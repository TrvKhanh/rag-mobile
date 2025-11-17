import uuid
import os
import logging
import chromadb
from jinja2 import Template
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk

from retrival.llm_router import Router
from retrival.re_rank import ReRank, HybridSearch
from generation.llm_stm import ChatWithMemory
from tools.comparison_tool import ComparisonTool

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Load API prompts
with open(os.path.join(base_dir, 'prompts', 'api', 'comparison.j2'), 'r', encoding='utf-8') as f:
    COMPARISON_PROMPT = f.read()
with open(os.path.join(base_dir, 'prompts', 'api', 'retrieval_with_docs.j2'), 'r', encoding='utf-8') as f:
    RETRIEVAL_WITH_DOCS_PROMPT = f.read()
with open(os.path.join(base_dir, 'prompts', 'api', 'retrieval_no_docs.j2'), 'r', encoding='utf-8') as f:
    RETRIEVAL_NO_DOCS_PROMPT = f.read()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialization ---
base_dir = os.path.dirname(__file__)
path_database = os.getenv("CHROMA_PATH", os.path.join(base_dir, "data.chromadb"))
logger.info(f"Initializing ChromaDB at path: {path_database}")

client = chromadb.PersistentClient(path_database)
collections = client.get_or_create_collection("production")
all_data = collections.get()

logger.info(f"Loaded {len(all_data.get('documents', []))} documents from ChromaDB")

documents = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
]

logger.info("Initializing HybridSearch")
hybrid_search = HybridSearch(topk=10, vector_path=path_database, collection_name="production", documents=documents)

logger.info("Initializing ReRank search")
search = ReRank(hybrid_search=hybrid_search, topk=3)


logger.info("Initializing ChatWithMemory model")
llm = ChatWithMemory().compile()

logger.info("Initializing ComparisonTool")
# Use the faster hybrid search for the tool, not the full re-ranker
comparison_tool = ComparisonTool(retriever=hybrid_search)

logger.info("Application initialized successfully")

LLL_ROUTER = Router()

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


async def stream_generator(thread_id: str, messages: list, initial_info: str = ""):
    """
    Generator for streaming the response from the LLM.
    It also accumulates the full response and logs it at the end.
    """
    yield f"thread_id:{thread_id}\n"
    if initial_info:
        yield initial_info
    
    config = {"configurable": {"thread_id": thread_id}}
    response_chunks = []
    try:
        async for event in llm.astream_events(
            {"messages": messages}, config=config, version="v1"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content = chunk.content
                    response_chunks.append(content)
                    logger.debug(f"Streaming chunk: {content}")
                    yield content
    except Exception as e:
        logger.error(f"Error during stream generation for thread_id {thread_id}: {e}", exc_info=True)
    finally:
        full_response = "".join(response_chunks)
        logger.info(f"Full response for thread_id {thread_id}: {full_response}")


@app.post("/chat/")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    logger.info(f"Received chat request - thread_id: {thread_id}, message: '{req.message}'")

    initial_info_to_stream = ""
    try:
        router_result = LLL_ROUTER.classify(req.message)
        logger.info(f"Result LLM: {router_result}")
        route = router_result.get("router")
        logger.info(f"Router result: {router_result}")

        if route == "comparison":
            product_names = router_result.get("products", [])
            logger.info(f"COMPARISON MODE: Comparing products: {product_names}")
            
            comparison_table = await comparison_tool.run.ainvoke({"product_names": product_names})
            
            initial_info_to_stream = f"COMPARISON_INFO:Đang so sánh {len(product_names)} sản phẩm.\n"
            
            template = Template(COMPARISON_PROMPT)
            system_prompt_template = template.render(comparison_table=comparison_table)
            messages = [
                SystemMessage(content=system_prompt_template),
                HumanMessage(content=req.message)
            ]

        elif route == "retrieval":
            query_info = router_result.get("infor")
            logger.info(f"RETRIEVAL MODE: Query: {query_info}")
            
            retrieved_docs = await search.query(query_info)
            logger.info(f"RAG search returned {len(retrieved_docs)} results")
            logger.info(f"Relust retrieval: {retrieved_docs}")

            initial_info_to_stream = f"RETRIEVAL_INFO:Tìm thấy {len(retrieved_docs)} kết quả.\n"

            if retrieved_docs:
                context_for_prompt = "\n\n---\n\n".join([
                    f"Nguồn: {doc.metadata.get('title', '')}\nURL: {doc.metadata.get('url', '')}\nNội dung: {doc.page_content}"
                    for doc in retrieved_docs
                ])
                
                template = Template(RETRIEVAL_WITH_DOCS_PROMPT)
                system_prompt_template = template.render(context_for_prompt=context_for_prompt)
            else:
                system_prompt_template = RETRIEVAL_NO_DOCS_PROMPT

            messages = [
                SystemMessage(content=system_prompt_template),
                HumanMessage(content=req.message)
            ]
        
        else: # Default to chat
            logger.info("CHAT MODE: Using simple chat without RAG")
            messages = [HumanMessage(content=req.message)]

        logger.info(f"Starting stream for thread_id: {thread_id}")
        return StreamingResponse(stream_generator(thread_id, messages, initial_info_to_stream), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error processing chat request - thread_id: {thread_id}, error: {str(e)}", exc_info=True)
        raise