import uuid
import os
import logging
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from retrival.llm_router import llm_router
from retrival.re_rank import ReRank
from generation.llm_stm import ChatWithMemory

# Cấu hình logging
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

logger.info("Initializing ReRank search")
search = ReRank(3, path_database, documents, "production")

logger.info("Initializing ChatWithMemory model")
llm = ChatWithMemory().compile()
logger.info("Application initialized successfully")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.post("/chat/")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    logger.info(f"Received chat request - thread_id: {thread_id}, message length: {len(req.message)}")
    
    try:
        router = llm_router(req.message)
        logger.info(f"Router result - router: {router['router']}, info: {router.get('infor', 'N/A')}")
        
        if router["router"] == "rag":
            logger.info(f"RAG mode: Querying with info: {router['infor']}")
            relust = search.query(router["infor"])
            logger.info(f"RAG search returned {len(relust) if isinstance(relust, list) else 'N/A'} results")
            
            system_prompt_template = f"""Bạn là một nhân viên tư vấn của shop điện thoại.
                                        Nhiệm vụ:
                                        - Hãy trả lời một cách thân thiện, minh bạch.
                                        - Luôn giữ văn phong lịch sự, chuyên nghiệp, không dùng ngôn ngữ lập lờ hay viết hoa toàn bộ.
                                        - Dựa vào thông tin có sẵn, hãy trả lời chi tiết cho người dùng dễ hiểu, tiếp cận thông tin như sau: {relust}
                                        """ 

            messages = [
                SystemMessage(content=system_prompt_template),
                HumanMessage(content=req.message) 
            ]
        else:
            logger.info("Chat mode: Using simple chat without RAG")
            messages = [HumanMessage(content=req.message)]

        logger.info(f"Invoking LLM with {len(messages)} messages")
        response = llm.invoke(
            input={"messages": messages},
            config={"configurable": {"thread_id": thread_id}}
        )

        ai_reply = response["messages"][-1].content
        logger.info(f"LLM response generated - length: {len(ai_reply)} characters")

        return {
            "thread_id": thread_id,
            "response": ai_reply
        }
    except Exception as e:
        logger.error(f"Error processing chat request - thread_id: {thread_id}, error: {str(e)}", exc_info=True)
        raise
