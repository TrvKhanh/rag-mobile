from langchain_core.tools import tool
from retrival.re_rank import ReRank
import logging

logger = logging.getLogger(__name__)

# Keywords that trigger the high-accuracy, slower re-ranking pipeline
RERANK_KEYWORDS = [
    "so sánh", "đánh giá", "nên mua", "khác biệt", 
    "tốt hơn", "ưu điểm", "nhược điểm", "phân tích"
]

def should_rerank(query: str) -> bool:
    """Checks if the query contains keywords that suggest a need for re-ranking."""
    query_lower = query.lower()
    for keyword in RERANK_KEYWORDS:
        if keyword in query_lower:
            return True
    return False

# This is a factory function: it takes the initialized ReRank instance
# and returns a fully configured, ready-to-use tool.
def create_rag_tool(search_instance: ReRank):
    
    @tool
    async def product_search_tool(query: str) -> str:
        """
        Use this tool to answer any questions about phone products, their specifications, 
        prices, comparisons, or reviews. The input should be the user's question.
        """
        
        # Use the conditional re-ranking logic we built before
        if should_rerank(query):
            logger.info(f"RAG TOOL: HIGH-ACCURACY MODE for query: {query}")
            retrieved_docs = await search_instance.query(query)
        else:
            logger.info(f"RAG TOOL: FAST MODE for query: {query}")
            retrieved_docs = await search_instance.llm.query(query)

        if not retrieved_docs:
            return "Không tìm thấy thông tin sản phẩm nào phù hợp với câu hỏi của bạn."

        # Format the retrieved documents into a single string for the agent
        context = "\n\n---\n\n".join([
            f"Nguồn: {doc.metadata.get('title', '')}\nURL: {doc.metadata.get('url', '')}\nNội dung: {doc.page_content}"
            for doc in retrieved_docs
        ])
        
        return context

    return product_search_tool
