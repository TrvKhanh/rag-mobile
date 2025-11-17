import torch
import os
import asyncio
import hashlib
import logging
from typing import List, Dict
from diskcache import Cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.disk_cache')
cache = Cache(cache_dir)

tokenizer = AutoTokenizer.from_pretrained('namdp-ptit/ViRanker')
model = AutoModelForSequenceClassification.from_pretrained('namdp-ptit/ViRanker')
model.to(device)
model.eval()

def hash_query(query: str, topk: int):
    return hashlib.md5(f"{query}_{topk}".encode("utf-8")).hexdigest()


#Hybrid search
class HybridSearch:
    def __init__(self, topk: int, vector_path: str, collection_name: str, documents: List[Document], bm25_weight: float = 0.5):
        self.k = topk
        self.bm25_weight = bm25_weight

        if not documents:
            documents = [Document(page_content="Welcome to shop!", metadata={"product_id": "default"})]

        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = topk

        self.vector_store = Chroma(persist_directory=vector_path, collection_name=collection_name)
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": topk}
        )

    async def query(self, query: str) -> List[Document]:
        cache_key = f"hybrid_{hash_query(query, self.k)}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        bm25_docs, vector_docs = await asyncio.gather(
            self.bm25.ainvoke(query),
            self.vector_retriever.ainvoke(query)
        )

        # Fusion weighted by bm25_weight
        score_dict: Dict[str, float] = {}
        doc_dict: Dict[str, Document] = {}

        for doc in bm25_docs:
            pid = doc.metadata.get("product_id")
            score_dict[pid] = score_dict.get(pid, 0) + self.bm25_weight
            doc_dict[pid] = doc

        for doc in vector_docs:
            pid = doc.metadata.get("product_id")
            score_dict[pid] = score_dict.get(pid, 0) + (1 - self.bm25_weight)
            doc_dict[pid] = doc

        # Sort product_ids by fused score
        sorted_pids = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        result = [doc_dict[pid] for pid, _ in sorted_pids[:self.k]]

        cache.set(cache_key, result, expire=86400)
        logger.info(f"Hybrid search for '{query}' returned {len(result)} products.")
        return result


# Rerank chunk-level but aggregate product-level 
def rerank_chunk_level(query: str, docs: List[Document], topk: int) -> List[Document]:
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        scores = model(**inputs).logits.squeeze().float()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

    # Aggregate by product_id (max score across chunks)
    product_scores: Dict[str, float] = {}
    product_docs: Dict[str, Document] = {}
    for doc, score in zip(docs, scores.tolist()):
        pid = doc.metadata.get("product_id")
        if pid not in product_scores or score > product_scores[pid]:
            product_scores[pid] = score
            product_docs[pid] = doc

    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [product_docs[pid] for pid, _ in sorted_products[:topk]]
    return top_docs


#ReRank wrapper
class ReRank:
    def __init__(self, hybrid_search: HybridSearch, topk: int = 5):
        self.hybrid_search = hybrid_search
        self.topk = topk

    async def query(self, query: str) -> List[Document]:
        cache_key = f"rerank_{hash_query(query, self.topk)}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        docs = await self.hybrid_search.query(query)
        reranked_docs = await asyncio.to_thread(rerank_chunk_level, query, docs, self.topk)

        cache.set(cache_key, reranked_docs, expire=86400)
        logger.info(f"Rerank for '{query}' returned {len(reranked_docs)} products.")
        return reranked_docs
