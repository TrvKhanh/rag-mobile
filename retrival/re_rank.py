import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from typing import List

tokenizer = AutoTokenizer.from_pretrained('namdp-ptit/ViRanker')
model = AutoModelForSequenceClassification.from_pretrained('namdp-ptit/ViRanker')
model.eval()



class HybridSearch:
    def __init__(self, topk: int, path: str, document: List[Document], collection_name: str):
        self.k = topk
        
        if not document:
            document = [Document(page_content="Chào mừng bạn đến với shop điện thoại!", metadata={"source": "default"})]
        
        self.bm25 = BM25Retriever.from_documents(document)
        self.bm25.k = self.k

        self.vector_stores = Chroma(
            persist_directory=path,
            collection_name=collection_name
        )
        self.vector_retriever = self.vector_stores.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25, self.vector_retriever],
            weights=[0.3, 0.7],
            mode="reciprocal_rerank"
        )

    def query(self, query: str) -> List[Document]:
        return self.hybrid_retriever.invoke(query)

class ReRank:
    def __init__(self, topk: int, path: str, document: List[Document], collection_name: str):
        self.llm = HybridSearch(topk, path, document, collection_name)

    def query(self, query: str, score_threshold: float = 5.0) -> List[Document]:
        results = self.llm.query(query)

        if not results:
            return []
            
        pairs = [[query, doc.page_content] for doc in results]

        with torch.no_grad():
            inputs = tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )

            scores = model(**inputs, return_dict=True).logits.squeeze().float()
            
            # Đảm bảo scores là 1D tensor
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

        
        reranked_docs = [
            doc for doc, score in zip(results, scores) if score > score_threshold
        ]
        return reranked_docs