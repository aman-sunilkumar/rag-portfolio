import yaml
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)


class HybridRetriever:
    def __init__(self, vectorstore, all_chunks):
        self.vectorstore = vectorstore
        self.all_chunks = all_chunks

        # Build BM25 index from all stored chunks
        tokenized = [doc.page_content.lower().split() for doc in all_chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Cross-encoder reranker (~100MB download on first use, then cached)
        self.reranker = CrossEncoder(config["retrieval"]["reranker_model"])

        self.top_k = config["retrieval"]["top_k"]
        self.reranker_top_n = config["retrieval"]["reranker_top_n"]
        self.bm25_weight = config["retrieval"]["bm25_weight"]
        self.vector_weight = config["retrieval"]["vector_weight"]

    def retrieve(self, query_text):
        # --- Vector retrieval (semantic) ---
        vector_docs = self.vectorstore.similarity_search(query_text, k=self.top_k)

        # --- BM25 retrieval (keyword) ---
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:self.top_k]
        bm25_docs = [self.all_chunks[i] for i in top_bm25_idx]

        # --- Reciprocal Rank Fusion (RRF) to merge results ---
        rrf = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            if key not in rrf:
                rrf[key] = {"doc": doc, "score": 0.0}
            rrf[key]["score"] += self.vector_weight / (rank + 1)

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            if key not in rrf:
                rrf[key] = {"doc": doc, "score": 0.0}
            rrf[key]["score"] += self.bm25_weight / (rank + 1)

        candidates = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)
        candidate_docs = [c["doc"] for c in candidates[:self.top_k]]

        # --- Cross-encoder reranking (reads query + chunk as a pair) ---
        pairs = [[query_text, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidate_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in reranked[:self.reranker_top_n]]
