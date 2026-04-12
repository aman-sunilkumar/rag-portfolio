import yaml
import numpy as np
from datetime import datetime, timezone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from memory.schemas import UserProfile

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)


def _recency(meta: dict) -> float:
    try:
        age = (datetime.now(timezone.utc) -
               datetime.fromisoformat(str(meta.get("created_at",""))).replace(tzinfo=timezone.utc)).days
        return max(0.0, 1.0 - age / 365.0)
    except Exception:
        return 0.5


def _role_match(meta: dict, profile: UserProfile) -> float:
    relevance = meta.get("role_relevance", "").split(",")
    score = 0.0
    if profile.role in relevance:
        score += 0.5
    if profile.team and meta.get("team", "").lower() == profile.team.lower():
        score += 0.3
    if meta.get("doc_type", "") in profile.preferred_doc_types:
        score += 0.2
    return min(1.0, score)


def _history_match(meta: dict, liked_docs: list, frequent_docs: list) -> float:
    doc_id = meta.get("doc_id", "")
    score = 0.6 if doc_id in liked_docs else 0.0
    score += 0.4 if doc_id in frequent_docs else 0.0
    return min(1.0, score)


def _session_relevance(meta: dict, recent_topics: list, session_context: list) -> float:
    topic = meta.get("topic", "").lower()
    tags = meta.get("tags", "").lower()
    signals = [s.lower() for s in recent_topics + session_context]
    if not signals:
        return 0.0
    hits = sum(1 for s in signals if s in topic or s in tags)
    return min(1.0, hits / max(len(signals), 1))


class PersonalizedRetriever:
    def __init__(self, vectorstore, all_chunks: list):
        self.vs = vectorstore
        self.all_chunks = all_chunks
        tokenized = [d.page_content.lower().split() for d in all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.reranker = CrossEncoder(config["retrieval"]["reranker_model"])
        self.w = config["personalization"]["weights"]
        self.top_k = config["retrieval"]["top_k"]
        self.reranker_top_n = config["retrieval"]["reranker_top_n"]

    def retrieve(self, query: str, profile: UserProfile,
                 liked_docs: list = None, expanded_query: str = None) -> list:
        liked_docs = liked_docs or []
        sq = expanded_query or query

        # Semantic + BM25 candidates
        vec_docs = self.vs.similarity_search(sq, k=self.top_k)
        bm25_scores = self.bm25.get_scores(sq.lower().split())
        bm25_top = [self.all_chunks[i] for i in np.argsort(bm25_scores)[::-1][:self.top_k]]

        # RRF merge into candidate pool
        pool = {}
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content[:80]
            pool[key] = {"doc": doc, "vr": rank + 1, "br": self.top_k + 1}
        for rank, doc in enumerate(bm25_top):
            key = doc.page_content[:80]
            if key not in pool:
                pool[key] = {"doc": doc, "vr": self.top_k + 1, "br": rank + 1}
            else:
                pool[key]["br"] = rank + 1

        # Score each candidate with personalization formula
        w = self.w
        scored = []
        for key, item in pool.items():
            doc, meta = item["doc"], item["doc"].metadata
            rrf = w["semantic"] / item["vr"] + w["bm25"] / item["br"]
            rec   = _recency(meta)
            role  = _role_match(meta, profile)
            hist  = _history_match(meta, liked_docs, profile.frequent_docs)
            sess  = _session_relevance(meta, profile.recent_topics, profile.session_context)
            final = rrf + w["recency"]*rec + w["role_match"]*role + w["history_match"]*hist + w["session_relevance"]*sess
            scored.append((doc, round(final, 3), {
                "recency": round(rec, 2), "role_match": round(role, 2),
                "history_match": round(hist, 2), "session_relevance": round(sess, 2)
            }))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.top_k]

        # Cross-encoder rerank
        pairs = [[query, c[0].page_content] for c in top]
        rerank_scores = self.reranker.predict(pairs)
        reranked = sorted(zip(top, rerank_scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in reranked[:self.reranker_top_n]]
