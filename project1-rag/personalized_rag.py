import yaml
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from personalized_retriever import PersonalizedRetriever
from query_expander import expand_query
from memory.store import init_db, get_profile, save_profile, add_session, get_recent_queries, get_liked_docs
from memory.schemas import UserProfile, SessionMessage

load_dotenv()

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)

client = OpenAI()
_embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])


def load_vectorstore():
    return Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)


def load_retriever(vs):
    raw = vs.get()
    docs = [Document(page_content=c, metadata=m)
            for c, m in zip(raw["documents"], raw["metadatas"])]
    return PersonalizedRetriever(vs, docs)


def _format_context(chunks_with_scores: list) -> str:
    parts = []
    for i, (doc, score, signals) in enumerate(chunks_with_scores):
        reasons = []
        if signals["role_match"] > 0.3:   reasons.append("matches your role")
        if signals["recency"] > 0.7:      reasons.append("recent doc")
        if signals["history_match"] > 0:  reasons.append("you use this doc often")
        if signals["session_relevance"] > 0.3: reasons.append("matches session topics")
        why = f" | Boosted: {', '.join(reasons)}" if reasons else ""
        parts.append(
            f"[Chunk {i+1} | {doc.metadata.get('source','?')} | type={doc.metadata.get('doc_type','?')}{why}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def personalized_query(question: str, user_id: str, retriever: PersonalizedRetriever) -> dict:
    init_db()
    profile = get_profile(user_id) or UserProfile(user_id=user_id, role="unknown", team="unknown")
    save_profile(profile)

    recent_queries = get_recent_queries(user_id, n=5)
    liked_docs = get_liked_docs(user_id)
    expanded = expand_query(question, profile, recent_queries)
    chunks = retriever.retrieve(question, profile, liked_docs=liked_docs, expanded_query=expanded)

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": [], "expanded_query": expanded}

    style_map = {
        "concise_with_citations": "Be concise. Cite every claim.",
        "technical_depth": "Be thorough and technically precise. Cite every claim.",
        "bullet_points": "Use bullet points throughout. Cite every claim."
    }
    system = config["prompts"]["system"] + f"\n\nStyle: {style_map.get(profile.preferred_answer_style,'Cite every claim.')}"

    resp = client.chat.completions.create(
        model=config["llm"]["model"], temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": config["prompts"]["user"].format(
                context=_format_context(chunks), question=question)}
        ]
    )
    answer = resp.choices[0].message.content
    sources = [{"source": c.metadata.get("source","?"), "doc_type": c.metadata.get("doc_type","?"),
                "score": s, "signals": sig, "snippet": c.page_content[:120]}
               for c, s, sig in chunks]

    add_session(SessionMessage(user_id=user_id, query=question, answer=answer,
                               sources=[s["source"] for s in sources]))
    return {"answer": answer, "sources": sources, "expanded_query": expanded, "role": profile.role}


if __name__ == "__main__":
    init_db()
    vs = load_vectorstore()
    print("Building retriever (downloads reranker if needed)...")
    retriever = load_retriever(vs)
    print("Ready. Enter user_id and question. Type 'quit' to exit.\n")

    while True:
        uid = input("User ID (pm_user / eng_user / sales_user): ").strip()
        if uid.lower() in ("quit","q"): break
        q = input("Question: ").strip()
        if not q: continue
        result = personalized_query(q, uid, retriever)
        print(f"\n[{result['role']}] Expanded: {result['expanded_query']}")
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources + signals:")
        for s in result["sources"]:
            sig = s["signals"]
            print(f"  {s['source']} | score={s['score']} | role={sig['role_match']} recency={sig['recency']} session={sig['session_relevance']}")
        print()
