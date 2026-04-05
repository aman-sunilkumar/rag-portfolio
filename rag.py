import yaml
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from retriever import HybridRetriever

load_dotenv()

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)

client = OpenAI()
embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])


def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


def load_retriever(vectorstore):
    """Reconstruct all Document objects and build the hybrid retriever."""
    raw = vectorstore.get()
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return HybridRetriever(vectorstore, docs)


def format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i+1} | Source: {source}, page {page}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def query(question, retriever):
    chunks = retriever.retrieve(question)
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": [], "grounded": False}

    context = format_context(chunks)
    response = client.chat.completions.create(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        messages=[
            {"role": "system", "content": config["prompts"]["system"]},
            {"role": "user", "content": config["prompts"]["user"].format(
                context=context, question=question
            )}
        ]
    )
    answer = response.choices[0].message.content
    grounded = "cannot find sufficient information" not in answer.lower()
    sources = [
        {
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page", "?"),
            "snippet": c.page_content[:200]
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources, "grounded": grounded}


if __name__ == "__main__":
    vs = load_vectorstore()
    retriever = load_retriever(vs)
    print("RAG ready (Phase 2 - hybrid + reranking). Type 'quit' to exit.")
    print("Note: first run downloads the reranker (~100MB). Be patient.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = query(q, retriever)
        print(f"\nAnswer:\n{result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  [{s['source']}] ...{s['snippet'][:80]}...")
        print(f"  Grounded: {result['grounded']}\n")
