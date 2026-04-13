import time
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# Langfuse is optional — if no credentials, tracing is silently skipped
_langfuse = None
try:
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import Langfuse
        _langfuse = Langfuse()
except Exception:
    pass


def traced_query(question: str, retriever, query_fn) -> dict:
    t_start = time.perf_counter()

    # --- Retrieval ---
    t0 = time.perf_counter()
    chunks = retriever.retrieve(question)
    ret_ms = round((time.perf_counter() - t0) * 1000, 1)

    # --- Generation ---
    t1 = time.perf_counter()
    result = _generate_with_chunks(question, chunks)
    gen_ms = round((time.perf_counter() - t1) * 1000, 1)

    total_ms = round(time.perf_counter() - t_start, 3) * 1000
    result["_timing"] = {"ret_ms": ret_ms, "gen_ms": gen_ms, "total_ms": total_ms}

    # --- Send to Langfuse if configured ---
    if _langfuse:
        try:
            from langfuse.decorators import langfuse_context
            _langfuse.create_trace(
                name="rag-query",
                input={"question": question},
                output={"answer": result["answer"][:300]},
                metadata={
                    "ret_ms": ret_ms,
                    "gen_ms": gen_ms,
                    "grounded": result["grounded"],
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                }
            )
            _langfuse.flush()
        except Exception as e:
            print(f"[tracing] Langfuse error (non-fatal): {e}")

    return result


def _generate_with_chunks(question, chunks):
    from openai import OpenAI
    from rag import format_context

    with open("prompts/config.yaml") as f:
        config = yaml.safe_load(f)

    client = OpenAI()

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": [], "grounded": False,
                "prompt_tokens": 0, "completion_tokens": 0}

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
        {"source": c.metadata.get("source", "unknown"),
         "page": c.metadata.get("page", "?"),
         "snippet": c.page_content[:200]}
        for c in chunks
    ]
    return {
        "answer": answer, "sources": sources, "grounded": grounded,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
