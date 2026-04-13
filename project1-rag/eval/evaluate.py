import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
FAITHFULNESS_THRESHOLD = 0.75


def score_faithfulness(question, answer, contexts):
    """Use GPT-4o as a judge to score whether the answer is grounded in context."""
    context_text = "\n---\n".join(contexts)
    prompt = f"""Evaluate whether the AI answer is grounded in the provided context.

Context:
{context_text}

Question: {question}
Answer: {answer}

Score faithfulness from 0.0 (fully hallucinated) to 1.0 (fully grounded).
Respond ONLY with valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.strip('`')
    if raw.startswith('json'):
        raw = raw[4:].strip()
    try:
        result = json.loads(raw)
        return result["score"], result["reason"]
    except Exception:
        return 0.5, "Could not parse judge response"



def write_results(golden, scores, avg):
    import json
    from pathlib import Path
    out = {
        "avg_faithfulness": round(avg, 3),
        "threshold": FAITHFULNESS_THRESHOLD,
        "n": len(scores),
        "passed": avg >= FAITHFULNESS_THRESHOLD,
        "per_question": [
            {"question": golden[i]["question"], "score": round(scores[i], 3)}
            for i in range(len(scores))
        ]
    }
    Path("eval/eval_results.json").write_text(json.dumps(out, indent=2))
    print("Results written → eval/eval_results.json")


def run_evaluation():
    with open("prompts/config.yaml") as f:
        config = yaml.safe_load(f)

    with open("eval/golden_set.json") as f:
        golden = json.load(f)

    embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    raw = vectorstore.get()
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]

    from retriever import HybridRetriever
    from rag import query

    retriever = HybridRetriever(vectorstore, docs)

    scores = []
    print(f"\nEvaluating {len(golden)} questions...\n")

    for i, item in enumerate(golden):
        result = query(item["question"], retriever)
        contexts = [s["snippet"] + "..." for s in result["sources"]]
        score, reason = score_faithfulness(item["question"], result["answer"], contexts)
        scores.append(score)
        status = "PASS" if score >= FAITHFULNESS_THRESHOLD else "FAIL"
        print(f"Q{i+1}: [{status}] score={score:.2f} — {reason[:80]}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n{'=' * 45}")
    print(f"Avg Faithfulness : {avg:.3f}")
    print(f"Threshold        : {FAITHFULNESS_THRESHOLD}")

    if avg < FAITHFULNESS_THRESHOLD:
        write_results(golden, scores, avg)
        print("RESULT: FAIL — build blocked.")
        sys.exit(1)
    else:
        print("RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    run_evaluation()

