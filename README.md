# Personalization-Aware RAG System

> A memory-aware enterprise retrieval system that adapts document ranking using user role, session history, prior queries, and recency signals — so two people asking the same question get different, more relevant answers.

---

## Why this project exists

Standard RAG treats every user identically: same query → same retrieved chunks → same answer.

This system does something different. It models **who is asking** and uses that signal to rerank retrieved evidence before generation. A product manager, an engineer, and a sales rep asking *"what are the key risks?"* should not get the same context. This project makes that possible.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────┐
│     Query Expander          │  GPT-4o rewrites query using
│  (memory-aware rewrite)     │  session history + role context
└─────────────┬───────────────┘
              │ expanded query
    ┌─────────▼──────────┐
    │  Hybrid Retrieval  │  BM25 (keyword) + ChromaDB (semantic)
    │  top-K candidates  │  merged via Reciprocal Rank Fusion
    └─────────┬──────────┘
              │ candidate pool
    ┌─────────▼──────────────────────────────────────┐
    │         Personalized Ranking Formula           │
    │                                                │
    │  score = 0.40 × semantic_rrf                   │
    │        + 0.20 × bm25_rrf                       │
    │        + 0.15 × recency_score                  │
    │        + 0.15 × role_match_score               │
    │        + 0.05 × history_match_score            │
    │        + 0.05 × session_relevance_score        │
    └─────────┬──────────────────────────────────────┘
              │ reranked candidates
    ┌─────────▼──────────┐
    │  Cross-Encoder     │  ms-marco-MiniLM-L-6-v2
    │  Reranking         │  scores query+chunk as a pair
    └─────────┬──────────┘
              │ top-N chunks with retrieval explanation
    ┌─────────▼──────────┐
    │  GPT-4o Generation │  citation-enforced, style-adapted
    │  + Citation Guard  │  per user preference
    └────────────────────┘
```

### Memory layers

| Layer | Storage | Contents |
|---|---|---|
| User profile | SQLite | role, team, preferred doc types, answer style |
| Session memory | SQLite | last 5 queries per user |
| Feedback memory | SQLite | liked/disliked docs per user |
| Document metadata | ChromaDB | doc type, team, role relevance, created_at, tags |

---

## Key results: personalization changes retrieval

Same question asked by three different users: *"What are the key risks and how should we handle them?"*

| User | Role | Top source retrieved | Role match score | Answer style |
|---|---|---|---|---|
| `pm_user` | Product Manager | `prd_onboarding.txt` | 1.0 | Concise with citations |
| `eng_user` | Engineer | `engineering_arch.txt` | 1.0 | Technical depth |
| `sales_user` | Sales Rep | `sales_battlecard.txt` | 1.0 | Bullet points |

Query expansion example:

```
Raw query:    "What should we prioritize next?"
pm_user:      "onboarding activation funnel Q1 growth team prioritization roadmap"
eng_user:     "platform infrastructure API performance backend scaling priorities"
sales_user:   "sales pipeline enterprise competitive deal prioritization Q1"
```

---

## Features

- **Hybrid retrieval** — BM25 keyword search + vector semantic search merged with Reciprocal Rank Fusion
- **Memory-aware query expansion** — GPT-4o rewrites queries using session history and user context
- **Personalized ranking formula** — 6-signal weighted score combining retrieval relevance with user signals
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reads query+chunk pairs for final precision
- **Retrieval explainability** — every chunk shows why it was boosted (role match, recency, session relevance)
- **Citation enforcement** — system declines to answer if retrieved chunks don't support a response
- **Cold/warm start handling** — new users fall back to pure semantic search; returning users get full personalization
- **Answer style adaptation** — concise, technical, or bullet-point format based on user preference
- **Persistent memory** — SQLite-backed profiles, sessions, and feedback survive across restarts

---

## Tech stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | GPT-4o |
| Keyword retrieval | BM25Okapi (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Memory store | SQLite (via Python stdlib) |
| Data validation | Pydantic v2 |
| Config management | YAML (versioned prompts) |

---

## Setup

```bash
# Clone and create environment
git clone https://github.com/YOUR_USERNAME/rag-portfolio
cd rag-portfolio
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Ingest documents
python ingest.py

# Seed demo user profiles
python seed_profiles.py

# Run interactive demo
python personalized_rag.py
```

---

## Project structure

```
rag-portfolio/
├── docs/
│   ├── metadata.json          # maps documents to role/team/type metadata
│   ├── prd_onboarding.txt     # PM-targeted document (Growth team)
│   ├── engineering_arch.txt   # Engineer-targeted document (Platform team)
│   └── sales_battlecard.txt   # Sales-targeted document
├── memory/
│   ├── schemas.py             # UserProfile, SessionMessage, FeedbackEvent
│   └── store.py               # SQLite persistence layer
├── personalized_retriever.py  # weighted ranking formula + cross-encoder
├── personalized_rag.py        # main pipeline
├── query_expander.py          # memory-aware query rewriting
├── seed_profiles.py           # creates demo user personas
├── ingest.py                  # document ingestion with metadata
├── eval/
│   └── compare_eval.py        # persona comparison evaluation
└── prompts/
    └── config.yaml            # versioned prompts + personalization weights
```

---

## Evaluation

Run the persona comparison:

```bash
python eval/compare_eval.py
```

This runs the same question through all 3 personas and prints which sources each retrieved, showing personalization working end-to-end.

---

## Design decisions

**Why SQLite over Redis?** For a portfolio project, zero-dependency persistence is the right tradeoff. The architecture supports swapping in Redis or Postgres — the store interface is intentionally simple.

**Why weight tuning in config.yaml?** Personalization weights are part of system behavior, not code. Versioning them alongside prompts means you can audit what changed when retrieval quality shifts.

**Why citation enforcement?** Uncited answers are untrustworthy in enterprise contexts. The system declines to answer rather than hallucinate — a deliberate product decision, not a technical limitation.

**Why cross-encoder after personalized ranking?** Personalization operates on metadata signals; the cross-encoder operates on semantic content. Running them in sequence gets the best of both.

---

## What's next

- [ ] FastAPI REST endpoint for multi-user serving
- [ ] Learning-to-rank: update user weights based on click feedback
- [ ] Memory decay: reduce weight of topics not touched in 60+ days
- [ ] Langfuse observability integration (Project 3)
- [ ] A/B eval: personalized vs. non-personalized retrieval precision@3
