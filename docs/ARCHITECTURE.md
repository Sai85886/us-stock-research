# Architecture

This document explains how `us-stock-research` fits together.

## Goals

Answer US equity research questions with grounded tools:

1. **Filings** — SEC 10-K text via RAG
2. **Market** — live prices / fundamentals
3. **Web** — recent news

A supervisor decides which specialist(s) to call and can hand off across them for multi-part questions.

## Offline ingestion pipeline

Run before asking filing questions:

```text
SEC EDGAR
  → download (HTML 10-K + metadata.json)
  → load/clean HTML + split Item sections
  → chunk (~1000 chars, overlap 200)
  → embed with all-MiniLM-L6-v2
  → store vectors in Chroma (data/chroma)
```

CLI:

```bash
stock-research download
stock-research chunk
stock-research embed --reset
```

`data/raw`, `data/processed`, and `data/chroma` are gitignored and regenerated locally.

## Online query path

```text
stock-research research "<question>"
  → configure optional LangSmith tracing
  → LangGraph supervisor loop
      → choose next route: rag | market | web | FINISH
      → run specialist ReAct agent (tool calling via Groq)
      → maybe hand off to another specialist
      → synthesize final answer when FINISH
```

### Specialists and tools

| Specialist | Tools |
|------------|-------|
| `rag` | `search_filings` |
| `market` | `get_quote`, `get_fundamentals`, `get_history` |
| `web` | `search_web`, `search_news` |

Each specialist is a small ReAct loop: model chooses tools → observations → final answer for its scope.

### Supervisor handoffs

Example:

> What is AAPL's current price and what are its main 10-K risk factors?

Typical route: `market` → `rag` → synthesize.

Max handoffs default: 3 (`--max-handoffs`).

## Key modules

| Path | Role |
|------|------|
| `ingestion/edgar_client.py` | SEC ticker→CIK + 10-K download |
| `ingestion/loader.py` | HTML → text/sections |
| `ingestion/chunker.py` | Section → overlapping chunks |
| `ingestion/embedder.py` | Local SentenceTransformer embeddings |
| `vectorstore/chroma_store.py` | Persist + query vectors |
| `rag/chain.py` | Simple non-agent RAG chain |
| `agents/react.py` | Tool-calling ReAct loop |
| `agents/specialists.py` | rag/market/web agent factories |
| `agents/supervisor.py` | LangGraph routing + synthesis |
| `cli.py` | Unified `stock-research` CLI |

## Observability

Set in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=us-stock-research
```

Then:

```bash
stock-research research "..." --trace --verbose
```

Traces cover the research chain and ReAct agent runs in LangSmith.

## Design notes

- Ingestion is batch/offline so query-time stays interactive.
- Specialists are intentionally narrow so the supervisor must hand off for mixed questions.
- Default Groq model is a free-tier-friendly ID (`openai/gpt-oss-20b`); override with `GROQ_MODEL`.
- This system should not be used as financial advice.
