# us-stock-research

Multi-agent stock research system for US equities (NYSE/NASDAQ). A LangGraph supervisor routes questions to specialized agents for SEC filings, live market data, and web/news.

> **Disclaimer:** This project is for education and research prototyping only. It is **not** investment advice.

## Architecture

```
User question
    └─► LangGraph supervisor
            ├─► rag agent      (Chroma + 10-K chunks)
            ├─► market agent   (yfinance)
            └─► web agent      (Tavily)
                    └─► optional handoffs + final synthesis
```

- **Supervisor** — chooses one or more specialists, then synthesizes
- **RAG agent** — SEC 10-K retrieval over local MiniLM embeddings in Chroma
- **Market agent** — live quotes, fundamentals, history via yfinance
- **Web agent** — recent news / web results via Tavily
- **LLM** — Groq (`openai/gpt-oss-20b` by default)
- **Observability** — optional LangSmith tracing

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Needed for |
|----------|------------|
| `SEC_EDGAR_USER_AGENT` | Downloading filings (`"Name email@example.com"`) |
| `GROQ_API_KEY` | Answering / agents |
| `TAVILY_API_KEY` | Web/news agent |
| `LANGCHAIN_API_KEY` | Optional LangSmith tracing |
| `GROQ_MODEL` | Optional model override |
| `EMBEDDING_MODEL` | Optional embedding model override |

## Quick start

```bash
# 1) Ingest starter filings (AAPL, MSFT, JPM)
stock-research download
stock-research chunk
stock-research embed --reset

# 2) Ask questions (supervisor routes + may hand off)
stock-research research "What is AAPL's price and what are its 10-K risk factors?" --verbose

# 3) Check local setup
stock-research status
```

## CLI

```bash
stock-research <command> [args]
python -m stock_research <command> [args]
```

| Command | Purpose |
|---------|---------|
| `download` | Download latest SEC 10-K filings |
| `chunk` | Chunk filings into `data/processed` |
| `embed` | Embed chunks into Chroma |
| `research` | Multi-agent supervised Q&A |
| `agent` | Run one specialist (`rag` / `market` / `web`) |
| `ask-filings` | Basic RAG over filings (no agent loop) |
| `quote` | Live yfinance quote / fundamentals |
| `web` | Tavily web/news search |
| `status` | Show config + local data status |

Individual entrypoints (`research`, `market-quote`, etc.) still work.

### Examples

```bash
stock-research quote AAPL --fundamentals
stock-research web "Microsoft cloud news" --news
stock-research agent market "What is MSFT trading at?" --verbose
stock-research research "Latest news on JPM" --trace --verbose
```

## Project structure

```text
src/stock_research/
  agents/          # ReAct specialists + LangGraph supervisor
  ingestion/       # EDGAR download, HTML load/chunk, embeddings
  rag/             # Basic retrieve-then-generate chain
  tools/           # yfinance + Tavily helpers
  vectorstore/     # Chroma persistence
  observability/   # LangSmith tracing setup
  scripts/         # Command implementations
  cli.py           # Unified stock-research entrypoint
tests/
docs/
data/              # Local only (gitignored): raw / processed / chroma
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Starter tickers

Apple (`AAPL`), Microsoft (`MSFT`), JPMorgan (`JPM`)
