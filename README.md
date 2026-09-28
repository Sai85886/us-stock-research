# us-stock-research

Multi-agent stock research system for US equities (NYSE/NASDAQ), built with a LangGraph supervisor that routes queries to specialized agents.

## Architecture

- **Supervisor (LangGraph)** — routes queries to the right specialist(s), with multi-agent handoffs
- **RAG agent** — answers from SEC 10-K filings via ChromaDB + local MiniLM embeddings
- **Market agent** — live prices and fundamentals via yfinance
- **Web agent** — recent news via Tavily
- **LLM** — Groq (`openai/gpt-oss-20b` by default)
- **Observability** — optional LangSmith tracing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

Required for full research flow:
- `SEC_EDGAR_USER_AGENT` — download filings
- `GROQ_API_KEY` — answering / agents
- `TAVILY_API_KEY` — web/news agent
- `LANGCHAIN_API_KEY` — optional LangSmith tracing

## Quick start

```bash
# 1) Ingest starter filings
stock-research download
stock-research chunk
stock-research embed --reset

# 2) Ask questions (supervisor routes to specialists)
stock-research research "What is AAPL's price and what are its 10-K risk factors?" --verbose

# 3) Check local setup
stock-research status
```

## CLI

Unified entrypoint:

```bash
stock-research <command> [args]
# or
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

Individual scripts (`research`, `market-quote`, etc.) still work.

### Examples

```bash
stock-research quote AAPL --fundamentals
stock-research web "Microsoft cloud news" --news
stock-research agent market "What is MSFT trading at?" --verbose
stock-research research "Latest news on JPM" --trace --verbose
```

## Starter tickers

Apple (`AAPL`), Microsoft (`MSFT`), JPMorgan (`JPM`)
