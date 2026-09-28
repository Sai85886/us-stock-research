# us-stock-research

Multi-agent stock research system for US equities (NYSE/NASDAQ), built with a LangGraph supervisor that routes queries to specialized agents.

## Architecture (target)

- **Supervisor (LangGraph)** — routes queries to the right specialist
- **RAG agent** — answers from SEC 10-K filings via LangChain + ChromaDB
- **Market agent** — live prices and fundamentals via yfinance
- **Web agent** — recent news via Tavily
- **LLM** — Groq (`openai/gpt-oss-20b` by default) for low-latency multi-agent reasoning
- **Observability** — LangSmith tracing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Starter tickers

Apple (`AAPL`), Microsoft (`MSFT`), JPMorgan (`JPM`)
