# us-stock-research

Multi-agent stock research system for US equities (NYSE/NASDAQ), built with a LangGraph supervisor that routes queries to specialized agents.

## Architecture (target)

- **Supervisor (LangGraph)** — routes queries to the right specialist
- **RAG agent** — answers from SEC 10-K filings via LangChain + ChromaDB
- **Market agent** — live prices and fundamentals via yfinance
- **Web agent** — recent news via Tavily
- **LLM** — Groq (Llama 3.3 70B) for low-latency multi-agent reasoning
- **Observability** — LangSmith tracing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Project status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Project scaffold + config | Done |
| 1 | Download SEC 10-K filings | Pending |
| 2 | Load & chunk 10-K documents | Pending |
| 3 | Embeddings + ChromaDB | Pending |
| 4 | Basic RAG chain | Pending |
| 5 | yfinance market data tools | Pending |
| 6 | Tavily web search | Pending |
| 7 | Specialized ReAct agents | Pending |
| 8 | LangGraph supervisor | Pending |
| 9 | Multi-agent handoffs | Pending |
| 10 | LangSmith tracing | Pending |
| 11 | CLI interface | Pending |
| 12 | Docs & polish | Pending |

## Starter tickers

Apple (`AAPL`), Microsoft (`MSFT`), JPMorgan (`JPM`)
