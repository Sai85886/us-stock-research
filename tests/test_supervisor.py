"""Tests for the LangGraph supervisor."""

from __future__ import annotations

from stock_research.agents.react import AgentResult, AgentStep
from stock_research.agents.supervisor import parse_route_decision, run_research
from stock_research.config import Settings


class FakeRouter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.text


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.questions: list[str] = []

    def run(self, question: str) -> AgentResult:
        self.questions.append(question)
        return AgentResult(
            agent=self.name,
            question=question,
            answer=f"{self.name} answered: {question}",
            steps=[
                AgentStep(
                    thought=None,
                    tool_name="dummy",
                    tool_args={"q": question},
                    observation='{"ok": true}',
                )
            ],
        )


def test_parse_route_decision_accepts_embedded_json():
    route, reason = parse_route_decision(
        'Sure. {"route":"market","reason":"Needs live price"}'
    )
    assert route == "market"
    assert "live price" in reason


def test_run_research_routes_to_market_specialist():
    settings = Settings()
    router = FakeRouter('{"route":"market","reason":"price question"}')
    market = FakeAgent("market")
    specialists = {
        "rag": FakeAgent("rag"),
        "market": market,
        "web": FakeAgent("web"),
    }

    result = run_research(
        "What is AAPL trading at?",
        settings,
        router=router,
        specialists=specialists,
    )

    assert result.route == "market"
    assert result.route_reason == "price question"
    assert result.answer == "market answered: What is AAPL trading at?"
    assert market.questions == ["What is AAPL trading at?"]
    assert result.agent_steps[0].tool_name == "dummy"
