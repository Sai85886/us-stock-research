"""Tests for the LangGraph supervisor and handoffs."""

from __future__ import annotations

from stock_research.agents.react import AgentResult, AgentStep
from stock_research.agents.supervisor import parse_route_decision, run_research
from stock_research.config import Settings


class ScriptedRouter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            return '{"route":"FINISH","reason":"fallback"}'
        return self.responses.pop(0)


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
                    tool_name=f"{self.name}_tool",
                    tool_args={"q": question},
                    observation='{"ok": true}',
                )
            ],
        )


def _specialists() -> dict[str, FakeAgent]:
    return {
        "rag": FakeAgent("rag"),
        "market": FakeAgent("market"),
        "web": FakeAgent("web"),
    }


def test_parse_route_decision_accepts_embedded_json():
    route, reason = parse_route_decision(
        'Sure. {"route":"market","reason":"Needs live price"}'
    )
    assert route == "market"
    assert "live price" in reason


def test_parse_route_decision_accepts_finish():
    route, reason = parse_route_decision(
        '{"route":"FINISH","reason":"Enough context"}'
    )
    assert route == "FINISH"
    assert "Enough" in reason


def test_run_research_single_specialist():
    settings = Settings()
    router = ScriptedRouter(
        [
            '{"route":"market","reason":"price question"}',
            '{"route":"FINISH","reason":"done"}',
        ]
    )
    specialists = _specialists()

    result = run_research(
        "What is AAPL trading at?",
        settings,
        router=router,
        specialists=specialists,
    )

    assert result.routes == ["market"]
    assert result.answer == "market answered: What is AAPL trading at?"
    assert specialists["market"].questions == ["What is AAPL trading at?"]
    assert specialists["rag"].questions == []
    assert result.agent_steps[0].tool_name == "market_tool"


def test_run_research_multi_agent_handoff_and_synthesize():
    settings = Settings()
    router = ScriptedRouter(
        [
            '{"route":"market","reason":"need price"}',
            '{"route":"rag","reason":"need risks"}',
            '{"route":"FINISH","reason":"have both"}',
            "Combined: price from market and risks from rag.",
        ]
    )
    specialists = _specialists()

    result = run_research(
        "What is AAPL's price and what are its 10-K risk factors?",
        settings,
        router=router,
        specialists=specialists,
    )

    assert result.routes == ["market", "rag"]
    assert specialists["market"].questions
    assert specialists["rag"].questions
    assert "Combined" in result.answer
    assert len(result.specialist_answers) == 2
    # 2 route calls + 1 finish call + 1 synthesize call
    assert len(router.calls) == 4
