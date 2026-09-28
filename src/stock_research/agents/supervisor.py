"""LangGraph supervisor that routes questions to specialist agents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from stock_research.agents.react import AgentResult, AgentStep
from stock_research.agents.specialists import SPECIALIST_BUILDERS, build_specialist
from stock_research.config import Settings

RouteName = Literal["rag", "market", "web"]
VALID_ROUTES = set(SPECIALIST_BUILDERS)

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor for a US stock research system.
Route each user question to exactly one specialist:

- rag: SEC 10-K filings (business, risk factors, MD&A, financial statement discussion)
- market: live prices, fundamentals, valuation metrics, price history
- web: recent news, headlines, and general web information

Pick the best single specialist for the question.
Respond with ONLY valid JSON of the form:
{"route":"rag"|"market"|"web","reason":"short justification"}
"""


class ResearchState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    answer: str
    agent_steps: list[dict[str, Any]]


@dataclass
class ResearchResult:
    question: str
    route: str
    route_reason: str
    answer: str
    agent_steps: list[AgentStep] = field(default_factory=list)


class RouterModel(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GroqRouterModel:
    """Plain chat completion used only for routing decisions."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is required to run the supervisor.")
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key.strip())

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Supervisor did not return JSON: {text!r}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"Supervisor JSON must be an object: {text!r}")
    return payload


def parse_route_decision(text: str) -> tuple[RouteName, str]:
    payload = _extract_json(text)
    route = str(payload.get("route", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    if route not in VALID_ROUTES:
        raise ValueError(f"Invalid route '{route}'. Expected one of {sorted(VALID_ROUTES)}")
    return route, reason  # type: ignore[return-value]


def _steps_to_dicts(steps: list[AgentStep]) -> list[dict[str, Any]]:
    return [asdict(step) for step in steps]


def build_research_graph(
    settings: Settings,
    *,
    router: RouterModel | None = None,
    specialists: dict[str, Any] | None = None,
):
    """Compile a LangGraph app: supervisor -> specialist -> END."""
    router_model = router or GroqRouterModel(settings.groq_api_key, settings.groq_model)

    def supervisor_node(state: ResearchState) -> dict[str, Any]:
        question = state["question"]
        raw = router_model.complete(SUPERVISOR_SYSTEM_PROMPT, question)
        route, reason = parse_route_decision(raw)
        return {"route": route, "route_reason": reason}

    def make_specialist_node(name: str):
        def _node(state: ResearchState) -> dict[str, Any]:
            if specialists is not None:
                agent = specialists[name]
            else:
                agent = build_specialist(name, settings)
            result: AgentResult = agent.run(state["question"])
            return {
                "answer": result.answer,
                "agent_steps": _steps_to_dicts(result.steps),
            }

        return _node

    graph = StateGraph(ResearchState)
    graph.add_node("supervisor", supervisor_node)
    for name in sorted(VALID_ROUTES):
        graph.add_node(name, make_specialist_node(name))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {name: name for name in sorted(VALID_ROUTES)},
    )
    for name in sorted(VALID_ROUTES):
        graph.add_edge(name, END)

    return graph.compile()


def run_research(
    question: str,
    settings: Settings,
    *,
    router: RouterModel | None = None,
    specialists: dict[str, Any] | None = None,
) -> ResearchResult:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    app = build_research_graph(settings, router=router, specialists=specialists)
    final_state = app.invoke({"question": cleaned})

    steps = [
        AgentStep(
            thought=item.get("thought"),
            tool_name=item.get("tool_name"),
            tool_args=item.get("tool_args"),
            observation=item.get("observation"),
        )
        for item in final_state.get("agent_steps") or []
    ]
    return ResearchResult(
        question=cleaned,
        route=str(final_state.get("route") or ""),
        route_reason=str(final_state.get("route_reason") or ""),
        answer=str(final_state.get("answer") or ""),
        agent_steps=steps,
    )
