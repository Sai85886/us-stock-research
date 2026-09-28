"""LangGraph supervisor with multi-agent handoffs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from stock_research.agents.react import AgentResult, AgentStep
from stock_research.agents.specialists import SPECIALIST_BUILDERS, build_specialist
from stock_research.config import Settings

RouteName = Literal["rag", "market", "web", "FINISH"]
VALID_SPECIALISTS = set(SPECIALIST_BUILDERS)
MAX_HANDOFFS = 3

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor for a US stock research system.
Choose the next specialist to consult, or FINISH if you have enough information.

Specialists:
- rag: SEC 10-K filings (business, risk factors, MD&A, financial statement discussion)
- market: live prices, fundamentals, valuation metrics, price history
- web: recent news, headlines, and general web information

Rules:
1. Prefer the minimum number of specialists needed.
2. Never route to a specialist already listed under "Already consulted".
3. Use FINISH only when consulted answers actually cover every part of the user question.
4. For multi-part questions (e.g. price + 10-K risks + news), hand off across specialists.
5. Filing/10-K/risk-factor questions require rag. Live price/fundamentals require market. Recent news requires web.
6. Do not FINISH early just because one specialist speculated beyond its tools.

Respond with ONLY valid JSON:
{"route":"rag"|"market"|"web"|"FINISH","reason":"short justification"}
"""

SYNTHESIZE_SYSTEM_PROMPT = """You combine specialist research answers into one final response.
Use only the specialist answers provided.
Be concise, factual, and clearly separate topics when multiple specialists contributed.
Cite specialist names when helpful (filings/market/web).
Do not give investment advice.
If only one specialist answered, you may lightly polish that answer without adding new facts."""


class ResearchState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    visited: list[str]
    route_history: list[str]
    specialist_answers: list[dict[str, Any]]
    agent_steps: list[dict[str, Any]]
    answer: str


@dataclass
class ResearchResult:
    question: str
    routes: list[str]
    route_reason: str
    answer: str
    specialist_answers: list[dict[str, Any]] = field(default_factory=list)
    agent_steps: list[AgentStep] = field(default_factory=list)

    @property
    def route(self) -> str:
        """Backward-compatible single-route view."""
        return ",".join(self.routes) if self.routes else ""


class RouterModel(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GroqRouterModel:
    """Plain chat completion used for routing and synthesis."""

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
    if route == "finish":
        route = "FINISH"
    reason = str(payload.get("reason", "")).strip() or "No reason provided."
    allowed = VALID_SPECIALISTS | {"FINISH"}
    if route not in allowed:
        raise ValueError(f"Invalid route '{route}'. Expected one of {sorted(allowed)}")
    return route, reason  # type: ignore[return-value]


def _steps_to_dicts(steps: list[AgentStep]) -> list[dict[str, Any]]:
    return [asdict(step) for step in steps]


def _format_consulted(specialist_answers: list[dict[str, Any]]) -> str:
    if not specialist_answers:
        return "None yet."
    blocks: list[str] = []
    for item in specialist_answers:
        blocks.append(
            f"- {item.get('agent')}: {item.get('answer')}"
        )
    return "\n".join(blocks)


def build_supervisor_user_prompt(state: ResearchState) -> str:
    visited = state.get("visited") or []
    return (
        f"User question:\n{state['question']}\n\n"
        f"Already consulted: {', '.join(visited) if visited else 'None'}\n\n"
        f"Specialist answers so far:\n{_format_consulted(state.get('specialist_answers') or [])}\n\n"
        "Choose the next route."
    )


def build_research_graph(
    settings: Settings,
    *,
    router: RouterModel | None = None,
    specialists: dict[str, Any] | None = None,
    max_handoffs: int = MAX_HANDOFFS,
):
    """Compile supervisor loop: route -> specialist* -> synthesize -> END."""
    router_model = router or GroqRouterModel(settings.groq_api_key, settings.groq_model)

    def supervisor_node(state: ResearchState) -> dict[str, Any]:
        visited = list(state.get("visited") or [])
        if len(visited) >= max_handoffs:
            return {
                "route": "FINISH",
                "route_reason": f"Reached max handoffs ({max_handoffs}).",
            }

        raw = router_model.complete(
            SUPERVISOR_SYSTEM_PROMPT,
            build_supervisor_user_prompt(state),
        )
        route, reason = parse_route_decision(raw)

        if route != "FINISH" and route in visited:
            return {
                "route": "FINISH",
                "route_reason": f"Avoided repeat handoff to {route}; finishing instead.",
            }
        return {"route": route, "route_reason": reason}

    def make_specialist_node(name: str):
        def _node(state: ResearchState) -> dict[str, Any]:
            if specialists is not None:
                agent = specialists[name]
            else:
                agent = build_specialist(name, settings)

            result: AgentResult = agent.run(state["question"])
            visited = list(state.get("visited") or [])
            route_history = list(state.get("route_history") or [])
            specialist_answers = list(state.get("specialist_answers") or [])
            agent_steps = list(state.get("agent_steps") or [])

            visited.append(name)
            route_history.append(name)
            specialist_answers.append(
                {
                    "agent": name,
                    "answer": result.answer,
                    "steps": _steps_to_dicts(result.steps),
                }
            )
            agent_steps.extend(_steps_to_dicts(result.steps))

            return {
                "visited": visited,
                "route_history": route_history,
                "specialist_answers": specialist_answers,
                "agent_steps": agent_steps,
            }

        return _node

    def synthesize_node(state: ResearchState) -> dict[str, Any]:
        answers = state.get("specialist_answers") or []
        if not answers:
            return {
                "answer": (
                    "No specialist was consulted, so I could not answer the question."
                )
            }
        if len(answers) == 1:
            return {"answer": str(answers[0].get("answer") or "")}

        synthesis_prompt = (
            f"User question:\n{state['question']}\n\n"
            f"Specialist answers:\n{_format_consulted(answers)}\n\n"
            "Write the final combined answer."
        )
        answer = router_model.complete(SYNTHESIZE_SYSTEM_PROMPT, synthesis_prompt)
        return {"answer": answer}

    def after_supervisor(state: ResearchState) -> str:
        route = state.get("route") or "FINISH"
        if route == "FINISH":
            return "synthesize"
        return route

    graph = StateGraph(ResearchState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("synthesize", synthesize_node)
    for name in sorted(VALID_SPECIALISTS):
        graph.add_node(name, make_specialist_node(name))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        after_supervisor,
        {
            **{name: name for name in sorted(VALID_SPECIALISTS)},
            "synthesize": "synthesize",
        },
    )
    for name in sorted(VALID_SPECIALISTS):
        graph.add_edge(name, "supervisor")
    graph.add_edge("synthesize", END)

    return graph.compile()


@traceable(name="research", run_type="chain")
def run_research(
    question: str,
    settings: Settings,
    *,
    router: RouterModel | None = None,
    specialists: dict[str, Any] | None = None,
    max_handoffs: int = MAX_HANDOFFS,
) -> ResearchResult:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    app = build_research_graph(
        settings,
        router=router,
        specialists=specialists,
        max_handoffs=max_handoffs,
    )
    final_state = app.invoke(
        {
            "question": cleaned,
            "visited": [],
            "route_history": [],
            "specialist_answers": [],
            "agent_steps": [],
        }
    )

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
        routes=list(final_state.get("route_history") or []),
        route_reason=str(final_state.get("route_reason") or ""),
        answer=str(final_state.get("answer") or ""),
        specialist_answers=list(final_state.get("specialist_answers") or []),
        agent_steps=steps,
    )
