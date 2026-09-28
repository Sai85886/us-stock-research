"""Tests for specialist ReAct agents."""

from __future__ import annotations

import json

from stock_research.agents.react import LLMResponse, ReActAgent, ToolCallRequest
from stock_research.agents.tools import AgentTool, build_market_tools


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            return LLMResponse(content="fallback", tool_calls=[])
        return self.responses.pop(0)


def test_react_agent_runs_tool_then_answers():
    def add_numbers(a: int, b: int) -> dict:
        return {"sum": a + b}

    tool = AgentTool(
        name="add_numbers",
        description="Add two integers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=add_numbers,
    )

    llm = ScriptedLLM(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="add_numbers",
                        arguments=json.dumps({"a": 2, "b": 3}),
                    )
                ],
            ),
            LLMResponse(content="The sum is 5.", tool_calls=[]),
        ]
    )

    agent = ReActAgent(
        name="math",
        system_prompt="Use tools to calculate.",
        tools=[tool],
        llm=llm,
    )
    result = agent.run("What is 2+3?")

    assert result.answer == "The sum is 5."
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "add_numbers"
    assert json.loads(result.steps[0].observation or "{}")["sum"] == 5
    assert len(llm.calls) == 2


def test_build_market_tools_expose_expected_names():
    names = {tool.name for tool in build_market_tools()}
    assert names == {"get_quote", "get_fundamentals", "get_history"}
