"""Minimal ReAct-style tool-calling agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from stock_research.agents.tools import AgentTool


@dataclass
class AgentStep:
    """One model/tool interaction inside the loop."""

    thought: str | None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    """Final agent answer plus trace of tool use."""

    agent: str
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest]


class ToolChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        ...


class GroqToolChatModel:
    """Groq chat completions with tool/function calling."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is required to run agents.")
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key.strip())

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls: list[ToolCallRequest] = []
        for call in message.tool_calls or []:
            tool_calls.append(
                ToolCallRequest(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments or "{}",
                )
            )
        return LLMResponse(content=message.content, tool_calls=tool_calls)


class ReActAgent:
    """Specialist agent that reasons and calls tools until it can answer."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[AgentTool],
        llm: ToolChatModel,
        max_steps: int = 5,
    ) -> None:
        if not tools:
            raise ValueError("ReActAgent requires at least one tool")
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.openai_schema() for tool in tools]
        self.llm = llm
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": cleaned},
        ]
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            response = self.llm.chat(messages, self.tool_schemas)

            if not response.tool_calls:
                answer = (response.content or "").strip()
                if not answer:
                    answer = "I could not produce an answer."
                return AgentResult(
                    agent=self.name,
                    question=cleaned,
                    answer=answer,
                    steps=steps,
                )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in response.tool_calls
                ],
            }
            messages.append(assistant_message)

            for call in response.tool_calls:
                observation = self._execute_tool(call.name, call.arguments)
                parsed_args: dict[str, Any]
                try:
                    loaded = json.loads(call.arguments) if call.arguments.strip() else {}
                    parsed_args = loaded if isinstance(loaded, dict) else {"raw": loaded}
                except json.JSONDecodeError:
                    parsed_args = {"raw": call.arguments}

                steps.append(
                    AgentStep(
                        thought=(response.content or None),
                        tool_name=call.name,
                        tool_args=parsed_args,
                        observation=observation,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )

        # Budget exceeded: ask once more without tools for a final answer.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool budget reached. Provide your best final answer "
                    "using the information gathered so far."
                ),
            }
        )
        final = self.llm.chat(messages, tools=[])
        answer = (final.content or "").strip() or "I ran out of tool steps before finishing."
        return AgentResult(
            agent=self.name,
            question=cleaned,
            answer=answer,
            steps=steps,
        )

    def _execute_tool(self, name: str, raw_arguments: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return tool.run(raw_arguments)
        except Exception as exc:  # noqa: BLE001 - return error to the agent loop
            return json.dumps({"error": str(exc)})
