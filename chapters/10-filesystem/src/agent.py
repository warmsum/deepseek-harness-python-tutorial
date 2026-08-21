"""第 10 章复用的真实模型—工具循环。

这部分已经在第 02、07 章建立。本章保留最小副本，让文件能力直接进入
真实模型流程，同时让章节仍可独立运行。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


def load_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key
    key = dotenv_values(Path(__file__).resolve().parents[3] / ".env").get(
        "DEEPSEEK_API_KEY"
    )
    if key:
        return key
    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请参考 .env.example 创建 .env")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    traces: tuple[ToolTrace, ...]


class DeepSeekClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    @staticmethod
    def _wire(message: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            wire["content"] = message.content
        elif message.role == "assistant":
            wire["content"] = ""
        if message.reasoning_content:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            wire["tool_call_id"] = message.tool_call_id
        return wire

    def chat(self, messages: list[Message], tools: list[Tool]) -> Message:
        payload = {
            "model": "deepseek-chat",
            "messages": [self._wire(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
        }
        with httpx.Client(timeout=60) as http:
            response = http.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]
        return Message(
            role="assistant",
            content=raw.get("content"),
            reasoning_content=raw.get("reasoning_content"),
            tool_calls=tuple(
                ToolCall(
                    item["id"],
                    item["function"]["name"],
                    item["function"]["arguments"],
                )
                for item in raw.get("tool_calls") or []
            ),
        )


def run_agent(
    client: DeepSeekClient,
    tools: list[Tool],
    system_prompt: str,
    user_prompt: str,
    *,
    max_steps: int = 8,
) -> AgentResult:
    history = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    by_name = {tool.name: tool for tool in tools}
    traces: list[ToolTrace] = []
    for _ in range(max_steps):
        reply = client.chat(history, tools)
        history.append(reply)
        if not reply.tool_calls:
            return AgentResult(reply.content or "", tuple(traces))
        for call in reply.tool_calls:
            arguments: dict[str, Any] = {}
            try:
                raw = json.loads(call.arguments)
                if not isinstance(raw, dict):
                    raise TypeError("工具参数必须是 JSON 对象")
                arguments = raw
                tool = by_name[call.name]
                result = tool.execute(arguments)
            except Exception as error:  # noqa: BLE001 - 工具边界把失败回灌给模型
                result = f"工具执行出错: {error}"
            traces.append(ToolTrace(call.name, arguments, result))
            history.append(Message(role="tool", content=result, tool_call_id=call.id))
    raise RuntimeError(f"模型在 {max_steps} 个步骤内没有结束")
