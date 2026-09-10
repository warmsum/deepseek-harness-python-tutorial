"""第 13 章复用的 DeepSeek 客户端、消息与工具结构。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


def load_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY") or dotenv_values(
        Path(__file__).resolve().parents[3] / ".env"
    ).get("DEEPSEEK_API_KEY")
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


class DeepSeekClient:
    MODEL = "deepseek-v4-flash"

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
            "model": self.MODEL,
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
