"""第 08 章复用的 DeepSeek 客户端、消息与工具结构。

模型—工具协议已在第 02 章讲解。本章保留一份可独立运行的最小副本，
把重点放在会话何时写入磁盘，而不是跨章节导入隐藏实现。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


def load_api_key() -> str:
    """优先读取进程环境，其次读取项目根目录的 .env。"""
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env
    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file
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
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str | None = None, model: str = "deepseek-v4-flash") -> None:
        self.api_key = api_key or load_api_key()
        self.model = model

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
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
        if message.tool_call_id is not None:
            wire["tool_call_id"] = message.tool_call_id
        return wire

    def chat(self, messages: list[Message], tools: list[Tool]) -> Message:
        payload = {
            "model": self.model,
            "messages": [self._wire_message(message) for message in messages],
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
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]
        calls = tuple(
            ToolCall(
                id=item["id"],
                name=item["function"]["name"],
                arguments=item["function"]["arguments"],
            )
            for item in raw.get("tool_calls") or []
        )
        return Message(
            role="assistant",
            content=raw.get("content"),
            reasoning_content=raw.get("reasoning_content"),
            tool_calls=calls,
        )
