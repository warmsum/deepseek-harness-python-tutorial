"""支持工具调用的 DeepSeek 客户端与消息模型。

这部分在第 02 章首次实现。后续章节保留一份副本，让每章都能从自己的
`src/` 目录独立运行，不需要跨目录导入前面章节的代码。

它提供三样基础能力：
1. `ToolCall` —— 模型发起的一次工具调用请求
2. `Message` —— assistant 可携带 reasoning_content 与 tool_calls；工具结果以 role="tool" 回灌
3. `DeepSeekClient.chat()` 支持传入工具清单、解析模型返回的 tool_calls
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import dotenv_values
from httpx_sse import aconnect_sse

# ---------------------------------------------------------------------------
# 1. 从 .env 读 API Key（与第 01 章相同）
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """先读取环境变量，再读取项目根目录的 .env。"""
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env
    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file
    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请参考 .env.example 创建 .env")


# ---------------------------------------------------------------------------
# 2. 消息模型：新增工具调用的三个概念
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用请求。

    OpenAI 兼容协议把 arguments 表示为 JSON 字符串。执行工具前需要先用
    json.loads 将其解析成 Python 对象。
    """

    id: str       # 调用编号，工具结果回灌时靠它一一对应
    name: str     # 要调用的工具名
    arguments: str  # 参数 JSON 字符串，例如 '{"expression": "1+2*3"}'


@dataclass(frozen=True)
class Message:
    """一条对话消息。相对第 01 章多了三个可选字段：

    - reasoning_content：模型的思考内容，后续请求必须按原文回传；
    - tool_calls：assistant 消息可以携带一组工具调用请求；
    - tool_call_id：role="tool" 的消息用它标明对应的工具调用。
    """

    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Tool:
    """一个 Agent 可用的工具。

    - name/description/parameters 是发送给模型的参数说明；
    - execute 是仅在 Agent 进程中运行的执行函数。
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema，描述参数长什么样
    execute: Callable[[dict[str, Any]], str]


# ---------------------------------------------------------------------------
# 3. 客户端：请求带工具清单，响应解析 tool_calls
# ---------------------------------------------------------------------------


class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    @staticmethod
    def _wire_message(m: Message) -> dict[str, Any]:
        """把内部消息转换成协议字典，并保留思考内容与工具调用编号。"""
        wire: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            wire["content"] = m.content
        elif m.role == "assistant":
            wire["content"] = ""
        if m.reasoning_content:
            wire["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in m.tool_calls
            ]
        if m.tool_call_id is not None:
            wire["tool_call_id"] = m.tool_call_id
        return wire

    def chat(self, messages: list[Message], tools: list[Tool] | None = None) -> Message:
        """执行非流式调用，返回含思考内容和工具调用的完整消息。"""
        payload: dict[str, Any] = {
            "model": self.MODEL,
            "messages": [self._wire_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        raw_message = choice["message"]
        tool_calls: list[ToolCall] = []
        for raw_call in raw_message.get("tool_calls") or []:
            tool_calls.append(
                ToolCall(
                    id=raw_call["id"],
                    name=raw_call["function"]["name"],
                    arguments=raw_call["function"]["arguments"],
                )
            )
        return Message(
            role="assistant",
            content=raw_message.get("content"),
            reasoning_content=raw_message.get("reasoning_content"),
            tool_calls=tuple(tool_calls),
        )

    # ---------- 流式方法（与第 01 章相同，用于终端展示） ----------

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """流式调用：只处理纯文本回答的展示。工具调用场景见 README 对照。"""
        completed = False
        async with httpx.AsyncClient(timeout=60) as client:
            async with aconnect_sse(
                client,
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [self._wire_message(m) for m in messages],
                    "stream": True,
                },
            ) as event_source:
                event_source.response.raise_for_status()
                async for event in event_source.aiter_sse():
                    if event.data == "[DONE]":
                        completed = True
                        break
                    payload = json.loads(event.data)
                    delta = payload["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        if not completed:
            raise RuntimeError("流式响应在 [DONE] 之前中断，拒绝保存不完整消息")
