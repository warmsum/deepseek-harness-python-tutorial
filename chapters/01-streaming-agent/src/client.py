"""第 01 章：最小流式 Agent。

这一章实现三样东西：
1. `load_api_key()`   —— 从项目根目录的 .env 读 API Key
2. `Message`          —— 一条不可变的对话消息
3. `DeepSeekClient`   —— 同时支持完整响应与流式响应的模型客户端
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from dotenv import dotenv_values
from httpx_sse import aconnect_sse

# ---------------------------------------------------------------------------
# 1. 从 .env 读 API Key
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """先读取环境变量，再读取项目根目录的 .env。"""
    # 第一步：环境变量（例如终端里 export DEEPSEEK_API_KEY=...）
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env

    # 第二步：项目根目录的 .env 文件。
    # 本文件位于 chapters/01-streaming-agent/src/，向上三级才是项目根目录。
    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file

    raise RuntimeError(
        "找不到 DEEPSEEK_API_KEY：请在项目根目录创建 .env，"
        "写入一行 DEEPSEEK_API_KEY=你的key（参考 .env.example）"
    )


# ---------------------------------------------------------------------------
# 2. 消息：对话历史里的最小单位
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """一条对话消息。

    `frozen=True` 表示创建后不能修改。对话历史会被反复读取并发送给模型；
    不可变消息可以防止后续代码意外改写已经记录的内容。
    """

    role: str  # "system"（规则）/"user"（用户）/"assistant"（模型）
    content: str


# ---------------------------------------------------------------------------
# 3. 模型客户端：完整响应与流式响应
# ---------------------------------------------------------------------------


class DeepSeekClient:
    """基于 httpx 与 httpx-sse 的 DeepSeek OpenAI 兼容客户端。"""

    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    # ---------- 3.1 非流式：一次拿回完整回答 ----------

    def chat(self, messages: list[Message]) -> str:
        """把整段对话发给模型，并在生成结束后返回完整回答。"""
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,  # 关键开关：False = 一次给全
                },
            )
            response.raise_for_status()  # 网络/认证出错时抛出带状态码的异常
            data = response.json()
            return cast(str, data["choices"][0]["message"]["content"])

    # ---------- 3.2 流式：边生成边产出 ----------

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """流式调用：模型每生成一小段，就立即交出一个分片（chunk）。

        这是一个「异步生成器」——调用方用 `async for` 遍历它，
        每迭代一次拿到一小段新文字，调用方立刻打印到终端。
        """
        completed = False
        async with httpx.AsyncClient(timeout=60) as client:
            # aconnect_sse 帮我们解析 SSE 协议。
            # SSE 是服务端持续推送数据的文本协议：每条数据以 "data: ..." 开头。
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
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": True,  # 关键开关：True = 边生成边给
                },
            ) as event_source:
                event_source.response.raise_for_status()
                async for event in event_source.aiter_sse():
                    if event.data == "[DONE]":
                        completed = True
                        break  # DeepSeek 用这一行表示响应正常结束
                    payload = json.loads(event.data)
                    delta = payload["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece  # 把这一小段文字交给调用方
        if not completed:
            raise RuntimeError("流式响应在 [DONE] 之前中断，拒绝保存不完整消息")

    # ---------- 3.3 组装：把分片拼成一条完整消息 ----------

    async def stream_message(self, messages: list[Message]) -> Message:
        """流式展示分片，并在正常结束后生成一条完整的历史消息。"""
        pieces: list[str] = []
        async for piece in self.stream(messages):
            pieces.append(piece)
        return Message(role="assistant", content="".join(pieces))
