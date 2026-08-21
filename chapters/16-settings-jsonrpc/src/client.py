"""第 16 章使用的最小 DeepSeek 客户端。"""

from __future__ import annotations

import os
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


class DeepSeekClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or load_api_key()

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=60) as http:
            response = http.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content:
            raise RuntimeError("模型没有返回文本")
        return content
