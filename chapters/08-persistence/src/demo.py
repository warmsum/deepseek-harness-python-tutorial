"""第 08 章：真实任务在运行中写入 JSONL，并从文件恢复。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent import run_agent
from calculator import calculator
from client import DeepSeekClient
from persistence import JsonlStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.jsonl"
        store = JsonlStore(path)

        print("=== 真实模型任务：计算并持续保存会话 ===")
        session = run_agent(
            DeepSeekClient(),
            calculator,
            store,
            "请计算 (18 + 6) / 3，并说明你使用了工具。",
        )
        final = [
            message.content
            for message in session.derive_messages()
            if message.role == "assistant" and message.content
        ][-1]
        print(f"模型最终回答: {final}")

        print("\n=== 磁盘中的真实事件顺序 ===")
        loaded = store.load()
        for event in loaded.events:
            print(f"#{event.id:<2} {event.type}")
        print(f"恢复后消息数量: {len(loaded.derive_messages())}")

        print("\n=== 使用同一日志模拟未完成写入 ===")
        with path.open("a", encoding="utf-8") as file:
            file.write('{"id":999,"type":"assistant/message"')
        recovered = store.load()
        print(f"残缺尾行已移除，最后事件仍是: {recovered.events[-1].type}")


if __name__ == "__main__":
    main()
