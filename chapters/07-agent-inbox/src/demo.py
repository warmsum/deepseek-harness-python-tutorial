"""第 07 章 demo：常驻 Agent 的连续两轮对话。

运行（在项目根目录，需要 .env）：
    uv run python chapters/07-agent-inbox/src/demo.py

演示两点：
1. 连续两轮对话共享同一份会话日志（第 2 轮模型记得第 1 轮）；
2. 事件日志里两个 turn/start…turn/end 边界清晰可辨。
"""

from __future__ import annotations

from agent import Agent
from calculator import calculator
from client import DeepSeekClient
from prompt import PromptAssembler
from registry import ToolRegistry


def main() -> None:
    assembler = PromptAssembler()
    assembler.section(
        "persona",
        "你是 {{name}}，一个数学助手。遇到算式时先调用 calculator 工具计算，"
        "再基于计算结果回答。",
        order=0,
    )
    assembler.section("rules", "回答要简洁：先给结论，再给过程。", order=100)

    registry = ToolRegistry()
    registry.register(calculator)

    agent = Agent(
        DeepSeekClient(), registry, assembler, variables={"name": "小算"}
    )

    print("=== 第 1 轮：问 1+2*3 ===")
    agent.followup("1+2*3 等于几？")
    agent.run()
    for message in agent.session.derive_messages():
        if message.role == "assistant" and message.content:
            print(f"  [assistant] {message.content}")

    print()
    print("=== 第 2 轮：followup 再问 8/4（同一会话，历史延续） ===")
    agent.followup("再帮我算 8/4")
    agent.run()
    for message in agent.session.derive_messages():
        if message.role == "assistant" and message.content:
            print(f"  [assistant] {message.content}")

    print()
    print("=== 事件日志：两个轮次边界 ===")
    for event in agent.session.snapshot_events():
        marker = "  ← 轮次边界" if event.type in ("turn/start", "turn/end") else ""
        print(f"  #{event.seq:<2} {event.type:<20}{marker}")


if __name__ == "__main__":
    main()
