"""第 02 章 demo：运行一次“模型 → 工具 → 模型”往返。

运行（在项目根目录）：
    uv run python chapters/02-tool-calling/src/demo.py
"""

from __future__ import annotations

from agent import run_agent
from calculator import calculator
from client import DeepSeekClient, Message


def print_history(history: list[Message]) -> None:
    """打印完整对话历史，注意观察 tool 消息如何回灌。"""
    for message in history:
        if message.role == "assistant" and message.tool_calls:
            calls = ", ".join(
                f"{call.name}({call.arguments})" for call in message.tool_calls
            )
            print(f"\n[assistant → 请求工具] {calls}")
        elif message.role == "tool":
            print(f"\n[tool → 结果 #{message.tool_call_id}] {message.content}")
        else:
            print(f"\n[{message.role}]\n{message.content}")


def main() -> None:
    client = DeepSeekClient()
    history = run_agent(
        client,
        tools=[calculator],
        system_prompt="你是一个数学助手。遇到算式时先调用 calculator 工具计算，"
        "再基于计算结果回答。",
        user_prompt="1+2*3 等于几？",
        max_steps=10,
    )
    print("=== 完整对话历史 ===")
    print_history(history)


if __name__ == "__main__":
    main()
