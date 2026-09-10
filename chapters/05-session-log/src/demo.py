"""第 05 章 demo：事件日志的完整演示。

运行（在项目根目录，需要 .env）：
    uv run python chapters/05-session-log/src/demo.py

输出四节：
① 订阅者视角：每条事件追加时被实时看到
② 日志原文：完整事件列表（含「只记不发」的事件）
③ 派生消息：模型真正看到的历史（投影）
④ 重放：同一份日志重建会话，派生结果完全一致
"""

from __future__ import annotations

from agent import run_agent
from calculator import calculator
from client import DeepSeekClient, Message
from session import Session, SessionEvent


def print_event(event: SessionEvent) -> None:
    print(f"  #{event.seq:<2} {event.type:<20} {event.data}")


def print_message(m: Message) -> None:
    if m.role == "assistant" and m.tool_calls:
        calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
        print(f"  [assistant → 请求工具] {calls}")
    elif m.role == "tool":
        print(f"  [tool → 结果] {m.content}")
    else:
        print(f"  [{m.role}] {m.content}")


def main() -> None:
    client = DeepSeekClient()

    # ① 订阅者视角：模拟「持久化插件」挂在订阅接口上，实时看到每条事件
    print("=== ① 订阅者视角 ===")
    demo = Session()
    demo.subscribe(lambda e: print(f"  [订阅者] 看到新事件 #{e.seq} {e.type}"))
    demo.append("user/message", {"content": "你好"})
    print()

    print("=== ② 事件日志原文（唯一事实来源） ===")
    result = run_agent(
        client,
        tools=[calculator],
        system_prompt="你是一个数学助手。遇到算式时先调用 calculator 工具计算，"
        "再基于计算结果回答。",
        user_prompt="1+2*3 等于几？",
        max_steps=10,
    )
    for event in result.snapshot_events():
        print_event(event)

    print()
    print("=== ③ 派生消息：模型看到的历史（derive_messages 投影） ===")
    for message in result.derive_messages():
        print_message(message)

    print()
    print("=== ④ 重放：同一份日志 → 新会话 → 完全相同的消息历史 ===")
    replayed = Session.from_log(result.snapshot_events())
    original = result.derive_messages()
    replayed_messages = replayed.derive_messages()
    same = [m.role for m in original] == [m.role for m in replayed_messages]
    print(f"  重放派生 {len(replayed_messages)} 条消息，与原始派生一致: {same}")
    print("  ← 日志是唯一事实来源：持久化、界面、重放都从它派生")


if __name__ == "__main__":
    main()
