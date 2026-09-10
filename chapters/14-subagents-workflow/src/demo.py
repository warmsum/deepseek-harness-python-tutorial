"""第 14 章 demo：Subagent、Jobs 与 Workflow。

运行（在项目根目录，需要 .env）：
    uv run python chapters/14-subagents-workflow/src/demo.py

演示：
1. 用本地 scripted client 演示 fork、continuable 与 owner 隔离；
2. 本地运行 Job 状态结算和 Workflow 编排；
3. 调用真实模型，并行委派两个子任务后汇总结果。
"""

from __future__ import annotations

import json
import time

from client import DeepSeekClient, Message, Tool
from jobs import LocalJobs
from session import Session
from subagent import SubagentManager, create_subagent_tool, run_subagents_parallel
from workflow import WorkflowEngine, WorkflowMeta


class ScriptedClient(DeepSeekClient):
    """不访问网络的子 Agent client，用于展示生命周期。"""

    def __init__(self) -> None:
        pass

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> Message:
        del tools
        return Message(role="assistant", content=f"已处理：{messages[-1].content}")


def run_local_lifecycles() -> None:
    print("=== ① fork + continuable：继承完整轮次并继续投递 ===")
    parent = Session()
    parent.append("turn/start", {"turn": 1})
    parent.append("step/start", {"turn": 1, "step": 1})
    parent.record_system_prompt("你是父 Agent。", turn=1, step=1)
    parent.append("user/message", {"content": "已完成的父任务"})
    parent.append("step/end", {"turn": 1, "step": 1})
    parent.append("turn/end", {"turn": 1, "reason": "completed"})
    parent.append("turn/start", {"turn": 2})
    parent.append("user/message", {"content": "尚未完成，不应被 fork"})
    manager = SubagentManager(ScriptedClient(), "你是本地演示子 Agent。")
    child = manager.create("root-1", parent_session=parent, fork=True)
    first = child.send_message("继续第一项")
    second = child.send_message("再处理第二项")
    user_messages = [
        message.content
        for message in child.session.derive_messages()
        if message.role == "user"
    ]
    print(f"  两次结果: {first.output} / {second.output}")
    print(f"  子会话用户消息: {user_messages}")
    try:
        manager.get("other-root", child.id)
    except KeyError:
        print("  owner 隔离: 其他 root 无法访问 child")
    manager.close()

    print()
    print("=== ② Jobs：后台执行、等待与首次终态 ===")
    jobs = LocalJobs(max_concurrency=2)
    job = jobs.start("root-1", lambda cancelled: "已完成" if not cancelled.is_set() else "")
    settled = jobs.wait("root-1", job.id, timeout=2)
    print(f"  {settled.id}: {settled.status}, output={settled.output}")
    jobs.close()

    print()
    print("=== ③ Workflow：并发与逐项 pipeline ===")
    workflow = WorkflowEngine(max_concurrency=2, max_agents=8)
    parallel = workflow.run(
        WorkflowMeta("two-tasks", "并行运行两个确定性任务"),
        [lambda: "A", lambda: "B"],
    )
    piped = workflow.pipeline([1, 2], lambda value, _item, _index: value * 2)
    print(f"  parallel={parallel.value}, pipeline={piped}")


def main() -> None:
    run_local_lifecycles()

    client = DeepSeekClient()
    subagent_tool = create_subagent_tool(
        client, child_system_prompt="你是一个速算助手，直接给出答案。"
    )

    print()
    print("=== ④ 主 agent 请求并行委派 ===")
    main_history: list[Message] = [
        Message(
            role="system",
            content=(
                "你是一个任务协调者。遇到多个独立的子问题时，"
                "在同一轮里用 subagent 工具并行委派，拿到全部结果后再汇总回答。"
            ),
        ),
        Message(role="user", content="同时帮我算两件事：3+3 等于几？7*2 等于几？"),
    ]

    reply = client.chat(main_history, [subagent_tool])
    main_history.append(reply)
    for call in reply.tool_calls:
        print(f"  [主 agent → 请求工具] {call.name}({call.arguments})")
    print(f"  （本轮请求了 {len(reply.tool_calls)} 个 subagent，将并行执行）")

    print()
    print("=== ⑤ 并行执行（计时） ===")
    specs = []
    for call in reply.tool_calls:
        task = json.loads(call.arguments)["task"]
        specs.append((task, "你是一个速算助手，直接给出答案。"))
    started = time.monotonic()
    results = run_subagents_parallel(specs, client)
    elapsed = time.monotonic() - started
    for index, result in enumerate(results, start=1):
        print(f"  [子 agent#{index}] {result.output}  ({result.stop_reason})")
    print(f"  总耗时: {elapsed:.1f}s")

    print()
    print("=== ⑥ 上下文隔离证据 ===")
    # 子 agent 的模型可见历史只有 system + task；这里用一个探测会话展示。
    probe = Session()
    probe.append("turn/start", {"turn": 1})
    probe.append("step/start", {"turn": 1, "step": 1})
    probe.record_system_prompt("你是一个速算助手，直接给出答案。", turn=1, step=1)
    probe.append("user/message", {"content": specs[0][0]})
    for message in probe.derive_messages():
        print(f"  [{message.role}] {(message.content or '')[:40]}…")
    print("  ← 子会话未自动继承父 agent 的对话历史")

    print()
    print("=== ⑦ 主 agent 汇总 ===")
    for call in reply.tool_calls:
        main_history.append(
            Message(
                role="tool",
                content=(
                    "[completed]\n"
                    + [r.output for r in results][
                        reply.tool_calls.index(call)
                    ]
                ),
                tool_call_id=call.id,
            )
        )
    final = client.chat(main_history, [subagent_tool])
    print(f"  [主 agent] {final.content}")


if __name__ == "__main__":
    main()
