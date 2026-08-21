"""把真实模型—工具循环接到 JSONL 持久化检查点。"""

from __future__ import annotations

import json

from checkpoint import CheckpointPolicy
from client import DeepSeekClient, Message, Tool
from persistence import JsonlStore
from session import Session


def run_agent(
    client: DeepSeekClient,
    tool: Tool,
    store: JsonlStore,
    user_prompt: str,
    *,
    max_steps: int = 6,
) -> Session:
    """运行一次真实任务，并在模型与工具边界前保存事件日志。"""
    session = Session()
    checkpoint = CheckpointPolicy(store.save)
    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"content": user_prompt})

    for step in range(1, max_steps + 1):
        if step > 1:
            checkpoint.before_step(session)
        session.append("step/start", {"turn": 1, "step": step})
        messages = [
            Message(
                role="system",
                content="你是计算助手。算术必须调用 calculator，不要自行心算。",
            ),
            *session.derive_messages(),
        ]
        session.append(
            "request/header",
            {"model": client.model, "tools": [tool.name], "step": step},
        )
        checkpoint.before_model(session)
        reply = client.chat(messages, [tool])
        session.append(
            "assistant/message",
            {
                "content": reply.content,
                "reasoning_content": reply.reasoning_content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in reply.tool_calls
                ],
            },
        )

        if not reply.tool_calls:
            session.append("step/end", {"turn": 1, "step": step})
            session.append("turn/end", {"turn": 1, "reason": "completed"})
            store.save(session)
            return session

        for call in reply.tool_calls:
            session.append(
                "tool/call",
                {"call_id": call.id, "name": call.name, "arguments": call.arguments},
            )
            checkpoint.before_tool(session)
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("工具参数必须是 JSON 对象")
                result = tool.execute(arguments)
                is_error = False
            except Exception as error:  # noqa: BLE001 - 工具边界把失败回灌给模型
                result = f"工具执行出错: {error}"
                is_error = True
            session.append(
                "tool/result",
                {"call_id": call.id, "content": result, "is_error": is_error},
            )
        session.append("step/end", {"turn": 1, "step": step})

    session.append("turn/end", {"turn": 1, "reason": "max_steps"})
    store.save(session)
    raise RuntimeError(f"模型在 {max_steps} 个步骤内没有结束")
