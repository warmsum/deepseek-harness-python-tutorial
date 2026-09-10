"""第 05 章：日志化的 Agent 循环。

相对第 02 章的核心变化：对话历史从「普通 list」升级为「事件日志」。
循环不再直接改 messages 列表，而是往 Session 追加事件；
每次请求前用 derive_messages() 投影出模型看到的历史。
"""

from __future__ import annotations

import json

from client import DeepSeekClient, Tool
from session import Session


def run_agent(
    client: DeepSeekClient,
    tools: list[Tool],
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 10,
) -> Session:
    """运行一轮工具对话，并把完整过程记录到事件日志。"""
    tools_by_name = {tool.name: tool for tool in tools}
    session = Session()

    session.append("turn/start", {"turn": 1})

    try:
        for step in range(1, max_steps + 1):
            session.append("step/start", {"turn": 1, "step": step})
            completed = False
            try:
                session.record_system_prompt(system_prompt, turn=1, step=step)
                if step == 1:
                    session.append("user/message", {"content": user_prompt})
                    header: dict[str, object] = {
                        "config": {
                            "provider": "deepseek-official",
                            "model": client.MODEL,
                        }
                    }
                    schemas = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in sorted(tools, key=lambda item: item.name)
                    ]
                    if schemas:
                        header["tools"] = schemas
                    session.append(
                        "request/header",
                        {
                            "header": header,
                            "reason": "initial",
                        },
                    )

                # 系统提示词和普通历史都来自同一份日志投影。
                messages = session.derive_messages()
                reply = client.chat(messages, tools)
                session.append(
                    "assistant/message",
                    {
                        "content": reply.content,
                        **(
                            {"reasoning_content": reply.reasoning_content}
                            if reply.reasoning_content
                            else {}
                        ),
                        "tool_calls": [
                            {"id": c.id, "name": c.name, "arguments": c.arguments}
                            for c in reply.tool_calls
                        ],
                    },
                )

                if not reply.tool_calls:
                    completed = True
                else:
                    for call in reply.tool_calls:
                        session.append(
                            "tool/call",
                            {
                                "call_id": call.id,
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        )
                        tool = tools_by_name.get(call.name)
                        is_error = tool is None
                        if tool is None:
                            result = f"Error: 模型请求了未注册的工具 {call.name!r}"
                        else:
                            try:
                                args = json.loads(call.arguments)
                                result = tool.execute(args)
                            except Exception as error:
                                is_error = True
                                result = f"工具执行出错: {error}"
                        session.append(
                            "tool/result",
                            {"call_id": call.id, "content": result, "is_error": is_error},
                        )
            finally:
                session.append("step/end", {"turn": 1, "step": step})
            if completed:
                session.append("turn/end", {"turn": 1, "reason": "completed"})
                return session
    except Exception as error:
        session.append(
            "turn/end", {"turn": 1, "reason": "error", "message": str(error)}
        )
        raise

    session.append("turn/end", {"turn": 1, "reason": "max-steps"})
    raise RuntimeError(f"Agent 在 {max_steps} 个 step 内没有结束")
