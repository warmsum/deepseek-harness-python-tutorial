"""把计划模式、目标和任务清单接进真实模型—工具循环。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from client import DeepSeekClient, Message, Tool
from plan import PlanModeController
from session import Session


@dataclass(frozen=True)
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    traces: tuple[ToolTrace, ...]


def run_agent(
    client: DeepSeekClient,
    session: Session,
    plan_mode: PlanModeController,
    tools: list[Tool],
    user_prompt: str,
    *,
    max_steps: int = 10,
) -> AgentResult:
    by_name = {tool.name: tool for tool in tools}
    traces: list[ToolTrace] = []
    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"content": user_prompt})

    for step in range(1, max_steps + 1):
        notice = plan_mode.apply_boundary()
        session.append("step/start", {"turn": 1, "step": step})
        if notice:
            session.append("user/message", {"content": notice})
        system_prompt = (
            "你是任务规划助手。严格按照用户指定的工具顺序工作；"
            "工具成功前不要声称已经完成。"
        )
        plan_section = plan_mode.prompt_section()
        if plan_section:
            system_prompt += "\n\n" + plan_section
        session.append(
            "request/header",
            {
                "step": step,
                "plan_mode": plan_mode.get().active,
                "tools": [tool.name for tool in tools],
            },
        )
        reply = client.chat(
            [Message("system", system_prompt), *session.derive_messages()], tools
        )
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
            return AgentResult(reply.content or "", tuple(traces))

        for call in reply.tool_calls:
            session.append(
                "tool/call",
                {"call_id": call.id, "name": call.name, "arguments": call.arguments},
            )
            arguments: dict[str, Any] = {}
            try:
                raw = json.loads(call.arguments)
                if not isinstance(raw, dict):
                    raise TypeError("工具参数必须是 JSON 对象")
                arguments = raw
                result = by_name[call.name].execute(arguments)
                is_error = False
            except Exception as error:  # noqa: BLE001 - 工具边界把失败回灌给模型
                result = f"工具执行出错: {error}"
                is_error = True
            traces.append(ToolTrace(call.name, arguments, result))
            session.append(
                "tool/result",
                {"call_id": call.id, "content": result, "is_error": is_error},
            )
        session.append("step/end", {"turn": 1, "step": step})

    session.append("turn/end", {"turn": 1, "reason": "max_steps"})
    raise RuntimeError(f"模型在 {max_steps} 个步骤内没有结束")
