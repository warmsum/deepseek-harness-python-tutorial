"""第 07 章：持续处理多轮消息的 Agent 循环。

第 06 章的 run_agent 只处理一个问题。本章的 Agent 可以接收
followup/steer 并逐轮处理，轮次之间保持同一份会话日志。

层级术语（与官方对齐）：
- turn（轮次）：一次「唤醒到完成」的边界，由 turn/start 与 turn/end 夹住；
- step（步骤）：一轮内部的一次「模型调用 + 工具执行」。
"""

from __future__ import annotations

import json

from client import DeepSeekClient, Message
from inbox import Inbox
from prompt import PromptAssembler
from registry import ToolRegistry
from retry import RetryPolicy
from session import Session


class _EmptyResponseError(RuntimeError):
    code = "EMPTY_RESPONSE"


class Agent:
    def __init__(
        self,
        client: DeepSeekClient,
        registry: ToolRegistry,
        assembler: PromptAssembler,
        variables: dict[str, str] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._assembler = assembler
        self._variables = variables
        self._session = Session()
        self._inbox = Inbox(self._session)
        self._turn_no = 0
        self._request_header: str | None = None
        self._request_generation = 0
        self._retry_policy = retry_policy

    # ------------------------------------------------------------------
    # 外部入口：投递消息
    # ------------------------------------------------------------------

    def followup(self, content: str) -> None:
        """用户的常规提问：进入下一轮队列。"""
        if not content.strip():
            raise ValueError("followup 内容不能为空")
        self._inbox.followup(Message(role="user", content=content))

    def steer(self, content: str) -> None:
        """中途引导：进入下一步队列，当前轮次内立刻生效。"""
        if not content.strip():
            raise ValueError("steer 内容不能为空")
        self._inbox.steer(Message(role="user", content=content))

    @property
    def session(self) -> Session:
        return self._session

    # ------------------------------------------------------------------
    # 主循环：领取 → 轮次 → 步骤
    # ------------------------------------------------------------------

    def run(self, max_turns: int = 5) -> Session:
        """处理 inbox 直到没有待处理消息（教学版同步实现；
        官方在这里是常驻驱动器，空闲时挂起等待唤醒）。"""
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
            raise ValueError("max_turns 必须是正整数")
        turns_run = 0
        while self._inbox.pending > 0 and turns_run < max_turns:
            turns_run += 1
            self._turn_no += 1
            self._session.append("turn/start", {"turn": self._turn_no})
            claimed = self._inbox.claim_turn()
            if not claimed:
                self._session.append(
                    "turn/end", {"turn": self._turn_no, "reason": "completed"}
                )
                break
            try:
                self._run_turn(claimed)
            except Exception as error:
                self._session.append(
                    "turn/end",
                    {"turn": self._turn_no, "reason": "error", "message": str(error)},
                )
                raise
            else:
                self._session.append(
                    "turn/end", {"turn": self._turn_no, "reason": "completed"}
                )
        return self._session

    def _run_turn(
        self,
        claimed: list[Message],
    ) -> None:
        """一轮内反复领取 steer、请求模型并执行工具，最多运行 10 步。"""
        for step in range(1, 11):
            if step > 1:
                claimed = self._inbox.claim_step()
            self._session.append("step/start", {"turn": self._turn_no, "step": step})
            completed = False
            try:
                system_prompt = self._assembler.render(self._variables)
                self._session.record_system_prompt(
                    system_prompt, turn=self._turn_no, step=step
                )
                for message in claimed:
                    self._session.append("user/message", {"content": message.content})

                tools = self._registry.all()
                tools_by_name = {tool.name: tool for tool in tools}
                header: dict[str, object] = {
                    "config": {
                        "provider": "deepseek-official",
                        "model": self._client.MODEL,
                    }
                }
                schemas = self._registry.schemas()
                if schemas:
                    header["tools"] = schemas
                header_fingerprint = json.dumps(
                    header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                surface_changed = (
                    self._session.replace_generation != self._request_generation
                )
                if header_fingerprint != self._request_header or surface_changed:
                    reason = (
                        "initial"
                        if self._request_header is None
                        else "change"
                        if header_fingerprint != self._request_header
                        else "series"
                    )
                    self._session.append(
                        "request/header",
                        {
                            "header": header,
                            "reason": reason,
                        },
                    )
                    self._request_header = header_fingerprint
                    self._request_generation = self._session.replace_generation

                messages = self._session.derive_messages()
                while True:
                    try:
                        reply = self._client.chat(messages, tools)
                        if (
                            not reply.content
                            and not reply.reasoning_content
                            and not reply.tool_calls
                        ):
                            raise _EmptyResponseError(
                                "model returned a completed response with no content"
                            )
                    except Exception as error:
                        if self._retry_policy is None or not self._retry_policy.recover(
                            self._session,
                            turn=self._turn_no,
                            step=step,
                            error=error,
                        ):
                            raise
                        continue
                    break
                self._session.append(
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
                        self._session.append(
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
                        self._session.append(
                            "tool/result",
                            {"call_id": call.id, "content": result, "is_error": is_error},
                        )
            finally:
                self._session.append(
                    "step/end", {"turn": self._turn_no, "step": step}
                )

            if completed and not self._inbox.has_next_step:
                return
        raise RuntimeError(f"第 {self._turn_no} 轮超过 10 个 step 仍未结束")
