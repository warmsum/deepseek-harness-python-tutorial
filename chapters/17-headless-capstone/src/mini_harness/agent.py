"""持续对话的 Agent 循环，以及供插件接入的生命周期事件。

Agent Loop 只固定 turn/step、消息配对和终止路径。Plan Mode、checkpoint、
retry、pruner 与 spill 都监听 Context 事件，因此增加或替换策略不需要修改
循环本身。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from .client import ChatClient, Message, Tool, ToolCall
from .cordis import Context
from .inbox import Inbox
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .rpc import RpcDispatcher
from .session import Session


class EmptyResponseError(RuntimeError):
    code = "EMPTY_RESPONSE"


@dataclass
class StepBoundary:
    """`agent/pre-step` 的可变载荷；插件可贡献边界 narration。"""

    agent: Agent
    session: Session
    turn: int
    step: int
    notices: list[str] = field(default_factory=list)


@dataclass
class ModelRequest:
    """`agent/prepare-request` 与 `llm/request` 共享的请求 envelope。"""

    agent: Agent
    session: Session
    turn: int
    step: int
    system_prompt: str
    messages: list[Message]
    tools: list[Tool]
    schemas: list[dict[str, Any]]


@dataclass
class ToolExecution:
    """工具管线载荷；post-execute 插件可以改写最终模型可见结果。"""

    agent: Agent
    session: Session
    call: ToolCall
    result: str = ""
    is_error: bool = False


class AgentRegistry:
    """保存当前进程中的实时 Agent，并按 id 返回精确实例。"""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> Callable[[], None]:
        if agent.id in self._agents:
            raise ValueError(f'Agent "{agent.id}" 已被注册')
        self._agents[agent.id] = agent
        agent._registered = True

        def unregister() -> None:
            if self._agents.get(agent.id) is agent:
                del self._agents[agent.id]
                agent._registered = False

        return unregister

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list(self) -> tuple[Agent, ...]:
        return tuple(self._agents.values())


class Agent:
    def __init__(
        self,
        ctx: Context,
        client: ChatClient,
        registry: ToolRegistry,
        assembler: PromptAssembler,
        *,
        variables: dict[str, str] | None = None,
        session: Session | None = None,
        agent_id: str = "main",
        session_id: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._client = client
        self._registry = registry
        self._assembler = assembler
        self._variables = variables
        self._session = session or Session()
        self._inbox = Inbox(self._session)
        self._turn_no = max(
            (
                int(event.data["turn"])
                for event in self._session.snapshot_events()
                if event.type == "turn/start"
            ),
            default=0,
        )
        prior_header = next(
            (
                event.data.get("header")
                for event in reversed(self._session.snapshot_events())
                if event.type == "request/header"
            ),
            None,
        )
        self._request_header = (
            json.dumps(
                prior_header,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if prior_header is not None
            else None
        )
        self._request_generation = self._session.replace_generation
        self._first_request = True
        self._registered = True
        self.id = agent_id
        self.session_id = session_id or uuid4().hex
        self.rpc_dispatcher: RpcDispatcher | None = None

    def followup(self, content: str) -> None:
        self._require_live()
        if not content.strip():
            raise ValueError("followup 内容不能为空")
        self._inbox.followup(Message(role="user", content=content))

    def steer(self, content: str) -> None:
        self._require_live()
        if not content.strip():
            raise ValueError("steer 内容不能为空")
        self._inbox.steer(Message(role="user", content=content))

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def session(self) -> Session:
        return self._session

    @property
    def inbox(self) -> Inbox:
        return self._inbox

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self._registry.all())

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._registry.schemas()

    def close(self) -> None:
        self._ctx.dispose()

    def run(self, max_turns: int = 5) -> Session:
        """同步处理 inbox；max_turns 限制本次调用处理的轮数。"""
        self._require_live()
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

    def _require_live(self) -> None:
        if not self._registered:
            raise RuntimeError(f'Agent "{self.id}" 已从注册表移除')

    def _run_turn(self, claimed: list[Message]) -> None:
        """执行一轮中的模型步骤和工具调用，最多运行 10 步。"""
        for step in range(1, 11):
            if step > 1:
                claimed = self._inbox.claim_step()

            boundary = StepBoundary(self, self._session, self._turn_no, step)
            self._ctx.emit("agent/pre-step", boundary)
            self._session.append("step/start", {"turn": self._turn_no, "step": step})
            completed = False
            try:
                system_prompt = self._assembler.render(self._variables)
                self._session.record_system_prompt(
                    system_prompt, turn=self._turn_no, step=step
                )
                for message in claimed:
                    self._session.append("user/message", {"content": message.content})
                for notice in boundary.notices:
                    self._session.append("user/message", {"content": notice})

                request = self._assemble_request(step, system_prompt)
                self._ctx.emit("agent/prepare-request", request)
                self._record_request_header(request)
                reply = cast(
                    Message,
                    self._ctx.waterfall("llm/request", request, self._request_model),
                )
                self._append_assistant(reply)

                if not reply.tool_calls:
                    completed = True
                else:
                    for call in reply.tool_calls:
                        self._run_tool(call)
            finally:
                self._session.append("step/end", {"turn": self._turn_no, "step": step})

            if completed and not self._inbox.has_next_step:
                return
        raise RuntimeError(f"第 {self._turn_no} 轮超过 10 个 step 仍未结束")

    def _assemble_request(self, step: int, system_prompt: str) -> ModelRequest:
        tools = self._registry.all()
        schemas = self._registry.schemas()
        return ModelRequest(
            agent=self,
            session=self._session,
            turn=self._turn_no,
            step=step,
            system_prompt=system_prompt,
            messages=self._session.derive_messages(),
            tools=tools,
            schemas=schemas,
        )

    def _record_request_header(self, request: ModelRequest) -> None:
        header: dict[str, Any] = {
            "config": {
                "provider": "deepseek-official",
                "model": self._client.MODEL,
            },
        }
        if request.schemas:
            header["tools"] = request.schemas
        fingerprint = json.dumps(
            header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        surface_changed = self._session.replace_generation != self._request_generation
        if (
            not self._first_request
            and fingerprint == self._request_header
            and not surface_changed
        ):
            return
        if self._first_request:
            reason = "resume" if self._request_header is not None else "initial"
        elif fingerprint != self._request_header:
            reason = "change"
        else:
            reason = "series"
        self._session.append(
            "request/header",
            {
                "header": header,
                "reason": reason,
            },
        )
        self._request_header = fingerprint
        self._request_generation = self._session.replace_generation
        self._first_request = False

    def _request_model(self, request: ModelRequest) -> Message:
        reply = self._client.chat(request.messages, request.tools)
        if not reply.content and not reply.reasoning_content and not reply.tool_calls:
            raise EmptyResponseError(
                "model returned a completed response with no content"
            )
        return reply

    def _append_assistant(self, reply: Message) -> None:
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
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in reply.tool_calls
                ],
            },
        )

    def _run_tool(self, call: ToolCall) -> None:
        self._session.append(
            "tool/call",
            {"call_id": call.id, "name": call.name, "arguments": call.arguments},
        )
        execution = ToolExecution(self, self._session, call)
        self._ctx.emit("tools/pre-execute", execution)
        cast(
            ToolExecution,
            self._ctx.waterfall("tools/execute", execution, self._execute_tool),
        )
        self._ctx.emit("tools/post-execute", execution)
        self._session.append(
            "tool/result",
            {
                "call_id": call.id,
                "content": execution.result,
                "is_error": execution.is_error,
            },
        )
        self._ctx.emit("tools/result", execution)

    def _execute_tool(self, execution: ToolExecution) -> ToolExecution:
        tool = self._registry.get(execution.call.name)
        if tool is None:
            execution.is_error = True
            execution.result = f"Error: 模型请求了未注册的工具 {execution.call.name!r}"
            return execution
        try:
            arguments = json.loads(execution.call.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("工具参数必须是 JSON 对象")
            execution.result = tool.execute(arguments)
        except Exception as error:  # noqa: BLE001 - 工具边界必须把失败配对成 result
            execution.is_error = True
            execution.result = f"工具执行出错: {error}"
        return execution
