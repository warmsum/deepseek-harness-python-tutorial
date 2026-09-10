"""第 14 章：子智能体委派与独立会话。

对应官方 packages/subagent/subagent + tool-subagent。
教学版沿用第 04 章讨论的“状态隔离”目标，但不使用那一章的 Context：
子 agent 使用独立的运行环境、会话和工具子集，只接收父 agent 提供的
task 描述，不自动继承父对话历史。

本章保留两个核心行为：
1. 上下文隔离：子 agent 看不到父历史，这正是 subagent 省 token 的原因；
2. 并行：一轮里的多个子任务同时跑（官方按 isConcurrencySafe 并行调度）。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Protocol
from uuid import uuid4

from client import DeepSeekClient, Tool
from session import Session


@dataclass(frozen=True)
class SubagentResult:
    """子 agent 的一次运行结果。

    对应官方 SubagentRun.result → { output, stopReason, diagnostic? }：
    output 只放子 agent 的回答，失败诊断单独存放。"""

    output: str
    stop_reason: str  # "completed" / "max-steps" / "interrupted" / "error"
    diagnostic: str | None = None


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


class _EitherCancellation:
    def __init__(self, first: CancellationSignal, second: CancellationSignal) -> None:
        self._first = first
        self._second = second

    def is_set(self) -> bool:
        return self._first.is_set() or self._second.is_set()


def run_subagent(
    client: DeepSeekClient,
    task: str,
    system_prompt: str,
    max_steps: int = 3,
    session: Session | None = None,
    cancelled: CancellationSignal | None = None,
) -> SubagentResult:
    """在独立 Session 中运行子 agent，只接收 task 和系统提示词。

    默认不继承父历史；需要继承已完成轮次时由调用方传入 fork 会话。"""
    child_session = session or Session()
    turn = 1 + sum(
        1 for event in child_session.snapshot_events() if event.type == "turn/start"
    )
    child_session.append("turn/start", {"turn": turn})

    partial: str = ""
    try:
        for step in range(1, max_steps + 1):
            if cancelled is not None and cancelled.is_set():
                child_session.append("turn/end", {"turn": turn, "reason": "interrupted"})
                return SubagentResult(partial, "interrupted")
            child_session.append("step/start", {"turn": turn, "step": step})
            interrupted = False
            completed = False
            try:
                child_session.record_system_prompt(
                    system_prompt, turn=turn, step=step
                )
                if step == 1:
                    child_session.append("user/message", {"content": task})
                reply = client.chat(child_session.derive_messages())
                if cancelled is not None and cancelled.is_set():
                    interrupted = True
                else:
                    child_session.append(
                        "assistant/message",
                        {
                            "content": reply.content,
                            **(
                                {"reasoning_content": reply.reasoning_content}
                                if reply.reasoning_content
                                else {}
                            ),
                            "tool_calls": [],
                        },
                    )
                    partial = reply.content or ""
                    completed = bool(reply.content)
            finally:
                child_session.append("step/end", {"turn": turn, "step": step})
            if interrupted:
                child_session.append(
                    "turn/end", {"turn": turn, "reason": "interrupted"}
                )
                return SubagentResult(partial, "interrupted")
            if completed:
                child_session.append("turn/end", {"turn": turn, "reason": "completed"})
                return SubagentResult(output=partial, stop_reason="completed")
        child_session.append("turn/end", {"turn": turn, "reason": "max-steps"})
        return SubagentResult(
            output=partial,
            stop_reason="max-steps",
        )
    except Exception as error:
        # 把运行失败写入 diagnostic，不把异常文本作为模型回答返回。
        child_session.append("turn/end", {"turn": turn, "reason": "error", "message": str(error)})
        return SubagentResult(
            output=partial,
            stop_reason="error",
            diagnostic=str(error),
        )


def fork_session(parent: Session) -> Session:
    """复制父日志的最后完整 turn 前缀；当前开放 turn 完全排除。"""
    last_turn_end = -1
    events = parent.snapshot_events()
    for index, event in enumerate(events):
        if event.type == "turn/end":
            last_turn_end = index
    if last_turn_end < 0:
        return Session()
    return Session.from_log(events[: last_turn_end + 1])


class ContinuableSubagent:
    """有独立 Session 的可继续子 Agent；消息由单线程队列按 FIFO 执行。"""

    def __init__(
        self,
        client: DeepSeekClient,
        system_prompt: str,
        *,
        seed: Session | None = None,
        id: str | None = None,
    ) -> None:
        self.id = id or f"child-{uuid4().hex}"
        self._client = client
        self._system_prompt = system_prompt
        self._session = seed or Session()
        self._cancelled = Event()
        self._state_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._status = "idle"
        self._closed = False

    @property
    def session(self) -> Session:
        return self._session

    @property
    def status(self) -> str:
        with self._state_lock:
            return self._status

    def submit_message(
        self, content: str, cancelled: CancellationSignal | None = None
    ) -> Future[SubagentResult]:
        """接收一条后续消息并立即返回；单 worker 保证投递顺序。"""
        if not content.strip():
            raise ValueError("content 必须是非空字符串")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("subagent 已关闭")
            return self._executor.submit(self._run_message, content, cancelled)

    def send_message(
        self, content: str, cancelled: CancellationSignal | None = None
    ) -> SubagentResult:
        """同步章节 API：仍走同一 FIFO 队列，但等待这一条消息完成。"""
        return self.submit_message(content, cancelled).result()

    def _run_message(self, content: str, cancelled: CancellationSignal | None) -> SubagentResult:
        with self._state_lock:
            if self._closed:
                return SubagentResult("", "interrupted")
            self._cancelled.clear()
            self._status = "running"
        signal: CancellationSignal = self._cancelled
        if cancelled is not None:
            signal = _EitherCancellation(self._cancelled, cancelled)
        result = run_subagent(
            self._client,
            content,
            self._system_prompt,
            session=self._session,
            cancelled=signal,
        )
        with self._state_lock:
            self._status = "interrupted" if result.stop_reason == "interrupted" else "idle"
        return result

    def interrupt(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.interrupt()
        self._executor.shutdown(wait=True, cancel_futures=True)


class SubagentManager:
    """按 owner 管理 continuable 子 Agent，隔离兄弟和其他根 Agent。"""

    def __init__(self, client: DeepSeekClient, system_prompt: str) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._children: dict[str, tuple[str, ContinuableSubagent]] = {}

    def create(
        self,
        owner_id: str,
        *,
        parent_session: Session | None = None,
        fork: bool = False,
    ) -> ContinuableSubagent:
        seed = fork_session(parent_session) if fork and parent_session is not None else None
        child = ContinuableSubagent(self._client, self._system_prompt, seed=seed)
        self._children[child.id] = (owner_id, child)
        return child

    def get(self, owner_id: str, child_id: str) -> ContinuableSubagent:
        record = self._children.get(child_id)
        if record is None or record[0] != owner_id:
            raise KeyError("subagent 不存在或不属于当前 owner")
        return record[1]

    def list(self, owner_id: str) -> tuple[ContinuableSubagent, ...]:
        return tuple(child for owner, child in self._children.values() if owner == owner_id)

    def close(self) -> None:
        for _owner, child in self._children.values():
            child.close()
        self._children.clear()


def run_subagents_parallel(
    specs: list[tuple[str, str]],  # (task, system_prompt)
    client: DeepSeekClient,
) -> list[SubagentResult]:
    """并行运行多个子 agent。

    DeepSeekClient.chat 是同步网络调用；线程池允许多个请求并发等待，
    总耗时通常接近最慢的子任务。
    官方在同一轮里对 isConcurrencySafe 的工具调用做并行调度，
    这里是对「多个 subagent 调用」的教学版并行。"""
    if not specs:
        return []
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [
            pool.submit(run_subagent, client, task, system_prompt) for task, system_prompt in specs
        ]
        return [future.result() for future in futures]


def create_subagent_tool(client: DeepSeekClient, child_system_prompt: str) -> Tool:
    """把 run_subagent 包装成第 02 章风格的 Tool。

    官方把「委派」做成一个基于已配置 provider 的模型工具——模型在需要
    拆分任务时主动调用它，参数就是子任务描述。"""

    def execute(args: dict[str, Any]) -> str:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("参数 task 必须是非空字符串")
        result = run_subagent(client, task, child_system_prompt)
        parts = [f"[{result.stop_reason}]"]
        if result.diagnostic is not None:
            parts.append(f"Diagnostic: {result.diagnostic}")
        if result.output:
            parts.append(result.output)
        return "\n".join(parts)

    return Tool(
        name="subagent",
        description=(
            "把一个独立的子任务委派给子 agent 执行并等待其完成。"
            "子 agent 看不到当前对话历史，只看到 task 描述。"
            "适合：独立的小任务、并行探索多个方向。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "要委派的子任务描述，要自包含（子 agent 没有上下文）",
                }
            },
            "required": ["task"],
        },
        execute=execute,
    )
