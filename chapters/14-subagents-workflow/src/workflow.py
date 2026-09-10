"""第 14 章：带并发和总量上限的 Python Workflow 教学引擎。

官方在 Worker Thread 中执行 JavaScript 编排脚本。教学版直接接收 Python
callable，保留 parallel、无 barrier 的逐项 pipeline、失败项变为 ``None``
和上限语义。线程不隔离代码权限，因此输入必须来自受信任的 Python 代码。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class WorkflowMeta:
    name: str
    description: str


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    agents_started: int
    value: list[Any]
    stop_reason: str
    error: str | None = None


Stage = Callable[[Any, Any, int], Any]


class WorkflowEngine:
    def __init__(self, max_concurrency: int = 4, max_agents: int = 32) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (max_concurrency, max_agents)
        ):
            raise ValueError("workflow 上限必须是正整数")
        self.max_concurrency = max_concurrency
        self.max_agents = max_agents

    def parallel(self, thunks: Sequence[Callable[[], Any]]) -> list[Any]:
        """并发运行并保持输入顺序；普通任务失败投影为 ``None``。"""
        self._check_count(len(thunks))
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [pool.submit(_settle, thunk) for thunk in thunks]
            return [future.result() for future in futures]

    def pipeline(self, items: Iterable[Any], *stages: Stage) -> list[Any]:
        """每个 item 依次完成各阶段，阶段之间不设置全局 barrier。"""
        materialized = list(items)
        self._check_count(len(materialized) * max(1, len(stages)))

        def run_item(index: int, item: Any) -> Any:
            value = item
            for stage in stages:
                try:
                    value = stage(value, item, index)
                except Exception:
                    return None
            return value

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [
                pool.submit(run_item, index, item)
                for index, item in enumerate(materialized)
            ]
            return [future.result() for future in futures]

    def run(
        self,
        meta: WorkflowMeta,
        tasks: Sequence[Callable[[], Any]],
    ) -> WorkflowResult:
        if not meta.name or not meta.description:
            raise ValueError("workflow meta 需要 name 和 description")
        run_id = f"workflow-{uuid4().hex}"
        try:
            value = self.parallel(tasks)
        except Exception as error:
            return WorkflowResult(run_id, 0, [], "error", str(error))
        return WorkflowResult(run_id, len(tasks), value, "completed")

    def _check_count(self, count: int) -> None:
        if count > self.max_agents:
            raise ValueError(
                f"workflow agent 上限为 {self.max_agents}，本次需要 {count}"
            )


def _settle(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except Exception:
        return None
