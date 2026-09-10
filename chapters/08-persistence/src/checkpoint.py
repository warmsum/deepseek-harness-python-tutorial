"""第 08 章：模型请求与工具副作用之前的语义 checkpoint。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from session import Session


@dataclass(frozen=True)
class CheckpointPolicy:
    """决定 flush 时机；保存失败时异常直接返回给调用方。"""

    flush: Callable[[Session], None]

    def before_model(self, session: Session) -> None:
        """完整请求前缀必须先持久化，再调用模型。"""
        self.flush(session)

    def before_tool(self, session: Session, *, nested: bool = False) -> None:
        """顶层工具正文前持久化 call；嵌套分派复用外层 checkpoint。"""
        if not nested:
            self.flush(session)

    def before_step(self, session: Session) -> None:
        """下一 step 派生请求前，持久化上一响应和有序工具结果。"""
        self.flush(session)

    def before_retry(self, session: Session) -> None:
        """退避开始前持久化已调度的 retry 事件。"""
        self.flush(session)
