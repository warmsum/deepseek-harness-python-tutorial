"""第 07 章：由会话事件保存的两级消息收件箱。

`followup` 写入下一轮队列，`steer` 写入下一步骤队列。每次插入和领取
都追加 `agent/inbox/spliced`，重新构造 Inbox 时可以从日志恢复尚未处理
的消息。领取方法属于循环内部；对外发送入口仍是 Agent。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from client import Message
from session import Session

InboxTarget = Literal["next-turn", "next-step"]


class Inbox:
    """把两条待处理队列投影到 Session 日志之上。"""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._state()

    def followup(self, message: Message) -> None:
        """把常规用户消息追加到下一轮队列。"""
        self._insert("next-turn", message)

    def steer(self, message: Message) -> None:
        """把中途引导追加到下一步骤队列。"""
        self._insert("next-step", message)

    def claim_turn(self) -> list[Message]:
        """领取全部 next-step 消息和最早的一条 next-turn 消息。"""
        state = self._state()
        claimed = list(state["next-step"])
        if claimed:
            self._splice("next-step", 0, len(claimed), [])
        next_turn = state["next-turn"]
        if next_turn:
            claimed.append(next_turn[0])
            self._splice("next-turn", 0, 1, [])
        return claimed

    def claim_step(self) -> list[Message]:
        """领取当前全部 next-step 输入。"""
        claimed = list(self._state()["next-step"])
        if claimed:
            self._splice("next-step", 0, len(claimed), [])
        return claimed

    def clear(self) -> None:
        """取消全部待处理消息，先清理 next-step，再清理 next-turn。"""
        state = self._state()
        if state["next-step"]:
            self._splice(
                "next-step", 0, len(state["next-step"]), [], outcome="canceled"
            )
        if state["next-turn"]:
            self._splice(
                "next-turn", 0, len(state["next-turn"]), [], outcome="canceled"
            )

    @property
    def next_turn(self) -> tuple[Message, ...]:
        return tuple(self._state()["next-turn"])

    @property
    def next_step(self) -> tuple[Message, ...]:
        return tuple(self._state()["next-step"])

    @property
    def pending(self) -> int:
        state = self._state()
        return len(state["next-turn"]) + len(state["next-step"])

    @property
    def has_next_step(self) -> bool:
        return bool(self._state()["next-step"])

    def _insert(self, target: InboxTarget, message: Message) -> None:
        if message.role != "user" or not isinstance(message.content, str):
            raise ValueError("Inbox 只接受带文本内容的 user 消息")
        self._splice(target, len(self._state()[target]), 0, [message])

    def _splice(
        self,
        target: InboxTarget,
        start: int,
        removed_count: int,
        inserted: list[Message],
        *,
        outcome: str | None = None,
    ) -> None:
        data: dict[str, object] = {
            "target": target,
            "start": start,
            "removed_count": removed_count,
            "inserted": [
                {"role": message.role, "content": message.content}
                for message in inserted
            ],
        }
        if outcome is not None:
            data["outcome"] = outcome
        self._session.append("agent/inbox/spliced", data)

    def _state(self) -> dict[InboxTarget, list[Message]]:
        state: dict[InboxTarget, list[Message]] = {
            "next-turn": [],
            "next-step": [],
        }
        for event in self._session.snapshot_events():
            if event.type != "agent/inbox/spliced":
                continue
            target = event.data.get("target")
            start = event.data.get("start")
            removed_count = event.data.get("removed_count")
            inserted = event.data.get("inserted")
            outcome = event.data.get("outcome")
            if not isinstance(target, str) or target not in state:
                raise ValueError(f"无效的 Inbox 事件 seq={event.seq}")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(removed_count, int)
                or isinstance(removed_count, bool)
                or not isinstance(inserted, tuple)
                or (outcome is not None and outcome != "canceled")
            ):
                raise ValueError(f"无效的 Inbox 事件 seq={event.seq}")
            queue = state[target]
            if start < 0 or removed_count < 0 or start + removed_count > len(queue):
                raise ValueError(f"无效的 Inbox splice 坐标 seq={event.seq}")
            messages = [_message_from_data(item, event.seq) for item in inserted]
            queue[start : start + removed_count] = messages
        return state


def _message_from_data(data: object, seq: int) -> Message:
    if not isinstance(data, Mapping):
        raise ValueError(f"无效的 Inbox 消息 seq={seq}")
    role = data.get("role")
    content = data.get("content")
    if role != "user" or not isinstance(content, str):
        raise ValueError(f"无效的 Inbox 消息 seq={seq}")
    return Message(role="user", content=content)
