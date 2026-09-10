"""事件日志：会话状态的唯一事实来源（第 05 章首次实现）。

对应官方 packages/core/session（事件溯源的会话日志）。
核心思想：
1. 日志是 append-only（只追加）：事件一旦写入永不修改；
2. 消息历史是派生视图：derive_messages() 每次从日志投影；
3. 模型请求、持久化、界面展示和重放都读取同一份日志。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import copysign, isfinite
from typing import Any, Never

from client import Message, ToolCall


class FrozenDict(dict[str, Any]):
    """JSON 可序列化、但写入后不可修改的字典。"""

    @staticmethod
    def _immutable() -> Never:
        raise TypeError("FrozenDict 不可修改")

    def __setitem__(self, key: str, value: Any) -> None:
        self._immutable()

    def __delitem__(self, key: str) -> None:
        self._immutable()

    def __ior__(self, value: Any) -> Never:  # type: ignore[misc]
        self._immutable()

    def clear(self) -> None:
        self._immutable()

    def pop(self, key: str, default: Any = None) -> Any:
        self._immutable()

    def popitem(self) -> Never:
        self._immutable()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._immutable()

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._immutable()

    @classmethod
    def build(cls, items: dict[str, Any]) -> "FrozenDict":
        frozen = cls()
        for key, value in items.items():
            dict.__setitem__(frozen, key, value)
        return frozen


@dataclass(frozen=True)
class SessionEvent:
    """日志里的一条事件。

    - seq：从 0 开始连续递增（重放校验靠它）；
    - type：事件类型（turn/start、user/message、tool/result……）；
    - time：Unix 毫秒时间戳；
    - data：事件内容，写入时冻结，之后不可修改。
    - surface_op：消息事件怎样进入模型可见表层。
    """

    seq: int
    type: str
    time: int
    data: Mapping[str, Any]
    surface_op: str | Mapping[str, Any] | None = None
    source_event_seqs: tuple[int, ...] = ()


SURFACE_EVENT_TYPES = frozenset(
    {"system/message", "user/message", "assistant/message", "tool/result"}
)


class Session:
    """一次对话的全部历史：一条只追加的事件日志。"""

    def __init__(self) -> None:
        self._log: list[SessionEvent] = []
        self._snapshot: tuple[SessionEvent, ...] | None = None
        self._listeners: list[Any] = []

    # ------------------------------------------------------------------
    # 追加：校验 + 冻结 + 通知
    # ------------------------------------------------------------------

    def append(
        self,
        type: str,
        data: dict[str, Any],
        *,
        surface_op: str | dict[str, Any] | None = None,
        source_event_seqs: tuple[int, ...] = (),
    ) -> SessionEvent:
        """追加一条事件。三个动作：

        1. 校验 data 是可序列化的纯 JSON（拒绝函数、集合等）；
        2. 冻结 data，防止后续代码修改已记录内容；
        3. 使缓存快照失效，并通知持久化等订阅者。
        """
        if not isinstance(type, str) or not type:
            raise ValueError("事件 type 必须是非空字符串")
        if not isinstance(data, dict):
            raise TypeError("事件 data 必须是对象")
        if type in SURFACE_EVENT_TYPES:
            surface_op = "append" if surface_op is None else surface_op
        elif surface_op is not None or source_event_seqs:
            raise ValueError(f'非表层事件 "{type}" 不能携带表层元数据')
        if type == "request/header":
            header = data.get("header")
            if not isinstance(header, dict):
                raise TypeError("request/header 的 header 必须是对象")
            if "system" in header:
                raise ValueError("request/header 不能包含 system；请使用 system/message")
            if header.get("tools") in ([], ()):
                raise ValueError("request/header 应省略空 tools")
        if type == "assistant/message" and source_event_seqs:
            raise ValueError("assistant/message 不能声明 source_event_seqs")
        frozen_data = _freeze_json(data)
        if not isinstance(frozen_data, FrozenDict):
            raise TypeError("事件 data 必须是对象")
        frozen_op = _freeze_json(surface_op)
        frozen_sources = _freeze_json(source_event_seqs)
        if frozen_op is not None and not isinstance(frozen_op, (str, FrozenDict)):
            raise TypeError("surface_op 必须是 append 或替换对象")
        if not isinstance(frozen_sources, tuple) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in frozen_sources
        ):
            raise TypeError("source_event_seqs 必须是整数序列")
        event = SessionEvent(
            seq=len(self._log),
            type=type,
            time=_now(),
            data=frozen_data,
            surface_op=frozen_op,
            source_event_seqs=frozen_sources,
        )
        _surface_events([*self._log, event])
        self._log.append(event)
        self._snapshot = None
        for listener in list(self._listeners):
            listener(event)
        return event

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def event_at(self, seq: int) -> SessionEvent | None:
        """读取一个已存在的事件位置；越界时返回 None。"""
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError("seq 必须是非负整数")
        return self._log[seq] if seq < len(self._log) else None

    def snapshot_events(
        self, start: int = 0, end: int | None = None
    ) -> tuple[SessionEvent, ...]:
        """返回半开区间 ``[start, end)`` 的稳定事件快照。"""
        stop = len(self._log) if end is None else end
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, stop)
        ) or not 0 <= start <= stop <= len(self._log):
            raise ValueError("事件快照范围无效")
        if start != 0 or stop != len(self._log):
            return tuple(self._log[start:stop])
        if self._snapshot is None:
            self._snapshot = tuple(self._log)
        return self._snapshot

    def surface_events(self) -> tuple[SessionEvent, ...]:
        """返回按 append/replace 规则折叠后的模型可见事件。"""
        return tuple(_surface_events(self._log))

    def record_system_prompt(
        self, content: str, *, turn: int, step: int
    ) -> SessionEvent | None:
        """记录当前系统提示词；变化时用新事件替换原有头节点。"""
        if not isinstance(content, str):
            raise TypeError("系统提示词必须是字符串")
        nodes = _surface_events(self._log)
        current = next(
            (event for event in nodes if event.type == "system/message"), None
        )
        if current is None:
            if nodes:
                raise ValueError("首条模型可见事件必须是 system/message")
            return self.append(
                "system/message",
                {"turn": turn, "step": step, "content": content},
            )
        if current.data.get("content") == content:
            return None
        return self.append(
            "system/message",
            {"turn": turn, "step": step, "content": content},
            surface_op={
                "op": "replace",
                "start_seq": current.seq,
                "end_seq": current.seq,
            },
            source_event_seqs=(current.seq,),
        )

    @property
    def replace_generation(self) -> int:
        """已提交的表层替换次数。"""
        return sum(
            1 for event in self._log if isinstance(event.surface_op, Mapping)
        )

    @property
    def seq(self) -> int:
        """下一条事件的序号，恒等于日志长度。"""
        return len(self._log)

    def subscribe(self, listener: Any) -> Any:
        """订阅新事件（返回解绑函数）。持久化插件挂在这里。"""
        self._listeners.append(listener)
        active = True

        def unsubscribe() -> None:
            nonlocal active
            if not active:
                return
            active = False
            self._listeners.remove(listener)

        return unsubscribe

    # ------------------------------------------------------------------
    # 投影：日志 → 模型看到的消息历史
    # ------------------------------------------------------------------

    def derive_messages(self) -> list[Message]:
        """把日志投影成 LLM 消息历史。

        四种事件会投影成消息（对应官方 surface 层）：
          system/message    → role="system"
          user/message      → role="user"
          assistant/message → role="assistant"（含 reasoning_content、tool_calls）
          tool/result       → role="tool"（带 tool_call_id）
        其余事件（turn/start、tool/call、turn/end……）只记日志，不发给模型。
        """
        messages: list[Message] = []
        for event in _surface_events(self._log):
            message = _derive_event_message(event)
            if message is not None:
                messages.append(message)
        return messages

    # ------------------------------------------------------------------
    # 重放：从既有日志重建会话
    # ------------------------------------------------------------------

    @classmethod
    def from_log(cls, events: Iterable[SessionEvent]) -> "Session":
        """从既有日志重建会话（恢复/重放的入口）。

        校验 seq 从 0 连续。发现缺号时拒绝恢复，避免使用残缺历史。
        """
        session = cls()
        for index, event in enumerate(events):
            if (
                not isinstance(event.seq, int)
                or isinstance(event.seq, bool)
                or event.seq != index
            ):
                raise ValueError(
                    f"重放失败：第 {index} 个事件 seq 为 {event.seq}（应为 {index}）"
                )
            if not isinstance(event.type, str) or not event.type:
                raise ValueError(f"重放失败：第 {index} 个事件 type 无效")
            if (
                not isinstance(event.time, int)
                or isinstance(event.time, bool)
                or abs(event.time) > 2**53 - 1
            ):
                raise ValueError(f"重放失败：第 {index} 个事件 time 无效")
            frozen_data = _freeze_json(event.data)
            if not isinstance(frozen_data, FrozenDict):
                raise ValueError(f"重放失败：第 {index} 个事件 data 必须是对象")
            if event.type == "request/header":
                header = frozen_data.get("header")
                if not isinstance(header, FrozenDict):
                    raise TypeError(
                        f"重放失败：第 {index} 个 request/header 缺少 header"
                    )
                if "system" in header:
                    raise ValueError(
                        "request/header 不能包含 system；请使用 system/message"
                    )
                if header.get("tools") == ():
                    raise ValueError("request/header 应省略空 tools")
            if event.type == "assistant/message" and event.source_event_seqs:
                raise ValueError("assistant/message 不能声明 source_event_seqs")
            frozen_op = _freeze_json(event.surface_op)
            frozen_sources = _freeze_json(event.source_event_seqs)
            if frozen_op is not None and not isinstance(frozen_op, (str, FrozenDict)):
                raise TypeError(f"重放失败：第 {index} 个事件 surface_op 无效")
            if not isinstance(frozen_sources, tuple) or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in frozen_sources
            ):
                raise TypeError(
                    f"重放失败：第 {index} 个事件 source_event_seqs 无效"
                )
            session._log.append(
                SessionEvent(
                    seq=event.seq,
                    type=event.type,
                    time=event.time,
                    data=frozen_data,
                    surface_op=frozen_op,
                    source_event_seqs=frozen_sources,
                )
            )
            _surface_events(session._log)
        return session


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _now() -> int:
    return time.time_ns() // 1_000_000


def _freeze_json(value: Any, _path: set[int] | None = None) -> Any:
    """冻结 + lossless JSON 校验。拒绝异常数字、非字符串键和循环引用。"""
    path = _path if _path is not None else set()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("事件 data 的整数超出 JSON 安全范围")
        return value
    if isinstance(value, float):
        if not isfinite(value) or (value == 0.0 and copysign(1.0, value) < 0):
            raise ValueError("事件 data 不能包含非有限数或负零")
        return value
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in path:
            raise ValueError("事件 data 不能包含循环引用")
        path.add(marker)
        try:
            return tuple(_freeze_json(item, path) for item in value)
        finally:
            path.remove(marker)
    if isinstance(value, dict):
        marker = id(value)
        if marker in path:
            raise ValueError("事件 data 不能包含循环引用")
        path.add(marker)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("事件 data 的对象键必须是字符串")
                result[key] = _freeze_json(item, path)
            return FrozenDict.build(result)
        finally:
            path.remove(marker)
    raise ValueError(f"事件 data 不能包含 {type(value).__name__}（仅限纯 JSON）")


def _derive_event_message(event: SessionEvent) -> Message | None:
    """单条事件的投影规则（对应官方 deriveEventMessage）。"""
    if event.type == "system/message":
        content = event.data.get("content")
        return Message(role="system", content=content) if content else None
    if event.type == "user/message":
        return Message(role="user", content=event.data["content"])
    if event.type == "assistant/message":
        raw_calls = event.data.get("tool_calls") or []
        tool_calls = tuple(
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in raw_calls
        )
        reasoning_content = event.data.get("reasoning_content")
        # 文本、思考与工具调用都没有的事件不投影（它只是日志记录）
        if event.data.get("content") is None and not reasoning_content and not tool_calls:
            return None
        return Message(
            role="assistant",
            content=event.data.get("content"),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
    if event.type == "tool/result":
        return Message(
            role="tool",
            content=event.data["content"],
            tool_call_id=event.data["call_id"],
        )
    return None


def _surface_events(events: list[SessionEvent]) -> list[SessionEvent]:
    """折叠模型可见事件，并校验替换范围与来源。"""
    nodes: list[SessionEvent] = []
    for event in events:
        if event.type not in SURFACE_EVENT_TYPES:
            if event.surface_op is not None or event.source_event_seqs:
                raise ValueError(f'非表层事件 "{event.type}" 不能携带表层元数据')
            continue
        if (
            len(set(event.source_event_seqs)) != len(event.source_event_seqs)
            or any(
                isinstance(source, bool) or source < 0 or source >= event.seq
                for source in event.source_event_seqs
            )
        ):
            raise ValueError("source_event_seqs 必须唯一并指向更早事件")
        operation = event.surface_op
        if operation == "append":
            nodes.append(event)
            continue
        if not isinstance(operation, Mapping) or set(operation) != {
            "op",
            "start_seq",
            "end_seq",
        }:
            raise ValueError("表层事件必须声明 append 或规范替换对象")
        start = operation.get("start_seq")
        end = operation.get("end_seq")
        if (
            operation.get("op") != "replace"
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start > end
            or end >= event.seq
        ):
            raise ValueError("surface replacement 范围无效")
        start_index = next(
            (index for index, node in enumerate(nodes) if node.seq == start), None
        )
        end_index = next(
            (index for index, node in enumerate(nodes) if node.seq == end), None
        )
        if start_index is None or end_index is None or start_index > end_index:
            raise ValueError("surface replacement 引用了不存在的表层节点")
        shadowed = nodes[start_index : end_index + 1]
        shadowed_seqs = {node.seq for node in shadowed}
        if not shadowed_seqs.issubset(set(event.source_event_seqs)):
            raise ValueError("source_event_seqs 必须包含全部被替换的表层节点")
        if event.type == "tool/result":
            if len(shadowed) != 1 or shadowed[0].type != "tool/result":
                raise ValueError("tool/result 替换必须只覆盖一条现有 tool/result")
            original = dict(shadowed[0].data)
            replacement = dict(event.data)
            original.pop("content", None)
            replacement.pop("content", None)
            if original != replacement:
                raise ValueError("tool/result 替换只能修改 content")
        if (
            start_index == 0
            and nodes[0].type == "system/message"
            and (event.type != "system/message" or len(shadowed) != 1)
        ):
            raise ValueError("首个 system/message 只能由 system/message 单点替换")
        nodes[start_index : end_index + 1] = [event]
    return nodes
