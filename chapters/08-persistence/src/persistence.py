"""第 08 章：JSONL 持久化 —— 把会话日志写进磁盘。

对应官方 packages/session/session-persistence-jsonl。
教学版实现三个核心机制：
1. JSONL 格式：首行 header + 每行一条事件；
2. 原子发布：先写临时文件，再用无覆盖硬链接发布；
3. 崩溃修复：加载时截断残缺尾行，并为开放的工具、step、turn 合成收尾。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from uuid import uuid4

from session import Session, SessionEvent

HEADER_FORMAT = "mini-harness-jsonl"
HEADER_VERSION = 2
TOOL_NOT_STARTED = "TOOL_NOT_STARTED"
TOOL_OUTCOME_UNKNOWN = "TOOL_OUTCOME_UNKNOWN"


class JsonlStore:
    """会话的 JSONL 落盘与加载。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # 保存：首次原子发布，之后只追加
    # ------------------------------------------------------------------

    def save(self, session: Session) -> None:
        """首次用临时文件原子发布，之后只追加尚未落盘的事件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            lines = [
                _header_line(),
                *(_event_line(event) for event in session.snapshot_events()),
            ]
            tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                with tmp_path.open("x", encoding="utf-8") as file:
                    file.write("\n".join(lines) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.link(tmp_path, self.path)
            except FileExistsError as error:
                raise RuntimeError(f"会话文件已由另一个写入方创建: {self.path}") from error
            finally:
                tmp_path.unlink(missing_ok=True)
            return

        persisted, torn_offset = self._read_records()
        if torn_offset is not None:
            raise ValueError("会话文件存在残缺尾行；请先 load() 修复后再保存")
        current = session.snapshot_events()
        if len(persisted) > len(current) or tuple(persisted) != current[: len(persisted)]:
            raise ValueError("会话文件不是当前日志的前缀，拒绝覆盖既有历史")
        pending = current[len(persisted) :]
        if not pending:
            return
        with self.path.open("a", encoding="utf-8") as file:
            for event in pending:
                file.write(_event_line(event) + "\n")
            file.flush()
            os.fsync(file.fileno())

    # ------------------------------------------------------------------
    # 加载：校验 + 崩溃修复
    # ------------------------------------------------------------------

    def load(self) -> Session:
        """从磁盘重建会话。

        两项处理：
        1. header 校验：格式或版本不符时拒绝读取，避免误解其他格式；
        2. 崩溃修复：截断末尾残缺行，并为开放的工具、步骤和轮次生成
           对应的收尾事件。
        """
        if not self.path.exists():
            raise FileNotFoundError(f"会话文件不存在: {self.path}")

        events, torn_offset = self._read_records()
        if torn_offset is not None:
            with self.path.open("r+b") as file:
                file.truncate(torn_offset)
                file.flush()
                os.fsync(file.fileno())
        # 崩溃可能恰好发生在完整行写完之后，因此即使没有 torn tail，
        # 也要根据已持久化事件检查开放状态。
        _append_recovery_closers(events)
        return Session.from_log(events)

    def _read_records(self) -> tuple[list[SessionEvent], int | None]:
        """读取完整行。只有最后一个未换行片段可以按崩溃尾部修复。"""
        raw_bytes = self.path.read_bytes()
        if not raw_bytes:
            raise ValueError("空的会话文件")
        complete_bytes = len(raw_bytes)
        torn_offset: int | None = None
        if not raw_bytes.endswith(b"\n"):
            last_newline = raw_bytes.rfind(b"\n")
            if last_newline < 0:
                raise ValueError("会话文件缺少完整 header")
            complete_bytes = last_newline + 1
            torn_offset = complete_bytes

        try:
            complete_lines = raw_bytes[:complete_bytes].decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("会话文件包含无效 UTF-8") from error
        if not complete_lines:
            raise ValueError("会话文件缺少 header")
        try:
            header = json.loads(complete_lines[0])
        except json.JSONDecodeError as error:
            raise ValueError("会话文件 header 不是合法 JSON") from error
        if not isinstance(header, dict) or set(header) != {"format", "version"}:
            raise ValueError(f"无法识别的会话文件头: {header}")
        if header["format"] != HEADER_FORMAT or header["version"] != HEADER_VERSION:
            raise ValueError(f"无法识别的会话文件头: {header}")

        events: list[SessionEvent] = []
        for line_number, line in enumerate(complete_lines[1:], start=2):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"会话文件第 {line_number} 行损坏") from error
            required = {"seq", "type", "time", "data"}
            optional = {"surface_op", "source_event_seqs"}
            if (
                not isinstance(raw, dict)
                or not required <= set(raw)
                or not set(raw) <= required | optional
            ):
                raise ValueError(f"会话文件第 {line_number} 行不是合法事件")
            raw_sources = raw.get("source_event_seqs", [])
            if not isinstance(raw_sources, list):
                raise ValueError(
                    f"会话文件第 {line_number} 行的 source_event_seqs 不是数组"
                )
            events.append(
                SessionEvent(
                    seq=raw["seq"],
                    type=raw["type"],
                    time=raw["time"],
                    data=raw["data"],
                    surface_op=raw.get("surface_op"),
                    source_event_seqs=tuple(raw_sources),
                )
            )
        # 先走 Session 的连续 seq 与 lossless JSON 校验，再交还普通列表。
        return list(Session.from_log(events).snapshot_events()), torn_offset


def _header_line() -> str:
    return json.dumps({"format": HEADER_FORMAT, "version": HEADER_VERSION}, ensure_ascii=False)


def _event_line(event: SessionEvent) -> str:
    record: dict[str, object] = {
        "seq": event.seq,
        "type": event.type,
        "time": event.time,
        "data": event.data,
    }
    if event.surface_op is not None:
        record["surface_op"] = event.surface_op
    if event.source_event_seqs:
        record["source_event_seqs"] = event.source_event_seqs
    return json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
    )


def _append_recovery_closers(events: list[SessionEvent]) -> None:
    """为开放的工具、step 与 turn 依次补上崩溃收尾。"""
    open_turn: int | None = None
    open_step: tuple[int, int] | None = None
    open_calls: dict[str, tuple[str, int | None]] = {}
    for event in events:
        if event.type == "turn/start":
            open_turn = event.data["turn"]
            open_step = None
            open_calls.clear()
        elif event.type == "turn/end":
            open_turn = None
            open_step = None
            open_calls.clear()
        elif event.type == "step/start":
            open_step = (event.data["turn"], event.data["step"])
        elif event.type == "step/end":
            open_step = None
            open_calls.clear()
        elif event.type == "assistant/message":
            for call in event.data.get("tool_calls") or ():
                open_calls[call["id"]] = (call["name"], None)
        elif event.type == "tool/call":
            open_calls[event.data["call_id"]] = (event.data["name"], event.seq)
        elif event.type == "tool/result":
            open_calls.pop(event.data["call_id"], None)

    if open_turn is None:
        return
    timestamp = events[-1].time if events else time.time_ns() // 1_000_000
    for call_id, (name, call_seq) in open_calls.items():
        started = call_seq is not None
        code = TOOL_OUTCOME_UNKNOWN if started else TOOL_NOT_STARTED
        content = (
            f"Error [{code}]: tool {name!r} was recorded, but no result was "
            "durably recorded. Its outcome is unknown. Retry only if the operation "
            "is read-only or idempotent; if it may have side effects, first verify "
            "external state or ask the user. Do not retry blindly."
            if started
            else f"Error [{code}]: tool {name!r} was requested, but the Harness "
            "did not record that execution started. Retry it if it is still needed."
        )
        events.append(
            SessionEvent(
                seq=len(events),
                type="tool/result",
                time=timestamp,
                data={
                    "call_id": call_id,
                    "content": content,
                    "is_error": True,
                },
                surface_op="append",
                source_event_seqs=(call_seq,) if call_seq is not None else (),
            )
        )
    if open_step is not None:
        turn, step = open_step
        events.append(
            SessionEvent(
                seq=len(events),
                type="step/end",
                time=timestamp,
                data={"turn": turn, "step": step, "reason": "crashed"},
            )
        )
    if open_turn is not None:
        events.append(
            SessionEvent(
                seq=len(events),
                type="turn/end",
                time=timestamp,
                data={"turn": open_turn, "reason": "crashed"},
            )
        )
