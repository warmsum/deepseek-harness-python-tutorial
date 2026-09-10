"""第 09 章：过大纯文本工具结果的外存接口、本地实现与策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class SaveTextSpill:
    session_id: str
    tool_name: str
    call_id: str
    label: str
    suggested_name: str
    content: str


@dataclass(frozen=True)
class SpillRef:
    locator: str
    bytes: int
    retrieval_hint: str


class SpillStore(Protocol):
    def save_text(self, request: SaveTextSpill) -> SpillRef: ...


class LocalSpillStore:
    """把内容放到 session 私有目录；suggested_name 只作为文件名提示。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def save_text(self, request: SaveTextSpill) -> SpillRef:
        owner = _safe_segment(request.session_id, "session")
        stem = _safe_segment(Path(request.suggested_name).stem, "spill")
        suffix = Path(request.suggested_name).suffix
        if not re.fullmatch(r"\.[a-zA-Z0-9]{1,10}", suffix):
            suffix = ".txt"
        directory = self.root / owner
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / f"{stem}-{uuid4().hex}{suffix}"
        target.write_text(request.content, encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
        return SpillRef(
            locator=str(target),
            bytes=len(request.content.encode("utf-8")),
            retrieval_hint="Use the read tool with offset and limit to inspect this file.",
        )


class SpillPolicy:
    def __init__(
        self,
        max_inline_bytes: int | None,
        store: SpillStore | None,
    ) -> None:
        if max_inline_bytes is not None and (
            not isinstance(max_inline_bytes, int)
            or isinstance(max_inline_bytes, bool)
            or max_inline_bytes < 0
        ):
            raise ValueError("max_inline_bytes 必须是非负整数或 None")
        self.max_inline_bytes = max_inline_bytes
        self.store = store

    def transform(
        self,
        text: str,
        *,
        session_id: str,
        tool_name: str,
        call_id: str,
        nested: bool = False,
    ) -> str:
        """尝试外存；保存失败、read 或嵌套调用均保留原文。"""
        cap = self.max_inline_bytes
        total = len(text.encode("utf-8"))
        if (
            cap is None
            or total <= cap
            or nested
            or tool_name == "read"
            or self.store is None
        ):
            return text
        try:
            ref = self.store.save_text(
                SaveTextSpill(
                    session_id=session_id,
                    tool_name=tool_name,
                    call_id=call_id,
                    label="result",
                    suggested_name=f"{tool_name}.txt",
                    content=text,
                )
            )
        except Exception:  # noqa: BLE001 - 外存失败时保留原工具结果
            return text
        worst_notice = _notice(total, ref)
        preview_budget = max(0, cap - len(("\n\n" + worst_notice).encode("utf-8")))
        preview = _head_tail_utf8(text, preview_budget)
        omitted = total - len(preview.encode("utf-8"))
        notice = _notice(omitted, ref)
        replacement = f"{preview}\n\n{notice}" if preview else notice
        if len(replacement.encode("utf-8")) > cap:
            return text
        return replacement


def _notice(omitted: int, ref: SpillRef) -> str:
    return (
        f"(Omitted {omitted} bytes. Full formatted result stored at: "
        f"{ref.locator}. {ref.retrieval_hint})"
    )


def _head_tail_utf8(text: str, budget: int) -> str:
    head_budget = (budget + 1) // 2
    tail_budget = budget // 2
    head = _take_utf8_prefix(text, head_budget)
    remaining = text[len(head) :]
    tail = _take_utf8_suffix(remaining, tail_budget)
    return head + tail


def _take_utf8_prefix(text: str, budget: int) -> str:
    used = 0
    kept: list[str] = []
    for character in text:
        size = len(character.encode("utf-8"))
        if used + size > budget:
            break
        kept.append(character)
        used += size
    return "".join(kept)


def _take_utf8_suffix(text: str, budget: int) -> str:
    used = 0
    kept: list[str] = []
    for character in reversed(text):
        size = len(character.encode("utf-8"))
        if used + size > budget:
            break
        kept.append(character)
        used += size
    return "".join(reversed(kept))


def _safe_segment(value: str, fallback: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    return safe[:80] or fallback
