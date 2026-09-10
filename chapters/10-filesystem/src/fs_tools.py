"""第 10 章：供模型调用的文件操作。

五个工具：read_file / write_file / edit_file / grep / glob。
外加「读后写」观察策略（对应官方 fs-observation-policy 的 CAS 思想）：
- 修改已存在的文件，必须先 read 过它（FS_NOT_OBSERVED）；
- read 之后文件被外部改动过，写入要拒绝（FS_STALE_VERSION），
  防止模型依据旧内容覆盖外部更新。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sandbox import SandboxPolicy

# ---------------------------------------------------------------------------
# 观察策略：read 时记录 mtime，write 前对比
# ---------------------------------------------------------------------------


@dataclass
class ObservationTracker:
    """读后写版本检查：记录模型最后一次读取文件时的修改时间。

    键统一用 resolve() 规范化——macOS 上 /var 是指向 /private/var 的
    符号链接，不做规范化会出现「读的是 A、写检查的是 B」的假阴性。"""

    _observed: dict[str, int] = field(default_factory=dict)

    def record_read(self, path: Path) -> None:
        key = str(path.resolve())
        try:
            self._observed[key] = path.stat().st_mtime_ns
        except FileNotFoundError:
            self._observed.pop(key, None)

    def check_write(self, path: Path) -> None:
        """写入前检查：没读过 → 拒绝；读过但 mtime 变了 → 拒绝。"""
        key = str(path.resolve())
        if key not in self._observed:
            raise PermissionError(
                f"[FS_NOT_OBSERVED] 修改 {path} 之前必须先 read 它"
            )
        current = path.stat().st_mtime_ns
        if current != self._observed[key]:
            raise PermissionError(
                f"[FS_STALE_VERSION] {path} 自上次读取后被外部修改"
                f"（mtime 变化），请重新 read 后再写"
            )


# ---------------------------------------------------------------------------
# 五个文件工具
# ---------------------------------------------------------------------------


def read_file(path: Path, tracker: ObservationTracker) -> str:
    """读文件，带行号与页脚（官方 read 的格式）。读取同时记录观察。"""
    text = path.read_text(encoding="utf-8")
    tracker.record_read(path)
    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1:>4}: {line}" for i, line in enumerate(lines))
    return f"{numbered}\n\n(End of file - total {len(lines)} lines)"


def write_file(
    path: Path, content: str, policy: SandboxPolicy, tracker: ObservationTracker
) -> str:
    """全量写入（创建或覆盖）。已存在文件必须先读过且未被外部修改。"""
    target = policy.fence_write(path)
    if target.exists():
        tracker.check_write(target)
    target.write_text(content, encoding="utf-8")
    tracker.record_read(target)
    return f"written {len(content)} chars to {target}"


def edit_file(
    path: Path,
    old_string: str,
    new_string: str,
    policy: SandboxPolicy,
    tracker: ObservationTracker,
    replace_all: bool = False,
) -> str:
    """执行字符串替换；默认要求 old_string 唯一匹配。"""
    target = policy.fence_write(path)
    if not target.exists():
        raise FileNotFoundError(f"[FS_NOT_FOUND] {target} 不存在")
    if not old_string:
        raise ValueError("[FS_INVALID_EDIT] old_string 不能为空")
    tracker.check_write(target)
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ValueError(f"[FS_EDIT_NOT_FOUND] old_string 未在 {target} 中找到")
    if count > 1 and not replace_all:
        raise ValueError(
            f"[FS_AMBIGUOUS_EDIT] old_string 在 {target} 中匹配了 {count} 处；"
            "请提供更具体的 old_string，或设置 replace_all=True"
        )
    updated = text.replace(old_string, new_string)
    target.write_text(updated, encoding="utf-8")
    tracker.record_read(target)
    return f"updated {target} ({count} 处替换)"


def grep(workspace: Path, pattern: str) -> str:
    """正则搜索工作区内全部文本文件（官方用打包 ripgrep，教学版用 re）。"""
    compiled = re.compile(pattern)
    hits: list[str] = []
    for file in sorted(workspace.rglob("*")):
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # 二进制/不可读文件跳过
        for line_no, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                hits.append(f"{file.relative_to(workspace)}:{line_no}: {line.strip()}")
    return "\n".join(hits) if hits else "(无匹配)"


def glob(workspace: Path, pattern: str) -> str:
    """按 glob 模式列出文件（* 匹配任意层级的文件名）。"""
    matched = sorted(
        str(p.relative_to(workspace)) for p in workspace.rglob(pattern) if p.is_file()
    )
    return "\n".join(matched) if matched else "(无匹配)"
