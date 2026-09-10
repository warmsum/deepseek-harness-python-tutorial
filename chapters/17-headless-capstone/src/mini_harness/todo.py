"""第 13 章：todo —— 任务清单工具。

对应官方 packages/todo/tool-todo。核心语义（tool-todo/README.zh.md）：
1. 整体替换：模型每次调用都发送完整列表，不存在部分更新；
2. status 三值：pending / in_progress / completed；
3. 校验：空/重复 content 拒绝、未知键拒绝；
4. 每次调用追加 todo/write 事件（完整列表快照），后写覆盖先写。
"""

from __future__ import annotations

from dataclasses import dataclass

from .session import Session

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
VALID_STATUSES = (STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED)


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: str


def validate_todos(
    items: list[TodoItem], *, allow_parallel_in_progress: bool
) -> list[str]:
    """校验整个列表（整体替换前的完整检查）。"""
    errors: list[str] = []
    seen: set[str] = set()
    active = 0
    for item in items:
        content = item.content.strip()
        if not content:
            errors.append("content 必须是非空字符串")
            continue
        if content in seen:
            errors.append(f'重复的 content: "{content}"')
        seen.add(content)
        if item.status not in VALID_STATUSES:
            errors.append(
                f'无效的 status: "{item.status}"（只允许 pending/in_progress/completed）'
            )
        elif item.status == STATUS_IN_PROGRESS:
            active += 1
    if not allow_parallel_in_progress and active > 1:
        errors.append(f"最多只能有一个 in_progress 项（当前 {active} 个）")
    return errors


def todo_write(
    session: Session,
    items: list[TodoItem],
    *,
    allow_parallel_in_progress: bool,
) -> str:
    """校验完整列表并追加 todo/write 快照；回放时采用最后一份。"""
    errors = validate_todos(
        items, allow_parallel_in_progress=allow_parallel_in_progress
    )
    if errors:
        return "Error: invalid todos: " + "; ".join(errors)
    snapshot = [
        {"content": item.content.strip(), "status": item.status} for item in items
    ]
    session.append("todo/write", {"todos": snapshot})
    pending = sum(1 for item in items if item.status == STATUS_PENDING)
    in_progress = sum(1 for item in items if item.status == STATUS_IN_PROGRESS)
    completed = sum(1 for item in items if item.status == STATUS_COMPLETED)
    return (
        f"Updated todo list: {pending} pending, {in_progress} in progress, "
        f"{completed} completed."
    )
