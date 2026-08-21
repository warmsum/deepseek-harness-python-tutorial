"""第 10 章：模型真实调用文件工具完成一次受约束的修改。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from agent import DeepSeekClient, Tool, run_agent
from fs_tools import ObservationTracker, edit_file, glob, grep, read_file, write_file
from sandbox import WORKSPACE_WRITE, SandboxPolicy


def _path(workspace: Path, arguments: dict[str, Any]) -> Path:
    raw = arguments.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("path 必须是非空字符串")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else workspace / candidate


def build_tools(
    workspace: Path, policy: SandboxPolicy, tracker: ObservationTracker
) -> list[Tool]:
    return [
        Tool(
            "read",
            "读取工作区文件。修改已有文件前必须先调用 read。",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            lambda args: read_file(_path(workspace, args), tracker),
        ),
        Tool(
            "write",
            "创建文件或覆盖已经读取且未被外部修改的文件。",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            lambda args: write_file(
                _path(workspace, args), str(args["content"]), policy, tracker
            ),
        ),
        Tool(
            "edit",
            "替换文件中的一段原文；文件必须先读取。",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            lambda args: edit_file(
                _path(workspace, args),
                str(args["old_string"]),
                str(args["new_string"]),
                policy,
                tracker,
                bool(args.get("replace_all", False)),
            ),
        ),
        Tool(
            "grep",
            "在工作区文本文件中搜索正则表达式。",
            {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            lambda args: grep(workspace, str(args["pattern"])),
        ),
        Tool(
            "glob",
            "按 glob 模式列出工作区文件。",
            {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            lambda args: glob(workspace, str(args["pattern"])),
        ),
    ]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        target = workspace / "todo.txt"
        target.write_text("学习 sandbox\n学习 subagent\n", encoding="utf-8")
        policy = SandboxPolicy(WORKSPACE_WRITE, workspace)
        tracker = ObservationTracker()
        result = run_agent(
            DeepSeekClient(),
            build_tools(workspace, policy, tracker),
            (
                f"你是文件维护助手。工作区是 {workspace}。路径优先使用相对路径。"
                "修改已有文件前必须先 read，修改后再次 read 确认。"
            ),
            (
                "请实际使用文件工具：读取 todo.txt，把唯一的“学习 sandbox”改为"
                "“复习 sandbox”，然后重新读取并汇报结果。不要只描述操作。"
            ),
        )

        print("=== 模型发起的文件操作 ===")
        for trace in result.traces:
            print(f"{trace.name}({trace.arguments})")
            print(trace.result)
        print(f"\n模型最终回答: {result.final_text}")
        print(f"\n磁盘最终内容:\n{target.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
