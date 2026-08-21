"""第 11 章：模型请求命令，审批通过后才真正执行。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from agent import DeepSeekClient, Tool, run_agent
from shell import APPROVAL_ALLOWED_ONCE, APPROVAL_REJECTED, ShellPolicy

ALLOWED_COMMANDS = {
    "echo approved > approved.txt",
    "cat approved.txt",
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()

        def approve(command: str) -> str:
            outcome = (
                APPROVAL_ALLOWED_ONCE
                if command in ALLOWED_COMMANDS
                else APPROVAL_REJECTED
            )
            print(f"审批请求: {command} -> {outcome}")
            return outcome

        policy = ShellPolicy(mode="workspace-write", approver=approve)

        def execute(arguments: dict[str, Any]) -> str:
            command = arguments.get("command")
            if not isinstance(command, str) or not command:
                raise ValueError("command 必须是非空字符串")
            result = policy.execute(command, cwd=str(workspace), timeout_seconds=5)
            return (
                f"exit_code={result.exit_code}\n"
                f"stdout={result.stdout.strip() or '(空)'}\n"
                f"stderr={result.stderr.strip() or '(空)'}"
            )

        shell_tool = Tool(
            "shell",
            "在当前工作区执行终端命令。所有命令都必须经过审批。",
            {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            execute,
        )
        result = run_agent(
            DeepSeekClient(),
            [shell_tool],
            (
                "你是命令执行助手。必须调用 shell，且只能使用用户给出的精确命令。"
                "命令能否执行由审批策略决定。"
            ),
            (
                "请依次执行精确命令 `echo approved > approved.txt` 和 "
                "`cat approved.txt`，再根据真实结果回答。"
            ),
        )

        print("\n=== 模型发起的命令 ===")
        for trace in result.traces:
            print(f"{trace.name}({trace.arguments})")
            print(trace.result)
        print(f"\n模型最终回答: {result.final_text}")
        created = workspace / "approved.txt"
        print(f"文件是否由获批命令创建: {created.exists()}")


if __name__ == "__main__":
    main()
