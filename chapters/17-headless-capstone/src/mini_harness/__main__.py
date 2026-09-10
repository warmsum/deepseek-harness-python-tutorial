"""mini_harness 入口：一次性任务运行器。

用法（在项目根目录运行已声明的命令行入口）：
    uv run mini-harness "你的任务"

也可以在本章 src 目录运行：
    uv run python -m mini_harness "你的任务"

对应官方 bundle/headless 的 runner 语义：
创建 Agent、把任务作为普通用户消息提交、等待完全停稳、
把最后一条 assistant 文本写入 stdout；最终 turn/end 完成 → 退出码 0，
否则 1。进程不打开任何监听端口。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

from .agent import Agent, AgentRegistry
from .bundle import BundleConfig, headless_bundle
from .capabilities import load_settings_document
from .cordis import Context
from .persistence import JsonlStore
from .session import Session

SESSION_DIR = Path(".mini-harness") / "sessions"


def build_agent(
    *,
    enable_console_questions: bool = True,
    checkpoint_flush: Callable[[Session], None] | None = None,
) -> Agent:
    """创建 Context 并挂载 headless Bundle；能力由插件依赖自动接线。"""
    ctx = Context()
    ctx.plugin(
        headless_bundle,
        BundleConfig(
            settings_document=load_settings_document(),
            enable_console_questions=enable_console_questions,
            checkpoint_flush=checkpoint_flush,
        ),
    )
    agents = cast(AgentRegistry, ctx.require("agents"))
    agent = agents.get("main")
    if agent is None:
        raise RuntimeError("headless Bundle 未创建 main Agent")
    return agent


def run_task(task: str, session_file: str | Path | None = None) -> tuple[str, bool]:
    """运行一次性任务，返回最后一条 assistant 文本和完成状态。"""
    path = Path(session_file) if session_file is not None else _new_session_file()
    store = JsonlStore(path)
    agent = build_agent(checkpoint_flush=store.save)

    try:
        agent.followup(task)
        session = agent.run()
        final_text = ""
        for message in session.derive_messages():
            if message.role == "assistant" and message.content:
                final_text = message.content
        turn_ends = [
            event for event in session.snapshot_events() if event.type == "turn/end"
        ]
        completed = bool(turn_ends and turn_ends[-1].data.get("reason") == "completed")
        return final_text, completed
    finally:
        try:
            store.save(agent.session)
        finally:
            agent.close()


def _new_session_file() -> Path:
    """为一次性任务分配新日志，避免下一次运行覆盖上一份会话。"""
    return SESSION_DIR / f"{uuid4().hex}.jsonl"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-harness",
        description="运行一个任务并输出最终回答，或以 stdio JSON-RPC 模式启动。",
    )
    parser.add_argument(
        "--rpc", action="store_true", help="从标准输入逐行处理 JSON-RPC 请求"
    )
    parser.add_argument("task", nargs="*", help="要交给智能体的一次性任务")
    return parser


def main() -> None:
    parser = _argument_parser()
    arguments = parser.parse_args()
    if arguments.rpc:
        if arguments.task:
            parser.error("--rpc 不能与任务文本同时使用")
        _run_rpc()
        return
    task = " ".join(arguments.task).strip()
    if not task:
        parser.print_usage(sys.stderr)
        print('mini-harness: error: 缺少任务，例如 mini-harness "运行测试"', file=sys.stderr)
        sys.exit(1)
    try:
        final_text, completed = run_task(task)
    except Exception as error:  # noqa: BLE001 - CLI 将运行失败映射为退出码 1
        print(f"mini-harness: {error}", file=sys.stderr)
        sys.exit(1)
    if final_text:
        print(final_text)
    sys.exit(0 if completed else 1)


def _run_rpc() -> None:
    """stdio 上的 JSON-RPC line transport；不打开监听端口。"""
    store = JsonlStore(_new_session_file())
    agent = build_agent(
        enable_console_questions=False,
        checkpoint_flush=store.save,
    )
    dispatcher = agent.rpc_dispatcher
    if dispatcher is None:
        raise RuntimeError("JSON-RPC dispatcher 未组装")
    try:
        for line in sys.stdin:
            response = dispatcher.dispatch(line)
            print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        try:
            store.save(agent.session)
        finally:
            agent.close()


if __name__ == "__main__":
    main()
