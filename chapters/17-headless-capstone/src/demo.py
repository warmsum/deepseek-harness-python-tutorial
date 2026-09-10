"""第 17 章 demo：使用组装后的 mini_harness 运行完整任务。

运行（在项目根目录，需要 .env）：
    uv run python chapters/17-headless-capstone/src/demo.py

演示：
1. 模块清单：前 16 章各贡献了哪一块；
2. 用组装的包跑真实任务（与 python -m mini_harness 等价）；
3. 会话落盘到临时目录并读回验证。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mini_harness.__main__ import run_task


def main() -> None:
    print("=== ① 组装清单：前 16 章各贡献了哪一块 ===")
    modules = [
        ("client.py", "第 01/02 章", "流式客户端 + 工具调用消息模型"),
        ("cordis.py / bundle.py", "第 03/04 章", "插件内核、服务依赖与 Bundle"),
        ("calculator.py", "第 02 章", "计算器工具"),
        ("session.py", "第 05 章", "事件日志与消息投影"),
        ("registry.py / prompt.py", "第 06 章", "工具注册表 + 提示词组装"),
        ("agent.py / inbox.py", "第 07 章", "常驻循环、inbox 与扩展事件"),
        ("persistence.py / checkpoint.py", "第 08 章", "JSONL 持久化 + 副作用屏障"),
        ("meter.py / pruner.py / spill.py", "第 09 章", "token 计量、剪枝与结果落盘"),
        ("sandbox.py / fs_tools.py", "第 10 章", "路径围栏与文件工具"),
        ("shell.py", "第 11 章", "Shell 策略、审批与超时"),
        ("skills.py", "第 12 章", "技能目录与渐进加载"),
        ("goal.py / plan.py / todo.py", "第 13 章", "目标、计划、问答与 Todo"),
        ("subagent.py / jobs.py / workflow.py", "第 14 章", "委派、后台任务与编排"),
        ("web_tools.py", "第 15 章", "Web Search 与 Web Fetch"),
        ("settings.py / rpc.py", "第 16 章", "分层配置与 JSON-RPC"),
        ("policies.py", "第 07–09 章", "retry、checkpoint、pruner 与 spill 插件"),
    ]
    for file, chapter, what in modules:
        print(f"  {file:<28} 来自 {chapter:<12} {what}")

    print()
    print("=== ② 用组装的包跑真实任务 ===")
    with tempfile.TemporaryDirectory() as tmp:
        session_file = str(Path(tmp) / "session.jsonl")
        final_text, completed = run_task("1+2*3 等于几？", session_file=session_file)
        print(f"  [stdout] {final_text}")
        print(f"  [exit] {'0（正常完成）' if completed else '1（异常）'}")

        print()
        print("=== ③ 会话落盘并读回 ===")
        from mini_harness.persistence import JsonlStore

        loaded = JsonlStore(session_file).load()
        print(
            f"  读回 {len(loaded.snapshot_events())} 条事件"
            "（第 08 章的持久化在工作）"
        )


if __name__ == "__main__":
    main()
