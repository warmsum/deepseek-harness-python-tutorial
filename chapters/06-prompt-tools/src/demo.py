"""第 06 章 demo：组装完整请求，然后运行真实对话。

运行（在项目根目录，需要 .env）：
    uv run python chapters/06-prompt-tools/src/demo.py

输出三节：
① 组装出的系统提示词（两段拼接 + 变量替换）
② 注册表投影出的工具说明书（不含 execute）
③ 运行真实请求：模型用 calculator 回答问题
"""

from __future__ import annotations

import json

from agent import run_agent
from calculator import calculator
from client import DeepSeekClient
from prompt import PromptAssembler
from registry import ToolRegistry


def main() -> None:
    # ① 组装系统提示词：两个来源各贡献一段，{{name}} 留待渲染时替换
    assembler = PromptAssembler()
    assembler.section(
        "persona",
        "你是 {{name}}，一个数学助手。遇到算式时先调用 calculator 工具计算，"
        "再基于计算结果回答。",
        order=0,
    )
    assembler.section(
        "rules",
        "回答要简洁：先给结论，再给过程。",
        order=100,
    )
    # ② 注册表：登记工具，投影说明书
    registry = ToolRegistry()
    registry.register(calculator)

    print("=== ① 组装出的系统提示词 ===")
    print(assembler.render(variables={"name": "小算"}))

    print()
    print("=== ② 注册表投影出的工具说明书（模型看到的清单） ===")
    print(json.dumps(registry.schemas(), ensure_ascii=False, indent=2))
    print("  ← 注意：只有 name/description/parameters，没有 execute")

    print()
    print("=== ③ 运行真实请求 ===")
    client = DeepSeekClient()
    session = run_agent(
        client,
        registry=registry,
        assembler=assembler,
        user_prompt="1+2*3 等于几？",
        max_steps=10,
        variables={"name": "小算"},
    )
    for message in session.derive_messages():
        if message.role == "assistant":
            print(f"  [assistant] {message.content}")


if __name__ == "__main__":
    main()
