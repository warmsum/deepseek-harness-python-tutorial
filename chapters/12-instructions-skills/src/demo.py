"""第 12 章：模型先看技能目录，再按需加载并使用技能正文。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent import DeepSeekClient, Tool, run_agent
from skills import SkillCatalog, estimate_tokens


def main() -> None:
    skills_root = Path(__file__).resolve().parent / "skills"
    catalog = SkillCatalog(skills_root)
    menu = catalog.catalog_text()

    def load_skill(arguments: dict[str, Any]) -> str:
        name = arguments.get("name")
        if not isinstance(name, str):
            raise TypeError("name 必须是字符串")
        return catalog.render(name)

    skill_tool = Tool(
        "skill",
        "按名称加载一项技能的完整操作说明。任务匹配技能时先调用此工具。",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        load_skill,
    )
    result = run_agent(
        DeepSeekClient(),
        [skill_tool],
        (
            "你是开发助手。下面是可用技能目录，只有摘要，没有正文。"
            "任务匹配某项技能时，必须先调用 skill 加载正文，再按正文完成任务。\n\n"
            + menu
        ),
        (
            "请为以下改动编写一条规范的 Git 提交信息："
            "“重写课程练习，使每章包含开放思考题和实践题”。"
        ),
    )

    print("=== 模型最初看到的技能目录 ===")
    print(menu)
    print(f"目录估算: {estimate_tokens(menu)} token")
    print("\n=== 模型按需加载的技能 ===")
    for trace in result.traces:
        print(f"{trace.name}({trace.arguments})")
        print(f"加载正文估算: {estimate_tokens(trace.result)} token")
        print(trace.result[:240] + "…")
    print(f"\n模型依据技能生成的结果:\n{result.final_text}")


if __name__ == "__main__":
    main()
