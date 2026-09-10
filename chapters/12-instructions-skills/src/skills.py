"""第 12 章：Skills 的目录与按需加载。

对应官方 packages/skill/skill。核心决策是渐进式加载：
get() 每次调用都向胜出提供方请求正文，
而不是在注册表里缓存正文。

教学版实现：
1. SkillCatalog —— 扫描目录，模型可见的只有 {name, description} 摘要；
2. load —— skill 工具被调用时才从磁盘读全文（每次重读，不缓存）；
3. render —— 渲染成 <skill_content> 块（官方 renderSkillContent 的形状）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 官方 renderSkillContent 的标签形状
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# token 估算（沿用第 09 章的启发式：4 字符 ≈ 1 token）
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class SkillSummary:
    """模型可见的技能摘要：只有名字与一句话描述。

    目录只包含技能名称和说明；完整正文由工具按需读取。"""

    name: str
    description: str


class SkillCatalog:
    """技能目录：root 下的每个子目录是一个技能，正文在 SKILL.md。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list(self) -> list[SkillSummary]:
        """扫描目录，解析每个 SKILL.md 的 frontmatter（name/description）。"""
        summaries: list[SkillSummary] = []
        for skill_dir in sorted(self.root.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            if not SKILL_NAME.fullmatch(skill_dir.name):
                raise ValueError(f"无效的技能目录名: {skill_dir.name!r}")
            frontmatter = _read_frontmatter(skill_file)
            if frontmatter["name"] != skill_dir.name:
                raise ValueError(
                    f"{skill_file} 的 name 必须与目录名 {skill_dir.name!r} 一致"
                )
            summaries.append(
                SkillSummary(
                    name=frontmatter["name"],
                    description=frontmatter["description"],
                )
            )
        return summaries

    def load(self, name: str) -> str:
        """渐进加载：每次调用都从磁盘重读正文（不缓存）。

        技能文件可能在会话期间更新；每次读取可以取得当前版本。"""
        if not SKILL_NAME.fullmatch(name):
            raise ValueError(f"无效的技能名: {name!r}")
        skill_file = self.root / name / "SKILL.md"
        if not skill_file.is_file():
            raise FileNotFoundError(f"技能 {name} 不存在")
        text = skill_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(text)
        if frontmatter["name"] != name:
            raise ValueError(f"{skill_file} 的 name 必须与目录名 {name!r} 一致")
        return _strip_frontmatter(text)

    def render(self, name: str) -> str:
        """加载并渲染成 <skill_content> 块（模型读到指令的形态）。"""
        body = self.load(name)
        base = (self.root / name).resolve()
        return "\n".join(
            [
                f'<skill_content name="{name}">',
                "<skill_resources>",
                f"Base directory for this skill: {base}",
                "Resolve relative paths mentioned by this skill against the base "
                "directory before using them. Load referenced resources only as needed.",
                "</skill_resources>",
                "",
                "<skill_instructions>",
                body,
                "</skill_instructions>",
                "</skill_content>",
            ]
        )

    def catalog_text(self) -> str:
        """生成每轮提供给模型的技能摘要目录。"""
        lines = ["可用技能："]
        for summary in self.list():
            lines.append(f"- {summary.name}: {summary.description}")
        return "\n".join(lines)


def _read_frontmatter(path: Path) -> dict[str, str]:
    """只读取文件开头的 frontmatter，不把技能正文加载进目录扫描。"""
    with path.open(encoding="utf-8") as file:
        first = file.readline()
        if first.strip() != "---":
            raise ValueError("SKILL.md 必须以 --- frontmatter 开头")
        lines = [first]
        for line in file:
            lines.append(line)
            if line.strip() == "---":
                return _parse_frontmatter("".join(lines))
    raise ValueError("SKILL.md 的 frontmatter 缺少结束 ---")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """解析 SKILL.md 开头的 --- 块（name/description 两行，教学版手写解析）。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 必须以 --- frontmatter 开头")
    fields: dict[str, str] = {}
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            if key in fields:
                raise ValueError(f"frontmatter 字段重复: {key}")
            fields[key] = value.strip()
    if not closed:
        raise ValueError("SKILL.md 的 frontmatter 缺少结束 ---")
    if "name" not in fields or "description" not in fields:
        raise ValueError("frontmatter 必须包含 name 与 description")
    if not fields["name"] or not fields["description"]:
        raise ValueError("frontmatter 的 name 与 description 不能为空")
    return fields


def _strip_frontmatter(text: str) -> str:
    """去掉 frontmatter，只返回正文。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 必须以 --- frontmatter 开头")
    end: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index + 1
            break
    if end is None:
        raise ValueError("SKILL.md 的 frontmatter 缺少结束 ---")
    return "\n".join(lines[end:]).strip()
