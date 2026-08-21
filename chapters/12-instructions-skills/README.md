# 12｜技能与按需加载

> 预计时间：50 分钟 ｜ 前置：完成第 11 章 ｜ 本章调用真实 DeepSeek 模型

第 10、11 章为智能体增加了文件和命令工具。随着能力继续增加，系统中还会出现许多面向具体任务的操作手册，例如怎样搜索网络、编写 Git 提交信息，或者运行某个项目的测试。这些内容只在特定任务中使用。如果全部放进系统提示词，每轮请求都要携带所有手册，即使当前任务与它们无关。

Skills（技能）把操作手册分成“简短介绍”和“完整正文”两层。模型平时只看到包含名称和一句话说明的技能菜单；确定要使用某项技能后，再通过 `skill` 工具读取完整手册。这种先展示摘要、用到时再读取正文的方法称为渐进式加载。

## 学习目标

完成本章后，你将能够：

- 解释技能摘要常驻、正文按需加载如何节省 token；
- 按 `skills/<name>/SKILL.md` 约定扫描技能目录；
- 解析文件开头的 frontmatter 元数据，并把技能正文包装成 `<skill_content>` 块；
- 说明每次重新读取正文而不缓存的行为意义；
- 让模型根据任务选择技能，调用 `skill` 加载正文后再完成任务。

## 12.1 原理：菜单常驻，正文按需

把技能拆成两层：

| 层 | 内容 | 什么时候给模型看 |
|----|------|------------------|
| 摘要 | 名称 `name` 和一句话说明 `description` | 每轮请求都提供 |
| 正文 | `SKILL.md` 中的完整操作手册 | 调用 `skill` 工具时读取 |

示例中的两个技能摘要约占 30 token，而完整正文需要数百 token。技能越多、正文越长，差距越明显。假设有 50 个技能，每份正文占 500 token，把正文全部放进提示词会让每轮请求增加约 25000 token；按需加载时，只有真正使用的那一份会进入上下文。

按需加载还可以及时反映文件变化。技能文件可能被用户编辑，`SkillCatalog.load()` 每次调用都重新读取磁盘，因此修改后的正文会在下一次加载时生效；如果长期缓存正文，智能体可能继续使用旧指令。

第三个好处与第 06 章相呼应：每个技能在自己的 SKILL.md 中维护正文，系统提示词组装器只需要提供技能目录，不必内置每份手册的内容。

## 12.2 扫描、解析和加载技能：SkillCatalog

每项技能存放在 `skills/<name>/SKILL.md` 中。文件开头是由两行 `---` 包围的 frontmatter 元数据，后面是完整操作说明：

```markdown
---
name: web-search-guide
description: 如何使用 Web Search 工具高效搜索网络、挑选来源、引用结果
---

# Web Search 使用指南

当用户的问题需要最新信息时，调用 web_search 工具。
...
```

目录实现：

```python
@dataclass(frozen=True)
class SkillSummary:
    name: str
    description: str


class SkillCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list(self) -> list[SkillSummary]:
        """扫描目录，解析每个 SKILL.md 的 frontmatter。"""
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
        """渐进加载：每次调用都从磁盘重读正文，不缓存。"""
        if not SKILL_NAME.fullmatch(name):
            raise ValueError(f"无效的技能名: {name!r}")
        skill_file = self.root / name / "SKILL.md"
        if not skill_file.is_file():
            raise FileNotFoundError(f"技能 {name} 不存在")
        text = skill_file.read_text(encoding="utf-8")
        return _strip_frontmatter(text)
```

四个要点：

- `list()` 通过 `_read_frontmatter()` 读到第二个 `---` 就停止，因此扫描菜单时不会加载正文。缺少结束分隔符时立即报错。
- `load()` 不缓存正文，每次调用都重新读取 `SKILL.md`。这是文件修改能够及时生效的关键。
- 教学版的解析器只读取 `name` 和 `description` 两个字段，足以生成技能菜单。
- 技能名只接受由小写字母、数字和连字符组成的格式，而且 frontmatter 中的 `name` 必须与目录名一致。这样既能阻止 `../other-file` 一类路径穿越，也能避免目录和菜单使用不同名称。

## 12.3 把技能正文包装后交给模型

技能正文不能只把一段文本直接交给模型，还需要说明技能名称和相对路径从哪里解析。`render()` 会把这些信息包装成统一的 `<skill_content>` 结构：

```python
    def render(self, name: str) -> str:
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
```

包装分成两块：`skill_resources` 告诉模型相对路径从哪个目录解析，并提醒它只在需要时加载其他资源；`skill_instructions` 保存操作手册正文。外层标签还记录技能名称，使模型能够看出这段内容的来源和范围。

## 12.4 让模型自己选择并加载技能

扫描目录后，程序只把 `catalog.catalog_text()` 生成的技能菜单放进系统提示词，同时注册一个很小的 `skill` 工具：

```python
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
```

用户要求编写 Git 提交信息时，模型最初只能看到 `git-commit-guide` 和 `web-search-guide` 的名称与说明。它需要先判断任务与哪项技能匹配，再调用 `skill({"name": "git-commit-guide"})`。工具从磁盘读取并包装正文，结果进入下一次模型请求后，模型才能按照其中的类型、范围、主题和正文规则生成提交信息。

这里没有在 Python 代码中预先指定要加载哪项技能。选择由模型完成，目录负责提供可选项，工具负责校验名称和读取内容，技能正文负责约束最终任务。三层职责分开后，新增技能不需要修改模型循环。

## 12.5 运行完整示例

```bash
uv run python chapters/12-instructions-skills/src/demo.py
```

下面是一次真实运行的主要输出。模型生成的提交信息可能变化，技能目录和加载内容来自当前文件：

```
=== 模型最初看到的技能目录 ===
可用技能：
- git-commit-guide: 如何编写规范的 git commit message（类型、范围、主题、正文）
- web-search-guide: 如何使用 Web Search 工具高效搜索网络、挑选来源、引用结果
目录估算: 30 token

=== 模型按需加载的技能 ===
skill({'name': 'git-commit-guide'})
加载正文估算: 182 token
<skill_content name="git-commit-guide">
<skill_resources>
Base directory for this skill: …/skills/git-commit-guide
Resolve relati…

模型依据技能生成的结果:
refactor(course): 重写练习使每章含开放思考题与实践题
```

调用记录说明模型选择了与任务匹配的 `git-commit-guide`，没有加载无关的网络搜索技能。目录约 30 token，完整正文只在工具调用后进入会话。最终结果遵循技能中的提交信息结构；如果修改 `SKILL.md` 后再次运行，工具会读取新正文而不是旧缓存。

## 本章小结

- 技能存储方式：`skills/<name>/SKILL.md` 和文件开头的 frontmatter 元数据
- `SkillCatalog`：扫描摘要时不读取正文，加载技能时重新读取文件
- `render`：用 `<skill_content>` 区分资源路径说明和操作手册正文
- 名称校验：技能名使用固定格式，并与目录名保持一致
- 真实模型流程：模型先看摘要目录，再调用 `skill` 读取匹配任务的完整正文
- 两个示例技能：web-search-guide、git-commit-guide

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/skill/skill/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/skill/skill/README.zh.md) | `SkillCatalog` | 官方同样提供摘要目录、统一内容渲染和每次重新取正文的渐进式加载 |
| 同上 | （未实现） | 官方还支持多种技能来源、宿主和作用域分层、按优先级解决重名、缓存失效和调用记录；教学版只读取一个磁盘目录 |
| [`packages/skill/tool-skill/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/skill/tool-skill/README.zh.md) | `skill` 工具 | 官方把技能目录和 `skill` 工具放在同一个插件中，由模型调用工具读取正文；教学版在本章的最小模型循环中直接注册该工具 |

## 练习

1. 角色规则、工具说明和技能都能向模型提供指导。请选择三个具体例子，判断它们应放在哪一层，并说明把所有内容都写进系统提示词会带来什么问题。
2. 为本项目设计一项新的技能，例如“运行本地质量检查”或“编写课程章节”。写出能够帮助模型正确选择它的名称与说明，并规划正文和参考资源如何按需展开。
3. 技能来自项目、用户或第三方目录时，可能出现同名冲突、内容更新和恶意指令。设计一套发现、优先级和信任规则，说明哪些检查应在扫描菜单时完成，哪些留到加载正文时完成。
4. 扩展 `SkillCatalog`，使它能够从项目级和用户级两个来源加载技能，并按明确规则解决冲突。把 `catalog.render` 接入一个 `skill` 工具，验证菜单保持精简、正文只在调用后读取且文件更新能够被下一次调用看到。
