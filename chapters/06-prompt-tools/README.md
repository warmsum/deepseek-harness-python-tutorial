# 06｜组装模型请求

> 预计时间：55 分钟 ｜ 前置：完成第 05 章 ｜ 本章调用真实 DeepSeek 模型

第 05 章结束时，每次发给模型的请求长这样：

```python
session.record_system_prompt(system_prompt, turn=1, step=step)
messages = session.derive_messages()
tools = [calculator]  # 固定工具列表
```

系统提示词目前是一段手写字符串，工具清单则是固定列表。当智能体继续加入角色设定、安全规则和运行时信息后，提示词会来自不同模块，可用工具也会发生变化。本章把二者改造成可以组合的结构：不同模块分别提供提示词片段，组装器负责排序；工具注册表负责管理工具，并生成模型需要的说明。

一次完整的模型请求包括系统提示词、消息历史和工具说明。它们都可能随着运行状态发生变化，因此程序会在每个步骤开始前重新组装。官方文档把这组内容称为 request envelope，本章正文统一使用“完整请求”。

## 学习目标

完成本章后，你将能够：

- 把系统提示词拆成可独立贡献和排序的多个段；
- 在渲染提示词时替换运行时变量，并保证结果稳定；
- 使用 `ToolRegistry` 管理工具并拒绝重名注册；
- 区分模型看到的工具参数说明与程序实际执行的工具函数。

## 6.1 系统提示词为什么要组装

先看一个完整智能体的系统提示词可能由哪些部分组成：

| 段 | 内容举例 | 贡献者 |
|----|----------|--------|
| 人设 | 你是一个数学助手，遇到算式先调用工具 | 人设插件 |
| 规则 | 回答先给结论，再给过程 | 沙箱插件 |
| 工具目录 | 可用工具：calculator…… | 工具注册表 |
| 运行时信息 | 当前模型：deepseek-v4-flash | 框架自身 |

四个来源各自只负责自己的内容。如果全部由主流程拼接，主流程会再次承担过多职责；如果让插件共同改写一个字符串，最终顺序会依赖加载时机，难以复现和调试。

组装器把“提供内容”和“拼接内容”分开。每个插件调用 `section(name, text, order)` 提供一段文字，组装时先按 `order`、再按名称排序。这样，同一组段无论按什么顺序安装，都能得到相同提示词。

提示词还可以包含 `{{model}}` 这样的变量。组装时，程序会用当前模型名、工作目录等运行时信息替换它们。这比各个插件自行拼接字符串更容易检查，也能在变量缺失时及时报错。

## 6.2 PromptAssembler：段的贡献与拼接

```python
@dataclass(frozen=True)
class PromptSection:
    order: int
    name: str
    text: str


class PromptAssembler:
    def __init__(self) -> None:
        self._sections: list[PromptSection] = []
        self._variables: dict[str, Callable[[], str]] = {}

    def section(self, name: str, text: str, order: int = 0) -> None:
        if any(section.name == name for section in self._sections):
            raise ValueError(f'提示词段 "{name}" 已被注册')
        self._sections.append(PromptSection(order=order, name=name, text=text))

    def variable(self, name: str, provider: Callable[[], str]) -> None:
        self._variables[name] = provider

    def render(self, variables: dict[str, str] | None = None) -> str:
        ordered = sorted(self._sections, key=lambda s: (s.order, s.name))
        text = "\n\n".join(section.text for section in ordered)
        resolved = {name: provider() for name, provider in self._variables.items()}
        resolved.update(variables or {})
        for name, value in resolved.items():
            text = text.replace("{{" + name + "}}", value)
        unresolved = re.findall(r"{{([a-zA-Z_][a-zA-Z0-9_]*)}}", text)
        if unresolved:
            raise KeyError(f"未注册的提示词变量: {', '.join(unresolved)}")
        return text
```

三个机制：

1. 先按 `order` 排序，同一顺序值再按名称排序，避免插件加载顺序改变请求前缀。
2. 同一注册表中的提示词段不能重名，重复注册会立即报错，避免后注册的内容覆盖已有段。
3. 变量由 `variable(name, provider)` 注册，其中 `provider` 是一个返回当前值的函数。每次调用 `render()` 都会重新取值；模板引用了未知变量时直接报错，避免把没有替换的 `{{typo}}` 发给模型。

## 6.3 ToolRegistry：从列表到注册表

第 02 章用 `list[Tool]` 保存工具。每个 `Tool` 同时包含两部分：模型需要看到的名称、用途和参数说明，以及 Python 程序真正调用的 `execute` 函数。工具注册表会把这两部分分开，避免请求组装代码接触不需要发送给模型的执行函数：

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f'工具 "{tool.name}" 已被注册')
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def schemas(self) -> list[dict[str, Any]]:
        """投影出给模型看的说明书清单：不含 execute。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.all()
        ]
```

注册时会检查重名。`all()` 默认按工具名排序，因此相同的工具集合总能生成相同顺序的请求。`schemas()` 只返回模型需要的工具说明，不包含本地执行函数。

## 6.4 把组装器和注册表接进循环

第 05 章的 `run_agent` 只改两处：

```python
def run_agent(client, registry, assembler, user_prompt,
              max_steps=10, variables=None) -> Session:
    session = Session()
    # ...turn/start 与第 05 章相同
    request_header = None

    for step in range(1, max_steps + 1):
        system_prompt = assembler.render(variables)
        session.record_system_prompt(system_prompt, turn=1, step=step)
        # 模型配置、工具 schema 或消息序列变化时追加 request/header
        # ...计算 header_fingerprint 并与 request_header 比较
        reply = client.chat(session.derive_messages(), registry.all())
        # ...其余与第 05 章相同
```

至此，完整请求中的系统提示词由组装器生成，工具说明由注册表提供。每个步骤都会重新生成提示词，因此运行时变量变化后，下一次模型调用就能看到新值。渲染结果写入 `system/message`；提示词变化时，新事件通过表层替换取代旧节点。`request/header` 只保存模型配置与工具 schema，系统提示词不再重复保存其中。提示词替换会开始新的消息序列，并记录 `reason="series"`。

## 6.5 运行完整示例

```bash
uv run python chapters/06-prompt-tools/src/demo.py
```

真实输出，模型回复内容每次不同，组装结果稳定：

```
=== ① 组装出的系统提示词 ===
你是 小算，一个数学助手。遇到算式时先调用 calculator 工具计算，再基于计算结果回答。

回答要简洁：先给结论，再给过程。

=== ② 注册表投影出的工具说明书（模型看到的清单） ===
[
  {
    "name": "calculator",
    "description": "计算一个四则运算表达式，支持 + - * / 与括号，例如 '1+2*3'",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "要计算的数学表达式，例如 '1+2*(3-1)'"
        }
      },
      "required": ["expression"]
    }
  }
]
  ← 只有 name/description/parameters，没有 execute

=== ③ 运行真实请求 ===
  [assistant]
  [assistant] 1+2*3 = **7**

先算乘法 2×3=6，再加 1，得 7。
```

三部分输出分别验证了三个机制：① 人设段（`order=0`）与规则段（`order=100`）按序拼接，`{{name}}` 被替换成“小算”；② 注册表只输出模型需要的说明字段，不包含 `execute`；③ 组装后的请求能够完成一次真实对话。第一行空的模型消息对应请求工具的步骤，它没有正文，只包含 `tool_calls`。

## 本章小结

- `PromptSection` 与 `PromptAssembler`：按 `order` 和名称稳定排序，拒绝重名，严格替换变量
- `ToolRegistry`：检查工具重名，并把模型说明与本地执行函数分开
- 运行循环：每一步重新取得系统提示词和工具清单，把提示词写入消息历史，并按请求序列记录请求头

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/core/system-prompt/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/core/system-prompt/README.zh.md) | `PromptAssembler` | 与官方一样按顺序值和名称排列片段，拒绝同一层重名，并在每次生成提示词时重新取得变量值 |
| 同上 | 作用域与重名 | 不同作用域中，较近的提示词片段可以遮蔽较远的同名片段；同一作用域内重名会报错 |
| 同上 | 稳定排序 | `order` 相同的片段按名称排序；工具也默认按名称排序，官方还允许通过 `toolOrder` 指定顺序 |
| [`packages/core/tools/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/core/tools/README.zh.md) | `ToolRegistry` | 与官方一样由注册表生成模型需要的参数说明；执行函数只保留在本地，不进入模型请求 |

官方把工具参数说明也视为提示词组装结果的一部分。注册表通过 `ctx.systemPrompt.tools()` 把工具说明交给组装器，适配器再将它作为协议中的独立字段发送。教学版分别生成系统提示词和 `tools` 字段，再一起交给模型客户端，数据流更容易观察。

## 练习

1. 安全、人设和项目插件共同提供提示词与工具时，顺序不稳定、名称冲突和变量缺失分别会造成什么问题？请设计一套注册与报错规则，使同一组插件无论按什么顺序安装都能生成相同请求，并在请求发出前暴露冲突。
2. 工具参数说明可以放在协议的 `tools` 字段，也可以写成普通提示词文本。比较两种方式在参数约束、模型选择和 token 成本上的差异，并说明为什么现代智能体通常优先使用前者。
3. 为本章组装器增加一段包含运行时变量的提示词，并注册一个与它配合使用的工具。完成一次真实或模拟请求，展示最终系统提示词、工具参数说明和执行函数分别来自哪里。
