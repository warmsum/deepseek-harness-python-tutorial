"""系统提示词组装器（第 06 章首次实现）。

对应官方 packages/core/system-prompt：插件可以贡献有序段、
工具 schema 和具名变量，循环在每个步骤组装一次。
教学版实现其中的段贡献与变量替换两个核心机制。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptSection:
    """系统提示词里的一段。order 决定排序（数字小的在前）。"""

    order: int
    name: str
    text: str


class PromptAssembler:
    """按 order 和 name 组装多个插件贡献的系统提示词段。"""

    def __init__(self) -> None:
        self._sections: list[PromptSection] = []
        self._variables: dict[str, Callable[[], str]] = {}

    def section(self, name: str, text: str, order: int = 0) -> None:
        """贡献一段提示词。同层同名段会立即报错。"""
        if any(section.name == name for section in self._sections):
            raise ValueError(f'提示词段 "{name}" 已被注册')
        self._sections.append(PromptSection(order=order, name=name, text=text))

    def variable(self, name: str, provider: Callable[[], str]) -> None:
        if name in self._variables:
            raise ValueError(f'提示词变量 "{name}" 已被注册')
        self._variables[name] = provider

    def render(self, variables: dict[str, str] | None = None) -> str:
        """按 order 排序拼接全部段，并替换 {{变量}} 占位符。

        变量 provider 的用途：提示词里需要
        provider 在渲染时提供模型名、当前目录和日期等运行时值。
        段文本使用 {{model}} 形式引用变量。
        """
        ordered = sorted(self._sections, key=lambda s: (s.order, s.name))
        text = "\n\n".join(section.text for section in ordered)
        resolved = {name: provider() for name, provider in self._variables.items()}
        resolved.update(variables or {})
        for name, value in resolved.items():
            text = text.replace("{{" + name + "}}", value)
        unresolved = sorted(set(re.findall(r"{{([a-zA-Z_][a-zA-Z0-9_]*)}}", text)))
        if unresolved:
            raise KeyError(f"未注册的提示词变量: {', '.join(unresolved)}")
        return text

    @property
    def sections(self) -> tuple[PromptSection, ...]:
        return tuple(sorted(self._sections, key=lambda s: (s.order, s.name)))
