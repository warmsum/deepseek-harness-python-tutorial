"""工具注册表（第 06 章首次实现）。

第 02 章用 list[Tool] 保存工具。本章增加注册、重名检查、注销和模型侧
schema 投影；执行函数只留在本地，不进入模型请求。
"""

from __future__ import annotations

from typing import Any

from client import Tool


class ToolRegistry:
    """工具注册表：登记、查找、投影说明书。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具；重名工具会造成调用歧义，因此直接报错。"""
        if tool.name in self._tools:
            raise ValueError(f'工具 "{tool.name}" 已被注册')
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        del self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def schemas(self) -> list[dict[str, Any]]:
        """返回模型侧 schema，不包含本地 execute 函数。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.all()
        ]
