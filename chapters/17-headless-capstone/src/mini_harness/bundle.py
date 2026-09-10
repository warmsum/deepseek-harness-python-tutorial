"""Python 命令行 Bundle：声明插件清单，不承载业务逻辑。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from .agent import Agent, AgentRegistry
from .calculator import calculator
from .capabilities import (
    delegation_consumer,
    delegation_provider,
    filesystem_consumer,
    filesystem_provider,
    goal_todo_consumer,
    goal_todo_provider,
    interaction_consumer,
    jobs_provider,
    plan_provider,
    questions_provider,
    rpc_provider,
    settings_provider,
    shell_consumer,
    shell_provider,
    skills_consumer,
    skills_provider,
    subagents_provider,
    web_consumer,
    web_provider,
    workflow_provider,
)
from .client import ChatClient, DeepSeekClient
from .cordis import Context, depends
from .policies import (
    checkpoint_plugin,
    meter_provider,
    pruner_plugin,
    retry_plugin,
    spill_plugin,
)
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .session import Session


@dataclass(frozen=True)
class BundleConfig:
    settings_document: Mapping[str, Any]
    enable_console_questions: bool = True
    checkpoint_flush: Callable[[Session], None] | None = None
    spill_root: str = ".mini-harness/spills"
    client: ChatClient | None = None
    session: Session | None = None


def session_provider(ctx: Context, session: Session | None) -> None:
    ctx.provide("session", session or Session())


def prompt_provider(ctx: Context, _config: Any) -> None:
    prompt = PromptAssembler()
    ctx.provide("prompt", prompt)
    ctx.effect(
        lambda: prompt.section(
            "persona",
            "你是 {{name}}，一个本地编程助手。先调查再修改，使用工具验证结果，"
            "长任务用 Goal/Todo，独立任务可委派给 Subagent。",
            order=0,
        )
    )
    ctx.effect(
        lambda: prompt.section(
            "rules",
            "回答要直接、客观，并先给结论。说明实际结果和使用方法；只有限制会影响"
            "结果判断时才说明，并具体指出未完成或未验证的内容。不要堆叠免责声明、"
            "重复边界，也不要通过贬低其他方案来证明当前选择。",
            order=100,
        )
    )


def tools_provider(ctx: Context, _config: Any) -> None:
    ctx.provide("tools", ToolRegistry())


@depends("tools")
def calculator_plugin(ctx: Context, _config: Any) -> None:
    tools = ctx.tools
    assert isinstance(tools, ToolRegistry)
    ctx.effect(lambda: tools.register(calculator))


def llm_provider(ctx: Context, client: ChatClient | None) -> None:
    ctx.provide("llm", client if client is not None else DeepSeekClient())


@depends("llm", "tools", "prompt", "session")
def agent_provider(ctx: Context, _config: Any) -> None:
    client = cast(ChatClient, ctx.llm)
    tools = ctx.tools
    prompt = ctx.prompt
    session = ctx.session
    assert isinstance(tools, ToolRegistry)
    assert isinstance(prompt, PromptAssembler)
    assert isinstance(session, Session)
    agents = AgentRegistry()
    agent = Agent(
        ctx,
        client,
        tools,
        prompt,
        variables={"name": "mini-harness"},
        session=session,
        agent_id="main",
    )
    ctx.provide("agents", agents)
    ctx.effect(lambda: agents.register(agent))
    # 使用者是 Agent provider 的子插件，会随这个精确实例一起卸载和重建。
    ctx.plugin(interaction_consumer, agent)
    ctx.plugin(delegation_consumer, agent)
    ctx.plugin(rpc_provider, agent)


def headless_bundle(ctx: Context, config: BundleConfig) -> None:
    """挂载插件树；列表顺序用于确定提供者和包装器的优先级。"""
    ctx.plugin(settings_provider, config.settings_document)
    ctx.plugin(session_provider, config.session)
    ctx.plugin(prompt_provider)
    ctx.plugin(tools_provider)
    ctx.plugin(llm_provider, config.client)
    ctx.plugin(meter_provider)
    ctx.plugin(calculator_plugin)

    # waterfall 外层先安装：retry 每次 next() 都会重新经过 checkpoint。
    # checkpoint 先于 Plan 插件监听 pre-step，保持“先落盘、再提交边界选择”。
    ctx.plugin(retry_plugin)
    ctx.plugin(checkpoint_plugin, config.checkpoint_flush)
    ctx.plugin(pruner_plugin)
    ctx.plugin(spill_plugin, config.spill_root)

    ctx.plugin(filesystem_provider)
    ctx.plugin(filesystem_consumer)
    ctx.plugin(shell_provider)
    ctx.plugin(shell_consumer)
    ctx.plugin(skills_provider)
    ctx.plugin(skills_consumer)
    ctx.plugin(goal_todo_provider)
    ctx.plugin(goal_todo_consumer)
    ctx.plugin(questions_provider, config.enable_console_questions)
    ctx.plugin(plan_provider)
    ctx.plugin(web_provider)
    ctx.plugin(web_consumer)
    ctx.plugin(subagents_provider)
    ctx.plugin(jobs_provider)
    ctx.plugin(workflow_provider)
    ctx.plugin(delegation_provider)

    ctx.plugin(agent_provider)
