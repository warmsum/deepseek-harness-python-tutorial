"""第 04 章 demo：服务依赖与顺序处理链的四个场景。

运行（无需 API，纯本地）：
    uv run python chapters/04-services-scopes/src/demo.py

对照 README 观察：
1. 服务后到时自动启动等待中的 agent
2. 提供者被卸载 → 依赖方自动卸载
3. 读服务必须 inject（严格访问报错）
4. waterfall：不修改核心执行器，给工具增加执行日志
"""

from __future__ import annotations

from collections.abc import Callable

from context import Context


def llm_provider(ctx: Context, _config: object) -> None:
    ctx.provide(
        "llm", {"provider": "deepseek-official", "model": "deepseek-v4-flash"}
    )
    print("  [llm-provider] 已提供 llm 服务")


def agent(ctx: Context, _config: object) -> None:
    # 读服务走 __getattr__：声明过 inject 才能读
    print(f"  [agent] 启动！llm={ctx.llm} tools={ctx.tools}")


setattr(agent, "inject", ["llm", "tools"])  # 官方 cordis 的 Object.assign 模式


def tools_provider(ctx: Context, _config: object) -> None:
    ctx.provide("tools", {"calculator": "recursive-descent"})
    print("  [tools-provider] 已提供 tools 服务")


def tools_provider_v2(ctx: Context, _config: object) -> None:
    ctx.provide("tools", {"calculator": "v2"})
    print("  [tools-provider-2] 已提供 tools v2")


def main() -> None:
    ctx = Context()
    print("=== 场景 1：服务后到，插件自动启动 ===")

    ctx.plugin(llm_provider)

    agent_handle = ctx.plugin(agent)
    print(f"  [agent] 当前状态: {agent_handle.state}   ← 等待缺失依赖")

    tools_handle = ctx.plugin(tools_provider)
    print(f"  [agent] 当前状态: {agent_handle.state}      ← 依赖齐了，自动启动！")

    print()
    print("=== 场景 2：提供者被卸载，依赖方自动卸载 ===")

    try:
        ctx.plugin(tools_provider_v2)
    except ValueError as error:
        print(f"  重名服务被拒绝: {error}")

    tools_handle.dispose()
    print(f"  卸载 tools v1 后 [agent] 状态: {agent_handle.state}")
    tools_handle = ctx.plugin(tools_provider_v2)
    print(f"  注册 tools v2 后 [agent] 状态: {agent_handle.state}")

    tools_handle.dispose()
    print(f"  卸载 tools v2 后 [agent] 状态: {agent_handle.state}   ← 级联卸载")

    print()
    print("=== 场景 3：读服务必须 inject ===")
    try:
        print(ctx.llm)  # llm 服务仍存在，但当前调用方没有声明依赖
    except AttributeError as error:
        print(f"  报错: {error}")
        print("  ← 未声明的服务访问被运行时拒绝")

    print()
    print("=== 场景 4：waterfall 顺序处理链 ===")

    def logging_policy(c: Context, _config: object) -> None:
        def wrap(exec_: dict[str, str], next_: Callable[[], str]) -> str:
            print(f"  [logging-policy] 开始执行工具 {exec_['name']}")
            result = next_()  # 放行进入内层，返回值沿链回传
            print(f"  [logging-policy] 工具 {exec_['name']} 完成")
            return result

        c.on("tools/execute", wrap)

    ctx.plugin(logging_policy)

    def core_executor(exec_: dict[str, str]) -> str:
        print(f"  [core] 真正执行 {exec_['name']}……")
        return "计算结果: 42"

    result = ctx.waterfall("tools/execute", {"name": "calculator"}, core_executor)
    print(f"  最终结果: {result}")


if __name__ == "__main__":
    main()
