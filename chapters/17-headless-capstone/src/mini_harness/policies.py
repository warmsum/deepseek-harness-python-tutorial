"""通过 Agent/Tools 事件接入的模型不可见策略插件。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent import ModelRequest, StepBoundary, ToolExecution
from .checkpoint import CheckpointPolicy
from .client import Message
from .cordis import Context, depends
from .meter import TokenMeter
from .pruner import ToolResultPruner
from .retry import RetryPolicy
from .session import Session
from .settings import SettingsScope
from .spill import LocalSpillStore, SpillPolicy


def meter_provider(ctx: Context, _config: Any) -> None:
    ctx.provide("meter", TokenMeter(context_window=1_000_000))


@depends("agent_settings")
def retry_plugin(ctx: Context, _config: Any) -> None:
    settings = ctx.agent_settings
    assert isinstance(settings, SettingsScope)
    policy = RetryPolicy(max_retries=int(settings.get()["retry_max_retries"]))
    ctx.provide("retry", policy)

    def request_with_retry(
        request: ModelRequest, next_fn: Callable[..., Any]
    ) -> Message:
        while True:
            try:
                result = next_fn(request)
                assert isinstance(result, Message)
                return result
            except Exception as error:
                recovered = policy.recover(
                    request.session,
                    turn=request.turn,
                    step=request.step,
                    error=error,
                    before_wait=lambda session: ctx.emit("checkpoint/retry", session),
                )
                if not recovered:
                    raise

    ctx.on("llm/request", request_with_retry)


def checkpoint_plugin(ctx: Context, flush: Callable[[Session], None] | None) -> None:
    policy = CheckpointPolicy(flush or (lambda _session: None))
    ctx.provide("checkpoint", policy)

    def before_step(boundary: StepBoundary) -> None:
        policy.before_step(boundary.session)

    def before_model(request: ModelRequest, next_fn: Callable[..., Any]) -> Any:
        policy.before_model(request.session)
        return next_fn(request)

    def before_tool(execution: ToolExecution, next_fn: Callable[..., Any]) -> Any:
        policy.before_tool(execution.session)
        return next_fn(execution)

    ctx.on("agent/pre-step", before_step)
    ctx.on("llm/request", before_model)
    ctx.on("tools/execute", before_tool)
    ctx.on("checkpoint/retry", policy.before_retry)


@depends("meter", "agent_settings")
def pruner_plugin(ctx: Context, _config: Any) -> None:
    meter = ctx.meter
    settings = ctx.agent_settings
    assert isinstance(meter, TokenMeter)
    assert isinstance(settings, SettingsScope)
    config = settings.get()
    pruner = ToolResultPruner(
        threshold_chars=int(config["prune_threshold_chars"]),
        head_chars=int(config["prune_head_chars"]),
        tail_chars=int(config["prune_tail_chars"]),
    )
    ctx.provide("pruner", pruner)

    def prepare(request: ModelRequest) -> None:
        pressure = meter.pressure(meter.measure(request.messages, request.schemas))
        if not pressure.over_threshold:
            return
        result = pruner.prune_session(request.session)
        if result.replacements:
            request.messages = request.session.derive_messages()

    ctx.on("agent/prepare-request", prepare)


@depends("agent_settings")
def spill_plugin(ctx: Context, root: str) -> None:
    settings = ctx.agent_settings
    assert isinstance(settings, SettingsScope)
    policy = SpillPolicy(
        int(settings.get()["spill_max_inline_bytes"]),
        LocalSpillStore(root),
    )
    ctx.provide("spill", policy)

    def transform(execution: ToolExecution) -> None:
        execution.result = policy.transform(
            execution.result,
            session_id=execution.agent.session_id,
            tool_name=execution.call.name,
            call_id=execution.call.id,
        )

    ctx.on("tools/post-execute", transform)
