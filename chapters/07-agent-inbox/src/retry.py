"""第 07 章：提供方级 LLM retry 策略。

官方把策略放在模型提供方配置上，把执行接入 Agent 的请求错误边界。
这里保留有限预算、错误码、Retry-After、有界指数退避、jitter 和两条持久
事件；时间函数可注入，让测试不用真的等待。
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import httpx
from session import Session

DEFAULT_RETRYABLE_CODES = (
    "EMPTY_RESPONSE",
    "RATE_LIMIT",
    "SERVER",
    "TIMEOUT",
    "TRANSPORT",
)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    retryable_codes: tuple[str, ...] = DEFAULT_RETRYABLE_CODES
    initial_delay_ms: float = 500
    max_delay_ms: float = 10_000
    jitter_ratio: float = 0.1
    provider: str = "deepseek-official"
    sleeper: Callable[[float], None] = time.sleep
    random_sample: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries 必须是非负整数")
        if (
            not self.retryable_codes
            or any(not isinstance(code, str) or not code for code in self.retryable_codes)
            or len(set(self.retryable_codes)) != len(self.retryable_codes)
        ):
            raise ValueError("retryable_codes 必须由非空且不重复的字符串组成")
        if (
            not math.isfinite(self.initial_delay_ms)
            or not math.isfinite(self.max_delay_ms)
            or self.initial_delay_ms <= 0
            or self.initial_delay_ms > self.max_delay_ms
        ):
            raise ValueError("initial_delay_ms 必须为正且不大于 max_delay_ms")
        if not math.isfinite(self.jitter_ratio) or not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio 必须在 0 到 1 之间")

    def recover(
        self,
        session: Session,
        *,
        turn: int,
        step: int,
        error: Exception,
        before_wait: Callable[[Session], None] | None = None,
    ) -> bool:
        """若错误符合策略，先记 scheduled，再等待并记 started。"""
        failure = classify_failure(error)
        policy_key = json.dumps(
            [
                "normal",
                self.max_retries,
                sorted(self.retryable_codes),
                self.initial_delay_ms,
                self.max_delay_ms,
                self.jitter_ratio,
            ],
            separators=(",", ":"),
        )
        prior = next(
            (
                event
                for event in reversed(session.snapshot_events())
                if event.type == "llm/retry"
                and event.data.get("turn") == turn
                and event.data.get("step") == step
                and event.data.get("provider") == self.provider
                and event.data.get("policy_key") == policy_key
            ),
            None,
        )
        previous_retry = prior.data.get("retry") if prior is not None else 0
        retry = previous_retry + 1 if isinstance(previous_retry, int) else 1
        if failure["code"] not in self.retryable_codes or retry > self.max_retries:
            return False
        provider_delay = failure.get("provider_retry_after_ms")
        if isinstance(provider_delay, float) and provider_delay > self.max_delay_ms:
            return False
        delay_ms = provider_delay if isinstance(provider_delay, float) else self._local_delay(retry)
        prior_id = prior.data.get("retry_id") if prior is not None else None
        retry_id = prior_id if isinstance(prior_id, str) else uuid4().hex
        session.append(
            "llm/retry",
            {
                "retry_id": retry_id,
                "turn": turn,
                "step": step,
                "provider": self.provider,
                "mode": "normal",
                "policy_key": policy_key,
                "retry": retry,
                "max_retries": self.max_retries,
                "delay_ms": delay_ms,
                "failure": failure,
            },
        )
        if before_wait is not None:
            before_wait(session)
        self.sleeper(delay_ms / 1000)
        session.append(
            "llm/retry-started",
            {"retry_id": retry_id, "turn": turn, "step": step, "retry": retry},
        )
        return True

    def _local_delay(self, retry: int) -> float:
        growth_steps = retry - 1
        saturation = math.ceil(
            math.log2(self.max_delay_ms / self.initial_delay_ms)
        )
        exponential = (
            self.max_delay_ms
            if growth_steps >= saturation
            else self.initial_delay_ms * 2**growth_steps
        )
        jitter = 1 - self.jitter_ratio + 2 * self.jitter_ratio * self.random_sample()
        return float(min(exponential * jitter, self.max_delay_ms))


def classify_failure(error: Exception) -> dict[str, object]:
    """把 httpx 异常投影为官方 retry 使用的稳定错误码。"""
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        failure: dict[str, object] = {"code": code, "message": str(error)}
        retry_after = getattr(error, "provider_retry_after_ms", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            failure["provider_retry_after_ms"] = float(retry_after)
        return failure
    if isinstance(error, httpx.TimeoutException):
        return {"code": "TIMEOUT", "message": str(error)}
    if isinstance(error, httpx.TransportError):
        return {"code": "TRANSPORT", "message": str(error)}
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        code = "RATE_LIMIT" if status == 429 else "SERVER" if status >= 500 else f"HTTP_{status}"
        failure = {"code": code, "message": str(error), "status": status}
        retry_after = _retry_after_ms(error.response.headers.get("Retry-After"))
        if retry_after is not None:
            failure["provider_retry_after_ms"] = retry_after
        return failure
    return {"code": "UNKNOWN", "message": str(error)}


def _retry_after_ms(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds * 1000 if seconds > 0 else None
