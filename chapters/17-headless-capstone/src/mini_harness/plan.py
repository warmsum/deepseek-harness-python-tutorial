"""第 13 章：Plan Mode——日志状态、边界提交与用户评审。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .client import Tool
from .session import Session, SessionEvent
from .user_questions import (
    Question,
    QuestionIntent,
    QuestionOption,
    QuestionRequest,
    UserQuestionError,
    UserQuestionService,
)

EXIT_PLAN_MODE = "exit_plan_mode"
REVIEW_ID = "plan-review"
APPROVE_LABEL = "Approve"
KEEP_PLANNING_LABEL = "Keep planning"


@dataclass(frozen=True)
class PlanModeState:
    active: bool
    pending: bool | None = None


def fold_plan_mode(events: tuple[SessionEvent, ...]) -> bool:
    """最后一条 ``plan/mode`` 获胜；没有事件时默认关闭。"""
    active = False
    for event in events:
        if event.type == "plan/mode":
            value = event.data.get("active")
            if not isinstance(value, bool):
                raise ValueError(f"plan/mode active 必须是布尔值（seq={event.seq}）")
            active = value
    return active


class PlanModeController:
    """维护选择与日志提交，不把 Plan Mode 误当成权限沙箱。"""

    def __init__(
        self,
        session: Session,
        section: str,
        questions: UserQuestionService | None = None,
        *,
        agent: object | None = None,
    ) -> None:
        if not isinstance(section, str) or not section.strip():
            raise ValueError("PlanModeConfig needs a non-empty `section`")
        self._session = session
        self._section = section
        self._questions = questions
        self._agent = agent
        self._pending: bool | None = None
        self._narrate = False

    def bind_agent(self, agent: object | None) -> None:
        """组装完成后绑定精确的运行时根 Agent。"""
        self._agent = agent

    def get(self) -> PlanModeState:
        return PlanModeState(
            fold_plan_mode(self._session.snapshot_events()), self._pending
        )

    def set(self, active: bool) -> str:
        """空闲时立即提交；turn 内只排队到下一次 step 边界。"""
        if not isinstance(active, bool):
            raise TypeError("active 必须是布尔值")
        logged = fold_plan_mode(self._session.snapshot_events())
        target = logged if self._pending is None else self._pending
        if active == target:
            return "noop"
        if _has_open_turn(self._session.snapshot_events()):
            self._pending = active
            self._narrate = True
            return "cancelled" if active == logged else "queued"
        if active == logged:
            self._pending = None
            return "cancelled"
        self._session.append("plan/mode", {"active": active})
        self._pending = None
        return "committed"

    def apply_boundary(self) -> str | None:
        """在请求组装前提交待选择；返回需要注入的用户切换通知。"""
        pending = self._pending
        if pending is None:
            return None
        if pending != fold_plan_mode(self._session.snapshot_events()):
            self._session.append("plan/mode", {"active": pending})
        self._pending = None
        narrate = self._narrate
        self._narrate = False
        if not narrate or _mode_at_last_header(self._session.snapshot_events()) in (
            None,
            pending,
        ):
            return None
        return (
            "The user switched this session to plan mode."
            if pending
            else "The user switched this session back to the default mode."
        )

    def prompt_section(self) -> str:
        target = (
            fold_plan_mode(self._session.snapshot_events())
            if self._pending is None
            else self._pending
        )
        return self._section if target else ""

    def exit(self, plan: str) -> str:
        """呈交完整计划；只有用户明确、唯一批准才排队退出。"""
        if not fold_plan_mode(self._session.snapshot_events()):
            raise RuntimeError(f"{EXIT_PLAN_MODE} is only available in plan mode")
        if re.match(r"^#\s+\S", plan.strip()) is None:
            raise RuntimeError(
                f"{EXIT_PLAN_MODE} requires a non-empty markdown plan starting with a # heading"
            )
        if self._questions is None:
            raise RuntimeError(
                "no user-questions channel is available to review the plan; "
                "ask the user to switch the session mode instead"
            )
        try:
            answer = self._questions.ask(
                QuestionRequest(
                    questions=(
                        Question(
                            id=REVIEW_ID,
                            header="Plan review",
                            question="Approve this plan and leave plan mode?",
                            detail=plan,
                            options=(
                                QuestionOption(
                                    APPROVE_LABEL,
                                    "Leave plan mode; the plan is carried out from the next step.",
                                ),
                                QuestionOption(
                                    KEEP_PLANNING_LABEL,
                                    "Stay in plan mode; feedback goes back to the model.",
                                ),
                            ),
                            intent=QuestionIntent("plan-review", APPROVE_LABEL),
                        ),
                    ),
                    agent=self._agent,
                )
            )
        except UserQuestionError as error:
            if error.code == "ASK_CANCELLED":
                raise RuntimeError(
                    "The user dismissed the plan review to speak instead; stay in plan "
                    "mode, stop here, and wait for their message."
                ) from error
            raise
        matches = [item for item in answer.answers if item.id == REVIEW_ID]
        item = matches[0] if len(matches) == 1 else None
        approved = (
            item is not None
            and item.selected == (APPROVE_LABEL,)
            and item.custom is None
        )
        if not approved:
            feedback = "" if item is None or item.custom is None else item.custom
            if feedback:
                raise RuntimeError(
                    f"The user chose to keep planning; their feedback: {feedback}"
                )
            raise RuntimeError(
                "The user chose to keep planning; revise the plan and present it again."
            )
        self._pending = False
        self._narrate = False
        return "Plan approved — plan mode exited; carry out the plan starting with your next step."


def make_exit_plan_mode_tool(controller: PlanModeController) -> Tool:
    """工具始终注册；当前模式只在执行时校验。"""
    return Tool(
        name=EXIT_PLAN_MODE,
        description=(
            "Use only in plan mode. Present your COMPLETE markdown plan for the "
            "user's review and, on approval, leave plan mode."
        ),
        parameters={
            "type": "object",
            "properties": {"plan": {"type": "string"}},
            "required": ["plan"],
        },
        execute=lambda arguments: controller.exit(_plan_argument(arguments)),
    )


def _plan_argument(arguments: dict[str, object]) -> str:
    plan = arguments.get("plan")
    if not isinstance(plan, str):
        raise TypeError("plan 必须是字符串")
    return plan


def _has_open_turn(events: tuple[SessionEvent, ...]) -> bool:
    open_turn = False
    for event in events:
        if event.type == "turn/start":
            open_turn = True
        elif event.type == "turn/end":
            open_turn = False
    return open_turn


def _mode_at_last_header(events: tuple[SessionEvent, ...]) -> bool | None:
    last_header = -1
    for index, event in enumerate(events):
        if event.type == "request/header":
            last_header = index
    if last_header < 0:
        return None
    return fold_plan_mode(events[: last_header + 1])
