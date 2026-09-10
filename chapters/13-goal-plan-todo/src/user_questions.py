"""第 13 章：用户问答服务。

对应官方 ``packages/interaction/user-questions`` 与 ``tool-ask-user``：
模型工具只负责提出结构化问题，展示方式由唯一的服务实现决定。
教学版循环是同步的，因此 ``ask`` 直接阻塞当前工具调用；等待结束后，答案仍
作为普通工具结果回到下一步模型请求。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from client import Tool

ASK_USER_QUESTION_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "detail": {"type": "string"},
                    "header": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["label"],
                        },
                    },
                    "multi_select": {"type": "boolean"},
                },
                "required": ["id", "question"],
            },
        }
    },
    "required": ["questions"],
}


@dataclass(frozen=True)
class QuestionOption:
    label: str
    description: str | None = None


@dataclass(frozen=True)
class QuestionIntent:
    kind: Literal["plan-review"]
    approve: str


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    detail: str | None = None
    header: str | None = None
    options: tuple[QuestionOption, ...] = ()
    multi_select: bool = False
    intent: QuestionIntent | None = None


@dataclass(frozen=True)
class AnswerItem:
    id: str
    selected: tuple[str, ...]
    custom: str | None = None


@dataclass(frozen=True)
class QuestionAnswer:
    answers: tuple[AnswerItem, ...]


@dataclass(frozen=True)
class QuestionRequest:
    questions: tuple[Question, ...]
    agent: object | None = None
    aborted: Callable[[], bool] | None = None


class UserQuestionProvider(Protocol):
    def ask(self, request: QuestionRequest) -> QuestionAnswer: ...


class AgentAuthority(Protocol):
    """判断调用者是否为当前运行时中可与用户交互的根 Agent。"""

    def classify(self, agent: object) -> Literal["root", "delegated", "stale"]: ...


class UserQuestionError(RuntimeError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class UserQuestionService:
    """管理唯一问答实现，并在 ``ask`` 时校验调用者。"""

    def __init__(self, authority: AgentAuthority | None = None) -> None:
        self._provider: UserQuestionProvider | None = None
        self._authority = authority

    def register_provider(self, provider: UserQuestionProvider) -> Callable[[], None]:
        if self._provider is not None:
            raise UserQuestionError(
                "a user-questions provider is already registered",
                "DUPLICATE_PROVIDER",
            )
        self._provider = provider
        active = True

        def dispose() -> None:
            nonlocal active
            if active:
                active = False
                self._provider = None

        return dispose

    def ask(self, request: QuestionRequest) -> QuestionAnswer:
        if request.aborted is not None and request.aborted():
            raise UserQuestionError(
                "ask_user_question was aborted before the user answered",
                "ASK_ABORTED",
            )
        if not request.questions:
            raise UserQuestionError(
                "ask_user_question requires at least one question",
                "EMPTY_QUESTIONS",
            )
        if request.agent is not None:
            classification = (
                "stale" if self._authority is None else self._authority.classify(request.agent)
            )
            if classification == "stale":
                raise UserQuestionError(
                    "human interaction requires the exact live calling agent when an agent is supplied",
                    "CALLER_NOT_LIVE",
                )
            if classification == "delegated":
                raise UserQuestionError(
                    "human interaction is unavailable while the calling agent is owned by another "
                    "live agent; include the unresolved question or decision in the child agent's "
                    "final result",
                    "DELEGATED_CALLER",
                )
        for question in request.questions:
            intent = question.intent
            if intent is None:
                continue
            if not any(option.label == intent.approve for option in question.options):
                raise UserQuestionError(
                    f"question {question.id} declares intent {intent.kind} whose approve label "
                    f"{json.dumps(intent.approve)} names none of its options",
                    "BAD_INTENT",
                )
            if question.detail is None:
                raise UserQuestionError(
                    f"question {question.id} declares intent {intent.kind} without the detail it reviews",
                    "BAD_INTENT",
                )
        if self._provider is None:
            raise UserQuestionError("no user-questions provider is registered", "NO_PROVIDER")
        return self._provider.ask(request)


def ask_user_question(
    service: UserQuestionService,
    arguments: Mapping[str, Any],
    *,
    agent: object | None = None,
) -> str:
    """解析问题，等待问答实现返回，再生成紧凑 JSON。"""
    raw_questions = arguments.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("questions 必须是数组")
    questions = tuple(_parse_question(item) for item in raw_questions)
    result = service.ask(QuestionRequest(questions=questions, agent=agent))
    payload = {
        "answers": [
            {
                "id": answer.id,
                "selected": list(answer.selected),
                **({"custom": answer.custom} if answer.custom is not None else {}),
            }
            for answer in result.answers
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_ask_user_tool(service: UserQuestionService) -> Tool:
    """注册表适配器；章节 demo 使用 agentless 的程序化调用路径。"""
    return Tool(
        name="ask_user_question",
        description=(
            "Ask the user a concise question when you need confirmation, a choice, "
            "or missing information before proceeding."
        ),
        parameters=ASK_USER_QUESTION_PARAMETERS,
        execute=lambda arguments: ask_user_question(service, arguments),
    )


def _parse_question(raw: Any) -> Question:
    if not isinstance(raw, Mapping):
        raise ValueError("question 必须是对象")
    id_ = raw.get("id")
    text = raw.get("question")
    if not isinstance(id_, str) or not id_:
        raise ValueError("question.id 必须是非空字符串")
    if not isinstance(text, str) or not text:
        raise ValueError("question.question 必须是非空字符串")
    raw_options = raw.get("options", [])
    if not isinstance(raw_options, list):
        raise ValueError("question.options 必须是数组")
    options: list[QuestionOption] = []
    for raw_option in raw_options:
        if not isinstance(raw_option, Mapping) or not isinstance(raw_option.get("label"), str):
            raise ValueError("option.label 必须是字符串")
        description = raw_option.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("option.description 必须是字符串")
        options.append(QuestionOption(raw_option["label"], description))
    header = raw.get("header")
    if header is not None and not isinstance(header, str):
        raise ValueError("question.header 必须是字符串")
    detail = raw.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("question.detail 必须是字符串")
    multi_select = raw.get("multi_select", False)
    if not isinstance(multi_select, bool):
        raise ValueError("question.multi_select 必须是布尔值")
    return Question(
        id=id_,
        question=text,
        detail=detail,
        header=header,
        options=tuple(options),
        multi_select=multi_select,
    )
