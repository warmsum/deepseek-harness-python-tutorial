"""第 13 章：模型真实使用问答、计划、目标和任务清单工具。"""

from __future__ import annotations

import json
from typing import Any

from agent import run_agent
from client import DeepSeekClient, Tool
from goal import GoalStore
from plan import PlanModeController, make_exit_plan_mode_tool
from session import Session
from todo import TodoItem, todo_write
from user_questions import (
    AnswerItem,
    QuestionAnswer,
    QuestionRequest,
    UserQuestionService,
    make_ask_user_tool,
)


class TeachingProvider:
    """终端教学用回答器：普通问题选第一项，计划评审选择批准。"""

    def ask(self, request: QuestionRequest) -> QuestionAnswer:
        question = request.questions[0]
        selected = (question.options[0].label,) if question.options else ()
        print(f"用户问答: {question.question} -> {selected[0] if selected else '(无选项)'}")
        return QuestionAnswer((AnswerItem(question.id, selected),))


def main() -> None:
    session = Session()
    goals = GoalStore(session)
    questions = UserQuestionService()
    questions.register_provider(TeachingProvider())
    plan_mode = PlanModeController(
        session,
        (
            "当前处于计划模式。先澄清范围并形成计划，不要开始实施。"
            "准备好完整计划后调用 exit_plan_mode 请求用户评审。"
        ),
        questions,
    )
    plan_mode.set(True)

    def create_goal(arguments: dict[str, Any]) -> str:
        if plan_mode.get().active:
            raise RuntimeError("必须先退出计划模式")
        objective = arguments.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective 必须是非空字符串")
        ref = goals.create(objective, max_rounds=5)
        return json.dumps({"id": ref.id, "revision": ref.revision}, ensure_ascii=False)

    def write_todos(arguments: dict[str, Any]) -> str:
        if plan_mode.get().active:
            raise RuntimeError("必须先退出计划模式")
        if goals.get() is None:
            raise RuntimeError("必须先创建目标")
        raw_items = arguments.get("todos")
        if not isinstance(raw_items, list):
            raise TypeError("todos 必须是数组")
        items = [
            TodoItem(str(item["content"]), str(item["status"]))
            for item in raw_items
            if isinstance(item, dict)
        ]
        if len(items) != len(raw_items):
            raise ValueError("每个 todo 必须是对象")
        return todo_write(session, items, allow_parallel_in_progress=False)

    tools = [
        make_ask_user_tool(questions),
        make_exit_plan_mode_tool(plan_mode),
        Tool(
            "create_goal",
            "退出计划模式后，创建唯一的长期目标。",
            {
                "type": "object",
                "properties": {"objective": {"type": "string"}},
                "required": ["objective"],
            },
            create_goal,
        ),
        Tool(
            "todo_write",
            "退出计划模式并创建目标后，整体写入当前任务清单。",
            {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            write_todos,
        ),
    ]
    result = run_agent(
        DeepSeekClient(),
        session,
        plan_mode,
        tools,
        (
            "请为“把旧工具迁移到新框架”建立任务状态。严格按顺序："
            "先用 ask_user_question 确认是否只迁移公开接口；再用 exit_plan_mode "
            "提交 Markdown 计划；获批后用 create_goal 建立目标；最后用 todo_write "
            "写入 3 项清单，其中第一项 in_progress，其余 pending，然后总结。"
        ),
    )

    print("\n=== 模型调用的任务工具 ===")
    for trace in result.traces:
        print(f"{trace.name}({trace.arguments})")
        print(trace.result)
    print(f"\n模型最终回答:\n{result.final_text}")
    current = goals.get()
    print(
        "\n日志恢复出的目标: "
        + (f"r{current.revision} [{current.phase}] {current.objective}" if current else "无")
    )
    todo_events = [event for event in session.events if event.type == "todo/write"]
    print(f"日志中的任务清单写入次数: {len(todo_events)}")


if __name__ == "__main__":
    main()
