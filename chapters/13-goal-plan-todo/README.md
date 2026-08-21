# 13｜目标、计划与任务清单

> 预计时间：60 分钟 ｜ 前置：完成第 12 章 ｜ 本章调用真实 DeepSeek 模型

第 12 章解决了智能体怎样按需获得任务指导，但操作手册不能代替任务本身的状态管理。第 07 章的智能体已经能够连续对话，连续对话却不等于能够管理长期任务。以“把一个旧工具迁移到新框架”为例，程序需要随时回答：当前目标是什么？是否应该继续？接下来有哪些步骤？如果这些信息只写在普通对话里，模型可能忽略它们；如果只存在内存中，程序退出后又会丢失。

本章把这些信息分成四类，各自解决一个问题：

- 长期目标（Goal）：记录当前唯一的目标，以及它正在进行、暂停、完成还是阻塞；
- 计划模式（Plan Mode）：表示当前会话是否只调查和设计，暂不开始实施；
- 用户问答（User Questions）：让智能体暂停并取得结构化的人类选择；
- 任务清单（Todo）：记录眼前需要完成的几项工作。

四类信息都写入第 05 章的会话日志，需要恢复时再从事件中重新计算当前状态。这样，任务管理不会成为一份与对话历史相互矛盾的独立数据。

## 学习目标

完成本章后，你将能够：

- 使用 `active`、`paused`、`complete` 和 `blocked` 表示目标状态；
- 用 `GoalRef` 中的目标编号和版本号拒绝基于旧状态的修改；
- 把目标变更写入会话日志，并通过重放恢复当前目标；
- 在下一步骤开始时提交计划模式的切换，并通过用户明确评审退出；
- 使用统一的用户问答服务传递问题与结构化答案；
- 使用整体替换的方式维护并校验任务清单；
- 让模型按顺序完成范围确认、计划评审、目标创建和任务清单写入。

## 13.1 计划模式与用户问答为什么分开

计划模式是一种协作状态，不是文件权限。开启后，下一次系统提示词会增加“先调查和设计、暂不实施”的要求；真正能否写文件或执行命令，仍由第 10、11 章的权限与审批规则决定。

`UserQuestionService` 只负责把结构化问题交给当前交互界面，并等待用户回答。同一时间只能有一个界面提供这项服务，空问题会被拒绝。只有当前任务最外层的智能体可以发起提问，子智能体不能让父任务停在一个无人处理的提问上。

退出计划模式的工具 `exit_plan_mode` 始终存在，但只有当前确实处于计划模式时才能成功执行。它要求智能体提交一份以 Markdown 标题开头的完整计划，再向用户发起确认。只有用户明确选择 `Approve`，并且没有附加修改意见，计划才算通过。

批准后不会立刻关闭计划模式。控制器先保存一项待生效的选择，到下一步骤开始时再写入 `plan/mode=false`。这样，同一批模型工具调用中的剩余操作仍使用原来的提示词和工具清单，不会在执行到一半时改变运行条件。

## 13.2 为什么目标需要明确状态

智能体执行一个持续数小时的任务时，可能遇到这些情况：

- 用户说先停一下，目标该暂停，还是销毁？
- 依赖的上游服务挂了，目标是失败，还是等待？
- 程序中断后从磁盘恢复，目标是否仍然存在，又是否应该自动继续？

这些情况要求目标拥有明确状态，并规定哪些状态可以相互转换。代码使用四个状态和六种操作：

| 操作 | 效果 | 状态变化 |
|------|------|----------|
| `create` | 建立目标 | revision=1、phase=active、启用续行 |
| `edit` | 改目标文本 | 保留 phase、blocker reason 与 activation |
| `pause` / `resume` | 暂停 / 恢复 | 停用 / 恢复续行；resume 清除阻塞原因 |
| `complete` | 完成 | 停用续行 |
| `block` | 阻塞 | 记录文本说明，只用一个持久 phase |

这里有两条重要规则：

第一，同一时间最多只有一个当前目标。已经完成的目标可以被新目标替换，但不能同时维护多个含义不清的进行中目标。

第二，“是否立即继续执行”不会写入持久化状态。会话从中断中恢复后，即使目标仍是 `active`，智能体也不会自动继续，必须显式调用 `resume`。恢复后的工作区和依赖可能已经变化，先停下来确认比直接继续更安全。

## 13.3 用 GoalStore 和版本号保护更新

目标状态的载体：

```python
@dataclass(frozen=True)
class Goal:
    id: str
    revision: int
    phase: str
    objective: str
    max_rounds: int
    rounds_started: int = 0
    blocker_reason: str | None = None
```

每个操作都遵循相同流程：校验当前状态，生成新快照，将版本号 `revision` 加一，再追加 `goal/change` 事件。以 `resume` 为例：

```python
    def resume(self, ref: GoalRef) -> GoalRef:
        current = self._require(ref)
        if current.rounds_started >= current.max_rounds:
            raise ValueError("目标轮次已达上限，无法 resume")
        self._commit(self._with_phase(current, PHASE_ACTIVE, None), "resume")
        return GoalRef(id=current.id, revision=current.revision + 1)
```

`_require` 会检查调用方持有的目标编号和版本号：

```python
    def _require(self, ref: GoalRef) -> Goal:
        if self._current is None:
            raise ValueError("没有当前目标")
        if self._current.id != ref.id:
            raise ValueError(f"引用指向不同的目标（ref={ref.id} != 当前={self._current.id}）")
        if self._current.revision != ref.revision:
            raise ValueError(
                f"陈旧的引用（ref r{ref.revision} != 当前 r{self._current.revision}），"
                "请重新 get() 后再操作"
            )
        return self._current
```

为什么每次操作都要检查 `revision`？假设智能体 A 根据 r3 版本决定暂停目标，而智能体 B 已经把目标推进到 r5。如果仍允许 A 提交，旧决定就可能覆盖 B 的新进展。版本检查会拒绝这种基于旧状态的修改，并要求调用方重新读取当前目标。这是一种乐观并发控制方法。

## 13.4 从事件恢复目标

每个动词最后都做同一件事，`_commit` 追加事件：

```python
    def _commit(self, goal: Goal, operation: str) -> None:
        self._current = goal
        self._session.append(
            "goal/change",
            {"version": 1, "operation": operation, "goal": _goal_to_dict(goal)},
        )
```

`goal/change` 同时记录操作名和变更后的完整快照，因此目标状态可以像第 05 章的会话一样通过事件重建。`clear()` 会写入一条删除标记，英文常称为 tombstone，明确记录“这个目标被删除了”，而不是让状态无缘无故消失。

```python
    @classmethod
    def replay(cls, session: Session) -> "GoalStore":
        store = cls(session)
        for event in session.events:
            if event.type == "user/message":
                source = event.data.get("source")
                if isinstance(source, dict) and source.get("kind") == "goal":
                    # 校验 goal id、revision 与 round 连续后，
                    # 只推进 rounds_started。
                    ...
                continue
            if event.type == "goal/change":
                # 校验完整快照、revision 或 clear tombstone。
                ...
        return store
```

`admit_round()` 追加的是一条注明来自目标的 `user/message`。它表示智能体开始处理新一轮目标消息，不是修改目标本身，因此只增加 `rounds_started`，不会增加 `revision`。版本连续性只在同一个目标内检查；创建新目标时重新从 r1 开始。

教学版会检查目标编号、版本号、轮次和删除标记是否连续，但没有实现官方的全部数据形状、非法状态迁移和时间戳顺序检查。练习 2 会继续设计状态迁移规则。

## 13.5 任务清单 Todo：每次写入完整列表

长期目标记录任务是否继续，任务清单 Todo 则记录眼前需要完成的步骤。`todo_write` 每次接收一份完整列表，并用它替换旧列表，不提供只修改其中一项的操作：

```python
def todo_write(
    session: Session,
    items: list[TodoItem],
    *,
    allow_parallel_in_progress: bool,
) -> str:
    errors = validate_todos(
        items, allow_parallel_in_progress=allow_parallel_in_progress
    )
    if errors:
        return "Error: invalid todos: " + "; ".join(errors)
    snapshot = [{"content": item.content, "status": item.status} for item in items]
    session.append("todo/write", {"todos": snapshot})
    # ...返回统计文本
```

三个设计意图：

1. 每个 `todo/write` 事件都是完整快照。恢复时只需采用最后一份列表，不会出现只增加或删除到一半的中间状态。
2. 每项任务只有等待中 `pending`、进行中 `in_progress` 和已完成 `completed` 三种状态，避免引入含义相近的额外状态。
3. 写入前会去除内容两端的空白，再检查空项、重复项和非法状态。`allow_parallel_in_progress=False` 时还会拒绝同时存在多个进行中任务。返回的错误会明确告诉模型应该修正什么。

## 13.6 把四类任务状态接入模型循环

示例从计划模式开始，并向模型同时注册四个工具：`ask_user_question`、`exit_plan_mode`、`create_goal` 和 `todo_write`。用户任务明确要求依次确认范围、提交计划、创建目标，再写入三项任务清单。模型需要根据每次工具结果决定下一步，而不是由 Python 代码直接调用四项能力。

`agent.py` 在每个步骤开始时调用 `plan_mode.apply_boundary()`，再根据当前模式重新组装系统提示词：

```python
for step in range(1, max_steps + 1):
    notice = plan_mode.apply_boundary()
    system_prompt = "你是任务规划助手……"
    plan_section = plan_mode.prompt_section()
    if plan_section:
        system_prompt += "\n\n" + plan_section
    reply = client.chat(
        [Message("system", system_prompt), *session.derive_messages()],
        tools,
    )
```

`exit_plan_mode` 获批时只把退出选择放入待生效状态。`create_goal` 和 `todo_write` 还会检查当前是否仍处于计划模式，因此模型即使把它们与退出工具放在同一批调用中，也不会提前实施。下一步骤提交 `plan/mode=false` 后，目标和清单工具才会成功。

命令行示例没有交互界面，因此 `TeachingProvider` 代替终端用户选择每个问题的第一项，并在计划评审时选择 `Approve`。这只固定了人类输入；问题内容、计划、目标和清单仍由真实模型生成，所有工具调用与结果都进入同一份会话日志。

## 13.7 运行完整示例

```bash
uv run python chapters/13-goal-plan-todo/src/demo.py
```

下面是一次真实运行的关键输出。模型生成的问题、计划文字和目标表述可能变化：

```
用户问答: 旧工具迁移到新框架时，是否只迁移公开接口？ -> 是，只迁移公开接口
用户问答: Approve this plan and leave plan mode? -> Approve

=== 模型调用的任务工具 ===
ask_user_question({...})
{"answers":[{"id":"scope","selected":["是，只迁移公开接口"]}]}
exit_plan_mode({'plan': '## 迁移计划……'})
工具执行出错: exit_plan_mode requires a non-empty markdown plan starting with a # heading
exit_plan_mode({'plan': '# 迁移计划：旧工具 → 新框架（仅公开接口）……'})
Plan approved — plan mode exited; carry out the plan starting with your next step.
create_goal({'objective': '将旧工具仅迁移公开接口到新框架……'})
{"id": "goal-26", "revision": 1}
todo_write({'todos': [...]})
Updated todo list: 2 pending, 1 in progress, 0 completed.

模型最终回答:
已确认迁移范围，计划已经批准，并已创建长期目标和三项任务清单。

日志恢复出的目标: r1 [active] 将旧工具仅迁移公开接口到新框架……
日志中的任务清单写入次数: 1
```

这次运行还展示了错误回灌。模型第一次提交的计划以二级标题 `##` 开头，不符合工具要求；错误作为工具结果进入下一次请求后，模型改用一级标题 `#` 并再次提交。批准后，模式在步骤边界切换，目标与清单依次写入事件日志。最后打印出的目标和 `todo/write` 次数来自真实日志，而不是模型的文字总结。

## 本章小结

- `Goal`：用四种状态和六种操作管理一个长期目标
- `UserQuestionService`：统一传递结构化问题与答案，并限制由当前根智能体发起
- `PlanModeController`：记录计划模式，让退出选择在下一步骤生效
- `GoalRef`：通过目标编号和版本号拒绝过期修改
- `goal/change`：保存完整目标快照，并支持从日志恢复
- `todo_write`：整体替换任务清单，校验内容和状态
- 真实模型流程：范围确认、计划评审、模式切换、目标和清单共用一个事件驱动循环
- 恢复后不会自动续行，必须重新确认当前环境后再继续

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/goal/goal/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/goal/goal/README.zh.md) | `GoalStore` | 保留从事件恢复状态、`GoalRef` 版本保护、单一目标、六种操作、完整快照和恢复后不自动续行；教学版只校验事件连续性 |
| 同上 | `admit_round` | 官方只有已经接收且来源为目标的 `user/message` 才会增加目标轮数；普通用户对话不会增加 `roundsStarted` |
| [`packages/todo/tool-todo/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/todo/tool-todo/README.zh.md) | `todo_write` | 对齐整体替换、完整快照、三值状态、内容校验与可配置的并行进行中策略 |
| [`packages/interaction/user-questions/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/interaction/user-questions/README.zh.md) | `UserQuestionService` | 与官方一样只允许一个交互界面，拒绝空问题和非法调用方；教学版会同步等待用户回答 |
| [`packages/plan/plan-mode/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/plan/plan-mode/README.zh.md) | `PlanModeController` | 与官方一样由最后一条事件决定模式，在步骤边界提交选择，并要求用户明确评审计划；教学版没有 `/plan` 命令 |

长期目标、计划模式与任务清单分别回答三个问题：长任务是否继续、当前如何协作、眼前有哪些步骤。它们不能互相替代。

## 练习

1. 长期目标、计划模式、用户问答和任务清单都与任务状态有关，但时间尺度和职责不同。请为“迁移一个旧服务”设计它们的分工，并说明哪些信息不应重复保存。
2. 一个长任务可能经历暂停、恢复、阻塞和完成。画出你认为合理的状态迁移，并说明哪些迁移必须拒绝、哪些需要人类确认，以及崩溃后如何从事件日志恢复。
3. 计划模式只改变协作提示，不是文件权限。假设模型在计划阶段仍尝试写文件，系统还需要哪些独立控制？为什么不能依赖提示词保证安全？
4. 两个调用方基于同一个目标版本同时更新时，其中一个操作会被拒绝。讨论这种乐观并发控制对自动化智能体和人类协作者的好处，并设计冲突后的重新读取或合并流程。
5. 扩展本章的真实模型流程：先让用户拒绝一次计划并给出反馈，再批准修订版；随后完成一项任务并更新目标。中途保存和恢复会话，验证当前模式、目标与清单都能从日志重新得到。
