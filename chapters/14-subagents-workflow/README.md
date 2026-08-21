# 14｜子智能体、后台任务与工作流

> 预计时间：60 分钟 ｜ 前置：完成第 13 章 ｜ 本章调用真实 DeepSeek 模型

第 13 章让一个智能体能够记录目标、计划和任务清单，但所有工作仍由它自己按顺序完成。遇到可以相互独立的任务，例如同时查询多份资料或验证两个方案，逐个执行会增加等待时间，把所有过程放进同一段对话又会混入无关信息。

本章用子智能体处理独立任务：父智能体提交一段完整的任务说明，子智能体在自己的会话中完成工作，再返回结果。随后继续加入三项能力：从父会话复制已完成历史、向同一个子智能体继续发送消息，以及在后台并发执行一批任务。代码中的 `Subagent`、`Jobs` 和 `Workflow` 分别对应子智能体、后台任务和工作流。

## 学习目标

完成本章后，你将能够：

- 判断哪些任务适合交给上下文隔离的子智能体；
- 用独立 `Session` 运行子任务并返回结构化结果；
- 使用线程池并行执行多个互不依赖的子任务；
- 只继承父会话中已经完整结束的轮次，创建分支子智能体；
- 向可继续的子智能体发送后续消息或中断当前任务；
- 使用所有权隔离的后台任务和带有并发、总量上限的工作流。

## 14.1 原理：为什么子任务要隔离

假设父智能体已经积累了数万 token 的对话历史，现在只需要委派一个“查询某个函数最新文档”的子任务。如果子智能体继承全部父会话，会带来三个问题：

- 每次子任务请求都要重复发送数万 token，增加成本；
- 父历史中的无关内容会干扰子任务；
- 子任务过程继续写入同一份历史，会让上下文增长得更快。

采用上下文隔离后，子智能体只接收一条内容完整的任务说明，不继承父会话。这样可以减少无关信息和输入成本，但父智能体必须把完成任务所需的背景写清楚。示例会展示子智能体的完整输入，其中只有系统提示词和任务两条消息。

有时子任务确实需要已有对话，例如从另一个角度继续分析当前问题。这时可以使用 fork（分支），把父会话中已经完整结束的轮次作为子会话的起点。代码只复制到最后一个 `turn/end`，不会带入尚未完成的当前轮，避免留下不完整的模型回复或工具调用。

## 14.2 运行一个独立子智能体：run_subagent

```python
@dataclass(frozen=True)
class SubagentResult:
    output: str
    stop_reason: str  # "completed" / "max-steps" / "error"
    diagnostic: str | None = None


def run_subagent(client, task, system_prompt, max_steps=3, session=None) -> SubagentResult:
    session = session or Session()  # 默认隔离，也可接收 fork/continuable seed
    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"content": task})

    partial = ""
    try:
        for _step in range(max_steps):
            reply = client.chat(
                [Message(role="system", content=system_prompt),
                 *session.derive_messages()]
            )
            event = {"content": reply.content, "tool_calls": []}
            if reply.reasoning_content:
                event["reasoning_content"] = reply.reasoning_content
            session.append("assistant/message", event)
            partial = reply.content or ""
            if reply.content:
                session.append("turn/end", {"turn": 1, "reason": "completed"})
                return SubagentResult(output=reply.content, stop_reason="completed")
        session.append("turn/end", {"turn": 1, "reason": "max-steps"})
        return SubagentResult(output=partial, stop_reason="max-steps")
    except Exception as error:
        session.append(
            "turn/end", {"turn": 1, "reason": "error", "message": str(error)}
        )
        return SubagentResult(
            output=partial, stop_reason="error", diagnostic=str(error)
        )
```

三个教学要点：

1. 默认创建全新 `Session`。子智能体的日志只记录当前任务和自己的回复，父会话不会自动进入其中；正常完成、达到步骤上限、中断和错误等路径都会写入 `turn/end`。
2. `SubagentResult` 把输出 `output`、停止原因 `stop_reason` 和诊断信息 `diagnostic` 分开。子智能体运行到一半失败时，已经生成的内容仍放在 `output`；模型服务或运行时错误只放在 `diagnostic`，不会伪装成子智能体的回答。
3. 异常会转换成 `stop_reason="error"` 的结果。父智能体因此能够区分正常回答、部分回答和失败原因，再决定是否重试或改用其他方案。

分支会话的初始内容由一个很小的日志切片函数生成：

```python
def fork_session(parent: Session) -> Session:
    last_turn_end = -1
    for index, event in enumerate(parent.events):
        if event.type == "turn/end":
            last_turn_end = index
    if last_turn_end < 0:
        return Session()
    return Session.from_log(parent.events[:last_turn_end + 1])
```

父会话还没有完整轮次时返回空会话。这里复制事件值，不共享父会话对象，因此父子双方后续追加内容时互不影响。

## 14.3 并行：多个子任务同时跑

如果父智能体一次请求两个子任务，依次执行的总耗时接近两项任务用时之和。对于主要等待网络响应的独立任务，可以使用线程池并行执行：

```python
def run_subagents_parallel(specs, client) -> list[SubagentResult]:
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [
            pool.submit(run_subagent, client, task, system_prompt)
            for task, system_prompt in specs
        ]
        return [future.result() for future in futures]
```

`DeepSeekClient.chat` 是同步调用，大部分时间在等待网络响应。线程池让这些等待同时发生，因此总耗时通常接近最慢的那个子任务。只有互不修改共享状态的任务才适合这样并行；返回结果仍按输入顺序排列。

## 14.4 委派工具：模型主动拆任务

把 `run_subagent` 包装成第 02 章介绍的工具后，模型就能在需要时主动拆分任务：

```python
subagent_tool = Tool(
    name="subagent",
    description=(
        "把一个独立的子任务委派给子 agent 执行并等待其完成。"
        "子 agent 看不到当前对话历史，只看到 task 描述。"
        "适合：独立的小任务、并行探索多个方向。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "要委派的子任务描述，要自包含（子 agent 没有上下文）",
            }
        },
        "required": ["task"],
    },
    execute=execute,
)
```

工具说明明确写出“子智能体看不到当前对话历史，只看到任务描述”，用于提醒模型提供完整背景。这是上下文隔离的必要条件，也说明工具描述会直接影响模型怎样使用工具。

## 14.5 向同一个子智能体继续发送消息

子智能体的会话是否保留，以及父智能体是否等待结果，是两个不同问题：

| 维度 | 选项 | 含义 |
|------|------|------|
| 会话方式 | 一次性 `one-shot` | 完成一个任务后结束；既可以等待结果，也可以放到后台运行 |
| 会话方式 | 可继续 `continuable` | 子智能体保留独立会话，后续还可以继续发送消息 |
| 运行方式 | 前台 `foreground` | 父智能体等待结果，再继续当前步骤 |
| 运行方式 | 后台 `background` | 立即返回任务或子智能体编号，稍后再查询结果 |

一次性子任务默认在前台运行，可继续子智能体默认在后台运行。但两组概念彼此独立：一次性任务也可以放到后台，可继续子智能体也可以暂时在前台等待。

`ContinuableSubagent` 保留一份独立会话，并使用只有一个工作线程的线程池按先到先得顺序处理消息。`submit_message()` 把消息放入队列后立即返回，即使子智能体仍在工作也能继续投递；`send_message()` 使用同一队列，但会等待当前消息处理完成。`interrupt()` 设置取消信号。同步模型请求无法在执行中途强制停止，因此程序会在调用前后检查信号，并把已经生成的部分内容与“已中断”状态分开返回。

`SubagentManager` 会记录每个子智能体属于哪个根智能体。读取、列举和发送后续消息时都要核对所有者，其他根智能体不能只凭子智能体编号访问它。关闭管理器时，会先中断并回收所有子智能体。

教学版没有实现子智能体的跨进程恢复、主动向父智能体报告进度、能力协商和最大委派深度。它保留了最重要的三条本地路径：一次性任务、继承已完成历史的分支任务，以及可以继续对话的子智能体。更完整的差异放在章末说明。

## 14.6 后台任务：查询、等待与取消

后台任务不能只返回一个线程对象，父智能体还需要一个稳定编号，以及查询状态、等待结果和取消任务的方法。`LocalJobs` 提供 `start`、`list`、`read`、`wait` 和 `kill`：

```python
job = jobs.start(owner_id, operation)
snapshot = jobs.wait(owner_id, job.id, timeout=1.0)
```

每个后台任务都绑定所有者，读取、等待和取消时都会重新检查。状态从排队中 `queued` 进入运行中 `running`；收到取消请求后先进入正在停止 `stopping`，任务真正结束后才记为已取消 `cancelled`。完成、失败或取消等最终状态只记录第一次结果，后来到达的结果不能覆盖它。

系统按所有者统计排队、运行和正在停止的任务。达到上限时，在分配编号和执行函数之前就拒绝新任务；已经结束的历史记录不占用名额。取消通过一个协作信号完成，任务函数必须主动检查，因为 Python 线程不能被安全地强制终止。`close()` 会向全部后台任务发送取消信号，并等待线程池回收。

## 14.7 工作流：有上限地批量执行

让模型逐轮决定“再启动一个子智能体”虽然灵活，却会增加模型调用次数。如果一开始就知道有一批相互独立的任务，更适合用工作流一次提交，再按照并发上限执行。

`WorkflowEngine.parallel()` 并发运行一组无参数任务函数，结果仍保持输入顺序，单项失败时对应位置为 `None`。`pipeline()` 让每个输入依次完成自己的多个阶段，不要求所有输入完成同一阶段后才能继续，因此较快的任务无需等待较慢任务。`max_concurrency` 限制同时运行数，`max_agents` 限制一次工作流最多启动多少个子任务。

教学版直接运行 Python 函数，只演示任务编排和数量限制。线程不构成安全隔离，因此不能用它执行不可信代码。官方参考版本使用 Worker Thread 执行受限 JavaScript 脚本，提供的接口和安全边界更完整。

## 14.8 运行完整示例

```bash
uv run python chapters/14-subagents-workflow/src/demo.py
```

真实输出，模型行为每次略有差异，结构稳定：

```
=== ① fork + continuable：继承完整轮次并继续投递 ===
  两次结果: 已处理：继续第一项 / 已处理：再处理第二项
  子会话用户消息: ['已完成的父任务', '继续第一项', '再处理第二项']
  owner 隔离: 其他 root 无法访问 child

=== ② Jobs：后台执行、等待与首次终态 ===
  job-…: completed, output=已完成

=== ③ Workflow：并发与逐项 pipeline ===
  parallel=['A', 'B'], pipeline=[2, 4]

=== ④ 主 agent 请求并行委派 ===
  [主 agent → 请求工具] subagent({"task": "请计算 3+3 等于多少，并给出结果。"})
  [主 agent → 请求工具] subagent({"task": "请计算 7*2 等于多少，并给出结果。"})
  （本轮请求了 2 个 subagent，将并行执行）

=== ⑤ 并行执行（计时） ===
  [子 agent#1] 6  (completed)
  [子 agent#2] 14  (completed)
  总耗时: 0.8s（若串行执行约为两倍）

=== ⑥ 上下文隔离证据 ===
  [system] 你是一个速算助手，直接给出答案。…
  [user] 请计算 3+3 等于多少，并给出结果。…
  ← 父 agent 的对话历史一个字都没进来

=== ⑦ 主 agent 汇总 ===
  [主 agent] 两个子任务都已完成，结果如下：

1. **3 + 3 = 6**
2. **7 × 2 = 14**
```

前三节不访问网络：① 展示分支子智能体只继承完整轮次、可继续子智能体能够接收后续消息，而且子智能体只能由自己的父智能体访问；② 展示后台任务的状态变化；③ 展示工作流的并发与分阶段执行。后四节才调用真实 API：模型发出两个内容完整的子任务，并行执行后把结果交给主智能体汇总。这样可以从同一个示例观察本章各模块的作用。

## 本章小结

- `run_subagent`：在独立会话中运行子智能体，并分开返回输出、停止原因和诊断信息
- `fork_session`：只继承到父会话最后一个完整轮次
- `ContinuableSubagent` 与 `SubagentManager`：保留子会话、继续发送消息、中断任务并检查所有者
- `run_subagents_parallel`：线程池并行
- `create_subagent_tool`：通过工具说明要求模型提供完整的子任务背景
- `LocalJobs`：管理后台任务状态、协作式取消和最终结果
- `WorkflowEngine`：并行执行、分阶段处理、保持结果顺序并限制任务数量

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/subagent/tool-subagent/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/subagent/tool-subagent/README.zh.md) | `create_subagent_tool`、第 17 章的 `subagent` | 对齐模型发起委派、分开返回失败诊断与部分文本，以及前后台选择 |
| [`packages/subagent/subagent/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/subagent/subagent/README.zh.md) | `run_subagent`、`ContinuableSubagent` | 教学版在当前进程中实现一次性和可继续对话的子智能体；没有具名外部服务和跨进程恢复 |
| [`packages/subagent/subagent-fork-in-process/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/subagent/subagent-fork-in-process/README.zh.md) | `fork_session` | 与官方一样只继承父会话到最后一个完整轮次为止；教学版没有创建期间的能力过滤 |
| [`packages/subagent/subagent-codex/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/subagent/subagent-codex/README.zh.md) / [`subagent-claude-code`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/subagent/subagent-claude-code/README.zh.md) | （未实现） | 官方可以把任务委派给独立的 Codex 或 Claude Code 进程；两者都使用隔离上下文完成一次性任务 |
| [`packages/jobs/jobs-local/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/jobs/jobs-local/README.zh.md) | `LocalJobs` | 与官方一样按所有者隔离任务，并支持查询、等待和取消；教学版没有完成通知、自动过期和持久化 |
| [`packages/workflow/tool-workflow/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/workflow/tool-workflow/README.zh.md) | `WorkflowEngine`、第 17 章的 `workflow` | 对齐批量智能体编排和数量限制；教学版使用 Python 函数，不执行官方 Worker Thread JavaScript |

## 练习

1. 哪些任务适合委派，哪些任务留在主智能体中更好？请从上下文依赖、可并行性、结果可验证性和沟通成本四个方面分析，并给出正反两个例子。
2. 全新会话、继承父会话和可继续对话的子智能体分别适合什么场景？为“独立查资料”“从现有讨论换个角度分析”“持续跟进测试修复”选择方式，并说明错误选择会浪费什么资源或丢失什么信息。
3. 为一个多来源研究任务设计后台任务和工作流：明确并发上限、总任务预算、取消方式、失败处理和结果顺序。部分来源失败时，父智能体应继续汇总还是让整批失败？说明判断标准。
4. 子智能体返回 `completed` 不代表结果一定正确。父智能体可以怎样利用引用、结构化输出、交叉验证和诊断信息判断结果质量？哪些内部错误适合进入诊断，哪些内容适合交给用户？
5. 实现一个包含两个阶段的子智能体工作流：第一阶段并行收集或分析独立材料，第二阶段汇总结果。它应处理至少一个失败或取消分支，并把最终输出、停止原因与诊断信息分开返回。
