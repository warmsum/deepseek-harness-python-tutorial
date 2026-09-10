# 07｜多轮运行与消息队列

> 预计时间：65 分钟 ｜ 前置：完成第 06 章 ｜ 本章调用真实 DeepSeek 模型

第 06 章的 `run_agent` 只处理一个问题，返回结果后便结束。真实会话通常包含多次连续提问，智能体需要保留已有上下文；如果用户在智能体执行工具期间补充要求，新消息还需要在合适的时机生效。本章把一次性函数改造成可以持续接收消息的 `Agent`，并回答两个问题：

1. 新消息在智能体忙碌时到达，该怎么排队、何时生效？
2. 一次任务中，哪些操作属于同一轮，哪些属于不同步骤？

为此，智能体会维护一个收件箱 `Inbox`，其中有两条消息队列：一条等待下一轮处理，另一条在当前轮的下一步生效。队列的每次插入和领取都会写入 Session，使尚未处理的消息也可以恢复。模型请求还可能遇到限流、超时或临时服务错误，因此本章也会加入次数有限的自动重试。

## 学习目标

完成本章后，你将能够：

- 准确区分 turn（轮次）与 step（步骤）；
- 使用 Session 事件保存 `Inbox` 的“下一轮”和“下一步”队列；
- 让同一个智能体连续处理多轮输入并保留会话历史；
- 说明后续问题 `followup` 与中途引导 `steer` 的生效时机为什么不同；
- 对可以恢复的模型错误进行有限重试，并记录每次等待和重试。

## 7.1 轮次与步骤

先明确本章会反复使用的两个术语。turn（轮次）表示智能体从接收一条输入到完成回答的一次完整运行，对应事件日志中 `turn/start` 与 `turn/end` 之间的区间。step（步骤）表示轮次内部的一次模型调用及其后续工具执行；如果模型需要连续调用工具，一个轮次就会包含多个步骤。

新消息根据生效时机进入不同队列：

| 消息类型 | 进入的队列 | 领取时机 |
|----------|------------|----------|
| 后续问题 `followup` | 下一轮 `next-turn` | 当前轮次结束后，作为下一轮输入 |
| 中途引导 `steer` | 下一步 `next-step` | 当前轮次的下一个步骤开始前 |

轮次为什么重要，两个实际用途：

1. 轮次边界适合执行维护操作。压缩（第 09 章）、持久化（第 08 章）可以安排在这里，因为此时没有进行到一半的模型调用或工具调用。
2. 轮次和步骤也便于分析成本和调试问题。日志能够回答一次任务包含几轮、每轮调用了几次模型，以及错误发生在哪一步。

## 7.2 两条队列，两个处理时机：Inbox

```python
class Inbox:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._state()  # 立即校验已有 splice 事件

    def followup(self, message: Message) -> None:
        self._insert("next-turn", message)

    def steer(self, message: Message) -> None:
        self._insert("next-step", message)

    def claim_turn(self) -> list[Message]:
        state = self._state()
        claimed = list(state["next-step"])
        # 领取也追加 agent/inbox/spliced，删除已领取项
        ...
        return claimed
```

三个细节：

- `_state()` 从 `agent/inbox/spliced` 事件折叠出当前两条队列；重新创建 Inbox 时不依赖旧进程内存。
- `claim_turn` 一次取出全部 `next-step` 消息，再取一条 `next-turn` 消息；中途引导因此会排在已经等待的下一轮问题之前。`claim_step` 也会一次取出当前批次的全部消息。
- 教学版由 `run()` 主动处理队列，保留了内部领取方法；官方只把追加、前插、替换、移除、清空和队列快照作为公共 `agent.inbox` 接口，领取仍由驱动器内部完成。

## 7.3 持续处理消息的智能体循环

`Agent` 持有消息队列、会话日志和组装模型请求所需的对象，对外提供两个消息入口：`followup` 提交下一轮问题，`steer` 提交希望在当前轮下一步生效的要求。

```python
class Agent:
    def __init__(self, client, registry, assembler, variables=None):
        self._client = client
        self._registry = registry
        self._assembler = assembler
        self._variables = variables
        self._session = Session()
        self._inbox = Inbox(self._session)
        self._turn_no = 0

    def followup(self, content: str) -> None:
        self._inbox.followup(Message(role="user", content=content))

    def steer(self, content: str) -> None:
        self._inbox.steer(Message(role="user", content=content))

    def run(self, max_turns: int = 5) -> Session:
        turns_run = 0
        while self._inbox.pending > 0 and turns_run < max_turns:
            turns_run += 1
            self._turn_no += 1
            self._session.append("turn/start", {"turn": self._turn_no})
            claimed = self._inbox.claim_turn()
            self._run_turn(claimed)
            self._session.append(
                "turn/end", {"turn": self._turn_no, "reason": "completed"}
            )
        return self._session
```

主循环按固定顺序运行：记录轮次开始、领取消息、执行本轮、记录轮次结束，直到消息队列清空或本次 `run()` 达到 `max_turns`。第 06 章的循环体移入 `_run_turn` 后，只增加了一个步骤：

```python
    def _run_turn(self, claimed) -> None:
        for step in range(1, 11):
            if step > 1:
                claimed = self._inbox.claim_step()
            self._session.append("step/start", {"turn": self._turn_no, "step": step})
            try:
                system_prompt = self._assembler.render(self._variables)
                self._session.record_system_prompt(
                    system_prompt, turn=self._turn_no, step=step
                )
                for message in claimed:
                    self._session.append("user/message", {"content": message.content})
                # 记录 request/header，请求模型并执行工具
                ...
            finally:
                self._session.append("step/end", {"turn": self._turn_no, "step": step})
            if completed and not self._inbox.has_next_step:
                return
```

每个步骤都有明确边界。即使模型已经给出最终文本，只要请求期间又收到中途引导，当前轮就会继续执行下一步，而不是先结束当前轮再另开一轮。这就是 `steer` 能在当前轮生效的原因。

教学版仍由调用方同步执行 `run()`：只要队列中有消息就继续运行，队列暂时为空时返回。`max_turns` 限制单次调用处理的轮数，不会阻止以后再次调用 `run()`。官方实现可以在空闲时等待并在收到消息后自动恢复，还支持按消息标识编辑和删除队列内容；本章只实现插入、领取、清空与日志恢复，并保留完成、错误和单次运行轮数限制。

## 7.4 模型请求失败后怎样重试

模型请求失败不等于任务无法继续。限流、服务端临时错误、超时和网络中断通常可以稍后再试；请求参数错误和认证失败则不会因为等待而自动恢复，应立即报告。`RetryPolicy` 根据错误类型决定是否重试。

```python
while True:
    try:
        reply = self._client.chat(messages, tools)
        if not reply.content and not reply.reasoning_content and not reply.tool_calls:
            raise _EmptyResponseError("model returned a completed response with no content")
    except Exception as error:
        if self._retry_policy is None or not self._retry_policy.recover(
            self._session,
            turn=self._turn_no,
            step=step,
            error=error,
        ):
            raise
        continue
    break
```

教学版默认只重试空响应、限流、服务端错误、超时和网络传输错误，最多五次。等待时间从 500 毫秒开始逐次增加，上限为 10 秒，并加入少量随机偏移，避免多个请求同时再次访问服务。如果响应中的 `Retry-After` 给出了合理的等待时间，就优先采用它；如果服务端要求等待过久，本轮直接失败，避免一次请求长时间占住智能体。

准备重试时，程序先追加 `llm/retry` 事件，记录错误类型、当前次数和等待时间；等待结束后再追加 `llm/retry-started`。失败的请求不会产生模型消息，也不会额外开始新的轮次或步骤。程序会复用当前步骤已经组装好的请求，只重新调用模型服务。

这样做有两个作用。第一，当前步骤开始前已经完成的准备工作不会因为临时网络错误重复执行。第二，轮次和步骤仍表示一次完整任务过程，同时日志又能看出模型请求尝试了几次、每次等待多久。

`RetryPolicy` 不只在内存中保存重试次数，而是从当前 `Session` 的已有事件中重新计算。程序恢复后仍能知道同一请求已经尝试过多少次，不会因为重启而重新获得一整份重试额度。

教学版只实现次数有限的重试模式。等待期间会占用当前线程，也没有实现始终重试、异步取消和多个重试插件组合等能力。

## 7.5 运行完整示例

```bash
uv run python chapters/07-agent-inbox/src/demo.py
```

下面是两轮都调用一次 calculator 时的代表性输出。模型可能直接回答或增加工具调用，因此 assistant 文本、step 数和事件总数不是固定值；turn/step 的嵌套规则保持不变：

```
=== 第 1 轮：问 1+2*3 ===
  [assistant] 1+2×3 = **7**

按运算顺序，先算乘法 2×3=6，再算加法 1+6=7。

=== 第 2 轮：followup 再问 8/4（同一会话，历史延续） ===
  [assistant] 1+2×3 = **7**

按运算顺序，先算乘法 2×3=6，再算加法 1+6=7。
  [assistant] 8÷4 = **2**

=== 事件日志：两个轮次边界 ===
  #0  agent/inbox/spliced   ← followup 入队
  #1  turn/start            ← 轮次边界
  #2  agent/inbox/spliced   ← 驱动器领取消息
  #3  step/start
  #4  system/message
  #5  user/message
  #6  request/header
  ... assistant/message / tool/call / tool/result ...
  #N  turn/end              ← 轮次边界
```

三个观察点：

1. 历史能够延续。第 2 轮打印出的模型消息中包含第 1 轮的答案，说明新一轮确实使用了前面的对话。
2. 轮次边界清楚。每轮都由 `turn/start` 与 `turn/end` 包围；模型行为即使改变，事件数量可能不同，轮次仍会完整闭合。
3. 这次每轮包含两个步骤：第一步请求工具，第二步生成最终文本。一次模型调用及其后续工具执行就是一个步骤，不能只根据模型消息数量判断。

## 本章小结

- `Inbox`：通过 `agent/inbox/spliced` 持久记录下一轮和下一步骤队列
- `Agent`：持续领取消息，并为每轮、每步记录明确边界
- `followup` 与 `steer`：分别在下一轮和当前轮的下一步生效
- `RetryPolicy`：只重试可能恢复的错误，并限制次数和等待时间
- turn（轮次）与 step（步骤）：描述一次任务和其中的模型调用

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/core/agent-loop/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/core/agent-loop/README.zh.md) | 术语 | 官方同样用 turn 表示从唤醒到结束的一轮，用 step 表示一次模型调用和后续工具执行 |
| [`packages/core/agent-loop/src/inbox.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/core/agent-loop/src/inbox.ts) | `Inbox` | 对齐两级队列和 `agent/inbox/spliced` 恢复；教学版没有消息标识以及按标识编辑、删除接口 |
| [`packages/core/agent/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/core/agent/README.zh.md) | `Agent` | 官方通过 `agent.inbox` 公开队列操作，具体领取由默认驱动器内部完成；教学版直接展示同步驱动器 |
| 同上 | `_run_turn` | 核心循环只负责调用模型、运行工具和重复，其余行为由插件与事件组合 |
| 同上 | 会话日志 | 已经接收的消息、请求边界和工具调用都会写入日志，并用于后续步骤重建请求 |
| [`packages/llm/llm-retry/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/llm/llm-retry/README.zh.md) | `RetryPolicy` | 教学版保留有限次数、服务端等待时间、有上限的退避和重试事件；没有始终重试、取消和多个重试插件组合 |

## 练习

1. 一次用户任务可能经历多次模型调用和工具执行。请用一个需要两次工具调用的例子划分轮次与步骤，并说明错误地把每次模型调用都当成新一轮会影响哪些日志和状态。
2. 智能体正在执行任务时，用户先补充下一项工作，随后又要求立刻改变当前回答格式。两条消息分别应进入哪个队列、何时生效？如果顺序反过来，结果是否应当变化？
3. 重试能够掩盖临时网络故障，也可能放大费用或重复副作用。请为限流、认证失败、空响应、工具失败和用户取消分别决定是否重试，并说明预算与退避策略。
4. 当下一轮消息持续到达、当前轮长时间不结束或 `max_turns` 提前耗尽时，消息队列中可能一直留下无法处理的内容。设计一种对用户可解释的收尾策略。
5. 使用可控的假客户端扩展本章智能体，完成一个包含工具调用、中途引导、后续问题和临时模型失败的多轮场景。输出事件时间线，证明消息在预期位置生效，重试没有创建额外轮次或重复接纳输入。
