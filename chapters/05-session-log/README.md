# 05｜会话日志

> 预计时间：60 分钟 ｜ 前置：完成第 04 章 ｜ 本章调用真实 DeepSeek 模型

第 03、04 章解决了智能体能力增多后怎样组织和协作。现在回到第 02 章留下的运行循环：它用一个 `list[Message]` 保存对话历史，循环直接修改这个列表，再把它发送给模型。这个设计适合最小示例，但随着智能体持续运行，会出现三个问题：

1. 不是所有历史都该给模型看。一次真实运行会产生大量过程记录：流式分片、生命周期边界、压缩动作。它们要留存，但发给模型只会浪费 token、制造噪音。
2. 同一份历史有多种用途。模型请求要读取它，持久化功能要把它写入磁盘，界面要展示它，程序恢复时还要重新载入它。如果每个功能各存一份，内容很容易不同步。
3. 历史一旦写入就不该再改。第 02 章的 list 谁都能 pop、insert，一个误操作就会毁掉整段对话的上下文。

本章用一份只允许追加的事件日志统一解决这三个问题。用户消息、模型回复、工具调用和运行边界都依次写入日志；需要向模型发送历史时，再从日志中挑出相关事件并转换成消息。这种“保存事件，再根据事件还原当前状态”的方法称为事件溯源。

## 学习目标

完成本章后，你将能够：

- 区分“完整事件日志”和“发送给模型的消息投影”；
- 实现只追加、可订阅且数据不可变的 `Session`；
- 把智能体循环中的用户消息、模型回复和工具调用记录为事件；
- 从已有日志重放会话，并校验事件编号的连续性。

## 5.1 日志是事实，消息是投影

先想清楚日志和消息历史的区别。日志回答的问题是发生了什么：用户说了什么、模型回了什么、哪个工具被调用了、结果是什么、这一轮何时开始何时结束。消息历史回答的问题是模型该看到什么：把日志里对模型有用的部分挑出来，按协议格式排好。

两者的要求不同：日志要完整，消息历史则只保留模型真正需要的内容。事件溯源把它们分成两层：

```
                     ┌─────────────────────────┐
   追加事件 ────────▶ │  Session（事件日志）      │
   （唯一写入路径）    │  #0 turn/start          │
                     │  #1 user/message        │
                     │  #2 assistant/message   │
                     │  #3 tool/call           │───▶ derive_messages() ──▶ 发给模型
                     │  #4 tool/result         │      （生成消息）
                     │  #5 turn/end            │
                     └─────────────────────────┘
```

这个结构带来三个好处：运行过程能够完整保留；模型、界面和持久化功能都从同一份日志读取信息；已经写入的事件不会被后续代码改写。把事件转换成特定用途数据的过程称为“投影”，`derive_messages()` 生成的就是供模型使用的消息投影。

## 5.2 SessionEvent：日志里的一条事件

事件是日志的最小单位：

```python
@dataclass(frozen=True)
class SessionEvent:
    id: int
    type: str
    ts: float
    data: Mapping[str, Any]
```

- `id` 从 0 开始连续递增，5.6 节的重放校验靠它。
- `type` 是事件类型。除消息和工具事件外，本章还记录 `request/header`、`step/start`、`step/end` 与轮次边界，用来说明每次模型请求属于哪一轮、哪一步。
- `ts` 是时间戳，记录事件发生的时刻。
- `data` 是事件内容，`frozen=True` 保证事件对象本身不可改。

## 5.3 Session.append：唯一的写入路径

`Session` 类只提供一个写入入口，`append`：

```python
class Session:
    def __init__(self) -> None:
        self._log: list[SessionEvent] = []
        self._snapshot: tuple[SessionEvent, ...] | None = None
        self._listeners: list[Any] = []

    def append(self, type: str, data: dict[str, Any]) -> SessionEvent:
        frozen_data = _freeze_json(data)
        event = SessionEvent(id=len(self._log), type=type, ts=_now(), data=frozen_data)
        self._log.append(event)
        self._snapshot = None
        for listener in list(self._listeners):
            listener(event)
        return event
```

三个动作，各对应一个设计意图：

1. `_freeze_json(data)` 在写入前校验并冻结。它拒绝循环引用、非字符串对象键、非有限数、负零、超出 JSON 安全范围的整数和非 JSON 类型；列表变成元组，字典变成仍可序列化的 `FrozenDict`。事件和 `events` 元组都不能被调用方改写。
2. `self._snapshot = None` 让外部读日志走 `events` 属性拿缓存快照，append 后缓存失效，下次读取重建，高频读取不被每次全量复制拖慢。
3. 通知订阅者。通过 `subscribe` 注册的函数会立即收到新事件。第 08 章会实现文件存储并显式调用 `save()`；也可以进一步让存储功能订阅事件，在需要时自动保存。

## 5.4 哪些事件进模型，哪些只进日志

`derive_messages()` 是投影规则的实现：

```python
    def derive_messages(self) -> list[Message]:
        messages: list[Message] = []
        for event in self._log:
            message = _derive_event_message(event)
            if message is not None:
                messages.append(message)
        return messages


def _derive_event_message(event: SessionEvent) -> Message | None:
    if event.type == "user/message":
        return Message(role="user", content=event.data["content"])
    if event.type == "assistant/message":
        raw_calls = event.data.get("tool_calls") or []
        tool_calls = tuple(
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in raw_calls
        )
        reasoning_content = event.data.get("reasoning_content")
        if event.data.get("content") is None and not reasoning_content and not tool_calls:
            return None
        return Message(
            role="assistant",
            content=event.data.get("content"),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
    if event.type == "tool/result":
        return Message(
            role="tool",
            content=event.data["content"],
            tool_call_id=event.data["call_id"],
        )
    return None
```

三种事件投影成消息，其余一律 `None`，留在日志里，不进模型：

| 事件 | 投影结果 | 说明 |
|------|----------|------|
| `user/message` | `role="user"` | 人的话 |
| `assistant/message` | `role="assistant"` | 模型的话，可能带 reasoning_content 与 tool_calls |
| `tool/result` | `role="tool"` | 工具结果，带 tool_call_id 对应 |
| `request/header`、step/turn 边界、`tool/call` | 不投影 | 请求与过程记录，只进日志 |

`tool/call` 记录模型请求调用工具这件事本身，其中包含原始参数，但不会再次转换成模型消息。`assistant/message` 里的 `tool_calls` 已经携带了同一请求，重复发送没有意义。日志保留独立的 `tool/call`，是为了让界面和调试工具能够直接查看每次调用。

## 5.5 循环日志化

第 02 章的循环直接操作 `history` 列表。日志化之后，循环的每一步都变成追加一条事件：

```python
def run_agent(client, tools, system_prompt, user_prompt, max_steps=10) -> Session:
    tools_by_name = {tool.name: tool for tool in tools}
    session = Session()

    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"content": user_prompt})

    session.append("request/header", {"header": {"system": system_prompt, ...}})
    for step in range(1, max_steps + 1):
        session.append("step/start", {"turn": 1, "step": step})
        try:
            # 投影历史 → 请求模型 → 记录 assistant 与工具事件
            ...
        finally:
            session.append("step/end", {"turn": 1, "step": step})
    # ...
```

与第 02 章相比，这里有两个变化：

1. 系统提示词和工具说明不属于对话消息，但会写入 `request/header`。恢复时因此能知道当时使用的模型、提示词和工具，而不是只剩聊天文本。
2. `turn/start` 与 `turn/end` 标记一轮任务的开始和结束，`step/start` 与 `step/end` 标记一次模型调用及后续工具执行。`finally` 保证发生异常时也会写入步骤结束事件。

## 5.6 重放：从日志重建一切

日志全量留存后，可以通过重放重建会话状态。`from_log` 是重建入口：

```python
    @classmethod
    def from_log(cls, events: list[SessionEvent]) -> "Session":
        session = cls()
        for index, event in enumerate(events):
            if event.id != index:
                raise ValueError(
                    f"重放失败：第 {index} 个事件 id 为 {event.id}（应为 {index}）"
                )
            frozen_data = _freeze_json(event.data)
            session._log.append(SessionEvent(..., data=frozen_data))
        return session
```

从磁盘读回的数据可能损坏，因此 `from_log` 除了检查连续编号，还会重新验证事件类型、时间戳和 JSON 数据，并再次冻结 `data`。更完整的实现还需要检查轮次与步骤是否闭合，以及每个工具调用是否都有对应结果。

重放有三个常见用途：程序中断后从磁盘日志恢复会话；为子智能体复制一段已经完成的对话，第 14 章会继续讲解；以及在调试时还原一次运行的完整过程。本章只演示最基础的情况：使用同一份日志重建出完全相同的消息历史。

## 5.7 运行完整示例

```bash
uv run python chapters/05-session-log/src/demo.py
```

下面是模型调用一次 calculator、经过两个 step 后作答时的代表性输出。回复文本、tool call id 和 step 数可能随模型行为变化，但每个 step 的开始与结束、请求头和 turn 边界都会被记录：

```
=== ① 订阅者视角 ===
  [订阅者] 看到新事件 #0 user/message

=== ② 事件日志原文（唯一事实来源） ===
  #0  turn/start           {'turn': 1}
  #1  user/message         {'content': '1+2*3 等于几？'}
  #2  request/header       {'header': {...}, 'reason': 'initial'}
  #3  step/start           {'turn': 1, 'step': 1}
  #4  assistant/message    {'content': '', 'tool_calls': ({'id': 'call_00_...', 'name': 'calculator', ...},)}
  #5  tool/call            {'call_id': 'call_00_...', 'name': 'calculator', 'arguments': '{"expression": "1+2*3"}'}
  #6  tool/result          {'call_id': 'call_00_...', 'content': '7.0', 'is_error': False}
  #7  step/end             {'turn': 1, 'step': 1}
  #8  step/start           {'turn': 1, 'step': 2}
  #9  assistant/message    {'content': '根据计算，**1 + 2 × 3 = 7**。', 'tool_calls': ()}
  #10 step/end             {'turn': 1, 'step': 2}
  #11 turn/end             {'turn': 1, 'reason': 'completed'}

=== ③ 派生消息：模型看到的历史（derive_messages 投影） ===
  [user] 1+2*3 等于几？
  [assistant → 请求工具] calculator({"expression": "1+2*3"})
  [tool → 结果] 7.0
  [assistant] 根据计算，**1 + 2 × 3 = 7**。

  这里需要注意运算顺序：先算乘法（2 × 3 = 6），再算加法（1 + 6 = 7）。

=== ④ 重放：同一份日志 → 新会话 → 完全相同的消息历史 ===
  重放派生 4 条消息，与原始派生一致: True
  ← 日志是唯一事实来源：持久化、界面、重放都从它派生
```

第 ② 节完整记录了发生过的事情。在这次包含两个步骤的示例中，共有 12 条事件，其中请求头、步骤边界和工具调用记录不会转换成模型消息。第 ③ 节因此只有 4 条消息。第 ④ 节证明生成结果只依赖日志：同一份日志会得到相同的消息历史。事件 `data` 中的列表在冻结后变成元组，也能直观看到数据已经不可修改。

## 本章小结

- `SessionEvent`：日志中的一条事件，编号连续，写入后不可修改
- `Session.append`：统一的写入入口，负责校验、冻结和通知订阅者
- `derive_messages`：从完整日志中生成模型需要的消息
- 运行边界：用事件记录请求信息、轮次、步骤和工具调用
- `from_log`：校验已有事件并重新建立会话

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/core/session/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/core/session/README.zh.md) | `Session` | 与官方一样使用事件重建状态，在追加事件时复制并冻结数据，外部代码不能修改事件快照 |
| 同上 | `derive_messages` | 官方使用增量投影，每个派生视图节点只处理一次；教学版每次重新处理全部事件，长会话成本更高 |
| 同上 | `subscribe` | 官方持久化插件订阅 `session/event`，需要保存时再写入磁盘；第 08 章实现文件存储 |
| [`packages/core/agent-loop/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/core/agent-loop/README.zh.md) | 运行过程日志 | 官方同样把已经接收的消息、请求边界和工具调用写入日志，并在后续步骤重建模型输入 |

## 练习

1. 为什么“发生过什么”和“下一次应给模型看什么”不应共用一份可变消息列表？请分别从审计、界面展示、上下文成本和崩溃恢复的角度分析。
2. 为一个代码助手设计事件日志。除用户、模型和工具消息外，再选择两类只用于检查过程或界面展示的事件，并说明它们为什么不应进入模型消息。
3. 只追加日志便于回放，却也会长期保存错误或敏感信息。面对合规删除、数据更正和调试可追溯性之间的冲突，你会怎样设计更高一层的删除或遮蔽机制？
4. 扩展本章 Session，使它能够记录并重放一种新的过程事件，同时为一个消费者提供派生视图。要求原始事件保持不可变，并用同一份日志验证重放前后的派生结果一致。
