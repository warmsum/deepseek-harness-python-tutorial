# 08｜会话持久化

> 预计时间：50 分钟 ｜ 前置：完成第 07 章 ｜ 本章调用真实 DeepSeek 模型

第 05–07 章已经用事件日志记录模型与工具的运行过程，并让智能体能够连续处理多轮任务。不过，这些日志仍只保存在内存中，程序退出后历史就会丢失。要支持崩溃恢复、稍后继续或迁移会话，事件日志必须写入磁盘。本章实现一个文件存储，并说明第 05 章的 `subscribe` 接口如何用于增量持久化。

本章使用 JSONL 保存日志。JSONL 文件的第一行记录格式和版本，之后每行保存一个 JSON 对象。它既方便直接打开检查，也允许程序只在文件末尾追加新事件；即使最后一行没有写完，前面的完整记录通常仍可读取。

只在任务结束时保存还不够。程序可能在调用模型或执行写文件等操作后突然退出，导致动作已经发生，日志却没有记录。为此，程序会在重要操作前先保存已有事件。这个保存节点称为 checkpoint，本章称为“检查点”。

## 学习目标

完成本章后，你将能够：

- 把 `SessionEvent` 按 JSONL 格式写入文件并重新加载；
- 首次创建时原子发布，后续只追加尚未写入磁盘的事件；
- 校验文件头、格式版本和事件编号；
- 识别没有写完的最后一行，并为未开始或结果未知的工具调用、未闭合步骤和轮次补充事件；
- 在模型请求、顶层工具执行、下一步骤和重试等待之前建立检查点，保存失败时停止后续操作。

## 8.1 三个必须回答的问题

把日志存进文件需要处理三个问题：

1. 第一次创建时怎样避免留下只写了一半的文件？程序先写唯一命名的临时文件，用 `fsync` 请求操作系统把内容写入磁盘，再通过 `os.link` 无覆盖发布为正式文件。
2. 后续事件怎样保存？正式文件已经存在后，不应每次重写全部内容。`save()` 先确认磁盘日志确实是当前内存日志的开头部分，再只追加新增事件并调用 `fsync`。
3. 哪些损坏可以自动修复？只有文件末尾没有换行的残缺片段，能够明确判断为一次没有完成的写入，可以安全截断。完整行或文件中间的内容解析失败时必须停止加载，不能借“崩溃恢复”丢弃后续数据。

## 8.2 写入磁盘：首次原子发布，随后仅追加

```python
class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, session: Session) -> None:
        if not self.path.exists():
            # header + 当前前缀写入临时文件，fsync 后原子发布
            ...
            os.link(tmp_path, self.path)  # 目标已存在时拒绝覆盖
            return

        persisted, torn_offset = self._read_records()
        if torn_offset is not None:
            raise ValueError("请先 load() 修复残缺尾部")
        current = session.snapshot_events()
        if tuple(persisted) != current[:len(persisted)]:
            raise ValueError("磁盘日志不是当前日志的前缀")
        # 只追加 pending，并 flush + fsync
```

逐段看：

- 第一行文件头记录 `format` 和 `version`。本章的自有格式当前为版本 2，用来标识 `seq`/`time` 信封和独立表层元数据；它不是官方 Session V3 文件格式。加载时会据此识别文件类型并拒绝不支持的版本。
- 每行保存一条事件的 `seq`、`type`、`time`、`data`，以及消息事件的表层操作和来源序号。`ensure_ascii=False` 让中文保持原样，文件更容易人工检查。
- 第一次保存使用唯一临时文件、`fsync` 和无覆盖硬链接；后续保存验证已有内容后只追加新事件。已经公开的历史不会被整份重写。

第一次发布不会覆盖另一个写入方已经创建的同名文件。后续追加仍假设只有一个进程持有写入权；教学版没有实现官方 `SessionHandle` 的进程内认领和跨进程文件锁。

## 8.3 读回：校验与崩溃修复

```python
    def load(self) -> Session:
        if not self.path.exists():
            raise FileNotFoundError(f"会话文件不存在: {self.path}")

        events, torn_offset = self._read_records()
        if torn_offset is not None:
            with self.path.open("r+b") as file:
                file.truncate(torn_offset)
        _append_recovery_closers(events)
        return Session.from_log(events)
```

加载时依次执行三项检查和恢复操作：

1. 文件不存在时直接抛出 `FileNotFoundError`，避免把缺失的会话误认为空会话。
2. 文件头校验失败时立即报错。读取了其他格式或未来版本的文件时，程序不能猜测其含义。
3. `_read_records` 只把末尾没有换行的片段视为未完成写入；其他损坏一律报错。截断残缺片段后，加载器会检查哪些工具请求、工具调用、步骤和轮次没有结束，并依次补充 `tool/result`、`step/end` 和对应轮次的 `turn/end`。

Assistant 已请求工具、但还没有 `tool/call` 时，恢复器写入 `TOOL_NOT_STARTED`，模型可以按需重试。已经写入 `tool/call` 却没有结果时，恢复器无法判断工具是否已经产生外部影响，因此写入 `TOOL_OUTCOME_UNKNOWN`。只读或幂等操作可以再次尝试；可能修改外部状态的操作应先检查实际结果或询问用户。即使文件最后一行完整，也要检查这些开放状态，因为程序可能恰好在写完某条事件后退出。

最后，`Session.from_log` 继续执行第 05 章建立的 `seq` 连续性和表层替换校验。

## 8.4 检查点：先保存记录，再执行重要操作

`save()` 解决“怎样保存”，`CheckpointPolicy` 解决“什么时候必须保存”。本章选择四个检查点：调用模型前、执行顶层工具前、开始下一步骤前，以及进入重试等待前。

```python
@dataclass(frozen=True)
class CheckpointPolicy:
    flush: Callable[[Session], None]

    def before_model(self, session: Session) -> None:
        self.flush(session)

    def before_tool(self, session: Session, *, nested: bool = False) -> None:
        if not nested:
            self.flush(session)

    def before_step(self, session: Session) -> None:
        self.flush(session)

    def before_retry(self, session: Session) -> None:
        self.flush(session)
```

- `before_model`：先保存组装模型请求所依据的事件，再调用模型服务；
- `before_tool`：先保存 `tool/call`，再执行顶层工具；工具内部继续调用其他工具时复用外层检查点，避免重复保存；
- `before_step`：先保存上一条模型回复和有序的工具结果，再开始下一步骤；
- `before_retry`：先保存 `llm/retry` 事件，再进入等待。

四个方法都会把保存错误交给调用方。保存失败时，程序停止调用模型、执行工具或进入重试等待。这种“检查失败就停止”的策略称为 fail closed。检查点能保证操作意图先于动作写入磁盘，便于恢复时判断程序运行到了哪里；但它不能保证外部操作只发生一次，因为程序仍可能在远端写入已经成功、结果事件尚未保存时退出。

## 8.5 把检查点接入模型与工具循环

存储能力只有进入实际运行流程，才能回答“应该在什么时候保存”。`agent.py` 保留第 02、05 章已经讲过的最小模型与工具循环，并在模型请求和工具执行之前调用检查点：

```python
session.record_system_prompt(system_prompt, turn=1, step=step)
session.append("request/header", {"header": {"config": ..., "tools": ...}})
checkpoint.before_model(session)
reply = client.chat(session.derive_messages(), [tool])
session.append("assistant/message", {...})

for call in reply.tool_calls:
    session.append("tool/call", {...})
    checkpoint.before_tool(session)
    result = tool.execute(arguments)
    session.append("tool/result", {...})
```

这段顺序保证模型请求前已经保存请求依据，工具执行前已经保存调用意图。模型回复和工具结果进入事件日志后，下一步骤开始前还会再次保存。`client.py` 是前面模型通信代码的最小副本，使本章能够单独运行；本章新增的重点仍是 `JsonlStore` 和 `CheckpointPolicy`。

示例让模型计算 `(18 + 6) / 3`。模型必须调用 `calculator`，因此一轮真实任务会同时经过模型边界、工具边界、事件追加和 JSONL 写入。任务完成后，程序从同一文件恢复会话，再在文件末尾追加半条 JSON，验证加载器只移除这段没有写完的内容。

## 8.6 运行完整示例

```bash
uv run python chapters/08-persistence/src/demo.py
```

下面是一次真实运行的主要输出，模型回答中间的解释已省略。回答措辞可能变化，事件顺序由程序控制：

```
=== 真实模型任务：计算并持续保存会话 ===
模型最终回答: 我使用了 calculator 工具来计算表达式 (18 + 6) / 3。
最终结果是 8。

=== 磁盘中的真实事件顺序 ===
#0  turn/start
#1  step/start
#2  system/message
#3  user/message
#4  request/header
#5  assistant/message
#6  tool/call
#7  tool/result
#8  step/end
#9  step/start
#10 assistant/message
#11 step/end
#12 turn/end
恢复后消息数量: 5

=== 使用同一日志模拟未完成写入 ===
残缺尾行已移除，最后事件仍是: turn/end
```

第一次模型回复提出工具调用，第二次模型回复给出最终答案，因此日志中出现两个步骤。`derive_messages()` 恢复出系统消息、用户消息、带工具调用的助手消息、工具结果和最终助手消息，共 5 条。最后追加的残缺 JSON 没有换行，加载器能够确认它是未完成写入并截断；前面的 `turn/end` 保持不变。如果完整日志停在开放的工具调用、步骤或轮次中，加载器会在内存会话中补充异常收尾，下一次 `save()` 再把这些事件写回磁盘。

## 本章小结

- `JsonlStore.save()`：首次原子发布、前缀校验、后续仅追加
- `JsonlStore.load()`：严格校验文件头和事件，只截断末尾未写完的片段，并为未结束的工具、步骤和轮次补充收尾
- `CheckpointPolicy`：在模型请求、顶层工具、下一步骤和重试等待前保存，保存失败就停止后续操作
- 真实任务路径：模型调用计算器的同时持续写入事件，并从同一份 JSONL 恢复消息
- 恢复过程：检查文件、修复可以确认的残缺内容，再由 `from_log` 校验事件连续性

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/session/session-persistence-jsonl/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/session/session-persistence-jsonl/README.zh.md) | `JsonlStore` | 与官方一样只向日志末尾追加并在恢复前校验；教学版使用自有格式版本 2，不实现官方 V0→V3 迁移链、Zstandard 分帧和跨进程写锁 |
| 同上 | `save` | 两者都在首次写入时原子发布文件，之后只追加；官方还支持批量追加、写入失败回滚和更多跨平台细节 |
| 同上 | （未实现） | 官方当前默认使用带校验和的 Zstandard 帧，并通过不可变后继 generation 迁移旧格式；教学版保持可直接阅读的 JSONL |
| 同上 | 崩溃修复 | 教学版只截断不完整尾部，再区分 `TOOL_NOT_STARTED` 与 `TOOL_OUTCOME_UNKNOWN` 并补齐步骤和轮次；中间损坏时拒绝加载 |
| [`packages/session/session-checkpoint-policy/README.zh.md`](https://github.com/deepseek-ai/deepseek-harness/blob/b2e3b2a0125854567a4a5fcba75782e42fe84901/packages/session/session-checkpoint-policy/README.zh.md) | `CheckpointPolicy` | 模型请求、顶层工具和步骤开始前的保存时机与官方一致；教学版没有后台批量保存控制器，因此会在等待重试前显式保存 `llm/retry` |

官方还会订阅 `session/event`，按时间窗口批量写入，并默认使用带校验信息的 zstd 压缩格式。教学版由调用方显式执行 `save()`，每次追加尚未保存的事件。

## 练习

1. JSONL、关系型数据库和每次整份覆盖的 JSON 文件都能保存会话。请从追加写、人工检查、并发、查询和损坏恢复几个方面比较它们，并说明教学版选择 JSONL 的理由。
2. 分别面对残缺尾行、中间坏行、未闭合工具调用和只有 `turn/start` 的日志，恢复器应该继续、补写收尾还是拒绝加载？为每种情况说明可以信任的证据。
3. 检查点保证“执行意图先写入磁盘”，但不能自动保证外部操作只执行一次。以发送邮件或写入远程数据库为例，说明崩溃恢复时仍可能发生什么，并提出一种补充机制。
4. 编写一个只读会话检查器，输入 JSONL 文件后报告版本、事件数量、最后完整 turn、开放状态和可恢复问题。检查过程不得修改源文件；若提供修复功能，应先明确展示将追加或截断的内容。
