# 10｜文件系统

> 预计时间：70 分钟 ｜ 前置：完成第 09 章 ｜ 本章调用真实 DeepSeek 模型

前九章已经建立了模型调用、工具循环、会话和上下文控制，但示例工具大多只在内存中返回结果。真正处理开发任务时，智能体还要读取和修改文件。和第 02 章的 calculator 这种纯函数工具相比，文件工具会改动外部环境，因此需要额外处理两个问题：

1. 边界。智能体能写哪些文件？代码缺陷或恶意提示会不会让它修改工作区之外的内容？
2. 并发。智能体读取文件后、写回之前，用户自己修改了同一个文件，怎样避免覆盖用户的新内容？

本章把文件操作分成三部分：五个基础文件工具、一条限制写入范围的路径围栏，以及一个防止覆盖新修改的读后写检查器。它们在官方源码中属于三个独立模块，章末会给出对应位置。

## 学习目标

完成本章后，你将能够：

- 说明 `read-only`、`workspace-write` 与 `danger-full-access` 三种模式的边界；
- 在写文件前规范化路径，并拒绝工作区之外的目标；
- 用读后写检查发现未读取文件和过期版本；
- 实现 read、write、edit、grep 与 glob 这组基础文件工具；
- 把文件函数注册为模型工具，让模型在同一条流程中读取、修改并复查文件。

## 10.1 两道保护：限制范围，检查变化

第一道保护是三种文件访问模式：

| 模式 | 能写哪里 |
|------|----------|
| `read-only` | 哪都不能写，读永远放行 |
| `workspace-write` | 工作区根 + 平台临时目录 |
| `danger-full-access` | 任何地方，显式无约束模式 |

这里有两个要点。第一，策略主要约束写操作；读取不会修改文件系统，写入则可能覆盖或删除数据。第二，workspace-write 的可写集合包含工作区和临时目录，因为程序经常需要在 `/tmp` 等位置保存中间产物。

第二道保护是读后写检查。程序要求智能体在修改已有文件前先读取它，并在读取时记录文件状态；真正写入前，再确认文件没有被其他程序修改。这种“先记录旧值，写入前再比较”的方法称为 CAS（compare-and-swap）。没有读过就写时返回 `FS_NOT_OBSERVED`；读过以后文件又发生变化时返回 `FS_STALE_VERSION`，要求重新读取。这与人工修改代码的习惯一致：先查看当前内容，发现别人改过后再重新确认。

## 10.2 沙箱围栏：fence_write

围栏是本章的安全核心，代码不长，每一行都有讲究：

```python
@dataclass(frozen=True)
class SandboxPolicy:
    mode: str = READ_ONLY
    workspace_root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        if self.mode not in WIDER_MODES:
            raise ValueError(f"未知 sandbox mode: {self.mode}")

    def writable_roots(self) -> list[Path]:
        return [
            self.workspace_root,
            Path(tempfile.gettempdir()),
            Path("/tmp"),
        ]

    def fence_write(self, target: Path) -> Path:
        if self.mode == DANGER_FULL_ACCESS:
            return target
        if self.mode == READ_ONLY:
            raise SandboxDeniedError(str(target), self.mode)
        resolved = target.resolve()
        for root in self.writable_roots():
            root_resolved = root.resolve()
            try:
                resolved.relative_to(root_resolved)
                return resolved
            except ValueError:
                continue
        raise SandboxDeniedError(str(target), self.mode)
```

三个关键点：

1. `target.resolve()` 先得到规范化后的真实路径。`workspace/../etc/passwd` 在原始字符串中带有工作区前缀，但规范化后已经位于工作区之外；指向工作区外的符号链接也会被解析到真实目标。官方同样要求在委托前再次规范化路径，以发现工具解析后发生变化的祖先符号链接。
2. `relative_to` 判断包含关系：解析后的目标必须在某个可写根之下。抛 ValueError 表示不在其下，换下一个根；全部试完仍不在，拒绝。
3. 拒绝结果要明确并带有结构：`SandboxDeniedError` 会记录当前模式，并生成模型能够识别的标记 `[sandbox: file access denied under <mode> mode]`。模型由此可以判断失败来自权限限制，而不是根据零散的错误输出猜测原因。

这里需要明确能力边界：路径围栏是一项约束，不是内核级安全隔离。它能够阻止模型误写工作区之外的路径，但不能抵御恶意代码主动绕过。第 11 章会进一步说明 shell 命令所需的内核级隔离。

## 10.3 观察器：读后写的两道门

```python
@dataclass
class ObservationTracker:
    _observed: dict[str, int] = field(default_factory=dict)

    def record_read(self, path: Path) -> None:
        key = str(path.resolve())
        try:
            self._observed[key] = path.stat().st_mtime_ns
        except FileNotFoundError:
            self._observed.pop(key, None)

    def check_write(self, path: Path) -> None:
        key = str(path.resolve())
        if key not in self._observed:
            raise PermissionError(f"[FS_NOT_OBSERVED] 修改 {path} 之前必须先 read 它")
        current = path.stat().st_mtime_ns
        if current != self._observed[key]:
            raise PermissionError(
                f"[FS_STALE_VERSION] {path} 自上次读取后被外部修改"
                f"（mtime 变化），请重新 read 后再写"
            )
```

- 观察记录使用 mtime 快照：读文件时记下修改时间，写入前再次比较。mtime 的实际时间分辨率取决于文件系统；如果两次修改间隔过短，记录值可能不变，因此 demo 在模拟外部修改前短暂等待。
- 键用 `resolve()` 规范化。macOS 上 `/var` 是指向 `/private/var` 的符号链接，读时记的键和写时查的键若不统一，会出现明明读过却报没读的假阴性。
- 两道检查主要防止无意覆盖。读取时保存的状态是比较基准，写入前文件没有变化才放行；发生变化就要求重新读取。成功写入后，观察器会记录新版本，因此同一智能体后续写入仍有新的比较基准。这是并发编辑的基本保护，不等同于文件锁。

## 10.4 五个文件工具

工具层把围栏和观察器串进每个操作。`read_file` 带行号输出并记录观察：

```python
def read_file(path: Path, tracker: ObservationTracker) -> str:
    text = path.read_text(encoding="utf-8")
    tracker.record_read(path)
    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1:>4}: {line}" for i, line in enumerate(lines))
    return f"{numbered}\n\n(End of file - total {len(lines)} lines)"
```

`edit_file` 是 str-replace 局部替换，处理一个经典问题：old_string 匹配多处时的歧义：

```python
def edit_file(path, old_string, new_string, policy, tracker, replace_all=False):
    target = policy.fence_write(path)
    if not target.exists():
        raise FileNotFoundError(f"[FS_NOT_FOUND] {target} 不存在")
    tracker.check_write(target)
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ValueError(f"[FS_EDIT_NOT_FOUND] old_string 未在 {target} 中找到")
    if count > 1 and not replace_all:
        raise ValueError(
            f"[FS_AMBIGUOUS_EDIT] old_string 在 {target} 中匹配了 {count} 处；"
            "请提供更具体的 old_string，或设置 replace_all=True"
        )
    updated = text.replace(old_string, new_string)
    target.write_text(updated, encoding="utf-8")
    return f"updated {target} ({count} 处替换)"
```

歧义报错是教学重点。模型经常写出太短的 old_string，两个字在文件里出现两次，这时拒绝并说明原因，比悄悄改第一处好得多，模型下一轮会给出更具体的匹配串。错误信息本身就是给模型看的指导。

`grep` 与 `glob` 是搜索工具，正则搜内容、模式找文件，完整实现见源码。

## 10.5 让模型实际使用文件工具

`fs_tools.py` 负责文件行为，`demo.py` 再把五个函数包装成第 02 章的 `Tool`。每个包装函数都会补上工作区路径、`SandboxPolicy` 和共享的 `ObservationTracker`：

```python
Tool(
    "edit",
    "替换文件中的一段原文；文件必须先读取。",
    {...},
    lambda args: edit_file(
        _path(workspace, args),
        str(args["old_string"]),
        str(args["new_string"]),
        policy,
        tracker,
        bool(args.get("replace_all", False)),
    ),
)
```

系统提示词告诉模型工作区位置，以及“修改已有文件前必须先读取，修改后再次读取确认”。这不是安全边界本身：真正的约束仍由工具中的围栏和观察器执行。即使模型跳过读取或给出越界路径，工具也会拒绝操作，并把错误结果送回下一次模型请求。

`agent.py` 保留一个最小的真实模型循环。它发送工具说明，接收模型给出的 `tool_calls`，执行对应函数，再把每项结果作为 `role="tool"` 消息送回模型。这样，本章不是单独调用几个文件函数，而是让模型根据每一步真实结果决定后续操作。

## 10.6 运行完整示例

```bash
uv run python chapters/10-filesystem/src/demo.py
```

下面是一次真实运行的主要输出，模型回答中间的分步说明已省略。临时路径和最终表述可能变化：

```
=== 模型发起的文件操作 ===
read({'path': 'todo.txt'})
   1: 学习 sandbox
   2: 学习 subagent

(End of file - total 2 lines)
edit({'path': 'todo.txt', 'old_string': '学习 sandbox', 'new_string': '复习 sandbox'})
updated …/workspace/todo.txt (1 处替换)
read({'path': 'todo.txt'})
   1: 复习 sandbox
   2: 学习 subagent

(End of file - total 2 lines)

模型最终回答: 任务完成。已读取 todo.txt，将“学习 sandbox”改为“复习 sandbox”，并重新读取确认。

磁盘最终内容:
复习 sandbox
学习 subagent
```

这次调用顺序不是 Python 代码预先写死的。模型先选择 `read`，获得带行号的真实内容后再选择 `edit`，最后再次 `read` 验证结果。三次调用共享同一个观察器，因此编辑时能够确认文件已经读取且没有被外部修改。围栏、歧义编辑、过期版本和越界拒绝仍由相同工具代码处理；可以在练习中修改任务提示，观察模型收到错误结果后如何修正调用。

## 本章小结

- `SandboxPolicy.fence_write`：三模式、可写根集合、resolve 规范化、结构化拒绝
- `ObservationTracker`：读取时记录修改时间，写入前检查文件是否变化
- 五个工具：read 带行号与页脚并记录观察，write 全量写入，edit 唯一匹配 str-replace，grep 搜内容，glob 找文件
- 真实模型流程：工具调用、文件结果回灌和修改后复查位于同一个循环中
- 升级审批：严格更宽表

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/fs/fs-sandbox/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/fs/fs-sandbox/README.zh.md) | `SandboxPolicy` | 对齐三种模式、可写根集合、“约束而非安全边界”的定位与结构化 FsError |
| [`packages/fs/fs-observation-policy/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/fs/fs-observation-policy/README.zh.md) | `ObservationTracker` | 官方同样要求写入基于最近一次读取的版本，并通过文件事件记录写入意图和观察结果；教学版只比较修改时间 `mtime` |
| [`packages/fs/tool-fs/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/fs/tool-fs/README.zh.md) | 五个工具函数 | 官方工具层还会把 `FsError` 转换成模型可以理解的沙箱错误标记 |

## 练习

1. 为一个能修改代码库的智能体分析安全风险，至少考虑路径穿越、符号链接、覆盖用户新改动和过宽写权限。哪些风险由路径围栏解决，哪些还需要观察策略或人工审批？
2. 文档问答、代码重构和系统运维三类智能体应分别使用哪种文件模式？说明为什么“能读但不能写”并不等于不存在隐私或数据泄露风险。
3. 读后写检查可能因为外部格式化、时间戳精度或跨进程修改产生误报和漏报。比较 mtime、内容哈希和文件事件三种观察方式，并选择一种适合本教学项目的方案。
4. 设计并实现一个新的安全文件操作，例如应用补丁或批量重命名。它必须复用工作区围栏和读后写检查，返回能指导智能体修正的结构化错误，并验证越界路径与过期版本不会被写入。
