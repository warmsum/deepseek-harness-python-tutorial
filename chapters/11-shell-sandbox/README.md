# 11｜命令执行与审批

> 预计时间：60 分钟 ｜ 前置：完成第 10 章 ｜ 本章调用真实 DeepSeek 模型

文件工具只能完成预先定义的读写操作，开发任务还需要运行测试、构建和 Git 命令。Shell 命令的能力范围更大，一条命令既可能删除大量文件，也可能读取不应暴露的数据。因此，执行命令前需要经过两层决策：

1. 运行模式：不同模式允许执行哪些类型的命令；
2. 人工审批：某条命令即使符合当前模式，也可能需要用户明确同意。

本章实现命令执行器和审批流程。教学版只能在命令执行前决定允许或拒绝，不能在系统层限制命令运行后的实际影响。官方的 `bash-sandbox` 还会使用 macOS Seatbelt 或 Linux Landlock，在操作系统层限制文件访问；即便如此，它也不是能够隔离所有风险的通用安全环境。

## 学习目标

完成本章后，你将能够：

- 使用 `subprocess` 捕获命令输出、退出码与超时状态；
- 按模式、审批策略和一次性授权决定是否执行命令；
- 解释“一次性授权”和“检查失败时默认拒绝”的安全含义；
- 区分教学版的文本决策层与官方内核级隔离；
- 把模型提出的命令接入审批、执行和结果回灌流程。

## 11.1 原理：执行、超时与审批的三个问题

问题一，如何处理输出。命令的标准输出 `stdout` 和错误输出 `stderr` 既不能不受控制地占满终端，又需要交给模型读取。`capture_output=True` 会把两路输出收集到 `CommandResult` 中，调用方可以按字段处理。

问题二，命令长时间不结束怎么办。`sleep 9999`、等待输入的交互程序或死循环都可能一直占用进程。`timeout` 参数为命令设置最长运行时间，超时后终止子进程，并在结果中把 `timed_out` 标记为真。

问题三，谁有权决定是否执行。运行模式先判断命令类别，例如 `read-only` 模式只允许只读命令；需要进一步确认时，再把具体命令交给用户审批。审批有四种结果：一次性允许 `allowed-once`、拒绝 `rejected`、取消 `cancelled` 和审批通道不可用 `unavailable`。只有一次性允许会放行当前命令，其余结果都拒绝执行。

## 11.2 run_command：捕获、超时、结构化结果

```python
@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_command(
    command: str,
    cwd: str,
    timeout_seconds: float = 30.0,
    *,
    use_shell: bool = True,
) -> CommandResult:
    try:
        argv: str | list[str] = command if use_shell else shlex.split(command)
        completed = subprocess.run(
            argv,
            shell=use_shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            exit_code=-1,
            stdout=(error.stdout or b"").decode()
            if isinstance(error.stdout, bytes)
            else "",
            stderr=f"命令超时（>{timeout_seconds}s），已强制终止",
            timed_out=True,
        )
```

几个关键参数的用意：

- `use_shell=True` 按 shell 语法解析，管道、重定向、`&&` 都能用；`False` 时先用 `shlex.split` 拆成参数列表，完全绕过 shell 语法。
- `cwd` 决定命令从哪个目录开始执行。教学版直接信任调用方传入的路径，没有验证它是否位于工作区；示例主动传入临时工作区。真实智能体必须由更外层统一限制 `cwd`，不能让不可信输入随意指定执行目录。
- `timeout` 到期后会直接终止子进程。命令超时不会继续向外抛出异常，而是转换成一条结构化结果：`exit_code` 为 −1，`timed_out` 为真。模型可以根据这个结果决定下一步，而整个智能体进程不会因此退出。

为什么还要提供 `use_shell=False`？因为文本白名单只能识别第一条命令。假如把 `ls; rm file` 交给 shell，第一词看起来是只读的 `ls`，后半句却会删除文件。因此，本章对 read-only 白名单命令关闭 shell 解析；只有经过更宽模式和审批的命令才使用完整 shell 语法。

## 11.3 决策链：票据、模式门、审批

`ShellPolicy.decide` 按顺序检查三类规则，任一规则得出结论后立即返回：

```python
    def decide(self, command: str) -> tuple[bool, str]:
        # 1) 一次性票据：绕过一切，用后即焚
        if self._granted_once == command:
            self._granted_once = None
            return True, "allowed-once（一次性授权）"

        try:
            words = shlex.split(command)
        except ValueError as error:
            return False, f"[sandbox] 无法解析命令: {error}"
        first_word = words[0] if words else ""

        # 2) 模式门
        if self.mode == "read-only":
            if first_word in READ_ONLY_COMMANDS:
                return True, "read-only 白名单放行"
            return False, f"[sandbox] read-only 模式拒绝写类/未知命令: {command}"

        # 3) 审批
        if self.approval_policy == POLICY_NEVER:
            return False, "[approval] 审批策略为 never，直接拒绝"
        outcome = self.approver(command)
        if outcome == APPROVAL_ALLOWED_ONCE:
            return True, "approved（本轮放行）"
        if outcome == APPROVAL_CANCELLED:
            return False, "[approval] 审批被取消"
        if outcome == APPROVAL_UNAVAILABLE:
            return False, "[approval] 无可用审批通道（fail closed）"
        return False, "[approval] 审批被拒绝"
```

逐条看三个决策。

一次性授权最先检查。`grant_once(command)` 只对完全相同的一条命令生效，使用后立即失效。例如，模型在只读模式下请求执行写命令，用户可以只批准当前动作；后续写命令仍需重新判断。这种结果在代码中记为 `allowed-once`。

模式检查发生在审批之前。read-only 模式下，白名单只做精确命令名匹配：`grep` 可以，`grep-and-delete` 不可以。放行后还会以 `shell=False` 执行，分号、管道和重定向只会成为普通参数，不能偷偷追加第二条命令。这里的白名单仍然只是教学近似，真实内核沙箱按系统调用产生的实际影响拦截，不依赖命令文本。

审批采用“失败时默认拒绝”的原则，也称为 fail closed。`never` 策略直接拒绝；`ask` 策略调用审批函数，只有 `allowed-once` 会放行。审批通道不可用时，智能体停止当前动作，不会绕过检查继续执行。

## 11.4 从模型请求到审批和执行

`demo.py` 把 `ShellPolicy.execute()` 包装成一个 `shell` 模型工具。模型只负责提出命令，程序仍掌握是否执行的决定权：

```python
def execute(arguments: dict[str, Any]) -> str:
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("command 必须是非空字符串")
    result = policy.execute(command, cwd=str(workspace), timeout_seconds=5)
    return (
        f"exit_code={result.exit_code}\n"
        f"stdout={result.stdout.strip() or '(空)'}\n"
        f"stderr={result.stderr.strip() or '(空)'}"
    )
```

本章任务要求模型依次执行两条精确命令：先用 `echo approved > approved.txt` 创建文件，再用 `cat approved.txt` 读取内容。审批回调只允许这两条完整字符串；模型如果改写命令、增加管道或提出其他操作，就会得到 `rejected`，命令不会执行。

这条流程包含三个彼此独立的决定：模型选择调用什么工具，审批器决定某条命令能否运行，执行器负责超时、退出码和输出。工具结果随后送回模型，最终回答必须依据真实的 `stdout` 和 `stderr`，而不是预测命令会产生什么结果。

## 11.5 运行完整示例

```bash
uv run python chapters/11-shell-sandbox/src/demo.py
```

下面是一次真实运行的主要输出，模型最终回答的分步说明已省略。回答措辞可能变化，审批和命令结果由程序产生：

```
审批请求: echo approved > approved.txt -> allowed-once
审批请求: cat approved.txt -> allowed-once

=== 模型发起的命令 ===
shell({'command': 'echo approved > approved.txt'})
exit_code=0
stdout=(空)
stderr=(空)
shell({'command': 'cat approved.txt'})
exit_code=0
stdout=approved
stderr=(空)

模型最终回答: 两个命令均执行成功，approved.txt 中的内容为 approved。
文件是否由获批命令创建: True
```

输出先记录审批结果，再显示模型实际提出的命令和子进程结果。第一条命令没有标准输出，但退出码为 0；第二条命令读取到 `approved`。文件最终存在，说明写入来自获批命令。若审批回调返回拒绝、取消或不可用，工具仍会把结构化失败结果交还模型，但不会启动子进程。

## 本章小结

- `run_command`：subprocess、cwd、timeout 与可选 shell 解析，超时转结构化结果
- `ShellPolicy.decide`：票据、模式门、审批的三段决策链
- read-only 白名单使用精确命令名和 `shell=False`，拒绝命令拼接绕过
- 审批结果：只有一次性允许会执行命令，审批失败或不可用时默认拒绝
- 真实模型流程：模型提出命令，审批器决定是否放行，执行结果再返回模型
- 实现边界：教学版不校验 cwd，也不包含内核隔离；官方 bash-sandbox 使用 seatbelt 与 landlock

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/shell/bash-sandbox/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/shell/bash-sandbox/README.zh.md) | `ShellPolicy` | 官方支持 danger-full-access，并明确沙箱只限制文件影响；教学版不实现内核隔离 |
| [`packages/interaction/user-approval/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/interaction/user-approval/README.zh.md) | 审批链 | 官方定义四种审批结果以及询问和从不询问两种策略；教学版同样在审批失败时默认拒绝 |
| [`packages/guard/timeout-policy/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/guard/timeout-policy/README.zh.md) | `timeout` | 官方通过 `exec.signal` 通知运行中的命令停止，教学版由 Python 子进程接口在超时后直接终止 |

## 练习

1. 文本规则能够决定“是否允许执行”，内核沙箱能够限制“执行后真正影响什么”。请比较二者能够防御的风险，并解释为什么审批通过也不能代替系统级隔离。
2. 为一个日常编码智能体制定命令策略：哪些命令可以自动执行，哪些必须逐次审批，哪些应始终拒绝？请考虑读取、测试、网络访问、Git 写操作和删除文件等类别。
3. 一次性授权降低了长期授权风险，却可能造成审批疲劳；会话级授权更方便，却可能被后续命令滥用。设计一种折中方案，并说明授权应绑定哪些信息。
4. 为命令执行器增加“预览决策”能力，在不运行命令的情况下返回模式门、审批和票据检查结果。先预览安全、需审批和拒绝三类命令，再真实执行一个会超时的安全命令，说明预览与执行如何共享策略、执行器又负责哪些运行期结果。
