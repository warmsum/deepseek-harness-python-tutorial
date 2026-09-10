"""第 11 章：命令执行、权限模式与审批。

对应官方 packages/shell（执行接口）和 packages/interaction/user-approval。
教学版实现三件事：
1. run_command —— subprocess 执行 + 超时 + 输出捕获；
2. ApprovalPolicy —— 审批策略：ask（询问）/ never（直接拒绝）；
3. grant_once —— 一次性授权：allowed-once 只放行所请求的那一个动作。

官方 bash-sandbox 使用内核机制（Seatbelt/Landlock）限制文件影响。
教学版只实现模式判断和审批决策，不提供内核级隔离。
"""

from __future__ import annotations

import math
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

# 审批结果四值（对应官方 dsh-user-approval 的四结果枚举）：
# allowed-once 是唯一放行值——一次性授权，只作用于所请求的那一个动作
APPROVAL_ALLOWED_ONCE = "allowed-once"
APPROVAL_REJECTED = "rejected"
APPROVAL_CANCELLED = "cancelled"
APPROVAL_UNAVAILABLE = "unavailable"

# 审批策略：ask = 走审批通道；never = 直接拒绝（官方 ApprovalPolicy）
POLICY_ASK = "ask"
POLICY_NEVER = "never"
SHELL_MODES = frozenset({"read-only", "workspace-write", "danger-full-access"})

# 只读命令白名单：read-only 模式下仅这些前缀的命令放行。
# （教学简化——真实内核沙箱按系统调用拦截，不看命令文本。）
READ_ONLY_COMMANDS = {"ls", "cat", "head", "tail", "grep", "pwd", "wc"}


@dataclass(frozen=True)
class CommandResult:
    """一次命令执行的完整结果。"""

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
    """执行一条 shell 命令，捕获输出，强制超时。

    关键参数：
    - capture_output：stdout/stderr 不刷屏，收进结果里；
    - timeout：命令超过时限（如 sleep 9999）时终止子进程；
    - shell=True：按 shell 语法解析（管道、重定向都可用）。
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 必须是非空字符串")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds 必须是正有限数")
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
            stdout=(error.stdout or b"").decode() if isinstance(error.stdout, bytes) else "",
            stderr=f"命令超时（>{timeout_seconds}s），已强制终止",
            timed_out=True,
        )


class ShellPolicy:
    """命令执行的决策层：运行模式与审批。

    决策顺序（对应官方 sandbox 决策的简化版）：
    1. allowed-once：一次性授权，用一次即失效；
    2. 运行模式：read-only 只放行白名单只读命令；更宽模式进入审批；
    3. 审批：policy=never 直接拒绝；policy=ask 调用审批回调。
    """

    def __init__(
        self,
        mode: str = "read-only",
        approval_policy: str = POLICY_ASK,
        approver: Callable[[str], str] | None = None,
    ) -> None:
        if mode not in SHELL_MODES:
            raise ValueError(f"未知 shell mode: {mode}")
        if approval_policy not in {POLICY_ASK, POLICY_NEVER}:
            raise ValueError(f"未知 approval policy: {approval_policy}")
        self.mode = mode
        self.approval_policy = approval_policy
        # 审批回调：返回 APPROVAL_ALLOWED_ONCE / APPROVAL_REJECTED / ...
        self.approver = approver or (lambda command: APPROVAL_REJECTED)
        # 一次性授权：仅匹配一条完整命令，匹配后立即失效。
        self._granted_once: str | None = None

    def grant_once(self, command: str) -> None:
        """签发一次性授权（对应官方 allowed-once：只作用于所请求的那一个动作）。"""
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command 必须是非空字符串")
        self._granted_once = command

    def decide(self, command: str) -> tuple[bool, str]:
        """决定一条命令能否执行。返回 (放行?, 理由)。

        决策顺序（每一步命中即返回）：
        1. 一次性授权（allowed-once）：匹配完整命令后直接放行并失效；
        2. 运行模式：read-only 白名单命令无需审批，其他命令直接拒绝；
        3. 审批：never 直接拒绝；ask 调用审批回调（fail closed）。
        """
        if not isinstance(command, str) or not command.strip():
            return False, "[sandbox] command 必须是非空字符串"

        # 1) 一次性授权只匹配完整命令，匹配后立即失效。
        if self._granted_once == command:
            self._granted_once = None
            return True, "allowed-once（一次性授权）"

        try:
            words = shlex.split(command)
        except ValueError as error:
            return False, f"[sandbox] 无法解析命令: {error}"
        first_word = words[0] if words else ""

        # 2) 运行模式
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

    def execute(self, command: str, cwd: str, timeout_seconds: float = 30.0) -> CommandResult:
        """先完成策略判断，再执行获准的命令。"""
        allowed, reason = self.decide(command)
        if not allowed:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"{reason}\n（命令未执行）",
            )
        # 教学版没有内核沙箱：只读白名单必须绕过 shell 解析，防止
        # `ls; rm file` 这类“首命令看似只读、后续命令产生写效应”的绕过。
        direct_read_only = reason == "read-only 白名单放行"
        return run_command(
            command,
            cwd,
            timeout_seconds,
            use_shell=not direct_read_only,
        )
