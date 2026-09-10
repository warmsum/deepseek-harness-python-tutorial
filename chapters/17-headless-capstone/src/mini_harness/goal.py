"""第 13 章：Goal —— 长任务的目标状态机。

对应官方 packages/goal/goal。核心语义（packages/goal/goal）：
1. 事件溯源：目标状态以 goal/change 事件进入会话日志，日志是唯一持久权威；
2. 单一目标：最多只有一个当前目标，revision 从 1 开始；
3. 动词集合：create / edit / pause / resume / complete / block；
4. 变更经 revision 比较并设置防护（Compare-and-Swap），拒绝陈旧引用；
5. 续行启用状态不持久化：会话恢复后需要显式 resume。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .session import Session

PHASE_ACTIVE = "active"
PHASE_PAUSED = "paused"
PHASE_COMPLETE = "complete"
PHASE_BLOCKED = "blocked"
MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class Goal:
    """一个目标状态的完整快照。"""

    id: str
    revision: int
    phase: str
    objective: str
    max_rounds: int
    rounds_started: int = 0
    blocker_reason: str | None = None


@dataclass(frozen=True)
class GoalRef:
    """目标引用：变更操作的比较凭证（id + revision）。"""

    id: str
    revision: int


class GoalStore:
    """同会话目标状态：动词 + 事件溯源 + revision 守卫。"""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._current: Goal | None = None
        self._armed = False

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get(self) -> Goal | None:
        """当前目标（与内部状态脱离的冻结快照）。"""
        return self._current

    def get_ref(self) -> GoalRef:
        """当前目标的引用（变更操作的凭证）。"""
        if self._current is None:
            raise ValueError("没有当前目标")
        return GoalRef(id=self._current.id, revision=self._current.revision)

    @property
    def activation(self) -> str:
        """进程内续行状态；恢复出的 Goal 默认处于 disarmed。"""
        return "armed" if self._armed else "disarmed"

    # ------------------------------------------------------------------
    # 动词（每个动词 = 校验 → 新快照 → goal/change 事件）
    # ------------------------------------------------------------------

    def create(self, objective: str, max_rounds: int = 256) -> GoalRef:
        """创建目标。官方：最多只有一个当前目标；create 生成
        revision=1、phase=active 的目标并启用续行。"""
        objective = objective.strip()
        if not objective:
            raise ValueError("objective 必须是非空字符串")
        if (
            not isinstance(max_rounds, int)
            or isinstance(max_rounds, bool)
            or max_rounds <= 0
            or max_rounds > MAX_SAFE_INTEGER
        ):
            raise ValueError("max_rounds 必须是 JSON 安全范围内的正整数")
        if self._current is not None and self._current.phase != PHASE_COMPLETE:
            raise ValueError("已有进行中的目标：必须先 complete 或 clear")
        goal = Goal(
            id=f"goal-{self._session.seq}",
            revision=1,
            phase=PHASE_ACTIVE,
            objective=objective,
            max_rounds=max_rounds,
        )
        self._commit(goal, "create", armed=True)
        return GoalRef(id=goal.id, revision=goal.revision)

    def edit(
        self,
        ref: GoalRef,
        objective: str | None = None,
        max_rounds: int | None = None,
    ) -> GoalRef:
        """编辑目标文本。官方语义保留 phase、blocker reason 与 activation。"""
        current = self._require(ref)
        if objective is None and max_rounds is None:
            raise ValueError("edit 至少需要 objective 或 max_rounds")
        next_objective = current.objective
        if objective is not None:
            next_objective = objective.strip()
            if not next_objective:
                raise ValueError("objective 必须是非空字符串")
        next_max_rounds = current.max_rounds
        if max_rounds is not None:
            if (
                not isinstance(max_rounds, int)
                or isinstance(max_rounds, bool)
                or max_rounds <= 0
                or max_rounds > MAX_SAFE_INTEGER
            ):
                raise ValueError("max_rounds 必须是 JSON 安全范围内的正整数")
            next_max_rounds = max_rounds
        if next_max_rounds < current.rounds_started:
            raise ValueError("max_rounds 不能小于已经开始的目标轮次数")
        goal = Goal(
            id=current.id,
            revision=current.revision + 1,
            phase=current.phase,
            objective=next_objective,
            max_rounds=next_max_rounds,
            rounds_started=current.rounds_started,
            blocker_reason=current.blocker_reason,
        )
        self._commit(goal, "edit")
        return GoalRef(id=goal.id, revision=goal.revision)

    def pause(self, ref: GoalRef) -> GoalRef:
        current = self._require(ref)
        if current.phase != PHASE_ACTIVE:
            raise ValueError("只有 active 目标可以 pause")
        self._commit(
            self._with_phase(current, PHASE_PAUSED, current.blocker_reason),
            "pause",
            armed=False,
        )
        return GoalRef(id=current.id, revision=current.revision + 1)

    def resume(self, ref: GoalRef) -> GoalRef:
        """恢复。官方：只有配置的 Round 上限仍有剩余容量时，resume 才接受
        已停止 phase 或 phase=active 但已停用续行的目标；清除 blocker reason。"""
        current = self._require(ref)
        if current.phase not in {PHASE_ACTIVE, PHASE_PAUSED, PHASE_BLOCKED}:
            raise ValueError(f"{current.phase} 目标不能 resume")
        if current.phase == PHASE_ACTIVE and self._armed:
            raise ValueError("目标已经 active 且已启用续行")
        if current.rounds_started >= current.max_rounds:
            raise ValueError("目标轮次已达上限，无法 resume")
        self._commit(
            self._with_phase(current, PHASE_ACTIVE, None), "resume", armed=True
        )
        return GoalRef(id=current.id, revision=current.revision + 1)

    def complete(self, ref: GoalRef) -> GoalRef:
        current = self._require(ref)
        if current.phase == PHASE_COMPLETE:
            raise ValueError("目标已经完成")
        self._commit(
            self._with_phase(current, PHASE_COMPLETE, None),
            "complete",
            armed=False,
        )
        return GoalRef(id=current.id, revision=current.revision + 1)

    def block(self, ref: GoalRef, reason: str) -> GoalRef:
        """阻塞：记录策略代码与规范化文本说明（教学版只留文本）。
        官方语义中阻塞会停用续行，只记录一个持久 phase。"""
        current = self._require(ref)
        if current.phase != PHASE_ACTIVE:
            raise ValueError("只有 active 目标可以 block")
        reason = reason.strip()
        if not reason:
            raise ValueError("阻塞原因不能为空")
        self._commit(
            self._with_phase(current, PHASE_BLOCKED, reason),
            "block",
            armed=False,
        )
        return GoalRef(id=current.id, revision=current.revision + 1)

    def clear(self, ref: GoalRef) -> None:
        """清除当前目标，并用带 revision 的 tombstone 留下持久记录。"""
        current = self._require(ref)
        cleared = {"id": current.id, "revision": current.revision + 1}
        self._session.append(
            "goal/change",
            {"version": 1, "operation": "clear", "cleared": cleared},
        )
        self._current = None
        self._armed = False

    def admit_round(self) -> GoalRef:
        """接纳一个目标轮次（官方：只有来源为 goal 且已准入的
        user/message 事件会推进正数 Round）。
        轮次是 goal 来源消息的投影，不是 goal/change；因此它不推进 revision。"""
        if (
            self._current is None
            or self._current.phase != PHASE_ACTIVE
            or not self._armed
        ):
            raise ValueError("没有 active 且已启用续行的目标，无法接纳轮次")
        next_round = self._current.rounds_started + 1
        if next_round > self._current.max_rounds:
            raise ValueError("目标轮次已达上限")
        current = self._current
        self._session.append(
            "user/message",
            {
                "content": current.objective,
                "source": {
                    "kind": "goal",
                    "goal_id": current.id,
                    "revision": current.revision,
                    "round": next_round,
                },
            },
        )
        self._current = replace(current, rounds_started=next_round)
        return GoalRef(id=self._current.id, revision=self._current.revision)

    # ------------------------------------------------------------------
    # 内部：revision 守卫 + 事件提交 + 重放
    # ------------------------------------------------------------------

    def _require(self, ref: GoalRef) -> Goal:
        """比较并交换：只接受与当前 id 和 revision 精确匹配的引用。"""
        if (
            not isinstance(ref, GoalRef)
            or not isinstance(ref.id, str)
            or not ref.id
            or ref.id != ref.id.strip()
            or not isinstance(ref.revision, int)
            or isinstance(ref.revision, bool)
            or ref.revision < 1
            or ref.revision > MAX_SAFE_INTEGER
        ):
            raise ValueError(
                "GoalRef 必须包含规范的非空 id 和 JSON 安全范围内的正整数 revision"
            )
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

    def _with_phase(self, goal: Goal, phase: str, blocker: str | None) -> Goal:
        return Goal(
            id=goal.id,
            revision=goal.revision + 1,
            phase=phase,
            objective=goal.objective,
            max_rounds=goal.max_rounds,
            rounds_started=goal.rounds_started,
            blocker_reason=blocker,
        )

    def _commit(
        self, goal: Goal, operation: str, *, armed: bool | None = None
    ) -> None:
        """每次变更追加 goal/change 事件，并携带变更后的完整快照。"""
        self._session.append(
            "goal/change",
            {"version": 1, "operation": operation, "goal": _goal_to_dict(goal)},
        )
        self._current = goal
        if armed is not None:
            self._armed = armed

    @classmethod
    def replay(cls, session: Session) -> "GoalStore":
        """连续性回放：只从 goal/change 与 goal 来源消息派生状态。

        revision 连续性只在同一目标内检查——每个新目标（id 不同）
        的 revision 都从 1 重新开始（create 生成 revision=1）。教学版未实现
        官方 invariant 的完整形状、生命周期迁移和时间戳校验。"""
        store = cls(session)
        seen_goal_ids: set[str] = set()
        for event in session.snapshot_events():
            if event.type == "user/message":
                source = event.data.get("source")
                if not isinstance(source, Mapping) or source.get("kind") != "goal":
                    continue
                current = store._current
                round_number = source.get("round")
                if (
                    current is None
                    or current.phase != PHASE_ACTIVE
                    or source.get("goal_id") != current.id
                    or source.get("revision") != current.revision
                    or not isinstance(round_number, int)
                    or isinstance(round_number, bool)
                    or round_number != current.rounds_started + 1
                    or round_number > current.max_rounds
                ):
                    raise ValueError("goal round 不连续或引用了错误的目标")
                store._current = replace(current, rounds_started=round_number)
                continue
            if event.type != "goal/change":
                continue
            if event.data.get("version") != 1:
                raise ValueError("goal/change version 必须为 1")
            operation = event.data.get("operation")
            previous = store._current
            if operation == "clear":
                cleared = event.data.get("cleared")
                if (
                    previous is None
                    or not isinstance(cleared, Mapping)
                    or cleared.get("id") != previous.id
                    or cleared.get("revision") != previous.revision + 1
                ):
                    raise ValueError("goal clear tombstone 无效")
                store._current = None
                continue
            raw_goal = event.data.get("goal")
            if not isinstance(raw_goal, Mapping):
                raise ValueError("goal/change 缺少完整 goal 快照")
            goal = _goal_from_dict(raw_goal)
            if operation == "create":
                if previous is not None and previous.phase != PHASE_COMPLETE:
                    raise ValueError("goal create 要求不存在未完成目标")
                if goal.id in seen_goal_ids or goal.revision != 1:
                    raise ValueError("goal create 必须使用新的 id 和 revision 1")
                if goal.phase != PHASE_ACTIVE or goal.rounds_started != 0:
                    raise ValueError("goal create 必须从 active 且 0 round 开始")
                seen_goal_ids.add(goal.id)
            else:
                if previous is None or goal.id != previous.id:
                    raise ValueError("goal 变更缺少匹配的当前目标")
                if goal.revision != previous.revision + 1:
                    raise ValueError(f"goal/change revision 不连续: {goal.revision}")
                _validate_transition(previous, goal, operation)
            store._current = goal
        return store


def _goal_to_dict(goal: Goal) -> dict[str, Any]:
    return {
        "id": goal.id,
        "revision": goal.revision,
        "phase": goal.phase,
        "objective": goal.objective,
        "max_rounds": goal.max_rounds,
        "rounds_started": goal.rounds_started,
        "blocker_reason": goal.blocker_reason,
    }


def _goal_from_dict(data: Mapping[str, Any]) -> Goal:
    required = {
        "id",
        "revision",
        "phase",
        "objective",
        "max_rounds",
        "rounds_started",
        "blocker_reason",
    }
    if set(data) != required:
        raise ValueError("goal 快照字段不完整")
    id_ = data["id"]
    revision = data["revision"]
    phase = data["phase"]
    objective = data["objective"]
    max_rounds = data["max_rounds"]
    rounds_started = data["rounds_started"]
    blocker_reason = data["blocker_reason"]
    if not isinstance(id_, str) or not id_:
        raise ValueError("goal id 必须是非空字符串")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not 1 <= revision <= MAX_SAFE_INTEGER
    ):
        raise ValueError("goal revision 必须是 JSON 安全范围内的正整数")
    if not isinstance(phase, str) or phase not in {
        PHASE_ACTIVE,
        PHASE_PAUSED,
        PHASE_BLOCKED,
        PHASE_COMPLETE,
    }:
        raise ValueError("goal phase 无效")
    if not isinstance(objective, str) or not objective.strip() or objective != objective.strip():
        raise ValueError("goal objective 必须是规范化的非空字符串")
    if (
        not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or not 1 <= max_rounds <= MAX_SAFE_INTEGER
    ):
        raise ValueError("goal max_rounds 必须是 JSON 安全范围内的正整数")
    if (
        not isinstance(rounds_started, int)
        or isinstance(rounds_started, bool)
        or not 0 <= rounds_started <= min(max_rounds, MAX_SAFE_INTEGER)
    ):
        raise ValueError("goal rounds_started 超出有效范围")
    if phase == PHASE_BLOCKED:
        if (
            not isinstance(blocker_reason, str)
            or not blocker_reason.strip()
            or blocker_reason != blocker_reason.strip()
        ):
            raise ValueError("blocked 目标必须包含阻塞原因")
    elif blocker_reason is not None:
        raise ValueError("非 blocked 目标不能包含阻塞原因")
    return Goal(
        id=id_,
        revision=revision,
        phase=phase,
        objective=objective,
        max_rounds=max_rounds,
        rounds_started=rounds_started,
        blocker_reason=blocker_reason,
    )


def _validate_transition(previous: Goal, goal: Goal, operation: object) -> None:
    if not isinstance(operation, str) or operation not in {
        "edit",
        "pause",
        "resume",
        "complete",
        "block",
    }:
        raise ValueError(f"goal change operation 无效: {operation!r}")
    if goal.rounds_started != previous.rounds_started:
        raise ValueError(f"goal {operation} 不能修改 rounds_started")
    if operation == "edit":
        if goal.phase != previous.phase or goal.blocker_reason != previous.blocker_reason:
            raise ValueError("goal edit 不能修改 phase 或 blocker_reason")
        return
    if goal.objective != previous.objective or goal.max_rounds != previous.max_rounds:
        raise ValueError(f"goal {operation} 不能修改 objective 或 max_rounds")
    valid = {
        "pause": previous.phase == PHASE_ACTIVE and goal.phase == PHASE_PAUSED,
        "resume": previous.phase in {PHASE_ACTIVE, PHASE_PAUSED, PHASE_BLOCKED}
        and goal.phase == PHASE_ACTIVE
        and goal.blocker_reason is None
        and previous.rounds_started < goal.max_rounds,
        "complete": previous.phase != PHASE_COMPLETE
        and goal.phase == PHASE_COMPLETE
        and goal.blocker_reason is None,
        "block": previous.phase == PHASE_ACTIVE
        and goal.phase == PHASE_BLOCKED
        and goal.blocker_reason is not None,
    }[operation]
    if not valid:
        raise ValueError(f"goal {operation} 状态转换无效")
