"""第 09 章 demo：摘要、工具结果剪枝与 spill。

运行（在项目根目录，需要 .env；压缩摘要由真实 DeepSeek 模型生成）：
    uv run python chapters/09-context-engineering/src/demo.py

演示：
1. 本地演示 append-only 工具结果剪枝；
2. 本地演示大结果 spill 与预算内预览；
3. 构造长会话，让上下文压力越过 80%；
4. 调用真实模型生成摘要 checkpoint，并对比压缩前后。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from client import DeepSeekClient, Message
from compaction import compact
from meter import TokenMeter
from pruner import ToolResultPruner
from session import Session
from spill import LocalSpillStore, SpillPolicy

# 教学用的小容量：4000 token 的「上下文窗口」，
# 让压力阈值（80% = 3200）能被十几条消息轻松触及。
CONTEXT_WINDOW = 4000


def build_long_conversation() -> list[Message]:
    """脚本构造一段长会话：模拟「实现 GRPO 训练项目」的连续问答。"""
    messages: list[Message] = [
        Message(role="system", content="你是一个 Python 编码助手，回答要详细。"),
    ]
    tasks = [
        "实现 GRPO 训练骨架",
        "实现奖励模型打分",
        "实现 loss 与 reward 曲线记录",
        "实现 --resume 断点续训",
        "实现流式数据加载",
        "实现 pass@k 评估脚本",
        "把训练配置抽到 yaml",
        "写 README 使用说明",
        "实现梯度累积与混合精度",
        "实现早停与 checkpoint 保存",
        "实现分布式数据并行",
        "实现日志与指标上报",
    ]
    for task in tasks:
        messages.append(Message(role="user", content=f"帮我{task}这部分。"))
        # 长回答：脚本化的伪代码块（约 1100 字符）
        messages.append(
            Message(
                role="assistant",
                content=(
                    f"好的，关于「{task}」，实现如下：\n\n```python\n"
                    + "\n".join(
                        f"def step_{i}(state, config):  # {task} 第 {i} 步：读取状态、更新指标、写回结果"
                        for i in range(20)
                    )
                    + "\n```\n\n关键点：状态按轮次推进，指标按步记录，配置经 yaml 注入。"
                ),
            )
        )
    return messages


def main() -> None:
    print("=== ① 工具结果剪枝：表层变短，原事件仍保留 ===")
    session = Session()
    original = "BEGIN-" + "中间内容" * 80 + "-END"
    session.append("tool/result", {"call_id": "call-1", "content": original})
    pruned = ToolResultPruner(
        threshold_chars=120,
        head_chars=40,
        tail_chars=20,
    ).prune_session(session)
    surface = session.derive_messages()[0].content or ""
    print(f"  replacement: {pruned.replacements} 条")
    print(f"  表层字符数: {len(original)} → {len(surface)}")
    print(
        "  原事件仍完整: "
        f"{session.snapshot_events()[0].data['content'] == original}"
    )

    print()
    print("=== ② spill：完整结果落盘，模型只收预算内预览 ===")
    with tempfile.TemporaryDirectory() as tmp:
        spill_text = "首部\n" + "数据行\n" * 300 + "尾部"
        policy = SpillPolicy(512, LocalSpillStore(tmp))
        inline = policy.transform(
            spill_text,
            session_id="demo-session",
            tool_name="shell",
            call_id="call-2",
        )
        stored = next(Path(tmp).rglob("*.txt"))
        print(
            f"  inline: {len(spill_text.encode('utf-8'))} → "
            f"{len(inline.encode('utf-8'))} bytes"
        )
        print(f"  落盘内容完整: {stored.read_text(encoding='utf-8') == spill_text}")

    meter = TokenMeter(context_window=CONTEXT_WINDOW)
    messages = build_long_conversation()

    print()
    print("=== ③ 长会话逐轮计量：压力变化 ===")
    threshold_tokens = int(CONTEXT_WINDOW * 0.8)
    for index, message in enumerate(messages[1:], start=1):
        ratio = meter.measure(messages[: index + 1]).total_tokens / CONTEXT_WINDOW
        if index % 3 == 0 or ratio > 0.8:
            flag = "  ← 达到 80% 阈值" if ratio >= 0.8 else ""
            print(f"  [{index:>2} 条] {ratio:>6.1%}{flag}")
    print(f"  阈值 = {threshold_tokens} token（{CONTEXT_WINDOW} × 0.8）")

    print()
    print("=== ④ 触发压缩：真实模型重放前缀 + 官方压缩指令 ===")
    before = meter.measure(messages)
    print(f"  压缩前：{len(messages)} 条消息，{before.total_tokens} token")
    result = compact(DeepSeekClient(), meter, messages)
    after = meter.measure(result.messages)

    print()
    print("=== ⑤ 压缩结果 ===")
    print(f"  ok: {result.ok}（{result.reason}）")
    print(f"  被压缩区：{result.shadowed_count} 条消息，{result.shadowed_tokens} token")
    print(f"  checkpoint：{result.checkpoint_tokens} token（含 preamble 与标签）")
    print(f"  压缩调用次数：{result.attempts}")
    print(f"  消息数：{len(messages)} → {len(result.messages)}")
    print(f"  token：{before.total_tokens} → {after.total_tokens}")
    print(f"  占用率：{before.total_tokens / CONTEXT_WINDOW:.1%} → "
          f"{after.total_tokens / CONTEXT_WINDOW:.1%}")

    print()
    print("=== ⑥ checkpoint 的真实内容（模型生成） ===")
    checkpoint = result.messages[1]
    print(f"  [role={checkpoint.role}]")
    for line in (checkpoint.content or "").splitlines()[:14]:
        print(f"  {line}")
    print("  …")


if __name__ == "__main__":
    main()
