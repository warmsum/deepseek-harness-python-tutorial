"""Run every chapter demo in sequence as a maintenance smoke check."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
EXPECTED_CHAPTERS = 17


def discover() -> tuple[Path, ...]:
    demos = tuple(sorted(CHAPTERS.glob("[0-9][0-9]-*/src/demo.py")))
    if len(demos) != EXPECTED_CHAPTERS:
        raise RuntimeError(f"expected {EXPECTED_CHAPTERS} chapter demos, discovered {len(demos)}")
    numbers = tuple(path.parent.parent.name[:2] for path in demos)
    expected = tuple(f"{number:02d}" for number in range(1, EXPECTED_CHAPTERS + 1))
    if numbers != expected:
        raise RuntimeError(f"chapter demo sequence is incomplete: {numbers!r}")
    return demos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="维护用：顺序运行全部章节示例，遇到非零退出码时停止。"
    )
    parser.parse_args()
    demos = discover()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    total = len(demos)
    for index, demo in enumerate(demos, start=1):
        relative = demo.relative_to(ROOT)
        print(f"[{index:02d}/{total:02d}] {relative}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(demo)],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            print(f"FAILED: {relative} (exit {completed.returncode})", file=sys.stderr)
            return completed.returncode or 1
    print(f"All {total} chapter demos completed without a non-zero exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
