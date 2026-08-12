#!/usr/bin/env python3
"""Run the frozen Phase1205 Qwen3-only pipeline in its only legal order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN = TEST_ROOT / "phase1205_qwen3_object_attribute_vertical_closure.py"
AUDIT = TEST_ROOT / "phase1205_qwen3_object_attribute_vertical_closure_audit.py"


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run(str(MAIN), "protocol")
    run(str(AUDIT), "preexecution", "--write")
    run(str(MAIN), "run")
    run(str(MAIN), "analyze")
    run(str(AUDIT), "result", "--write")
    run(str(MAIN), "finalize")


if __name__ == "__main__":
    main()
