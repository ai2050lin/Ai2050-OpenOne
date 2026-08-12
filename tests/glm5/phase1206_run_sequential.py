#!/usr/bin/env python3
"""Run the frozen Phase1206 pipeline in protocol order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "tests/glm5/phase1206_qwen3_object_attribute_causal_transfer.py"
AUDIT = ROOT / "tests/glm5/phase1206_qwen3_object_attribute_causal_transfer_audit.py"


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
