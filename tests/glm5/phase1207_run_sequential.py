#!/usr/bin/env python3
"""Run the frozen Phase1207 pipeline in protocol order."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "tests/glm5/phase1207_qwen3_causal_ancestry_necessity.py"
AUDIT = ROOT / "tests/glm5/phase1207_qwen3_causal_ancestry_necessity_audit.py"
ONSET = ROOT / "tests/glm5/result/phase1207_qwen3_causal_ancestry_necessity/analysis/onset_verdict.json"
NECESSITY = ROOT / "tests/glm5/result/phase1207_qwen3_causal_ancestry_necessity/analysis/necessity_verdict.json"


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run(str(MAIN), "protocol")
    run(str(AUDIT), "preexecution", "--write")
    run(str(MAIN), "capture")
    run(str(MAIN), "run-onset")
    run(str(MAIN), "analyze-onset")
    onset = read(ONSET)
    if onset["authorization"]["necessity_run"]:
        run(str(MAIN), "run-necessity")
        run(str(MAIN), "analyze-necessity")
        necessity = read(NECESSITY)
        if necessity["authorization"]["rescue_run"]:
            run(str(MAIN), "run-rescue")
            run(str(MAIN), "analyze-rescue")
    run(str(AUDIT), "result", "--write")
    run(str(MAIN), "finalize")


if __name__ == "__main__":
    main()
