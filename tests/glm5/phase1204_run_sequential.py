#!/usr/bin/env python3
"""Run the three sealed Phase1204 behavior jobs and final audits in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1204_object_attribute_behavior_execution as execution


def checked(command: list[str]) -> None:
    print("[phase1204]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    for model_name in execution.MODEL_ORDER:
        checked(
            [
                python,
                str(TEST_ROOT / "phase1204_object_attribute_behavior_execution.py"),
                "run",
                model_name,
            ]
        )
    checked([python, str(TEST_ROOT / "phase1204_object_attribute_behavior_finalize.py"), "analyze"])
    checked(
        [
            python,
            str(TEST_ROOT / "phase1204_object_attribute_behavior_result_audit.py"),
            "--write",
        ]
    )
    checked([python, str(TEST_ROOT / "phase1204_object_attribute_behavior_finalize.py"), "finalize"])


if __name__ == "__main__":
    main()
