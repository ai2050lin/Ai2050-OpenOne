#!/usr/bin/env python3
"""Run Phase1105 protocol and the three local FP16 models sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TEST_ROOT = ROOT / "tests" / "glm5"


def run(script: str, *args: str) -> None:
    subprocess.run([str(PYTHON), str(TEST_ROOT / script), *args], cwd=ROOT, check=True)


def main() -> None:
    run("phase1105_natural_synonym_address_protocol.py")
    for model in ("qwen3", "glm4", "deepseek7b"):
        run("phase1105_natural_synonym_address_behavior.py", model)
    run("phase1105_natural_synonym_address_finalize.py")
    run("phase1105_natural_synonym_address_result_audit.py")


if __name__ == "__main__":
    main()
