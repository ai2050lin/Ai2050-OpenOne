#!/usr/bin/env python3
"""Reproduce Phase1023 sequentially without co-loading local models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
RESULT_ROOT = (
    TEST_ROOT
    / "result"
    / "phase1023_ecological_niche_execution_fork"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, "-u", str(TEST_ROOT / script), *arguments]
    print("[phase1023-run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun stages even when their audited output exists",
    )
    args = parser.parse_args()

    protocol_summary = RESULT_ROOT / "protocol" / "summary.json"
    if args.force or not protocol_summary.exists():
        run("phase1023_ecological_niche_protocol.py")

    for model in MODELS:
        behavior_summary = (
            RESULT_ROOT / "behavior" / model / "summary.json"
        )
        if args.force or not behavior_summary.exists():
            run("phase1023_fp16_behavior.py", model)

    pairing_summary = RESULT_ROOT / "pairing" / "summary.json"
    if args.force or not pairing_summary.exists():
        run("phase1023_strict_pairing.py")

    for model in MODELS:
        ecology_summary = RESULT_ROOT / "ecology" / model / "summary.json"
        if args.force or not ecology_summary.exists():
            run("phase1023_fp16_ecology_scan.py", model)

    run("phase1023_finalize.py")
    run("phase1023_audit.py")


if __name__ == "__main__":
    main()
