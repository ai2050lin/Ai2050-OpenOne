#!/usr/bin/env python3
"""Run Phase399 trace shards sequentially in the frozen model order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")
GROUPS = {"discovery": 30, "calibration": 15, "physical_holdout": 15}
SHARD_SIZE = 3


def run_command(arguments: list[str]) -> None:
    print("[phase399 orchestrator]", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main(stage: str, only_model: str | None) -> None:
    models = (only_model,) if only_model else MODELS
    shard_count = (GROUPS[stage] + SHARD_SIZE - 1) // SHARD_SIZE
    for model in models:
        for shard_index in range(shard_count):
            run_command(
                [
                    sys.executable,
                    "tests/gpt5/phase399_dynamic_trace_collection.py",
                    "--model",
                    model,
                    "--stage",
                    stage,
                    "--shard-index",
                    str(shard_index),
                    "--shard-size",
                    str(SHARD_SIZE),
                ]
            )
        run_command(
            [
                sys.executable,
                "tests/gpt5/phase399_dynamic_trace_shard_merge.py",
                "--model",
                model,
                "--stage",
                stage,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(GROUPS), required=True)
    parser.add_argument("--only-model", choices=MODELS)
    args = parser.parse_args()
    main(args.stage, args.only_model)
