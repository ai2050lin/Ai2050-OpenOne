#!/usr/bin/env python3
"""Run Phase400 trace shards sequentially in the frozen model order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase400_partial_order"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("discovery", "calibration", "physical_holdout")
SHARD_SIZE = 2


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(arguments: list[str]) -> None:
    print("[phase400 orchestrator]", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main(stage: str, only_model: str | None) -> None:
    freeze = read_json(OUT / "phase400_behavior_freeze_summary.json")
    group_count = sum(
        len(splits[stage]) for splits in freeze["selected_groups_private"].values()
    )
    if stage == "discovery":
        audit = read_json(OUT / "phase400_instrument_audit.json")
        if not audit["authorization"]["run_discovery_trace"]:
            raise RuntimeError("Phase400 discovery is not authorized")
    elif stage == "calibration":
        discovery = read_json(OUT / "phase400_partial_order_discovery.json")
        if not discovery["authorization"]["run_calibration_trace"]:
            raise RuntimeError("Phase400 calibration is not authorized")
    else:
        calibration = read_json(OUT / "phase400_partial_order_calibration.json")
        if not calibration["authorization"]["open_physical_holdout"]:
            raise RuntimeError("Phase400 physical holdout is not authorized")
    models = (only_model,) if only_model else MODELS
    shard_count = (group_count + SHARD_SIZE - 1) // SHARD_SIZE
    for model in models:
        for shard_index in range(shard_count):
            run_command(
                [
                    sys.executable,
                    "tests/gpt5/phase400_dynamic_trace_collection.py",
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
                "tests/gpt5/phase400_dynamic_trace_shard_merge.py",
                "--model",
                model,
                "--stage",
                stage,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--only-model", choices=MODELS)
    args = parser.parse_args()
    main(args.stage, args.only_model)
