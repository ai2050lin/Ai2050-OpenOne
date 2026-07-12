#!/usr/bin/env python3
"""Collect Phase391 calibration/physical ledgers from the sealed Phase390 groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase386_multitime_event_collection as collector  # noqa: E402


P390 = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
OUT = ROOT / "tests/gpt5/result/phase391_local_parent_graph"
OUTPUT_ROOT = OUT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("calibration", "physical_holdout")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_file(split: str) -> Path:
    return P390 / f"protocol/private/phase390_{split}_cases.jsonl"


def split_authorized(split: str) -> None:
    if split == "calibration":
        path = OUT / "phase391_discovery_candidate_freeze.json"
        allowed = path.is_file() and read_json(path)["authorization"][
            "calibration_collection"
        ]
    else:
        path = OUT / "phase391_calibration_summary.json"
        allowed = path.is_file() and read_json(path)["authorization"][
            "physical_holdout_collection"
        ]
    if not allowed:
        raise RuntimeError(f"Phase391 split is not authorized: {split}")


def run(model: str, split: str) -> dict[str, Any]:
    collector.case_file = case_file
    collector.split_authorized = split_authorized
    return collector.run_model(
        model,
        split,
        OUTPUT_ROOT,
        schema_version="65.1.0",
        phase_id="Phase391-IncrementalLocalParentCollection",
        group_field="phase390_public_parallel_group_id",
        collection_label="Phase391",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    run(args.model, args.split)
