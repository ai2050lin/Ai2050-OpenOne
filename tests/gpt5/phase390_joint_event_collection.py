#!/usr/bin/env python3
"""Run the audited incremental event collector against the Phase390 denominator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase386_multitime_event_collection as collector  # noqa: E402


PHASE_ROOT = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
OUTPUT_ROOT = PHASE_ROOT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "discovery", "calibration", "physical_holdout")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_file(split: str) -> Path:
    name = (
        "phase390_instrument_audit_cases.jsonl"
        if split == "instrument_audit"
        else f"phase390_{split}_cases.jsonl"
    )
    return PHASE_ROOT / "protocol/private" / name


def split_authorized(split: str) -> None:
    freeze = read_json(PHASE_ROOT / "phase390_behavior_freeze_summary.json")
    if split == "instrument_audit":
        allowed = freeze["authorization"]["run_instrument_audit"]
    elif split == "discovery":
        path = PHASE_ROOT / "phase390_instrument_audit_summary.json"
        allowed = path.is_file() and read_json(path)["authorization"][
            "discovery_collection"
        ]
    elif split == "calibration":
        path = PHASE_ROOT / "phase390_discovery_candidate_freeze.json"
        allowed = path.is_file() and read_json(path)["authorization"][
            "calibration_collection"
        ]
    else:
        path = PHASE_ROOT / "phase390_calibration_summary.json"
        allowed = path.is_file() and read_json(path)["authorization"].get(
            "physical_holdout_collection", False
        )
    if not allowed:
        raise RuntimeError(f"Phase390 split is not authorized: {split}")


def run(model: str, split: str) -> dict[str, Any]:
    collector.PHASE_ROOT = PHASE_ROOT
    collector.case_file = case_file
    collector.split_authorized = split_authorized
    return collector.run_model(
        model,
        split,
        OUTPUT_ROOT,
        schema_version="64.3.0",
        phase_id="Phase390-IncrementalJointEventCollection",
        group_field="phase390_public_parallel_group_id",
        collection_label="Phase390",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    run(args.model, args.split)
