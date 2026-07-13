#!/usr/bin/env python3
"""Collect Phase395 exact multitime events under staged authorization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase386_multitime_event_collection as collector  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
OUTPUT_ROOT = OUT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "discovery", "calibration", "physical_holdout")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_file(split: str) -> Path:
    return OUT / f"protocol/private/phase395_{split}_cases.jsonl"


def split_authorized(split: str) -> None:
    if split == "instrument_audit":
        source = OUT / "phase395_behavior_freeze_summary.json"
        allowed = source.is_file() and read_json(source)["authorization"]["run_instrument_audit"]
    elif split == "discovery":
        source = OUT / "phase395_instrument_audit_summary.json"
        allowed = source.is_file() and read_json(source)["authorization"]["discovery_collection"]
    elif split == "calibration":
        source = OUT / "phase395_discovery_candidate_freeze.json"
        allowed = source.is_file() and read_json(source)["authorization"]["calibration_collection"]
    else:
        source = OUT / "phase395_calibration_summary.json"
        allowed = source.is_file() and read_json(source)["authorization"]["physical_holdout_collection"]
    if not allowed:
        raise RuntimeError(f"Phase395 split is not authorized: {split}")


def run(model: str, split: str) -> dict[str, Any]:
    collector.case_file = case_file
    collector.split_authorized = split_authorized
    return collector.run_model(
        model,
        split,
        OUTPUT_ROOT,
        schema_version="69.3.0",
        phase_id="Phase395-NaturalBindingEventCollection",
        group_field="phase395_public_parallel_group_id",
        collection_label="Phase395",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    run(args.model, args.split)
