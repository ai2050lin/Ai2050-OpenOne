#!/usr/bin/env python3
"""Freeze the 192 cases not used by the Phase365 engineering pilot."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
P365_FREEZE = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection_freeze"
COLLECTION = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection"
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/phase366_remaining_collection"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    pilot = read_jsonl(P365_FREEZE / "private" / "phase365_collection_execution_cases.jsonl")
    pilot_ids = {row["blind_case_id"] for row in pilot}
    all_cases = [
        row for row in read_jsonl(P362 / "private" / "phase362_execution_cases.jsonl")
        if row["phase362_split"] == "independent_calibration"
    ]
    remaining_source = [row for row in all_cases if row["blind_case_id"] not in pilot_ids]
    remaining = [{
        "schema_version": "43.1.0", "phase_id": "Phase366",
        "blind_case_id": row["blind_case_id"], "anonymous_model_id": row["anonymous_model_id"],
        "private_execution_model": row["model"], "anonymous_group_id": row["phase362_group_id"],
        "anonymous_condition_slot": "slot_" + hashlib.sha256(
            f"phase365-slot-v1:{row['phase362_group_id']}:{row['contrast_condition']}".encode()
        ).hexdigest()[:12],
        "prompt": row["prompt"], "raw_prompt": row["raw_prompt"],
        "source_fragment": row["source_fragment"], "query_fragment": row["query_fragment"],
        "tokenization_add_special_tokens": row["tokenization_add_special_tokens"],
        "phase365_split": "blind_motif_full_collection",
    } for row in remaining_source]
    if len(remaining) != 192:
        raise RuntimeError(f"Expected 192 remaining cases, got {len(remaining)}")
    if any(sum(row["private_execution_model"] == model for row in remaining) != 64 for model in MODELS):
        raise RuntimeError("Remaining model denominator is not 64 each")
    for model in MODELS:
        source = COLLECTION / "models" / model / "manifest.json"
        target = COLLECTION / "models" / model / "manifest_phase365_pilot.json"
        if not target.exists():
            shutil.copy2(source, target)
    write_jsonl(OUT / "private" / "phase366_remaining_execution_cases.jsonl", remaining)
    write_jsonl(OUT / "private" / "phase366_all_execution_cases.jsonl", [*pilot, *remaining])
    summary = {
        "schema_version": "43.1.0", "phase_id": "Phase366",
        "denominator": {
            "all_discovery_case_count": len(all_cases), "phase365_pilot_case_count": len(pilot),
            "remaining_case_count": len(remaining), "remaining_case_count_per_model": 64,
            "physical_confirmation_overlap_count": 0,
        },
        "quality": {
            "pilot_cases_not_reexecuted": True,
            "model_effects_used_for_selection": False,
            "semantic_labels_available_to_collector": False,
            "target_specific_competition_available_to_collector": False,
        },
        "authorization": {"remaining_collection_authorized": True, "physical_confirmation_authorized": False},
    }
    write_json(OUT / "phase366_remaining_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
