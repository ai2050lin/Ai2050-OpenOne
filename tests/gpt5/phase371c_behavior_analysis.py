#!/usr/bin/env python3
"""Freeze common three-model Phase371C behavior-qualified internal denominators."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
CASE_ROOT = PHASE371 / "phase371c_case_bank"
BEHAVIOR = PHASE371 / "phase371c_behavior_qualification"
OUT = PHASE371 / "phase371c_behavior_analysis"
MODELS = ("qwen3", "glm4", "deepseek7b")
MIN_DISCOVERY_GROUPS = 8
MIN_CALIBRATION_GROUPS = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    behavior_rows = []
    model_summaries = []
    for model in MODELS:
        behavior_rows.extend(read_jsonl(
            BEHAVIOR / "private/models" / model / "phase371c_behavior_rows.jsonl"
        ))
        model_summaries.append(json.loads(
            (BEHAVIOR / "models" / model / "complete.json").read_text(encoding="utf-8")
        ))
    if len(behavior_rows) != 864:
        raise RuntimeError(f"Expected 864 nonphysical behavior rows, got {len(behavior_rows)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata = {}
    for row in behavior_rows:
        parallel = row["anonymous_parallel_group_id"]
        grouped[(row["model"], parallel)].append(row)
        metadata[parallel] = (row["mechanism_id"], row["phase371c_split"])
    qualified = {
        key: len(rows) == 4 and all(row["strict_behavior_correct"] for row in rows)
        for key, rows in grouped.items()
    }
    common_groups = {
        parallel for parallel in metadata
        if all(qualified.get((model, parallel), False) for model in MODELS)
    }
    common_counts = Counter(metadata[parallel] for parallel in common_groups)
    mechanisms = sorted({mechanism for mechanism, _split in metadata.values()})
    mechanism_rows = []
    eligible_mechanisms = []
    for mechanism in mechanisms:
        discovery = common_counts[(mechanism, "fresh_discovery")]
        calibration = common_counts[(mechanism, "sealed_calibration")]
        eligible = discovery >= MIN_DISCOVERY_GROUPS and calibration >= MIN_CALIBRATION_GROUPS
        if eligible:
            eligible_mechanisms.append(mechanism)
        mechanism_rows.append({
            "mechanism_id": mechanism,
            "common_qualified_discovery_groups": discovery,
            "common_qualified_calibration_groups": calibration,
            "minimum_discovery_groups": MIN_DISCOVERY_GROUPS,
            "minimum_calibration_groups": MIN_CALIBRATION_GROUPS,
            "internal_discovery_eligible": eligible,
        })
    execution_rows = read_jsonl(CASE_ROOT / "private/phase371c_nonphysical_execution_cases.jsonl")
    selected_groups = {
        parallel for parallel in common_groups
        if metadata[parallel][0] in eligible_mechanisms and metadata[parallel][1] == "fresh_discovery"
    }
    forbidden = {
        "family_id", "mechanism_id", "semantic_group_id", "contrast_condition",
        "operation_demanded", "target", "target_aliases", "distractors", "instruction",
        "question", "language",
    }
    collector_rows = []
    for row in execution_rows:
        if row["anonymous_parallel_group_id"] not in selected_groups:
            continue
        collector = {key: value for key, value in row.items() if key not in forbidden}
        collector["semantic_labels_available_to_collector"] = False
        collector["target_specific_competition_available_to_collector"] = False
        collector_rows.append(collector)
    expected_cases = len(selected_groups) * 4 * len(MODELS)
    if len(collector_rows) != expected_cases:
        raise RuntimeError(f"Collector denominator mismatch: {len(collector_rows)} != {expected_cases}")
    summary = {
        "schema_version": "47.10.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_common_three_model_four_condition_behavior_qualified_internal_discovery_denominator",
        "behavior": {
            "nonphysical_case_count": len(behavior_rows),
            "strict_correct_case_count": sum(row["strict_behavior_correct"] for row in behavior_rows),
            "physical_case_count_loaded": sum(row["physical_case_count_loaded"] for row in model_summaries),
            "common_qualified_group_count": len(common_groups),
            "model_summaries": model_summaries,
        },
        "mechanisms": mechanism_rows,
        "results": {
            "eligible_mechanisms": eligible_mechanisms,
            "eligible_mechanism_count": len(eligible_mechanisms),
            "four_mechanism_behavior_gate_pass": len(eligible_mechanisms) == 4,
            "partial_discovery_cycle_authorized": bool(eligible_mechanisms),
            "behavior_failure_groups_replaced": False,
            "language_mechanism_claimed": False,
        },
        "internal_discovery": {
            "parallel_group_count": len(selected_groups),
            "case_count": len(collector_rows),
            "case_count_per_model": len(collector_rows) // len(MODELS),
            "all_four_conditions_retained": True,
            "semantic_labels_available_to_collector": False,
            "calibration_internal_states_opened": False,
            "physical_holdout_opened": False,
        },
        "authorization": {
            "freeze_internal_collection_code_and_hashes": bool(eligible_mechanisms),
            "run_internal_discovery_collection_before_hash_freeze": False,
            "run_calibration_internal_collection": False,
            "run_physical": False,
        },
        "next_decision": "freeze_sufficient_state_internal_collection_for_two_behavior_eligible_mechanisms",
    }
    write_json(OUT / "phase371c_behavior_analysis_summary.json", summary)
    write_jsonl(OUT / "private/phase371c_discovery_collector_cases.jsonl", collector_rows)
    write_jsonl(OUT / "phase371c_discovery_blind_case_registry.jsonl", [
        {key: value for key, value in row.items() if key != "private_execution_model"}
        for row in collector_rows
    ])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
