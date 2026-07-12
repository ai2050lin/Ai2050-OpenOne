#!/usr/bin/env python3
"""Freeze reconstruction, repeat, and same-operation template floors before motif scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
DESCRIPTORS = PHASE_ROOT / "label_blind_flow_descriptors"
EXECUTION = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time/private/phase362_execution_cases.jsonl"
COLLECTION = PHASE_ROOT / "engineering_collection" / "phase366_full_collection_summary.json"
REPEAT = PHASE_ROOT / "repeat_noise_format_gate" / "phase365_repeat_noise_summary.json"
OUT = PHASE_ROOT / "blind_threshold_custodian"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot take a quantile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def operation_pair(condition: str) -> tuple[str, str]:
    prefix = condition.split("_", 1)[0]
    mapping = {
        "A": ("demand", "x"),
        "C": ("demand", "y"),
        "B": ("control", "x"),
        "D": ("control", "y"),
    }
    if prefix not in mapping:
        raise ValueError(f"Unknown frozen condition slot: {condition}")
    return mapping[prefix]


def structural_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["generation_time"],
        row["relative_depth_bin"],
        row["source_role_alias"],
        row["receiver_role_alias"],
    )


def main() -> None:
    execution_rows = [
        row for row in read_jsonl(EXECUTION)
        if row["phase362_split"] == "independent_calibration"
    ]
    private_case = {}
    for row in execution_rows:
        pair_kind, lexical_slot = operation_pair(row["contrast_condition"])
        private_case[row["blind_case_id"]] = {
            "model": row["model"],
            "group": row["phase362_group_id"],
            "pair_kind": pair_kind,
            "lexical_slot": lexical_slot,
        }

    collection = read_json(COLLECTION)
    repeat = read_json(REPEAT)
    reconstruction_error = max(collection["results"]["max_errors"].values())
    reconstruction_feature_floor = 4.0 * reconstruction_error
    repeat_floor = repeat["results"]["observed_fixed_execution_repeat_noise_floor"]

    all_threshold_rows = []
    model_summaries = []
    combined_private_pair_hash = hashlib.sha256()
    calibration_descriptor_count = 0
    for model in MODELS:
        descriptor_path = DESCRIPTORS / "private" / f"{model}_directed_flow_descriptors.jsonl"
        pending: dict[tuple[Any, ...], tuple[str, dict[str, float]]] = {}
        differences: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        pair_observation_count = 0
        feature_difference_count = 0
        discovery_descriptor_count = 0
        paired_case_ids = set()
        for row in read_jsonl(descriptor_path):
            if row["split"] == "blind_calibration":
                calibration_descriptor_count += 1
                continue
            discovery_descriptor_count += 1
            case = private_case[row["anonymous_case_id"]]
            if case["model"] != model:
                raise RuntimeError("Private threshold case/model mismatch")
            route_key = (
                case["group"], case["pair_kind"],
                row["generation_time"], row["layer_index"],
                row["source_role_alias"], row["receiver_role_alias"],
            )
            current = (case["lexical_slot"], row["features"])
            previous = pending.pop(route_key, None)
            if previous is None:
                pending[route_key] = current
                continue
            if previous[0] == current[0]:
                raise RuntimeError(f"Duplicate lexical slot in same-operation pair: {route_key}")
            pair_observation_count += 1
            paired_case_ids.add(row["anonymous_case_id"])
            combined_private_pair_hash.update("|".join(map(str, route_key)).encode())
            group_key = structural_key(row)
            for feature, value in row["features"].items():
                difference = abs(float(value) - float(previous[1][feature]))
                differences[group_key + (feature,)].append(difference)
                feature_difference_count += 1
        if pending:
            raise RuntimeError(f"Unpaired discovery descriptor rows for {model}: {len(pending)}")

        anonymous_model_id = next(
            row["anonymous_model_id"] for row in read_jsonl(descriptor_path)
        )
        model_threshold_rows = []
        for key, values in sorted(differences.items()):
            generation_time, depth, source, receiver, feature = key
            median = quantile(values, 0.5)
            q75 = quantile(values, 0.75)
            maximum = max(values)
            frozen_floor = max(repeat_floor, reconstruction_feature_floor, q75)
            threshold_row = {
                "anonymous_model_id": anonymous_model_id,
                "generation_time": generation_time,
                "relative_depth_bin": depth,
                "source_role_alias": source,
                "receiver_role_alias": receiver,
                "feature": feature,
                "same_operation_template_difference_count": len(values),
                "template_absolute_difference_median": median,
                "template_absolute_difference_q75": q75,
                "template_absolute_difference_max_diagnostic_only": maximum,
                "repeat_floor": repeat_floor,
                "reconstruction_feature_floor": reconstruction_feature_floor,
                "frozen_effect_floor": frozen_floor,
            }
            model_threshold_rows.append(threshold_row)
            all_threshold_rows.append(threshold_row)
        model_summaries.append({
            "model": model,
            "anonymous_model_id": anonymous_model_id,
            "discovery_descriptor_count": discovery_descriptor_count,
            "paired_route_observation_count": pair_observation_count,
            "feature_difference_count": feature_difference_count,
            "threshold_row_count": len(model_threshold_rows),
        })

    threshold_path = OUT / "phase366_frozen_descriptor_floors.jsonl"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_path.open("w", encoding="utf-8") as handle:
        for row in all_threshold_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    threshold_digest = hashlib.sha256(threshold_path.read_bytes()).hexdigest()
    expected_calibration = 96 * (179712 + 199680 + 139776) // 288
    summary = {
        "schema_version": "43.4.0",
        "phase_id": "Phase366-D",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_nonsemantic_descriptor_effect_floors_before_blind_motif_scoring",
        "denominator": {
            "model_count": len(MODELS),
            "discovery_group_count": 48,
            "same_operation_case_pair_count": 96,
            "paired_route_observation_count": sum(row["paired_route_observation_count"] for row in model_summaries),
            "feature_difference_count": sum(row["feature_difference_count"] for row in model_summaries),
            "threshold_row_count": len(all_threshold_rows),
            "calibration_descriptor_count_not_used": calibration_descriptor_count,
        },
        "floors": {
            "fixed_execution_repeat_floor": repeat_floor,
            "max_engineering_reconstruction_error": reconstruction_error,
            "reconstruction_feature_floor_multiplier": 4.0,
            "global_reconstruction_feature_floor": reconstruction_feature_floor,
            "template_floor_statistic": "upper_quartile_of_absolute_same_operation_cross_lexical_difference",
            "frozen_floor_formula": "max(repeat_floor, reconstruction_feature_floor, template_difference_q75)",
        },
        "results": {
            "calibration_rows_used_for_thresholds": False,
            "target_or_correct_answer_used": False,
            "family_or_mechanism_used": False,
            "condition_effect_counted_as_noise": False,
            "same_operation_cross_lexical_template_floor_used": True,
            "thresholds_frozen_before_scoring": True,
            "all_thresholds_finite": all(
                math.isfinite(row["frozen_effect_floor"]) for row in all_threshold_rows
            ),
        },
        "models": model_summaries,
        "threshold_table": {
            "relative_path": str(threshold_path.relative_to(OUT)),
            "sha256": threshold_digest,
        },
        "private_pair_registry_digest": combined_private_pair_hash.hexdigest(),
        "authorization": {
            "blind_discovery_scoring_authorized": (
                len(all_threshold_rows) > 0
                and calibration_descriptor_count == expected_calibration
            ),
            "blind_calibration_scoring_authorized_before_candidate_freeze": False,
            "semantic_label_reveal_authorized": False,
            "physical_confirmation_authorized": False,
        },
        "claim_boundary": {
            "thresholds_are_noise_model": False,
            "thresholds_are_conservative_engineering_and_template_variation_floors": True,
            "motif_candidate_count": 0,
            "language_path_discovered": False,
        },
        "next_decision": "score_only_blind_discovery_groups_then_freeze_candidates_before_calibration",
    }
    write_json(OUT / "phase366_threshold_custodian_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
