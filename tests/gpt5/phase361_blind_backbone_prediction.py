#!/usr/bin/env python3
"""Separate a blind public backbone and test next-layer persistence on calibration."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
ROOT_OUT = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTH_BINS = ("early", "middle", "late")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def depth_bin(value: float) -> str:
    if value < 1 / 3:
        return "early"
    if value < 2 / 3:
        return "middle"
    return "late"


def feature_values(row: dict[str, Any]) -> dict[str, float]:
    values = {}
    for component, role_values in row["component_norms"].items():
        for role, value in zip(row["role_names"], role_values, strict=True):
            values[f"state::{component}::{role}"] = float(value)
    return values


def main() -> None:
    rows = [
        row for model in MODELS
        for row in read_jsonl(ROOT_OUT / "models" / model / "phase361_r0_r1_ledger_rows.jsonl")
    ]
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["blind_case_id"]][row["layer_index"]] = row

    discovery_groups: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    calibration_transitions = []
    for blind_case_id, case_rows in by_case.items():
        for layer, row in case_rows.items():
            if layer + 1 not in case_rows:
                continue
            next_row = case_rows[layer + 1]
            current_features = feature_values(row)
            next_features = feature_values(next_row)
            model = row["anonymous_model_id"]
            depth = depth_bin(float(row["relative_depth"]))
            for feature, current in current_features.items():
                pair = (current, next_features[feature])
                if row["split"] == "blind_discovery":
                    discovery_groups[(model, depth, feature)].append(pair)
                else:
                    calibration_transitions.append((blind_case_id, layer, model, depth, feature, *pair))

    backbones = []
    lookup = {}
    for key, values in discovery_groups.items():
        model, depth, feature = key
        current_median = median(value[0] for value in values)
        next_median = median(value[1] for value in values)
        residuals = [abs(value[1] - next_median) for value in values]
        next_mad = median(residuals)
        row = {
            "schema_version": "38.1.0", "phase_id": "Phase361", "created_at": now(),
            "anonymous_model_id": model, "depth_bin": depth, "feature_id": feature,
            "discovery_transition_count": len(values),
            "current_median": round(current_median, 7),
            "next_median": round(next_median, 7),
            "median_delta": round(next_median - current_median, 7),
            "next_mad": round(next_mad, 7),
            "semantic_label_used": False,
        }
        backbones.append(row)
        lookup[key] = row

    evaluation_rows = []
    aggregate: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for blind_case_id, layer, model, depth, feature, current, actual_next in calibration_transitions:
        backbone = lookup[(model, depth, feature)]
        scale = max(abs(float(backbone["next_median"])), 1e-6)
        baseline_error = abs(actual_next - float(backbone["next_median"])) / scale
        state_prediction = current + float(backbone["median_delta"])
        state_error = abs(actual_next - state_prediction) / scale
        gain = baseline_error - state_error
        evaluation_rows.append({
            "schema_version": "38.1.0", "phase_id": "Phase361",
            "blind_case_id": blind_case_id, "layer_index": layer,
            "anonymous_model_id": model, "depth_bin": depth, "feature_id": feature,
            "normalized_baseline_error": round(baseline_error, 9),
            "normalized_state_error": round(state_error, 9),
            "prediction_gain": round(gain, 9),
            "semantic_label_used": False,
        })
        aggregate[(depth, feature)][model].append(gain)

    candidates = []
    for (depth, feature), model_values in aggregate.items():
        model_gains = {
            model: sum(values) / len(values)
            for model, values in model_values.items()
        }
        if len(model_gains) == 3 and all(value > 0 for value in model_gains.values()):
            candidates.append({
                "schema_version": "38.1.0", "phase_id": "Phase361", "created_at": now(),
                "candidate_id": f"next_layer::{depth}::{feature}",
                "depth_bin": depth, "feature_id": feature,
                "model_mean_prediction_gain": {key: round(value, 9) for key, value in model_gains.items()},
                "all_three_models_positive": True,
                "semantic_label_used": False,
                "candidate_is_operation_specific": False,
                "candidate_is_causal": False,
            })

    global_baseline = sum(row["normalized_baseline_error"] for row in evaluation_rows) / len(evaluation_rows)
    global_state = sum(row["normalized_state_error"] for row in evaluation_rows) / len(evaluation_rows)
    summary = {
        "schema_version": "38.1.0", "phase_id": "Phase361", "created_at": now(),
        "denominator": {
            "blind_case_count": len(by_case),
            "discovery_backbone_cell_count": len(backbones),
            "calibration_transition_feature_count": len(evaluation_rows),
            "shared_positive_candidate_count": len(candidates),
        },
        "results": {
            "global_normalized_baseline_error": round(global_baseline, 9),
            "global_normalized_state_error": round(global_state, 9),
            "global_prediction_gain": round(global_baseline - global_state, 9),
            "state_persistence_beats_public_backbone": global_state < global_baseline,
        },
        "claim_boundary": {
            "next_layer_prediction_tested": True,
            "next_generation_step_prediction_tested": False,
            "shared_candidates_are_language_operations": False,
            "semantic_labels_revealed": False,
            "physical_heldout_revealed": False,
            "causal_intervention_executed": False,
        },
        "next_decision": (
            "posthoc_function_audit_of_frozen_predictive_candidates"
            if candidates else "stop_no_cross_model_predictive_candidate"
        ),
    }
    write_jsonl(ROOT_OUT / "phase361_public_backbone_rows.jsonl", backbones)
    write_jsonl(ROOT_OUT / "phase361_next_layer_evaluation_rows.jsonl", evaluation_rows)
    write_jsonl(ROOT_OUT / "phase361_frozen_predictive_candidates.jsonl", candidates)
    write_json(ROOT_OUT / "phase361_blind_prediction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
