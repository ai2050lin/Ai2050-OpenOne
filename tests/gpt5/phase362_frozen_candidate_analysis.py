#!/usr/bin/env python3
"""Test frozen Phase361 next-layer formulas on unseen Phase362 groups."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
P361 = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def linear_fit(values: list[tuple[float, float]]) -> tuple[float, float]:
    x_mean = sum(value[0] for value in values) / len(values)
    y_mean = sum(value[1] for value in values) / len(values)
    variance = sum((value[0] - x_mean) ** 2 for value in values)
    slope = (
        sum((value[0] - x_mean) * (value[1] - y_mean) for value in values) / variance
        if variance > 1e-12 else 0.0
    )
    return slope, y_mean - slope * x_mean


def main() -> None:
    candidates = read_jsonl(P361 / "phase361_frozen_predictive_candidates.jsonl")
    p361_labels = {row["blind_case_id"]: row["model"] for row in read_jsonl(P361 / "private" / "phase361_label_key.jsonl")}
    p362_labels = {row["blind_case_id"]: row["model"] for row in read_jsonl(P362 / "private" / "phase362_label_key.jsonl")}
    p361_rows = [
        row for model in MODELS
        for row in read_jsonl(P361 / "models" / model / "phase361_r0_r1_ledger_rows.jsonl")
        if row["split"] == "blind_discovery"
    ]
    p362_rows = [
        row for model in MODELS
        for row in read_jsonl(P362 / "models" / model / "phase362_generation_time_rows.jsonl")
    ]

    train_cases: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in p361_rows:
        train_cases[row["blind_case_id"]][row["layer_index"]] = row
    training: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for case_id, layers in train_cases.items():
        model = p361_labels[case_id]
        for layer, row in layers.items():
            if layer + 1 not in layers:
                continue
            current, following = feature_values(row), feature_values(layers[layer + 1])
            depth = depth_bin(float(row["relative_depth"]))
            for feature, value in current.items():
                training[(model, depth, feature)].append((value, following[feature]))
    predictors = {}
    for key, values in training.items():
        slope, intercept = linear_fit(values)
        predictors[key] = {
            "b0_next_median": median(value[1] for value in values),
            "b2_slope": slope, "b2_intercept": intercept,
            "b3_median_delta": median(value[1] - value[0] for value in values),
        }

    test_cases: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in p362_rows:
        test_cases[(row["blind_case_id"], row["generation_time"])][row["layer_index"]] = row
    candidate_lookup = {(row["depth_bin"], row["feature_id"]): row for row in candidates}
    group_errors: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    temporal_identity: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (case_id, generation_time), layers in test_cases.items():
        model = p362_labels[case_id]
        for layer, row in layers.items():
            current_features = feature_values(row)
            depth = depth_bin(float(row["relative_depth"]))
            if layer + 1 in layers:
                next_features = feature_values(layers[layer + 1])
                for feature, current in current_features.items():
                    candidate = candidate_lookup.get((depth, feature))
                    if candidate is None:
                        continue
                    params = predictors[(model, depth, feature)]
                    actual = next_features[feature]
                    scale = max(abs(params["b0_next_median"]), 1e-6)
                    predictions = {
                        "b0_public": params["b0_next_median"],
                        "b1_identity": current,
                        "b2_linear": params["b2_slope"] * current + params["b2_intercept"],
                        "b3_frozen_delta": current + params["b3_median_delta"],
                    }
                    key = (candidate["candidate_id"], model, row["anonymous_group_id"])
                    for name, prediction in predictions.items():
                        group_errors[key][name].append(abs(actual - prediction) / scale)
            if generation_time < 2 and (case_id, generation_time + 1) in test_cases:
                later = test_cases[(case_id, generation_time + 1)].get(layer)
                if later is not None:
                    later_features = feature_values(later)
                    for feature, current in current_features.items():
                        candidate = candidate_lookup.get((depth, feature))
                        if candidate is None:
                            continue
                        scale = max(abs(current), 1e-6)
                        temporal_identity[(candidate["candidate_id"], model, row["anonymous_group_id"], f"t{generation_time}_t{generation_time+1}")].append(
                            abs(later_features[feature] - current) / scale
                        )

    group_means = {
        key: {name: sum(values) / len(values) for name, values in errors.items()}
        for key, errors in group_errors.items()
    }
    audit_rows = []
    group_rows = []
    for (candidate_id, model, group_id), errors in group_means.items():
        strongest = min(errors[name] for name in ("b0_public", "b1_identity", "b2_linear"))
        group_rows.append({
            "candidate_id": candidate_id, "model": model,
            "anonymous_group_id": group_id,
            "mean_errors": {key: round(value, 9) for key, value in errors.items()},
            "b3_gain_over_strongest_alternative": round(strongest - errors["b3_frozen_delta"], 9),
        })
    for candidate in candidates:
        model_results = {}
        for model in MODELS:
            values = [errors for (candidate_id, value_model, _group), errors in group_means.items() if candidate_id == candidate["candidate_id"] and value_model == model]
            means = {name: sum(value[name] for value in values) / len(values) for name in ("b0_public", "b1_identity", "b2_linear", "b3_frozen_delta")}
            strongest_alternative = min(means[name] for name in ("b0_public", "b1_identity", "b2_linear"))
            model_results[model] = {
                "independent_group_count": len(values),
                "mean_errors": {key: round(value, 9) for key, value in means.items()},
                "b3_gain_over_strongest_alternative": round(strongest_alternative - means["b3_frozen_delta"], 9),
                "b3_beats_all_alternatives": means["b3_frozen_delta"] < strongest_alternative,
            }
        temporal = {}
        for transition in ("t0_t1", "t1_t2"):
            transition_values = [
                sum(values) / len(values)
                for (candidate_id, _model, _group, value_transition), values in temporal_identity.items()
                if candidate_id == candidate["candidate_id"] and value_transition == transition
            ]
            temporal[transition] = round(sum(transition_values) / len(transition_values), 9)
        audit_rows.append({
            **candidate,
            "independent_model_results": model_results,
            "b3_beats_all_alternatives_all_models": all(value["b3_beats_all_alternatives"] for value in model_results.values()),
            "temporal_identity_error_descriptive_only": temporal,
            "temporal_predictor_was_frozen_in_phase361": False,
        })

    independently_best = [row for row in audit_rows if row["b3_beats_all_alternatives_all_models"]]
    summary = {
        "schema_version": "39.2.0", "phase_id": "Phase362", "created_at": now(),
        "denominator": {
            "frozen_candidate_count": len(candidates),
            "independent_case_count": len({row["blind_case_id"] for row in p362_rows}),
            "independent_group_count": len({row["anonymous_group_id"] for row in p362_rows}),
            "generation_time_count": 3,
            "ledger_row_count": len(p362_rows),
        },
        "results": {
            "b3_independently_best_all_models_count": len(independently_best),
            "b3_not_best_count": len(candidates) - len(independently_best),
        },
        "identifiability_audit": {
            "phase361_candidate_contains_next_layer_formula": True,
            "phase361_candidate_contains_next_generation_formula": False,
            "phase361_candidate_contains_competition_formula": False,
            "phase361_candidate_contains_divergence_formula": False,
            "next_generation_predictive_gate_identifiable_without_new_rule": False,
        },
        "claim_boundary": {
            "unseen_next_layer_test_completed": True,
            "strongest_baseline_comparison_completed": True,
            "temporal_values_recorded": True,
            "temporal_prediction_completed": False,
            "competition_prediction_completed": False,
            "physical_confirmation_opened": False,
            "operation_specific_mechanism_count": 0,
        },
        "next_decision": "close_phase361_candidate_route_if_no_independent_strong_baseline_survivor_else_freeze_survivors_for_new_temporal_discovery",
    }
    write_jsonl(P362 / "phase362_frozen_candidate_audit_rows.jsonl", audit_rows)
    write_jsonl(P362 / "phase362_candidate_group_errors.jsonl", group_rows)
    write_json(P362 / "phase362_frozen_candidate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
