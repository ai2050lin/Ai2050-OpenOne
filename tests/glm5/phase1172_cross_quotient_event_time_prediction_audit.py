#!/usr/bin/env python3
"""Independent integrity and endpoint audit for Phase1172."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402


OUT = p1172.OUT_ROOT / "audit/independent_audit.json"


def canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


def recompute_predictions(predictor: dict[str, Any], split: str) -> dict[str, Any]:
    trajectories, training_groups = p1172.grouped_trajectories(split)
    baseline_x, augmented_x, labels, task_names = [], [], [], []
    for trajectory in trajectories:
        baseline_names, baseline_values = p1172.feature_vector(training_groups[trajectory["trajectory_id"]], augmented=False)
        augmented_names, augmented_values = p1172.feature_vector(training_groups[trajectory["trajectory_id"]], augmented=True)
        if tuple(baseline_names) != p1172.BASELINE_FEATURE_NAMES:
            raise RuntimeError("baseline feature order mismatch")
        if tuple(augmented_names) != p1172.BASELINE_FEATURE_NAMES + p1172.STRUCTURE_FEATURE_NAMES:
            raise RuntimeError("augmented feature order mismatch")
        baseline_x.append(baseline_values)
        augmented_x.append(augmented_values)
        labels.append(p1172.event_labels(trajectory))
        task_names.append(trajectory["task_name"])
    baseline_array = np.asarray(baseline_x, dtype=np.float64)
    augmented_array = np.asarray(augmented_x, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    constant = np.tile(np.asarray(predictor["predictors"]["constant_probability"]), (len(label_array), 1))
    baseline = p1172.apply_ridge(predictor["predictors"]["baseline_ridge"], baseline_array)
    augmented = p1172.apply_ridge(predictor["predictors"]["augmented_ridge"], augmented_array)
    scores = {
        "constant_brier": p1172.brier(label_array, constant),
        "baseline_ridge_brier": p1172.brier(label_array, baseline),
        "augmented_ridge_brier": p1172.brier(label_array, augmented),
    }
    per_task = []
    for task_name in sorted(set(task_names)):
        mask = np.asarray([name == task_name for name in task_names], dtype=bool)
        row = {
            "task_name": task_name,
            "constant_brier": p1172.brier(label_array[mask], constant[mask]),
            "baseline_ridge_brier": p1172.brier(label_array[mask], baseline[mask]),
            "augmented_ridge_brier": p1172.brier(label_array[mask], augmented[mask]),
        }
        row["augmented_beats_both"] = row["augmented_ridge_brier"] < min(row["constant_brier"], row["baseline_ridge_brier"])
        per_task.append(row)
    return {"trajectories": trajectories, "scores": scores, "per_task": per_task}


def main() -> None:
    protocol = base.read_json(p1172.OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(p1172.OUT_ROOT / "runs/training/seal.json")
    training_rows = base.read_jsonl(p1172.OUT_ROOT / "runs/training/training_metrics.jsonl")
    predictor = base.read_json(p1172.OUT_ROOT / "analysis/predictor_seal.json")
    score = base.read_json(p1172.OUT_ROOT / "analysis/score.json")
    final = base.read_json(p1172.OUT_ROOT / "analysis/final.json")
    protocol_without_digest = dict(protocol)
    protocol_digest = protocol_without_digest.pop("protocol_digest")
    signatures = {task.name: p1172.quotient_signature(task.name) for task in p1172.TASK_SPECS}
    checks: dict[str, bool] = {}
    checks["protocol_digest"] = base.digest(protocol_without_digest) == protocol_digest
    checks["primary_script_hash"] = base.sha256_file(p1172.SCRIPT) == protocol["script_sha256"]
    checks["audit_script_hash"] = base.sha256_file(Path(__file__).resolve()) == protocol["audit_script_sha256"]
    checks["prior_state"] = not base.read_json(p1172.P1171_FINAL)["decision"]["auto_continue"]
    checks["pilot_excluded"] = protocol["pilot_excluded_from_evidence"] and base.read_json(p1172.PILOT_RESULT)["formal_evidence"] is False
    checks["task_count"] = len(protocol["tasks"]) == 12 == len(p1172.TASK_SPECS)
    checks["task_split"] = len(protocol["discovery_task_names"]) == 8 and len(protocol["confirmation_task_names"]) == 4
    checks["quotient_signature_recompute"] = all(canonical(task["quotient_signature"]) == canonical(signatures[task["name"]]) for task in protocol["tasks"])
    checks["quotient_signatures_unique"] = len({value["digest"] for value in signatures.values()}) == 12
    checks["fixed_dimensions"] = protocol["modulus"] == 61 and protocol["model_width"] == 128 and protocol["parameter_count"] == 39808
    checks["trajectory_count"] = protocol["trajectory_count"] == 96 and seal["trajectory_count"] == 96
    checks["training_row_count"] = len(training_rows) == 96 * len(p1172.CHECKPOINT_STEPS) == seal["checkpoint_count"]
    trajectory_counts = Counter(row["trajectory_id"] for row in training_rows)
    checks["trajectory_checkpoint_counts"] = len(trajectory_counts) == 96 and set(trajectory_counts.values()) == {len(p1172.CHECKPOINT_STEPS)}
    checks["checkpoint_steps"] = all(sorted(row["step"] for row in training_rows if row["trajectory_id"] == trajectory_id) == list(p1172.CHECKPOINT_STEPS) for trajectory_id in trajectory_counts)
    checks["training_metrics_hash"] = base.sha256_file(p1172.OUT_ROOT / "runs/training/training_metrics.jsonl") == seal["training_metrics_sha256"]
    checks["checkpoint_hashes"] = len(seal["checkpoint_hashes"]) == len(training_rows) and all(base.sha256_file(p1172.OUT_ROOT / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt") == row["checkpoint_sha256"] == seal["checkpoint_hashes"][row["checkpoint_id"]] for row in training_rows)
    checks["sealed_before_holdout"] = seal["holdout_outcomes_absent_at_sealing"] and seal["training_sealed"]
    checks["no_holdout_training_eval"] = seal["no_holdout_evaluated"] and all(not row["holdout_evaluated_during_training"] for row in training_rows)
    checks["no_holdout_gradient"] = seal["no_holdout_gradient"] and all(not row["holdout_used_by_gradient"] for row in training_rows)
    checks["exact_training_finite"] = seal["all_training_logits_exactly_finite"] and all(row["train"]["exact_all_finite"] for row in training_rows)
    manifest = {row["trajectory_id"]: row for row in protocol["manifest"]}
    checks["allocation_manifest"] = all(row["trajectory_id"] in manifest and row["task_split"] == manifest[row["trajectory_id"]]["task_split"] and row["train_pair_digest"] == manifest[row["trajectory_id"]]["train_pair_digest"] and row["train_label_digest"] == manifest[row["trajectory_id"]]["train_label_digest"] for row in training_rows)
    discovery_summary = base.read_json(p1172.OUT_ROOT / "runs/holdout/discovery/summary.json")
    discovery_rows = base.read_jsonl(p1172.OUT_ROOT / "runs/holdout/discovery/holdout_metrics.jsonl")
    checks["discovery_row_count"] = len(discovery_rows) == 8 * 8 * len(p1172.CHECKPOINT_STEPS) == discovery_summary["row_count"]
    checks["discovery_only"] = {row["task_name"] for row in discovery_rows} == set(protocol["discovery_task_names"])
    checks["discovery_hash"] = base.sha256_file(p1172.OUT_ROOT / "runs/holdout/discovery/holdout_metrics.jsonl") == discovery_summary["holdout_metrics_sha256"]
    checks["discovery_exact_finite"] = discovery_summary["all_holdout_logits_exactly_finite"] and all(row["holdout"]["exact_all_finite"] for row in discovery_rows)
    discovery_trajectories, discovery_training = p1172.grouped_trajectories("discovery")
    object_decision = p1172.discovery_object_decision(discovery_trajectories)
    checks["object_decision_recompute"] = canonical(object_decision) == canonical(predictor["object_decision"])
    checks["confirmation_absent_at_predictor_seal"] = predictor["confirmation_absent_at_predictor_seal"]
    if predictor["confirmation_reveal_authorized"]:
        baseline_x, augmented_x, labels = [], [], []
        for trajectory in discovery_trajectories:
            _, baseline_values = p1172.feature_vector(discovery_training[trajectory["trajectory_id"]], augmented=False)
            _, augmented_values = p1172.feature_vector(discovery_training[trajectory["trajectory_id"]], augmented=True)
            baseline_x.append(baseline_values)
            augmented_x.append(augmented_values)
            labels.append(p1172.event_labels(trajectory))
        baseline_model = p1172.fit_ridge(np.asarray(baseline_x), np.asarray(labels))
        augmented_model = p1172.fit_ridge(np.asarray(augmented_x), np.asarray(labels))
        checks["predictor_refit"] = canonical(baseline_model) == canonical(predictor["predictors"]["baseline_ridge"]) and canonical(augmented_model) == canonical(predictor["predictors"]["augmented_ridge"])
        confirmation_summary = base.read_json(p1172.OUT_ROOT / "runs/holdout/confirmation/summary.json")
        confirmation_rows = base.read_jsonl(p1172.OUT_ROOT / "runs/holdout/confirmation/holdout_metrics.jsonl")
        checks["confirmation_row_count"] = len(confirmation_rows) == 4 * 8 * len(p1172.CHECKPOINT_STEPS) == confirmation_summary["row_count"]
        checks["confirmation_only"] = {row["task_name"] for row in confirmation_rows} == set(protocol["confirmation_task_names"])
        checks["confirmation_hash"] = base.sha256_file(p1172.OUT_ROOT / "runs/holdout/confirmation/holdout_metrics.jsonl") == confirmation_summary["holdout_metrics_sha256"]
        checks["confirmation_exact_finite"] = confirmation_summary["all_holdout_logits_exactly_finite"] and all(row["holdout"]["exact_all_finite"] for row in confirmation_rows)
        recomputed = recompute_predictions(predictor, "confirmation")
        checks["confirmation_score_recompute"] = canonical(recomputed["scores"]) == canonical(score["confirmation_scores"])
        checks["per_task_score_recompute"] = canonical(recomputed["per_task"]) == canonical(score["per_confirmation_task"])
        best = min(recomputed["scores"]["constant_brier"], recomputed["scores"]["baseline_ridge_brier"])
        improvement = (best - recomputed["scores"]["augmented_ridge_brier"]) / best if best > 0 else 0.0
        breadth = sum(row["augmented_beats_both"] for row in recomputed["per_task"])
        endpoint = all(row["fit_step"] is not None and row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in recomputed["trajectories"]) and improvement >= p1172.THRESHOLDS["confirmation_relative_brier_improvement_min"] and breadth >= p1172.THRESHOLDS["confirmation_class_advantage_min"]
        checks["endpoint_recompute"] = endpoint == score["primary_endpoint_pass"]
    else:
        checks["predictor_refit"] = predictor["predictors"] is None
        checks["confirmation_not_revealed"] = not (p1172.OUT_ROOT / "runs/holdout/confirmation").exists()
        checks["endpoint_recompute"] = score["stage"] == "discovery_object_gate_failure" and not score["primary_endpoint_pass"]
    checks["predictor_digest"] = base.digest({key: value for key, value in predictor.items() if key != "predictor_digest"}) == predictor["predictor_digest"]
    checks["score_digest"] = base.digest({key: value for key, value in score.items() if key != "score_digest"}) == score["score_digest"]
    checks["final_consistency"] = final["protocol_digest"] == protocol["protocol_digest"] and final["seal_digest"] == seal["seal_digest"] and final["predictor_digest"] == predictor["predictor_digest"] and final["score_digest"] == score["score_digest"]
    checks["continuation_consistency"] = final["decision"]["auto_continue"] == score["primary_endpoint_pass"]
    checks["hidden_scan_denied"] = not final["decision"]["hidden_scan_authorized"]
    checks["causal_intervention_denied"] = not final["decision"]["causal_intervention_authorized"]
    checks["feature_search_denied"] = not final["decision"]["feature_search_authorized"]
    report = {
        "phase": p1172.PHASE,
        "audited_at_utc": base.utc_now(),
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "overall_pass": all(checks.values()),
        "object_gate_pass": predictor["object_decision"]["pass"],
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "scope": "Recomputes quotient separation, allocation, seals, event labels, frozen predictors, confirmation scores, endpoint, and continuation. It does not establish causal formation or language external validity.",
    }
    report["audit_digest"] = base.digest(report)
    base.write_json(OUT, report)
    print(json.dumps({"passed": report["passed"], "total": report["total"], "overall_pass": report["overall_pass"], "audit_digest": report["audit_digest"]}))


if __name__ == "__main__":
    main()
