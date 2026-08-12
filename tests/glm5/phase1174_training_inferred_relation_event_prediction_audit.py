#!/usr/bin/env python3
"""Independent audit for Phase1174."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1174_training_inferred_relation_event_prediction as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()
OUT = phase.OUT_ROOT / "audit/independent_audit.json"


def independent_edges(mask: np.ndarray, contexts: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    source, target = [], []
    for background in map(int, contexts):
        for left in range(phase.MODULUS):
            shifted = (left + shift) % phase.MODULUS
            if mask[left, background] and mask[shifted, background]:
                source.append((left, background))
                target.append((shifted, background))
    return np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


@torch.inference_mode()
def independent_grid(model: torch.nn.Module, mask: np.ndarray, device: torch.device) -> np.ndarray:
    coordinates = np.argwhere(mask)
    inputs = torch.tensor(coordinates, dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = model.hidden(model.left_embedding(inputs[:, 0]) + model.right_embedding(inputs[:, 1])).square()
    result = np.full((phase.MODULUS, phase.MODULUS, hidden.shape[-1]), np.nan, dtype=np.float64)
    result[coordinates[:, 0], coordinates[:, 1]] = hidden.float().cpu().numpy().astype(np.float64)
    return result


def independent_variant(
    grid: np.ndarray,
    mask: np.ndarray,
    fit_contexts: np.ndarray,
    test_contexts: np.ndarray,
    randomized: bool,
    random_seed: int,
) -> dict[str, float]:
    fit_edges = {shift: independent_edges(mask, fit_contexts, shift) for shift in phase.RELATION_SHIFTS}
    fit_coordinates = np.unique(
        np.concatenate([np.concatenate(value, axis=0) for value in fit_edges.values()], axis=0), axis=0,
    )
    values = grid[fit_coordinates[:, 0], fit_coordinates[:, 1]]
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > max(float(eigenvalues.max()), 1.0e-12) * 1.0e-10
    whitener = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))

    def project(coordinates: np.ndarray) -> np.ndarray:
        return (grid[coordinates[:, 0], coordinates[:, 1]] - mean) @ whitener

    def fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        design = np.column_stack((source, np.ones(len(source))))
        penalty = np.eye(design.shape[1]) * phase.CAMERA_RIDGE
        penalty[-1, -1] = 0.0
        return np.linalg.solve(design.T @ design + penalty, design.T @ target)

    def apply(source: np.ndarray, operator: np.ndarray) -> np.ndarray:
        return np.column_stack((source, np.ones(len(source)))) @ operator

    operators = {}
    for shift, (source_coordinates, target_coordinates) in fit_edges.items():
        source = project(source_coordinates)
        target = project(target_coordinates)
        if randomized:
            rng = np.random.default_rng(random_seed + shift * 101)
            target = target[rng.permutation(len(target))]
        operators[shift] = fit(source, target)

    reuse_num = reuse_den = 0.0
    for shift in phase.RELATION_SHIFTS:
        source_coordinates, target_coordinates = independent_edges(mask, test_contexts, shift)
        source, target = project(source_coordinates), project(target_coordinates)
        reuse_num += float(np.sum((apply(source, operators[shift]) - target) ** 2))
        reuse_den += float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))

    closure_source, closure_target = [], []
    for background in map(int, test_contexts):
        for left in range(phase.MODULUS):
            if mask[left, background] and mask[(left + 1) % phase.MODULUS, background] and mask[(left + 3) % phase.MODULUS, background]:
                closure_source.append((left, background))
                closure_target.append(((left + 3) % phase.MODULUS, background))
    closure_source = np.asarray(closure_source, dtype=np.int64)
    closure_target = np.asarray(closure_target, dtype=np.int64)
    source, target = project(closure_source), project(closure_target)
    direct = apply(source, operators[3])
    composed = apply(apply(source, operators[1]), operators[2])
    closure_num = float(np.sum((direct - composed) ** 2) + np.sum((composed - target) ** 2))
    closure_den = float(2.0 * np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    reuse = float(np.clip(1.0 - reuse_num / max(reuse_den, 1.0e-12), 0.0, 1.0))
    closure = float(np.clip(1.0 - closure_num / max(closure_den, 1.0e-12), 0.0, 1.0))
    return {"reuse": reuse, "closure": closure, "score": float(math.sqrt(reuse * closure)), "effective_rank": int(keep.sum())}


def recompute_camera(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = phase.make_data(row["task_name"], row["seed"] + 17)
    key = phase.infer_relation_key(data)
    if not key["all_closure_relations_eligible"]:
        return {"status": "NoEligibleRelation", "actual": {"score": 0.0}, "random_pairing": {"score": 0.0}}
    path = phase.OUT_ROOT / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt"
    model = phase.load_checkpoint(path, device)
    grid = independent_grid(model, data["train_mask"], device)
    actual = independent_variant(
        grid, data["train_mask"], data["contexts"]["fit"], data["contexts"]["test"], False, row["seed"] + 90_001,
    )
    random_pairing = independent_variant(
        grid, data["train_mask"], data["contexts"]["fit"], data["contexts"]["test"], True, row["seed"] + 90_001,
    )
    del model
    return {"status": "EligibleRelation", "actual": actual, "random_pairing": random_pairing}


def exact_digest_without(value: dict[str, Any], field: str) -> str:
    return base.digest({key: item for key, item in value.items() if key != field})


def main() -> None:
    protocol = base.read_json(phase.OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(phase.OUT_ROOT / "runs/training/seal.json")
    predictor = base.read_json(phase.OUT_ROOT / "analysis/predictor_seal.json")
    score = base.read_json(phase.OUT_ROOT / "analysis/score.json")
    final = base.read_json(phase.OUT_ROOT / "analysis/final.json")
    prior = base.read_json(phase.P1173_FINAL)
    prior_audit = base.read_json(phase.P1173_AUDIT)
    training_rows = base.read_jsonl(phase.OUT_ROOT / "runs/training/training_metrics.jsonl")
    discovery_summary = base.read_json(phase.OUT_ROOT / "runs/holdout/discovery/summary.json")
    confirmation_exists = (phase.OUT_ROOT / "runs/holdout/confirmation/summary.json").exists()
    confirmation_summary = base.read_json(phase.OUT_ROOT / "runs/holdout/confirmation/summary.json") if confirmation_exists else None

    checks: dict[str, bool] = {}
    checks["phase_identity"] = protocol["phase"] == seal["phase"] == predictor["phase"] == score["phase"] == final["phase"] == phase.PHASE
    checks["protocol_script_hash"] = protocol["script_sha256"] == base.sha256_file(phase.SCRIPT)
    checks["protocol_audit_hash"] = protocol["audit_script_sha256"] == base.sha256_file(SCRIPT)
    checks["protocol_digest"] = protocol["protocol_digest"] == exact_digest_without(protocol, "protocol_digest")
    checks["prior_calibration"] = prior["relation_closure_camera_calibrated"] and prior["confirmation_passed"]
    checks["prior_audit"] = prior_audit["passed"] and prior_audit["passed_count"] == 34
    checks["prior_digests"] = protocol["prior_phase1173_final_digest"] == prior["final_digest"] and protocol["prior_phase1173_audit_digest"] == prior_audit["audit_digest"]
    checks["material_gate"] = protocol["material_gate"]["pass"] and all(protocol["material_gate"]["checks"].values())
    checks["task_count"] = protocol["task_count"] == 12 and len(protocol["task_specs"]) == 12
    checks["trajectory_count"] = protocol["trajectory_count"] == 96 and len(protocol["manifest"]) == 96
    signature_digests = [task["quotient_signature"]["digest"] for task in protocol["task_specs"]]
    checks["quotient_unique"] = len(set(signature_digests)) == 12
    checks["quotient_new"] = set(signature_digests).isdisjoint(protocol["material_gate"]["phase1172_signatures"].values())
    checks["random_null_abstention"] = protocol["material_gate"]["random_null_eligible_counts"] == [0] * 16

    manifest_keys_ok = True
    context_partition_ok = True
    for item in protocol["manifest"]:
        data = phase.make_data(item["task_name"], item["seed"] + 17)
        key = phase.infer_relation_key(data)
        manifest_keys_ok &= key["key_digest"] == item["relation_key"]["key_digest"]
        expected = 3 if item["relation_expected"] else 0
        manifest_keys_ok &= key["eligible_count"] == expected
        contexts = data["contexts"]
        context_partition_ok &= len(set(contexts["key"]).intersection(contexts["fit"])) == 0
        context_partition_ok &= len(set(contexts["key"]).intersection(contexts["test"])) == 0
        context_partition_ok &= len(set(contexts["fit"]).intersection(contexts["test"])) == 0
        context_partition_ok &= len(contexts["key"]) + len(contexts["fit"]) + len(contexts["test"]) == phase.MODULUS
    checks["training_only_relation_keys_recompute"] = bool(manifest_keys_ok)
    checks["background_partitions_disjoint"] = bool(context_partition_ok)
    checks["key_declares_no_leakage"] = all(
        not item["relation_key"]["uses_holdout_inputs"]
        and not item["relation_key"]["uses_holdout_labels"]
        and not item["relation_key"]["uses_task_name_or_formula"]
        and not item["relation_key"]["uses_future_generalization"]
        for item in protocol["manifest"]
    )

    checks["seal_protocol_link"] = seal["protocol_digest"] == protocol["protocol_digest"]
    checks["training_row_count"] = len(training_rows) == 96 * len(phase.CHECKPOINT_STEPS) == seal["checkpoint_count"]
    checks["training_trajectory_count"] = len({row["trajectory_id"] for row in training_rows}) == seal["trajectory_count"] == 96
    checks["training_metrics_hash"] = seal["training_metrics_sha256"] == base.sha256_file(phase.OUT_ROOT / "runs/training/training_metrics.jsonl")
    checks["no_holdout_during_training"] = seal["no_holdout_evaluated"] and seal["no_holdout_gradient"] and seal["no_holdout_camera"]
    checks["strict_inductive_whitening"] = seal["strict_inductive_whitening"] and all(
        row["relation_camera"]["actual"]["whitening_fit_background_only"]
        and not row["relation_camera"]["actual"]["test_background_used_for_whitening"]
        for row in training_rows
    )
    checks["training_finite"] = seal["all_training_logits_exactly_finite"] and all(row["train"]["exact_all_finite"] for row in training_rows)
    checks["all_checkpoint_hashes"] = all(
        base.sha256_file(phase.OUT_ROOT / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt") == row["checkpoint_sha256"] == seal["checkpoint_hashes"][row["checkpoint_id"]]
        for row in training_rows
    )
    checks["discovery_summary_hash"] = discovery_summary["holdout_metrics_sha256"] == base.sha256_file(phase.OUT_ROOT / "runs/holdout/discovery/holdout_metrics.jsonl")
    checks["discovery_summary_digest"] = discovery_summary["summary_digest"] == exact_digest_without(discovery_summary, "summary_digest")
    checks["predictor_discovery_link"] = predictor["discovery_summary_digest"] == discovery_summary["summary_digest"]
    checks["predictor_digest"] = predictor["predictor_digest"] == exact_digest_without(predictor, "predictor_digest")
    checks["confirmation_seal_order"] = predictor["confirmation_absent_at_predictor_seal"] and (
        not confirmation_exists or predictor["sealed_at_utc"] <= confirmation_summary["evaluated_at_utc"]
    )
    checks["confirmation_authorization_consistent"] = confirmation_exists == predictor["confirmation_reveal_authorized"]
    checks["feature_cutoff_frozen"] = predictor["prediction_cutoff"] == phase.PREDICTION_CUTOFF == 250
    checks["feature_models_complete"] = (
        not predictor["confirmation_reveal_authorized"]
        or set(predictor["feature_names"]) == set(phase.MODEL_NAMES)
    )

    selected = []
    for task in phase.TASK_SPECS:
        task_rows = [row for row in training_rows if row["task_name"] == task.name and row["replicate"] == 0 and row["step"] in {phase.PREDICTION_CUTOFF, max(phase.CHECKPOINT_STEPS)}]
        selected.extend(task_rows)
    device = torch.device("cuda")
    camera_diffs = []
    statuses_match = True
    for row in selected:
        recomputed = recompute_camera(row, device)
        statuses_match &= recomputed["status"] == row["relation_camera"]["status"]
        for variant in ("actual", "random_pairing"):
            camera_diffs.append(abs(recomputed[variant]["score"] - row["relation_camera"][variant]["score"]))
    checks["independent_camera_status"] = bool(statuses_match)
    checks["independent_camera_numeric"] = max(camera_diffs, default=0.0) <= 1.0e-9

    checks["score_digest"] = score["score_digest"] == exact_digest_without(score, "score_digest")
    if confirmation_exists:
        checks["confirmation_summary_hash"] = confirmation_summary["holdout_metrics_sha256"] == base.sha256_file(phase.OUT_ROOT / "runs/holdout/confirmation/holdout_metrics.jsonl")
        trajectories, groups = phase.grouped_trajectories("confirmation")
        labels = np.asarray([phase.event_labels(row) for row in trajectories], dtype=np.float64)
        recomputed_scores = {}
        constant = np.tile(np.asarray(predictor["predictors"]["constant_probability"]), (len(labels), 1))
        recomputed_scores["constant"] = phase.brier(labels, constant)
        for model_name in phase.MODEL_NAMES:
            features = np.asarray([phase.feature_vector(groups[row["trajectory_id"]], model_name)[1] for row in trajectories])
            probability = phase.apply_ridge(predictor["predictors"][model_name], features)
            recomputed_scores[model_name] = phase.brier(labels, probability)
        checks["confirmation_brier_recompute"] = max(
            abs(recomputed_scores[name] - score["confirmation_scores"][name]) for name in recomputed_scores
        ) <= 1.0e-12
        controls = [name for name in recomputed_scores if name != "relation"]
        best_name = min(controls, key=lambda name: recomputed_scores[name])
        checks["best_control_recompute"] = best_name == score["best_control_name"]
        best = recomputed_scores[best_name]
        improvement = (best - recomputed_scores["relation"]) / best if best > 0 else 0.0
        checks["relative_improvement_recompute"] = abs(improvement - score["relative_brier_improvement"]) <= 1.0e-12
        expected_primary = all(score["endpoint_checks"].values())
        checks["primary_endpoint_logic"] = score["primary_endpoint_pass"] == expected_primary
    else:
        checks["confirmation_absent_after_object_failure"] = not predictor["object_decision"]["pass"] and score["stage"] == "discovery_object_gate_failure"
        checks["primary_endpoint_logic"] = not score["primary_endpoint_pass"]

    checks["final_digest"] = final["final_digest"] == exact_digest_without(final, "final_digest")
    checks["final_links"] = final["protocol_digest"] == protocol["protocol_digest"] and final["seal_digest"] == seal["seal_digest"] and final["predictor_digest"] == predictor["predictor_digest"] and final["score_digest"] == score["score_digest"]
    checks["final_decision_matches_score"] = final["decision"]["primary_endpoint_pass"] == score["primary_endpoint_pass"]
    checks["no_nonlinear_escalation"] = not final["decision"]["nonlinear_camera_search_authorized"] and not final["decision"]["hidden_feature_search_authorized"]
    checks["causal_authorization_logic"] = final["decision"]["causal_use_authorized"] == score["primary_endpoint_pass"]
    checks["auto_continue_logic"] = final["decision"]["auto_continue"] == score["primary_endpoint_pass"]

    audit = {
        "phase": phase.PHASE,
        "audit": "independent Phase1174 recomputation and leakage audit",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "failed_count": sum(not value for value in checks.values()),
        "passed": bool(all(checks.values())),
        "independent_camera_sample_count": len(selected),
        "independent_camera_max_abs_score_error": max(camera_diffs, default=0.0),
    }
    audit["audit_digest"] = base.digest(audit)
    base.write_json(OUT, audit)
    print(json.dumps({
        "passed": audit["passed"],
        "passed_count": audit["passed_count"],
        "check_count": audit["check_count"],
        "failed": [name for name, value in checks.items() if not value],
        "audit_digest": audit["audit_digest"],
    }))


if __name__ == "__main__":
    main()
