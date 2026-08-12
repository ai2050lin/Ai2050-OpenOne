#!/usr/bin/env python3
"""Independent integrity and numerical audit for Phase1175."""

from __future__ import annotations

import argparse
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
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402
import phase1174_training_inferred_relation_event_prediction as p1174  # noqa: E402
import phase1175_late_relation_weight_decay_fork as main  # noqa: E402


OUT_ROOT = ROOT / "tests/glm5/result/phase1175_late_relation_weight_decay_fork"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"


def digest_without(payload: dict[str, Any], key: str) -> str:
    clean = dict(payload)
    clean.pop(key, None)
    return base.digest(clean)


def median(values) -> float:
    return float(np.median(list(values)))


def optimizer_weight_decay(snapshot: dict[str, Any]) -> list[float]:
    return [float(group["weight_decay"]) for group in snapshot["optimizer_state_dict"]["param_groups"]]


def run_audit(camera_sample_count: int) -> dict[str, Any]:
    protocol_path = OUT_ROOT / "protocol/preregistration.json"
    seal_path = OUT_ROOT / "runs/training/seal.json"
    training_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    holdout_path = OUT_ROOT / "runs/holdout/holdout_metrics.jsonl"
    holdout_summary_path = OUT_ROOT / "runs/holdout/summary.json"
    score_path = OUT_ROOT / "analysis/score.json"
    final_path = OUT_ROOT / "analysis/final.json"
    required = (
        protocol_path, seal_path, training_path, holdout_path,
        holdout_summary_path, score_path, final_path,
    )
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise RuntimeError(f"missing Phase1175 artifacts: {missing}")

    protocol = base.read_json(protocol_path)
    seal = base.read_json(seal_path)
    training_rows = base.read_jsonl(training_path)
    holdout_rows = base.read_jsonl(holdout_path)
    holdout_summary = base.read_json(holdout_summary_path)
    score = base.read_json(score_path)
    final = base.read_json(final_path)
    prior = base.read_json(main.P1174_FINAL)
    prior_audit = base.read_json(main.P1174_AUDIT)
    probe = base.read_json(main.MATERIAL_PROBE)

    expected_parent_rows = (
        len(main.TASK_SPECS) * main.REPLICATES * len(main.PARENT_CHECKPOINT_STEPS)
    )
    expected_arm_rows = (
        len(main.TASK_SPECS) * main.REPLICATES * len(main.ARMS)
        * len(main.ARM_CHECKPOINT_STEPS)
    )
    expected_rows = expected_parent_rows + expected_arm_rows
    expected_trajectories = len(main.TASK_SPECS) * main.REPLICATES

    old_digests = {
        p1172.quotient_signature(task.name)["digest"] for task in p1172.TASK_SPECS
    }
    old_digests.update({
        p1174.quotient_signature(task.name)["digest"] for task in p1174.TASK_SPECS
    })
    recomputed_signatures = {
        task.name: main.quotient_signature(task.name) for task in main.TASK_SPECS
    }
    manifest_by_id = {row["trajectory_id"]: row for row in protocol["manifest"]}
    key_recompute_exact = True
    for row in protocol["manifest"]:
        data = main.make_data(row["task_name"], row["seed"] + 17)
        recomputed = p1174.infer_relation_key(data)
        key_recompute_exact &= recomputed["key_digest"] == row["relation_key"]["key_digest"]

    branch_model_exact = True
    branch_optimizer_exact = True
    branch_weight_decay_exact = True
    branch_snapshot_hash_exact = True
    for trajectory_id in sorted(manifest_by_id):
        snapshots = {}
        for arm in main.ARMS:
            path = OUT_ROOT / "runs/training/branch_snapshots" / f"{trajectory_id}_{arm}.pt"
            branch_snapshot_hash_exact &= (
                base.sha256_file(path) == seal["branch_snapshot_hashes"][f"{trajectory_id}:{arm}"]
            )
            snapshots[arm] = torch.load(path, map_location="cpu", weights_only=False)
        model_digests = {
            arm: main.tensor_state_digest(snapshot["model_state_dict"])
            for arm, snapshot in snapshots.items()
        }
        optimizer_digests = {
            arm: snapshot["optimizer_moment_digest"] for arm, snapshot in snapshots.items()
        }
        branch_model_exact &= len(set(model_digests.values())) == 1
        branch_optimizer_exact &= len(set(optimizer_digests.values())) == 1
        for arm, snapshot in snapshots.items():
            branch_weight_decay_exact &= optimizer_weight_decay(snapshot) == [
                main.ARM_WEIGHT_DECAY[arm]
            ]

    training_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        training_by_trajectory.setdefault(row["trajectory_id"], []).append(row)
    step_sets_exact = True
    arm_sets_exact = True
    for trajectory_id, rows in training_by_trajectory.items():
        parent_steps = {row["step"] for row in rows if row["arm"] == "parent"}
        step_sets_exact &= parent_steps == set(main.PARENT_CHECKPOINT_STEPS)
        for arm in main.ARMS:
            arm_steps = {row["step"] for row in rows if row["arm"] == arm}
            step_sets_exact &= arm_steps == set(main.ARM_CHECKPOINT_STEPS)
        arm_sets_exact &= {row["arm"] for row in rows} == {"parent", *main.ARMS}

    checkpoint_hash_exact = True
    for row in training_rows:
        path = OUT_ROOT / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt"
        checkpoint_hash_exact &= base.sha256_file(path) == row["checkpoint_sha256"]

    norm_pair_errors = []
    for trajectory_id, rows in training_by_trajectory.items():
        continued = {
            row["step"]: row["training_only_structure"]["parameter_l2_norm"]
            for row in rows if row["arm"] == "continued_decay"
        }
        matched = {
            row["step"]: row["training_only_structure"]["parameter_l2_norm"]
            for row in rows if row["arm"] == "norm_matched_no_decay"
        }
        for step in main.ARM_CHECKPOINT_STEPS:
            norm_pair_errors.append(
                abs(continued[step] - matched[step]) / max(continued[step], 1.0e-12)
            )
    max_checkpoint_norm_error = max(norm_pair_errors)

    holdout_by_checkpoint = {row["checkpoint_id"]: row for row in holdout_rows}
    holdout_digests_exact = all(
        row["sealed_holdout_pair_digest"]
        == base.digest(main.make_data(row["task_name"], row["seed"] + 17)["holdout_x"].tolist())
        and row["sealed_holdout_label_digest"]
        == base.digest(main.make_data(row["task_name"], row["seed"] + 17)["holdout_y"].tolist())
        for row in training_rows
    )

    pair_rows = score["pair_rows"]
    branch_quiet = [row for row in pair_rows if row["branch_camera_quiet"]]
    matched = [row for row in branch_quiet if row["behavior_matched"]]
    recomputed_actual_effect = median(
        row["late_camera_effect_continued_minus_off"] for row in matched
    ) if matched else 0.0
    recomputed_random_effect = median(
        row["late_random_effect_continued_minus_off"] for row in matched
    ) if matched else 0.0
    per_task_effects = {}
    per_task_endpoint = {}
    for task_name in sorted({row["task_name"] for row in matched}):
        rows = [row for row in matched if row["task_name"] == task_name]
        per_task_effects[task_name] = median(
            row["late_camera_effect_continued_minus_off"] for row in rows
        )
        per_task_endpoint[task_name] = (
            median(row["arms"]["continued_decay"]["late_camera_score"] for row in rows)
            >= main.THRESHOLDS["camera_score_min"]
            and median(row["arms"]["continued_decay"]["late_camera_advantage"] for row in rows)
            >= main.THRESHOLDS["camera_advantage_min"]
        )
    recomputed_effect_breadth = sum(
        value >= main.THRESHOLDS["minimum_per_class_late_camera_effect"]
        for value in per_task_effects.values()
    )
    recomputed_endpoint_breadth = sum(per_task_endpoint.values())

    camera_candidates = [
        row for row in training_rows
        if row["step"] in {main.BRANCH_STEP, main.MAX_STEP}
    ]
    camera_candidates.sort(key=lambda row: (
        row["task_name"], row["replicate"], row["arm"], row["step"]
    ))
    if camera_sample_count <= 0:
        camera_sample = []
    else:
        indices = np.linspace(
            0, len(camera_candidates) - 1,
            num=min(camera_sample_count, len(camera_candidates)),
            dtype=int,
        )
        camera_sample = [camera_candidates[int(index)] for index in indices]
    device = torch.device("cuda")
    camera_errors = []
    for row in camera_sample:
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt"
        model = main.load_checkpoint(checkpoint_path, device)
        data = main.make_data(row["task_name"], row["seed"] + 17)
        relation_key = p1174.infer_relation_key(data)
        recomputed = p1174.relation_camera(
            model, data, relation_key, device, row["seed"] + 95_001
        )
        camera_errors.extend((
            abs(recomputed["actual"]["score"] - row["relation_camera"]["actual"]["score"]),
            abs(
                recomputed["random_pairing"]["score"]
                - row["relation_camera"]["random_pairing"]["score"]
            ),
            abs(recomputed["score_advantage"] - row["relation_camera"]["score_advantage"]),
        ))
        del model
    max_camera_error = max(camera_errors, default=0.0)

    checks = {
        "phase_number": protocol["phase"] == main.PHASE == 1175,
        "formal_script_hash": protocol["script_sha256"] == base.sha256_file(main.SCRIPT),
        "audit_script_hash": protocol["audit_script_sha256"] == base.sha256_file(Path(__file__)),
        "prior_final_digest": protocol["prior_phase1174_final_digest"] == prior["final_digest"],
        "prior_audit_digest": protocol["prior_phase1174_audit_digest"] == prior_audit["audit_digest"],
        "prior_primary_failed": not prior["decision"]["primary_endpoint_pass"],
        "prior_endpoint_externality_passed": prior["decision"]["free_network_endpoint_camera_externality"],
        "closed_branch_not_reopened": not protocol["separation_from_closed_branch"]["formation_prediction_reopened"],
        "nonlinear_search_forbidden": not protocol["separation_from_closed_branch"]["nonlinear_camera_search"],
        "probe_digest": protocol["material_gate"]["material_probe_digest"] == probe["digest"],
        "probe_sha256": protocol["material_gate"]["material_probe_sha256"] == base.sha256_file(main.MATERIAL_PROBE),
        "material_gate": protocol["material_gate"]["pass"],
        "task_signature_recompute": all(
            recomputed_signatures[task.name] == next(
                row["quotient_signature"] for row in protocol["task_specs"]
                if row["name"] == task.name
            ) for task in main.TASK_SPECS
        ),
        "task_signatures_unique": len({
            value["digest"] for value in recomputed_signatures.values()
        }) == len(main.TASK_SPECS),
        "task_signatures_new": all(
            value["digest"] not in old_digests for value in recomputed_signatures.values()
        ),
        "manifest_count": len(protocol["manifest"]) == expected_trajectories,
        "relation_key_recompute": key_recompute_exact,
        "all_formal_keys_eligible": all(
            row["relation_key"]["eligible_count"] == 3 for row in protocol["manifest"]
        ),
        "random_nulls_abstain": all(
            count == 0 for count in protocol["material_gate"]["random_null_eligible_counts"]
        ),
        "seal_protocol_digest": seal["protocol_digest"] == protocol["protocol_digest"],
        "seal_digest": digest_without(seal, "seal_digest") == seal["seal_digest"],
        "training_metrics_hash": seal["training_metrics_sha256"] == base.sha256_file(training_path),
        "training_row_count": len(training_rows) == expected_rows == seal["checkpoint_count"],
        "parent_row_count": sum(row["arm"] == "parent" for row in training_rows) == expected_parent_rows,
        "arm_row_count": sum(row["arm"] != "parent" for row in training_rows) == expected_arm_rows,
        "trajectory_count": len(training_by_trajectory) == expected_trajectories,
        "step_sets_exact": step_sets_exact,
        "arm_sets_exact": arm_sets_exact,
        "checkpoint_hashes_exact": checkpoint_hash_exact,
        "branch_snapshot_hashes_exact": branch_snapshot_hash_exact,
        "branch_model_identity": branch_model_exact,
        "branch_optimizer_moment_identity": branch_optimizer_exact,
        "branch_weight_decay_assignment": branch_weight_decay_exact,
        "seal_branch_identity": seal["all_branch_models_and_optimizer_moments_exact"],
        "norm_match_raw_bound": seal["maximum_norm_match_relative_error"] <= main.THRESHOLDS["maximum_norm_match_relative_error"],
        "norm_match_checkpoint_recompute": max_checkpoint_norm_error <= main.THRESHOLDS["maximum_norm_match_relative_error"],
        "holdout_absent_at_seal": seal["holdout_outcomes_absent_at_sealing"],
        "no_holdout_training": seal["no_holdout_evaluated"] and seal["no_holdout_gradient"] and seal["no_holdout_camera"],
        "strict_inductive_whitening": seal["strict_inductive_whitening"],
        "training_all_finite": seal["all_training_logits_exactly_finite"],
        "holdout_summary_seal": holdout_summary["seal_digest"] == seal["seal_digest"],
        "holdout_summary_digest": digest_without(holdout_summary, "summary_digest") == holdout_summary["summary_digest"],
        "holdout_metrics_hash": holdout_summary["holdout_metrics_sha256"] == base.sha256_file(holdout_path),
        "holdout_row_count": len(holdout_rows) == len(training_rows) == holdout_summary["row_count"],
        "holdout_checkpoint_bijection": set(holdout_by_checkpoint) == {row["checkpoint_id"] for row in training_rows},
        "holdout_digests_exact": holdout_digests_exact,
        "holdout_all_finite": holdout_summary["all_holdout_logits_exactly_finite"],
        "score_digest": digest_without(score, "score_digest") == score["score_digest"],
        "score_protocol_link": score["protocol_digest"] == protocol["protocol_digest"],
        "score_seal_link": score["seal_digest"] == seal["seal_digest"],
        "score_holdout_link": score["holdout_summary_digest"] == holdout_summary["summary_digest"],
        "pair_row_count": len(pair_rows) == expected_trajectories,
        "matched_count_recompute": len(matched) == score["behavior_matched_trajectory_count"],
        "behavior_fraction_recompute": abs(
            len(matched) / max(len(branch_quiet), 1) - score["behavior_match_fraction"]
        ) <= 1.0e-12,
        "actual_effect_recompute": abs(recomputed_actual_effect - score["median_late_camera_effect"]) <= 1.0e-12,
        "random_effect_recompute": abs(recomputed_random_effect - score["median_late_random_effect"]) <= 1.0e-12,
        "effect_breadth_recompute": recomputed_effect_breadth == score["effect_class_breadth"],
        "endpoint_breadth_recompute": recomputed_endpoint_breadth == score["continued_decay_endpoint_class_breadth"],
        "primary_decision_recompute": score["primary_endpoint_pass"] == all(score["checks"].values()),
        "final_digest": digest_without(final, "final_digest") == final["final_digest"],
        "final_score_link": final["score_digest"] == score["score_digest"],
        "final_decision_link": final["decision"]["primary_endpoint_pass"] == score["primary_endpoint_pass"],
        "auto_continue_false": not final["decision"]["auto_continue"],
        "no_unauthorized_search": (
            not final["decision"]["nonlinear_camera_search_authorized"]
            and not final["decision"]["hidden_feature_search_authorized"]
            and not final["decision"]["branch_time_or_decay_tuning_authorized"]
        ),
        "camera_sample_recompute": max_camera_error <= 1.0e-12,
    }
    passed_count = sum(checks.values())
    audit: dict[str, Any] = {
        "phase": main.PHASE,
        "audited_at_utc": base.utc_now(),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "passed_count": passed_count,
        "total_count": len(checks),
        "camera_sample_count": len(camera_sample),
        "camera_scalar_comparison_count": len(camera_errors),
        "maximum_camera_recompute_error": max_camera_error,
        "maximum_checkpoint_norm_match_relative_error": max_checkpoint_norm_error,
        "recomputed_headlines": {
            "branch_quiet_count": len(branch_quiet),
            "behavior_matched_count": len(matched),
            "median_actual_effect": recomputed_actual_effect,
            "median_random_effect": recomputed_random_effect,
            "effect_class_breadth": recomputed_effect_breadth,
            "endpoint_class_breadth": recomputed_endpoint_breadth,
        },
    }
    audit["audit_digest"] = base.digest(audit)
    return audit


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-samples", type=int, default=24)
    args = parser.parse_args()
    if AUDIT_PATH.exists():
        raise RuntimeError("Phase1175 audit already exists")
    audit = run_audit(args.camera_samples)
    base.write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "passed": audit["passed"],
        "passed_count": audit["passed_count"],
        "total_count": audit["total_count"],
        "camera_samples": audit["camera_sample_count"],
        "max_camera_error": audit["maximum_camera_recompute_error"],
        "max_norm_error": audit["maximum_checkpoint_norm_match_relative_error"],
        "audit_digest": audit["audit_digest"],
    }))


if __name__ == "__main__":
    main_cli()
