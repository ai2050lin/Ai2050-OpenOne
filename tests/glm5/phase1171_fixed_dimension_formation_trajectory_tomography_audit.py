#!/usr/bin/env python3
"""Independent recomputation audit for Phase1171."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as phase  # noqa: E402


def main() -> None:
    root = phase.OUT_ROOT
    protocol = phase.base.read_json(root / "protocol/preregistration.json")
    seal = phase.base.read_json(root / "runs/training/seal.json")
    holdout_summary = phase.base.read_json(root / "runs/holdout/summary.json")
    training = phase.base.read_jsonl(root / "runs/training/training_metrics.jsonl")
    holdout = phase.base.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    score = phase.base.read_json(root / "analysis/score.json")
    final = phase.base.read_json(root / "analysis/final.json")

    independent_sample = tuple(random.Random(phase.TASK_SELECTION_SEED).sample(phase.eligible_operations(), 12))
    grouped: dict[str, list[dict]] = {}
    for row in holdout:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    recomputed_trajectories = [phase.trajectory_summary(rows) for rows in grouped.values()]
    recomputed_decision = phase.endpoint_decision(recomputed_trajectories)
    checkpoint_hashes = {
        row["checkpoint_id"]: phase.base.sha256_file(root / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt")
        for row in training
    }
    allocation_lookup = {(row["task_name"], row["replicate"]): row for row in protocol["allocation"]}
    allocation_matches = all(
        row["seed"] == allocation_lookup[(row["task_name"], row["replicate"])]["seed"]
        and row["train_pair_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["train_pair_digest"]
        and row["sealed_holdout_pair_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["holdout_pair_digest"]
        and row["train_label_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["train_label_digest"]
        and row["sealed_holdout_label_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["sealed_holdout_label_digest"]
        for row in training
    )
    operation_parameter_counts = set()
    for operation in phase.FORMAL_OPERATIONS:
        model = phase.RoleSquareNetwork(phase.RoleSquareConfig())
        operation_parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
    prior_audit = phase.base.read_json(phase.P1170_AUDIT)
    prior_finite = phase.base.read_json(phase.P1170_FINITE_AUDIT)
    checks = {
        "protocol_digest": protocol["protocol_digest"] == phase.base.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "primary_script_hash": protocol["source_hashes"]["primary_script"] == phase.base.sha256_file(phase.SCRIPT),
        "audit_script_hash": protocol["source_hashes"]["audit_script"] == phase.base.sha256_file(phase.AUDIT_SCRIPT),
        "prior_hashes": (
            protocol["prerequisite"]["phase1170_final_sha256"] == phase.base.sha256_file(phase.P1170_FINAL)
            and protocol["prerequisite"]["phase1170_audit_sha256"] == phase.base.sha256_file(phase.P1170_AUDIT)
            and protocol["prerequisite"]["phase1170_exact_finite_audit_sha256"] == phase.base.sha256_file(phase.P1170_FINITE_AUDIT)
        ),
        "prior_state": prior_audit["passed"] == 29 and prior_audit["total"] == 30 and prior_finite["overall_pass"],
        "pilot_files_hashed": all(
            protocol["engineering_calibration_excluded_from_evidence"][key] == phase.base.sha256_file(path)
            for key, path in (
                ("p31_script_sha256", phase.PILOT31_SCRIPT),
                ("p31_result_sha256", phase.PILOT31_RESULT),
                ("p61_script_sha256", phase.PILOT61_SCRIPT),
                ("p61_result_sha256", phase.PILOT61_RESULT),
            )
        ),
        "operation_sample": tuple(tuple(row) for row in protocol["task_selection"]["sampled_operations"]) == independent_sample,
        "formal_operations": tuple(tuple(row) for row in protocol["task_selection"]["formal_operations"]) == independent_sample[: phase.FORMAL_TASK_COUNT],
        "reserved_operations": tuple(tuple(row) for row in protocol["task_selection"]["reserved_fresh_operations"]) == independent_sample[phase.FORMAL_TASK_COUNT :],
        "operation_disjointness": not set(phase.FORMAL_OPERATIONS).intersection(phase.RESERVED_OPERATIONS) and phase.PILOT_OPERATION not in independent_sample,
        "fixed_dimensions": len(operation_parameter_counts) == 1 and protocol["model"]["parameter_count_fixed_across_tasks"] in operation_parameter_counts,
        "training_row_count": len(training) == len(phase.TASKS) * phase.REPLICATES * len(phase.CHECKPOINT_STEPS),
        "holdout_row_count": len(holdout) == len(training),
        "trajectory_count": len(grouped) == len(phase.TASKS) * phase.REPLICATES,
        "task_replicate_counts": all(sum(row["task_name"] == task_name for row in recomputed_trajectories) == phase.REPLICATES for task_name in phase.TASKS),
        "checkpoint_steps": all(tuple(sorted(row["step"] for row in rows)) == phase.CHECKPOINT_STEPS for rows in grouped.values()),
        "allocation_matches": allocation_matches,
        "checkpoint_hashes": checkpoint_hashes == seal["checkpoint_hashes"],
        "training_metrics_hash": seal["training_metrics_sha256"] == phase.base.sha256_file(root / "runs/training/training_metrics.jsonl"),
        "holdout_metrics_hash": holdout_summary["holdout_metrics_sha256"] == phase.base.sha256_file(root / "runs/holdout/holdout_metrics.jsonl"),
        "sealed_before_holdout": seal["training_sealed"] and seal["holdout_outcomes_absent_at_sealing"],
        "no_holdout_training_eval": seal["no_holdout_evaluated"] and all(not row["holdout_evaluated_during_training"] for row in training),
        "no_holdout_gradient": seal["no_holdout_gradient"] and all(not row["holdout_used_by_gradient"] for row in training),
        "exact_training_finite": seal["all_training_logits_exactly_finite"] and all(row["train"]["exact_all_finite"] for row in training),
        "exact_holdout_finite": holdout_summary["all_holdout_logits_exactly_finite"] and all(row["holdout"]["exact_all_finite"] for row in holdout),
        "reserved_operations_not_evaluated": not set(tuple(row["operation"]) for row in holdout).intersection(phase.RESERVED_OPERATIONS),
        "trajectory_recompute": sorted(recomputed_trajectories, key=lambda row: row["trajectory_id"]) == sorted(score["trajectories"], key=lambda row: row["trajectory_id"]),
        "endpoint_recompute": all(recomputed_decision[key] == score[key] for key in recomputed_decision),
        "final_consistency": final["decision"]["primary_endpoint_pass"] == recomputed_decision["primary_endpoint_pass"],
        "continuation_consistency": final["decision"]["prospective_predictor_phase_authorized"] == recomputed_decision["primary_endpoint_pass"],
        "hidden_scan_denied": final["decision"]["hidden_scan_authorized"] is False,
        "causal_intervention_denied": final["decision"]["causal_intervention_authorized"] is False,
        "operation_search_denied": final["decision"]["operation_search_authorized"] is False,
    }
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": phase.base.utc_now(),
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "overall_pass": all(checks.values()),
        "recomputed_global_regime_counts": recomputed_decision["global_regime_counts"],
        "recomputed_mixed_task_count": recomputed_decision["mixed_task_count"],
        "recomputed_primary_endpoint_pass": recomputed_decision["primary_endpoint_pass"],
        "scope": "Recomputes fixed allocation, exact finite status, sealing, event-time regimes, breadth endpoint, and claim scope. It does not fit or validate a process predictor.",
    }
    audit["audit_digest"] = phase.base.digest(audit)
    phase.base.write_json(root / "audit/independent_audit.json", audit)
    print(json.dumps({"overall_pass": audit["overall_pass"], "passed": audit["passed"], "total": audit["total"], "audit_digest": audit["audit_digest"]}))
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
