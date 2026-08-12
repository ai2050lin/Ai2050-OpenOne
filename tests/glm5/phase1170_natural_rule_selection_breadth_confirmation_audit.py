#!/usr/bin/env python3
"""Independent recomputation audit for Phase1170."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1170_natural_rule_selection_breadth_confirmation as phase  # noqa: E402


def main() -> None:
    root = phase.OUT_ROOT
    protocol = phase.base.read_json(root / "protocol/preregistration.json")
    seal = phase.base.read_json(root / "runs/training/seal.json")
    holdout_summary = phase.base.read_json(root / "runs/holdout/summary.json")
    training = phase.base.read_jsonl(root / "runs/training/training_metrics.jsonl")
    holdout = phase.base.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    score = phase.base.read_json(root / "analysis/score.json")
    final = phase.base.read_json(root / "analysis/final.json")
    prior_final = phase.base.read_json(phase.P1169_FINAL)
    prior_audit = phase.base.read_json(phase.P1169_AUDIT)

    independent_permutation = tuple(random.Random(phase.TASK_SELECTION_SEED).sample(phase.PRIME_POOL, len(phase.PRIME_POOL)))
    grouped: dict[str, list[dict]] = {}
    for row in holdout:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    recomputed_trajectories = [phase.trajectory_summary(rows) for rows in grouped.values()]
    recomputed_decision = phase.breadth_decision(recomputed_trajectories)
    checkpoint_hashes = {
        row["checkpoint_id"]: phase.base.sha256_file(root / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt")
        for row in training
    }
    allocation_lookup = {
        (row["task_name"], row["replicate"]): row
        for row in protocol["allocation"]
    }
    allocation_matches = all(
        row["seed"] == allocation_lookup[(row["task_name"], row["replicate"])]["seed"]
        and row["train_pair_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["train_pair_digest"]
        and row["sealed_holdout_pair_digest"] == allocation_lookup[(row["task_name"], row["replicate"])]["sealed_holdout_pair_digest"]
        for row in training
    )
    checks = {
        "protocol_digest": protocol["protocol_digest"] == phase.base.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "primary_script_hash": protocol["source_hashes"]["primary_script"] == phase.base.sha256_file(phase.SCRIPT),
        "audit_script_hash": protocol["source_hashes"]["audit_script"] == phase.base.sha256_file(phase.AUDIT_SCRIPT),
        "prior_final_hash": protocol["prerequisite"]["phase1169_final_sha256"] == phase.base.sha256_file(phase.P1169_FINAL),
        "prior_audit_hash": protocol["prerequisite"]["phase1169_audit_sha256"] == phase.base.sha256_file(phase.P1169_AUDIT),
        "prior_state": prior_final["decision"]["primary_endpoint_pass"] is False and prior_audit["overall_pass"] is True,
        "random_permutation": tuple(protocol["task_selection"]["random_permutation"]) == independent_permutation,
        "selected_allocation": tuple(protocol["task_selection"]["selected_moduli"]) == independent_permutation[: phase.TASK_COUNT],
        "reserved_allocation": tuple(protocol["task_selection"]["reserved_fresh_moduli"]) == independent_permutation[phase.TASK_COUNT :],
        "allocation_disjoint": not set(protocol["task_selection"]["selected_moduli"]).intersection(protocol["task_selection"]["reserved_fresh_moduli"]),
        "phase1169_regime_unchanged": (
            tuple(protocol["checkpoint_steps"]) == phase.base.CHECKPOINT_STEPS
            and protocol["train_fraction"] == phase.base.TRAIN_FRACTION
            and protocol["model"]["width"] == phase.base.MODEL_WIDTH
            and protocol["training"] == phase.base.TRAINING
            and protocol["trajectory_thresholds"] == phase.TRAJECTORY_THRESHOLDS
        ),
        "training_row_count": len(training) == len(phase.TASKS) * phase.REPLICATES * len(phase.CHECKPOINT_STEPS),
        "holdout_row_count": len(holdout) == len(training),
        "trajectory_count": len(grouped) == len(phase.TASKS) * phase.REPLICATES,
        "checkpoint_steps": all(tuple(sorted(row["step"] for row in rows)) == phase.CHECKPOINT_STEPS for rows in grouped.values()),
        "allocation_matches": allocation_matches,
        "checkpoint_hashes": checkpoint_hashes == seal["checkpoint_hashes"],
        "training_metrics_hash": seal["training_metrics_sha256"] == phase.base.sha256_file(root / "runs/training/training_metrics.jsonl"),
        "holdout_metrics_hash": holdout_summary["holdout_metrics_sha256"] == phase.base.sha256_file(root / "runs/holdout/holdout_metrics.jsonl"),
        "sealed_before_holdout": seal["training_sealed"] and seal["holdout_outcomes_absent_at_sealing"],
        "no_holdout_training_eval": seal["no_holdout_evaluated"] and all(not row["holdout_evaluated_during_training"] for row in training),
        "no_holdout_gradient": seal["no_holdout_gradient"] and all(not row["holdout_used_by_gradient"] for row in training),
        "finite": holdout_summary["finite"],
        "trajectory_recompute": sorted(recomputed_trajectories, key=lambda row: row["trajectory_id"]) == sorted(score["trajectories"], key=lambda row: row["trajectory_id"]),
        "breadth_recompute": all(recomputed_decision[key] == score[key] for key in recomputed_decision),
        "final_consistency": final["decision"]["primary_endpoint_pass"] == recomputed_decision["primary_endpoint_pass"],
        "continuation_consistency": final["decision"]["prospective_predictor_phase_authorized"] == recomputed_decision["primary_endpoint_pass"],
        "hidden_scan_denied": final["decision"]["hidden_scan_authorized"] is False,
        "mechanism_claim_denied": final["decision"]["mechanism_claim_authorized"] is False,
        "modulus_search_denied": final["decision"]["modulus_search_authorized"] is False,
    }
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": phase.base.utc_now(),
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "overall_pass": all(checks.values()),
        "recomputed_task_summaries": recomputed_decision["task_summaries"],
        "recomputed_primary_endpoint_pass": recomputed_decision["primary_endpoint_pass"],
        "scope": "Recomputes allocation, sealing, trajectory labels, mixed-panel endpoint, and claim scope. It does not promote training-only correlations to mechanisms.",
    }
    audit["audit_digest"] = phase.base.digest(audit)
    phase.base.write_json(root / "audit/independent_audit.json", audit)
    print(json.dumps({"overall_pass": audit["overall_pass"], "passed": audit["passed"], "total": audit["total"], "audit_digest": audit["audit_digest"]}))
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
