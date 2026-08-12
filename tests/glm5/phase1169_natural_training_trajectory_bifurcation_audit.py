#!/usr/bin/env python3
"""Independent recomputation audit for Phase1169."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as phase  # noqa: E402


def main() -> None:
    root = phase.OUT_ROOT
    protocol = phase.read_json(root / "protocol/preregistration.json")
    seal = phase.read_json(root / "runs/training/seal.json")
    holdout_summary = phase.read_json(root / "runs/holdout/summary.json")
    training = phase.read_jsonl(root / "runs/training/training_metrics.jsonl")
    holdout = phase.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    score = phase.read_json(root / "analysis/score.json")
    final = phase.read_json(root / "analysis/final.json")

    grouped: dict[str, list[dict]] = {}
    for row in holdout:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    recomputed = [phase.trajectory_summary(rows) for rows in grouped.values()]
    split_counts = {
        split: sum(row["transition_present"] for row in recomputed if row["split"] == split)
        for split in phase.TASKS
    }
    primary = all(value >= phase.THRESHOLDS["successful_trajectories_per_split_min"] for value in split_counts.values())
    checkpoint_hashes = {
        row["checkpoint_id"]: phase.sha256_file(root / "runs/training/checkpoints" / f"{row['checkpoint_id']}.pt")
        for row in training
    }
    checks = {
        "protocol_digest": protocol["protocol_digest"] == phase.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        "primary_script_hash": protocol["source_hashes"]["primary_script"] == phase.sha256_file(phase.SCRIPT),
        "audit_script_hash": protocol["source_hashes"]["audit_script"] == phase.sha256_file(phase.AUDIT_SCRIPT),
        "prior_final_hash": protocol["prerequisite"]["phase1168_final_sha256"] == phase.sha256_file(phase.P1168_FINAL),
        "prior_audit_hash": protocol["prerequisite"]["phase1168_audit_sha256"] == phase.sha256_file(phase.P1168_AUDIT),
        "training_row_count": len(training) == len(phase.TASKS) * phase.REPLICATES * len(phase.CHECKPOINT_STEPS),
        "holdout_row_count": len(holdout) == len(training),
        "trajectory_count": len(grouped) == len(phase.TASKS) * phase.REPLICATES,
        "checkpoint_steps": all(tuple(sorted(row["step"] for row in rows)) == phase.CHECKPOINT_STEPS for rows in grouped.values()),
        "checkpoint_hashes": checkpoint_hashes == seal["checkpoint_hashes"],
        "training_metrics_hash": seal["training_metrics_sha256"] == phase.sha256_file(root / "runs/training/training_metrics.jsonl"),
        "holdout_metrics_hash": holdout_summary["holdout_metrics_sha256"] == phase.sha256_file(root / "runs/holdout/holdout_metrics.jsonl"),
        "sealed_before_holdout": seal["training_sealed"] and seal["holdout_outcomes_absent_at_sealing"],
        "no_holdout_training_eval": seal["no_holdout_evaluated"] and all(not row["holdout_evaluated_during_training"] for row in training),
        "no_holdout_gradient": seal["no_holdout_gradient"] and all(not row["holdout_used_by_gradient"] for row in training),
        "finite": holdout_summary["finite"],
        "split_counts": split_counts == score["split_transition_counts"],
        "primary_recompute": primary == score["primary_endpoint_pass"],
        "final_consistency": final["decision"]["primary_endpoint_pass"] == primary,
        "hidden_scan_denied": final["decision"]["hidden_scan_authorized"] is False,
        "mechanism_claim_denied": final["decision"]["mechanism_claim_authorized"] is False,
    }
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": phase.utc_now(),
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "overall_pass": all(checks.values()),
        "recomputed_split_transition_counts": split_counts,
        "recomputed_primary_endpoint_pass": primary,
        "scope": "Recomputes sealing, hashes, transition labels, and final decision; it does not convert exploratory features into mechanisms.",
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.write_json(root / "audit/independent_audit.json", audit)
    print(json.dumps({"overall_pass": audit["overall_pass"], "passed": audit["passed"], "total": audit["total"], "audit_digest": audit["audit_digest"]}))
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
