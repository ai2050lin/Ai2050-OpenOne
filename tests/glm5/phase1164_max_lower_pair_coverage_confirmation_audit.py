#!/usr/bin/env python3
"""Independent audit for Phase1164 max-lower-pair confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1164_max_lower_pair_coverage_confirmation as phase  # noqa: E402


def record(checks: dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


def audit_command() -> None:
    root = phase.OUT_ROOT
    protocol = phase.p1163.read_json(root / "protocol/preregistration.json")
    protocol_audit = phase.p1163.read_json(root / "protocol/audit.json")
    calibration_summary = phase.p1163.read_json(root / "runs/calibration/summary.json")
    holdout_summary = phase.p1163.read_json(root / "runs/holdout/summary.json")
    metadata = phase.p1163.read_json(root / "predictions/metadata.json")
    score = phase.p1163.read_json(root / "analysis/score.json")
    final = phase.p1163.read_json(root / "analysis/final.json")
    truth = phase.p1163.read_jsonl(root / "runs/calibration/sealed_truth.jsonl")
    public = phase.p1163.read_jsonl(root / "runs/calibration/public_manifest.jsonl")
    training = phase.p1163.read_jsonl(root / "runs/calibration/training_metrics.jsonl")
    with np.load(root / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(root / "predictions/sealed_predictions.npz") as pack:
        predictions = {algorithm: np.asarray(pack[algorithm], dtype=np.float64) for algorithm in phase.ALGORITHMS}
    with np.load(root / "runs/holdout/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)

    checks: dict[str, bool] = {}
    body = dict(protocol)
    stored_protocol = body.pop("protocol_digest")
    record(checks, "protocol_digest", phase.p1163.digest(body) == stored_protocol)
    record(checks, "protocol_audit", protocol_audit["all_checks_passed"])
    record(checks, "protocol_check_count", protocol_audit["passed_count"] == protocol_audit["check_count"])
    record(checks, "primary_script_hash", phase.p1163.sha256_file(phase.SCRIPT) == protocol["source_hashes"]["primary_script"])
    record(checks, "audit_script_hash", phase.p1163.sha256_file(phase.AUDIT_SCRIPT) == protocol["source_hashes"]["audit_script"])
    record(checks, "phase1163_script_hash", phase.p1163.sha256_file(phase.P1163_SCRIPT) == protocol["source_hashes"]["phase1163_script"])
    record(checks, "phase1163_diagnostic_hash", phase.p1163.sha256_file(phase.P1163_DIAGNOSTIC_SCRIPT) == protocol["source_hashes"]["phase1163_diagnostic_script"])
    record(checks, "holdout_exact", protocol["broad_holdout_subsets"] == [list(row) for row in phase.broad_holdout_subsets()])
    record(checks, "holdout_count", len(phase.broad_holdout_subsets()) == 512)
    record(checks, "holdout_unique", len(set(phase.broad_holdout_subsets())) == 512)
    record(checks, "stress_count", len(phase.stress_subsets()) == 12)
    record(checks, "calibration_gate", calibration_summary["calibration_gate_passed"])
    record(checks, "calibration_checks", all(calibration_summary["checks"].values()))
    record(checks, "calibration_shape", list(calibration.shape) == [8, 3, 121])
    record(checks, "calibration_finite", np.isfinite(calibration).all())
    record(checks, "model_count", len(truth) == len(public) == len(training) == 8)
    record(checks, "manifest_alignment", all(a["model_id"] == b["model_id"] for a, b in zip(public, truth, strict=True)))
    record(checks, "architectures_sealed", all("architecture" not in row for row in public))
    record(checks, "two_architectures", set(row["architecture"] for row in truth) == set(phase.ARCHITECTURES))
    record(checks, "behavior_qualified", all(row["qualified"] for row in training))
    record(checks, "behavior_accuracy", min(row["accuracy"] for row in training) >= phase.THRESHOLDS["behavior_accuracy_min"])
    record(checks, "behavior_probability", min(row["minimum_probability"] for row in training) >= phase.THRESHOLDS["behavior_min_probability_min"])
    record(checks, "calibration_hash", phase.p1163.sha256_file(root / "runs/calibration/calibration_responses.npz") == metadata["calibration_pack_sha256"])
    record(checks, "prediction_hash", phase.p1163.sha256_file(root / "predictions/sealed_predictions.npz") == metadata["prediction_pack_sha256"])
    record(checks, "prediction_precedes_holdout", metadata["created_at_utc"] < holdout_summary["created_at_utc"])
    record(checks, "holdout_absent_at_seal", metadata["holdout_outcomes_absent_at_sealing"])
    record(checks, "holdout_gate", holdout_summary["holdout_gate_passed"])
    record(checks, "holdout_checks", all(holdout_summary["checks"].values()))
    record(checks, "holdout_shape", list(observed.shape) == [8, 3, 524])
    record(checks, "holdout_finite", np.isfinite(observed).all())
    record(checks, "holdout_hash", phase.p1163.sha256_file(root / "runs/holdout/holdout_responses.npz") == holdout_summary["holdout_pack_sha256"])

    targets = phase.all_test_subsets()
    recomputed_max_single = phase.max_lower_prediction(calibration, targets, 1)
    recomputed_max_pair = phase.max_lower_prediction(calibration, targets, 2)
    record(checks, "max_single_recompute", float(np.max(np.abs(predictions["max_single"] - recomputed_max_single))) <= 1e-6)
    record(checks, "max_pair_recompute", float(np.max(np.abs(predictions["max_pair"] - recomputed_max_pair))) <= 1e-6)
    for algorithm in ("cardinality", "layout", "main", "pairwise"):
        recomputed = np.zeros_like(predictions[algorithm])
        for model_index in range(calibration.shape[0]):
            for factor_index in range(calibration.shape[1]):
                coefficient = phase.p1161.fit_coefficients(
                    algorithm, phase.calibration_subsets(), calibration[model_index, factor_index]
                )
                recomputed[model_index, factor_index] = phase.p1161.predict_values(
                    algorithm, coefficient, targets
                )
        record(checks, f"{algorithm}_recompute", float(np.max(np.abs(predictions[algorithm] - recomputed))) <= 1e-6)

    recomputed_results = phase.calculate_results(protocol, predictions, observed, truth)
    record(checks, "results_recompute", phase.p1163.canonical(recomputed_results) == phase.p1163.canonical(score["results"]))
    score_body = dict(score)
    stored_score = score_body.pop("score_digest")
    record(checks, "score_digest", phase.p1163.digest(score_body) == stored_score)
    record(checks, "score_integrity", all(score["integrity_checks"].values()))
    record(checks, "branch_closed_score", score["branch_status"] == "closed_after_independent_confirmation")
    final_body = dict(final)
    stored_final = final_body.pop("final_digest")
    record(checks, "final_digest", phase.p1163.digest(final_body) == stored_final)
    record(checks, "final_matches_score", final["decision"] == score["results"]["decision"])
    record(checks, "no_natural_mechanism_claim", not final["natural_mechanism_recovered"])
    record(checks, "branch_closed_final", final["branch_status"] == "closed_after_independent_confirmation")
    record(checks, "auto_continue_false", not final["auto_continue"])

    audit = {
        "phase": phase.PHASE,
        "created_at_utc": phase.p1163.now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = phase.p1163.digest(audit)
    phase.p1163.write_json(root / "audit/independent_audit.json", audit)
    print(phase.p1163.canonical({"all_checks_passed": audit["all_checks_passed"], "passed_count": audit["passed_count"], "check_count": audit["check_count"], "audit_digest": audit["audit_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit_command()


if __name__ == "__main__":
    main()
