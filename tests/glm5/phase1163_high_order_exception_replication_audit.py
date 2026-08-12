#!/usr/bin/env python3
"""Independent artifact and metric audit for Phase1163."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1163_high_order_exception_replication as phase  # noqa: E402


def record(checks: dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


def audit_command() -> None:
    root = phase.OUT_ROOT
    protocol = phase.read_json(root / "protocol/preregistration.json")
    protocol_audit = phase.read_json(root / "protocol/audit.json")
    calibration_summary = phase.read_json(root / "runs/calibration/summary.json")
    diagnostic_summary = phase.read_json(root / "runs/diagnostics/summary.json")
    metadata = phase.read_json(root / "predictions/metadata.json")
    score = phase.read_json(root / "analysis/score.json")
    final = phase.read_json(root / "analysis/final.json")
    public = phase.read_jsonl(root / "runs/calibration/public_manifest.jsonl")
    truth = phase.read_jsonl(root / "runs/calibration/sealed_truth.jsonl")
    training = phase.read_jsonl(root / "runs/calibration/training_metrics.jsonl")
    with np.load(root / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(root / "predictions/diagnostic_predictions.npz") as pack:
        prediction = np.asarray(pack["prediction"], dtype=np.float64)
    with np.load(root / "predictions/pairwise_coefficients.npz") as pack:
        coefficients = np.asarray(pack["coefficients"], dtype=np.float64)
    with np.load(root / "runs/diagnostics/diagnostic_responses.npz") as pack:
        matched = np.asarray(pack["matched"], dtype=np.float64)
        wrong = np.asarray(pack["wrong"], dtype=np.float64)
        matched_case = np.asarray(pack["matched_case"], dtype=np.float64)
        wrong_case = np.asarray(pack["wrong_case"], dtype=np.float64)

    checks: dict[str, bool] = {}
    body = dict(protocol)
    stored_protocol_digest = body.pop("protocol_digest")
    record(checks, "protocol_digest", phase.digest(body) == stored_protocol_digest)
    record(checks, "protocol_audit_passed", protocol_audit["all_checks_passed"])
    record(checks, "protocol_checks_complete", protocol_audit["passed_count"] == protocol_audit["check_count"])
    record(checks, "primary_script_hash", phase.sha256_file(phase.SCRIPT) == protocol["source_hashes"]["primary_script"])
    record(checks, "audit_script_hash", phase.sha256_file(phase.AUDIT_SCRIPT) == protocol["source_hashes"]["audit_script"])
    record(checks, "phase1161_script_hash", phase.sha256_file(phase.P1161_SCRIPT) == protocol["source_hashes"]["phase1161_script"])
    record(checks, "phase1161_audit_script_hash", phase.sha256_file(phase.P1161_AUDIT) == protocol["source_hashes"]["phase1161_audit_script"])
    record(checks, "model_source_hash", phase.sha256_file(phase.SOURCE_SCRIPT) == protocol["source_hashes"]["model_source_script"])
    record(checks, "registry_exact", protocol["diagnostic_registry"] == phase.diagnostic_registry())
    record(checks, "registry_count", len(protocol["diagnostic_registry"]) == protocol["diagnostic_registry_count"])
    record(checks, "frozen_target_exact", tuple(protocol["frozen_a_star"]) == phase.A_STAR)
    record(checks, "frozen_target_first", phase.registry_index("frozen_a_star") == 0)
    record(checks, "leave_one_out_complete", sum("leave_one_out" in row["categories"] for row in protocol["diagnostic_registry"]) == 4)
    record(checks, "entry_family_complete", sum("entry_query_chain" in row["categories"] for row in protocol["diagnostic_registry"]) == 12)
    record(checks, "random_controls_complete", sum("matched_cardinality_control" in row["categories"] for row in protocol["diagnostic_registry"]) == 32)
    record(checks, "calibration_gate", calibration_summary["calibration_gate_passed"])
    record(checks, "calibration_checks", all(calibration_summary["checks"].values()))
    record(checks, "calibration_shape", list(calibration.shape) == [8, 3, 121])
    record(checks, "calibration_finite", np.isfinite(calibration).all())
    record(checks, "calibration_null", float(np.max(np.abs(calibration[:, :, 0]))) <= phase.THRESHOLDS["null_abs_max"])
    record(checks, "model_count", len(public) == len(truth) == len(training) == 8)
    record(checks, "manifest_alignment", all(a["model_id"] == b["model_id"] for a, b in zip(public, truth, strict=True)))
    record(checks, "architecture_sealed", all("architecture" not in row for row in public))
    record(checks, "two_architectures", set(row["architecture"] for row in truth) == set(phase.ARCHITECTURES))
    record(checks, "all_behavior_qualified", all(row["qualified"] for row in training))
    record(checks, "behavior_accuracy", min(row["accuracy"] for row in training) >= phase.THRESHOLDS["behavior_accuracy_min"])
    record(checks, "behavior_probability", min(row["minimum_probability"] for row in training) >= phase.THRESHOLDS["behavior_min_probability_min"])
    record(checks, "calibration_hash", phase.sha256_file(root / "runs/calibration/calibration_responses.npz") == metadata["calibration_pack_sha256"])
    record(checks, "prediction_hash", phase.sha256_file(root / "predictions/diagnostic_predictions.npz") == metadata["prediction_pack_sha256"])
    record(checks, "coefficient_hash", phase.sha256_file(root / "predictions/pairwise_coefficients.npz") == metadata["coefficient_pack_sha256"])
    record(checks, "prediction_precedes_diagnostics", metadata["created_at_utc"] < diagnostic_summary["created_at_utc"])
    record(checks, "diagnostic_absent_at_seal", metadata["diagnostic_outcomes_absent_at_sealing"])
    record(checks, "diagnostic_gate", diagnostic_summary["diagnostic_gate_passed"])
    record(checks, "diagnostic_checks", all(diagnostic_summary["checks"].values()))
    record(checks, "matched_shape", list(matched.shape) == [8, 3, protocol["diagnostic_registry_count"]])
    record(checks, "wrong_shape", list(wrong.shape) == list(matched.shape))
    record(checks, "case_shape_prefix", list(matched_case.shape[:3]) == list(matched.shape))
    record(checks, "case_shapes_equal", matched_case.shape == wrong_case.shape)
    record(checks, "diagnostic_finite", np.isfinite(matched).all() and np.isfinite(wrong).all())
    record(checks, "diagnostic_hash", phase.sha256_file(root / "runs/diagnostics/diagnostic_responses.npz") == diagnostic_summary["diagnostic_pack_sha256"])

    recomputed_coefficients = np.zeros_like(coefficients)
    recomputed_prediction = np.zeros_like(prediction)
    subsets = phase.registry_subsets()
    for model_index in range(calibration.shape[0]):
        for factor_index in range(calibration.shape[1]):
            coefficient = phase.p1161.fit_coefficients(
                "pairwise", phase.calibration_subsets(), calibration[model_index, factor_index]
            )
            recomputed_coefficients[model_index, factor_index] = coefficient
            recomputed_prediction[model_index, factor_index] = phase.p1161.predict_values(
                "pairwise", coefficient, subsets
            )
    record(checks, "coefficient_recompute", float(np.max(np.abs(coefficients - recomputed_coefficients))) <= 1e-10)
    record(checks, "prediction_recompute", float(np.max(np.abs(prediction - recomputed_prediction))) <= 1e-6)

    recomputed_results = phase.calculate_results(
        protocol, calibration, prediction, matched, wrong, matched_case, wrong_case, truth
    )
    record(checks, "results_recompute", phase.canonical(recomputed_results) == phase.canonical(score["results"]))
    score_body = dict(score)
    stored_score_digest = score_body.pop("score_digest")
    record(checks, "score_digest", phase.digest(score_body) == stored_score_digest)
    record(checks, "score_integrity_checks", all(score["integrity_checks"].values()))
    record(checks, "registry_closed_in_score", score["registry_status"] == "closed_to_further_schedule_search")
    final_body = dict(final)
    stored_final_digest = final_body.pop("final_digest")
    record(checks, "final_digest", phase.digest(final_body) == stored_final_digest)
    record(checks, "final_matches_score", final["decision"] == score["results"]["decision"])
    record(checks, "no_unique_mechanism_claim", not final["unique_natural_mechanism_identified"])
    record(checks, "no_exact_mobius_claim", not final["exact_mobius_order_identified"])
    record(checks, "registry_closed_in_final", final["registry_status"] == "closed_to_further_schedule_search")
    record(checks, "auto_continue_false", not final["auto_continue"])

    audit = {
        "phase": phase.PHASE,
        "created_at_utc": phase.now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "max_coefficient_recompute_error": float(np.max(np.abs(coefficients - recomputed_coefficients))),
        "max_prediction_recompute_error": float(np.max(np.abs(prediction - recomputed_prediction))),
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.write_json(root / "audit/independent_audit.json", audit)
    print(phase.canonical({"all_checks_passed": audit["all_checks_passed"], "passed_count": audit["passed_count"], "check_count": audit["check_count"], "audit_digest": audit["audit_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit_command()


if __name__ == "__main__":
    main()
