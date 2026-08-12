#!/usr/bin/env python3
"""Recover Phase1164 summaries after a numpy.bool_ JSON serialization error.

No model, response, prediction, threshold, or registry value is changed.  The
script only reconstructs the already computed summary with native Python bool
values so the frozen pipeline can continue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1164_max_lower_pair_coverage_confirmation as phase  # noqa: E402


def recover_calibration() -> None:
    protocol = phase.verify_protocol()
    root = phase.OUT_ROOT / "runs/calibration"
    with np.load(root / "calibration_responses.npz") as pack:
        response = np.asarray(pack["response"], dtype=np.float64)
    public_rows = phase.p1163.read_jsonl(root / "public_manifest.jsonl")
    training_rows = phase.p1163.read_jsonl(root / "training_metrics.jsonl")
    diagnostic_rows = phase.p1163.read_jsonl(root / "diagnostics.jsonl")
    denominator_min = min(
        row["factor"][factor]["denominator_min"]
        for row in diagnostic_rows
        for factor in phase.FACTORS
    )
    checks = {
        "model_count": bool(len(public_rows) == protocol["model_count"]),
        "all_models_qualified": bool(all(row["qualified"] for row in training_rows)),
        "behavior_accuracy": bool(
            min(row["accuracy"] for row in training_rows)
            >= phase.THRESHOLDS["behavior_accuracy_min"]
        ),
        "behavior_probability": bool(
            min(row["minimum_probability"] for row in training_rows)
            >= phase.THRESHOLDS["behavior_min_probability_min"]
        ),
        "finite": bool(np.isfinite(response).all()),
        "positive_denominator": bool(
            denominator_min > phase.THRESHOLDS["denominator_min"]
        ),
        "null": bool(
            float(np.max(np.abs(response[:, :, 0])))
            <= phase.THRESHOLDS["null_abs_max"]
        ),
        "architecture_hidden": bool(
            all("architecture" not in row for row in public_rows)
        ),
    }
    summary = {
        "phase": phase.PHASE,
        "created_at_utc": phase.p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "response_shape": list(response.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(
            row["minimum_probability"] for row in training_rows
        ),
        "denominator_min": denominator_min,
        "null_max_abs": float(np.max(np.abs(response[:, :, 0]))),
        "calibration_pack_sha256": phase.p1163.sha256_file(
            root / "calibration_responses.npz"
        ),
        "checks": checks,
        "calibration_gate_passed": bool(all(checks.values())),
        "serialization_recovery": {
            "reason": "numpy.bool_ is not serializable by the frozen canonical JSON function",
            "scientific_values_changed": False,
        },
    }
    summary["summary_digest"] = phase.p1163.digest(summary)
    phase.p1163.write_json(root / "summary.json", summary)
    print(phase.p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def recover_holdout() -> None:
    protocol = phase.verify_protocol()
    metadata = phase.p1163.read_json(phase.OUT_ROOT / "predictions/metadata.json")
    root = phase.OUT_ROOT / "runs/holdout"
    with np.load(root / "holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    diagnostic_rows = phase.p1163.read_jsonl(root / "diagnostics.jsonl")
    denominator_min = min(
        row["factor"][factor]["denominator_min"]
        for row in diagnostic_rows
        for factor in phase.FACTORS
    )
    checks = {
        "model_count": bool(len(diagnostic_rows) == protocol["model_count"]),
        "target_count": bool(observed.shape[2] == len(phase.all_test_subsets())),
        "finite": bool(np.isfinite(observed).all()),
        "positive_denominator": bool(
            denominator_min > phase.THRESHOLDS["denominator_min"]
        ),
        "prediction_integrity": bool(
            phase.p1163.sha256_file(phase.OUT_ROOT / "predictions/sealed_predictions.npz")
            == metadata["prediction_pack_sha256"]
        ),
        "prediction_precedes_holdout": bool(metadata["created_at_utc"] < phase.p1163.now()),
    }
    summary = {
        "phase": phase.PHASE,
        "created_at_utc": phase.p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "response_shape": list(observed.shape),
        "denominator_min": denominator_min,
        "holdout_pack_sha256": phase.p1163.sha256_file(root / "holdout_responses.npz"),
        "checks": checks,
        "holdout_gate_passed": bool(all(checks.values())),
        "serialization_recovery": {
            "reason": "numpy.bool_ is not serializable by the frozen canonical JSON function",
            "scientific_values_changed": False,
        },
    }
    summary["summary_digest"] = phase.p1163.digest(summary)
    phase.p1163.write_json(root / "summary.json", summary)
    print(phase.p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("calibration", "holdout"))
    args = parser.parse_args()
    if args.target == "calibration":
        recover_calibration()
    else:
        recover_holdout()


if __name__ == "__main__":
    main()
