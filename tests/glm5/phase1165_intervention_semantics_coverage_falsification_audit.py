#!/usr/bin/env python3
"""Independent artifact audit for Phase1165."""

from __future__ import annotations

import hashlib
import json
import py_compile
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PRIMARY_SCRIPT = ROOT / "tests/glm5/phase1165_intervention_semantics_coverage_falsification.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1165_intervention_semantics_coverage_falsification"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1165_intervention_semantics_coverage_falsification as phase  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> None:
    py_compile.compile(str(PRIMARY_SCRIPT), doraise=True)
    py_compile.compile(str(SCRIPT), doraise=True)
    protocol = phase.p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    calibration_summary = phase.p1163.read_json(OUT_ROOT / "runs/calibration/summary.json")
    prediction_metadata = phase.p1163.read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = phase.p1163.read_json(OUT_ROOT / "runs/holdout/summary.json")
    score = phase.p1163.read_json(OUT_ROOT / "analysis/score.json")
    final = phase.p1163.read_json(OUT_ROOT / "analysis/final.json")
    body = dict(protocol)
    protocol_digest = body.pop("protocol_digest")
    with np.load(OUT_ROOT / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(OUT_ROOT / "predictions/sealed_predictions.npz") as pack:
        predictions = {algorithm: np.asarray(pack[algorithm], dtype=np.float64) for algorithm in phase.ALGORITHMS}
    with np.load(OUT_ROOT / "runs/holdout/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    recalculated = phase.calculate_results(predictions, observed)
    broad = phase.broad_holdout_subsets()
    expected_calibration_shape = (
        protocol["model_count"],
        len(phase.FACTORS),
        len(phase.SEMANTICS),
        len(phase.calibration_subsets()),
    )
    expected_holdout_shape = (
        protocol["model_count"],
        len(phase.FACTORS),
        len(phase.SEMANTICS),
        len(phase.all_test_subsets()),
    )
    checks = {
        "primary_compiles": True,
        "audit_compiles": True,
        "protocol_digest": digest(body) == protocol_digest,
        "primary_hash_frozen": phase.p1163.sha256_file(PRIMARY_SCRIPT) == protocol["source_hashes"]["primary_script"],
        "audit_hash_frozen": phase.p1163.sha256_file(SCRIPT) == protocol["source_hashes"]["audit_script"],
        "source_hash_frozen": phase.p1163.sha256_file(phase.P1164_SCRIPT) == protocol["source_hashes"]["phase1164_script"],
        "protocol_audit_passed": all(protocol["checks"].values()),
        "calibration_gate_passed": calibration_summary["calibration_gate_passed"],
        "holdout_gate_passed": holdout_summary["holdout_gate_passed"],
        "calibration_shape": calibration.shape == expected_calibration_shape,
        "holdout_shape": observed.shape == expected_holdout_shape,
        "calibration_finite": bool(np.isfinite(calibration).all()),
        "holdout_finite": bool(np.isfinite(observed).all()),
        "null_exact": float(np.max(np.abs(calibration[:, :, :, 0]))) <= phase.THRESHOLDS["null_abs_max"],
        "holdout_registry_frozen": protocol["broad_holdout_subsets"] == [list(row) for row in broad],
        "holdout_registry_unique": len(broad) == len(set(broad)),
        "holdout_disjoint_prior": not bool(set(broad).intersection(phase.prior_used_subsets())),
        "prediction_precedes_holdout": prediction_metadata["created_at_utc"] < holdout_summary["created_at_utc"],
        "prediction_hash": phase.p1163.sha256_file(OUT_ROOT / "predictions/sealed_predictions.npz") == prediction_metadata["prediction_pack_sha256"],
        "holdout_hash": phase.p1163.sha256_file(OUT_ROOT / "runs/holdout/holdout_responses.npz") == holdout_summary["holdout_pack_sha256"],
        "recalculated_decision": recalculated["decision"] == score["results"]["decision"],
        "recalculated_mode_results": recalculated["mode_results"] == score["results"]["mode_results"],
        "score_digest": digest({key: value for key, value in score.items() if key != "score_digest"}) == score["score_digest"],
        "final_score_link": final["score_digest"] == score["score_digest"],
        "final_decision_link": final["decision"] == score["results"]["decision"],
        "branch_closed": final["branch_status"] == "closed_after_one_shot_semantic_comparison",
        "auto_continue_false": final["auto_continue"] is False,
        "natural_mechanism_not_claimed": final["natural_mechanism_recovered"] is False,
        "four_semantics": len(protocol["semantics"]) == 4,
        "three_algorithms": len(protocol["algorithms"]) == 3,
        "transport_gate_present": "transport_median_q95_min" in protocol["thresholds"],
        "coverage_gate_present": "coverage_median_relative_mae_max" in protocol["thresholds"],
        "one_shot_rule": protocol["checks"]["one_shot_semantic_axis"],
    }
    audit = {
        "phase": phase.PHASE,
        "created_at_utc": phase.p1163.now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "decision": final["decision"],
        "mode_decisions": final["mode_decisions"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = digest(audit)
    phase.p1163.write_json(OUT_ROOT / "audit/independent_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
