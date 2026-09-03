#!/usr/bin/env python3
"""C204: fit simple odd dose-response laws on C196 doses 0.25/0.5 and predict 1.0."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1738_c204_odd_nonlinear_dose_response"
C196 = RESULT / "phase1730_c196_multidose_orthogonal_identification"
C203 = RESULT / "phase1737_c203_bf16_intervention_representability"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1738, "C204"
FIT_DOSES = np.asarray([0.25, 0.5], dtype=np.float64)
HOLDOUT_DOSE = 1.0
MODELS = ("registered_c195_superposition", "half_dose_proportional", "two_dose_linear_ls", "two_dose_odd_cubic")


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C203 / "audit/independent_final_audit.json")
    source = np.load(C196 / "raw/orthogonal_actual.float16.npy", mmap_mode="r")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C204_nonlinear_basis_response_without_precision_repair",
        "source_shape": list(source.shape) == [14, 3, 16, 2, 6, 2560],
        "fit_holdout": list(FIT_DOSES) == [0.25, 0.5] and HOLDOUT_DOSE == 1.0,
        "models": len(MODELS) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "odd_dose_response_frozen",
        "source": "C196 frozen symmetric odd response tensor; no model rerun",
        "fit_doses": FIT_DOSES.tolist(), "holdout_dose": HOLDOUT_DOSE, "models": list(MODELS),
        "linear_ls": "y(d)=a*d, fitted on d in {0.25,0.5}",
        "odd_cubic": "y(d)=a*d+c*d^3, exactly fitted on d in {0.25,0.5}, prospectively evaluated only at d=1",
        "primary_gate": {"odd_cubic_nrmse_max": 0.50, "improvement_over_best_linear_min": 0.05},
        "claim_boundary": "Dose-response interpolation/extrapolation for one frozen intervention family; not a language operator, cross-program law, or causal mechanism.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "fitting on dose 1", "gate changes"],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error2 = float(np.square(prediction - truth, dtype=np.float64).sum())
    truth2 = float(np.square(truth, dtype=np.float64).sum())
    weight = np.minimum(np.abs(prediction), np.abs(truth)).astype(np.float64)
    sign = float((weight * (np.signbit(prediction) == np.signbit(truth))).sum() / max(float(weight.sum()), 1e-30))
    return {"nrmse": float(np.sqrt(error2 / max(truth2, 1e-30))), "weighted_sign_agreement": sign}


def analyze() -> None:
    actual = np.load(C196 / "raw/orthogonal_actual.float16.npy", mmap_mode="r")
    registered = np.load(C196 / "raw/orthogonal_predicted.float16.npy", mmap_mode="r")
    y1 = np.asarray(actual[:, 0], dtype=np.float32)
    y2 = np.asarray(actual[:, 1], dtype=np.float32)
    truth = np.asarray(actual[:, 2], dtype=np.float32)
    d1, d2 = FIT_DOSES
    linear_a = (d1 * y1 + d2 * y2) / float(d1 * d1 + d2 * d2)
    matrix = np.asarray([[d1, d1 ** 3], [d2, d2 ** 3]], dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    cubic_a = inverse[0, 0] * y1 + inverse[0, 1] * y2
    cubic_c = inverse[1, 0] * y1 + inverse[1, 1] * y2
    predictions = {
        "registered_c195_superposition": np.asarray(registered[:, 2], dtype=np.float32),
        "half_dose_proportional": 2.0 * y2,
        "two_dose_linear_ls": (linear_a * HOLDOUT_DOSE).astype(np.float32),
        "two_dose_odd_cubic": (cubic_a * HOLDOUT_DOSE + cubic_c * HOLDOUT_DOSE ** 3).astype(np.float32),
    }
    result_metrics = {name: metrics(value, truth) for name, value in predictions.items()}
    linear_names = ("registered_c195_superposition", "half_dose_proportional", "two_dose_linear_ls")
    best_linear = min(linear_names, key=lambda name: result_metrics[name]["nrmse"])
    improvement = result_metrics[best_linear]["nrmse"] - result_metrics["two_dose_odd_cubic"]["nrmse"]
    gate = core.load(OUT / "protocol/preregistration.json")["primary_gate"]
    passed = result_metrics["two_dose_odd_cubic"]["nrmse"] <= gate["odd_cubic_nrmse_max"] and improvement >= gate["improvement_over_best_linear_min"]
    curvature_ratio = float(np.sqrt(np.square(cubic_c, dtype=np.float64).sum() / max(np.square(cubic_a, dtype=np.float64).sum(), 1e-30)))
    report = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "odd_dose_response_analyzed",
        "fit_doses": FIT_DOSES.tolist(), "holdout_dose": HOLDOUT_DOSE,
        "holdout_metrics": result_metrics, "best_linear": best_linear,
        "odd_cubic_improvement_over_best_linear": float(improvement),
        "cubic_to_linear_coefficient_rms": curvature_ratio,
        "primary_gate_passed": bool(passed),
        "interpretation": "The cubic model is an elementwise odd interpolation basis. Holdout success would establish dose curvature only for this response family; failure rejects this simple basis without implying global linearity.",
        "next_authorization": "C205_prospective_nonlinear_dose_replication" if passed else "C205_abandon_simple_odd_polynomial_and_map_response_regimes",
    }
    core.save(OUT / "analysis/holdout.json", report)
    checks = {"models": set(result_metrics) == set(MODELS), "finite": bool(np.isfinite([[row["nrmse"], row["weighted_sign_agreement"]] for row in result_metrics.values()]).all()), "holdout_only": HOLDOUT_DOSE not in FIT_DOSES}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/holdout.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "analyze", "close")); args = parser.parse_args(); {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__": main()
