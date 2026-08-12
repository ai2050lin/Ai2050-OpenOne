#!/usr/bin/env python3
"""Independent recomputation audit for Phase1161."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PRIMARY = ROOT / "tests/glm5/phase1161_ordered_intervention_response_prediction.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1161_ordered_intervention_response_prediction"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1161_ordered_intervention_response_prediction as phase  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def recompute_score(
    predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    truth: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for algorithm in phase.ALGORITHMS:
        units = []
        for model_index, truth_row in enumerate(truth):
            for factor_index, factor in enumerate(phase.FACTORS):
                detail = phase.metrics(
                    predictions[algorithm][model_index, factor_index],
                    observed[model_index, factor_index],
                )
                units.append(
                    {
                        "model_index": model_index,
                        "architecture": truth_row["architecture"],
                        "factor": factor,
                        **detail,
                    }
                )
        output[algorithm] = {
            "unit_metrics": units,
            "median_mae": float(np.median([row["mae"] for row in units])),
            "median_correlation": float(np.median([row["correlation"] for row in units])),
            "unit_pass_count": int(
                sum(
                    row["mae"] <= phase.THRESHOLDS["confirmation_unit_mae_max"]
                    and row["correlation"] >= phase.THRESHOLDS["confirmation_unit_correlation_min"]
                    for row in units
                )
            ),
            "unit_count": len(units),
            "architecture_median_mae": {
                architecture: float(
                    np.median([row["mae"] for row in units if row["architecture"] == architecture])
                )
                for architecture in phase.ARCHITECTURES
            },
        }
    return output


def audit_command() -> None:
    required = [
        OUT_ROOT / "protocol/preregistration.json",
        OUT_ROOT / "runs/discovery/responses.npz",
        OUT_ROOT / "analysis/discovery_fit.json",
        OUT_ROOT / "runs/confirmation/calibration_responses.npz",
        OUT_ROOT / "predictions/confirmation_predictions.npz",
        OUT_ROOT / "predictions/metadata.json",
        OUT_ROOT / "runs/confirmation/holdout_responses.npz",
        OUT_ROOT / "analysis/confirmation_score.json",
        OUT_ROOT / "analysis/final.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing artifacts: {missing}")
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_body = dict(protocol)
    protocol_digest = protocol_body.pop("protocol_digest")
    discovery_summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    fit = read_json(OUT_ROOT / "analysis/discovery_fit.json")
    calibration_summary = read_json(OUT_ROOT / "runs/confirmation/calibration_summary.json")
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = read_json(OUT_ROOT / "runs/confirmation/holdout_summary.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    calibration = [tuple(row) for row in protocol["calibration_subsets"]]
    discovery_holdout = [tuple(row) for row in protocol["discovery_holdout_subsets"]]
    confirmation_holdout = [tuple(row) for row in protocol["confirmation_holdout_subsets"]]
    discovery_public = read_jsonl(OUT_ROOT / "runs/discovery/public_manifest.jsonl")
    confirmation_public = read_jsonl(OUT_ROOT / "runs/confirmation/public_manifest.jsonl")
    confirmation_truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    with np.load(OUT_ROOT / "runs/discovery/responses.npz") as pack:
        discovery_response = np.asarray(pack["response"], dtype=np.float64)
    calibration_count = len(calibration)
    fit_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(discovery_public) if row["analysis_partition"] == "validation"]
    recomputed_fit_metrics = phase.evaluate_algorithms(
        discovery_response[:, :, :calibration_count],
        discovery_response[:, :, calibration_count:],
        discovery_holdout,
        fit_indices,
    )
    recomputed_validation_metrics = phase.evaluate_algorithms(
        discovery_response[:, :, :calibration_count],
        discovery_response[:, :, calibration_count:],
        discovery_holdout,
        validation_indices,
    )
    structural_mae = {
        name: recomputed_fit_metrics[name]["median_mae"] for name in phase.STRUCTURAL_ALGORITHMS
    }
    best_mae = min(structural_mae.values())
    recomputed_selected = next(
        name
        for name in phase.STRUCTURAL_ALGORITHMS
        if structural_mae[name] <= best_mae + phase.THRESHOLDS["complexity_tie_mae"]
    )
    with np.load(OUT_ROOT / "predictions/confirmation_predictions.npz") as pack:
        predictions = {name: np.asarray(pack[name], dtype=np.float64) for name in phase.ALGORITHMS}
    with np.load(OUT_ROOT / "runs/confirmation/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    recomputed_algorithms = recompute_score(predictions, observed, confirmation_truth)
    selected = fit["selected_algorithm"]
    selected_result = recomputed_algorithms[selected]
    layout_advantage = recomputed_algorithms["layout"]["median_mae"] - selected_result["median_mae"]
    recomputed_checks = {
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/confirmation_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "selected_algorithm_frozen": selected == metadata["selected_algorithm"],
        "median_mae": selected_result["median_mae"]
        <= phase.THRESHOLDS["confirmation_median_mae_max"],
        "median_correlation": selected_result["median_correlation"]
        >= phase.THRESHOLDS["confirmation_median_correlation_min"],
        "unit_pass": selected_result["unit_pass_count"] >= phase.THRESHOLDS["confirmation_unit_pass_min"],
        "unit_total": selected_result["unit_count"] == phase.THRESHOLDS["confirmation_unit_total"],
        "architecture_median_mae": all(
            value <= phase.THRESHOLDS["confirmation_architecture_median_mae_max"]
            for value in selected_result["architecture_median_mae"].values()
        ),
        "beats_layout_baseline": layout_advantage
        >= phase.THRESHOLDS["confirmation_layout_mae_advantage_min"],
    }
    prediction_time = datetime.fromisoformat(metadata["created_at_utc"])
    outcome_time = datetime.fromisoformat(holdout_summary["created_at_utc"])
    checks = {
        "protocol_digest": digest(protocol_body) == protocol_digest,
        "primary_source_frozen": sha256_file(PRIMARY) == protocol["source_hashes"]["primary_script"],
        "audit_source_frozen": sha256_file(SCRIPT) == protocol["source_hashes"]["audit_script"],
        "site_count": protocol["site_count"] == 15 == len(protocol["sites"]),
        "depths_chronological": [row["depth"] for row in protocol["sites"]]
        == sorted(row["depth"] for row in protocol["sites"]),
        "calibration_cardinality": {len(row) for row in calibration} == {0, 1, 2},
        "discovery_holdout_cardinality": {len(row) for row in discovery_holdout} == {3, 4},
        "confirmation_holdout_cardinality": {len(row) for row in confirmation_holdout} == {3, 4},
        "holdouts_disjoint": not bool(set(discovery_holdout).intersection(confirmation_holdout)),
        "calibration_holdouts_disjoint": not bool(
            set(calibration).intersection(discovery_holdout + confirmation_holdout)
        ),
        "discovery_shape": discovery_response.shape
        == (8, len(phase.FACTORS), len(calibration) + len(discovery_holdout)),
        "discovery_null_exact": float(np.max(np.abs(discovery_response[:, :, 0])))
        <= phase.THRESHOLDS["null_abs_max"],
        "discovery_summary_pack_hash": sha256_file(OUT_ROOT / "runs/discovery/responses.npz")
        == discovery_summary["effect_pack_sha256"],
        "discovery_architecture_hidden": all("architecture" not in row for row in discovery_public),
        "fit_indices": fit_indices == fit["fit_indices"],
        "validation_indices": validation_indices == fit["validation_indices"],
        "fit_metrics_recomputed": canonical(recomputed_fit_metrics) == canonical(fit["fit_metrics"]),
        "validation_metrics_recomputed": canonical(recomputed_validation_metrics)
        == canonical(fit["validation_metrics"]),
        "selection_recomputed": recomputed_selected == selected,
        "fit_digest": digest({key: value for key, value in fit.items() if key != "fit_digest"})
        == fit["fit_digest"],
        "calibration_pack_hash": sha256_file(OUT_ROOT / "runs/confirmation/calibration_responses.npz")
        == calibration_summary["effect_pack_sha256"]
        == metadata["calibration_pack_sha256"],
        "confirmation_architecture_hidden": all("architecture" not in row for row in confirmation_public),
        "prediction_pack_hash": sha256_file(OUT_ROOT / "predictions/confirmation_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "prediction_digest": digest({key: value for key, value in metadata.items() if key != "prediction_digest"})
        == metadata["prediction_digest"],
        "prediction_precedes_outcome": prediction_time < outcome_time,
        "prediction_shapes": all(
            array.shape == (8, len(phase.FACTORS), len(confirmation_holdout))
            for array in predictions.values()
        ),
        "observed_shape": observed.shape == (8, len(phase.FACTORS), len(confirmation_holdout)),
        "finite_predictions": all(np.isfinite(array).all() for array in predictions.values()),
        "finite_observed": bool(np.isfinite(observed).all()),
        "score_algorithms_recomputed": canonical(recomputed_algorithms) == canonical(score["algorithm_results"]),
        "score_layout_advantage": close(layout_advantage, score["layout_mae_advantage"]),
        "score_checks_recomputed": recomputed_checks == score["checks"],
        "score_digest": digest({key: value for key, value in score.items() if key != "score_digest"})
        == score["score_digest"],
        "final_decision_consistent": final["ordered_response_prediction_confirmed"]
        == all(recomputed_checks.values()),
        "final_no_graph_overclaim": not final["causal_graph_recovered"]
        and not final["physical_hyperedges_recovered"]
        and not final["full_mechanism_recovery_complete"],
        "final_digest": digest({key: value for key, value in final.items() if key != "final_digest"})
        == final["final_digest"],
    }
    audit = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "recomputed_selected_algorithm": recomputed_selected,
        "recomputed_selected_metrics": selected_result,
        "recomputed_layout_mae_advantage": layout_advantage,
        "recomputed_confirmation_checks": recomputed_checks,
        "primary_final_digest": final["final_digest"],
    }
    audit["audit_digest"] = digest(audit)
    write_json(OUT_ROOT / "audit/independent_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit_command()


if __name__ == "__main__":
    main()
