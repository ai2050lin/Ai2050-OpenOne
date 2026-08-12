#!/usr/bin/env python3
"""Independent audit for Phase1162 modular-task transfer."""

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
PRIMARY = ROOT / "tests/glm5/phase1162_modular_task_response_transfer.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1162_modular_task_response_transfer"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1162_modular_task_response_transfer as phase  # noqa: E402


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


def recompute_results(
    predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    truth: list[dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for algorithm in phase.prior.ALGORITHMS:
        units = []
        for model_index, truth_row in enumerate(truth):
            for factor_index, factor in enumerate(phase.FACTORS):
                detail = phase.prior.metrics(
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
        results[algorithm] = {
            "unit_metrics": units,
            "median_mae": float(np.median([row["mae"] for row in units])),
            "median_correlation": float(np.median([row["correlation"] for row in units])),
            "unit_pass_count": int(
                sum(
                    row["mae"] <= phase.THRESHOLDS["global_unit_mae_max"]
                    and row["correlation"] >= phase.THRESHOLDS["global_unit_correlation_min"]
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
    return results


def audit_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_body = dict(protocol)
    protocol_digest = protocol_body.pop("protocol_digest")
    calibration_summary = read_json(OUT_ROOT / "runs/models/calibration_summary.json")
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = read_json(OUT_ROOT / "runs/models/holdout_summary.json")
    score = read_json(OUT_ROOT / "analysis/score.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    public = read_jsonl(OUT_ROOT / "runs/models/public_manifest.jsonl")
    truth = read_jsonl(OUT_ROOT / "runs/models/sealed_truth.jsonl")
    with np.load(OUT_ROOT / "runs/models/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(OUT_ROOT / "predictions/predictions.npz") as pack:
        predictions = {name: np.asarray(pack[name], dtype=np.float64) for name in phase.prior.ALGORITHMS}
    with np.load(OUT_ROOT / "runs/models/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    holdout = [tuple(row) for row in protocol["holdout_subsets"]]
    stress = [tuple(row) for row in protocol["stress_subsets"]]
    recomputed_results = recompute_results(predictions, observed, truth)
    selected = recomputed_results[phase.SELECTED_ALGORITHM]
    layout_advantage = recomputed_results["layout"]["median_mae"] - selected["median_mae"]
    global_checks = {
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/predictions.npz")
        == metadata["prediction_pack_sha256"],
        "median_mae": selected["median_mae"] <= phase.THRESHOLDS["global_median_mae_max"],
        "median_correlation": selected["median_correlation"]
        >= phase.THRESHOLDS["global_median_correlation_min"],
        "unit_pass": selected["unit_pass_count"] >= phase.THRESHOLDS["global_unit_pass_min"],
        "unit_total": selected["unit_count"] == phase.THRESHOLDS["global_unit_total"],
        "architecture_median_mae": all(
            value <= phase.THRESHOLDS["architecture_median_mae_max"]
            for value in selected["architecture_median_mae"].values()
        ),
        "beats_layout_baseline": layout_advantage >= phase.THRESHOLDS["layout_mae_advantage_min"],
    }
    stress_indices = [holdout.index(row) for row in stress]
    absolute_error = np.abs(predictions[phase.SELECTED_ALGORITHM] - observed)
    stress_subset_medians = {
        phase.prior.subset_id(holdout[index]): float(np.median(absolute_error[:, :, index]))
        for index in stress_indices
    }
    stress_median = float(np.median(absolute_error[:, :, stress_indices]))
    stress_checks = {
        "stress_median_absolute_error": stress_median
        <= phase.THRESHOLDS["stress_median_absolute_error_max"],
        "each_stress_subset": all(
            value <= phase.THRESHOLDS["stress_each_subset_median_absolute_error_max"]
            for value in stress_subset_medians.values()
        ),
    }
    prediction_time = datetime.fromisoformat(metadata["created_at_utc"])
    outcome_time = datetime.fromisoformat(holdout_summary["created_at_utc"])
    checks = {
        "protocol_digest": digest(protocol_body) == protocol_digest,
        "primary_source_frozen": sha256_file(PRIMARY) == protocol["source_hashes"]["primary_script"],
        "audit_source_frozen": sha256_file(SCRIPT) == protocol["source_hashes"]["audit_script"],
        "task_formula_non_cartesian": protocol["task"]["output_classes"] == 8
        and protocol["task"]["input_combinations"] == 32,
        "calibration_shape": calibration.shape == (8, len(phase.FACTORS), len(phase.prior.calibration_subsets())),
        "holdout_shape": observed.shape == (8, len(phase.FACTORS), len(holdout)),
        "prediction_shapes": all(array.shape == observed.shape for array in predictions.values()),
        "finite": bool(np.isfinite(calibration).all() and np.isfinite(observed).all())
        and all(np.isfinite(array).all() for array in predictions.values()),
        "calibration_null_exact": float(np.max(np.abs(calibration[:, :, 0])))
        <= phase.THRESHOLDS["null_abs_max"],
        "calibration_hash": sha256_file(OUT_ROOT / "runs/models/calibration_responses.npz")
        == calibration_summary["effect_pack_sha256"]
        == metadata["calibration_pack_sha256"],
        "prediction_hash": sha256_file(OUT_ROOT / "predictions/predictions.npz")
        == metadata["prediction_pack_sha256"],
        "prediction_digest": digest({key: value for key, value in metadata.items() if key != "prediction_digest"})
        == metadata["prediction_digest"],
        "prediction_precedes_outcome": prediction_time < outcome_time,
        "architecture_hidden": all("architecture" not in row for row in public),
        "new_random_holdout": not bool(
            (set(holdout) - set(stress)).intersection(
                phase.prior.discovery_holdout_subsets() + phase.prior.confirmation_holdout_subsets()
            )
        ),
        "stress_present": all(row in holdout for row in stress) and len(stress) == 3,
        "algorithm_results_recomputed": canonical(recomputed_results) == canonical(score["algorithm_results"]),
        "layout_advantage_recomputed": abs(layout_advantage - score["layout_mae_advantage"]) <= 1e-12,
        "global_checks_recomputed": global_checks == score["global_checks"],
        "stress_indices_recomputed": stress_indices == score["stress_indices"],
        "stress_medians_recomputed": canonical(stress_subset_medians)
        == canonical(score["stress_subset_median_absolute_errors"]),
        "stress_checks_recomputed": stress_checks == score["stress_checks"],
        "score_digest": digest({key: value for key, value in score.items() if key != "score_digest"})
        == score["score_digest"],
        "final_global_consistent": final["global_transfer_passed"] == all(global_checks.values()),
        "final_stress_consistent": final["high_order_stress_passed"] == all(stress_checks.values()),
        "final_no_overclaim": not final["causal_graph_recovered"]
        and not final["physical_hyperedges_recovered"]
        and not final["full_mechanism_recovery_complete"],
        "final_auto_stop": not final["auto_continue"],
        "final_digest": digest({key: value for key, value in final.items() if key != "final_digest"})
        == final["final_digest"],
    }
    audit = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "recomputed_selected_metrics": selected,
        "recomputed_layout_mae_advantage": layout_advantage,
        "recomputed_global_checks": global_checks,
        "recomputed_stress_subset_medians": stress_subset_medians,
        "recomputed_stress_checks": stress_checks,
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
