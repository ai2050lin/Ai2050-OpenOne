#!/usr/bin/env python3
"""Prospectively confirm a max-lower-pair coverage response estimator.

Phase1163 showed post-hoc that additive pairwise extrapolation confuses
algebraic order with intervention necessity on saturating/redundant response
surfaces.  This one-shot phase freezes the simpler rule

    F_hat(A) = max { F(B) : B subset A, |B| <= 2 }

and tests it on new networks and 512 entirely new triple/quad schedules.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1164_max_lower_pair_coverage_confirmation_audit.py"
P1163_SCRIPT = ROOT / "tests/glm5/phase1163_high_order_exception_replication.py"
P1163_DIAGNOSTIC_SCRIPT = ROOT / "tests/glm5/phase1163_posthoc_sublattice_coverage_diagnostic.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1164_max_lower_pair_coverage_confirmation"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1163_high_order_exception_replication as p1163  # noqa: E402


p1161 = p1163.p1161
source = p1163.source
PHASE = 1164
FACTORS = source.FACTORS
ARCHITECTURES = source.ARCHITECTURES
REPLICATES = 4
HOLDOUT_SEED = 1164007
HOLDOUT_PER_CARDINALITY = 256
ALGORITHMS = ("cardinality", "layout", "main", "pairwise", "max_single", "max_pair")
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "denominator_min": 1e-5,
    "null_abs_max": 1e-8,
    "max_pair_median_unit_mae_max": 0.03,
    "max_pair_pairwise_advantage_min": 0.01,
    "max_pair_max_single_advantage_min": 0.005,
    "unit_mae_max": 0.05,
    "unit_pairwise_advantage_min": 0.005,
    "unit_pass_min": 20,
    "unit_total": 24,
    "architecture_median_mae_max": 0.04,
    "schedule_abs_error_q95_max": 0.10,
    "stress_median_unit_mae_max": 0.03,
    "stress_a_star_abs_error_median_max": 0.05,
}


def model_seed(architecture: str, replicate: int) -> int:
    return 1164100 + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def model_id(seed: int) -> str:
    return p1163.digest({"phase": PHASE, "seed": seed})[:16]


def calibration_subsets() -> list[tuple[int, ...]]:
    return p1163.calibration_subsets()


def broad_holdout_subsets() -> list[tuple[int, ...]]:
    excluded = set(p1161.discovery_holdout_subsets())
    excluded.update(p1161.confirmation_holdout_subsets())
    excluded.update(p1163.registry_subsets())
    rng = np.random.default_rng(HOLDOUT_SEED)
    result = []
    for cardinality in (3, 4):
        population = [
            row
            for row in itertools.combinations(range(len(p1163.sites())), cardinality)
            if row not in excluded
        ]
        selected = sorted(
            rng.choice(len(population), size=HOLDOUT_PER_CARDINALITY, replace=False).tolist()
        )
        result.extend(population[int(index)] for index in selected)
    return result


def stress_subsets() -> list[tuple[int, ...]]:
    return [
        tuple(row["subset"])
        for row in p1163.diagnostic_registry()
        if "entry_query_chain" in row["categories"]
    ]


def all_test_subsets() -> list[tuple[int, ...]]:
    return broad_holdout_subsets() + stress_subsets()


def max_lower_prediction(
    calibration: np.ndarray,
    targets: list[tuple[int, ...]],
    maximum_order: int,
) -> np.ndarray:
    lookup = {subset: index for index, subset in enumerate(calibration_subsets())}
    prediction = np.zeros(calibration.shape[:-1] + (len(targets),), dtype=np.float64)
    for target_index, target in enumerate(targets):
        target_set = set(target)
        lower_indices = [
            index
            for subset, index in lookup.items()
            if len(subset) <= maximum_order and set(subset).issubset(target_set)
        ]
        prediction[..., target_index] = np.max(calibration[..., lower_indices], axis=-1)
    return prediction


def prior_artifacts() -> dict[str, Any]:
    return {
        "final": p1163.read_json(p1163.OUT_ROOT / "analysis/final.json"),
        "audit": p1163.read_json(p1163.OUT_ROOT / "audit/independent_audit.json"),
        "diagnostic": p1163.read_json(
            p1163.OUT_ROOT / "analysis/posthoc_sublattice_coverage_diagnostic.json"
        ),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1164 artifacts")
    prior = prior_artifacts()
    broad = broad_holdout_subsets()
    stress = stress_subsets()
    earlier = set(p1161.discovery_holdout_subsets()) | set(p1161.confirmation_holdout_subsets()) | set(
        p1163.registry_subsets()
    )
    checks = {
        "phase1163_exception_confirmed": prior["final"]["operational_exception_replication_confirmed"],
        "phase1163_audit_passed": prior["audit"]["all_checks_passed"],
        "posthoc_candidate_non_upgrading": prior["diagnostic"]["evidence_upgrade_forbidden"],
        "posthoc_max_pair_advantage_positive": prior["diagnostic"]["coverage_baseline"]["all_registry_max_pair_advantage_median"] > 0.20,
        "calibration_count": len(calibration_subsets()) == 121,
        "broad_holdout_count": len(broad) == 2 * HOLDOUT_PER_CARDINALITY,
        "broad_holdout_triple_quad": {len(row) for row in broad} == {3, 4},
        "broad_holdout_unique": len(broad) == len(set(broad)),
        "broad_holdout_new": not bool(set(broad).intersection(earlier)),
        "stress_count": len(stress) == 12,
        "stress_disjoint_from_broad": not bool(set(stress).intersection(broad)),
        "predictions_before_outcomes": True,
        "one_shot_confirmation": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": "prospective max-lower-pair coverage response confirmation",
        "source_digests": {
            "phase1163_final": prior["final"]["final_digest"],
            "phase1163_audit": prior["audit"]["audit_digest"],
            "phase1163_posthoc_diagnostic": prior["diagnostic"]["diagnostic_digest"],
        },
        "source_hashes": {
            "primary_script": p1163.sha256_file(SCRIPT),
            "audit_script": p1163.sha256_file(AUDIT_SCRIPT),
            "phase1163_script": p1163.sha256_file(P1163_SCRIPT),
            "phase1163_diagnostic_script": p1163.sha256_file(P1163_DIAGNOSTIC_SCRIPT),
        },
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_count": len(ARCHITECTURES) * REPLICATES,
        "factors": list(FACTORS),
        "calibration_subsets": [list(row) for row in calibration_subsets()],
        "broad_holdout_subsets": [list(row) for row in broad],
        "stress_subsets": [list(row) for row in stress],
        "algorithms": list(ALGORITHMS),
        "primary_algorithm": "max_pair",
        "primary_algorithm_formula": "maximum calibration response over contained null/single/pair subsets",
        "broad_holdout_seed": HOLDOUT_SEED,
        "holdout_per_cardinality": HOLDOUT_PER_CARDINALITY,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "max-lower-pair predicts new triple/quad ordered schedules in new networks and beats additive pairwise plus max-single baselines",
        "secondary_endpoint": "the predeclared query-chain stress family, including A*, remains predictable",
        "allowed_decisions": ["coverage_response_rule_confirmed", "coverage_response_rule_not_confirmed"],
        "hard_stops": [
            "All algorithm predictions are sealed before any holdout response is generated.",
            "Failure may not be repaired by clipping, changing max to another order statistic, changing thresholds, or resampling schedules.",
            "Success is limited to this deterministic micro-task and full-residual ordered patch operation.",
            "A predictive coverage rule is not a recovered natural neural graph or language mechanism.",
            "This algorithm branch closes after one independent confirmation regardless of outcome.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = p1163.digest(protocol)
    p1163.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    p1163.write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(checks.values()),
            "all_checks_passed": all(checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(p1163.canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if p1163.digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1163_script", P1163_SCRIPT),
        ("phase1163_diagnostic_script", P1163_DIAGNOSTIC_SCRIPT),
    ):
        if p1163.sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    if protocol["broad_holdout_subsets"] != [list(row) for row in broad_holdout_subsets()]:
        raise RuntimeError("holdout registry drift")
    return protocol


def run_calibration_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/calibration"
    if root.exists():
        raise RuntimeError("refusing to overwrite calibration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_rows = []
    truth_rows = []
    training_rows = []
    diagnostic_rows = []
    model_arrays = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(architecture, replicate)
            identifier = model_id(seed)
            lexicon = source.make_lexicon(seed + 18017)
            model, training = source.train_model(config, seed, lexicon, device)
            if not training["qualified"]:
                raise RuntimeError(f"behavior gate failed: {identifier}")
            factor_arrays = []
            factor_diagnostics = {}
            for factor in FACTORS:
                matrices, detail = p1163.ordered_factor_surfaces(
                    model, config, lexicon, factor, calibration_subsets(), ("matched",)
                )
                factor_arrays.append(np.median(matrices["matched"], axis=1).astype(np.float32))
                factor_diagnostics[factor] = detail
            model_arrays.append(np.stack(factor_arrays, axis=0))
            public_rows.append({"model_id": identifier, "factor_count": len(FACTORS)})
            truth_rows.append(
                {
                    "model_id": identifier,
                    "architecture": architecture,
                    "replicate": replicate,
                    "seed": seed,
                    "lexicon_digest": p1163.digest(lexicon),
                }
            )
            training_rows.append({"model_id": identifier, **training})
            diagnostic_rows.append({"model_id": identifier, "factor": factor_diagnostics})
            checkpoint = root / "checkpoints" / f"{identifier}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(p1163.checkpoint_payload(model, config, lexicon), checkpoint)
            del model
            torch.cuda.empty_cache()
    response = np.stack(model_arrays, axis=0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "calibration_responses.npz", response=response)
    p1163.write_jsonl(root / "public_manifest.jsonl", public_rows)
    p1163.write_jsonl(root / "sealed_truth.jsonl", truth_rows)
    p1163.write_jsonl(root / "training_metrics.jsonl", training_rows)
    p1163.write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(public_rows) == protocol["model_count"],
        "all_models_qualified": all(row["qualified"] for row in training_rows),
        "behavior_accuracy": min(row["accuracy"] for row in training_rows) >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": min(row["minimum_probability"] for row in training_rows) >= THRESHOLDS["behavior_min_probability_min"],
        "finite": np.isfinite(response).all(),
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "null": float(np.max(np.abs(response[:, :, 0]))) <= THRESHOLDS["null_abs_max"],
        "architecture_hidden": all("architecture" not in row for row in public_rows),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "response_shape": list(response.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "denominator_min": denominator_min,
        "null_max_abs": float(np.max(np.abs(response[:, :, 0]))),
        "calibration_pack_sha256": p1163.sha256_file(root / "calibration_responses.npz"),
        "checks": checks,
        "calibration_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    summary = p1163.read_json(OUT_ROOT / "runs/calibration/summary.json")
    if not summary["calibration_gate_passed"]:
        raise RuntimeError("calibration gate failed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout outcomes already exist")
    prediction_root = OUT_ROOT / "predictions"
    if prediction_root.exists():
        raise RuntimeError("refusing to overwrite predictions")
    with np.load(OUT_ROOT / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    targets = all_test_subsets()
    predictions = {}
    for algorithm in ("cardinality", "layout", "main", "pairwise"):
        output = np.zeros(calibration.shape[:-1] + (len(targets),), dtype=np.float64)
        for model_index in range(calibration.shape[0]):
            for factor_index in range(calibration.shape[1]):
                coefficient = p1161.fit_coefficients(
                    algorithm, calibration_subsets(), calibration[model_index, factor_index]
                )
                output[model_index, factor_index] = p1161.predict_values(
                    algorithm, coefficient, targets
                )
        predictions[algorithm] = output.astype(np.float32)
    predictions["max_single"] = max_lower_prediction(calibration, targets, 1).astype(np.float32)
    predictions["max_pair"] = max_lower_prediction(calibration, targets, 2).astype(np.float32)
    prediction_root.mkdir(parents=True)
    np.savez_compressed(prediction_root / "sealed_predictions.npz", **predictions)
    metadata = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "algorithms": list(ALGORITHMS),
        "primary_algorithm": "max_pair",
        "test_subset_ids": [p1163.subset_id(row) for row in targets],
        "broad_count": len(broad_holdout_subsets()),
        "stress_count": len(stress_subsets()),
        "holdout_outcomes_absent_at_sealing": True,
        "architecture_labels_used": False,
        "calibration_pack_sha256": summary["calibration_pack_sha256"],
        "prediction_pack_sha256": p1163.sha256_file(prediction_root / "sealed_predictions.npz"),
    }
    metadata["prediction_digest"] = p1163.digest(metadata)
    p1163.write_json(prediction_root / "metadata.json", metadata)
    print(p1163.canonical(metadata))


def run_holdout_command() -> None:
    protocol = verify_protocol()
    metadata = p1163.read_json(OUT_ROOT / "predictions/metadata.json")
    if not metadata["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid prediction seal")
    root = OUT_ROOT / "runs/holdout"
    if root.exists():
        raise RuntimeError("refusing to overwrite holdout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    calibration_root = OUT_ROOT / "runs/calibration"
    public = p1163.read_jsonl(calibration_root / "public_manifest.jsonl")
    truth = p1163.read_jsonl(calibration_root / "sealed_truth.jsonl")
    targets = all_test_subsets()
    arrays = []
    diagnostic_rows = []
    for public_row, truth_row in zip(public, truth, strict=True):
        if public_row["model_id"] != truth_row["model_id"]:
            raise RuntimeError("manifest order mismatch")
        model, config, lexicon = p1163.load_checkpoint(
            calibration_root / "checkpoints" / f"{public_row['model_id']}.pt", device
        )
        factor_arrays = []
        factor_diagnostics = {}
        for factor in FACTORS:
            matrices, detail = p1163.ordered_factor_surfaces(
                model, config, lexicon, factor, targets, ("matched",)
            )
            factor_arrays.append(np.median(matrices["matched"], axis=1).astype(np.float32))
            factor_diagnostics[factor] = detail
        arrays.append(np.stack(factor_arrays, axis=0))
        diagnostic_rows.append({"model_id": public_row["model_id"], "factor": factor_diagnostics})
        del model
        torch.cuda.empty_cache()
    observed = np.stack(arrays, axis=0)
    root.mkdir(parents=True)
    np.savez_compressed(root / "holdout_responses.npz", response=observed)
    p1163.write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(arrays) == protocol["model_count"],
        "target_count": observed.shape[2] == len(targets),
        "finite": np.isfinite(observed).all(),
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "prediction_integrity": p1163.sha256_file(OUT_ROOT / "predictions/sealed_predictions.npz") == metadata["prediction_pack_sha256"],
        "prediction_precedes_holdout": metadata["created_at_utc"] < p1163.now(),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "response_shape": list(observed.shape),
        "denominator_min": denominator_min,
        "holdout_pack_sha256": p1163.sha256_file(root / "holdout_responses.npz"),
        "checks": checks,
        "holdout_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def calculate_results(
    protocol: dict[str, Any],
    predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    truth: list[dict[str, Any]],
) -> dict[str, Any]:
    broad_count = len(broad_holdout_subsets())
    broad_observed = observed[:, :, :broad_count]
    stress_observed = observed[:, :, broad_count:]
    algorithm_results = {}
    for algorithm in ALGORITHMS:
        broad_prediction = predictions[algorithm][:, :, :broad_count]
        error = np.abs(broad_prediction - broad_observed)
        unit_mae = np.mean(error, axis=2)
        architecture_medians = {}
        for architecture in ARCHITECTURES:
            model_indices = [index for index, row in enumerate(truth) if row["architecture"] == architecture]
            architecture_medians[architecture] = float(np.median(unit_mae[model_indices]))
        algorithm_results[algorithm] = {
            "median_unit_mae": float(np.median(unit_mae)),
            "mean_unit_mae": float(np.mean(unit_mae)),
            "schedule_abs_error_q95": float(np.quantile(error, 0.95)),
            "schedule_abs_error_max": float(np.max(error)),
            "architecture_median_unit_mae": architecture_medians,
            "unit_mae": unit_mae.tolist(),
        }
    max_pair_unit = np.asarray(algorithm_results["max_pair"]["unit_mae"], dtype=np.float64)
    pairwise_unit = np.asarray(algorithm_results["pairwise"]["unit_mae"], dtype=np.float64)
    max_single_unit = np.asarray(algorithm_results["max_single"]["unit_mae"], dtype=np.float64)
    unit_pass = (max_pair_unit <= THRESHOLDS["unit_mae_max"]) & (
        pairwise_unit - max_pair_unit >= THRESHOLDS["unit_pairwise_advantage_min"]
    )
    stress_error = np.abs(predictions["max_pair"][:, :, broad_count:] - stress_observed)
    stress_unit_mae = np.mean(stress_error, axis=2)
    a_star_stress_index = stress_subsets().index(p1163.A_STAR)
    a_star_error = stress_error[:, :, a_star_stress_index]
    checks = {
        "median_unit_mae": algorithm_results["max_pair"]["median_unit_mae"]
        <= THRESHOLDS["max_pair_median_unit_mae_max"],
        "beats_pairwise": float(np.median(pairwise_unit - max_pair_unit))
        >= THRESHOLDS["max_pair_pairwise_advantage_min"],
        "beats_max_single": float(np.median(max_single_unit - max_pair_unit))
        >= THRESHOLDS["max_pair_max_single_advantage_min"],
        "unit_pass": int(np.sum(unit_pass)) >= THRESHOLDS["unit_pass_min"],
        "unit_total": int(unit_pass.size) == THRESHOLDS["unit_total"],
        "architecture_mae": all(
            value <= THRESHOLDS["architecture_median_mae_max"]
            for value in algorithm_results["max_pair"]["architecture_median_unit_mae"].values()
        ),
        "schedule_q95": algorithm_results["max_pair"]["schedule_abs_error_q95"]
        <= THRESHOLDS["schedule_abs_error_q95_max"],
        "stress_mae": float(np.median(stress_unit_mae)) <= THRESHOLDS["stress_median_unit_mae_max"],
        "a_star_stress": float(np.median(a_star_error))
        <= THRESHOLDS["stress_a_star_abs_error_median_max"],
    }
    confirmed = all(checks.values())
    return {
        "decision": "coverage_response_rule_confirmed" if confirmed else "coverage_response_rule_not_confirmed",
        "coverage_response_rule_confirmed": confirmed,
        "unit_count": int(unit_pass.size),
        "broad_schedule_count": broad_count,
        "stress_schedule_count": len(stress_subsets()),
        "algorithm_results": algorithm_results,
        "max_pair_pairwise_advantage_median": float(np.median(pairwise_unit - max_pair_unit)),
        "max_pair_max_single_advantage_median": float(np.median(max_single_unit - max_pair_unit)),
        "unit_pass_count": int(np.sum(unit_pass)),
        "stress_median_unit_mae": float(np.median(stress_unit_mae)),
        "stress_a_star_abs_error_median": float(np.median(a_star_error)),
        "checks": checks,
        "claim_scope": "predictive upper-envelope rule for ordered full-residual interventions in one deterministic micro-task",
        "not_recovered": ["natural neural graph", "exact semantic identity", "pretrained language mechanism"],
    }


def score_command() -> None:
    protocol = verify_protocol()
    metadata = p1163.read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = p1163.read_json(OUT_ROOT / "runs/holdout/summary.json")
    if not holdout_summary["holdout_gate_passed"]:
        raise RuntimeError("holdout gate failed")
    with np.load(OUT_ROOT / "predictions/sealed_predictions.npz") as pack:
        predictions = {algorithm: np.asarray(pack[algorithm], dtype=np.float64) for algorithm in ALGORITHMS}
    with np.load(OUT_ROOT / "runs/holdout/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    truth = p1163.read_jsonl(OUT_ROOT / "runs/calibration/sealed_truth.jsonl")
    results = calculate_results(protocol, predictions, observed, truth)
    integrity_checks = {
        "prediction_integrity": p1163.sha256_file(OUT_ROOT / "predictions/sealed_predictions.npz") == metadata["prediction_pack_sha256"],
        "holdout_integrity": p1163.sha256_file(OUT_ROOT / "runs/holdout/holdout_responses.npz") == holdout_summary["holdout_pack_sha256"],
        "prediction_precedes_holdout": metadata["created_at_utc"] < holdout_summary["created_at_utc"],
        "one_shot_branch_closed": True,
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"integrity checks failed: {integrity_checks}")
    score = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "results": results,
        "integrity_checks": integrity_checks,
        "branch_status": "closed_after_independent_confirmation",
    }
    score["score_digest"] = p1163.digest(score)
    p1163.write_json(OUT_ROOT / "analysis/score.json", score)
    print(p1163.canonical({"decision": results["decision"], "checks": results["checks"], "key_metrics": {"max_pair_mae": results["algorithm_results"]["max_pair"]["median_unit_mae"], "pairwise_mae": results["algorithm_results"]["pairwise"]["median_unit_mae"], "max_single_mae": results["algorithm_results"]["max_single"]["median_unit_mae"], "unit_pass_count": results["unit_pass_count"]}, "score_digest": score["score_digest"]}))


def finalize_command() -> None:
    protocol = verify_protocol()
    score = p1163.read_json(OUT_ROOT / "analysis/score.json")
    results = score["results"]
    final = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": results["decision"],
        "coverage_response_rule_confirmed": results["coverage_response_rule_confirmed"],
        "natural_mechanism_recovered": False,
        "branch_status": "closed_after_independent_confirmation",
        "auto_continue": False,
        "auto_continue_reason": "The frozen coverage candidate has received its one independent large holdout test; further work must change task family or intervention semantics, not retune this rule.",
        "non_implications": [
            "Prediction under full-residual patching does not recover the unpatched natural computation graph.",
            "A maximum lower-set rule does not prove neurons literally compute a max operation.",
            "The result is not evidence about Qwen3, GLM4, DS7B, or natural language.",
        ],
    }
    final["final_digest"] = p1163.digest(final)
    p1163.write_json(OUT_ROOT / "analysis/final.json", final)
    print(p1163.canonical(final))


def smoke_command() -> None:
    print(p1163.canonical({
        "calibration_count": len(calibration_subsets()),
        "broad_holdout_count": len(broad_holdout_subsets()),
        "stress_count": len(stress_subsets()),
        "holdout_disjoint": not bool(set(broad_holdout_subsets()).intersection(p1163.registry_subsets())),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "protocol", "run-calibration", "seal-predictions", "run-holdout", "score", "finalize"))
    args = parser.parse_args()
    commands = {
        "smoke": smoke_command,
        "protocol": protocol_command,
        "run-calibration": run_calibration_command,
        "seal-predictions": seal_predictions_command,
        "run-holdout": run_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
