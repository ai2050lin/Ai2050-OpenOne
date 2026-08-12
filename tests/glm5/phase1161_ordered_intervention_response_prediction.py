#!/usr/bin/env python3
"""Predict held-out ordered multi-site interventions in freely trained Transformers.

This phase tests a bounded inverse problem after Phase1160 abstained on stable
factor identity.  A subset of sites is interpreted as a mask over a fixed
chronological intervention schedule, not as an unordered physical hyperedge.
The estimator observes null, singleton, and pair interventions for each
trained network and must predict sealed triple and quadruple interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1161_ordered_intervention_response_prediction_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1161_ordered_intervention_response_prediction"
SOURCE1160 = ROOT / "tests/glm5/phase1160_interior_factor_identity_purification.py"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1159_free_transformer_causal_use_external_validity as source  # noqa: E402


PHASE = 1161
FACTORS = source.FACTORS
ROLES = source.ROLES
ARCHITECTURES = source.ARCHITECTURES
INTERIOR_DEPTHS = (0.25, 0.5, 0.75)
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
RIDGE = 1e-3
ALGORITHMS = ("cardinality", "layout", "main", "pairwise")
STRUCTURAL_ALGORITHMS = ("main", "pairwise")
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "denominator_min": 1e-5,
    "discovery_validation_median_mae_max": 0.15,
    "discovery_validation_median_correlation_min": 0.75,
    "discovery_validation_unit_mae_max": 0.20,
    "discovery_validation_unit_correlation_min": 0.60,
    "discovery_validation_unit_pass_min": 5,
    "discovery_validation_unit_total": 6,
    "discovery_layout_mae_advantage_min": 0.03,
    "confirmation_median_mae_max": 0.15,
    "confirmation_median_correlation_min": 0.75,
    "confirmation_unit_mae_max": 0.20,
    "confirmation_unit_correlation_min": 0.60,
    "confirmation_unit_pass_min": 20,
    "confirmation_unit_total": 24,
    "confirmation_architecture_median_mae_max": 0.18,
    "confirmation_layout_mae_advantage_min": 0.03,
    "complexity_tie_mae": 0.01,
    "null_abs_max": 1e-8,
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sites() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "depth": float(depth),
            "role": role,
            "site_id": f"d{depth:.2f}:{role}",
        }
        for index, (depth, role) in enumerate(itertools.product(INTERIOR_DEPTHS, ROLES))
    ]


def subset_id(subset: tuple[int, ...]) -> str:
    return "empty" if not subset else ".".join(f"s{value:02d}" for value in subset)


def calibration_subsets() -> list[tuple[int, ...]]:
    count = len(sites())
    return [tuple()] + [(index,) for index in range(count)] + list(itertools.combinations(range(count), 2))


def sampled_holdout_subsets(seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, ...]] = []
    for cardinality in (3, 4):
        population = list(itertools.combinations(range(len(sites())), cardinality))
        chosen = rng.choice(len(population), size=64, replace=False)
        rows.extend(population[int(index)] for index in sorted(chosen.tolist()))
    return rows


def discovery_holdout_subsets() -> list[tuple[int, ...]]:
    return sampled_holdout_subsets(1161001)


def confirmation_holdout_subsets() -> list[tuple[int, ...]]:
    discovery = set(discovery_holdout_subsets())
    candidate = sampled_holdout_subsets(1161002)
    if discovery.intersection(candidate):
        # Deterministically replace accidental overlap without inspecting model outcomes.
        rng = np.random.default_rng(1161003)
        candidate = []
        for cardinality in (3, 4):
            population = [
                row
                for row in itertools.combinations(range(len(sites())), cardinality)
                if row not in discovery
            ]
            chosen = rng.choice(len(population), size=64, replace=False)
            candidate.extend(population[int(index)] for index in sorted(chosen.tolist()))
    return candidate


def model_seed(split: str, architecture: str, replicate: int) -> int:
    base = 1161100 if split == "discovery" else 1161900
    return base + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def model_id(split: str, seed: int) -> str:
    return digest({"phase": PHASE, "split": split, "seed": seed})[:16]


def features(algorithm: str, subsets: list[tuple[int, ...]]) -> np.ndarray:
    masks = np.zeros((len(subsets), len(sites())), dtype=np.float64)
    for row_index, subset in enumerate(subsets):
        masks[row_index, list(subset)] = 1.0
    cardinality = np.sum(masks, axis=1, keepdims=True)
    if algorithm == "cardinality":
        return np.concatenate([np.ones((len(subsets), 1)), cardinality, cardinality**2], axis=1)
    if algorithm == "layout":
        depth_counts = np.stack(
            [np.sum(masks[:, [i for i, site in enumerate(sites()) if site["depth"] == depth]], axis=1) for depth in INTERIOR_DEPTHS],
            axis=1,
        )
        role_counts = np.stack(
            [np.sum(masks[:, [i for i, site in enumerate(sites()) if site["role"] == role]], axis=1) for role in ROLES],
            axis=1,
        )
        return np.concatenate([np.ones((len(subsets), 1)), cardinality, cardinality**2, depth_counts, role_counts], axis=1)
    if algorithm == "main":
        return np.concatenate([np.ones((len(subsets), 1)), masks], axis=1)
    if algorithm == "pairwise":
        pair_columns = [masks[:, left] * masks[:, right] for left, right in itertools.combinations(range(len(sites())), 2)]
        pairs = np.stack(pair_columns, axis=1)
        return np.concatenate([np.ones((len(subsets), 1)), masks, pairs], axis=1)
    raise ValueError(algorithm)


def fit_coefficients(algorithm: str, subsets: list[tuple[int, ...]], values: np.ndarray) -> np.ndarray:
    design = features(algorithm, subsets)
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(values, dtype=np.float64))


def predict_values(algorithm: str, coefficients: np.ndarray, subsets: list[tuple[int, ...]]) -> np.ndarray:
    return features(algorithm, subsets) @ np.asarray(coefficients, dtype=np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    error = prediction - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "correlation": correlation(prediction, truth),
    }


def source_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    root = ROOT / "tests/glm5/result/phase1160_interior_factor_identity_purification"
    return read_json(root / "analysis/final.json"), read_json(root / "audit/independent_audit.json")


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1161 artifacts")
    prior, prior_audit = source_artifacts()
    calibration = calibration_subsets()
    discovery_holdout = discovery_holdout_subsets()
    confirmation_holdout = confirmation_holdout_subsets()
    checks = {
        "phase1160_abstained": prior["discovery_decision"] == "abstain",
        "phase1160_confirmation_denied": prior["score_digest"] is None
        and not bool(prior["graph_hyperedge_scan_authorized"]),
        "phase1160_audit_passed": bool(prior_audit["all_checks_passed"]),
        "interior_only": all(site["depth"] not in (0.0, 1.0) for site in sites()),
        "fixed_chronological_schedule": True,
        "calibration_null_single_pair_only": {len(row) for row in calibration} == {0, 1, 2},
        "discovery_holdout_triple_quad_only": {len(row) for row in discovery_holdout} == {3, 4},
        "confirmation_holdout_triple_quad_only": {len(row) for row in confirmation_holdout} == {3, 4},
        "discovery_confirmation_holdout_disjoint": not bool(set(discovery_holdout).intersection(confirmation_holdout)),
        "holdouts_absent_from_calibration": not bool(set(calibration).intersection(discovery_holdout + confirmation_holdout)),
        "strong_baselines_present": set(("cardinality", "layout")).issubset(ALGORITHMS),
        "prediction_before_confirmation_outcome": True,
        "architecture_labels_forbidden_during_selection": True,
        "abstention_required": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": "ordered multi-site intervention response prediction",
        "source_phase1160_final_digest": prior["final_digest"],
        "source_phase1160_audit_digest": prior_audit["audit_digest"],
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
            "phase1160_script": sha256_file(SOURCE1160),
        },
        "factors": list(FACTORS),
        "roles": list(ROLES),
        "interior_depths": list(INTERIOR_DEPTHS),
        "sites": sites(),
        "site_count": len(sites()),
        "intervention_algebra": "site subsets mask a fixed early-to-late schedule; same-depth replacements are simultaneous",
        "response": "median normalized donor-minus-receiver answer margin over the frozen case panel",
        "calibration_subsets": [list(row) for row in calibration],
        "discovery_holdout_subsets": [list(row) for row in discovery_holdout],
        "confirmation_holdout_subsets": [list(row) for row in confirmation_holdout],
        "algorithms": list(ALGORITHMS),
        "structural_algorithms": list(STRUCTURAL_ALGORITHMS),
        "ridge": RIDGE,
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "thresholds": THRESHOLDS,
        "primary_endpoint": "predict unseen triple/quadruple ordered interventions from null/singleton/pair calibration and beat layout baselines",
        "allowed_outputs": ["ordered_response_predictable", "abstain"],
        "hard_stops": [
            "Discovery selects the simplest structural estimator within the frozen MAE tie window.",
            "Discovery validation failure denies confirmation model training.",
            "Confirmation predictions are sealed before any confirmation holdout intervention is run.",
            "A passing predictor is not called a recovered causal graph, a stable factor identity, or a language mechanism.",
            "Pairwise coefficients are conditional coefficients under the fixed intervention schedule, not symmetric physical hyperedges.",
            "No natural-language, pretrained-model, neuron, head, or full-mechanism claim is authorized.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(checks.values()),
            "all_checks_passed": all(checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1160_script", SOURCE1160),
    ):
        if sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    return protocol


def ordered_surface(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    split: str,
    factor: str,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, float]]:
    device = next(model.parameters()).device
    receiver_cpu, donor_cpu, _control_cpu, receiver_target_cpu, donor_target_cpu, positions_cpu = source.scan_batch(
        lexicon, split, factor
    )
    receiver = receiver_cpu.to(device)
    donor = donor_cpu.to(device)
    receiver_targets = receiver_target_cpu.to(device)
    donor_targets = donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    candidates = source.answer_ids(lexicon, device)
    batch_index = torch.arange(len(receiver), device=device)
    role_positions = {role: positions[:, ROLES.index(role)] for role in ROLES}
    actual_by_depth = {depth: source.actual_depth_index(config, depth) for depth in INTERIOR_DEPTHS}
    site_rows = sites()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw = model(receiver)
        donor_raw, donor_states = model(donor, return_states=True)
    receiver_logits = source.candidate_logits(receiver_raw, candidates)
    donor_logits = source.candidate_logits(donor_raw, candidates)
    base_margin = source.target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = source.target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    if float(torch.min(denominator).item()) <= THRESHOLDS["denominator_min"]:
        raise RuntimeError(f"nonpositive denominator: {split}/{factor}")
    values = []
    for subset in subsets:
        if not subset:
            values.append(0.0)
            continue
        selected = {int(value) for value in subset}
        hidden = model.embed(receiver)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for layer_index, block in enumerate(model.blocks, start=1):
                hidden = block(hidden)
                patch_indices = [
                    index
                    for index in selected
                    if actual_by_depth[float(site_rows[index]["depth"])] == layer_index
                ]
                if patch_indices:
                    hidden = hidden.clone()
                    for site_index in patch_indices:
                        site = site_rows[site_index]
                        token_positions = role_positions[str(site["role"])]
                        hidden[batch_index, token_positions] = donor_states[layer_index][batch_index, token_positions]
            patched_raw = model.lm_head(model.final_norm(hidden))
        patched_logits = source.candidate_logits(patched_raw, candidates)
        effect = (
            source.target_margin(patched_logits, donor_targets, receiver_targets) - base_margin
        ) / denominator
        values.append(float(torch.median(effect.float()).item()))
    result = np.asarray(values, dtype=np.float32)
    return result, {
        "case_count": int(len(receiver)),
        "denominator_min": float(torch.min(denominator).item()),
        "denominator_median": float(torch.median(denominator).item()),
        "finite_fraction": float(np.isfinite(result).mean()),
        "null_abs": float(abs(result[0])) if subsets and not subsets[0] else 0.0,
    }


def collect_model_surfaces(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    split: str,
    subset_registry: list[tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.zeros((len(FACTORS), len(subset_registry)), dtype=np.float32)
    diagnostics: dict[str, Any] = {"factor": {}}
    for factor_index, factor in enumerate(FACTORS):
        values, detail = ordered_surface(model, config, lexicon, split, factor, subset_registry)
        rows[factor_index] = values
        diagnostics["factor"][factor] = detail
    return rows, diagnostics


def checkpoint_payload(model: torch.nn.Module, config: Any, lexicon: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "lexicon": lexicon,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = source.ModelConfig(**payload["config"])
    model = source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def run_models(
    split: str,
    subset_registry: list[tuple[int, ...]],
    output_name: str,
    train_new: bool,
) -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs" / split
    if split == "discovery":
        if root.exists():
            raise RuntimeError("refusing to overwrite discovery run")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if (root / output_name).exists():
            raise RuntimeError(f"refusing to overwrite {root / output_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_path = root / "public_manifest.jsonl"
    truth_path = root / "sealed_truth.jsonl"
    if train_new:
        public_rows: list[dict[str, Any]] = []
        truth_rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        diagnostics_rows: list[dict[str, Any]] = []
        arrays = []
        for architecture, config in ARCHITECTURES.items():
            for replicate in range(REPLICATES):
                seed = model_seed(split, architecture, replicate)
                identifier = model_id(split, seed)
                lexicon = source.make_lexicon(seed + 17)
                model, training = source.train_model(config, seed, lexicon, device)
                if not training["qualified"]:
                    raise RuntimeError(f"training failed: {split}/{identifier}")
                values, diagnostics = collect_model_surfaces(model, config, lexicon, split, subset_registry)
                arrays.append(values)
                public_rows.append(
                    {
                        "model_id": identifier,
                        "analysis_partition": "fit" if replicate in FIT_REPLICATES else "validation",
                        "factor_count": len(FACTORS),
                        "subset_count": len(subset_registry),
                    }
                )
                truth_rows.append(
                    {
                        "model_id": identifier,
                        "architecture": architecture,
                        "replicate": replicate,
                        "seed": seed,
                    }
                )
                training_rows.append({"model_id": identifier, **training})
                diagnostics_rows.append({"model_id": identifier, **diagnostics})
                checkpoint = root / "checkpoints" / f"{identifier}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, config, lexicon), checkpoint)
                del model
                torch.cuda.empty_cache()
        write_jsonl(public_path, public_rows)
        write_jsonl(truth_path, truth_rows)
        write_jsonl(root / "training_metrics.jsonl", training_rows)
        write_jsonl(root / "diagnostics.jsonl", diagnostics_rows)
        stacked = np.stack(arrays)
    else:
        public_rows = read_jsonl(public_path)
        truth_rows = read_jsonl(truth_path)
        diagnostics_rows = []
        arrays = []
        for public, truth in zip(public_rows, truth_rows, strict=True):
            if public["model_id"] != truth["model_id"]:
                raise RuntimeError("manifest order mismatch")
            checkpoint = root / "checkpoints" / f"{public['model_id']}.pt"
            model, config, lexicon = load_checkpoint(checkpoint, device)
            values, diagnostics = collect_model_surfaces(model, config, lexicon, split, subset_registry)
            arrays.append(values)
            diagnostics_rows.append({"model_id": public["model_id"], **diagnostics})
            del model
            torch.cuda.empty_cache()
        write_jsonl(root / "holdout_diagnostics.jsonl", diagnostics_rows)
        stacked = np.stack(arrays)
    output_path = root / output_name
    np.savez_compressed(output_path, response=stacked)
    training_rows = read_jsonl(root / "training_metrics.jsonl")
    denominator_min = min(
        detail["denominator_min"]
        for row in diagnostics_rows
        for detail in row["factor"].values()
    )
    finite_fraction = float(np.isfinite(stacked).mean())
    null_max = float(np.max(np.abs(stacked[:, :, 0]))) if subset_registry and not subset_registry[0] else 0.0
    summary_name = "summary.json" if split == "discovery" else ("calibration_summary.json" if train_new else "holdout_summary.json")
    summary = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "output_name": output_name,
        "model_count": len(public_rows),
        "factor_count": len(FACTORS),
        "subset_count": len(subset_registry),
        "response_shape": list(stacked.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "finite_fraction": finite_fraction,
        "denominator_min": denominator_min,
        "null_max_abs": null_max,
        "effect_pack_sha256": sha256_file(output_path),
        "public_manifest_sha256": sha256_file(public_path),
        "sealed_truth_sha256": sha256_file(truth_path),
    }
    checks = {
        "model_count": len(public_rows) == len(ARCHITECTURES) * REPLICATES,
        "behavior_accuracy": summary["behavior_accuracy_min"] >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": summary["behavior_min_probability_min"] >= THRESHOLDS["behavior_min_probability_min"],
        "finite": finite_fraction >= THRESHOLDS["finite_fraction_min"],
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "null": null_max <= THRESHOLDS["null_abs_max"],
        "architecture_hidden_from_public": all("architecture" not in row for row in public_rows),
    }
    summary["checks"] = checks
    summary["run_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(root / summary_name, summary)
    print(canonical({"split": split, "output": output_name, "summary_digest": summary["summary_digest"], "checks": checks}))


def run_discovery_command() -> None:
    calibration = calibration_subsets()
    holdout = discovery_holdout_subsets()
    run_models("discovery", calibration + holdout, "responses.npz", train_new=True)


def evaluate_algorithms(
    calibration_values: np.ndarray,
    holdout_values: np.ndarray,
    holdout_subsets: list[tuple[int, ...]],
    model_indices: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    calibration = calibration_subsets()
    for algorithm in ALGORITHMS:
        units = []
        for model_index in model_indices:
            for factor_index, factor in enumerate(FACTORS):
                coefficients = fit_coefficients(algorithm, calibration, calibration_values[model_index, factor_index])
                predicted = predict_values(algorithm, coefficients, holdout_subsets)
                detail = metrics(predicted, holdout_values[model_index, factor_index])
                units.append({"model_index": model_index, "factor": factor, **detail})
        results[algorithm] = {
            "unit_metrics": units,
            "median_mae": float(np.median([row["mae"] for row in units])),
            "median_correlation": float(np.median([row["correlation"] for row in units])),
            "unit_pass_count": int(
                sum(
                    row["mae"] <= THRESHOLDS["discovery_validation_unit_mae_max"]
                    and row["correlation"] >= THRESHOLDS["discovery_validation_unit_correlation_min"]
                    for row in units
                )
            ),
            "unit_count": len(units),
        }
    return results


def fit_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/discovery"
    summary = read_json(root / "summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("discovery run gate failed")
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "responses.npz") as pack:
        response = np.asarray(pack["response"], dtype=np.float64)
    calibration_count = len(calibration_subsets())
    calibration_values = response[:, :, :calibration_count]
    holdout_values = response[:, :, calibration_count:]
    fit_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "validation"]
    fit_metrics = evaluate_algorithms(calibration_values, holdout_values, discovery_holdout_subsets(), fit_indices)
    structural_mae = {name: fit_metrics[name]["median_mae"] for name in STRUCTURAL_ALGORITHMS}
    best_mae = min(structural_mae.values())
    selected = next(
        name
        for name in STRUCTURAL_ALGORITHMS
        if structural_mae[name] <= best_mae + THRESHOLDS["complexity_tie_mae"]
    )
    validation_metrics = evaluate_algorithms(
        calibration_values, holdout_values, discovery_holdout_subsets(), validation_indices
    )
    selected_metrics = validation_metrics[selected]
    layout_metrics = validation_metrics["layout"]
    layout_advantage = layout_metrics["median_mae"] - selected_metrics["median_mae"]
    checks = {
        "validation_median_mae": selected_metrics["median_mae"]
        <= THRESHOLDS["discovery_validation_median_mae_max"],
        "validation_median_correlation": selected_metrics["median_correlation"]
        >= THRESHOLDS["discovery_validation_median_correlation_min"],
        "validation_unit_pass": selected_metrics["unit_pass_count"]
        >= THRESHOLDS["discovery_validation_unit_pass_min"],
        "validation_unit_total": selected_metrics["unit_count"]
        == THRESHOLDS["discovery_validation_unit_total"],
        "beats_layout_baseline": layout_advantage >= THRESHOLDS["discovery_layout_mae_advantage_min"],
    }
    fit = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "selection_rule": "simplest structural algorithm within 0.01 MAE of the best fit-partition structural algorithm",
        "selected_algorithm": selected,
        "validation_layout_mae_advantage": layout_advantage,
        "checks": checks,
        "confirmation_authorized": all(checks.values()),
        "discovery_summary_digest": summary["summary_digest"],
    }
    fit["fit_digest"] = digest(fit)
    write_json(OUT_ROOT / "analysis/discovery_fit.json", fit)
    print(canonical({"selected_algorithm": selected, "checks": checks, "confirmation_authorized": fit["confirmation_authorized"], "fit_digest": fit["fit_digest"]}))


def run_confirmation_calibration_command() -> None:
    fit = read_json(OUT_ROOT / "analysis/discovery_fit.json")
    if not fit["confirmation_authorized"]:
        raise RuntimeError("discovery denied confirmation")
    run_models("confirmation", calibration_subsets(), "calibration_responses.npz", train_new=True)


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/discovery_fit.json")
    if not fit["confirmation_authorized"]:
        raise RuntimeError("discovery denied confirmation")
    calibration_root = OUT_ROOT / "runs/confirmation"
    summary = read_json(calibration_root / "calibration_summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("confirmation calibration run gate failed")
    if (calibration_root / "holdout_responses.npz").exists():
        raise RuntimeError("confirmation holdout outcomes already exist")
    prediction_root = OUT_ROOT / "predictions"
    if prediction_root.exists():
        raise RuntimeError("refusing to overwrite predictions")
    with np.load(calibration_root / "calibration_responses.npz") as pack:
        calibration_values = np.asarray(pack["response"], dtype=np.float64)
    holdout = confirmation_holdout_subsets()
    predictions = {
        name: np.zeros((calibration_values.shape[0], len(FACTORS), len(holdout)), dtype=np.float32)
        for name in ALGORITHMS
    }
    coefficient_rows = []
    for model_index in range(calibration_values.shape[0]):
        for factor_index, factor in enumerate(FACTORS):
            for algorithm in ALGORITHMS:
                coefficients = fit_coefficients(
                    algorithm, calibration_subsets(), calibration_values[model_index, factor_index]
                )
                predictions[algorithm][model_index, factor_index] = predict_values(
                    algorithm, coefficients, holdout
                ).astype(np.float32)
                coefficient_rows.append(
                    {
                        "model_index": model_index,
                        "factor": factor,
                        "algorithm": algorithm,
                        "coefficient_count": int(len(coefficients)),
                        "coefficient_l2": float(np.linalg.norm(coefficients)),
                    }
                )
    prediction_root.mkdir(parents=True)
    np.savez_compressed(prediction_root / "confirmation_predictions.npz", **predictions)
    write_jsonl(prediction_root / "coefficient_diagnostics.jsonl", coefficient_rows)
    metadata = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "selected_algorithm": fit["selected_algorithm"],
        "holdout_subset_ids": [subset_id(row) for row in holdout],
        "holdout_outcomes_absent_at_sealing": True,
        "architecture_labels_used": False,
        "prediction_pack_sha256": sha256_file(prediction_root / "confirmation_predictions.npz"),
        "calibration_pack_sha256": summary["effect_pack_sha256"],
    }
    metadata["prediction_digest"] = digest(metadata)
    write_json(prediction_root / "metadata.json", metadata)
    print(canonical(metadata))


def run_confirmation_holdout_command() -> None:
    verify_protocol()
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    if not metadata["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid prediction seal")
    run_models(
        "confirmation",
        confirmation_holdout_subsets(),
        "holdout_responses.npz",
        train_new=False,
    )


def score_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/discovery_fit.json")
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = read_json(OUT_ROOT / "runs/confirmation/holdout_summary.json")
    if not holdout_summary["run_gate_passed"]:
        raise RuntimeError("confirmation holdout run gate failed")
    with np.load(OUT_ROOT / "predictions/confirmation_predictions.npz") as pack:
        predicted = {name: np.asarray(pack[name], dtype=np.float64) for name in ALGORITHMS}
    with np.load(OUT_ROOT / "runs/confirmation/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    algorithm_results: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        units = []
        for model_index, truth_row in enumerate(truth):
            for factor_index, factor in enumerate(FACTORS):
                detail = metrics(predicted[algorithm][model_index, factor_index], observed[model_index, factor_index])
                units.append(
                    {
                        "model_index": model_index,
                        "architecture": truth_row["architecture"],
                        "factor": factor,
                        **detail,
                    }
                )
        architecture_medians = {
            architecture: float(np.median([row["mae"] for row in units if row["architecture"] == architecture]))
            for architecture in ARCHITECTURES
        }
        algorithm_results[algorithm] = {
            "unit_metrics": units,
            "median_mae": float(np.median([row["mae"] for row in units])),
            "median_correlation": float(np.median([row["correlation"] for row in units])),
            "unit_pass_count": int(
                sum(
                    row["mae"] <= THRESHOLDS["confirmation_unit_mae_max"]
                    and row["correlation"] >= THRESHOLDS["confirmation_unit_correlation_min"]
                    for row in units
                )
            ),
            "unit_count": len(units),
            "architecture_median_mae": architecture_medians,
        }
    selected = fit["selected_algorithm"]
    selected_result = algorithm_results[selected]
    layout_advantage = algorithm_results["layout"]["median_mae"] - selected_result["median_mae"]
    checks = {
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/confirmation_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "selected_algorithm_frozen": selected == metadata["selected_algorithm"],
        "median_mae": selected_result["median_mae"] <= THRESHOLDS["confirmation_median_mae_max"],
        "median_correlation": selected_result["median_correlation"]
        >= THRESHOLDS["confirmation_median_correlation_min"],
        "unit_pass": selected_result["unit_pass_count"] >= THRESHOLDS["confirmation_unit_pass_min"],
        "unit_total": selected_result["unit_count"] == THRESHOLDS["confirmation_unit_total"],
        "architecture_median_mae": all(
            value <= THRESHOLDS["confirmation_architecture_median_mae_max"]
            for value in selected_result["architecture_median_mae"].values()
        ),
        "beats_layout_baseline": layout_advantage >= THRESHOLDS["confirmation_layout_mae_advantage_min"],
    }
    score = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "selected_algorithm": selected,
        "algorithm_results": algorithm_results,
        "layout_mae_advantage": layout_advantage,
        "checks": checks,
        "ordered_response_prediction_confirmed": all(checks.values()),
        "holdout_summary_digest": holdout_summary["summary_digest"],
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", score)
    print(canonical({"selected_algorithm": selected, "selected_result": selected_result, "layout_mae_advantage": layout_advantage, "checks": checks, "score_digest": score["score_digest"]}))


def finalize_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/discovery_fit.json")
    confirmation_executed = (OUT_ROOT / "analysis/confirmation_score.json").exists()
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json") if confirmation_executed else None
    confirmed = bool(score and score["ordered_response_prediction_confirmed"])
    final = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "decision": "ordered_response_predictable" if confirmed else "abstain",
        "selected_algorithm": fit["selected_algorithm"],
        "discovery_confirmation_authorized": bool(fit["confirmation_authorized"]),
        "confirmation_executed": confirmation_executed,
        "ordered_response_prediction_confirmed": confirmed,
        "full_mechanism_recovery_complete": False,
        "stable_factor_identity_recovered": False,
        "causal_graph_recovered": False,
        "physical_hyperedges_recovered": False,
        "claim_scope": (
            "A low-order conditional response model predicts unseen ordered interventions in new freely trained networks."
            if confirmed
            else "The frozen low-order response estimators did not pass the held-out ordered-intervention gate."
        ),
        "non_implications": [
            "Prediction does not identify a unique internal graph.",
            "Pairwise coefficients are not physical hyperedges without independent interventions and invariance.",
            "The task is deterministic and contains every input combination during training, so it does not test compositional generalization.",
            "Common depth-role coordinates and matched donors are supplied to the estimator; this is not fully blind recovery.",
            "The result does not transfer to natural language or pretrained models.",
        ],
        "next_phase_authorized": confirmed,
        "next_phase_scope": (
            "one independent task-family test of the frozen estimator and intervention algebra"
            if confirmed
            else "none; revise the mechanism object before further graph search"
        ),
        "score_digest": score["score_digest"] if score else None,
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def smoke_command() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    config = ARCHITECTURES["compact"]
    seed = 11619991
    lexicon = source.make_lexicon(seed + 17)
    model, training = source.train_model(config, seed, lexicon, device)
    calibration = calibration_subsets()
    holdout = discovery_holdout_subsets()[:8]
    response, diagnostics = collect_model_surfaces(model, config, lexicon, "discovery", calibration + holdout)
    report = {"training_qualified": training["qualified"], "response_shape": list(response.shape), "diagnostics": diagnostics, "metrics": {}}
    for algorithm in ALGORITHMS:
        unit = []
        for factor_index in range(len(FACTORS)):
            coefficients = fit_coefficients(algorithm, calibration, response[factor_index, : len(calibration)])
            prediction = predict_values(algorithm, coefficients, holdout)
            unit.append(metrics(prediction, response[factor_index, len(calibration) :]))
        report["metrics"][algorithm] = unit
    print(canonical(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "protocol",
            "run-discovery",
            "fit",
            "run-confirmation-calibration",
            "seal-predictions",
            "run-confirmation-holdout",
            "score",
            "finalize",
        ),
    )
    args = parser.parse_args()
    commands = {
        "smoke": smoke_command,
        "protocol": protocol_command,
        "run-discovery": run_discovery_command,
        "fit": fit_command,
        "run-confirmation-calibration": run_confirmation_calibration_command,
        "seal-predictions": seal_predictions_command,
        "run-confirmation-holdout": run_confirmation_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
