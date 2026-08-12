#!/usr/bin/env python3
"""Sealed cross-quotient prediction of modular generalization event time.

Phase1171 found 64 delayed trajectories inside one affine task-isomorphism
class.  Phase1172 changes the scientific object before training: twelve tasks
are separated by exact permutation-invariant signatures, stable generalization
is an interval-censored event, and predictors may read training history only.

All models and training summaries are sealed first.  Holdout outcomes for eight
discovery task classes are then revealed and used to freeze a simple predictor.
Only after that seal may four disjoint confirmation classes be evaluated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1172_cross_quotient_event_time_prediction_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1172_cross_quotient_event_time_prediction"
P1171_FINAL = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography/analysis/final.json"
P1171_AUDIT = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography/audit/independent_audit.json"
P1171_REPRESENTATION_AUDIT = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography/audit/endpoint_representation_recompute.json"
PILOT_SCRIPT = ROOT / "tests/glm5_temp/phase1172_cross_quotient_task_pilot.py"
PILOT_RESULT = ROOT / "tests/glm5_temp/phase1172_cross_quotient_task_pilot.json"

PHASE = 1172
MODULUS = 61
MODEL_WIDTH = 128
TRAIN_FRACTION = 0.50
REPLICATES = 8
PREDICTION_CUTOFF = 150
RIDGE_L2 = 1.0
CHECKPOINT_STEPS = (
    25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1250, 1500,
    1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4500,
    5000, 5500, 6000, 7000, 8000, 9000, 10000, 12000,
)
PREDICTION_HORIZONS = (500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12000)
HISTORY_STEPS = (25, 50, 75, 100, 150)
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": max(CHECKPOINT_STEPS),
}
THRESHOLDS = {
    "train_fit_accuracy_min": 0.99,
    "stable_generalization_accuracy_min": 0.90,
    "stable_generalization_adjacent_checkpoint_count": 2,
    "discovery_event_trajectory_min": 24,
    "discovery_censored_trajectory_min": 16,
    "discovery_event_task_class_min": 4,
    "discovery_censored_task_class_min": 2,
    "discovery_distinct_event_upper_bounds_min": 3,
    "confirmation_relative_brier_improvement_min": 0.10,
    "confirmation_class_advantage_min": 3,
    "all_trajectories_must_fit": True,
    "all_logits_must_be_finite": True,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str
    family: str
    formula: str


TASK_SPECS = (
    TaskSpec("affine_relabel", "discovery", "affine", "(7*a + 11*b + 13) mod 61"),
    TaskSpec("shifted_product", "discovery", "bilinear", "((a+3)*(b+5) + 7) mod 61"),
    TaskSpec("left_square_add", "discovery", "quadratic_unary", "((a+7)^2 + 3*b + 11) mod 61"),
    TaskSpec("distance_square", "discovery", "quadratic_relation", "(3*a - 5*b + 7)^2 + 11 mod 61"),
    TaskSpec("maximum_shift", "discovery", "ordered_selector", "max(a+7,b+13) + 17 with residues in 0..60"),
    TaskSpec("quad_mixed", "discovery", "quadratic_mixed", "2*(a+1)^2 + 3*(a+1)*(b+2) + 5*(b+2) + 7 mod 61"),
    TaskSpec("circle_form", "discovery", "quadratic_binary", "2*a^2 + 3*a*b + 5*b^2 + 7 mod 61"),
    TaskSpec("ordered_gate", "discovery", "piecewise_ordered", "2*x+3*z if x<z else 5*x+7*z+11, x=a+7,z=b+13"),
    TaskSpec("square_sum", "confirmation", "quadratic_separable", "(a+2)^2 + 5*(b+9)^2 + 17 mod 61"),
    TaskSpec("left_cube_add", "confirmation", "cubic_unary", "(a+4)^3 + 7*b + 19 mod 61"),
    TaskSpec("xor_shift", "confirmation", "bitwise", "((a xor b) mod 61 + 7) mod 61"),
    TaskSpec("diagonal_bump", "confirmation", "sparse_relation", "2*a+3*b+5 + 11*I[a+7=b+13] mod 61"),
)
TASK_BY_NAME = {task.name: task for task in TASK_SPECS}


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "affine_relabel": lambda a, b: (7 * a + 11 * b + 13) % p,
        "shifted_product": lambda a, b: (((a + 3) % p) * ((b + 5) % p) + 7) % p,
        "left_square_add": lambda a, b: (((a + 7) % p) ** 2 + 3 * b + 11) % p,
        "distance_square": lambda a, b: ((3 * a - 5 * b + 7) ** 2 + 11) % p,
        "maximum_shift": lambda a, b: (max((a + 7) % p, (b + 13) % p) + 17) % p,
        "quad_mixed": lambda a, b: (2 * ((a + 1) % p) ** 2 + 3 * ((a + 1) % p) * ((b + 2) % p) + 5 * ((b + 2) % p) + 7) % p,
        "circle_form": lambda a, b: (2 * a * a + 3 * a * b + 5 * b * b + 7) % p,
        "ordered_gate": lambda a, b: ((2 * ((a + 7) % p) + 3 * ((b + 13) % p)) if ((a + 7) % p) < ((b + 13) % p) else (5 * ((a + 7) % p) + 7 * ((b + 13) % p) + 11)) % p,
        "square_sum": lambda a, b: (((a + 2) % p) ** 2 + 5 * ((b + 9) % p) ** 2 + 17) % p,
        "left_cube_add": lambda a, b: (((a + 4) % p) ** 3 + 7 * b + 19) % p,
        "xor_shift": lambda a, b: (((a ^ b) % p) + 7) % p,
        "diagonal_bump": lambda a, b: (2 * a + 3 * b + 5 + 11 * int((a + 7) % p == (b + 13) % p)) % p,
    }


def model_seed(task_index: int, replicate: int) -> int:
    return 11720000 + int(task_index) * 100_003 + int(replicate) * 1_009


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray([[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)], dtype=np.int64)


def quotient_invariant_payload(table: np.ndarray) -> dict[str, Any]:
    """Permutation invariants; unequal payloads prove task non-isomorphism."""

    global_counts = np.bincount(table.ravel(), minlength=MODULUS)
    row_histograms = np.sort(np.stack([np.bincount(row, minlength=MODULUS) for row in table]), axis=1)
    column_histograms = np.sort(np.stack([np.bincount(column, minlength=MODULUS) for column in table.T]), axis=1)
    row_agreement, column_agreement = [], []
    for first in range(MODULUS):
        for second in range(first + 1, MODULUS):
            row_agreement.append(int(np.sum(table[first] == table[second])))
            column_agreement.append(int(np.sum(table[:, first] == table[:, second])))
    return {
        "global_output_multiplicities": sorted(int(value) for value in global_counts),
        "row_histogram_multiset": sorted(tuple(map(int, row)) for row in row_histograms.tolist()),
        "column_histogram_multiset": sorted(tuple(map(int, row)) for row in column_histograms.tolist()),
        "row_pair_agreement_multiset": sorted(row_agreement),
        "column_pair_agreement_multiset": sorted(column_agreement),
        "distinct_row_count": len({tuple(map(int, row)) for row in table.tolist()}),
        "distinct_column_count": len({tuple(map(int, row)) for row in table.T.tolist()}),
        "row_distinct_output_counts": sorted(len(set(map(int, row))) for row in table.tolist()),
        "column_distinct_output_counts": sorted(len(set(map(int, row))) for row in table.T.tolist()),
    }


def quotient_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    payload = quotient_invariant_payload(table)
    return {
        "digest": base.digest(payload),
        "table_digest": base.digest(table.tolist()),
        "global_output_count_range": [min(payload["global_output_multiplicities"]), max(payload["global_output_multiplicities"])],
        "row_distinct_output_range": [min(payload["row_distinct_output_counts"]), max(payload["row_distinct_output_counts"])],
        "column_distinct_output_range": [min(payload["column_distinct_output_counts"]), max(payload["column_distinct_output_counts"])],
        "distinct_row_count": payload["distinct_row_count"],
        "distinct_column_count": payload["distinct_column_count"],
    }


def make_data(task_name: str, seed: int) -> dict[str, torch.Tensor]:
    table = task_table(task_name)
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(table.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {"train_x": x[mask_t], "train_y": y[mask_t], "holdout_x": x[~mask_t], "holdout_y": y[~mask_t]}


@torch.inference_mode()
def generic_local_scores(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(train_x.to(device)).float().cpu()
    centered = logits - logits.mean(dim=1, keepdim=True)
    lookup = torch.full((MODULUS, MODULUS), -1, dtype=torch.long)
    lookup[train_x[:, 0], train_x[:, 1]] = torch.arange(len(train_x), dtype=torch.long)
    aligned_edges: list[torch.Tensor] = []
    path_scores: list[torch.Tensor] = []
    for da, db in ((1, 0), (0, 1)):
        neighbor = lookup[(train_x[:, 0] + da) % MODULUS, (train_x[:, 1] + db) % MODULUS]
        valid = neighbor >= 0
        source = centered[valid]
        target = centered[neighbor[valid]]
        shifts = (train_y[valid] - train_y[neighbor[valid]]).tolist()
        aligned = torch.stack([torch.roll(vector, shifts=int(shift), dims=0) for vector, shift in zip(target, shifts)])
        aligned_edges.append(F.cosine_similarity(source, aligned, dim=1))
    for index, (a, b) in enumerate(train_x.tolist()):
        right = int(lookup[a, (b + 1) % MODULUS])
        down = int(lookup[(a + 1) % MODULUS, b])
        diagonal = int(lookup[(a + 1) % MODULUS, (b + 1) % MODULUS])
        if min(right, down, diagonal) < 0:
            continue
        base_vector = centered[index]
        aligned = []
        for other in (right, down, diagonal):
            shift = int(train_y[index] - train_y[other])
            aligned.append(torch.roll(centered[other], shifts=shift, dims=0))
        path_scores.append(torch.stack([F.cosine_similarity(base_vector, vector, dim=0) for vector in aligned]).mean())
    edge_values = torch.cat(aligned_edges)
    path_values = torch.stack(path_scores) if path_scores else torch.zeros(1)
    return {
        "label_aligned_local_cosine": float(edge_values.mean().item()),
        "label_aligned_local_edge_count": int(edge_values.numel()),
        "label_aligned_path_cosine": float(path_values.mean().item()),
        "label_aligned_path_cell_count": len(path_scores),
    }


def training_only_structure(model: p1171.RoleSquareNetwork, data: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    left = F.linear(model.left_embedding.weight.detach().float(), model.hidden.weight.detach().float())
    right = F.linear(model.right_embedding.weight.detach().float(), model.hidden.weight.detach().float())
    output = model.output.weight.detach().float()
    result = {
        "left_embedding_circulant_gram": base.circulant_gram_score(left),
        "right_embedding_circulant_gram": base.circulant_gram_score(right),
        "mean_embedding_circulant_gram": 0.5 * (base.circulant_gram_score(left) + base.circulant_gram_score(right)),
        "output_circulant_gram": base.circulant_gram_score(output),
        "left_embedding_fourier_top4_share": base.fourier_top_share(left),
        "right_embedding_fourier_top4_share": base.fourier_top_share(right),
        "mean_embedding_fourier_top4_share": 0.5 * (base.fourier_top_share(left) + base.fourier_top_share(right)),
        "output_fourier_top4_share": base.fourier_top_share(output),
        "parameter_l2_norm": math.sqrt(sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters())),
    }
    result.update(generic_local_scores(model, data["train_x"], data["train_y"], device))
    return result


def gradient_l2_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().float().square().sum().item()) for parameter in model.parameters() if parameter.grad is not None))


def checkpoint_payload(model: p1171.RoleSquareNetwork, task: TaskSpec, task_index: int, replicate: int, seed: int, step: int) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task.name,
        "task_split": task.split,
        "task_index": task_index,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> p1171.RoleSquareNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["step"])
    fit_index = next((i for i, row in enumerate(ordered) if row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]), None)
    event_flags = [row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"] and row["holdout"]["accuracy"] >= THRESHOLDS["stable_generalization_accuracy_min"] for row in ordered]
    stable_index = next((i for i in range(len(ordered) - 1) if event_flags[i] and event_flags[i + 1]), None)
    fit_step = ordered[fit_index]["step"] if fit_index is not None else None
    stable_step = ordered[stable_index]["step"] if stable_index is not None else None
    stable_lower = ordered[stable_index - 1]["step"] if stable_index is not None and stable_index > 0 else 0 if stable_index == 0 else max(CHECKPOINT_STEPS)
    return {
        "trajectory_id": ordered[0]["trajectory_id"],
        "task_name": ordered[0]["task_name"],
        "task_split": ordered[0]["task_split"],
        "task_index": ordered[0]["task_index"],
        "replicate": ordered[0]["replicate"],
        "seed": ordered[0]["seed"],
        "fit_step": fit_step,
        "fit_interval": [ordered[fit_index - 1]["step"] if fit_index is not None and fit_index > 0 else 0, fit_step] if fit_step is not None else [max(CHECKPOINT_STEPS), None],
        "stable_generalization_step": stable_step,
        "stable_generalization_interval": [stable_lower, stable_step] if stable_step is not None else [max(CHECKPOINT_STEPS), None],
        "event_observed": stable_step is not None,
        "maximum_holdout_accuracy": max(row["holdout"]["accuracy"] for row in ordered),
        "final_holdout_accuracy": ordered[-1]["holdout"]["accuracy"],
        "all_train_logits_finite": all(row["train"]["exact_all_finite"] for row in ordered),
        "all_holdout_logits_finite": all(row["holdout"]["exact_all_finite"] for row in ordered),
    }


BASELINE_FEATURE_NAMES = tuple(
    [f"loss_{step}" for step in HISTORY_STEPS]
    + [f"train_accuracy_{step}" for step in HISTORY_STEPS]
    + [f"mean_target_probability_{step}" for step in HISTORY_STEPS]
    + ["parameter_l2_norm_150", "gradient_l2_norm_150"]
)
STRUCTURE_FEATURE_NAMES = (
    "left_embedding_circulant_gram",
    "right_embedding_circulant_gram",
    "mean_embedding_circulant_gram",
    "output_circulant_gram",
    "left_embedding_fourier_top4_share",
    "right_embedding_fourier_top4_share",
    "mean_embedding_fourier_top4_share",
    "output_fourier_top4_share",
    "label_aligned_local_cosine",
    "label_aligned_path_cosine",
)


def feature_vector(training_rows: list[dict[str, Any]], augmented: bool) -> tuple[list[str], list[float]]:
    rows = {row["step"]: row for row in training_rows}
    values: list[float] = []
    for step in HISTORY_STEPS:
        values.append(float(rows[step]["loss"]))
    for step in HISTORY_STEPS:
        values.append(float(rows[step]["train"]["accuracy"]))
    for step in HISTORY_STEPS:
        values.append(float(rows[step]["train"]["mean_target_probability"]))
    cutoff = rows[PREDICTION_CUTOFF]
    values.extend([float(cutoff["training_only_structure"]["parameter_l2_norm"]), float(cutoff["gradient_l2_norm"])])
    names = list(BASELINE_FEATURE_NAMES)
    if augmented:
        names.extend(STRUCTURE_FEATURE_NAMES)
        values.extend(float(cutoff["training_only_structure"][name]) for name in STRUCTURE_FEATURE_NAMES)
    return names, values


def event_labels(trajectory: dict[str, Any]) -> list[float]:
    event_step = trajectory["stable_generalization_step"]
    return [float(event_step is not None and event_step <= horizon) for horizon in PREDICTION_HORIZONS]


def fit_ridge(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    y_mean = y.mean(axis=0)
    gram = standardized.T @ standardized + RIDGE_L2 * np.eye(standardized.shape[1])
    coefficient = np.linalg.solve(gram, standardized.T @ (y - y_mean))
    return {"feature_mean": mean.tolist(), "feature_scale": scale.tolist(), "target_mean": y_mean.tolist(), "coefficient": coefficient.tolist()}


def apply_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    standardized = (x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    raw = np.asarray(model["target_mean"]) + standardized @ np.asarray(model["coefficient"])
    clipped = np.clip(raw, 0.0, 1.0)
    return np.maximum.accumulate(clipped, axis=1)


def brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - y) ** 2))


def protocol_command() -> None:
    if (OUT_ROOT / "protocol/preregistration.json").exists():
        raise RuntimeError("Phase1172 protocol already exists")
    prior = base.read_json(P1171_FINAL)
    prior_audit = base.read_json(P1171_AUDIT)
    prior_representation = base.read_json(P1171_REPRESENTATION_AUDIT)
    pilot = base.read_json(PILOT_RESULT)
    if prior["decision"]["auto_continue"] or prior["decision"]["primary_endpoint_pass"]:
        raise RuntimeError("Phase1171 terminal decision mismatch")
    if prior_audit["passed"] != 32 or not prior_representation["normalized_endpoint_match"]:
        raise RuntimeError("Phase1171 audit boundary mismatch")
    if pilot["status"] != "excluded_engineering_pilot" or pilot["formal_evidence"]:
        raise RuntimeError("engineering pilot exclusion mismatch")
    signatures = {task.name: quotient_signature(task.name) for task in TASK_SPECS}
    if len({value["digest"] for value in signatures.values()}) != len(TASK_SPECS):
        raise RuntimeError("formal tasks are not separated by the frozen quotient invariant")
    manifest = []
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = make_data(task.name, seed + 17)
            manifest.append({
                "trajectory_id": f"{task.name}_r{replicate}_s{seed}",
                "task_name": task.name,
                "task_split": task.split,
                "task_index": task_index,
                "replicate": replicate,
                "seed": seed,
                "train_pair_digest": base.digest(data["train_x"].tolist()),
                "train_label_digest": base.digest(data["train_y"].tolist()),
                "sealed_holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
                "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
            })
    probe = p1171.RoleSquareNetwork(p1171.RoleSquareConfig())
    protocol = {
        "phase": PHASE,
        "created_at_utc": base.utc_now(),
        "authorization": "New explicit user request after Phase1171 AutoContinue=0; this is a new registry and does not reveal Phase1171 reserved rules.",
        "prior_final_digest": prior["final_digest"],
        "prior_audit_digest": prior_audit["audit_digest"],
        "prior_representation_audit_digest": prior_representation["report_digest"],
        "pilot_sha256": base.sha256_file(PILOT_SCRIPT),
        "pilot_result_sha256": base.sha256_file(PILOT_RESULT),
        "pilot_excluded_from_evidence": True,
        "task_equivalence": "g(a,b)=pi_Y^{-1}(f(pi_L(a),pi_R(b))) for independent permutations pi_L, pi_R, pi_Y",
        "invariant_scope": "Unequal signatures are sufficient proof of non-equivalence; equal signatures would not prove equivalence.",
        "tasks": [asdict(task) | {"quotient_signature": signatures[task.name]} for task in TASK_SPECS],
        "discovery_task_names": [task.name for task in TASK_SPECS if task.split == "discovery"],
        "confirmation_task_names": [task.name for task in TASK_SPECS if task.split == "confirmation"],
        "task_count": len(TASK_SPECS),
        "trajectory_count": len(TASK_SPECS) * REPLICATES,
        "replicates_per_task": REPLICATES,
        "modulus": MODULUS,
        "model_width": MODEL_WIDTH,
        "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
        "train_fraction": TRAIN_FRACTION,
        "training": TRAINING,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "prediction_cutoff": PREDICTION_CUTOFF,
        "history_steps": HISTORY_STEPS,
        "prediction_horizons": PREDICTION_HORIZONS,
        "baseline_feature_names": BASELINE_FEATURE_NAMES,
        "structure_feature_names": STRUCTURE_FEATURE_NAMES,
        "ridge_l2": RIDGE_L2,
        "thresholds": THRESHOLDS,
        "manifest": manifest,
        "primary_endpoint": "On task-class-held-out confirmation, augmented training-only ridge must reduce multi-horizon Brier score by >=10% versus the better of a discovery-rate constant and a baseline ridge, and win in >=3/4 confirmation classes.",
        "object_gate": "Discovery must contain >=24 observed events, >=16 right-censored trajectories, >=4 event classes, >=2 censored classes, and >=3 distinct event upper bounds; all trajectories must fit and remain finite.",
        "staged_reveal": [
            "All 96 training trajectories and 3072 checkpoints are sealed before any holdout evaluation.",
            "Only discovery holdout is revealed before predictor fitting.",
            "Predictor coefficients and discovery score are sealed before confirmation holdout exists.",
            "A discovery object-gate failure forbids confirmation reveal and all predictor reinterpretation.",
        ],
        "forbidden": [
            "No Phase1171 reserved rule may be evaluated.",
            "No task, split, feature, cutoff, horizon, threshold, or ridge constant may change after protocol creation.",
            "No holdout value or holdout-derived Gamma may enter predictor features.",
            "No hidden-state scan or causal intervention is authorized by object formation alone.",
            "This predictor registry is one-shot; failure closes it without task replacement or feature search.",
        ],
        "script_sha256": base.sha256_file(SCRIPT),
        "audit_script_sha256": base.sha256_file(AUDIT_SCRIPT),
    }
    protocol["protocol_digest"] = base.digest(protocol)
    base.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    print(json.dumps({"protocol_digest": protocol["protocol_digest"], "tasks": len(TASK_SPECS), "trajectories": protocol["trajectory_count"], "quotient_signatures_unique": True}))


def smoke_command() -> None:
    signatures = [quotient_signature(task.name)["digest"] for task in TASK_SPECS]
    if len(set(signatures)) != 12:
        raise RuntimeError("quotient signatures are not pairwise distinct")
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig())
    if sum(parameter.numel() for parameter in model.parameters()) != 39808:
        raise RuntimeError("parameter count mismatch")
    for task_index, task in enumerate(TASK_SPECS):
        data = make_data(task.name, model_seed(task_index, 0) + 17)
        overlap = set(map(tuple, data["train_x"].tolist())).intersection(map(tuple, data["holdout_x"].tolist()))
        if overlap or len(data["train_x"]) != 1860 or len(data["holdout_x"]) != 1861:
            raise RuntimeError(f"data split mismatch for {task.name}")
        if not bool(((data["train_y"] >= 0) & (data["train_y"] < MODULUS)).all()):
            raise RuntimeError(f"label range mismatch for {task.name}")
    print(json.dumps({"smoke_pass": True, "task_count": 12, "unique_quotient_signatures": 12, "parameter_count": 39808}))


def train_and_seal_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout exists before training seal")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            base.set_seed(seed)
            data = make_data(task.name, seed + 17)
            model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
            train_x, train_y = data["train_x"].to(device), data["train_y"].to(device)
            trajectory_id = f"{task.name}_r{replicate}_s{seed}"
            for step in range(1, max(CHECKPOINT_STEPS) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(train_x).float()
                    loss = F.cross_entropy(logits, train_y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss: {trajectory_id}/{step}")
                loss.backward()
                grad_norm = gradient_l2_norm(model)
                optimizer.step()
                if step not in CHECKPOINT_STEPS:
                    continue
                train_metrics = p1171.evaluate(model, data["train_x"], data["train_y"], device)
                structure = training_only_structure(model, data, device)
                checkpoint_id = f"{trajectory_id}_step{step:05d}"
                checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, task, task_index, replicate, seed, step), checkpoint_path)
                checkpoint_hash = base.sha256_file(checkpoint_path)
                hashes[checkpoint_id] = checkpoint_hash
                rows.append({
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": checkpoint_id,
                    "task_name": task.name,
                    "task_split": task.split,
                    "task_index": task_index,
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.item()),
                    "gradient_l2_norm": grad_norm,
                    "train": train_metrics,
                    "training_only_structure": structure,
                    "train_pair_digest": base.digest(data["train_x"].tolist()),
                    "train_label_digest": base.digest(data["train_y"].tolist()),
                    "sealed_holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
                    "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
                    "holdout_evaluated_during_training": False,
                    "holdout_used_by_gradient": False,
                    "checkpoint_sha256": checkpoint_hash,
                })
            print(json.dumps({"trained": trajectory_id, "checkpoints": len(CHECKPOINT_STEPS)}), flush=True)
            del model, optimizer, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()
    metrics_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    base.write_jsonl(metrics_path, rows)
    seal = {
        "phase": PHASE,
        "sealed_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": len(TASK_SPECS) * REPLICATES,
        "checkpoint_count": len(rows),
        "training_metrics_sha256": base.sha256_file(metrics_path),
        "checkpoint_hashes": hashes,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "all_training_logits_exactly_finite": all(row["train"]["exact_all_finite"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({"seal_digest": seal["seal_digest"], "trajectories": seal["trajectory_count"], "checkpoints": seal["checkpoint_count"]}))


def reveal_split(split: str) -> None:
    if split not in {"discovery", "confirmation"}:
        raise ValueError(split)
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    holdout_root = OUT_ROOT / "runs/holdout" / split
    if holdout_root.exists():
        raise RuntimeError(f"{split} holdout already exists")
    if split == "discovery" and (OUT_ROOT / "runs/holdout/confirmation").exists():
        raise RuntimeError("confirmation was revealed before discovery")
    if split == "confirmation":
        predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
        if not predictor["confirmation_reveal_authorized"]:
            raise RuntimeError("confirmation reveal not authorized")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    output_rows = []
    for training_row in training_rows:
        if training_row["task_split"] != split:
            continue
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{training_row['checkpoint_id']}.pt"
        if base.sha256_file(checkpoint_path) != training_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {training_row['checkpoint_id']}")
        data = make_data(training_row["task_name"], training_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
        output_rows.append({
            "trajectory_id": training_row["trajectory_id"],
            "checkpoint_id": training_row["checkpoint_id"],
            "task_name": training_row["task_name"],
            "task_split": split,
            "task_index": training_row["task_index"],
            "replicate": training_row["replicate"],
            "seed": training_row["seed"],
            "step": training_row["step"],
            "train": training_row["train"],
            "holdout": holdout,
        })
        del model
    path = holdout_root / "holdout_metrics.jsonl"
    base.write_jsonl(path, output_rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "evaluated_at_utc": base.utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(output_rows),
        "trajectory_count": len({row["trajectory_id"] for row in output_rows}),
        "all_holdout_logits_exactly_finite": all(row["holdout"]["exact_all_finite"] for row in output_rows),
        "holdout_metrics_sha256": base.sha256_file(path),
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_json(holdout_root / "summary.json", summary)
    print(json.dumps({"split": split, "summary_digest": summary["summary_digest"], "rows": len(output_rows), "all_finite": summary["all_holdout_logits_exactly_finite"]}))


def grouped_trajectories(split: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    holdout_rows = base.read_jsonl(OUT_ROOT / f"runs/holdout/{split}/holdout_metrics.jsonl")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    grouped_holdout: dict[str, list[dict[str, Any]]] = {}
    grouped_training: dict[str, list[dict[str, Any]]] = {}
    for row in holdout_rows:
        grouped_holdout.setdefault(row["trajectory_id"], []).append(row)
    for row in training_rows:
        if row["task_split"] == split:
            grouped_training.setdefault(row["trajectory_id"], []).append(row)
    summaries = [trajectory_summary(rows) for rows in grouped_holdout.values()]
    return sorted(summaries, key=lambda row: (row["task_index"], row["replicate"])), grouped_training


def discovery_object_decision(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    events = [row for row in trajectories if row["event_observed"]]
    censored = [row for row in trajectories if not row["event_observed"]]
    event_tasks = {row["task_name"] for row in events}
    censored_tasks = {row["task_name"] for row in censored}
    upper_bounds = sorted({row["stable_generalization_interval"][1] for row in events})
    checks = {
        "all_fit": all(row["fit_step"] is not None for row in trajectories),
        "all_finite": all(row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in trajectories),
        "event_count": len(events) >= THRESHOLDS["discovery_event_trajectory_min"],
        "censored_count": len(censored) >= THRESHOLDS["discovery_censored_trajectory_min"],
        "event_task_breadth": len(event_tasks) >= THRESHOLDS["discovery_event_task_class_min"],
        "censored_task_breadth": len(censored_tasks) >= THRESHOLDS["discovery_censored_task_class_min"],
        "event_time_breadth": len(upper_bounds) >= THRESHOLDS["discovery_distinct_event_upper_bounds_min"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "event_count": len(events),
        "right_censored_count": len(censored),
        "event_task_names": sorted(event_tasks),
        "censored_task_names": sorted(censored_tasks),
        "event_upper_bounds": upper_bounds,
    }


def fit_and_seal_predictor_command() -> None:
    if (OUT_ROOT / "analysis/predictor_seal.json").exists():
        raise RuntimeError("predictor already sealed")
    if (OUT_ROOT / "runs/holdout/confirmation").exists():
        raise RuntimeError("confirmation exists before predictor seal")
    discovery_summary = base.read_json(OUT_ROOT / "runs/holdout/discovery/summary.json")
    trajectories, training_groups = grouped_trajectories("discovery")
    object_decision = discovery_object_decision(trajectories)
    seal: dict[str, Any] = {
        "phase": PHASE,
        "sealed_at_utc": base.utc_now(),
        "discovery_summary_digest": discovery_summary["summary_digest"],
        "object_decision": object_decision,
        "confirmation_absent_at_predictor_seal": not (OUT_ROOT / "runs/holdout/confirmation").exists(),
        "confirmation_reveal_authorized": bool(object_decision["pass"]),
        "feature_cutoff": PREDICTION_CUTOFF,
        "prediction_horizons": PREDICTION_HORIZONS,
        "baseline_feature_names": BASELINE_FEATURE_NAMES,
        "structure_feature_names": STRUCTURE_FEATURE_NAMES,
        "ridge_l2": RIDGE_L2,
        "predictors": None,
        "discovery_scores": None,
    }
    if object_decision["pass"]:
        baseline_x, augmented_x, y = [], [], []
        for trajectory in trajectories:
            baseline_names, baseline_values = feature_vector(training_groups[trajectory["trajectory_id"]], augmented=False)
            augmented_names, augmented_values = feature_vector(training_groups[trajectory["trajectory_id"]], augmented=True)
            if tuple(baseline_names) != BASELINE_FEATURE_NAMES or tuple(augmented_names) != BASELINE_FEATURE_NAMES + STRUCTURE_FEATURE_NAMES:
                raise RuntimeError("feature order mismatch")
            baseline_x.append(baseline_values)
            augmented_x.append(augmented_values)
            y.append(event_labels(trajectory))
        baseline_array = np.asarray(baseline_x, dtype=np.float64)
        augmented_array = np.asarray(augmented_x, dtype=np.float64)
        label_array = np.asarray(y, dtype=np.float64)
        baseline_model = fit_ridge(baseline_array, label_array)
        augmented_model = fit_ridge(augmented_array, label_array)
        constant_probability = np.maximum.accumulate(label_array.mean(axis=0)).tolist()
        constant_pred = np.tile(np.asarray(constant_probability), (len(label_array), 1))
        baseline_pred = apply_ridge(baseline_model, baseline_array)
        augmented_pred = apply_ridge(augmented_model, augmented_array)
        seal["predictors"] = {
            "constant_probability": constant_probability,
            "baseline_ridge": baseline_model,
            "augmented_ridge": augmented_model,
        }
        seal["discovery_scores"] = {
            "constant_brier": brier(label_array, constant_pred),
            "baseline_ridge_brier": brier(label_array, baseline_pred),
            "augmented_ridge_brier": brier(label_array, augmented_pred),
        }
    seal["predictor_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "analysis/predictor_seal.json", seal)
    print(json.dumps({"object_gate_pass": object_decision["pass"], "confirmation_reveal_authorized": seal["confirmation_reveal_authorized"], "predictor_digest": seal["predictor_digest"]}))


def score_command() -> None:
    predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
    if not predictor["confirmation_reveal_authorized"]:
        score = {
            "phase": PHASE,
            "scored_at_utc": base.utc_now(),
            "stage": "discovery_object_gate_failure",
            "primary_endpoint_pass": False,
            "object_decision": predictor["object_decision"],
            "confirmation_evaluated": False,
            "interpretation": "The cross-quotient event-time panel was insufficient; predictor confirmation and all mechanism work are forbidden.",
        }
    else:
        confirmation_summary = base.read_json(OUT_ROOT / "runs/holdout/confirmation/summary.json")
        trajectories, training_groups = grouped_trajectories("confirmation")
        baseline_x, augmented_x, y, task_names = [], [], [], []
        for trajectory in trajectories:
            _, baseline_values = feature_vector(training_groups[trajectory["trajectory_id"]], augmented=False)
            _, augmented_values = feature_vector(training_groups[trajectory["trajectory_id"]], augmented=True)
            baseline_x.append(baseline_values)
            augmented_x.append(augmented_values)
            y.append(event_labels(trajectory))
            task_names.append(trajectory["task_name"])
        baseline_array = np.asarray(baseline_x, dtype=np.float64)
        augmented_array = np.asarray(augmented_x, dtype=np.float64)
        label_array = np.asarray(y, dtype=np.float64)
        constant_pred = np.tile(np.asarray(predictor["predictors"]["constant_probability"]), (len(label_array), 1))
        baseline_pred = apply_ridge(predictor["predictors"]["baseline_ridge"], baseline_array)
        augmented_pred = apply_ridge(predictor["predictors"]["augmented_ridge"], augmented_array)
        scores = {
            "constant_brier": brier(label_array, constant_pred),
            "baseline_ridge_brier": brier(label_array, baseline_pred),
            "augmented_ridge_brier": brier(label_array, augmented_pred),
        }
        best_baseline = min(scores["constant_brier"], scores["baseline_ridge_brier"])
        relative_improvement = (best_baseline - scores["augmented_ridge_brier"]) / best_baseline if best_baseline > 0 else 0.0
        per_task = []
        for task_name in sorted(set(task_names)):
            mask = np.asarray([name == task_name for name in task_names], dtype=bool)
            task_scores = {
                "task_name": task_name,
                "constant_brier": brier(label_array[mask], constant_pred[mask]),
                "baseline_ridge_brier": brier(label_array[mask], baseline_pred[mask]),
                "augmented_ridge_brier": brier(label_array[mask], augmented_pred[mask]),
            }
            task_scores["augmented_beats_both"] = task_scores["augmented_ridge_brier"] < min(task_scores["constant_brier"], task_scores["baseline_ridge_brier"])
            per_task.append(task_scores)
        class_advantage = sum(row["augmented_beats_both"] for row in per_task)
        confirmation_quality = {
            "all_fit": all(row["fit_step"] is not None for row in trajectories),
            "all_finite": all(row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in trajectories),
        }
        endpoint_checks = {
            **confirmation_quality,
            "relative_brier_improvement": relative_improvement >= THRESHOLDS["confirmation_relative_brier_improvement_min"],
            "class_breadth": class_advantage >= THRESHOLDS["confirmation_class_advantage_min"],
        }
        score = {
            "phase": PHASE,
            "scored_at_utc": base.utc_now(),
            "stage": "confirmation_prediction",
            "predictor_digest": predictor["predictor_digest"],
            "confirmation_summary_digest": confirmation_summary["summary_digest"],
            "primary_endpoint_pass": all(endpoint_checks.values()),
            "endpoint_checks": endpoint_checks,
            "confirmation_scores": scores,
            "best_baseline_brier": best_baseline,
            "relative_brier_improvement": relative_improvement,
            "confirmation_class_advantage_count": class_advantage,
            "per_confirmation_task": per_task,
            "discovery_trajectories": grouped_trajectories("discovery")[0],
            "confirmation_trajectories": trajectories,
            "interpretation": {
                "if_pass": "Frozen training-only structural features add task-class-held-out information about future stable-generalization timing beyond generic training baselines.",
                "if_fail": "This fixed camera/predictor does not transfer event-time information across task quotient classes; no feature search, hidden scan, or causal claim is authorized.",
                "scope": "Prediction concerns a controlled role-square learner, not natural-language encoding or a causal formation mechanism.",
            },
        }
    score["score_digest"] = base.digest(score)
    base.write_json(OUT_ROOT / "analysis/score.json", score)
    print(json.dumps({"stage": score["stage"], "primary_endpoint_pass": score["primary_endpoint_pass"], "score_digest": score["score_digest"], "relative_brier_improvement": score.get("relative_brier_improvement")}))


def finalize_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
    score = base.read_json(OUT_ROOT / "analysis/score.json")
    passed = bool(score["primary_endpoint_pass"])
    final = {
        "phase": PHASE,
        "finalized_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "predictor_digest": predictor["predictor_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "primary_endpoint_pass": passed,
            "cross_quotient_event_time_prediction_confirmed": passed,
            "hidden_scan_authorized": False,
            "causal_intervention_authorized": False,
            "feature_search_authorized": False,
            "auto_continue": passed,
            "authorized_next": "Phase1173: separately preregister one training-process intervention on the frozen predictive feature, with fit/loss/norm matching" if passed else None,
        },
        "claims": [
            "Twelve formal task tables have pairwise unequal frozen permutation-invariant signatures.",
            "The predictor reads training-domain history through step 150 only and never reads holdout-derived Gamma as an input.",
            "Discovery and confirmation are separated by task equivalence class and by a predictor seal.",
            "A positive endpoint would establish prospective cross-class prediction, not causal mechanism identity or language external validity.",
        ],
    }
    final["final_digest"] = base.digest(final)
    base.write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({"final_digest": final["final_digest"], "auto_continue": final["decision"]["auto_continue"], "authorized_next": final["decision"]["authorized_next"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "smoke", "train-and-seal", "reveal-discovery", "fit-and-seal-predictor", "reveal-confirmation", "score", "finalize"))
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "smoke": smoke_command,
        "train-and-seal": train_and_seal_command,
        "reveal-discovery": lambda: reveal_split("discovery"),
        "fit-and-seal-predictor": fit_and_seal_predictor_command,
        "reveal-confirmation": lambda: reveal_split("confirmation"),
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
