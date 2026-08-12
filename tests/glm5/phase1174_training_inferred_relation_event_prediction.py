#!/usr/bin/env python3
"""Training-inferred relation keys and prospective formation prediction.

Phase1173 calibrated an affine relation-transfer/closure camera when the
relation key and representation generator were known.  This phase asks a
strictly harder question in freely trained role-square networks.  Candidate
input transformations are frozen before training, relation keys are inferred
only from training pairs and labels, whitening is fit on operator-fit
backgrounds only, and the resulting early camera must predict held-out
generalization time across all-new task quotient classes.

The experiment is staged.  All models and training-only measurements are
sealed first.  Discovery holdout is then revealed, predictors are frozen, and
only then can confirmation holdout be evaluated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1174_training_inferred_relation_event_prediction_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1174_training_inferred_relation_event_prediction"
P1173_FINAL = ROOT / "tests/glm5/result/phase1173_task_conditioned_relation_closure_calibration/analysis/final.json"
P1173_AUDIT = ROOT / "tests/glm5/result/phase1173_task_conditioned_relation_closure_calibration/audit/independent_audit.json"
KEY_PILOT = ROOT / "tests/glm5_temp/phase1174_training_only_relation_key_pilot.json"
CAMERA_PILOT = ROOT / "tests/glm5_temp/phase1174_inductive_relation_camera_training_pilot.json"
MATERIAL_PROBE = ROOT / "tests/glm5_temp/phase1174_task_material_probe.json"

PHASE = 1174
MODULUS = 61
MODEL_WIDTH = 128
TRAIN_FRACTION = 0.50
REPLICATES = 8
RELATION_SHIFTS = (1, 2, 3)
KEY_CONTEXT_COUNT = 20
FIT_CONTEXT_COUNT = 20
TEST_CONTEXT_COUNT = 21
PREDICTION_CUTOFF = 250
HISTORY_STEPS = (25, 50, 75, 100, 150, 200, 250)
CHECKPOINT_STEPS = (
    25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1250,
    1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750,
    4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 12000,
)
PREDICTION_HORIZONS = (500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12000)
RIDGE_L2 = 1.0
CAMERA_RIDGE = 1.0e-7
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": max(CHECKPOINT_STEPS),
}
THRESHOLDS = {
    "key_edge_min": 200,
    "source_support_min": 2,
    "source_coverage_min": 0.70,
    "key_consistency_min": 0.98,
    "mapping_injectivity_min": 0.95,
    "validation_edge_min": 200,
    "validation_consistency_min": 0.98,
    "train_fit_accuracy_min": 0.99,
    "stable_generalization_accuracy_min": 0.90,
    "stable_generalization_adjacent_checkpoint_count": 2,
    "camera_score_min": 0.15,
    "camera_advantage_min": 0.10,
    "camera_adjacent_checkpoint_count": 2,
    "discovery_event_trajectory_min": 24,
    "discovery_censored_trajectory_min": 8,
    "discovery_event_task_class_min": 3,
    "discovery_censored_task_class_min": 1,
    "discovery_distinct_event_upper_bounds_min": 2,
    "discovery_endpoint_camera_class_min": 4,
    "confirmation_endpoint_camera_class_min": 3,
    "confirmation_relative_brier_improvement_min": 0.10,
    "confirmation_class_advantage_min": 3,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str
    family: str
    formula: str
    relation_expected: bool


TASK_SPECS = (
    TaskSpec("power2_outer1", "discovery", "power_quotient_2", "a + b^2 mod 61", True),
    TaskSpec("power3_outer7", "discovery", "power_quotient_3", "(a + b^3)^7 mod 61", True),
    TaskSpec("power5_outer11", "discovery", "power_quotient_5", "(a + b^5)^11 mod 61", True),
    TaskSpec("power10_outer13", "discovery", "power_quotient_10", "(a + b^10)^13 mod 61", True),
    TaskSpec("power15_outer17", "discovery", "power_quotient_15", "(a + b^15)^17 mod 61", True),
    TaskSpec("power30_outer19", "discovery", "power_quotient_30", "(a + b^30)^19 mod 61", True),
    TaskSpec("interaction_one", "discovery", "non_equivariant_cubic", "a*b^3 + 7a + 11b + 13 mod 61", False),
    TaskSpec("interaction_two", "discovery", "non_equivariant_quadratic", "(a+2)(b+3)^2 + 5a + 17 mod 61", False),
    TaskSpec("power4_outer23", "confirmation", "power_quotient_4", "(a + b^4)^23 mod 61", True),
    TaskSpec("power12_outer29", "confirmation", "power_quotient_12", "(a + b^12)^29 mod 61", True),
    TaskSpec("power60_outer31", "confirmation", "power_quotient_60", "(a + b^60)^31 mod 61", True),
    TaskSpec("interaction_three", "confirmation", "non_equivariant_mixed", "(a^2+3)(b+5) + 7a + 19 mod 61", False),
)
TASK_BY_NAME = {task.name: task for task in TASK_SPECS}


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "power2_outer1": lambda a, b: (a + pow(b, 2, p)) % p,
        "power3_outer7": lambda a, b: pow((a + pow(b, 3, p)) % p, 7, p),
        "power5_outer11": lambda a, b: pow((a + pow(b, 5, p)) % p, 11, p),
        "power10_outer13": lambda a, b: pow((a + pow(b, 10, p)) % p, 13, p),
        "power15_outer17": lambda a, b: pow((a + pow(b, 15, p)) % p, 17, p),
        "power30_outer19": lambda a, b: pow((a + pow(b, 30, p)) % p, 19, p),
        "power4_outer23": lambda a, b: pow((a + pow(b, 4, p)) % p, 23, p),
        "power12_outer29": lambda a, b: pow((a + pow(b, 12, p)) % p, 29, p),
        "power60_outer31": lambda a, b: pow((a + pow(b, 60, p)) % p, 31, p),
        "interaction_one": lambda a, b: (a * pow(b, 3, p) + 7 * a + 11 * b + 13) % p,
        "interaction_two": lambda a, b: (((a + 2) * pow((b + 3) % p, 2, p)) + 5 * a + 17) % p,
        "interaction_three": lambda a, b: (((a * a + 3) * ((b + 5) % p)) + 7 * a + 19) % p,
    }


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray([[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)], dtype=np.int64)


def quotient_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    payload = p1172.quotient_invariant_payload(table)
    return {
        "digest": base.digest(payload),
        "table_digest": base.digest(table.tolist()),
        "distinct_row_count": payload["distinct_row_count"],
        "distinct_column_count": payload["distinct_column_count"],
        "row_distinct_output_range": [min(payload["row_distinct_output_counts"]), max(payload["row_distinct_output_counts"])],
        "column_distinct_output_range": [min(payload["column_distinct_output_counts"]), max(payload["column_distinct_output_counts"])],
    }


def model_seed(task_index: int, replicate: int) -> int:
    return 11_740_000 + task_index * 100_003 + replicate * 1_009


def make_data(task_name: str, seed: int) -> dict[str, Any]:
    table = task_table(task_name)
    pairs = np.asarray([(a, b) for a in range(MODULUS) for b in range(MODULUS)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    train_mask = np.zeros(MODULUS * MODULUS, dtype=bool)
    train_mask[order[: int(round(len(pairs) * TRAIN_FRACTION))]] = True
    flat_mask = train_mask.copy()
    train_mask = train_mask.reshape(MODULUS, MODULUS)
    backgrounds = rng.permutation(MODULUS)
    contexts = {
        "key": backgrounds[:KEY_CONTEXT_COUNT].astype(np.int64),
        "fit": backgrounds[KEY_CONTEXT_COUNT:KEY_CONTEXT_COUNT + FIT_CONTEXT_COUNT].astype(np.int64),
        "test": backgrounds[KEY_CONTEXT_COUNT + FIT_CONTEXT_COUNT:].astype(np.int64),
    }
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(table.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(flat_mask, dtype=torch.bool)
    return {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "holdout_x": x[~mask_t],
        "holdout_y": y[~mask_t],
        "train_mask": train_mask,
        "contexts": contexts,
    }


def edge_coordinates(train_mask: np.ndarray, contexts: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    source, target = [], []
    for b in map(int, contexts):
        for a in range(MODULUS):
            aa = (a + shift) % MODULUS
            if bool(train_mask[a, b]) and bool(train_mask[aa, b]):
                source.append((a, b))
                target.append((aa, b))
    return np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


def infer_relation(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: np.ndarray,
    key_contexts: np.ndarray,
    fit_contexts: np.ndarray,
    shift: int,
) -> dict[str, Any]:
    label_lookup = np.full((MODULUS, MODULUS), -1, dtype=np.int64)
    coordinates = train_x.numpy()
    label_lookup[coordinates[:, 0], coordinates[:, 1]] = train_y.numpy()
    key_source, key_target = edge_coordinates(train_mask, key_contexts, shift)
    counts = np.zeros((MODULUS, MODULUS), dtype=np.int64)
    for source, target in zip(key_source, key_target):
        counts[int(label_lookup[source[0], source[1]]), int(label_lookup[target[0], target[1]])] += 1
    totals = counts.sum(axis=1)
    supported = np.flatnonzero(totals >= THRESHOLDS["source_support_min"])
    mapping = np.full(MODULUS, -1, dtype=np.int64)
    if len(supported):
        mapping[supported] = np.argmax(counts[supported], axis=1)
    key_correct = int(sum(int(counts[source, mapping[source]]) for source in supported))
    key_total = int(sum(int(totals[source]) for source in supported))
    key_consistency = key_correct / max(key_total, 1)
    coverage = len(supported) / MODULUS
    injectivity = len(set(map(int, mapping[supported]))) / max(len(supported), 1)

    validation_source, validation_target = edge_coordinates(train_mask, fit_contexts, shift)
    validation_total = 0
    validation_correct = 0
    for source, target in zip(validation_source, validation_target):
        source_label = int(label_lookup[source[0], source[1]])
        if mapping[source_label] < 0:
            continue
        validation_total += 1
        validation_correct += int(mapping[source_label] == int(label_lookup[target[0], target[1]]))
    validation_consistency = validation_correct / max(validation_total, 1)
    checks = {
        "key_edges": len(key_source) >= THRESHOLDS["key_edge_min"],
        "coverage": coverage >= THRESHOLDS["source_coverage_min"],
        "key_consistency": key_consistency >= THRESHOLDS["key_consistency_min"],
        "injectivity": injectivity >= THRESHOLDS["mapping_injectivity_min"],
        "validation_edges": validation_total >= THRESHOLDS["validation_edge_min"],
        "validation_consistency": validation_consistency >= THRESHOLDS["validation_consistency_min"],
    }
    return {
        "shift": shift,
        "key_edge_count": int(len(key_source)),
        "supported_source_count": int(len(supported)),
        "source_coverage": float(coverage),
        "key_consistency": float(key_consistency),
        "mapping_injectivity": float(injectivity),
        "validation_edge_count": int(validation_total),
        "validation_consistency": float(validation_consistency),
        "mapping": mapping.tolist(),
        "mapping_digest": base.digest(mapping.tolist()),
        "eligible": bool(all(checks.values())),
        "checks": checks,
    }


def infer_relation_key(data: dict[str, Any]) -> dict[str, Any]:
    relations = [
        infer_relation(
            data["train_x"], data["train_y"], data["train_mask"],
            data["contexts"]["key"], data["contexts"]["fit"], shift,
        )
        for shift in RELATION_SHIFTS
    ]
    payload = {
        "candidate_transform_family": "left_coordinate_cyclic_shift",
        "relations": relations,
        "eligible_count": int(sum(row["eligible"] for row in relations)),
        "all_closure_relations_eligible": bool(all(row["eligible"] for row in relations)),
        "uses_training_inputs": True,
        "uses_training_labels": True,
        "uses_holdout_inputs": False,
        "uses_holdout_labels": False,
        "uses_task_name_or_formula": False,
        "uses_future_generalization": False,
    }
    payload["key_digest"] = base.digest(payload)
    return payload


def relation_key_features(key: dict[str, Any]) -> list[float]:
    relations = key["relations"]
    return [
        float(key["eligible_count"]),
        float(np.mean([row["source_coverage"] for row in relations])),
        float(np.mean([row["key_consistency"] for row in relations])),
        float(np.mean([row["mapping_injectivity"] for row in relations])),
        float(np.mean([row["validation_consistency"] for row in relations])),
        float(np.log1p(sum(row["key_edge_count"] for row in relations))),
    ]


@torch.inference_mode()
def hidden_grid(model: p1171.RoleSquareNetwork, train_mask: np.ndarray, device: torch.device) -> np.ndarray:
    coordinates = np.argwhere(train_mask)
    inputs = torch.tensor(coordinates, dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        summed = model.left_embedding(inputs[:, 0]) + model.right_embedding(inputs[:, 1])
        hidden = model.hidden(summed).square()
    grid = np.full((MODULUS, MODULUS, hidden.shape[-1]), np.nan, dtype=np.float64)
    grid[coordinates[:, 0], coordinates[:, 1]] = hidden.float().cpu().numpy().astype(np.float64)
    return grid


def fit_whitener(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    cutoff = max(float(eigenvalues.max()), 1.0e-12) * 1.0e-10
    keep = eigenvalues > cutoff
    transform = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    return mean, transform, eigenvalues[keep]


def apply_whitener(values: np.ndarray, mean: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    return (values - mean) @ whitener


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * CAMERA_RIDGE
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def apply_affine(source: np.ndarray, operator: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    return design @ operator


def score_camera_variant(
    grid: np.ndarray,
    train_mask: np.ndarray,
    fit_contexts: np.ndarray,
    test_contexts: np.ndarray,
    randomize_targets: bool,
    random_seed: int,
) -> dict[str, Any]:
    fit_edges = {shift: edge_coordinates(train_mask, fit_contexts, shift) for shift in RELATION_SHIFTS}
    fit_coordinates = np.unique(
        np.concatenate([np.concatenate(pair, axis=0) for pair in fit_edges.values()], axis=0), axis=0,
    )
    mean, whitener, retained_eigenvalues = fit_whitener(grid[fit_coordinates[:, 0], fit_coordinates[:, 1]])
    operators = {}
    for shift, (source_coordinates, target_coordinates) in fit_edges.items():
        source = apply_whitener(grid[source_coordinates[:, 0], source_coordinates[:, 1]], mean, whitener)
        target = apply_whitener(grid[target_coordinates[:, 0], target_coordinates[:, 1]], mean, whitener)
        if randomize_targets:
            rng = np.random.default_rng(random_seed + shift * 101)
            target = target[rng.permutation(len(target))]
        operators[shift] = fit_affine(source, target)

    reuse_num = 0.0
    reuse_den = 0.0
    test_edge_count = 0
    for shift in RELATION_SHIFTS:
        source_coordinates, target_coordinates = edge_coordinates(train_mask, test_contexts, shift)
        source = apply_whitener(grid[source_coordinates[:, 0], source_coordinates[:, 1]], mean, whitener)
        target = apply_whitener(grid[target_coordinates[:, 0], target_coordinates[:, 1]], mean, whitener)
        prediction = apply_affine(source, operators[shift])
        reuse_num += float(np.sum((prediction - target) ** 2))
        reuse_den += float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
        test_edge_count += len(source_coordinates)

    closure_source, closure_target = [], []
    for b in map(int, test_contexts):
        for a in range(MODULUS):
            if train_mask[a, b] and train_mask[(a + 1) % MODULUS, b] and train_mask[(a + 3) % MODULUS, b]:
                closure_source.append((a, b))
                closure_target.append(((a + 3) % MODULUS, b))
    closure_source = np.asarray(closure_source, dtype=np.int64)
    closure_target = np.asarray(closure_target, dtype=np.int64)
    source = apply_whitener(grid[closure_source[:, 0], closure_source[:, 1]], mean, whitener)
    actual_target = apply_whitener(grid[closure_target[:, 0], closure_target[:, 1]], mean, whitener)
    direct = apply_affine(source, operators[3])
    composed = apply_affine(apply_affine(source, operators[1]), operators[2])
    closure_num = float(np.sum((direct - composed) ** 2) + np.sum((composed - actual_target) ** 2))
    closure_den = float(2.0 * np.sum((actual_target - actual_target.mean(axis=0, keepdims=True)) ** 2))
    reuse = float(np.clip(1.0 - reuse_num / max(reuse_den, 1.0e-12), 0.0, 1.0))
    closure = float(np.clip(1.0 - closure_num / max(closure_den, 1.0e-12), 0.0, 1.0))
    return {
        "reuse": reuse,
        "closure": closure,
        "score": float(math.sqrt(reuse * closure)),
        "effective_rank": int(whitener.shape[1]),
        "fit_state_count": int(len(fit_coordinates)),
        "test_edge_count": int(test_edge_count),
        "closure_test_count": int(len(closure_source)),
        "retained_eigenvalue_min": float(retained_eigenvalues.min()),
        "retained_eigenvalue_max": float(retained_eigenvalues.max()),
        "whitening_fit_background_only": True,
        "test_background_used_for_whitening": False,
        "randomized_target_pairing": bool(randomize_targets),
    }


def relation_camera(
    model: p1171.RoleSquareNetwork,
    data: dict[str, Any],
    relation_key: dict[str, Any],
    device: torch.device,
    random_seed: int,
) -> dict[str, Any]:
    if not relation_key["all_closure_relations_eligible"]:
        empty = {
            "reuse": 0.0,
            "closure": 0.0,
            "score": 0.0,
            "effective_rank": 0,
            "fit_state_count": 0,
            "test_edge_count": 0,
            "closure_test_count": 0,
            "whitening_fit_background_only": True,
            "test_background_used_for_whitening": False,
        }
        return {
            "status": "NoEligibleRelation",
            "actual": empty | {"randomized_target_pairing": False},
            "random_pairing": empty | {"randomized_target_pairing": True},
            "score_advantage": 0.0,
        }
    grid = hidden_grid(model, data["train_mask"], device)
    actual = score_camera_variant(
        grid, data["train_mask"], data["contexts"]["fit"], data["contexts"]["test"], False, random_seed,
    )
    random_pairing = score_camera_variant(
        grid, data["train_mask"], data["contexts"]["fit"], data["contexts"]["test"], True, random_seed,
    )
    return {
        "status": "EligibleRelation",
        "actual": actual,
        "random_pairing": random_pairing,
        "score_advantage": float(actual["score"] - random_pairing["score"]),
    }


def checkpoint_payload(
    model: p1171.RoleSquareNetwork,
    task: TaskSpec,
    task_index: int,
    replicate: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
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


def random_relation_null(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    table = rng.integers(0, MODULUS, size=(MODULUS, MODULUS), dtype=np.int64)
    pairs = np.asarray([(a, b) for a in range(MODULUS) for b in range(MODULUS)], dtype=np.int64)
    order = rng.permutation(len(pairs))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[: int(round(len(pairs) * TRAIN_FRACTION))]] = True
    backgrounds = rng.permutation(MODULUS)
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(table.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    data = {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "train_mask": mask.reshape(MODULUS, MODULUS),
        "contexts": {
            "key": backgrounds[:KEY_CONTEXT_COUNT],
            "fit": backgrounds[KEY_CONTEXT_COUNT:KEY_CONTEXT_COUNT + FIT_CONTEXT_COUNT],
            "test": backgrounds[KEY_CONTEXT_COUNT + FIT_CONTEXT_COUNT:],
        },
    }
    return infer_relation_key(data)


def material_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_signatures = {task.name: p1172.quotient_signature(task.name)["digest"] for task in p1172.TASK_SPECS}
    old_digests = set(old_signatures.values())
    signatures = {task.name: quotient_signature(task.name) for task in TASK_SPECS}
    manifest = []
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = make_data(task.name, seed + 17)
            key = infer_relation_key(data)
            manifest.append({
                "trajectory_id": f"{task.name}_r{replicate}_s{seed}",
                "task_name": task.name,
                "task_split": task.split,
                "task_index": task_index,
                "replicate": replicate,
                "seed": seed,
                "relation_expected": task.relation_expected,
                "relation_key": key,
                "train_pair_digest": base.digest(data["train_x"].tolist()),
                "train_label_digest": base.digest(data["train_y"].tolist()),
                "context_digest": base.digest({name: values.tolist() for name, values in data["contexts"].items()}),
            })
    random_nulls = [random_relation_null(11_749_000 + index * 103) for index in range(16)]
    checks = {
        "candidate_quotient_signatures_unique": len({value["digest"] for value in signatures.values()}) == len(TASK_SPECS),
        "no_phase1172_quotient_collision": all(value["digest"] not in old_digests for value in signatures.values()),
        "all_expected_relations_identified": all(
            row["relation_key"]["eligible_count"] == 3 for row in manifest if row["relation_expected"]
        ),
        "all_expected_abstentions_observed": all(
            row["relation_key"]["eligible_count"] == 0 for row in manifest if not row["relation_expected"]
        ),
        "random_table_nulls_abstain": all(key["eligible_count"] == 0 for key in random_nulls),
        "context_partition_exact": all(
            len(make_data(row["task_name"], row["seed"] + 17)["contexts"]["key"]) == KEY_CONTEXT_COUNT
            and len(make_data(row["task_name"], row["seed"] + 17)["contexts"]["fit"]) == FIT_CONTEXT_COUNT
            and len(make_data(row["task_name"], row["seed"] + 17)["contexts"]["test"]) == TEST_CONTEXT_COUNT
            for row in manifest
        ),
        "relation_key_training_only": all(
            not row["relation_key"]["uses_holdout_inputs"]
            and not row["relation_key"]["uses_holdout_labels"]
            and not row["relation_key"]["uses_task_name_or_formula"]
            and not row["relation_key"]["uses_future_generalization"]
            for row in manifest
        ),
    }
    material = {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "signatures": signatures,
        "phase1172_signatures": old_signatures,
        "random_null_count": len(random_nulls),
        "random_null_eligible_counts": [key["eligible_count"] for key in random_nulls],
    }
    return manifest, material


def protocol_command() -> None:
    path = OUT_ROOT / "protocol/preregistration.json"
    if path.exists():
        raise RuntimeError("Phase1174 protocol already exists")
    prior = base.read_json(P1173_FINAL)
    prior_audit = base.read_json(P1173_AUDIT)
    key_pilot = base.read_json(KEY_PILOT)
    camera_pilot = base.read_json(CAMERA_PILOT)
    probe = base.read_json(MATERIAL_PROBE)
    if not prior["relation_closure_camera_calibrated"] or not prior["confirmation_passed"]:
        raise RuntimeError("Phase1173 calibration boundary mismatch")
    if not prior_audit["passed"] or prior_audit["passed_count"] != 34:
        raise RuntimeError("Phase1173 independent audit boundary mismatch")
    if key_pilot["scope"] != "exclusion_only_non_evidential_pilot" or camera_pilot["scope"] != "exclusion_only_non_evidential_pilot":
        raise RuntimeError("Phase1174 pilot exclusion boundary mismatch")
    if not probe["all_candidate_signatures_unique"] or probe["old_collision_count"] != 0:
        raise RuntimeError("material probe boundary mismatch")
    manifest, material = material_manifest()
    if not material["pass"]:
        raise RuntimeError(f"material gate failed: {material['checks']}")
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": base.utc_now(),
        "title": "Training-inferred relation key transfer/closure and prospective event-time prediction",
        "script_sha256": base.sha256_file(SCRIPT),
        "audit_script_sha256": base.sha256_file(AUDIT_SCRIPT),
        "prior_phase1173_final_digest": prior["final_digest"],
        "prior_phase1173_audit_digest": prior_audit["audit_digest"],
        "pilot_scope": "exclusion-only engineering; no formal task formula or seed was trained",
        "task_specs": [asdict(task) | {"quotient_signature": material["signatures"][task.name]} for task in TASK_SPECS],
        "manifest": manifest,
        "material_gate": material,
        "task_count": len(TASK_SPECS),
        "trajectory_count": len(TASK_SPECS) * REPLICATES,
        "replicates": REPLICATES,
        "model": {"architecture": "RoleSquareNetwork", "modulus": MODULUS, "width": MODEL_WIDTH, "parameter_count": 39_808},
        "training": TRAINING,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "prediction_cutoff": PREDICTION_CUTOFF,
        "prediction_horizons": PREDICTION_HORIZONS,
        "history_steps": HISTORY_STEPS,
        "relation_key": {
            "candidate_input_transforms": [f"left_shift_{shift}" for shift in RELATION_SHIFTS],
            "definition": "r=(g,pi_hat_g), where pi_hat_g is inferred only from paired training labels",
            "key_background_count": KEY_CONTEXT_COUNT,
            "validation_and_operator_fit_background_count": FIT_CONTEXT_COUNT,
            "operator_test_background_count": TEST_CONTEXT_COUNT,
            "abstention_allowed": True,
            "thresholds": {name: THRESHOLDS[name] for name in (
                "key_edge_min", "source_support_min", "source_coverage_min", "key_consistency_min",
                "mapping_injectivity_min", "validation_edge_min", "validation_consistency_min",
            )},
        },
        "camera": {
            "state": "post-square hidden state on training examples only",
            "map_family": "affine ridge",
            "camera_ridge": CAMERA_RIDGE,
            "whitening": "mean, covariance, effective rank fit only on operator-fit backgrounds and frozen on operator-test backgrounds",
            "test_background_marginals_used": False,
            "relation_score": "geometric mean of held-out-background reuse and closure scores",
            "random_pairing_control": "same edges and counts with deterministic target permutation",
        },
        "zero_models": [
            "constant event rate", "task relation-class only", "training-label graph summaries only",
            "basic training dynamics only", "static state geometry only", "random relation pairing",
        ],
        "thresholds": THRESHOLDS,
        "primary_endpoint": (
            "On four task-class-held-out confirmation classes, the early training-only relation camera must reduce "
            "multi-horizon Brier score by >=10% versus every frozen zero model, beat every zero model in >=3/4 "
            "classes, and the final-state camera must independently pass all three eligible confirmation classes."
        ),
        "hard_stops": [
            "If discovery lacks a final-state relation object, do not reveal confirmation.",
            "If prediction fails, close this affine-camera branch; do not search a nonlinear camera.",
            "A pass authorizes only an independently preregistered causal-use experiment, not a mechanism claim.",
        ],
        "claim_scope": "Controlled free role-square networks; no natural-language, causal-use, or neural-necessity claim.",
    }
    protocol["protocol_digest"] = base.digest(protocol)
    base.write_json(path, protocol)
    print(json.dumps({
        "protocol_digest": protocol["protocol_digest"],
        "material_gate_pass": material["pass"],
        "tasks": len(TASK_SPECS),
        "trajectories": protocol["trajectory_count"],
    }))


def smoke_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if not protocol["material_gate"]["pass"]:
        raise RuntimeError("material gate is not closed")
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH))
    if sum(parameter.numel() for parameter in model.parameters()) != 39_808:
        raise RuntimeError("parameter count mismatch")
    for task_index, task in enumerate(TASK_SPECS):
        data = make_data(task.name, model_seed(task_index, 0) + 17)
        if len(data["train_x"]) != 1860 or len(data["holdout_x"]) != 1861:
            raise RuntimeError(f"split mismatch: {task.name}")
        key = infer_relation_key(data)
        expected_count = 3 if task.relation_expected else 0
        if key["eligible_count"] != expected_count:
            raise RuntimeError(f"relation key mismatch: {task.name}")
    print(json.dumps({"smoke_pass": True, "task_count": 12, "parameter_count": 39_808}))


def gradient_l2_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in model.parameters() if parameter.grad is not None
    ))


def train_and_seal_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout exists before training seal")
    manifest_by_id = {row["trajectory_id"]: row for row in protocol["manifest"]}
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            base.set_seed(seed)
            data = make_data(task.name, seed + 17)
            relation_key = infer_relation_key(data)
            trajectory_id = f"{task.name}_r{replicate}_s{seed}"
            if relation_key["key_digest"] != manifest_by_id[trajectory_id]["relation_key"]["key_digest"]:
                raise RuntimeError(f"relation key drift: {trajectory_id}")
            model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
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
                structure = p1172.training_only_structure(model, data, device)
                camera = relation_camera(model, data, relation_key, device, seed + 90_001)
                checkpoint_id = f"{trajectory_id}_step{step:05d}"
                checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, task, task_index, replicate, seed, step), checkpoint_path)
                checkpoint_hash = base.sha256_file(checkpoint_path)
                checkpoint_hashes[checkpoint_id] = checkpoint_hash
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
                    "relation_key": relation_key,
                    "relation_key_features": relation_key_features(relation_key),
                    "relation_camera": camera,
                    "train_pair_digest": base.digest(data["train_x"].tolist()),
                    "train_label_digest": base.digest(data["train_y"].tolist()),
                    "sealed_holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
                    "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
                    "holdout_evaluated_during_training": False,
                    "holdout_used_by_gradient": False,
                    "camera_used_holdout": False,
                    "checkpoint_sha256": checkpoint_hash,
                })
            print(json.dumps({
                "trained": trajectory_id,
                "checkpoints": len(CHECKPOINT_STEPS),
                "relation_status": rows[-1]["relation_camera"]["status"],
                "final_camera_score": rows[-1]["relation_camera"]["actual"]["score"],
            }), flush=True)
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
        "checkpoint_hashes": checkpoint_hashes,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "no_holdout_camera": all(not row["camera_used_holdout"] for row in rows),
        "strict_inductive_whitening": all(
            row["relation_camera"]["actual"]["whitening_fit_background_only"]
            and not row["relation_camera"]["actual"]["test_background_used_for_whitening"]
            for row in rows
        ),
        "all_training_logits_exactly_finite": all(row["train"]["exact_all_finite"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({
        "seal_digest": seal["seal_digest"],
        "trajectories": seal["trajectory_count"],
        "checkpoints": seal["checkpoint_count"],
        "strict_inductive_whitening": seal["strict_inductive_whitening"],
    }))


def reveal_split(split: str) -> None:
    if split not in {"discovery", "confirmation"}:
        raise ValueError(split)
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    output_root = OUT_ROOT / "runs/holdout" / split
    if output_root.exists():
        raise RuntimeError(f"{split} already revealed")
    if split == "confirmation":
        predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
        if not predictor["confirmation_reveal_authorized"]:
            raise RuntimeError("confirmation reveal not authorized")
    elif (OUT_ROOT / "runs/holdout/confirmation").exists():
        raise RuntimeError("confirmation predates discovery")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    rows = []
    for training_row in training_rows:
        if training_row["task_split"] != split:
            continue
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{training_row['checkpoint_id']}.pt"
        if base.sha256_file(checkpoint_path) != training_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {training_row['checkpoint_id']}")
        data = make_data(training_row["task_name"], training_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
        rows.append({
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
    path = output_root / "holdout_metrics.jsonl"
    base.write_jsonl(path, rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "evaluated_at_utc": base.utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "all_holdout_logits_exactly_finite": all(row["holdout"]["exact_all_finite"] for row in rows),
        "holdout_metrics_sha256": base.sha256_file(path),
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_json(output_root / "summary.json", summary)
    print(json.dumps({"split": split, "rows": len(rows), "summary_digest": summary["summary_digest"]}))


def first_stable_index(flags: list[bool], count: int = 2) -> int | None:
    if len(flags) < count:
        return None
    return next((index for index in range(len(flags) - count + 1) if all(flags[index:index + count])), None)


def trajectory_summary(training_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    training = sorted(training_rows, key=lambda row: row["step"])
    holdout = {row["step"]: row for row in holdout_rows}
    fit_index = next((index for index, row in enumerate(training) if row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]), None)
    generalization_flags = [
        row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]
        and holdout[row["step"]]["holdout"]["accuracy"] >= THRESHOLDS["stable_generalization_accuracy_min"]
        for row in training
    ]
    generalization_index = first_stable_index(generalization_flags, THRESHOLDS["stable_generalization_adjacent_checkpoint_count"])
    camera_flags = [
        row["relation_camera"]["status"] == "EligibleRelation"
        and row["relation_camera"]["actual"]["score"] >= THRESHOLDS["camera_score_min"]
        and row["relation_camera"]["score_advantage"] >= THRESHOLDS["camera_advantage_min"]
        for row in training
    ]
    camera_index = first_stable_index(camera_flags, THRESHOLDS["camera_adjacent_checkpoint_count"])
    fit_step = training[fit_index]["step"] if fit_index is not None else None
    generalization_step = training[generalization_index]["step"] if generalization_index is not None else None
    camera_step = training[camera_index]["step"] if camera_index is not None else None
    generalization_lower = training[generalization_index - 1]["step"] if generalization_index is not None and generalization_index > 0 else 0
    camera_lower = training[camera_index - 1]["step"] if camera_index is not None and camera_index > 0 else 0
    final = training[-1]
    cutoff = next(row for row in training if row["step"] == PREDICTION_CUTOFF)
    return {
        "trajectory_id": training[0]["trajectory_id"],
        "task_name": training[0]["task_name"],
        "task_split": training[0]["task_split"],
        "task_index": training[0]["task_index"],
        "replicate": training[0]["replicate"],
        "seed": training[0]["seed"],
        "relation_eligible": training[0]["relation_key"]["all_closure_relations_eligible"],
        "fit_step": fit_step,
        "fit_interval": [training[fit_index - 1]["step"] if fit_index is not None and fit_index > 0 else 0, fit_step] if fit_step else [max(CHECKPOINT_STEPS), None],
        "stable_generalization_step": generalization_step,
        "stable_generalization_interval": [generalization_lower, generalization_step] if generalization_step else [max(CHECKPOINT_STEPS), None],
        "event_observed": generalization_step is not None,
        "relation_camera_step": camera_step,
        "relation_camera_interval": [camera_lower, camera_step] if camera_step else [max(CHECKPOINT_STEPS), None],
        "relation_camera_event_observed": camera_step is not None,
        "relation_camera_strictly_precedes_generalization": bool(
            camera_step is not None and generalization_step is not None and camera_step <= generalization_lower
        ),
        "relation_camera_follows_generalization": bool(
            camera_step is not None and generalization_step is not None and camera_lower >= generalization_step
        ),
        "maximum_holdout_accuracy": max(row["holdout"]["accuracy"] for row in holdout.values()),
        "final_holdout_accuracy": holdout[final["step"]]["holdout"]["accuracy"],
        "cutoff_relation_score": cutoff["relation_camera"]["actual"]["score"],
        "cutoff_random_score": cutoff["relation_camera"]["random_pairing"]["score"],
        "final_relation_score": final["relation_camera"]["actual"]["score"],
        "final_random_score": final["relation_camera"]["random_pairing"]["score"],
        "final_relation_advantage": final["relation_camera"]["score_advantage"],
        "all_train_logits_finite": all(row["train"]["exact_all_finite"] for row in training),
        "all_holdout_logits_finite": all(row["holdout"]["exact_all_finite"] for row in holdout.values()),
    }


def grouped_trajectories(split: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    holdout_rows = base.read_jsonl(OUT_ROOT / f"runs/holdout/{split}/holdout_metrics.jsonl")
    training_groups: dict[str, list[dict[str, Any]]] = {}
    holdout_groups: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        if row["task_split"] == split:
            training_groups.setdefault(row["trajectory_id"], []).append(row)
    for row in holdout_rows:
        holdout_groups.setdefault(row["trajectory_id"], []).append(row)
    summaries = [trajectory_summary(training_groups[key], holdout_groups[key]) for key in sorted(training_groups)]
    return sorted(summaries, key=lambda row: (row["task_index"], row["replicate"])), training_groups


STATIC_FEATURE_NAMES = (
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
LABEL_FEATURE_NAMES = (
    "eligible_relation_count",
    "mean_source_coverage",
    "mean_key_consistency",
    "mean_mapping_injectivity",
    "mean_validation_consistency",
    "log_relation_edge_count",
)
CAMERA_FEATURE_NAMES = ("reuse", "closure", "score")
MODEL_NAMES = ("dynamics", "class_only", "label_only", "static_only", "common", "random_pairing", "relation")


def dynamics_names() -> tuple[str, ...]:
    return tuple(
        [f"loss_{step}" for step in HISTORY_STEPS]
        + [f"train_accuracy_{step}" for step in HISTORY_STEPS]
        + [f"mean_target_probability_{step}" for step in HISTORY_STEPS]
        + [f"parameter_l2_norm_{PREDICTION_CUTOFF}", f"gradient_l2_norm_{PREDICTION_CUTOFF}"]
    )


def feature_vector(training_rows: list[dict[str, Any]], model_name: str) -> tuple[list[str], list[float]]:
    rows = {row["step"]: row for row in training_rows}
    cutoff = rows[PREDICTION_CUTOFF]
    dynamics = (
        [float(rows[step]["loss"]) for step in HISTORY_STEPS]
        + [float(rows[step]["train"]["accuracy"]) for step in HISTORY_STEPS]
        + [float(rows[step]["train"]["mean_target_probability"]) for step in HISTORY_STEPS]
        + [float(cutoff["training_only_structure"]["parameter_l2_norm"]), float(cutoff["gradient_l2_norm"])]
    )
    class_values = [float(cutoff["relation_key"]["eligible_count"] == len(RELATION_SHIFTS))]
    label_values = list(map(float, cutoff["relation_key_features"]))
    static_values = [float(cutoff["training_only_structure"][name]) for name in STATIC_FEATURE_NAMES]
    common_names = list(dynamics_names()) + ["eligible_relation_class"] + list(LABEL_FEATURE_NAMES) + list(STATIC_FEATURE_NAMES)
    common_values = dynamics + class_values + label_values + static_values
    if model_name == "dynamics":
        return list(dynamics_names()), dynamics
    if model_name == "class_only":
        return ["eligible_relation_class"], class_values
    if model_name == "label_only":
        return list(LABEL_FEATURE_NAMES), label_values
    if model_name == "static_only":
        return list(STATIC_FEATURE_NAMES), static_values
    if model_name == "common":
        return common_names, common_values
    if model_name == "random_pairing":
        camera = cutoff["relation_camera"]["random_pairing"]
        return common_names + [f"random_{name}" for name in CAMERA_FEATURE_NAMES], common_values + [float(camera[name]) for name in CAMERA_FEATURE_NAMES]
    if model_name == "relation":
        camera = cutoff["relation_camera"]["actual"]
        return common_names + [f"relation_{name}" for name in CAMERA_FEATURE_NAMES], common_values + [float(camera[name]) for name in CAMERA_FEATURE_NAMES]
    raise ValueError(model_name)


def event_labels(trajectory: dict[str, Any]) -> list[float]:
    event_step = trajectory["stable_generalization_step"]
    return [float(event_step is not None and event_step <= horizon) for horizon in PREDICTION_HORIZONS]


def fit_ridge(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (x - mean) / scale
    target_mean = y.mean(axis=0)
    gram = standardized.T @ standardized + RIDGE_L2 * np.eye(standardized.shape[1])
    coefficient = np.linalg.solve(gram, standardized.T @ (y - target_mean))
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "target_mean": target_mean.tolist(),
        "coefficient": coefficient.tolist(),
    }


def apply_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    standardized = (x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    raw = np.asarray(model["target_mean"]) + standardized @ np.asarray(model["coefficient"])
    return np.maximum.accumulate(np.clip(raw, 0.0, 1.0), axis=1)


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def endpoint_camera_by_task(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task_name in sorted({row["task_name"] for row in trajectories if row["relation_eligible"]}):
        group = [row for row in trajectories if row["task_name"] == task_name]
        median_score = float(np.median([row["final_relation_score"] for row in group]))
        median_random = float(np.median([row["final_random_score"] for row in group]))
        median_advantage = float(np.median([row["final_relation_advantage"] for row in group]))
        rows.append({
            "task_name": task_name,
            "trajectory_count": len(group),
            "median_relation_score": median_score,
            "median_random_score": median_random,
            "median_advantage": median_advantage,
            "pass": median_score >= THRESHOLDS["camera_score_min"] and median_advantage >= THRESHOLDS["camera_advantage_min"],
        })
    return rows


def discovery_object_decision(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    events = [row for row in trajectories if row["event_observed"]]
    censored = [row for row in trajectories if not row["event_observed"]]
    event_tasks = {row["task_name"] for row in events}
    censored_tasks = {row["task_name"] for row in censored}
    endpoint_rows = endpoint_camera_by_task(trajectories)
    endpoint_breadth = sum(row["pass"] for row in endpoint_rows)
    checks = {
        "all_fit": all(row["fit_step"] is not None for row in trajectories),
        "all_finite": all(row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in trajectories),
        "event_count": len(events) >= THRESHOLDS["discovery_event_trajectory_min"],
        "censored_count": len(censored) >= THRESHOLDS["discovery_censored_trajectory_min"],
        "event_task_breadth": len(event_tasks) >= THRESHOLDS["discovery_event_task_class_min"],
        "censored_task_breadth": len(censored_tasks) >= THRESHOLDS["discovery_censored_task_class_min"],
        "event_time_breadth": len({row["stable_generalization_step"] for row in events}) >= THRESHOLDS["discovery_distinct_event_upper_bounds_min"],
        "endpoint_relation_camera_breadth": endpoint_breadth >= THRESHOLDS["discovery_endpoint_camera_class_min"],
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "event_count": len(events),
        "right_censored_count": len(censored),
        "event_task_names": sorted(event_tasks),
        "censored_task_names": sorted(censored_tasks),
        "event_upper_bounds": sorted({row["stable_generalization_step"] for row in events}),
        "endpoint_camera_by_task": endpoint_rows,
        "endpoint_camera_class_pass_count": endpoint_breadth,
        "cutoff_relation_score_std": float(np.std([row["cutoff_relation_score"] for row in trajectories])),
    }


def fit_and_seal_predictor_command() -> None:
    path = OUT_ROOT / "analysis/predictor_seal.json"
    if path.exists():
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
        "prediction_cutoff": PREDICTION_CUTOFF,
        "prediction_horizons": PREDICTION_HORIZONS,
        "ridge_l2": RIDGE_L2,
        "feature_names": {},
        "predictors": None,
        "discovery_scores": None,
    }
    if object_decision["pass"]:
        labels = np.asarray([event_labels(row) for row in trajectories], dtype=np.float64)
        predictors = {}
        scores = {}
        for model_name in MODEL_NAMES:
            names_and_values = [feature_vector(training_groups[row["trajectory_id"]], model_name) for row in trajectories]
            names = names_and_values[0][0]
            if any(item[0] != names for item in names_and_values):
                raise RuntimeError(f"feature order mismatch: {model_name}")
            features = np.asarray([item[1] for item in names_and_values], dtype=np.float64)
            predictor = fit_ridge(features, labels)
            predictors[model_name] = predictor
            seal["feature_names"][model_name] = names
            scores[model_name] = brier(labels, apply_ridge(predictor, features))
        constant_probability = np.maximum.accumulate(labels.mean(axis=0)).tolist()
        constant = np.tile(np.asarray(constant_probability), (len(labels), 1))
        scores["constant"] = brier(labels, constant)
        seal["predictors"] = {"constant_probability": constant_probability, **predictors}
        seal["discovery_scores"] = scores
    seal["predictor_digest"] = base.digest(seal)
    base.write_json(path, seal)
    print(json.dumps({
        "object_gate_pass": object_decision["pass"],
        "confirmation_reveal_authorized": seal["confirmation_reveal_authorized"],
        "predictor_digest": seal["predictor_digest"],
        "endpoint_camera_class_pass_count": object_decision["endpoint_camera_class_pass_count"],
        "cutoff_relation_score_std": object_decision["cutoff_relation_score_std"],
    }))


def score_command() -> None:
    predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
    if not predictor["confirmation_reveal_authorized"]:
        score: dict[str, Any] = {
            "phase": PHASE,
            "scored_at_utc": base.utc_now(),
            "stage": "discovery_object_gate_failure",
            "primary_endpoint_pass": False,
            "confirmation_evaluated": False,
            "object_decision": predictor["object_decision"],
            "interpretation": "The free-network endpoint relation object did not pass discovery; confirmation and all camera reinterpretation are forbidden.",
        }
    else:
        confirmation_summary = base.read_json(OUT_ROOT / "runs/holdout/confirmation/summary.json")
        trajectories, training_groups = grouped_trajectories("confirmation")
        labels = np.asarray([event_labels(row) for row in trajectories], dtype=np.float64)
        probabilities: dict[str, np.ndarray] = {
            "constant": np.tile(np.asarray(predictor["predictors"]["constant_probability"]), (len(labels), 1))
        }
        for model_name in MODEL_NAMES:
            features = np.asarray([
                feature_vector(training_groups[row["trajectory_id"]], model_name)[1] for row in trajectories
            ], dtype=np.float64)
            probabilities[model_name] = apply_ridge(predictor["predictors"][model_name], features)
        scores = {name: brier(labels, value) for name, value in probabilities.items()}
        control_names = tuple(name for name in probabilities if name != "relation")
        best_control_name = min(control_names, key=lambda name: scores[name])
        best_control = scores[best_control_name]
        relative_improvement = (best_control - scores["relation"]) / best_control if best_control > 0 else 0.0
        per_task = []
        for task_name in sorted({row["task_name"] for row in trajectories}):
            mask = np.asarray([row["task_name"] == task_name for row in trajectories], dtype=bool)
            task_scores = {name: brier(labels[mask], value[mask]) for name, value in probabilities.items()}
            per_task.append({
                "task_name": task_name,
                "scores": task_scores,
                "best_control_name": min(control_names, key=lambda name: task_scores[name]),
                "relation_beats_every_control": task_scores["relation"] < min(task_scores[name] for name in control_names),
            })
        class_advantage = sum(row["relation_beats_every_control"] for row in per_task)
        endpoint_rows = endpoint_camera_by_task(trajectories)
        endpoint_breadth = sum(row["pass"] for row in endpoint_rows)
        eligible_events = [row for row in trajectories if row["relation_eligible"] and row["event_observed"]]
        temporal = {
            "eligible_event_trajectory_count": len(eligible_events),
            "relation_camera_event_count": sum(row["relation_camera_event_observed"] for row in eligible_events),
            "strictly_precedes_generalization_count": sum(row["relation_camera_strictly_precedes_generalization"] for row in eligible_events),
            "follows_generalization_count": sum(row["relation_camera_follows_generalization"] for row in eligible_events),
            "median_generalization_step": float(np.median([row["stable_generalization_step"] for row in eligible_events])) if eligible_events else None,
            "median_relation_camera_step": float(np.median([row["relation_camera_step"] for row in eligible_events if row["relation_camera_step"] is not None])) if any(row["relation_camera_step"] is not None for row in eligible_events) else None,
        }
        endpoint_checks = {
            "all_fit": all(row["fit_step"] is not None for row in trajectories),
            "all_finite": all(row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in trajectories),
            "endpoint_relation_camera_breadth": endpoint_breadth >= THRESHOLDS["confirmation_endpoint_camera_class_min"],
            "relative_brier_improvement": relative_improvement >= THRESHOLDS["confirmation_relative_brier_improvement_min"],
            "prediction_class_breadth": class_advantage >= THRESHOLDS["confirmation_class_advantage_min"],
        }
        score = {
            "phase": PHASE,
            "scored_at_utc": base.utc_now(),
            "stage": "confirmation_prediction",
            "predictor_digest": predictor["predictor_digest"],
            "confirmation_summary_digest": confirmation_summary["summary_digest"],
            "primary_endpoint_pass": bool(all(endpoint_checks.values())),
            "endpoint_checks": endpoint_checks,
            "confirmation_scores": scores,
            "best_control_name": best_control_name,
            "best_control_brier": best_control,
            "relative_brier_improvement": relative_improvement,
            "confirmation_class_advantage_count": class_advantage,
            "per_confirmation_task": per_task,
            "endpoint_camera_by_task": endpoint_rows,
            "endpoint_camera_class_pass_count": endpoint_breadth,
            "temporal_order": temporal,
            "discovery_trajectories": grouped_trajectories("discovery")[0],
            "confirmation_trajectories": trajectories,
            "interpretation": {
                "if_pass": "A training-inferred relation camera adds prospective cross-quotient information and may proceed to a separately preregistered causal-use test.",
                "if_fail": "The calibrated affine camera lacks the frozen prospective external validity required here; close this branch without nonlinear-camera search.",
                "temporal_caution": "A camera event after generalization is compatible with late compression and is not formation-causal evidence.",
                "scope": "Controlled free role-square networks only.",
            },
        }
    score["score_digest"] = base.digest(score)
    base.write_json(OUT_ROOT / "analysis/score.json", score)
    print(json.dumps({
        "stage": score["stage"],
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "relative_brier_improvement": score.get("relative_brier_improvement"),
        "score_digest": score["score_digest"],
    }))


def finalize_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    predictor = base.read_json(OUT_ROOT / "analysis/predictor_seal.json")
    score = base.read_json(OUT_ROOT / "analysis/score.json")
    passed = bool(score["primary_endpoint_pass"])
    confirmation_tested = score["stage"] == "confirmation_prediction"
    endpoint_externality = bool(
        confirmation_tested and score["endpoint_checks"]["endpoint_relation_camera_breadth"]
    )
    final: dict[str, Any] = {
        "phase": PHASE,
        "finalized_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "predictor_digest": predictor["predictor_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "primary_endpoint_pass": passed,
            "training_inferred_relation_prediction_confirmed": passed,
            "free_network_endpoint_camera_externality": endpoint_externality,
            "confirmation_tested": confirmation_tested,
            "causal_use_authorized": passed,
            "nonlinear_camera_search_authorized": False,
            "hidden_feature_search_authorized": False,
            "auto_continue": passed,
            "authorized_next": "Phase1175: one independently preregistered relation causal-use intervention" if passed else None,
        },
        "claims": [
            "Relation keys were inferred from training pairs and labels without task formulas, holdout labels, or future event times.",
            "Whitening, effective rank, and affine maps used operator-fit backgrounds only; operator-test marginals were untouched.",
            "Nine relation-supporting and three abstention task classes were all new under the frozen Phase1172 quotient invariant.",
            "A positive endpoint would be prospective representation externality, not causal use or a natural-language mechanism.",
        ],
        "hard_boundary": (
            "Failure closes this frozen affine-camera branch.  It does not prove that no relation mechanism exists, "
            "but it forbids post-hoc relation-key replacement and nonlinear camera escalation."
        ),
    }
    final["final_digest"] = base.digest(final)
    base.write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({
        "primary_endpoint_pass": passed,
        "auto_continue": final["decision"]["auto_continue"],
        "final_digest": final["final_digest"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "protocol", "smoke", "train-and-seal", "reveal-discovery",
        "fit-and-seal-predictor", "reveal-confirmation", "score", "finalize",
    ))
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
