"""Exclusion-only free-network pilot for the Phase1174 camera.

The task and seed are excluded from the formal registry. Whitening is fit only
on operator-fit backgrounds and is then frozen before test backgrounds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402


OUT = ROOT / "tests/glm5_temp/phase1174_inductive_relation_camera_training_pilot.json"
P = 61
SEED = 1_174_123
STEPS = (0, 25, 50, 100, 150, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000)
TRANSFORMS = (1, 2, 3)


def table() -> np.ndarray:
    return np.asarray(
        [[pow((a + pow(b, 8, P)) % P, 47, P) for b in range(P)] for a in range(P)],
        dtype=np.int64,
    )


def material() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    labels = table()
    pairs = np.asarray([(a, b) for a in range(P) for b in range(P)], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    mask = np.zeros(P * P, dtype=bool)
    mask[order[: (P * P) // 2]] = True
    mask = mask.reshape(P, P)
    backgrounds = rng.permutation(P)
    return pairs, labels, mask, (backgrounds[:20], backgrounds[20:40], backgrounds[40:])


def edge_coordinates(mask: np.ndarray, contexts: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    source, target = [], []
    for b in map(int, contexts):
        for a in range(P):
            aa = (a + shift) % P
            if mask[a, b] and mask[aa, b]:
                source.append((a, b))
                target.append((aa, b))
    return np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)


def infer_mappings(labels: np.ndarray, mask: np.ndarray, key_contexts: np.ndarray) -> dict[int, np.ndarray]:
    mappings = {}
    for shift in TRANSFORMS:
        source, target = edge_coordinates(mask, key_contexts, shift)
        counts = np.zeros((P, P), dtype=np.int64)
        for (a, b), (aa, bb) in zip(source, target):
            counts[int(labels[a, b]), int(labels[aa, bb])] += 1
        totals = counts.sum(axis=1)
        mapping = np.full(P, -1, dtype=np.int64)
        supported = totals >= 2
        mapping[supported] = np.argmax(counts[supported], axis=1)
        mappings[shift] = mapping
    return mappings


@torch.inference_mode()
def hidden_grid(model: p1171.RoleSquareNetwork, mask: np.ndarray, device: torch.device) -> np.ndarray:
    coordinates = np.argwhere(mask)
    inputs = torch.tensor(coordinates, dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        summed = model.left_embedding(inputs[:, 0]) + model.right_embedding(inputs[:, 1])
        hidden = model.hidden(summed).square()
    grid = np.full((P, P, hidden.shape[-1]), np.nan, dtype=np.float64)
    grid[coordinates[:, 0], coordinates[:, 1]] = hidden.float().cpu().numpy().astype(np.float64)
    return grid


def fit_whitener(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > max(float(eigenvalues.max()), 1.0e-12) * 1.0e-10
    transform = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    return mean, transform


def transform(values: np.ndarray, mean: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    return (values - mean) @ whitener


def fit_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    penalty = np.eye(design.shape[1]) * 1.0e-7
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def apply_map(source: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return np.concatenate((source, np.ones((len(source), 1))), axis=1) @ operator


def score_camera(
    grid: np.ndarray,
    mask: np.ndarray,
    fit_contexts: np.ndarray,
    test_contexts: np.ndarray,
    randomize: bool,
) -> dict[str, float]:
    fit_edges = {shift: edge_coordinates(mask, fit_contexts, shift) for shift in TRANSFORMS}
    all_fit_coordinates = np.unique(
        np.concatenate([np.concatenate(pair, axis=0) for pair in fit_edges.values()], axis=0),
        axis=0,
    )
    fit_values = grid[all_fit_coordinates[:, 0], all_fit_coordinates[:, 1]]
    mean, whitener = fit_whitener(fit_values)
    operators = {}
    rng = np.random.default_rng(SEED + 9001)
    for shift, (source_coordinates, target_coordinates) in fit_edges.items():
        source = transform(grid[source_coordinates[:, 0], source_coordinates[:, 1]], mean, whitener)
        target = transform(grid[target_coordinates[:, 0], target_coordinates[:, 1]], mean, whitener)
        if randomize:
            target = target[rng.permutation(len(target))]
        operators[shift] = fit_map(source, target)

    reuse_num = reuse_den = 0.0
    for shift in TRANSFORMS:
        source_coordinates, target_coordinates = edge_coordinates(mask, test_contexts, shift)
        source = transform(grid[source_coordinates[:, 0], source_coordinates[:, 1]], mean, whitener)
        target = transform(grid[target_coordinates[:, 0], target_coordinates[:, 1]], mean, whitener)
        prediction = apply_map(source, operators[shift])
        reuse_num += float(np.sum((prediction - target) ** 2))
        reuse_den += float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))

    closure_num = closure_den = 0.0
    source_coordinates, direct_coordinates = [], []
    for b in map(int, test_contexts):
        for a in range(P):
            if mask[a, b] and mask[(a + 1) % P, b] and mask[(a + 3) % P, b]:
                source_coordinates.append((a, b))
                direct_coordinates.append(((a + 3) % P, b))
    source_coordinates = np.asarray(source_coordinates, dtype=np.int64)
    direct_coordinates = np.asarray(direct_coordinates, dtype=np.int64)
    source = transform(grid[source_coordinates[:, 0], source_coordinates[:, 1]], mean, whitener)
    actual = transform(grid[direct_coordinates[:, 0], direct_coordinates[:, 1]], mean, whitener)
    direct = apply_map(source, operators[3])
    composed = apply_map(apply_map(source, operators[1]), operators[2])
    closure_num = float(np.sum((direct - composed) ** 2) + np.sum((composed - actual) ** 2))
    closure_den = float(2.0 * np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))
    reuse_error = reuse_num / max(reuse_den, 1.0e-12)
    closure_error = closure_num / max(closure_den, 1.0e-12)
    reuse = float(np.clip(1.0 - reuse_error, 0.0, 1.0))
    closure = float(np.clip(1.0 - closure_error, 0.0, 1.0))
    return {
        "reuse": reuse,
        "closure": closure,
        "score": float(math.sqrt(reuse * closure)),
        "effective_rank": int(whitener.shape[1]),
        "fit_state_count": int(len(all_fit_coordinates)),
        "closure_test_count": int(len(source_coordinates)),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    pairs, labels, mask, (key_contexts, fit_contexts, test_contexts) = material()
    mappings = infer_mappings(labels, mask, key_contexts)
    config = p1171.RoleSquareConfig(modulus=P, width=128)
    model = p1171.RoleSquareNetwork(config).to(device)
    train_coordinates = np.argwhere(mask)
    holdout_coordinates = np.argwhere(~mask)
    train_x = torch.tensor(train_coordinates, dtype=torch.long, device=device)
    train_y = torch.tensor(labels[mask], dtype=torch.long, device=device)
    holdout_x = torch.tensor(holdout_coordinates, dtype=torch.long, device=device)
    holdout_y = torch.tensor(labels[~mask], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
    rows = []
    for step in range(max(STEPS) + 1):
        if step in STEPS:
            grid = hidden_grid(model, mask, device)
            actual = score_camera(grid, mask, fit_contexts, test_contexts, randomize=False)
            random = score_camera(grid, mask, fit_contexts, test_contexts, randomize=True)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(train_x).float()
                accuracy = float((logits.argmax(dim=-1) == train_y).float().mean().item())
                loss = float(F.cross_entropy(logits, train_y).item())
                holdout_logits = model(holdout_x).float()
                holdout_accuracy = float((holdout_logits.argmax(dim=-1) == holdout_y).float().mean().item())
            rows.append({"step": step, "loss": loss, "train_accuracy": accuracy, "holdout_accuracy": holdout_accuracy, "actual": actual, "random": random})
            print(json.dumps(rows[-1]), flush=True)
        if step == max(STEPS):
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    payload = {
        "scope": "exclusion_only_non_evidential_pilot",
        "task": "inner_power_8_outer_power_47",
        "seed": SEED,
        "formal_task_or_seed": False,
        "key_mapping_supported_counts": {str(key): int(np.sum(value >= 0)) for key, value in mappings.items()},
        "whitening_fit_background_only": True,
        "test_background_marginals_used_for_whitening": False,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
