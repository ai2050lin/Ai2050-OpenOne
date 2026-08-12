#!/usr/bin/env python3
"""Excluded engineering pilot for Phase1172 cross-quotient task families.

This file is not evidence.  It checks fixed-size role-square trainability and
the observation-window suitability of candidate function families before the
formal task instances, seeds, endpoint, and discovery/confirmation split are
frozen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402


OUT = ROOT / "tests/glm5_temp/phase1172_cross_quotient_task_pilot.json"
P = 61
CHECKPOINTS = (100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000)
PILOT_SEED = 11720091


def functions() -> dict[str, Callable[[int, int], int]]:
    return {
        "add": lambda a, b: (a + b) % P,
        "mul": lambda a, b: (a * b) % P,
        "left_square_add": lambda a, b: (a * a + b) % P,
        "square_sum": lambda a, b: (a * a + b * b) % P,
        "left_cube_add": lambda a, b: (a * a * a + b) % P,
        "quad_mixed": lambda a, b: (a * a + a * b + b) % P,
        "circle": lambda a, b: (a * a + a * b + b * b) % P,
        "cubic_mixed": lambda a, b: (a * a * a + a * b + b) % P,
        "square_product_plus": lambda a, b: (a * a * b + a + b) % P,
        "biquadratic": lambda a, b: (a * a * b * b + a + b) % P,
        "distance_square": lambda a, b: ((a - b) * (a - b)) % P,
        "diagonal_bump": lambda a, b: (a + b + int(a == b)) % P,
        "xor": lambda a, b: (a ^ b) % P,
        "maximum": lambda a, b: max(a, b),
        "ordered_affine_gate": lambda a, b: ((a + b) if a < b else (a + 2 * b)) % P,
    }


def make_data(function: Callable[[int, int], int], seed: int) -> dict[str, torch.Tensor]:
    pairs = [(a, b) for a in range(P) for b in range(P)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * 0.5))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor([function(a, b) for a, b in pairs], dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {"train_x": x[mask_t], "train_y": y[mask_t], "holdout_x": x[~mask_t], "holdout_y": y[~mask_t]}


def quotient_signature(function: Callable[[int, int], int]) -> dict[str, object]:
    table = np.asarray([[function(a, b) for b in range(P)] for a in range(P)], dtype=np.int64)
    global_counts = np.bincount(table.ravel(), minlength=P)
    row_histograms = np.sort(np.stack([np.bincount(row, minlength=P) for row in table]), axis=1)
    column_histograms = np.sort(np.stack([np.bincount(column, minlength=P) for column in table.T]), axis=1)
    row_agreement, column_agreement = [], []
    for first in range(P):
        for second in range(first + 1, P):
            row_agreement.append(int(np.sum(table[first] == table[second])))
            column_agreement.append(int(np.sum(table[:, first] == table[:, second])))
    payload = {
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
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return {
        "digest": hashlib.sha256(canonical).hexdigest(),
        "global_count_range": [int(global_counts.min()), int(global_counts.max())],
        "row_distinct_range": [min(payload["row_distinct_output_counts"]), max(payload["row_distinct_output_counts"])],
        "column_distinct_range": [min(payload["column_distinct_output_counts"]), max(payload["column_distinct_output_counts"])],
        "distinct_row_count": payload["distinct_row_count"],
        "distinct_column_count": payload["distinct_column_count"],
    }


def run_one(name: str, function: Callable[[int, int], int], index: int) -> dict[str, object]:
    seed = PILOT_SEED + index * 1009
    base.set_seed(seed)
    data = make_data(function, seed + 17)
    device = torch.device("cuda")
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
    x, y = data["train_x"].to(device), data["train_y"].to(device)
    rows = []
    for step in range(1, max(CHECKPOINTS) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x).float()
            loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite pilot loss for {name} at {step}")
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            train = p1171.evaluate(model, data["train_x"], data["train_y"], device)
            holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
            rows.append({"step": step, "loss": float(loss.item()), "train_accuracy": train["accuracy"], "holdout_accuracy": holdout["accuracy"]})
    fit = next((row["step"] for row in rows if row["train_accuracy"] >= 0.99), None)
    stable = next((rows[i]["step"] for i in range(len(rows) - 1) if rows[i]["train_accuracy"] >= 0.99 and rows[i]["holdout_accuracy"] >= 0.90 and rows[i + 1]["train_accuracy"] >= 0.99 and rows[i + 1]["holdout_accuracy"] >= 0.90), None)
    return {
        "name": name,
        "seed": seed,
        "quotient_signature": quotient_signature(function),
        "fit_step": fit,
        "stable_generalization_step": stable,
        "maximum_holdout_accuracy": max(row["holdout_accuracy"] for row in rows),
        "final_holdout_accuracy": rows[-1]["holdout_accuracy"],
        "checkpoints": rows,
    }


def main() -> None:
    results = []
    for index, (name, function) in enumerate(functions().items()):
        result = run_one(name, function, index)
        results.append(result)
        print(json.dumps({key: result[key] for key in ("name", "fit_step", "stable_generalization_step", "maximum_holdout_accuracy")}), flush=True)
    payload = {
        "status": "excluded_engineering_pilot",
        "formal_evidence": False,
        "modulus": P,
        "parameter_count": 39808,
        "checkpoint_steps": CHECKPOINTS,
        "results": results,
    }
    base.write_json(OUT, payload)
    print(json.dumps({"output": str(OUT), "task_count": len(results)}))


if __name__ == "__main__":
    main()
