"""Development-only task-family qualification for Phase1183.

The pilot uses a different modulus, coefficients, and seeds from the formal
panel.  It may reject task families, but it must not tune any camera threshold.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402


MODULUS = 23
WIDTH = 128
STEPS = 3500
REPLICATES = 2
OUT = ROOT / "tests/glm5_temp/phase1183_task_family_pilot.json"


def task_functions():
    p = MODULUS
    return {
        "affine": lambda a, b: (5 * a + 8 * b + 9) % p,
        "product": lambda a, b: (((a + 2) % p) * ((b + 6) % p) + 4) % p,
        "left_square": lambda a, b: (((a + 3) % p) ** 2 + 7 * b + 5) % p,
        "right_square": lambda a, b: (9 * a + ((b + 4) % p) ** 2 + 7) % p,
        "left_cube": lambda a, b: (((a + 5) % p) ** 3 + 6 * b + 11) % p,
        "right_cube": lambda a, b: (4 * a + ((b + 7) % p) ** 3 + 2) % p,
        "square_sum": lambda a, b: (((a + 2) % p) ** 2 + 3 * ((b + 8) % p) ** 2 + 6) % p,
        "cube_square_sum": lambda a, b: (((a + 4) % p) ** 3 + 5 * ((b + 3) % p) ** 2 + 10) % p,
        "maximum": lambda a, b: (max((a + 5) % p, (b + 9) % p) + 3) % p,
        "minimum": lambda a, b: (min((a + 8) % p, (b + 4) % p) + 12) % p,
        "xor": lambda a, b: (((a ^ b) % p) + 7) % p,
        "or": lambda a, b: (((a | b) % p) + 5) % p,
        "absdiff": lambda a, b: (abs((a + 6) % p - (b + 10) % p) + 8) % p,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data(function, seed: int):
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = len(pairs) // 2
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor([function(a, b) for a, b in pairs], dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    return x[mask], y[mask], x[~mask], y[~mask]


@torch.inference_mode()
def accuracy(model, x, y, device):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x.to(device)).float()
    return float((logits.argmax(dim=1).cpu() == y).float().mean().item())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = []
    for task_index, (name, function) in enumerate(task_functions().items()):
        for replicate in range(REPLICATES):
            seed = 11830000 + task_index * 1009 + replicate * 97
            set_seed(seed)
            train_x, train_y, holdout_x, holdout_y = make_data(function, seed + 17)
            model = p1171.RoleSquareNetwork(
                p1171.RoleSquareConfig(modulus=MODULUS, width=WIDTH)
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
            tx, ty = train_x.to(device), train_y.to(device)
            for _ in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(tx).float(), ty)
                loss.backward()
                optimizer.step()
            row = {
                "task": name,
                "replicate": replicate,
                "seed": seed,
                "train_accuracy": accuracy(model, train_x, train_y, device),
                "holdout_accuracy": accuracy(model, holdout_x, holdout_y, device),
                "finite": bool(all(torch.isfinite(p).all() for p in model.parameters())),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model, optimizer, tx, ty
            torch.cuda.empty_cache()
    summary = {
        "development_only": True,
        "modulus": MODULUS,
        "width": WIDTH,
        "steps": STEPS,
        "replicates": REPLICATES,
        "tasks": {
            name: {
                "minimum_train_accuracy": min(r["train_accuracy"] for r in rows if r["task"] == name),
                "minimum_holdout_accuracy": min(r["holdout_accuracy"] for r in rows if r["task"] == name),
            }
            for name in task_functions()
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
