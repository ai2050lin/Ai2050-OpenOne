#!/usr/bin/env python3
"""Engineering-only p=61 smoke test for the Phase1171 role learner."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5_temp"))

import phase1171_fixed_dimension_role_square_pilot as pilot  # noqa: E402


MODULUS = 61
OPERATION = (2, 3, 5)
SEEDS = (11712001, 11713010)
CHECKPOINTS = pilot.CHECKPOINTS


def make_data(seed: int) -> tuple[torch.Tensor, ...]:
    alpha, beta, gamma = OPERATION
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * 0.50))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor([(alpha * a + beta * b + gamma) % MODULUS for a, b in pairs], dtype=torch.long)
    return x[mask], y[mask], x[~mask], y[~mask]


@torch.inference_mode()
def accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x.cuda()).float()
    return float((logits.argmax(dim=-1).cpu() == y).float().mean().item())


def main() -> None:
    rows = []
    for replicate, seed in enumerate(SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_x, train_y, holdout_x, holdout_y = make_data(seed + 17)
        model = pilot.RoleSquareNetwork(pilot.Config(modulus=MODULUS, width=128)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
        train_x_device = train_x.cuda()
        train_y_device = train_y.cuda()
        for step in range(1, max(CHECKPOINTS) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(train_x_device).float(), train_y_device)
            loss.backward()
            optimizer.step()
            if step in CHECKPOINTS:
                rows.append({
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.item()),
                    "train_accuracy": accuracy(model, train_x, train_y),
                    "holdout_accuracy": accuracy(model, holdout_x, holdout_y),
                })
        print(json.dumps({"replicate": replicate, "final": rows[-1]}), flush=True)
    output = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_p61_pilot.json"
    output.write_text(json.dumps({"evidence_status": "engineering_only_excluded", "operation": OPERATION, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
