#!/usr/bin/env python3
"""Engineering-only pilot for the Phase1171 fixed-dimension learner.

Modulus 31 and every result from this script are permanently excluded from
formal evidence.  The pilot only checks representability and checkpoint timing
for a role-separated square network on asymmetric affine modular rules.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402


MODULUS = 31
WIDTH = 128
TRAIN_FRACTION = 0.50
CHECKPOINTS = (25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 10000)
OPERATIONS = ((2, 3, 5), (7, 11, 13))
SEEDS = (11710001, 11711010)


@dataclass(frozen=True)
class Config:
    modulus: int = MODULUS
    width: int = WIDTH


class RoleSquareNetwork(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.left_embedding = nn.Embedding(config.modulus, config.width)
        self.right_embedding = nn.Embedding(config.modulus, config.width)
        self.hidden = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.modulus, bias=False)
        for module in (self.left_embedding, self.right_embedding, self.hidden, self.output):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        summed = self.left_embedding(inputs[:, 0]) + self.right_embedding(inputs[:, 1])
        return self.output(self.hidden(summed).square())


def make_data(alpha: int, beta: int, gamma: int, seed: int) -> tuple[torch.Tensor, ...]:
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor([(alpha * a + beta * b + gamma) % MODULUS for a, b in pairs], dtype=torch.long)
    return x[mask], y[mask], x[~mask], y[~mask]


@torch.inference_mode()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x.cuda()).float()
    return float((logits.argmax(dim=-1).cpu() == y).float().mean().item())


def main() -> None:
    rows = []
    for operation_index, operation in enumerate(OPERATIONS):
        for replicate, seed in enumerate(SEEDS):
            run_seed = seed + operation_index * 100_003
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            train_x, train_y, holdout_x, holdout_y = make_data(*operation, run_seed + 17)
            model = RoleSquareNetwork(Config()).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)
            for step in range(1, max(CHECKPOINTS) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(train_x.cuda()).float(), train_y.cuda())
                loss.backward()
                optimizer.step()
                if step in CHECKPOINTS:
                    rows.append({
                        "operation": operation,
                        "replicate": replicate,
                        "seed": run_seed,
                        "step": step,
                        "loss": float(loss.item()),
                        "train_accuracy": accuracy(model, train_x, train_y),
                        "holdout_accuracy": accuracy(model, holdout_x, holdout_y),
                    })
            print(json.dumps({"operation": operation, "replicate": replicate, "final": rows[-1]}), flush=True)
    output = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_pilot.json"
    output.write_text(json.dumps({"evidence_status": "engineering_only_excluded", "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
