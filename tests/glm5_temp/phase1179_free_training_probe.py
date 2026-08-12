#!/usr/bin/env python3
"""Development-only probe for a freely trained dual-path competition object."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ProbeTask:
    modulus: int = 7
    coefficients: tuple[int, ...] = (1, 2, 3)

    def target(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(a)
        for power, coefficient in enumerate(self.coefficients):
            total = total + coefficient * torch.pow(b, power)
        return (a + total) % self.modulus


class SymmetricExpert(nn.Module):
    def __init__(self, modulus: int) -> None:
        super().__init__()
        self.modulus = modulus
        self.table = nn.Parameter(torch.empty(modulus * modulus, modulus))
        self.relation = nn.Parameter(torch.empty(modulus, modulus))
        self.register_buffer("table_projection", torch.empty(modulus * modulus, 32))
        self.register_buffer("relation_projection", torch.empty(modulus, 32))
        nn.init.normal_(self.table, std=0.02)
        nn.init.normal_(self.relation, std=0.02)
        nn.init.normal_(self.table_projection, std=1.0 / math.sqrt(32))
        nn.init.normal_(self.relation_projection, std=1.0 / math.sqrt(32))

    def execute(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mode: str,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.modulus
        if mode == "table":
            address = a * p + b
            if intervention == "roll_table_address":
                address = (address + 1) % (p * p)
            logits = self.table[address]
            hidden = self.table_projection[address]
            if intervention == "disable_table":
                logits = torch.zeros_like(logits)
                hidden = torch.zeros_like(hidden)
        elif mode == "relation":
            address = b
            relation = self.relation[address]
            if intervention == "roll_relation_offset":
                relation = torch.roll(relation, shifts=1, dims=-1)
            indices = (torch.arange(p, device=a.device)[None, :] - a[:, None]) % p
            logits = torch.gather(relation, 1, indices)
            hidden = self.relation_projection[address]
            if intervention == "disable_relation":
                logits = torch.zeros_like(logits)
                hidden = torch.zeros_like(hidden)
        else:
            raise ValueError(mode)
        return logits, hidden


class FreeDualPath(nn.Module):
    def __init__(self, modulus: int, seed: int, swap: bool) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.modulus = modulus
        self.paths = nn.ModuleList([SymmetricExpert(modulus), SymmetricExpert(modulus)])
        self.modes = ("relation", "table") if swap else ("table", "relation")
        self.gate_logits = nn.Parameter(0.25 * torch.randn(2))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate_logits, dim=0)
        logits = []
        hidden = []
        for path, mode in zip(self.paths, self.modes):
            value, state = path.execute(a, b, mode, intervention)
            logits.append(value)
            hidden.append(state)
        contributions = torch.stack([weights[i] * logits[i] for i in range(2)])
        hidden_contributions = torch.stack([weights[i] * hidden[i] for i in range(2)])
        return contributions.sum(dim=0), hidden_contributions, weights


def correct_margin(logits: torch.Tensor, target: torch.Tensor) -> float:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target[:, None], -torch.inf)
    return float(torch.mean(correct - torch.max(masked, dim=1).values).item())


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((torch.argmax(logits, dim=1) == target).float()).item())


def split_pairs(task: ProbeTask, device: torch.device) -> tuple[torch.Tensor, ...]:
    pairs = torch.cartesian_prod(torch.arange(task.modulus, device=device), torch.arange(task.modulus, device=device))
    a, b = pairs[:, 0], pairs[:, 1]
    train = ((a + 2 * b) % task.modulus) < 4
    return a[train], b[train], a[~train], b[~train]


def run_one(seed: int, partial: bool, device: torch.device) -> dict[str, object]:
    task = ProbeTask()
    train_a, train_b, test_a, test_b = split_pairs(task, device)
    if not partial:
        train_a = torch.cat((train_a, test_a))
        train_b = torch.cat((train_b, test_b))
        test_a, test_b = train_a, train_b
    model = FreeDualPath(task.modulus, 1000 + seed, swap=bool(seed % 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    checkpoints = {}
    for step in range(1001):
        if step in (0, 10, 25, 50, 100, 200, 500, 1000):
            with torch.no_grad():
                train_logits, _, weights = model(train_a, train_b)
                test_logits, _, _ = model(test_a, test_b)
                checkpoints[str(step)] = {
                    "weights": [float(value) for value in weights.detach().cpu()],
                    "train_accuracy": accuracy(train_logits, task.target(train_a, train_b)),
                    "test_accuracy": accuracy(test_logits, task.target(test_a, test_b)),
                    "loss": float(torch.nn.functional.cross_entropy(train_logits, task.target(train_a, train_b)).item()),
                }
        if step == 1000:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, _, weights = model(train_a, train_b)
        loss = torch.nn.functional.cross_entropy(logits, task.target(train_a, train_b))
        loss = loss + 0.35 * weights[0] * weights[1]
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        all_a = torch.cartesian_prod(torch.arange(task.modulus, device=device), torch.arange(task.modulus, device=device))
        a, b = all_a[:, 0], all_a[:, 1]
        target = task.target(a, b)
        baseline, _, weights = model(a, b)
        baseline_margin = correct_margin(baseline, target)
        spectrum = {}
        for intervention in ("disable_table", "disable_relation", "roll_table_address", "roll_relation_offset"):
            changed, _, _ = model(a, b, intervention)
            spectrum[intervention] = correct_margin(changed, target) - baseline_margin
        return {
            "seed": seed,
            "partial": partial,
            "modes": list(model.modes),
            "final_weights": [float(value) for value in weights.cpu()],
            "winner": model.modes[int(torch.argmax(weights).item())],
            "all_accuracy": accuracy(baseline, target),
            "spectrum": spectrum,
            "checkpoints": checkpoints,
        }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = [run_one(seed, partial, device) for partial in (False, True) for seed in range(32)]
    summary = {}
    for partial in (False, True):
        subset = [row for row in rows if row["partial"] == partial]
        summary[str(partial)] = {
            "winner_counts": {name: sum(row["winner"] == name for row in subset) for name in ("table", "relation")},
            "all_accuracy_median": float(np.median([row["all_accuracy"] for row in subset])),
            "all_accuracy_min": float(np.min([row["all_accuracy"] for row in subset])),
            "gate_extreme_count": sum(max(row["final_weights"]) >= 0.95 for row in subset),
        }
    payload = {"device": torch.cuda.get_device_name(0), "summary": summary, "rows": rows}
    output = Path(__file__).with_name("phase1179_free_training_probe.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
