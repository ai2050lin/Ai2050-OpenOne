#!/usr/bin/env python3
"""Phase1180 norm-safe overlay for the frozen Phase1179 training library.

The scientific object and cameras are unchanged. Only the non-functional norm
ballast is widened and stored in float64 so that all preregistered moduli can be
trained without crossing the ballast ceiling.
"""

from __future__ import annotations

import torch
from torch import nn

import phase1179_free_training_library as base
from phase1179_free_training_library import *  # noqa: F401,F403


NORM_TARGET_MULTIPLIER = 100.0


class FreeDualPathNetwork(base.FreeDualPathNetwork):
    architecture_version = "phase1180.free_symmetric_dual_path.norm_safe.v1"

    def __init__(self, task: TaskSpec, seed: int, config_index: int, device: torch.device) -> None:
        super().__init__(task, seed, config_index, device)
        self.norm_target = NORM_TARGET_MULTIPLIER * task.modulus
        self.norm_ballast = nn.Parameter(
            torch.zeros(4, dtype=torch.float64, device=device),
            requires_grad=True,
        )
        self.match_parameter_norm()


def train_system(
    task: TaskSpec,
    cohort: str,
    seed: int,
    config_index: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = FreeDualPathNetwork(task, seed, config_index, device)
    train_a, train_b, holdout_a, holdout_b = train_holdout_pairs(task, cohort, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoints = []
    for step in range(TRAIN_STEPS + 1):
        if step in CHECKPOINT_STEPS:
            checkpoints.append(base.evaluate_checkpoint(
                model, task, cohort, step, train_a, train_b, holdout_a, holdout_b,
            ))
        if step == TRAIN_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        logits, _, weights = model(train_a, train_b)
        target = task.target(train_a, train_b)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss = loss + COMMITMENT_WEIGHT * weights[0] * weights[1]
        loss.backward()
        optimizer.step()
        model.match_parameter_norm()

    all_a, all_b = all_pairs(task, device)
    target = task.target(all_a, all_b)
    spectrum = response_spectrum(model, all_a, all_b, target)
    final = checkpoints[-1]
    truth = {
        "response_spectrum": spectrum,
        "response_family": family_from_spectrum(spectrum),
        "modes_by_slot": list(model.modes),
        "final_holdout_accuracy": final["holdout_accuracy"],
        "final_all_accuracy": final["all_accuracy"],
        "final_gate_weights": final["gate_weights"],
    }
    return checkpoints, truth
