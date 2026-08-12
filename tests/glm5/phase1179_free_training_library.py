#!/usr/bin/env python3
"""Free-training components and fixed cameras for Phase1179.

The model learns two competing implementations under one optimizer and one
architecture. Physical slot, channel gauge, and implementation semantics are
balanced independently. Public cameras only receive gauge-invariant summaries
of the common residual sequence and non-diagnostic training scalars.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


MODES = ("table", "relation")
INTERVENTIONS = (
    "disable_table",
    "disable_relation",
    "roll_table_address",
    "roll_relation_offset",
)
CHECKPOINT_STEPS = (0, 5, 10, 25, 50, 100, 300)
PREFIX_STEP = 0
HIDDEN_WIDTH = 32
GATE_BIAS = 0.12
COMMITMENT_WEIGHT = 0.35
LEARNING_RATE = 0.04
TRAIN_STEPS = 300


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    modulus: int
    coefficients: tuple[int, ...]
    mask_shift: int

    def target(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(a)
        for power, coefficient in enumerate(self.coefficients):
            total = total + coefficient * torch.pow(b, power)
        return (a + total) % self.modulus

    def table(self) -> np.ndarray:
        a, b = np.meshgrid(np.arange(self.modulus), np.arange(self.modulus), indexing="ij")
        total = np.zeros_like(a)
        for power, coefficient in enumerate(self.coefficients):
            total += coefficient * np.power(b, power)
        return (a + total) % self.modulus


def all_pairs(task: TaskSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.cartesian_prod(
        torch.arange(task.modulus, device=device),
        torch.arange(task.modulus, device=device),
    )
    return pairs[:, 0], pairs[:, 1]


def train_holdout_pairs(
    task: TaskSpec,
    cohort: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = all_pairs(task, device)
    if cohort == "endpoint":
        return a, b, a, b
    if cohort != "formation":
        raise ValueError(cohort)
    width = int(math.ceil(0.60 * task.modulus))
    mask = ((a + 2 * b + task.mask_shift) % task.modulus) < width
    return a[mask], b[mask], a[~mask], b[~mask]


class SymmetricExpert(nn.Module):
    """One physical slot; both slots have the same parameterization and budget."""

    def __init__(self, modulus: int) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.table = nn.Parameter(torch.empty(modulus * modulus, modulus))
        self.relation = nn.Parameter(torch.empty(modulus, modulus))
        self.register_buffer("table_projection", torch.empty(modulus * modulus, HIDDEN_WIDTH))
        self.register_buffer("relation_projection", torch.empty(modulus, HIDDEN_WIDTH))
        nn.init.normal_(self.table, std=0.02)
        nn.init.normal_(self.relation, std=0.02)
        nn.init.normal_(self.table_projection, std=1.0 / math.sqrt(HIDDEN_WIDTH))
        nn.init.normal_(self.relation_projection, std=1.0 / math.sqrt(HIDDEN_WIDTH))

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
            return logits, hidden
        if mode == "relation":
            address = b
            values = self.relation[address]
            if intervention == "roll_relation_offset":
                values = torch.roll(values, shifts=1, dims=-1)
            indices = (torch.arange(p, device=a.device)[None, :] - a[:, None]) % p
            logits = torch.gather(values, 1, indices)
            hidden = self.relation_projection[address]
            if intervention == "disable_relation":
                logits = torch.zeros_like(logits)
                hidden = torch.zeros_like(hidden)
            return logits, hidden
        raise ValueError(mode)


class FreeDualPathNetwork(nn.Module):
    architecture_version = "phase1179.free_symmetric_dual_path.v1"

    def __init__(self, task: TaskSpec, seed: int, config_index: int, device: torch.device) -> None:
        super().__init__()
        if config_index not in range(4):
            raise ValueError(config_index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.task = task
        self.paths = nn.ModuleList((SymmetricExpert(task.modulus), SymmetricExpert(task.modulus)))
        self.modes = ("table", "relation") if config_index < 2 else ("relation", "table")
        first_wins = config_index in (0, 2)
        initial = torch.tensor(
            (GATE_BIAS, -GATE_BIAS) if first_wins else (-GATE_BIAS, GATE_BIAS),
            dtype=torch.float32,
        )
        self.gate_logits = nn.Parameter(initial)
        # Non-functional ballast removes global parameter-norm identity leakage.
        # It receives no forward gradient and follows the same task-level target
        # in every system, regardless of implementation outcome.
        self.norm_target = 10.0 * task.modulus
        self.norm_ballast = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.to(device)
        self.match_parameter_norm()

    @property
    def architecture_digest(self) -> str:
        return digest({
            "version": self.architecture_version,
            "path_count": 2,
            "expert": type(self.paths[0]).__name__,
            "shapes": [tuple(parameter.shape) for parameter in self.paths[0].parameters()],
        })

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
        logit_contributions = torch.stack([weights[index] * logits[index] for index in range(2)])
        hidden_contributions = torch.stack([weights[index] * hidden[index] for index in range(2)])
        return logit_contributions.sum(dim=0), hidden_contributions, weights

    def parameter_l2(self) -> float:
        total = sum(float(torch.sum(parameter.detach().double() ** 2).item()) for parameter in self.parameters())
        return math.sqrt(total)

    @torch.no_grad()
    def match_parameter_norm(self) -> None:
        total = sum(
            float(torch.sum(parameter.detach().double() ** 2).item())
            for name, parameter in self.named_parameters()
            if name != "norm_ballast"
        )
        if total >= self.norm_target ** 2:
            raise RuntimeError(
                f"functional parameter norm exceeded ballast target: {math.sqrt(total)} >= {self.norm_target}"
            )
        self.norm_ballast.zero_()
        self.norm_ballast[0] = math.sqrt(self.norm_target ** 2 - total)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((torch.argmax(logits, dim=1) == target).float()).item())


def confidence(logits: torch.Tensor, target: torch.Tensor) -> float:
    values = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    return float(torch.mean(values).item())


def correct_margin(logits: torch.Tensor, target: torch.Tensor) -> float:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target[:, None], -torch.inf)
    alternative = torch.max(masked, dim=1).values
    return float(torch.mean(correct - alternative).item())


def response_spectrum(
    model: FreeDualPathNetwork,
    a: torch.Tensor,
    b: torch.Tensor,
    target: torch.Tensor,
) -> list[float]:
    baseline, _, _ = model(a, b)
    base_margin = correct_margin(baseline, target)
    values = []
    for intervention in INTERVENTIONS:
        changed, _, _ = model(a, b, intervention)
        values.append(correct_margin(changed, target) - base_margin)
    return values


def family_from_spectrum(spectrum: list[float] | np.ndarray) -> str:
    values = np.abs(np.asarray(spectrum, dtype=np.float64))
    return "table" if values[0] >= values[1] else "relation"


def _mask_mean(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = matrix[mask]
    return float(np.mean(values)) if values.size else 0.0


def topology_summary(hidden: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, ...]:
    energy = float(np.mean(hidden * hidden))
    gram = hidden @ hidden.T
    diagonal = np.diag(gram)
    scale = max(float(np.mean(np.abs(diagonal))), 1.0e-12)
    gram = gram / scale
    eye = np.eye(len(a), dtype=bool)
    same_a = (a[:, None] == a[None, :]) & ~eye
    same_b = (b[:, None] == b[None, :]) & ~eye
    other = ~(same_a | same_b | eye)
    diag_mean = float(np.mean(np.diag(gram)))
    same_a_mean = _mask_mean(gram, same_a)
    same_b_mean = _mask_mean(gram, same_b)
    other_mean = _mask_mean(gram, other)
    return energy, same_a_mean - other_mean, same_b_mean - other_mean, diag_mean


def public_features(
    model: FreeDualPathNetwork,
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, list[float]]:
    with torch.no_grad():
        logits, contributions, weights = model(a, b)
    # The camera sees a common residual sequence. Increments are recovered from
    # consecutive residual states, as in a layerwise residual stream.
    residual_1 = contributions[0]
    residual_2 = contributions[0] + contributions[1]
    increments = (residual_1, residual_2 - residual_1)
    aa = a.detach().cpu().numpy()
    bb = b.detach().cpu().numpy()
    summaries = [topology_summary(value.detach().cpu().numpy(), aa, bb) for value in increments]
    energies = np.asarray([row[0] for row in summaries], dtype=np.float64)
    energy_ratio = energies / max(float(np.sum(energies)), 1.0e-12)
    topology = np.asarray([value for row in summaries for value in row[1:]], dtype=np.float64)
    relation_signature = np.asarray([row[2] for row in summaries], dtype=np.float64)
    table_signature = -relation_signature
    interaction = np.asarray([
        energy_ratio[0] * relation_signature[0],
        energy_ratio[1] * relation_signature[1],
        energy_ratio[0] * table_signature[0],
        energy_ratio[1] * table_signature[1],
        (energy_ratio[0] - energy_ratio[1]) * (relation_signature[0] - relation_signature[1]),
    ], dtype=np.float64)
    joint = np.concatenate((energy_ratio, topology, interaction))
    output = np.asarray([
        float(torch.mean(logits).item()),
        float(torch.std(logits).item()),
        float(torch.max(logits).item()),
        float(torch.min(logits).item()),
    ])
    return {
        "joint_topology_energy": joint.tolist(),
        "energy_only": energy_ratio.tolist(),
        "topology_only": topology.tolist(),
        "gate_only": weights.detach().cpu().double().numpy().tolist(),
        "output_only": output.tolist(),
    }


def evaluate_checkpoint(
    model: FreeDualPathNetwork,
    task: TaskSpec,
    cohort: str,
    step: int,
    train_a: torch.Tensor,
    train_b: torch.Tensor,
    holdout_a: torch.Tensor,
    holdout_b: torch.Tensor,
) -> dict[str, Any]:
    all_a, all_b = all_pairs(task, train_a.device)
    with torch.no_grad():
        train_logits, _, weights = model(train_a, train_b)
        holdout_logits, _, _ = model(holdout_a, holdout_b)
        all_logits, _, _ = model(all_a, all_b)
        train_target = task.target(train_a, train_b)
        holdout_target = task.target(holdout_a, holdout_b)
        all_target = task.target(all_a, all_b)
        loss = float(torch.nn.functional.cross_entropy(train_logits, train_target).item())
        behavior = [
            accuracy(train_logits, train_target),
            loss,
            confidence(train_logits, train_target),
            correct_margin(train_logits, train_target),
            model.parameter_l2(),
        ]
        payload = {
            "step": step,
            "cohort": cohort,
            "train_accuracy": behavior[0],
            "train_loss": behavior[1],
            "train_confidence": behavior[2],
            "train_margin": behavior[3],
            "parameter_l2": behavior[4],
            "progress": float(step) / TRAIN_STEPS,
            "holdout_accuracy": accuracy(holdout_logits, holdout_target),
            "all_accuracy": accuracy(all_logits, all_target),
            "gate_weights": weights.detach().cpu().double().numpy().tolist(),
        }
    gradient_logits, _, gradient_weights = model(train_a, train_b)
    gradient_loss = torch.nn.functional.cross_entropy(gradient_logits, task.target(train_a, train_b))
    gradient_loss = gradient_loss + COMMITMENT_WEIGHT * gradient_weights[0] * gradient_weights[1]
    gradients = torch.autograd.grad(gradient_loss, tuple(model.parameters()), allow_unused=True)
    gradient_l2 = math.sqrt(sum(
        float(torch.sum(value.detach().double() ** 2).item())
        for value in gradients if value is not None
    ))
    behavior.append(gradient_l2)
    behavior.append(float(step) / TRAIN_STEPS)
    payload["gradient_l2"] = gradient_l2
    payload["features"] = {
        **public_features(model, all_a, all_b),
        "behavior_only": behavior,
        "progress_only": [float(step) / TRAIN_STEPS],
    }
    return payload


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
            checkpoints.append(evaluate_checkpoint(
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


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1.0e-6) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-12] = 1.0
    z = (x - mean) / scale
    z = np.column_stack((np.ones(len(z)), z))
    penalty = ridge * np.eye(z.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist()}


def apply_ridge(camera: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(camera["mean"], dtype=np.float64)
    scale = np.asarray(camera["scale"], dtype=np.float64)
    weights = np.asarray(camera["weights"], dtype=np.float64)
    z = (x - mean) / scale
    z = np.column_stack((np.ones(len(z)), z))
    return z @ weights


def camera_metrics(prediction: np.ndarray, truth: np.ndarray, scale: float) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    predicted_family = np.where(np.abs(prediction[:, 0]) >= np.abs(prediction[:, 1]), "table", "relation")
    true_family = np.where(np.abs(truth[:, 0]) >= np.abs(truth[:, 1]), "table", "relation")
    error = np.max(np.abs(prediction - truth), axis=1)
    return {
        "family_accuracy": float(np.mean(predicted_family == true_family)),
        "median_linf_error": float(np.median(error)),
        "normalized_median_linf_error": float(np.median(error) / max(scale, 1.0e-12)),
    }
