"""Reusable Phase1178 dual-path implementation-library components.

This module deliberately builds a *known-truth calibration object*, not a
model of natural language.  Two implementations compute the same finite task:

* ``instance_interpolation`` reads a complete pair table;
* ``relation_transfer`` reads a reusable offset and transports the first item.

Both live in the same symmetric two-slot supernetwork.  Their identity is
defined by a frozen intervention/repair response spectrum, never by a class
name, path slot, channel, norm, or architecture difference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


IMPLEMENTATIONS = ("instance_interpolation", "relation_transfer")
NEUTRAL_INTERVENTIONS = (
    "post_merge_scale_075",
    "post_merge_scale_050",
    "post_merge_common_bias",
    "post_merge_class_roll",
)
DIAGNOSTIC_INTERVENTIONS = (
    "disable_instance",
    "disable_relation",
    "roll_instance_address",
    "roll_relation_offset",
)
LOGIT_SCALE = 8.0
OBSERVATION_WIDTH = 24
MAX_MODULUS = 13


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    modulus: int
    formula: str
    coefficients: tuple[int, ...]

    def offsets(self) -> np.ndarray:
        """Return the task's reusable relation r(b), with arithmetic modulo p."""
        p = self.modulus
        values = []
        for b in range(p):
            total = 0
            for power, coefficient in enumerate(self.coefficients):
                total += coefficient * pow(b, power, p)
            values.append(total % p)
        return np.asarray(values, dtype=np.int64)

    def table(self) -> np.ndarray:
        p = self.modulus
        relation = self.offsets()
        return np.asarray(
            [[(a + int(relation[b])) % p for b in range(p)] for a in range(p)],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class ImplementationPayload:
    task: TaskSpec
    table_targets: np.ndarray
    relation_offsets: np.ndarray
    table_logits: np.ndarray
    relation_logits: np.ndarray
    payload_digest: str


class ImplementationFamilyGenerator:
    """Generate matched table and relation implementations for one task.

    The two physical experts receive identical payload tensors.  Which rule is
    executed is sealed runtime truth, so parameter count and tensor norm cannot
    disclose the implementation family.
    """

    def __init__(self, logit_scale: float = LOGIT_SCALE) -> None:
        self.logit_scale = float(logit_scale)

    def generate(self, task: TaskSpec) -> ImplementationPayload:
        p = task.modulus
        table = task.table()
        offsets = task.offsets()
        table_logits = np.zeros((p * p, p), dtype=np.float32)
        table_logits[np.arange(p * p), table.reshape(-1)] = self.logit_scale
        relation_logits = np.zeros((p, p), dtype=np.float32)
        relation_logits[np.arange(p), offsets] = self.logit_scale
        payload = {
            "task": task.name,
            "modulus": p,
            "table_targets": table.tolist(),
            "relation_offsets": offsets.tolist(),
            "logit_scale": self.logit_scale,
        }
        return ImplementationPayload(
            task=task,
            table_targets=table,
            relation_offsets=offsets,
            table_logits=table_logits,
            relation_logits=relation_logits,
            payload_digest=digest(payload),
        )


class SymmetricExpertPath(nn.Module):
    """One of two physically symmetric slots in the shared supernetwork."""

    def __init__(self, max_modulus: int = MAX_MODULUS) -> None:
        super().__init__()
        self.max_modulus = int(max_modulus)
        self.table_bank = nn.Parameter(
            torch.zeros(max_modulus * max_modulus, max_modulus),
            requires_grad=False,
        )
        self.relation_bank = nn.Parameter(
            torch.zeros(max_modulus, max_modulus),
            requires_grad=False,
        )

    @torch.no_grad()
    def load_payload(self, payload: ImplementationPayload) -> None:
        self.table_bank.zero_()
        self.relation_bank.zero_()
        p = payload.task.modulus
        self.table_bank[: p * p, :p].copy_(torch.from_numpy(payload.table_logits).to(self.table_bank))
        self.relation_bank[:p, :p].copy_(torch.from_numpy(payload.relation_logits).to(self.relation_bank))

    def execute(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        modulus: int,
        implementation: str,
        intervention: str | None = None,
    ) -> torch.Tensor:
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation: {implementation}")
        if implementation == "instance_interpolation":
            output = self.table_bank[a * modulus + b, :modulus]
            if intervention == "disable_instance":
                output = torch.zeros_like(output)
            elif intervention == "roll_instance_address":
                output = torch.roll(output, shifts=1, dims=-1)
        else:
            offset_logits = self.relation_bank[b, :modulus]
            indices = (torch.arange(modulus, device=a.device)[None, :] - a[:, None]) % modulus
            output = torch.gather(offset_logits, 1, indices)
            if intervention == "disable_relation":
                output = torch.zeros_like(output)
            elif intervention == "roll_relation_offset":
                output = torch.roll(output, shifts=1, dims=-1)
        return output


class SymmetricDualPathHypernetwork(nn.Module):
    """A same-architecture, same-budget two-path known-truth supernetwork."""

    architecture_version = "phase1178.symmetric_dual_path.v1"

    def __init__(self, payload: ImplementationPayload, device: torch.device) -> None:
        super().__init__()
        self.modulus = payload.task.modulus
        self.payload_digest = payload.payload_digest
        self.paths = nn.ModuleList((SymmetricExpertPath(), SymmetricExpertPath()))
        self.to(device)
        for path in self.paths:
            path.load_payload(payload)

    @property
    def architecture_digest(self) -> str:
        return digest({
            "version": self.architecture_version,
            "path_count": len(self.paths),
            "path_type": type(self.paths[0]).__name__,
            "parameter_shapes": [tuple(parameter.shape) for parameter in self.paths[0].parameters()],
        })

    @property
    def parameter_budget(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_l2(self) -> float:
        return math.sqrt(sum(float(torch.sum(parameter.double() ** 2).item()) for parameter in self.parameters()))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mechanisms_by_slot: tuple[str, str],
        active_slot: int,
        intervention: str | None = None,
        destroy_slot: int | None = None,
        rescue_slot: int | None = None,
        rescue_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for slot, (path, mechanism) in enumerate(zip(self.paths, mechanisms_by_slot)):
            value = path.execute(a, b, self.modulus, mechanism, intervention)
            if destroy_slot == slot:
                value = torch.zeros_like(value)
            if rescue_slot == slot and rescue_logits is not None:
                value = rescue_logits
            outputs.append(value)
        merged = outputs[active_slot]
        if intervention == "post_merge_scale_075":
            merged = 0.75 * merged
        elif intervention == "post_merge_scale_050":
            merged = 0.50 * merged
        elif intervention == "post_merge_common_bias":
            merged = merged + 3.0
        elif intervention == "post_merge_class_roll":
            merged = torch.roll(merged, shifts=1, dims=-1)
        return merged


def mechanisms_for(family: str, active_slot: int) -> tuple[str, str]:
    other = IMPLEMENTATIONS[1] if family == IMPLEMENTATIONS[0] else IMPLEMENTATIONS[0]
    values = [other, other]
    values[active_slot] = family
    values[1 - active_slot] = other
    return values[0], values[1]


def all_pairs(modulus: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.cartesian_prod(
        torch.arange(modulus, device=device),
        torch.arange(modulus, device=device),
    )
    return pairs[:, 0], pairs[:, 1]


def correct_margin(logits: torch.Tensor, target: torch.Tensor) -> float:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target[:, None], -torch.inf)
    alternative = torch.max(masked, dim=1).values
    return float(torch.mean(correct - alternative).item())


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((torch.argmax(logits, dim=1) == target).float()).item())


def response_spectrum(
    model: SymmetricDualPathHypernetwork,
    a: torch.Tensor,
    b: torch.Tensor,
    target: torch.Tensor,
    mechanisms_by_slot: tuple[str, str],
    active_slot: int,
    interventions: Iterable[str],
) -> dict[str, float]:
    baseline = correct_margin(model(a, b, mechanisms_by_slot, active_slot), target)
    return {
        name: correct_margin(
            model(a, b, mechanisms_by_slot, active_slot, intervention=name), target,
        ) - baseline
        for name in interventions
    }


def observation_state(
    a: torch.Tensor,
    b: torch.Tensor,
    logits: torch.Tensor,
    modulus: int,
    permutation: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Create the only state visible to a future camera.

    It is downstream of path merging.  A signed channel permutation is applied
    per matched block; the key remains sealed.  Matched implementation/slot
    systems therefore have byte-identical observations.
    """
    target = torch.argmax(logits, dim=1).double()
    af = a.double()
    bf = b.double()
    columns = []
    for source in (af, bf, target):
        for harmonic in range(1, 5):
            angle = 2.0 * math.pi * harmonic * source / modulus
            columns.extend((torch.sin(angle), torch.cos(angle)))
    state = torch.stack(columns, dim=1).detach().cpu().numpy().astype(np.float64)
    state = state[:, permutation] * signs[None, :]
    return state


def make_channel_gauge(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(OBSERVATION_WIDTH)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=OBSERVATION_WIDTH)
    return permutation.astype(np.int64), signs.astype(np.float64)


def gauge_invariant_gram(state: np.ndarray) -> np.ndarray:
    return state @ state.T


def diagnostic_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(float(left[name]) - float(right[name])) for name in DIAGNOSTIC_INTERVENTIONS)


def ridge_binary_accuracy(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    """Frozen basic linear null camera; no hyperparameter search is permitted."""
    train_x = np.asarray(train_x, dtype=np.float64).reshape(len(train_y), -1)
    test_x = np.asarray(test_x, dtype=np.float64).reshape(len(test_y), -1)
    mean = np.mean(train_x, axis=0, keepdims=True)
    scale = np.std(train_x, axis=0, keepdims=True)
    scale[scale < 1.0e-12] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    ridge = 1.0e-6 * np.eye(train.shape[1])
    ridge[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + ridge, train.T @ train_y)
    prediction = np.where(test @ weights >= 0.0, 1, -1)
    return float(np.mean(prediction == test_y))


def normalized_random_logits(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)
    random = torch.randn(reference.shape, generator=generator, device=reference.device, dtype=reference.dtype)
    ref_norm = torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    random_norm = torch.linalg.vector_norm(random, dim=1, keepdim=True).clamp_min(1.0e-12)
    return random * (ref_norm / random_norm)

