#!/usr/bin/env python3
"""Development probe for a semantic-free causal-response material gate.

This file is deliberately not a Phase1181 formal protocol.  It tests whether a
plain freely trained RoleSquareNetwork yields a non-degenerate, permutation-
quotiented response spectrum before a new camera protocol is preregistered.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1171_fixed_dimension_formation_trajectory_tomography import (
    RoleSquareConfig,
    RoleSquareNetwork,
)


DEFAULT_OUTPUT = ROOT / "tests/glm5_temp/phase1181_natural_response_material_probe.json"
MODULUS = 23
WIDTH = 64
TRAIN_FRACTION = 0.50
STEPS = 3000
TASKS = (
    (2, 5, 1),
    (3, 7, 4),
    (6, 11, 8),
    (9, 13, 3),
)
REPLICATES = 8


@dataclass(frozen=True)
class DataPanel:
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    holdout_mask: torch.Tensor


def make_data(operation: tuple[int, int, int], seed: int) -> DataPanel:
    alpha, beta, gamma = operation
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    train_mask = torch.zeros(len(pairs), dtype=torch.bool)
    train_mask[torch.tensor(order[:cutoff], dtype=torch.long)] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(
        [(alpha * a + beta * b + gamma) % MODULUS for a, b in pairs],
        dtype=torch.long,
    )
    return DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def correct_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], -torch.inf)
    return correct - masked.max(dim=1).values


@torch.inference_mode()
def logits_and_hidden(
    model: RoleSquareNetwork,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    x_device = x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        summed = model.left_embedding(x_device[:, 0]) + model.right_embedding(x_device[:, 1])
        hidden = model.hidden(summed)
        logits = model.output(hidden.square())
    return logits.float(), hidden.float()


@torch.inference_mode()
def response_spectrum(
    model: RoleSquareNetwork,
    panel: DataPanel,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    _, hidden = logits_and_hidden(model, panel.x, device)
    targets = panel.y.to(device)
    selected = mask.to(device)
    output_weight = model.output.weight.detach().float()
    logits = F.linear(hidden.square(), output_weight)
    baseline_margin = correct_margin(logits, targets)
    responses: list[float] = []
    channel_batch = 16
    squared = hidden.square()
    for start in range(0, hidden.shape[1], channel_batch):
        stop = min(start + channel_batch, hidden.shape[1])
        channels = torch.arange(start, stop, device=device)
        contribution = (
            squared[:, channels].transpose(0, 1)[:, :, None]
            * output_weight[:, channels].transpose(0, 1)[:, None, :]
        )
        changed_logits = logits[None, :, :] - contribution
        flat_targets = targets.repeat(stop - start)
        changed_margin = correct_margin(
            changed_logits.reshape(-1, changed_logits.shape[-1]),
            flat_targets,
        ).reshape(stop - start, -1)
        batch_response = (
            baseline_margin[None, selected] - changed_margin[:, selected]
        ).mean(dim=1)
        responses.extend(float(value) for value in batch_response.cpu().tolist())
    raw = np.asarray(responses, dtype=np.float64)
    ordered = np.sort(raw)
    centered = ordered - ordered.mean()
    centered_norm = float(np.linalg.norm(centered))
    unit_shape = centered / max(centered_norm, 1e-12)
    return {
        "raw": raw.tolist(),
        "ordered": ordered.tolist(),
        "unit_shape": unit_shape.tolist(),
        "mean": float(raw.mean()),
        "standard_deviation": float(raw.std()),
        "centered_norm": centered_norm,
        "mean_absolute_response": float(np.abs(raw).mean()),
        "maximum_absolute_response": float(np.abs(raw).max()),
    }


@torch.inference_mode()
def evaluate(
    model: RoleSquareNetwork,
    panel: DataPanel,
    device: torch.device,
) -> dict[str, float]:
    logits, _ = logits_and_hidden(model, panel.x, device)
    targets = panel.y.to(device)
    margins = correct_margin(logits, targets)
    predictions = logits.argmax(dim=1)
    result: dict[str, float] = {}
    for name, mask in (("train", panel.train_mask), ("holdout", panel.holdout_mask)):
        selected = mask.to(device)
        result[f"{name}_accuracy"] = float((predictions[selected] == targets[selected]).float().mean().item())
        result[f"{name}_loss"] = float(F.cross_entropy(logits[selected], targets[selected]).item())
        result[f"{name}_mean_margin"] = float(margins[selected].mean().item())
    result["parameter_norm"] = math.sqrt(
        sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters())
    )
    return result


def train_model(
    operation: tuple[int, int, int],
    task_index: int,
    replicate: int,
    device: torch.device,
) -> tuple[RoleSquareNetwork, DataPanel, dict[str, float]]:
    seed = 11810000 + task_index * 100_003 + replicate * 1_009
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    panel = make_data(operation, seed + 17)
    model = RoleSquareNetwork(RoleSquareConfig(modulus=MODULUS, width=WIDTH)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.5)
    train_x = panel.x[panel.train_mask].to(device)
    train_y = panel.y[panel.train_mask].to(device)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    return model, panel, evaluate(model, panel, device)


def gauge_transform(
    model: RoleSquareNetwork,
    seed: int,
    device: torch.device,
) -> RoleSquareNetwork:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = model.config.width
    permutation = torch.randperm(width, generator=generator)
    signs = torch.where(
        torch.rand(width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    transformed = RoleSquareNetwork(model.config).to(device)
    with torch.no_grad():
        transformed.left_embedding.weight.copy_(model.left_embedding.weight)
        transformed.right_embedding.weight.copy_(model.right_embedding.weight)
        transformed.hidden.weight.copy_(
            signs[:, None].to(device) * model.hidden.weight[permutation.to(device)]
        )
        transformed.output.weight.copy_(model.output.weight[:, permutation.to(device)])
    return transformed


def pairwise_distances(vectors: list[list[float]]) -> list[float]:
    array = np.asarray(vectors, dtype=np.float64)
    distances: list[float] = []
    for left in range(len(array)):
        for right in range(left + 1, len(array)):
            distances.append(float(np.linalg.norm(array[left] - array[right])))
    return distances


def correlation(values_a: list[float], values_b: list[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    systems: list[dict[str, Any]] = []
    gauge_checks: list[dict[str, float]] = []
    for task_index, operation in enumerate(TASKS):
        for replicate in range(REPLICATES):
            model, panel, behavior = train_model(operation, task_index, replicate, device)
            holdout_response = response_spectrum(model, panel, panel.holdout_mask, device)
            full_response = response_spectrum(model, panel, torch.ones_like(panel.train_mask), device)
            replay_response = response_spectrum(model, panel, panel.holdout_mask, device)
            replay_error = float(
                np.max(
                    np.abs(
                        np.asarray(holdout_response["ordered"])
                        - np.asarray(replay_response["ordered"])
                    )
                )
            )
            if task_index == 0 and replicate < 2:
                transformed = gauge_transform(model, 11819000 + replicate, device)
                original_logits, _ = logits_and_hidden(model, panel.x, device)
                transformed_logits, _ = logits_and_hidden(transformed, panel.x, device)
                transformed_response = response_spectrum(
                    transformed, panel, panel.holdout_mask, device
                )
                gauge_checks.append(
                    {
                        "maximum_logit_error": float(
                            (original_logits - transformed_logits).abs().max().item()
                        ),
                        "maximum_ordered_response_error": float(
                            np.max(
                                np.abs(
                                    np.asarray(holdout_response["ordered"])
                                    - np.asarray(transformed_response["ordered"])
                                )
                            )
                        ),
                    }
                )
            systems.append(
                {
                    "task_index": task_index,
                    "operation": list(operation),
                    "replicate": replicate,
                    "behavior": behavior,
                    "holdout_response": holdout_response,
                    "full_response": full_response,
                    "replay_maximum_error": replay_error,
                }
            )
            del model
            torch.cuda.empty_cache()

    qualified = [
        item
        for item in systems
        if item["behavior"]["train_accuracy"] >= 0.99
        and item["behavior"]["holdout_accuracy"] >= 0.90
        and item["holdout_response"]["centered_norm"] > 1e-8
    ]
    by_task: dict[str, Any] = {}
    for task_index in range(len(TASKS)):
        subset = [item for item in qualified if item["task_index"] == task_index]
        distances = pairwise_distances(
            [item["holdout_response"]["unit_shape"] for item in subset]
        )
        by_task[str(task_index)] = {
            "qualified_count": len(subset),
            "pair_count": len(distances),
            "median_unit_shape_distance": float(np.median(distances)) if distances else None,
            "minimum_unit_shape_distance": min(distances) if distances else None,
            "maximum_unit_shape_distance": max(distances) if distances else None,
        }

    response_scales = [item["holdout_response"]["centered_norm"] for item in qualified]
    summary = {
        "system_count": len(systems),
        "qualified_count": len(qualified),
        "minimum_train_accuracy": min(item["behavior"]["train_accuracy"] for item in systems),
        "minimum_holdout_accuracy": min(item["behavior"]["holdout_accuracy"] for item in systems),
        "maximum_replay_error": max(item["replay_maximum_error"] for item in systems),
        "maximum_gauge_logit_error": max(item["maximum_logit_error"] for item in gauge_checks),
        "maximum_gauge_response_error": max(
            item["maximum_ordered_response_error"] for item in gauge_checks
        ),
        "by_task": by_task,
        "response_scale_coefficient_of_variation": float(
            np.std(response_scales) / max(np.mean(response_scales), 1e-12)
        ) if response_scales else None,
        "response_scale_vs_holdout_accuracy_correlation": correlation(
            response_scales,
            [item["behavior"]["holdout_accuracy"] for item in qualified],
        ) if qualified else None,
        "response_scale_vs_parameter_norm_correlation": correlation(
            response_scales,
            [item["behavior"]["parameter_norm"] for item in qualified],
        ) if qualified else None,
    }
    payload = {
        "status": "development_probe_only",
        "config": {
            "modulus": MODULUS,
            "width": WIDTH,
            "steps": STEPS,
            "tasks": [list(task) for task in TASKS],
            "replicates": REPLICATES,
            "intervention": "single hidden channel pre-square ablation",
            "quotient": "sort channel response spectrum under signed channel permutations",
        },
        "summary": summary,
        "gauge_checks": gauge_checks,
        "systems": systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
