#!/usr/bin/env python3
"""Development-only gauge-aligned donor rescue pilot for Phase1182."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402


TARGET_ROWS = p1181.DISCOVERY_ROWS
CHECKPOINT_ROOT = p1171.OUT_ROOT / "runs/training/checkpoints"
OUTPUT = ROOT / "tests/glm5_temp/phase1182_donor_rescue_pilot.json"
FUTURE_MASK_COUNT = 32
FUTURE_MASK_SIZE = 8


def margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return p1181.correct_margin(logits, targets)


def split_holdout(panel: p1181.DataPanel) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.where(panel.holdout_mask)[0]
    code = panel.x[indices, 0] * 131 + panel.x[indices, 1] * 17
    calibration = torch.zeros_like(panel.holdout_mask)
    calibration[indices[code % 2 == 0]] = True
    evaluation = panel.holdout_mask & ~calibration
    return calibration, evaluation


@torch.inference_mode()
def state_bundle(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> dict[str, Any]:
    logits, hidden = p1181.fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    calibration_mask, evaluation_mask = split_holdout(panel)
    calibration = calibration_mask.to(device)
    evaluation = evaluation_mask.to(device)
    q = hidden.square()
    weight = model.output.weight.detach().float()
    base_margin = margin(logits, targets)
    responses: list[float] = []
    for channel in range(model.config.width):
        changed = logits - q[:, channel, None] * weight[:, channel][None, :]
        responses.append(float((base_margin[calibration] - margin(changed, targets)[calibration]).mean().item()))
    behavior = p1181.behavior_metrics(model, panel, device)
    behavior_vector = np.asarray([behavior[name] for name in p1181.BEHAVIOR_FEATURES], dtype=np.float64)
    return {
        "q_eval": q[evaluation].cpu(),
        "weight": weight.cpu(),
        "targets_eval": targets[evaluation].cpu(),
        "evaluation_mask": evaluation_mask,
        "calibration_response": np.asarray(responses, dtype=np.float64),
        "behavior": behavior,
        "behavior_vector": behavior_vector,
    }


def future_masks(width: int, seed: int) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    return [np.sort(generator.choice(width, size=FUTURE_MASK_SIZE, replace=False)) for _ in range(FUTURE_MASK_COUNT)]


@torch.inference_mode()
def evaluate_hybrid(
    q: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    masks: list[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    q = q.to(device)
    weight = weight.to(device)
    targets = targets.to(device)
    logits = q @ weight.T
    baseline_margin = margin(logits, targets)
    response: list[float] = []
    for channels in masks:
        index = torch.tensor(channels, dtype=torch.long, device=device)
        changed = logits - q[:, index] @ weight[:, index].T
        response.append(float((baseline_margin - margin(changed, targets)).mean().item()))
    return {
        "accuracy": float((logits.argmax(dim=1) == targets).float().mean().item()),
        "mean_margin": float(baseline_margin.mean().item()),
        "future_response": response,
    }


def normalized_error(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    scale = max(float(np.linalg.norm(right_array - right_array.mean())), 1e-8)
    return float(np.linalg.norm(left_array - right_array) / scale)


def run_task(task_rows: list[dict[str, Any]], injury_size: int, device: torch.device) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for row in task_rows:
        payload = torch.load(CHECKPOINT_ROOT / row["checkpoint"], map_location="cpu", weights_only=False)
        model = p1181.load_model(payload, device)
        panel = p1181.load_panel(payload, "discovery")
        bundle = state_bundle(model, panel, device)
        bundle["replicate"] = row["replicate"]
        bundle["model"] = model
        bundle["panel"] = panel
        bundles.append(bundle)
    behavior_matrix = np.stack([bundle["behavior_vector"] for bundle in bundles])
    behavior_matrix = (behavior_matrix - behavior_matrix.mean(axis=0)) / np.maximum(behavior_matrix.std(axis=0), 1e-12)
    results: list[dict[str, Any]] = []
    masks = future_masks(128, 11820000 + int(task_rows[0]["task_index"]) * 101)
    for recipient_index, recipient in enumerate(bundles):
        candidate_indices = [index for index in range(len(bundles)) if index != recipient_index]
        candidate_indices.sort(key=lambda index: float(np.linalg.norm(behavior_matrix[recipient_index] - behavior_matrix[index])))
        behavior_pool = candidate_indices[:4]
        recipient_ordered = np.sort(recipient["calibration_response"])
        response_distance = {
            index: float(np.linalg.norm(np.sort(bundles[index]["calibration_response"]) - recipient_ordered))
            for index in behavior_pool
        }
        correct_index = min(behavior_pool, key=lambda index: response_distance[index])
        wrong_index = max(behavior_pool, key=lambda index: response_distance[index])
        recipient_order = np.argsort(recipient["calibration_response"])
        recipient_rank = np.empty(len(recipient_order), dtype=np.int64)
        recipient_rank[recipient_order] = np.arange(len(recipient_order))
        injured_channels = np.argsort(np.abs(recipient["calibration_response"]))[-injury_size:]

        baseline = evaluate_hybrid(recipient["q_eval"], recipient["weight"], recipient["targets_eval"], masks, device)
        injured_q = recipient["q_eval"].clone()
        injured_q[:, injured_channels] = 0.0
        injured = evaluate_hybrid(injured_q, recipient["weight"], recipient["targets_eval"], masks, device)

        rescues: dict[str, Any] = {}
        for label, donor_index in (("correct", correct_index), ("wrong", wrong_index)):
            donor = bundles[donor_index]
            donor_order = np.argsort(donor["calibration_response"])
            _, donor_hidden = p1181.fp32_state(
                donor["model"], recipient["panel"].x, device
            )
            donor_q_eval = donor_hidden.square()[recipient["evaluation_mask"].to(device)].cpu()
            hybrid_q = recipient["q_eval"].clone()
            hybrid_weight = recipient["weight"].clone()
            for recipient_channel in injured_channels:
                donor_channel = donor_order[recipient_rank[recipient_channel]]
                hybrid_q[:, recipient_channel] = donor_q_eval[:, donor_channel]
                hybrid_weight[:, recipient_channel] = donor["weight"][:, donor_channel]
            evaluated = evaluate_hybrid(hybrid_q, hybrid_weight, recipient["targets_eval"], masks, device)
            evaluated["future_response_error"] = normalized_error(
                evaluated["future_response"], baseline["future_response"]
            )
            evaluated["donor_replicate"] = donor["replicate"]
            evaluated["calibration_response_distance"] = response_distance[donor_index]
            rescues[label] = evaluated
        injured["future_response_error"] = normalized_error(
            injured["future_response"], baseline["future_response"]
        )
        results.append(
            {
                "recipient_replicate": recipient["replicate"],
                "baseline": baseline,
                "injured": injured,
                "correct": rescues["correct"],
                "wrong": rescues["wrong"],
            }
        )
    for bundle in bundles:
        del bundle["model"]
    torch.cuda.empty_cache()
    return results


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(path: tuple[str, str]) -> float:
        return float(np.mean([row[path[0]][path[1]] for row in rows]))
    return {
        "baseline_accuracy": mean(("baseline", "accuracy")),
        "injured_accuracy": mean(("injured", "accuracy")),
        "correct_accuracy": mean(("correct", "accuracy")),
        "wrong_accuracy": mean(("wrong", "accuracy")),
        "injured_future_error": mean(("injured", "future_response_error")),
        "correct_future_error": mean(("correct", "future_response_error")),
        "wrong_future_error": mean(("wrong", "future_response_error")),
        "correct_minus_wrong_accuracy": mean(("correct", "accuracy")) - mean(("wrong", "accuracy")),
        "wrong_minus_correct_future_error": mean(("wrong", "future_response_error")) - mean(("correct", "future_response_error")),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    rows = p1181.read_jsonl(TARGET_ROWS)
    test_rows = [row for row in rows if row["task_index"] in (6, 7)]
    results: dict[str, Any] = {}
    for injury_size in (8, 16, 32):
        injury_rows: list[dict[str, Any]] = []
        for task_index in (6, 7):
            injury_rows.extend(
                run_task([row for row in test_rows if row["task_index"] == task_index], injury_size, device)
            )
        results[str(injury_size)] = {
            "aggregate": aggregate(injury_rows),
            "rows": injury_rows,
        }
    payload = {
        "status": "development_only",
        "task_indices": [6, 7],
        "future_mask_count": FUTURE_MASK_COUNT,
        "future_mask_size": FUTURE_MASK_SIZE,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value["aggregate"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
