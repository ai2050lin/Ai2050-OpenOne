#!/usr/bin/env python3
"""Development-only endpoint/prefix camera pilot for Phase1182."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402


TARGET_ROWS = p1181.DISCOVERY_ROWS
CHECKPOINT_ROOT = p1171.OUT_ROOT / "runs/training/checkpoints"
OUTPUT = ROOT / "tests/glm5_temp/phase1182_camera_increment_pilot.json"
QUANTILES = np.linspace(0.0, 1.0, 17)
HISTORY_STEPS = (25, 50, 75, 100, 150)


@torch.inference_mode()
def output_features(model: p1171.RoleSquareNetwork, panel: p1181.DataPanel, device: torch.device) -> list[float]:
    logits, _ = p1181.fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    margins = p1181.correct_margin(logits, targets)
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities.max(dim=1).values
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    logit_norm = logits.norm(dim=1)
    features: list[float] = []
    for mask in (panel.train_mask, panel.holdout_mask):
        selected = mask.to(device)
        predictions = logits[selected].argmax(dim=1)
        features.extend(
            [
                float((predictions == targets[selected]).float().mean().item()),
                float(F.cross_entropy(logits[selected], targets[selected]).item()),
            ]
        )
        for values in (margins[selected], confidence[selected], target_probability[selected], entropy[selected], logit_norm[selected]):
            array = values.detach().cpu().numpy().astype(np.float64)
            features.extend([float(array.mean()), float(array.std())])
            features.extend(float(value) for value in np.quantile(array, (0.1, 0.25, 0.5, 0.75, 0.9)))
    return features


def gradient_features(model: p1171.RoleSquareNetwork, panel: p1181.DataPanel, device: torch.device) -> list[float]:
    model.zero_grad(set_to_none=True)
    train_x = panel.x[panel.train_mask].to(device)
    train_y = panel.y[panel.train_mask].to(device)
    logits = model(train_x).float()
    loss = F.cross_entropy(logits, train_y)
    loss.backward()
    groups = (
        model.left_embedding.weight.grad,
        model.right_embedding.weight.grad,
        model.hidden.weight.grad,
        model.output.weight.grad,
    )
    norms = [float(gradient.detach().float().norm().item()) for gradient in groups]
    model.zero_grad(set_to_none=True)
    return [float(loss.item()), *norms, float(math.sqrt(sum(value * value for value in norms)))]


@torch.inference_mode()
def internal_features(model: p1171.RoleSquareNetwork, panel: p1181.DataPanel, device: torch.device) -> list[float]:
    _, hidden = p1181.fp32_state(model, panel.x, device)
    hidden_weight = model.hidden.weight.detach().float()
    output_weight = model.output.weight.detach().float()
    left_projection = F.linear(model.left_embedding.weight.detach().float(), hidden_weight)
    right_projection = F.linear(model.right_embedding.weight.detach().float(), hidden_weight)
    arrays = [
        hidden_weight.norm(dim=1).cpu().numpy(),
        output_weight.norm(dim=0).cpu().numpy(),
        left_projection.norm(dim=0).cpu().numpy(),
        right_projection.norm(dim=0).cpu().numpy(),
        hidden.abs().mean(dim=0).cpu().numpy(),
        hidden.std(dim=0).cpu().numpy(),
        hidden.square().mean(dim=0).cpu().numpy(),
        output_weight.abs().mean(dim=0).cpu().numpy(),
    ]
    features: list[float] = []
    for array in arrays:
        features.extend(float(value) for value in np.quantile(array, QUANTILES))
    matrix = np.stack(arrays, axis=1)
    correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
    features.extend(float(correlation[left, right]) for left in range(len(arrays)) for right in range(left + 1, len(arrays)))
    singular = torch.linalg.svdvals(hidden - hidden.mean(dim=0, keepdim=True)).cpu().numpy()
    singular = singular / max(float(np.linalg.norm(singular)), 1e-12)
    features.extend(float(value) for value in np.quantile(singular, QUANTILES))
    return features


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, l2: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-8)
    x = (train_x - mean) / scale
    z = (test_x - mean) / scale
    x = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    z = np.concatenate((np.ones((len(z), 1)), z), axis=1)
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ train_y)
    return z @ weights


def normalize_rows(array: np.ndarray) -> np.ndarray:
    centered = array - array.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)


def trajectory_summary(rows: list[list[float]]) -> list[float]:
    matrix = np.asarray(rows, dtype=np.float64)
    time = np.log(np.asarray(HISTORY_STEPS, dtype=np.float64))
    time = (time - time.mean()) / time.std()
    slope = (time[:, None] * matrix).sum(axis=0) / float((time * time).sum())
    return np.concatenate((matrix[0], matrix[-1], matrix[-1] - matrix[0], slope)).tolist()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = normalize_rows(prediction)
    target = normalize_rows(target)
    cosine = np.sum(prediction * target, axis=1)
    error = np.linalg.norm(prediction - target, axis=1)
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "mean_l2_error": float(error.mean()),
        "mean_squared_error": float(np.mean((prediction - target) ** 2)),
    }


def residual_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    prediction_residual = np.zeros_like(prediction)
    target_residual = np.zeros_like(target)
    for group in np.unique(groups):
        selected = groups == group
        prediction_residual[selected] = prediction[selected] - prediction[selected].mean(axis=0)
        target_residual[selected] = target[selected] - target[selected].mean(axis=0)
    prediction_residual = normalize_rows(prediction_residual)
    target_residual = normalize_rows(target_residual)
    cosine = np.sum(prediction_residual * target_residual, axis=1)
    error = np.linalg.norm(prediction_residual - target_residual, axis=1)
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "mean_l2_error": float(error.mean()),
        "mean_squared_error": float(np.mean((prediction_residual - target_residual) ** 2)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    target_rows = p1181.read_jsonl(TARGET_ROWS)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(target_rows):
        endpoint_path = CHECKPOINT_ROOT / row["checkpoint"]
        endpoint_payload = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        panel = p1181.load_panel(endpoint_payload, "discovery")
        endpoint_model = p1181.load_model(endpoint_payload, device)
        endpoint_output = output_features(endpoint_model, panel, device)
        endpoint_gradient = gradient_features(endpoint_model, panel, device)
        endpoint_internal = internal_features(endpoint_model, panel, device)
        history_null_rows: list[list[float]] = []
        history_internal_rows: list[list[float]] = []
        for step in HISTORY_STEPS:
            history_path = CHECKPOINT_ROOT / row["checkpoint"].replace("step10000", f"step{step:05d}")
            history_payload = torch.load(history_path, map_location="cpu", weights_only=False)
            history_model = p1181.load_model(history_payload, device)
            history_null_rows.append(
                output_features(history_model, panel, device)
                + gradient_features(history_model, panel, device)
            )
            history_internal_rows.append(internal_features(history_model, panel, device))
            del history_model
        prefix_null = trajectory_summary(history_null_rows)
        prefix_internal = trajectory_summary(history_internal_rows)
        records.append(
            {
                "task_index": row["task_index"],
                "replicate": row["replicate"],
                "target": row["response"]["ordered"],
                "endpoint_null": endpoint_output + endpoint_gradient + prefix_null,
                "endpoint_internal": endpoint_internal,
                "prefix_null": prefix_null,
                "prefix_internal": prefix_internal,
            }
        )
        del endpoint_model
        torch.cuda.empty_cache()
        print(json.dumps({"completed": index + 1, "total": len(target_rows)}), flush=True)

    train = [record for record in records if record["task_index"] < 6]
    test = [record for record in records if record["task_index"] >= 6]
    target_train = np.asarray([record["target"] for record in train], dtype=np.float64)
    target_test = np.asarray([record["target"] for record in test], dtype=np.float64)
    test_groups = np.asarray([record["task_index"] for record in test], dtype=np.int64)
    results: dict[str, Any] = {}
    for stage in ("endpoint", "prefix"):
        null_train = np.asarray([record[f"{stage}_null"] for record in train], dtype=np.float64)
        null_test = np.asarray([record[f"{stage}_null"] for record in test], dtype=np.float64)
        internal_train = np.asarray([record[f"{stage}_internal"] for record in train], dtype=np.float64)
        internal_test = np.asarray([record[f"{stage}_internal"] for record in test], dtype=np.float64)
        stage_results: dict[str, Any] = {}
        for l2 in (0.1, 1.0, 10.0, 100.0, 1000.0):
            null_prediction = fit_ridge(null_train, target_train, null_test, l2)
            joint_prediction = fit_ridge(
                np.concatenate((null_train, internal_train), axis=1),
                target_train,
                np.concatenate((null_test, internal_test), axis=1),
                l2,
            )
            null_metrics = metrics(null_prediction, target_test)
            joint_metrics = metrics(joint_prediction, target_test)
            null_residual = residual_metrics(null_prediction, target_test, test_groups)
            joint_residual = residual_metrics(joint_prediction, target_test, test_groups)
            stage_results[str(l2)] = {
                "null": null_metrics,
                "joint": joint_metrics,
                "risk_improvement": null_metrics["mean_squared_error"] - joint_metrics["mean_squared_error"],
                "cosine_improvement": joint_metrics["mean_cosine"] - null_metrics["mean_cosine"],
                "null_residual": null_residual,
                "joint_residual": joint_residual,
                "residual_risk_improvement": null_residual["mean_squared_error"] - joint_residual["mean_squared_error"],
                "residual_cosine_improvement": joint_residual["mean_cosine"] - null_residual["mean_cosine"],
            }
        results[stage] = stage_results
    payload = {
        "status": "development_only",
        "train_task_indices": list(range(6)),
        "test_task_indices": [6, 7],
        "record_count": len(records),
        "null_dimension": len(records[0]["endpoint_null"]),
        "internal_dimension": len(records[0]["endpoint_internal"]),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
