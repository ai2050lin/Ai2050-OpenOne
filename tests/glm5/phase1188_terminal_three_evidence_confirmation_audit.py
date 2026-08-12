#!/usr/bin/env python3
"""Independent audit for Phase1188's terminal three-evidence decision."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
import phase1188_terminal_three_evidence_confirmation as p  # noqa: E402


AUDIT_PATH = p.AUDIT_PATH
TOLERANCE = 2e-6
FEATURE_TOLERANCE = 2e-10
SCORE_TOLERANCE = 2e-6


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def maximum_error(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return math.inf
    return float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0


def independent_panel(payload: dict[str, Any]) -> p1181.DataPanel:
    operation = tuple(int(value) for value in payload["operation"])
    data = p1171.make_data(operation, int(payload["seed"]) + 17)
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    train_mask[: len(data["train_x"])] = True
    return p1181.DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def fp32_state(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = x.to(device)
    left = F.embedding(ids[:, 0], model.left_embedding.weight.float())
    right = F.embedding(ids[:, 1], model.right_embedding.weight.float())
    hidden = F.linear(left + right, model.hidden.weight.float())
    return F.linear(hidden.square(), model.output.weight.float()), hidden


def correct_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], -torch.inf)
    return correct - masked.max(dim=1).values


@torch.inference_mode()
def behavior_metrics(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> dict[str, float]:
    targets = panel.y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(panel.x.to(device)).float()
    predictions = logits.argmax(dim=1)
    margins = correct_margin(logits, targets)
    result: dict[str, float] = {}
    for name, mask in (("train", panel.train_mask), ("holdout", panel.holdout_mask)):
        selected = mask.to(device)
        result[f"{name}_accuracy"] = float((predictions[selected] == targets[selected]).float().mean().item())
        result[f"{name}_loss"] = float(F.cross_entropy(logits[selected], targets[selected]).item())
        result[f"{name}_mean_margin"] = float(margins[selected].mean().item())
    result["parameter_norm"] = math.sqrt(
        sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters())
    )
    result["all_logits_finite"] = float(torch.isfinite(logits).all().item())
    return result


@torch.inference_mode()
def response_spectrum(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> dict[str, Any]:
    logits, hidden = fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    selected = panel.holdout_mask.to(device)
    baseline = correct_margin(logits, targets)
    squared = hidden.square()
    weight = model.output.weight.detach().float()
    responses: list[float] = []
    for channel in range(model.config.width):
        changed = logits - squared[:, channel, None] * weight[:, channel][None, :]
        responses.append(float((baseline[selected] - correct_margin(changed, targets)[selected]).mean().item()))
    ordered = np.sort(np.asarray(responses, dtype=np.float64))
    centered = ordered - ordered.mean()
    unit_shape = centered / max(float(np.linalg.norm(centered)), 1e-12)
    return {"ordered": ordered.tolist(), "unit_shape": unit_shape.tolist()}


def canonical_sum(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return float(math.fsum(float(value) for value in ordered))


def stable_moment(values: np.ndarray, power: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    return canonical_sum(np.power(array, power)) / max(array.size, 1)


def scalar_moments(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    scale = math.sqrt(max(stable_moment(array, 2), 1e-30))
    normalized = array / scale
    return [float(np.log1p(scale)), *(stable_moment(normalized, power) for power in (1, 2, 3, 4))]


@torch.inference_mode()
def output_features(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> list[float]:
    logits, _ = fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    margins = correct_margin(logits, targets)
    probabilities = torch.softmax(logits, dim=1)
    arrays = (
        margins,
        probabilities.max(dim=1).values,
        probabilities.gather(1, targets[:, None]).squeeze(1),
        logits.norm(dim=1),
    )
    features: list[float] = []
    for mask in (panel.train_mask, panel.holdout_mask):
        selected = mask.to(device)
        features.extend(
            (
                float((logits[selected].argmax(dim=1) == targets[selected]).float().mean().item()),
                float(F.cross_entropy(logits[selected], targets[selected]).item()),
            )
        )
        for array in arrays:
            features.extend(scalar_moments(array[selected].cpu().numpy()))
    features.extend(float(parameter.detach().float().norm().item()) for parameter in model.parameters())
    return features


def generator_arrays(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> tuple[np.ndarray, ...]:
    left = model.left_embedding.weight.detach().cpu().double().numpy()
    right = model.right_embedding.weight.detach().cpu().double().numpy()
    hidden_weight = model.hidden.weight.detach().cpu().double().numpy()
    output_weight = model.output.weight.detach().cpu().double().numpy()
    ids = x.cpu().numpy()
    left_projection = left @ hidden_weight.T
    right_projection = right @ hidden_weight.T
    hidden = (left[ids[:, 0]] + right[ids[:, 1]]) @ hidden_weight.T
    return left_projection, right_projection, hidden_weight, output_weight, hidden


def channel_matrix(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> np.ndarray:
    left_projection, right_projection, hidden_weight, output_weight, hidden = generator_arrays(model, x)

    def column_moment(matrix: np.ndarray, power: int) -> np.ndarray:
        powered = np.power(matrix, power)
        return np.asarray(
            [canonical_sum(powered[:, channel]) / powered.shape[0] for channel in range(powered.shape[1])],
            dtype=np.float64,
        )

    return np.stack(
        (
            np.sum(hidden_weight * hidden_weight, axis=1),
            np.sum(output_weight * output_weight, axis=0),
            np.sum(left_projection * left_projection, axis=0),
            np.sum(right_projection * right_projection, axis=0),
            column_moment(hidden, 2),
            column_moment(hidden, 4),
            column_moment(output_weight, 4),
            column_moment(left_projection * right_projection, 1),
        ),
        axis=1,
    )


def algebraic_internal_features(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> list[float]:
    channels = channel_matrix(model, x)
    scales = np.sqrt(
        np.asarray([stable_moment(channels[:, index], 2) for index in range(channels.shape[1])])
    )
    normalized = channels / np.maximum(scales[None, :], 1e-30)
    features: list[float] = []
    for index in range(channels.shape[1]):
        features.append(float(np.log1p(scales[index])))
        features.extend(stable_moment(normalized[:, index], power) for power in (1, 2, 3, 4))
    for left in range(channels.shape[1]):
        for right in range(left + 1, channels.shape[1]):
            features.append(stable_moment(normalized[:, left] * normalized[:, right], 1))
            features.append(stable_moment(normalized[:, left] ** 2 * normalized[:, right], 1))
    return features


def canonical_channel_signature(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> list[float]:
    channels = channel_matrix(model, x)
    scales = np.sqrt(np.mean(channels * channels, axis=0))
    normalized = channels / np.maximum(scales[None, :], 1e-30)
    order = np.lexsort(tuple(normalized[:, index] for index in reversed(range(normalized.shape[1]))))
    return np.concatenate((np.log1p(scales), normalized[order].reshape(-1))).tolist()


def trajectory_summary(rows: list[list[float]]) -> list[float]:
    matrix = np.asarray(rows, dtype=np.float64)
    time = np.log(np.asarray(p.HISTORY_STEPS, dtype=np.float64))
    time = (time - time.mean()) / time.std()
    slope = (time[:, None] * matrix).sum(axis=0) / float((time * time).sum())
    return np.concatenate((matrix[0], matrix[-1], matrix[-1] - matrix[0], slope)).tolist()


def reconstruct_record(endpoint: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    panel = independent_panel(payload)
    model = load_model(payload, device)
    behavior = behavior_metrics(model, panel, device)
    response = response_spectrum(model, panel, device)
    output_rows: list[list[float]] = []
    internal_rows: list[list[float]] = []
    for step in p.HISTORY_STEPS:
        history_payload = torch.load(p.history_path(endpoint, step), map_location="cpu", weights_only=False)
        history_model = load_model(history_payload, device)
        output_rows.append(output_features(history_model, panel, device))
        internal_rows.append(canonical_channel_signature(history_model, panel.x))
        del history_model
    prefix_null = trajectory_summary(output_rows)
    result = {
        "checkpoint": endpoint.name,
        "task_name": str(payload["task_name"]),
        "replicate": int(payload["replicate"]),
        "behavior": behavior,
        "target": response["unit_shape"],
        "ordered": response["ordered"],
        "endpoint_null": output_features(model, panel, device) + prefix_null,
        "endpoint_internal": algebraic_internal_features(model, panel.x),
        "prefix_null": prefix_null,
        "prefix_internal": trajectory_summary(internal_rows),
    }
    del model
    torch.cuda.empty_cache()
    return result


def fit_dual_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    standardized = (x - mean) / scale
    y_mean = y.mean(axis=0)
    alpha = np.linalg.solve(
        standardized @ standardized.T + l2 * np.eye(len(standardized)),
        y - y_mean,
    )
    return {"mean": mean, "scale": scale, "train_z": standardized, "alpha": alpha, "y_mean": y_mean}


def predict_dual(camera: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (x - camera["mean"]) / camera["scale"]
    return standardized @ camera["train_z"].T @ camera["alpha"] + camera["y_mean"]


def residual_metrics(prediction: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    prediction_residual = np.zeros_like(prediction)
    target_residual = np.zeros_like(target)
    for group in np.unique(groups):
        selected = groups == group
        prediction_residual[selected] = prediction[selected] - prediction[selected].mean(axis=0)
        target_residual[selected] = target[selected] - target[selected].mean(axis=0)
    prediction_residual /= np.maximum(np.linalg.norm(prediction_residual, axis=1, keepdims=True), 1e-12)
    target_residual /= np.maximum(np.linalg.norm(target_residual, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(prediction_residual * target_residual, axis=1)
    squared_error = np.mean((prediction_residual - target_residual) ** 2, axis=1)
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "mean_squared_error": float(squared_error.mean()),
    }


def fit_stage(rows: list[dict[str, Any]], stage: str, l2: float) -> dict[str, dict[str, np.ndarray]]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    null = np.asarray([row[f"{stage}_null"] for row in rows], dtype=np.float64)
    internal = np.asarray([row[f"{stage}_internal"] for row in rows], dtype=np.float64)
    return {
        "null": fit_dual_ridge(null, target, l2),
        "joint": fit_dual_ridge(np.concatenate((null, internal), axis=1), target, l2),
    }


def score_stage(
    rows: list[dict[str, Any]],
    stage: str,
    cameras: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["task_name"] for row in rows])
    null = np.asarray([row[f"{stage}_null"] for row in rows], dtype=np.float64)
    internal = np.asarray([row[f"{stage}_internal"] for row in rows], dtype=np.float64)
    null_score = residual_metrics(predict_dual(cameras["null"], null), target, groups)
    joint_score = residual_metrics(
        predict_dual(cameras["joint"], np.concatenate((null, internal), axis=1)), target, groups
    )
    return {
        "system_count": len(rows),
        "task_count": len(np.unique(groups)),
        "null": null_score,
        "joint": joint_score,
        "residual_cosine_improvement": joint_score["mean_cosine"] - null_score["mean_cosine"],
        "residual_risk_improvement": null_score["mean_squared_error"] - joint_score["mean_squared_error"],
    }


def load_camera() -> dict[str, Any]:
    arrays = np.load(p.DEVELOPMENT_NPZ)
    return {
        stage: {
            label: {
                key: arrays[f"{stage}__{label}__{key}"]
                for key in ("mean", "scale", "train_z", "alpha", "y_mean")
            }
            for label in ("null", "joint")
        }
        for stage in ("endpoint", "prefix")
    }


def numeric_score_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    values: list[float] = []
    for key, value in left.items():
        if isinstance(value, dict):
            values.append(numeric_score_error(value, right[key]))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(abs(float(value) - float(right[key])))
    return max(values, default=0.0)


def bounded(value: float, threshold: float, comparator: str) -> bool:
    if not (math.isfinite(float(value)) and math.isfinite(float(threshold))):
        return False
    return {
        "<=": value <= threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        ">": value > threshold,
        "==": value == threshold,
    }[comparator]


def universal(source: dict[str, Any]) -> bool:
    return bool(
        int(source["eligible_count"]) > 0
        and not bool(source["abstained"])
        and int(source["agree_count"]) == int(source["eligible_count"])
    )


def numerical_expected(case: dict[str, Any]) -> bool:
    fp32 = case["fp32"]
    fp64 = case["fp64"]
    decision = case["exact_decision"]
    values = [
        bounded(case["feature_error"], p.THRESHOLDS["algebraic_feature_error_max"], "<="),
        bounded(fp64["absolute_max"], fp64["mixed_absolute_bound"], "<="),
        bounded(fp64["scaled_max"], p.p1185.THRESHOLDS["fp64_scaled_error_max"], "<="),
        bounded(fp64["rms_relative"], p.p1185.THRESHOLDS["fp64_scaled_error_max"], "<="),
        bounded(fp32["absolute_max"], fp32["mixed_absolute_bound"], "<="),
        bounded(fp32["scaled_max"], p.p1185.THRESHOLDS["fp32_scaled_error_max"], "<="),
        bounded(fp32["rms_relative"], p.p1185.THRESHOLDS["fp32_scaled_error_max"], "<="),
        universal(decision["decision"]),
        universal(decision["margin_sign"]),
    ]
    return all(values)


def independent_rescue_summary(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for task in task_results for row in task["rows"]]

    def average(rows: list[dict[str, Any]], condition: str, metric: str) -> float:
        return float(np.mean([row[condition][metric] for row in rows]))

    task_summaries: dict[str, Any] = {}
    positive_task_count = 0
    for task in task_results:
        rows = task["rows"]
        advantage = average(rows, "wrong", "future_response_error") - average(
            rows, "correct", "future_response_error"
        )
        positive_fraction = float(
            np.mean(
                [row["correct"]["future_response_error"] < row["wrong"]["future_response_error"] for row in rows]
            )
        )
        positive_task_count += int(advantage > 0)
        task_summaries[task["task_name"]] = {
            "recipient_count": len(rows),
            "baseline_accuracy": average(rows, "baseline", "accuracy"),
            "injured_accuracy": average(rows, "injured", "accuracy"),
            "correct_accuracy": average(rows, "correct", "accuracy"),
            "wrong_accuracy": average(rows, "wrong", "accuracy"),
            "correct_future_error": average(rows, "correct", "future_response_error"),
            "wrong_future_error": average(rows, "wrong", "future_response_error"),
            "future_response_error_advantage": advantage,
            "recipient_positive_future_advantage_fraction": positive_fraction,
        }
    baseline = average(all_rows, "baseline", "accuracy")
    injured = average(all_rows, "injured", "accuracy")
    correct = average(all_rows, "correct", "accuracy")
    wrong = average(all_rows, "wrong", "accuracy")
    future_advantage = average(all_rows, "wrong", "future_response_error") - average(
        all_rows, "correct", "future_response_error"
    )
    positive_fraction = float(
        np.mean(
            [row["correct"]["future_response_error"] < row["wrong"]["future_response_error"] for row in all_rows]
        )
    )
    gate = bool(
        baseline - injured >= p.THRESHOLDS["injury_accuracy_drop_min"]
        and abs(correct - baseline) <= p.THRESHOLDS["correct_rescue_accuracy_gap_from_baseline_max"]
        and abs(wrong - baseline) <= p.THRESHOLDS["wrong_rescue_accuracy_gap_from_baseline_max"]
        and abs(correct - wrong) <= p.THRESHOLDS["correct_wrong_accuracy_difference_max"]
        and future_advantage >= p.THRESHOLDS["future_response_error_advantage_min"]
        and positive_fraction >= p.THRESHOLDS["recipient_positive_future_advantage_fraction_min"]
        and positive_task_count >= p.THRESHOLDS["confirmation_positive_task_count_min"]
    )
    return {
        "recipient_count": len(all_rows),
        "task_count": len(task_results),
        "baseline_accuracy": baseline,
        "injured_accuracy": injured,
        "correct_accuracy": correct,
        "wrong_accuracy": wrong,
        "correct_wrong_accuracy_difference": abs(correct - wrong),
        "correct_future_error": average(all_rows, "correct", "future_response_error"),
        "wrong_future_error": average(all_rows, "wrong", "future_response_error"),
        "future_response_error_advantage": future_advantage,
        "recipient_positive_future_advantage_fraction": positive_fraction,
        "positive_task_count": positive_task_count,
        "required_positive_task_count": p.THRESHOLDS["confirmation_positive_task_count_min"],
        "task_summaries": task_summaries,
        "gate_pass": gate,
    }


def independent_future_masks(task_name: str) -> list[np.ndarray]:
    seed = int(hashlib.sha256(("phase1188:" + task_name).encode("utf-8")).hexdigest()[:16], 16)
    generator = np.random.default_rng(seed)
    return [
        np.sort(generator.choice(p.WIDTH, size=p.FUTURE_MASK_SIZE, replace=False))
        for _ in range(p.FUTURE_MASK_COUNT)
    ]


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
    base_margin = correct_margin(logits, targets)
    response: list[float] = []
    for channels in masks:
        index = torch.tensor(channels, dtype=torch.long, device=device)
        changed = logits - q[:, index] @ weight[:, index].T
        response.append(float((base_margin - correct_margin(changed, targets)).mean().item()))
    return {
        "accuracy": float((logits.argmax(dim=1) == targets).float().mean().item()),
        "mean_margin": float(base_margin.mean().item()),
        "future_response": response,
    }


def response_error(candidate: list[float], reference: list[float]) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.linalg.norm(reference_array - reference_array.mean())), 1e-8)
    return float(np.linalg.norm(candidate_array - reference_array) / scale)


@torch.inference_mode()
def rescue_bundle(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> dict[str, Any]:
    logits, hidden = fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    indices = torch.where(panel.holdout_mask)[0]
    code = panel.x[indices, 0] * 131 + panel.x[indices, 1] * 17
    calibration_mask = torch.zeros_like(panel.holdout_mask)
    calibration_mask[indices[code % 2 == 0]] = True
    evaluation_mask = panel.holdout_mask & ~calibration_mask
    calibration = calibration_mask.to(device)
    evaluation = evaluation_mask.to(device)
    q = hidden.square()
    weight = model.output.weight.detach().float()
    base_margin = correct_margin(logits, targets)
    response: list[float] = []
    for channel in range(model.config.width):
        changed = logits - q[:, channel, None] * weight[:, channel][None, :]
        response.append(float((base_margin[calibration] - correct_margin(changed, targets)[calibration]).mean().item()))
    behavior = behavior_metrics(model, panel, device)
    return {
        "q_evaluation": q[evaluation].cpu(),
        "weight": weight.cpu(),
        "targets_evaluation": targets[evaluation].cpu(),
        "evaluation_mask": evaluation_mask,
        "calibration_response": np.asarray(response, dtype=np.float64),
        "behavior_vector": np.asarray([behavior[name] for name in p1181.BEHAVIOR_FEATURES], dtype=np.float64),
    }


def replay_first_rescue(
    task_name: str,
    records: list[dict[str, Any]],
    endpoint_map: dict[str, Path],
    stored_task: dict[str, Any],
    device: torch.device,
) -> float:
    bundles: list[dict[str, Any]] = []
    for record in records:
        payload = torch.load(endpoint_map[record["checkpoint"]], map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        panel = independent_panel(payload)
        bundle = rescue_bundle(model, panel, device)
        bundle.update({"record": record, "model": model, "panel": panel})
        bundles.append(bundle)
    behavior = np.stack([bundle["behavior_vector"] for bundle in bundles])
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    recipient_index = 0
    recipient = bundles[recipient_index]
    candidates = [index for index in range(len(bundles)) if index != recipient_index]
    candidates.sort(key=lambda index: float(np.linalg.norm(behavior[recipient_index] - behavior[index])))
    pool = candidates[:4]
    recipient_ordered = np.sort(recipient["calibration_response"])
    distances = {
        index: float(np.linalg.norm(np.sort(bundles[index]["calibration_response"]) - recipient_ordered))
        for index in pool
    }
    correct_index = min(pool, key=lambda index: distances[index])
    wrong_index = max(pool, key=lambda index: distances[index])
    recipient_order = np.argsort(recipient["calibration_response"])
    recipient_rank = np.empty(len(recipient_order), dtype=np.int64)
    recipient_rank[recipient_order] = np.arange(len(recipient_order))
    injured_channels = np.argsort(np.abs(recipient["calibration_response"]))[-p.INJURY_CHANNEL_COUNT :]
    masks = independent_future_masks(task_name)
    baseline = evaluate_hybrid(
        recipient["q_evaluation"], recipient["weight"], recipient["targets_evaluation"], masks, device
    )
    injured_q = recipient["q_evaluation"].clone()
    injured_q[:, injured_channels] = 0.0
    injured = evaluate_hybrid(injured_q, recipient["weight"], recipient["targets_evaluation"], masks, device)
    injured["future_response_error"] = response_error(injured["future_response"], baseline["future_response"])
    computed: dict[str, Any] = {"baseline": baseline, "injured": injured}
    for label, donor_index in (("correct", correct_index), ("wrong", wrong_index)):
        donor = bundles[donor_index]
        _, donor_hidden = fp32_state(donor["model"], recipient["panel"].x, device)
        donor_q = donor_hidden.square()[recipient["evaluation_mask"].to(device)].cpu()
        donor_order = np.argsort(donor["calibration_response"])
        hybrid_q = recipient["q_evaluation"].clone()
        hybrid_weight = recipient["weight"].clone()
        for recipient_channel in injured_channels:
            donor_channel = donor_order[recipient_rank[recipient_channel]]
            hybrid_q[:, recipient_channel] = donor_q[:, donor_channel]
            hybrid_weight[:, recipient_channel] = donor["weight"][:, donor_channel]
        evaluated = evaluate_hybrid(
            hybrid_q, hybrid_weight, recipient["targets_evaluation"], masks, device
        )
        evaluated["future_response_error"] = response_error(
            evaluated["future_response"], baseline["future_response"]
        )
        evaluated["donor_replicate"] = donor["record"]["replicate"]
        evaluated["calibration_response_distance"] = distances[donor_index]
        computed[label] = evaluated
    stored = next(
        row for row in stored_task["rows"] if int(row["recipient_replicate"]) == int(recipient["record"]["replicate"])
    )
    errors: list[float] = []
    for condition in ("baseline", "injured", "correct", "wrong"):
        for key, value in computed[condition].items():
            if isinstance(value, list):
                errors.append(maximum_error(value, stored[condition][key]))
            elif isinstance(value, (int, float)):
                errors.append(abs(float(value) - float(stored[condition][key])))
    for bundle in bundles:
        del bundle["model"]
    torch.cuda.empty_cache()
    return max(errors, default=0.0)


def main() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("Phase1188 independent audit already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checks: list[dict[str, Any]] = []
    protocol = read_json(p.PROTOCOL_PATH)
    summary = read_json(p.SUMMARY_PATH)
    seal = read_json(p.TRAINING_SEAL)
    systems = read_jsonl(p.SYSTEM_ROWS)
    positives = read_jsonl(p.POSITIVE_ROWS)
    rescue_raw = read_json(p.RESCUE_RAW)
    development = read_json(p.DEVELOPMENT_META)
    device = torch.device("cuda")

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    add_check(checks, "protocol_digest", digest(protocol_copy) == protocol_digest)
    add_check(checks, "source_hashes", p.source_hashes() == protocol["source_hashes"])
    development_copy = dict(development)
    metadata_digest = development_copy.pop("metadata_digest")
    add_check(checks, "development_metadata_digest", digest(development_copy) == metadata_digest)
    add_check(checks, "development_npz_hash", file_sha256(p.DEVELOPMENT_NPZ) == development["npz_sha256"])
    phase1187 = read_json(p.P1187_AUDIT)
    add_check(checks, "phase1187_authorization", bool(phase1187["phase1188_authorized_after_audit"]))
    add_check(
        checks,
        "task_operations",
        tuple(tuple(value) for value in protocol["operations"]) == p.OPERATIONS
        and not (set(p.OPERATIONS) & (set(p1171.OPERATION_SAMPLE) | {p1171.PILOT_OPERATION, (19, 23, 7), (29, 31, 11)})),
    )
    allocation = protocol["allocation"]
    add_check(checks, "allocation_cardinality", len(allocation) == p.TASK_COUNT * p.REPLICATES)
    allocation_valid = True
    for row in allocation:
        data = p1171.make_data(tuple(row["operation"]), int(row["seed"]) + 17)
        allocation_valid &= row["train_pair_digest"] == digest(data["train_x"].tolist())
        allocation_valid &= row["sealed_holdout_pair_digest"] == digest(data["holdout_x"].tolist())
        allocation_valid &= row["train_label_digest"] == digest(data["train_y"].tolist())
        allocation_valid &= row["sealed_holdout_label_digest"] == digest(data["holdout_y"].tolist())
    add_check(checks, "allocation_recomputation", allocation_valid)

    seal_copy = dict(seal)
    seal_digest = seal_copy.pop("seal_digest")
    add_check(checks, "training_seal_digest", digest(seal_copy) == seal_digest)
    add_check(checks, "training_protocol_binding", seal["protocol_digest"] == protocol_digest)
    training_rows = read_jsonl(p.TRAINING_METRICS)
    add_check(checks, "training_metrics_hash", file_sha256(p.TRAINING_METRICS) == seal["training_metrics_sha256"])
    add_check(checks, "training_checkpoint_count", len(training_rows) == 384 == len(seal["checkpoint_hashes"]))
    add_check(
        checks,
        "training_holdout_firewall",
        bool(seal["holdout_outcomes_absent_at_sealing"] and seal["no_holdout_evaluated"] and seal["no_holdout_gradient"]),
    )
    checkpoint_hashes_valid = all(
        file_sha256(p.CHECKPOINT_ROOT / name) == expected for name, expected in seal["checkpoint_hashes"].items()
    )
    add_check(checks, "checkpoint_hashes", checkpoint_hashes_valid)
    add_check(checks, "systems_hash", file_sha256(p.SYSTEM_ROWS) == summary["rows_sha256"])
    add_check(checks, "positive_hash", file_sha256(p.POSITIVE_ROWS) == summary["positive_rows_sha256"])
    add_check(checks, "rescue_hash", file_sha256(p.RESCUE_RAW) == summary["rescue_raw_sha256"])
    add_check(checks, "system_cardinality", len(systems) == 64 and len({row["checkpoint"] for row in systems}) == 64)

    # Refit and verify the frozen development camera from all 64 historical systems.
    development_paths = sorted((p1171.OUT_ROOT / "runs/training/checkpoints").glob("*step10000.pt"))
    development_rows = [reconstruct_record(path, device) for path in development_paths]
    fit_names = set(development["fit_task_names"])
    test_names = set(development["development_holdout_task_names"])
    fit_rows = [row for row in development_rows if row["task_name"] in fit_names]
    test_rows = [row for row in development_rows if row["task_name"] in test_names]
    provisional = {
        "endpoint": fit_stage(fit_rows, "endpoint", p.ENDPOINT_L2),
        "prefix": fit_stage(fit_rows, "prefix", p.PREFIX_L2),
    }
    development_endpoint = score_stage(test_rows, "endpoint", provisional["endpoint"])
    development_prefix = score_stage(test_rows, "prefix", provisional["prefix"])
    add_check(
        checks,
        "development_endpoint_score",
        numeric_score_error(development_endpoint, development["endpoint_development_score"]) <= SCORE_TOLERANCE,
        numeric_score_error(development_endpoint, development["endpoint_development_score"]),
    )
    add_check(
        checks,
        "development_prefix_score",
        numeric_score_error(development_prefix, development["prefix_development_score"]) <= SCORE_TOLERANCE,
        numeric_score_error(development_prefix, development["prefix_development_score"]),
    )
    independent_camera = {
        "endpoint": fit_stage(development_rows, "endpoint", p.ENDPOINT_L2),
        "prefix": fit_stage(development_rows, "prefix", p.PREFIX_L2),
    }
    sealed_camera = load_camera()
    camera_array_error = max(
        maximum_error(independent_camera[stage][label][key], sealed_camera[stage][label][key])
        for stage in ("endpoint", "prefix")
        for label in ("null", "joint")
        for key in ("mean", "scale", "train_z", "alpha", "y_mean")
    )
    add_check(checks, "frozen_camera_reconstruction", camera_array_error <= SCORE_TOLERANCE, camera_array_error)

    endpoint_map = {path.name: path for path in p.CHECKPOINT_ROOT.glob("*step10000.pt")}
    stored_by_name = {row["checkpoint"]: row for row in systems}
    reconstructed: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    behavior_errors: list[float] = []
    response_errors: list[float] = []
    for endpoint in sorted(endpoint_map.values()):
        audit_row = reconstruct_record(endpoint, device)
        stored = stored_by_name[endpoint.name]
        reconstructed.append(audit_row)
        behavior_errors.append(
            max(abs(float(audit_row["behavior"][key]) - float(stored["behavior"][key])) for key in audit_row["behavior"])
        )
        response_errors.append(maximum_error(audit_row["target"], stored["target"]))
        reconstruction_errors.extend(
            maximum_error(audit_row[key], stored[key])
            for key in ("endpoint_null", "endpoint_internal", "prefix_null", "prefix_internal")
        )
    add_check(checks, "all_behavior_recomputed", max(behavior_errors) <= TOLERANCE, max(behavior_errors))
    add_check(checks, "all_responses_recomputed", max(response_errors) <= TOLERANCE, max(response_errors))
    add_check(checks, "all_camera_inputs_recomputed", max(reconstruction_errors) <= FEATURE_TOLERANCE, max(reconstruction_errors))

    science_names = {
        row["checkpoint"]
        for row in systems
        if row["task_name"] in summary["behavior"]["passing_tasks"]
        and row["behavior"]["all_logits_finite"] == 1.0
        and row["behavior"]["train_accuracy"] >= p.THRESHOLDS["train_accuracy_min"]
        and row["behavior"]["holdout_accuracy"] >= p.THRESHOLDS["holdout_accuracy_min"]
        and row["numerical"]["typed_gate"]["authorizes"]
    }
    science_rows = [row for row in reconstructed if row["checkpoint"] in science_names]
    endpoint_score = score_stage(science_rows, "endpoint", sealed_camera["endpoint"])
    prefix_score = score_stage(science_rows, "prefix", sealed_camera["prefix"])
    add_check(
        checks,
        "endpoint_score_recomputed",
        numeric_score_error(endpoint_score, summary["endpoint"]) <= SCORE_TOLERANCE,
        numeric_score_error(endpoint_score, summary["endpoint"]),
    )
    add_check(
        checks,
        "prefix_score_recomputed",
        numeric_score_error(prefix_score, summary["prefix"]) <= SCORE_TOLERANCE,
        numeric_score_error(prefix_score, summary["prefix"]),
    )
    add_check(checks, "endpoint_gate_recomputed", p.camera_gate("endpoint", endpoint_score) == summary["endpoint"]["gate_pass"])
    add_check(checks, "prefix_gate_recomputed", p.camera_gate("prefix", prefix_score) == summary["prefix"]["gate_pass"])

    numerical_valid = True
    for row in systems:
        endpoint_expected = numerical_expected(row["numerical"]["endpoint"])
        prefix_expected = numerical_expected(row["numerical"]["prefix"])
        numerical_valid &= endpoint_expected == bool(row["numerical"]["endpoint"]["typed_gate"]["authorizes"])
        numerical_valid &= prefix_expected == bool(row["numerical"]["prefix"]["typed_gate"]["authorizes"])
        numerical_valid &= (endpoint_expected and prefix_expected) == bool(row["numerical"]["typed_gate"]["authorizes"])
    add_check(checks, "all_typed_numerical_claims_recompiled", numerical_valid)
    add_check(
        checks,
        "numerical_summary_recomputed",
        sum(bool(row["numerical"]["typed_gate"]["authorizes"]) for row in systems)
        == summary["numerical"]["typed_system_pass_count"],
    )
    replay_errors: list[float] = []
    for task_name in sorted(p.TASKS):
        row = next(row for row in systems if row["task_name"] == task_name)
        case_index = sorted(endpoint_map).index(row["checkpoint"])
        payload = torch.load(endpoint_map[row["checkpoint"]], map_location="cpu", weights_only=False)
        panel = independent_panel(payload)
        endpoint_model = load_model(payload, device)
        prefix_payload = torch.load(p.history_path(endpoint_map[row["checkpoint"]], p.HISTORY_STEPS[-1]), map_location="cpu", weights_only=False)
        prefix_model = load_model(prefix_payload, device)
        endpoint_replay = p.numerical_transform_case(
            endpoint_model, panel, 11885000 + case_index * 2, "endpoint_algebraic", device
        )
        prefix_replay = p.numerical_transform_case(
            prefix_model, panel, 11885001 + case_index * 2, "prefix_canonical", device
        )
        replay_errors.extend(
            (
                abs(endpoint_replay["feature_error"] - row["numerical"]["endpoint"]["feature_error"]),
                abs(prefix_replay["feature_error"] - row["numerical"]["prefix"]["feature_error"]),
                abs(endpoint_replay["fp32"]["scaled_max"] - row["numerical"]["endpoint"]["fp32"]["scaled_max"]),
                abs(prefix_replay["fp32"]["scaled_max"] - row["numerical"]["prefix"]["fp32"]["scaled_max"]),
            )
        )
        del endpoint_model, prefix_model
    add_check(checks, "numerical_replay_one_per_task", max(replay_errors) <= TOLERANCE, max(replay_errors))

    positive_valid = all(
        bounded(row["strength"], p.THRESHOLDS["positive_sentinel_error_min"], ">=")
        == bool(row["typed_gate"]["authorizes"])
        for row in positives
    )
    add_check(checks, "positive_sentinels_recompiled", positive_valid and len(positives) == 8)
    positive_replay_errors: list[float] = []
    first_by_task: dict[str, Path] = {}
    for endpoint in sorted(endpoint_map.values()):
        payload = torch.load(endpoint, map_location="cpu", weights_only=False)
        first_by_task.setdefault(str(payload["task_name"]), endpoint)
    positive_by_task = {row["task_name"]: row for row in positives}
    for case_index, (task_name, endpoint) in enumerate(sorted(first_by_task.items())):
        replay = p.positive_sentinel(endpoint, case_index, device)
        positive_replay_errors.append(abs(replay["strength"] - positive_by_task[task_name]["strength"]))
    add_check(checks, "positive_sentinel_replay", max(positive_replay_errors) <= TOLERANCE, max(positive_replay_errors))

    rescue_summary = independent_rescue_summary(rescue_raw["tasks"])
    rescue_score_error = numeric_score_error(rescue_summary, summary["rescue"])
    add_check(checks, "rescue_summary_recomputed", rescue_score_error <= TOLERANCE, rescue_score_error)
    stored_rescue_by_task = {task["task_name"]: task for task in rescue_raw["tasks"]}
    rescue_replay_errors: list[float] = []
    for task_name in summary["behavior"]["passing_tasks"]:
        task_records = [
            row for row in systems if row["task_name"] == task_name and row["checkpoint"] in science_names
        ]
        rescue_replay_errors.append(
            replay_first_rescue(
                task_name, task_records, endpoint_map, stored_rescue_by_task[task_name], device
            )
        )
    add_check(checks, "rescue_replay_one_recipient_per_task", max(rescue_replay_errors) <= TOLERANCE, max(rescue_replay_errors))

    summary_copy = dict(summary)
    summary_digest = summary_copy.pop("summary_digest")
    add_check(checks, "summary_digest", digest(summary_copy) == summary_digest)
    components = (
        bool(summary["behavior"]["gate_pass"]),
        bool(summary["numerical"]["gate_pass"]),
        bool(summary["positive_sentinel"]["gate_pass"]),
        bool(summary["endpoint"]["gate_pass"]),
        bool(summary["prefix"]["gate_pass"]),
        bool(summary["rescue"]["gate_pass"]),
    )
    add_check(checks, "main_conjunction", all(components) == bool(summary["main_pass_before_audit"]), components)

    audit_pass = all(check["pass"] for check in checks)
    main_pass = bool(summary["main_pass_before_audit"])
    result = {
        "phase": p.PHASE,
        "protocol_digest": protocol_digest,
        "summary_digest": summary_digest,
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "audit_pass": audit_pass,
        "main_pass": main_pass,
        "joint_pass": bool(main_pass and audit_pass),
        "scientific_status_after_audit": (
            "narrow_three_evidence_mechanism_confirmed"
            if main_pass and audit_pass
            else "terminal_three_evidence_confirmation_failed_or_unaudited"
        ),
        "k165_status_after_audit": (
            "E3_KT_narrow_RoleSquare_affine_family"
            if main_pass and audit_pass
            else "E1_KT_discovery_candidate_not_confirmed"
        ),
        "phase1189_authorized_after_audit": bool(main_pass and audit_pass),
        "registry": "closed_after_one_formal_decision",
    }
    result["audit_digest"] = digest(result)
    write_json(AUDIT_PATH, result)
    print(canonical_json(result))
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
