#!/usr/bin/env python3
"""Phase1182: quotient response camera and sealed donor rescue.

Phase1181 established that freely trained RoleSquareNetwork endpoints contain
non-degenerate causal-response material after quotienting signed channel
permutations.  Phase1182 asks three separate questions:

1. Do passive endpoint invariants improve response-residual prediction beyond
   a strong behavior/output/gradient/history null?
2. Do fixed early trajectory invariants predict the endpoint residual before
   endpoint behavior is observed?
3. Can a calibration-response-matched donor restore a sealed future response
   spectrum better than a behavior-matched but response-distant donor, while
   both donors restore natural behavior?

The three gates are reported separately and the primary claim requires all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402


PHASE = 1182
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1182_quotient_response_camera_and_rescue_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1182_quotient_response_camera_and_rescue"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
DISCOVERY_ROWS = OUT_ROOT / "runs/discovery/systems.jsonl"
DISCOVERY_RESCUE = OUT_ROOT / "runs/discovery/rescue.json"
DISCOVERY_SUMMARY = OUT_ROOT / "runs/discovery/summary.json"
CAMERA_SEAL = OUT_ROOT / "analysis/camera_seal.npz"
CAMERA_METADATA = OUT_ROOT / "analysis/camera_seal.json"
CONFIRMATION_ROWS = OUT_ROOT / "runs/confirmation/systems.jsonl"
CONFIRMATION_RESCUE = OUT_ROOT / "runs/confirmation/rescue.json"
CONFIRMATION_SUMMARY = OUT_ROOT / "runs/confirmation/summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

P1171_CHECKPOINTS = p1171.OUT_ROOT / "runs/training/checkpoints"
P1172_CHECKPOINTS = p1172.OUT_ROOT / "runs/training/checkpoints"
HISTORY_STEPS = (25, 50, 75, 100, 150)
QUANTILES = np.linspace(0.0, 1.0, 17)
ENDPOINT_L2 = 100.0
PREFIX_L2 = 1.0
INJURY_CHANNEL_COUNT = 32
FUTURE_MASK_COUNT = 32
FUTURE_MASK_SIZE = 8

THRESHOLDS = {
    "train_accuracy_min": 0.95,
    "holdout_accuracy_min": 0.90,
    "qualified_system_count_per_task_min": 6,
    "qualified_task_count_confirmation_min": 4,
    "feature_gauge_max_error_max": 1e-5,
    "endpoint_joint_residual_cosine_min": 0.35,
    "endpoint_residual_cosine_improvement_min": 0.02,
    "endpoint_residual_risk_improvement_min": 0.0,
    "prefix_joint_residual_cosine_min": 0.15,
    "prefix_residual_cosine_improvement_min": 0.04,
    "prefix_residual_risk_improvement_min": 0.0,
    "injury_accuracy_drop_min": 0.50,
    "correct_rescue_accuracy_gap_from_baseline_max": 0.05,
    "wrong_rescue_accuracy_gap_from_baseline_max": 0.05,
    "correct_wrong_accuracy_difference_max": 0.03,
    "future_response_error_advantage_min": 0.10,
    "recipient_positive_future_advantage_fraction_min": 0.75,
    "discovery_positive_task_count_min": 2,
    "confirmation_positive_task_count_min": 3,
}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def endpoint_paths(split: str) -> list[Path]:
    if split == "discovery":
        paths = sorted(P1171_CHECKPOINTS.glob("*step10000.pt"))
        if len(paths) != 64:
            raise RuntimeError(f"expected 64 discovery endpoints, found {len(paths)}")
        return paths
    if split == "confirmation":
        allowed = {task.name for task in p1172.TASK_SPECS if task.split == "discovery"}
        paths = sorted(
            path
            for path in P1172_CHECKPOINTS.glob("*step12000.pt")
            if any(path.name.startswith(task_name + "_") for task_name in allowed)
        )
        if len(paths) != 64:
            raise RuntimeError(f"expected 64 confirmation endpoints, found {len(paths)}")
        return paths
    raise ValueError(split)


def endpoint_step(split: str) -> int:
    return 10000 if split == "discovery" else 12000


def checkpoint_manifest(split: str) -> dict[str, str]:
    paths: set[Path] = set(endpoint_paths(split))
    end_step = endpoint_step(split)
    for endpoint in list(paths):
        for step in HISTORY_STEPS:
            paths.add(endpoint.with_name(endpoint.name.replace(f"step{end_step:05d}", f"step{step:05d}")))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing history checkpoints: {missing[:3]}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(paths)
    }


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1182 is already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("missing audit script")
    phase1181_final = read_json(p1181.FINAL_PATH)
    phase1181_audit = read_json(p1181.OUT_ROOT / "audit/independent_audit.json")
    if not phase1181_final["primary_pass"] or not phase1181_audit["integrity_and_recompute_pass"]:
        raise RuntimeError("Phase1181 did not authorize Phase1182")
    protocol = {
        "phase": PHASE,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "phase1181_final_digest": phase1181_final["final_digest"],
            "phase1181_audit_digest": phase1181_audit["audit_digest"],
        },
        "scientific_object": (
            "Prediction and rescue of task-centered quotient causal-response residuals in freely trained, "
            "ungated RoleSquareNetwork systems."
        ),
        "claim_exclusions": [
            "A camera pass is prediction, not behavioral necessity.",
            "A donor pass is limited to the declared channel-contribution intervention algebra.",
            "No semantic, Transformer, language, or general intelligence mechanism is inferred.",
            "The primary claim requires endpoint, prefix, and donor gates; gates are also reported separately.",
        ],
        "development_disclosure": (
            "All Phase1171 responses were used for development. Phase1172 confirmation-split tasks were used "
            "by Phase1181 and are excluded here. The Phase1172 discovery-split response spectra were not "
            "inspected before this registration and form the only Phase1182 confirmation panel."
        ),
        "splits": {
            "discovery": {
                "source_phase": 1171,
                "system_count": 64,
                "camera_train_task_indices": [0, 1, 2, 3, 4, 5],
                "camera_test_and_rescue_task_indices": [6, 7],
            },
            "confirmation": {
                "source_phase": 1172,
                "task_names": [task.name for task in p1172.TASK_SPECS if task.split == "discovery"],
                "system_count": 64,
            },
        },
        "camera": {
            "target": "ordered single-channel holdout margin-response spectrum",
            "score": "task-centered, row-normalized response residual",
            "endpoint_null": "endpoint behavior/output/gradient plus fixed early behavior/output/gradient trajectory",
            "endpoint_candidate": "passive endpoint channel-distribution and activation-geometry invariants",
            "prefix_null": "behavior/output/gradient trajectory at fixed steps 25,50,75,100,150",
            "prefix_candidate": "passive internal-geometry trajectory at the same five fixed steps",
            "ridge_l2": {"endpoint": ENDPOINT_L2, "prefix": PREFIX_L2},
            "risk_sign": "Delta = Risk(null) - Risk(null plus internal); positive is internal improvement",
        },
        "rescue": {
            "calibration": "single-channel effects on deterministic half of sealed holdout inputs",
            "injury": f"zero the {INJURY_CHANNEL_COUNT} largest-absolute calibration-effect channels",
            "behavior_pool": "four closest donors by behavior features within the same task",
            "correct_donor": "minimum ordered calibration-response distance in behavior pool",
            "wrong_donor": "maximum ordered calibration-response distance in behavior pool",
            "alignment": "calibration-response rank alignment",
            "future_test": f"{FUTURE_MASK_COUNT} frozen masks of {FUTURE_MASK_SIZE} channels on the disjoint holdout half",
        },
        "thresholds": THRESHOLDS,
        "decision": {
            "primary": "endpoint_gate and prefix_gate and donor_gate on confirmation",
            "failure_action": "close this camera/rescue registry; do not retune l2, times, injury size, or masks",
            "pass_action": "authorize an independently trained new-network replication before language-model transfer",
        },
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1181": file_sha256(Path(p1181.__file__)),
            "phase1171": file_sha256(Path(p1171.__file__)),
            "phase1172": file_sha256(Path(p1172.__file__)),
        },
        "checkpoint_manifests": {
            "discovery": checkpoint_manifest("discovery"),
            "confirmation": checkpoint_manifest("confirmation"),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1182 is not preregistered")
    protocol = read_json(PROTOCOL_PATH)
    stored = protocol.pop("protocol_digest")
    if digest(protocol) != stored:
        raise RuntimeError("protocol digest mismatch")
    protocol["protocol_digest"] = stored
    scripts = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1181": Path(p1181.__file__),
        "phase1171": Path(p1171.__file__),
        "phase1172": Path(p1172.__file__),
    }
    for name, path in scripts.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen script changed: {name}")
    for split in ("discovery", "confirmation"):
        if checkpoint_manifest(split) != protocol["checkpoint_manifests"][split]:
            raise RuntimeError(f"checkpoint manifest changed: {split}")
    return protocol


@torch.inference_mode()
def output_features(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> list[float]:
    logits, _ = p1181.fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    margins = p1181.correct_margin(logits, targets)
    probabilities = torch.softmax(logits, dim=1)
    values_by_name = (
        margins,
        probabilities.max(dim=1).values,
        probabilities.gather(1, targets[:, None]).squeeze(1),
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1),
        logits.norm(dim=1),
    )
    features: list[float] = []
    for mask in (panel.train_mask, panel.holdout_mask):
        selected = mask.to(device)
        features.extend(
            [
                float((logits[selected].argmax(dim=1) == targets[selected]).float().mean().item()),
                float(F.cross_entropy(logits[selected], targets[selected]).item()),
            ]
        )
        for values in values_by_name:
            array = values[selected].cpu().numpy().astype(np.float64)
            features.extend((float(array.mean()), float(array.std())))
            features.extend(float(value) for value in np.quantile(array, (0.1, 0.25, 0.5, 0.75, 0.9)))
    return features


def gradient_features(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> list[float]:
    model.zero_grad(set_to_none=True)
    logits = model(panel.x[panel.train_mask].to(device)).float()
    targets = panel.y[panel.train_mask].to(device)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    gradients = (
        model.left_embedding.weight.grad,
        model.right_embedding.weight.grad,
        model.hidden.weight.grad,
        model.output.weight.grad,
    )
    norms = [float(value.detach().float().norm().item()) for value in gradients]
    model.zero_grad(set_to_none=True)
    return [float(loss.item()), *norms, float(math.sqrt(sum(value * value for value in norms)))]


@torch.inference_mode()
def internal_features(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    device: torch.device,
) -> list[float]:
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
    features.extend(
        float(correlation[left, right])
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    )
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0).sqrt().cpu().numpy()
    eigenvalues = eigenvalues / max(float(np.linalg.norm(eigenvalues)), 1e-12)
    features.extend(float(value) for value in np.quantile(eigenvalues, QUANTILES))
    return features


def trajectory_summary(rows: list[list[float]]) -> list[float]:
    matrix = np.asarray(rows, dtype=np.float64)
    time = np.log(np.asarray(HISTORY_STEPS, dtype=np.float64))
    time = (time - time.mean()) / time.std()
    slope = (time[:, None] * matrix).sum(axis=0) / float((time * time).sum())
    return np.concatenate((matrix[0], matrix[-1], matrix[-1] - matrix[0], slope)).tolist()


def history_path(endpoint: Path, split: str, step: int) -> Path:
    return endpoint.with_name(
        endpoint.name.replace(f"step{endpoint_step(split):05d}", f"step{step:05d}")
    )


def load_panel(payload: dict[str, Any], split: str) -> p1181.DataPanel:
    return p1181.load_panel(payload, "discovery" if split == "discovery" else "confirmation")


def build_record(
    endpoint: Path,
    split: str,
    device: torch.device,
    gauge: bool,
) -> dict[str, Any]:
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    panel = load_panel(payload, split)
    model = p1181.load_model(payload, device)
    behavior = p1181.behavior_metrics(model, panel, device)
    target = p1181.response_spectrum(model, panel, device)["ordered"]
    endpoint_null_current = output_features(model, panel, device) + gradient_features(model, panel, device)
    endpoint_internal = internal_features(model, panel, device)
    history_null_rows: list[list[float]] = []
    history_internal_rows: list[list[float]] = []
    prefix_model_for_gauge: p1171.RoleSquareNetwork | None = None
    for step in HISTORY_STEPS:
        history_payload = torch.load(history_path(endpoint, split, step), map_location="cpu", weights_only=False)
        history_model = p1181.load_model(history_payload, device)
        history_null_rows.append(
            output_features(history_model, panel, device) + gradient_features(history_model, panel, device)
        )
        history_internal_rows.append(internal_features(history_model, panel, device))
        if step == HISTORY_STEPS[-1] and gauge:
            prefix_model_for_gauge = history_model
        else:
            del history_model
    prefix_null = trajectory_summary(history_null_rows)
    prefix_internal = trajectory_summary(history_internal_rows)
    gauge_result = None
    if gauge:
        transformed_endpoint = p1181.gauge_model(model, 11820000 + int(payload["task_index"]), device)
        transformed_prefix = p1181.gauge_model(
            prefix_model_for_gauge,
            11821000 + int(payload["task_index"]),
            device,
        )
        endpoint_gauge_error = float(
            np.max(np.abs(np.asarray(endpoint_internal) - np.asarray(internal_features(transformed_endpoint, panel, device))))
        )
        prefix_last_original = history_internal_rows[-1]
        prefix_gauge_error = float(
            np.max(
                np.abs(
                    np.asarray(prefix_last_original)
                    - np.asarray(internal_features(transformed_prefix, panel, device))
                )
            )
        )
        gauge_result = {
            "endpoint_internal_feature_max_error": endpoint_gauge_error,
            "prefix_internal_feature_max_error": prefix_gauge_error,
        }
        del transformed_endpoint, transformed_prefix, prefix_model_for_gauge
    record = {
        "split": split,
        "source_phase": int(payload["phase"]),
        "checkpoint": endpoint.name,
        "checkpoint_sha256": file_sha256(endpoint),
        "task_name": str(payload["task_name"]),
        "task_index": int(payload["task_index"]),
        "replicate": int(payload["replicate"]),
        "seed": int(payload["seed"]),
        "behavior": behavior,
        "target": target,
        "endpoint_null": endpoint_null_current + prefix_null,
        "endpoint_internal": endpoint_internal,
        "prefix_null": prefix_null,
        "prefix_internal": prefix_internal,
        "gauge": gauge_result,
    }
    del model
    torch.cuda.empty_cache()
    return record


def qualified(record: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return bool(
        record["behavior"]["all_logits_finite"] == 1.0
        and record["behavior"]["train_accuracy"] >= thresholds["train_accuracy_min"]
        and record["behavior"]["holdout_accuracy"] >= thresholds["holdout_accuracy_min"]
    )


def qualifying_task_names(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> list[str]:
    return sorted(
        task_name
        for task_name in {row["task_name"] for row in rows}
        if sum(row["task_name"] == task_name and qualified(row, thresholds) for row in rows)
        >= thresholds["qualified_system_count_per_task_min"]
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    standardized = (x - mean) / scale
    design = np.concatenate((np.ones((len(x), 1)), standardized), axis=1)
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_ridge(seal: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (x - seal["mean"]) / seal["scale"]
    design = np.concatenate((np.ones((len(x), 1)), standardized), axis=1)
    return design @ seal["weights"]


def residual_metrics(prediction: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    predicted_residual = np.zeros_like(prediction)
    target_residual = np.zeros_like(target)
    for group in np.unique(groups):
        selected = groups == group
        predicted_residual[selected] = prediction[selected] - prediction[selected].mean(axis=0)
        target_residual[selected] = target[selected] - target[selected].mean(axis=0)
    predicted_residual /= np.maximum(np.linalg.norm(predicted_residual, axis=1, keepdims=True), 1e-12)
    target_residual /= np.maximum(np.linalg.norm(target_residual, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(predicted_residual * target_residual, axis=1)
    squared_error = np.mean((predicted_residual - target_residual) ** 2, axis=1)
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
        "null": fit_ridge(null, target, l2),
        "joint": fit_ridge(np.concatenate((null, internal), axis=1), target, l2),
    }


def score_stage(
    rows: list[dict[str, Any]],
    stage: str,
    seals: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["task_name"] for row in rows])
    null = np.asarray([row[f"{stage}_null"] for row in rows], dtype=np.float64)
    internal = np.asarray([row[f"{stage}_internal"] for row in rows], dtype=np.float64)
    null_prediction = predict_ridge(seals["null"], null)
    joint_prediction = predict_ridge(seals["joint"], np.concatenate((null, internal), axis=1))
    null_metrics = residual_metrics(null_prediction, target, groups)
    joint_metrics = residual_metrics(joint_prediction, target, groups)
    return {
        "system_count": len(rows),
        "task_count": len(np.unique(groups)),
        "null": null_metrics,
        "joint": joint_metrics,
        "residual_cosine_improvement": joint_metrics["mean_cosine"] - null_metrics["mean_cosine"],
        "residual_risk_improvement": null_metrics["mean_squared_error"] - joint_metrics["mean_squared_error"],
    }


def save_camera_seal(seals: dict[str, dict[str, dict[str, np.ndarray]]], metadata: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for stage, stage_seals in seals.items():
        for camera_name, camera in stage_seals.items():
            for key, value in camera.items():
                arrays[f"{stage}__{camera_name}__{key}"] = value
    CAMERA_SEAL.parent.mkdir(parents=True, exist_ok=True)
    temporary = CAMERA_SEAL.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, CAMERA_SEAL)
    metadata = dict(metadata)
    metadata["npz_sha256"] = file_sha256(CAMERA_SEAL)
    metadata["array_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
    metadata["metadata_digest"] = digest(metadata)
    write_json(CAMERA_METADATA, metadata)


def load_camera_seal() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    metadata = read_json(CAMERA_METADATA)
    if file_sha256(CAMERA_SEAL) != metadata["npz_sha256"]:
        raise RuntimeError("camera seal hash mismatch")
    arrays = np.load(CAMERA_SEAL)
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for stage in ("endpoint", "prefix"):
        result[stage] = {}
        for camera_name in ("null", "joint"):
            result[stage][camera_name] = {
                key: arrays[f"{stage}__{camera_name}__{key}"]
                for key in ("mean", "scale", "weights")
            }
    return result


def split_holdout(panel: p1181.DataPanel) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.where(panel.holdout_mask)[0]
    code = panel.x[indices, 0] * 131 + panel.x[indices, 1] * 17
    calibration = torch.zeros_like(panel.holdout_mask)
    calibration[indices[code % 2 == 0]] = True
    return calibration, panel.holdout_mask & ~calibration


@torch.inference_mode()
def rescue_bundle(
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
    base_margin = p1181.correct_margin(logits, targets)
    response: list[float] = []
    for channel in range(model.config.width):
        changed = logits - q[:, channel, None] * weight[:, channel][None, :]
        response.append(
            float(
                (
                    base_margin[calibration]
                    - p1181.correct_margin(changed, targets)[calibration]
                ).mean().item()
            )
        )
    behavior = p1181.behavior_metrics(model, panel, device)
    return {
        "q_evaluation": q[evaluation].cpu(),
        "weight": weight.cpu(),
        "targets_evaluation": targets[evaluation].cpu(),
        "evaluation_mask": evaluation_mask,
        "calibration_response": np.asarray(response, dtype=np.float64),
        "behavior_vector": np.asarray(
            [behavior[name] for name in p1181.BEHAVIOR_FEATURES], dtype=np.float64
        ),
    }


def future_masks(task_name: str, width: int) -> list[np.ndarray]:
    seed = int(hashlib.sha256(("phase1182:" + task_name).encode("utf-8")).hexdigest()[:16], 16)
    generator = np.random.default_rng(seed)
    return [
        np.sort(generator.choice(width, size=FUTURE_MASK_SIZE, replace=False))
        for _ in range(FUTURE_MASK_COUNT)
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
    base_margin = p1181.correct_margin(logits, targets)
    response: list[float] = []
    for channels in masks:
        index = torch.tensor(channels, dtype=torch.long, device=device)
        changed = logits - q[:, index] @ weight[:, index].T
        response.append(
            float((base_margin - p1181.correct_margin(changed, targets)).mean().item())
        )
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


def rescue_task(
    task_name: str,
    records: list[dict[str, Any]],
    path_map: dict[str, Path],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    bundles: list[dict[str, Any]] = []
    for record in records:
        payload = torch.load(path_map[record["checkpoint"]], map_location="cpu", weights_only=False)
        model = p1181.load_model(payload, device)
        panel = load_panel(payload, split)
        bundle = rescue_bundle(model, panel, device)
        bundle.update({"record": record, "model": model, "panel": panel})
        bundles.append(bundle)
    behavior = np.stack([bundle["behavior_vector"] for bundle in bundles])
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    masks = future_masks(task_name, 128)
    rows: list[dict[str, Any]] = []
    for recipient_index, recipient in enumerate(bundles):
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
        injured_channels = np.argsort(np.abs(recipient["calibration_response"]))[-INJURY_CHANNEL_COUNT:]
        baseline = evaluate_hybrid(
            recipient["q_evaluation"], recipient["weight"], recipient["targets_evaluation"], masks, device
        )
        injured_q = recipient["q_evaluation"].clone()
        injured_q[:, injured_channels] = 0.0
        injured = evaluate_hybrid(
            injured_q, recipient["weight"], recipient["targets_evaluation"], masks, device
        )
        rescues: dict[str, Any] = {}
        for label, donor_index in (("correct", correct_index), ("wrong", wrong_index)):
            donor = bundles[donor_index]
            _, donor_hidden = p1181.fp32_state(
                donor["model"], recipient["panel"].x, device
            )
            donor_q = donor_hidden.square()[recipient["evaluation_mask"].to(device)].cpu()
            donor_order = np.argsort(donor["calibration_response"])
            hybrid_q = recipient["q_evaluation"].clone()
            hybrid_weight = recipient["weight"].clone()
            for recipient_channel in injured_channels:
                donor_channel = donor_order[recipient_rank[recipient_channel]]
                hybrid_q[:, recipient_channel] = donor_q[:, donor_channel]
                hybrid_weight[:, recipient_channel] = donor["weight"][:, donor_channel]
            evaluated = evaluate_hybrid(
                hybrid_q,
                hybrid_weight,
                recipient["targets_evaluation"],
                masks,
                device,
            )
            evaluated["future_response_error"] = response_error(
                evaluated["future_response"], baseline["future_response"]
            )
            evaluated["donor_replicate"] = donor["record"]["replicate"]
            evaluated["calibration_response_distance"] = distances[donor_index]
            rescues[label] = evaluated
        injured["future_response_error"] = response_error(
            injured["future_response"], baseline["future_response"]
        )
        rows.append(
            {
                "task_name": task_name,
                "recipient_replicate": recipient["record"]["replicate"],
                "baseline": baseline,
                "injured": injured,
                "correct": rescues["correct"],
                "wrong": rescues["wrong"],
            }
        )
    for bundle in bundles:
        del bundle["model"]
    torch.cuda.empty_cache()
    return {"task_name": task_name, "rows": rows}


def rescue_summary(task_results: list[dict[str, Any]], split: str, thresholds: dict[str, Any]) -> dict[str, Any]:
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
                [
                    row["correct"]["future_response_error"]
                    < row["wrong"]["future_response_error"]
                    for row in rows
                ]
            )
        )
        if advantage > 0:
            positive_task_count += 1
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
            [
                row["correct"]["future_response_error"]
                < row["wrong"]["future_response_error"]
                for row in all_rows
            ]
        )
    )
    required_positive_tasks = (
        thresholds["discovery_positive_task_count_min"]
        if split == "discovery"
        else thresholds["confirmation_positive_task_count_min"]
    )
    gate = bool(
        baseline - injured >= thresholds["injury_accuracy_drop_min"]
        and abs(correct - baseline) <= thresholds["correct_rescue_accuracy_gap_from_baseline_max"]
        and abs(wrong - baseline) <= thresholds["wrong_rescue_accuracy_gap_from_baseline_max"]
        and abs(correct - wrong) <= thresholds["correct_wrong_accuracy_difference_max"]
        and future_advantage >= thresholds["future_response_error_advantage_min"]
        and positive_fraction >= thresholds["recipient_positive_future_advantage_fraction_min"]
        and positive_task_count >= required_positive_tasks
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
        "required_positive_task_count": required_positive_tasks,
        "task_summaries": task_summaries,
        "gate_pass": gate,
    }


def camera_gate(stage: str, score: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return bool(
        score["joint"]["mean_cosine"] >= thresholds[f"{stage}_joint_residual_cosine_min"]
        and score["residual_cosine_improvement"]
        >= thresholds[f"{stage}_residual_cosine_improvement_min"]
        and score["residual_risk_improvement"]
        >= thresholds[f"{stage}_residual_risk_improvement_min"]
    )


def build_split_records(split: str, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gauged_tasks: set[str] = set()
    paths = endpoint_paths(split)
    for index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task_name = str(payload["task_name"])
        row = build_record(path, split, device, task_name not in gauged_tasks)
        gauged_tasks.add(task_name)
        rows.append(row)
        print(canonical_json({"split": split, "completed": index + 1, "total": len(paths)}), flush=True)
    return rows


def run_discovery() -> None:
    protocol = validate_protocol()
    if DISCOVERY_ROWS.exists():
        raise RuntimeError("discovery already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = build_split_records("discovery", device)
    thresholds = protocol["thresholds"]
    fit_rows = [row for row in rows if row["task_index"] < 6 and qualified(row, thresholds)]
    score_rows = [row for row in rows if row["task_index"] >= 6 and qualified(row, thresholds)]
    development_seals = {
        "endpoint": fit_stage(fit_rows, "endpoint", ENDPOINT_L2),
        "prefix": fit_stage(fit_rows, "prefix", PREFIX_L2),
    }
    endpoint_score = score_stage(score_rows, "endpoint", development_seals["endpoint"])
    prefix_score = score_stage(score_rows, "prefix", development_seals["prefix"])
    endpoint_pass = camera_gate("endpoint", endpoint_score, thresholds)
    prefix_pass = camera_gate("prefix", prefix_score, thresholds)
    path_map = {path.name: path for path in endpoint_paths("discovery")}
    rescue_tasks = []
    for task_name in sorted({row["task_name"] for row in score_rows}):
        task_rows = [row for row in score_rows if row["task_name"] == task_name]
        rescue_tasks.append(rescue_task(task_name, task_rows, path_map, "discovery", device))
    rescue = {
        "tasks": rescue_tasks,
        "summary": rescue_summary(rescue_tasks, "discovery", thresholds),
    }
    gauges = [row["gauge"] for row in rows if row["gauge"] is not None]
    gauge_max = max(
        max(item["endpoint_internal_feature_max_error"], item["prefix_internal_feature_max_error"])
        for item in gauges
    )
    gauge_pass = gauge_max <= thresholds["feature_gauge_max_error_max"]
    discovery_pass = bool(endpoint_pass and prefix_pass and rescue["summary"]["gate_pass"] and gauge_pass)
    final_fit_rows = [row for row in rows if qualified(row, thresholds)]
    final_seals = {
        "endpoint": fit_stage(final_fit_rows, "endpoint", ENDPOINT_L2),
        "prefix": fit_stage(final_fit_rows, "prefix", PREFIX_L2),
    }
    metadata = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_system_count": len(final_fit_rows),
        "fit_task_count": len({row["task_name"] for row in final_fit_rows}),
        "endpoint_l2": ENDPOINT_L2,
        "prefix_l2": PREFIX_L2,
        "discovery_pass": discovery_pass,
    }
    save_camera_seal(final_seals, metadata)
    summary = {
        "phase": PHASE,
        "split": "discovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_count": len(rows),
        "fit_system_count": len(fit_rows),
        "score_system_count": len(score_rows),
        "endpoint": {**endpoint_score, "gate_pass": endpoint_pass},
        "prefix": {**prefix_score, "gate_pass": prefix_pass},
        "rescue": rescue["summary"],
        "feature_gauge_max_error": gauge_max,
        "feature_gauge_pass": gauge_pass,
        "discovery_pass": discovery_pass,
        "rows_digest": digest(rows),
        "camera_seal_sha256": file_sha256(CAMERA_SEAL),
    }
    summary["summary_digest"] = digest(summary)
    write_jsonl(DISCOVERY_ROWS, rows)
    write_json(DISCOVERY_RESCUE, rescue)
    write_json(DISCOVERY_SUMMARY, summary)
    print(canonical_json(summary))


def run_confirmation() -> None:
    protocol = validate_protocol()
    if not DISCOVERY_SUMMARY.exists() or not read_json(DISCOVERY_SUMMARY)["discovery_pass"]:
        raise RuntimeError("confirmation denied because discovery did not pass")
    if CONFIRMATION_ROWS.exists():
        raise RuntimeError("confirmation already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = build_split_records("confirmation", device)
    thresholds = protocol["thresholds"]
    task_names = qualifying_task_names(rows, thresholds)
    score_rows = [
        row for row in rows if row["task_name"] in task_names and qualified(row, thresholds)
    ]
    seals = load_camera_seal()
    endpoint_score = score_stage(score_rows, "endpoint", seals["endpoint"])
    prefix_score = score_stage(score_rows, "prefix", seals["prefix"])
    enough_tasks = len(task_names) >= thresholds["qualified_task_count_confirmation_min"]
    endpoint_pass = bool(enough_tasks and camera_gate("endpoint", endpoint_score, thresholds))
    prefix_pass = bool(enough_tasks and camera_gate("prefix", prefix_score, thresholds))
    path_map = {path.name: path for path in endpoint_paths("confirmation")}
    rescue_tasks = []
    for task_name in task_names:
        task_rows = [row for row in score_rows if row["task_name"] == task_name]
        rescue_tasks.append(rescue_task(task_name, task_rows, path_map, "confirmation", device))
    rescue = {
        "tasks": rescue_tasks,
        "summary": rescue_summary(rescue_tasks, "confirmation", thresholds),
    }
    gauges = [row["gauge"] for row in rows if row["gauge"] is not None]
    gauge_max = max(
        max(item["endpoint_internal_feature_max_error"], item["prefix_internal_feature_max_error"])
        for item in gauges
    )
    gauge_pass = gauge_max <= thresholds["feature_gauge_max_error_max"]
    primary_pass = bool(
        enough_tasks
        and endpoint_pass
        and prefix_pass
        and rescue["summary"]["gate_pass"]
        and gauge_pass
    )
    summary = {
        "phase": PHASE,
        "split": "confirmation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_count": len(rows),
        "qualifying_task_names": task_names,
        "qualifying_task_count": len(task_names),
        "score_system_count": len(score_rows),
        "enough_tasks": enough_tasks,
        "endpoint": {**endpoint_score, "gate_pass": endpoint_pass},
        "prefix": {**prefix_score, "gate_pass": prefix_pass},
        "rescue": rescue["summary"],
        "feature_gauge_max_error": gauge_max,
        "feature_gauge_pass": gauge_pass,
        "primary_pass": primary_pass,
        "rows_digest": digest(rows),
        "camera_seal_sha256": file_sha256(CAMERA_SEAL),
    }
    summary["summary_digest"] = digest(summary)
    write_jsonl(CONFIRMATION_ROWS, rows)
    write_json(CONFIRMATION_RESCUE, rescue)
    write_json(CONFIRMATION_SUMMARY, summary)
    print(canonical_json(summary))


def analyze() -> None:
    protocol = validate_protocol()
    if not CONFIRMATION_SUMMARY.exists():
        raise RuntimeError("confirmation has not completed")
    discovery = read_json(DISCOVERY_SUMMARY)
    confirmation = read_json(CONFIRMATION_SUMMARY)
    final = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "discovery": discovery,
        "confirmation": confirmation,
        "primary_pass": confirmation["primary_pass"],
        "component_decisions": {
            "endpoint_increment": confirmation["endpoint"]["gate_pass"],
            "prefix_increment": confirmation["prefix"]["gate_pass"],
            "donor_future_response_rescue": confirmation["rescue"]["gate_pass"],
        },
        "claim_scope": (
            "Each passing component is limited to task-centered quotient response prediction or rescue in "
            "the declared freely trained micro-network and intervention algebra."
        ),
        "auto_continue": {
            "authorized": bool(confirmation["primary_pass"]),
            "next_phase": (
                "independently trained new-network replication"
                if confirmation["primary_pass"]
                else None
            ),
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preregister", "run-discovery", "run-confirmation", "analyze"),
    )
    args = parser.parse_args()
    commands: dict[str, Callable[[], None]] = {
        "preregister": preregister,
        "run-discovery": run_discovery,
        "run-confirmation": run_confirmation,
        "analyze": analyze,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
