#!/usr/bin/env python3
"""Phase1188: terminal prospective confirmation of the K165 three-evidence candidate.

The camera is selected and fitted only on the frozen Phase1171 development
panel.  Phase1188 then trains 64 new free networks on eight untouched affine
tasks and makes one terminal decision from endpoint prediction, early-prefix
prediction, and selective response-spectrum rescue.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
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
import phase1182_quotient_response_camera_and_rescue as p1182  # noqa: E402
import phase1183_gauge_exact_prospective_mechanism_closure as p1183  # noqa: E402
import phase1185_orthogonal_numerical_behavior_qualification as p1185  # noqa: E402
import phase1186_reducer_safe_numerical_qualification as p1186  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402


PHASE = 1188
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1188_terminal_three_evidence_confirmation_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1188_terminal_three_evidence_confirmation"
DEVELOPMENT_NPZ = OUT_ROOT / "development/frozen_camera.npz"
DEVELOPMENT_META = OUT_ROOT / "development/frozen_camera.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
TRAINING_METRICS = OUT_ROOT / "runs/training/training_metrics.jsonl"
TRAINING_SEAL = OUT_ROOT / "runs/training/seal.json"
CHECKPOINT_ROOT = OUT_ROOT / "runs/training/checkpoints"
SYSTEM_ROWS = OUT_ROOT / "analysis/systems.jsonl"
POSITIVE_ROWS = OUT_ROOT / "analysis/positive_sentinels.jsonl"
RESCUE_RAW = OUT_ROOT / "analysis/rescue_raw.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

P1171_SEAL = p1171.OUT_ROOT / "runs/training/seal.json"
P1187_AUDIT = p1187.OUT_ROOT / "audit/independent_audit.json"
EVIDENCE_CONTRACT = p1187.CONTRACT_PATH

MODULUS = 61
WIDTH = 128
REPLICATES = 8
TASK_COUNT = 8
TASK_SELECTION_SEED = 11880017
TRAIN_FRACTION = 0.50
HISTORY_STEPS = (25, 50, 75, 100, 150)
ENDPOINT_STEP = 10_000
CHECKPOINT_STEPS = (*HISTORY_STEPS, ENDPOINT_STEP)
ENDPOINT_L2 = 100.0
PREFIX_L2 = 1.0
INJURY_CHANNEL_COUNT = 32
FUTURE_MASK_COUNT = 32
FUTURE_MASK_SIZE = 8
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": ENDPOINT_STEP,
}
THRESHOLDS = {
    "train_accuracy_min": 0.95,
    "holdout_accuracy_min": 0.90,
    "qualified_system_count_per_task_min": 6,
    "qualified_task_count_min": 4,
    "endpoint_joint_residual_cosine_min": 0.35,
    "endpoint_residual_cosine_improvement_min": 0.02,
    "endpoint_residual_risk_improvement_min": 0.0,
    "prefix_joint_residual_cosine_min": 0.15,
    "prefix_residual_cosine_improvement_min": 0.04,
    "prefix_residual_risk_improvement_min": 0.0,
    "algebraic_feature_error_max": 1e-12,
    "positive_sentinel_error_min": 1e-4,
    "injury_accuracy_drop_min": 0.50,
    "correct_rescue_accuracy_gap_from_baseline_max": 0.05,
    "wrong_rescue_accuracy_gap_from_baseline_max": 0.05,
    "correct_wrong_accuracy_difference_max": 0.03,
    "future_response_error_advantage_min": 0.10,
    "recipient_positive_future_advantage_fraction_min": 0.75,
    "confirmation_positive_task_count_min": 3,
    "discovery_positive_task_count_min": 2,
}


def eligible_operations() -> list[tuple[int, int, int]]:
    excluded = set(p1171.OPERATION_SAMPLE) | {
        p1171.PILOT_OPERATION,
        (19, 23, 7),
        (29, 31, 11),
    }
    return [operation for operation in p1171.eligible_operations() if operation not in excluded]


OPERATIONS = tuple(random.Random(TASK_SELECTION_SEED).sample(eligible_operations(), TASK_COUNT))
TASKS = {
    f"terminal_affine_{index:02d}_a{operation[0]}_b{operation[1]}_g{operation[2]}": operation
    for index, operation in enumerate(OPERATIONS)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
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


def model_seed(task_index: int, replicate: int) -> int:
    return 11880000 + task_index * 100_003 + replicate * 1_009


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data(operation: tuple[int, int, int], seed: int) -> dict[str, torch.Tensor]:
    return p1171.make_data(operation, seed)


def panel_from_payload(payload: dict[str, Any]) -> p1181.DataPanel:
    data = make_data(tuple(payload["operation"]), int(payload["seed"]) + 17)
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    train_mask[: len(data["train_x"])] = True
    return p1181.DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def checkpoint_path(task_name: str, replicate: int, seed: int, step: int) -> Path:
    return CHECKPOINT_ROOT / f"{task_name}_r{replicate}_s{seed}_step{step:05d}.pt"


def history_path(endpoint: Path, step: int) -> Path:
    return endpoint.with_name(endpoint.name.replace("step10000", f"step{step:05d}"))


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def checkpoint_payload(
    model: p1171.RoleSquareNetwork,
    task_name: str,
    task_index: int,
    operation: tuple[int, int, int],
    replicate: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task_name,
        "task_index": task_index,
        "operation": operation,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": {"modulus": MODULUS, "width": WIDTH},
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


@torch.inference_mode()
def canonical_channel_signature(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> list[float]:
    """Signed-permutation invariant while preserving within-channel joint structure."""

    left = model.left_embedding.weight.detach().cpu().double().numpy()
    right = model.right_embedding.weight.detach().cpu().double().numpy()
    hidden_weight = model.hidden.weight.detach().cpu().double().numpy()
    output_weight = model.output.weight.detach().cpu().double().numpy()
    ids = x.cpu().numpy()
    left_projection = left @ hidden_weight.T
    right_projection = right @ hidden_weight.T
    hidden = (left[ids[:, 0]] + right[ids[:, 1]]) @ hidden_weight.T
    channels = np.stack(
        (
            np.sum(hidden_weight * hidden_weight, axis=1),
            np.sum(output_weight * output_weight, axis=0),
            np.sum(left_projection * left_projection, axis=0),
            np.sum(right_projection * right_projection, axis=0),
            np.mean(hidden * hidden, axis=0),
            np.mean(hidden ** 4, axis=0),
            np.mean(output_weight ** 4, axis=0),
            np.mean(left_projection * right_projection, axis=0),
        ),
        axis=1,
    )
    scales = np.sqrt(np.mean(channels * channels, axis=0))
    normalized = channels / np.maximum(scales[None, :], 1e-30)
    order = np.lexsort(tuple(normalized[:, index] for index in reversed(range(normalized.shape[1]))))
    return np.concatenate((np.log1p(scales), normalized[order].reshape(-1))).tolist()


def fit_dual_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    standardized = (x - mean) / scale
    y_mean = y.mean(axis=0)
    alpha = np.linalg.solve(
        standardized @ standardized.T + l2 * np.eye(len(standardized)),
        y - y_mean,
    )
    return {
        "mean": mean,
        "scale": scale,
        "train_z": standardized,
        "alpha": alpha,
        "y_mean": y_mean,
    }


def predict_dual(camera: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (x - camera["mean"]) / camera["scale"]
    return standardized @ camera["train_z"].T @ camera["alpha"] + camera["y_mean"]


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
    null_metrics = residual_metrics(predict_dual(cameras["null"], null), target, groups)
    joint_metrics = residual_metrics(
        predict_dual(cameras["joint"], np.concatenate((null, internal), axis=1)),
        target,
        groups,
    )
    return {
        "system_count": len(rows),
        "task_count": len(np.unique(groups)),
        "null": null_metrics,
        "joint": joint_metrics,
        "residual_cosine_improvement": joint_metrics["mean_cosine"] - null_metrics["mean_cosine"],
        "residual_risk_improvement": null_metrics["mean_squared_error"] - joint_metrics["mean_squared_error"],
    }


def camera_gate(stage: str, score: dict[str, Any]) -> bool:
    return bool(
        score["joint"]["mean_cosine"] >= THRESHOLDS[f"{stage}_joint_residual_cosine_min"]
        and score["residual_cosine_improvement"]
        >= THRESHOLDS[f"{stage}_residual_cosine_improvement_min"]
        and score["residual_risk_improvement"]
        >= THRESHOLDS[f"{stage}_residual_risk_improvement_min"]
    )


def save_camera(cameras: dict[str, Any], metadata: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for stage, stage_cameras in cameras.items():
        for label, camera in stage_cameras.items():
            for key, value in camera.items():
                arrays[f"{stage}__{label}__{key}"] = value
    DEVELOPMENT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEVELOPMENT_NPZ.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, DEVELOPMENT_NPZ)
    metadata = dict(metadata)
    metadata["npz_sha256"] = file_sha256(DEVELOPMENT_NPZ)
    metadata["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
    metadata["metadata_digest"] = digest(metadata)
    write_json(DEVELOPMENT_META, metadata)


def load_camera() -> dict[str, Any]:
    metadata = read_json(DEVELOPMENT_META)
    if file_sha256(DEVELOPMENT_NPZ) != metadata["npz_sha256"]:
        raise RuntimeError("development camera hash mismatch")
    arrays = np.load(DEVELOPMENT_NPZ)
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


def build_record(endpoint: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    panel = panel_from_payload(payload)
    model = load_model(payload, device)
    behavior = p1181.behavior_metrics(model, panel, device)
    response = p1181.response_spectrum(model, panel, device)
    replay = p1181.response_spectrum(model, panel, device)
    output_rows: list[list[float]] = []
    internal_rows: list[list[float]] = []
    for step in HISTORY_STEPS:
        history_payload = torch.load(history_path(endpoint, step), map_location="cpu", weights_only=False)
        history_model = load_model(history_payload, device)
        output_rows.append(p1183.output_features(history_model, panel, device))
        internal_rows.append(canonical_channel_signature(history_model, panel.x))
        del history_model
    prefix_null = p1182.trajectory_summary(output_rows)
    record = {
        "phase": PHASE,
        "source_phase": int(payload["phase"]),
        "checkpoint": endpoint.name,
        "checkpoint_sha256": file_sha256(endpoint),
        "task_name": str(payload["task_name"]),
        "task_index": int(payload["task_index"]),
        "operation": list(payload["operation"]),
        "replicate": int(payload["replicate"]),
        "seed": int(payload["seed"]),
        "behavior": behavior,
        "response": response,
        "target": response["unit_shape"],
        "endpoint_null": p1183.output_features(model, panel, device) + prefix_null,
        "endpoint_internal": p1183.algebraic_internal_features(model, panel.x),
        "prefix_null": prefix_null,
        "prefix_internal": p1182.trajectory_summary(internal_rows),
        "replay_maximum_error": float(
            np.max(np.abs(np.asarray(response["ordered"]) - np.asarray(replay["ordered"])))
        ),
    }
    del model
    torch.cuda.empty_cache()
    return record


def build_development_seal() -> None:
    if DEVELOPMENT_META.exists() or DEVELOPMENT_NPZ.exists():
        raise RuntimeError("development camera is already sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    paths = sorted((p1171.OUT_ROOT / "runs/training/checkpoints").glob("*step10000.pt"))
    if len(paths) != 64:
        raise RuntimeError("expected 64 frozen Phase1171 development endpoints")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        rows.append(build_record(path, device))
        print(canonical_json({"development_record": index + 1, "total": len(paths)}), flush=True)
    task_names = sorted({row["task_name"] for row in rows})
    fit_names = set(task_names[:6])
    test_names = set(task_names[6:])
    fit_rows = [row for row in rows if row["task_name"] in fit_names]
    test_rows = [row for row in rows if row["task_name"] in test_names]
    provisional = {
        "endpoint": fit_stage(fit_rows, "endpoint", ENDPOINT_L2),
        "prefix": fit_stage(fit_rows, "prefix", PREFIX_L2),
    }
    endpoint = score_stage(test_rows, "endpoint", provisional["endpoint"])
    prefix = score_stage(test_rows, "prefix", provisional["prefix"])
    endpoint["gate_pass"] = camera_gate("endpoint", endpoint)
    prefix["gate_pass"] = camera_gate("prefix", prefix)
    if not (endpoint["gate_pass"] and prefix["gate_pass"]):
        raise RuntimeError("development camera failed before preregistration")
    final_camera = {
        "endpoint": fit_stage(rows, "endpoint", ENDPOINT_L2),
        "prefix": fit_stage(rows, "prefix", PREFIX_L2),
    }
    metadata = {
        "phase": PHASE,
        "status": "development_only_camera_selection_and_fit",
        "source_phase": 1171,
        "source_training_seal_sha256": file_sha256(P1171_SEAL),
        "source_endpoint_count": len(paths),
        "source_task_count": len(task_names),
        "fit_task_names": sorted(fit_names),
        "development_holdout_task_names": sorted(test_names),
        "endpoint_feature": "phase1183_algebraic_moments",
        "prefix_feature": "canonical_sorted_joint_channel_signature_trajectory",
        "null_feature": "phase1183_output_features_with_prefix_output_trajectory",
        "selection_disclosure": (
            "Stage-specific features were selected after development-only comparisons on Phase1171. "
            "No Phase1188 task, seed, checkpoint, response, or outcome existed during selection."
        ),
        "endpoint_development_score": endpoint,
        "prefix_development_score": prefix,
        "camera_fit_system_count": len(rows),
        "endpoint_l2": ENDPOINT_L2,
        "prefix_l2": PREFIX_L2,
        "development_pass": True,
    }
    save_camera(final_camera, metadata)
    print(canonical_json(metadata))


def future_masks(task_name: str) -> list[np.ndarray]:
    seed = int(hashlib.sha256(("phase1188:" + task_name).encode("utf-8")).hexdigest()[:16], 16)
    generator = np.random.default_rng(seed)
    return [
        np.sort(generator.choice(WIDTH, size=FUTURE_MASK_SIZE, replace=False))
        for _ in range(FUTURE_MASK_COUNT)
    ]


def source_hashes() -> dict[str, str]:
    paths = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1171": Path(p1171.__file__),
        "phase1181": Path(p1181.__file__),
        "phase1182": Path(p1182.__file__),
        "phase1183": Path(p1183.__file__),
        "phase1185": Path(p1185.__file__),
        "phase1186": Path(p1186.__file__),
        "phase1187": Path(p1187.__file__),
        "phase1187_audit": P1187_AUDIT,
        "evidence_contract": EVIDENCE_CONTRACT,
        "phase1171_training_seal": P1171_SEAL,
        "development_camera_npz": DEVELOPMENT_NPZ,
        "development_camera_metadata": DEVELOPMENT_META,
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1188 already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("audit script must exist before preregistration")
    phase1187 = read_json(P1187_AUDIT)
    development = read_json(DEVELOPMENT_META)
    if not phase1187["phase1188_authorized_after_audit"] or not development["development_pass"]:
        raise RuntimeError("Phase1188 prerequisites did not authorize execution")
    expected = (
        (53, 5, 58),
        (28, 47, 47),
        (48, 39, 13),
        (51, 43, 4),
        (59, 58, 27),
        (55, 33, 55),
        (43, 26, 20),
        (1, 37, 2),
    )
    if OPERATIONS != expected:
        raise RuntimeError("task selection changed")
    allocation: list[dict[str, Any]] = []
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = make_data(operation, seed + 17)
            allocation.append(
                {
                    "task_name": task_name,
                    "task_index": task_index,
                    "operation": operation,
                    "replicate": replicate,
                    "seed": seed,
                    "train_pair_digest": digest(data["train_x"].tolist()),
                    "sealed_holdout_pair_digest": digest(data["holdout_x"].tolist()),
                    "train_label_digest": digest(data["train_y"].tolist()),
                    "sealed_holdout_label_digest": digest(data["holdout_y"].tolist()),
                }
            )
    protocol = {
        "phase": PHASE,
        "scientific_question": (
            "Does the frozen K165 camera prospectively predict endpoint and early-prefix response spectra, "
            "and does matched-donor transplantation selectively restore future response geometry?"
        ),
        "authorization": {
            "phase1187_audit_digest": phase1187["audit_digest"],
            "qualification_development_frozen": True,
            "one_terminal_registry_only": True,
        },
        "development_camera": {
            "metadata_digest": development["metadata_digest"],
            "npz_sha256": development["npz_sha256"],
            "development_pass": development["development_pass"],
            "selection_status": development["status"],
        },
        "task_selection_seed": TASK_SELECTION_SEED,
        "tasks": TASKS,
        "operations": OPERATIONS,
        "all_operations_excluded_from_prior_development": True,
        "replicates_per_task": REPLICATES,
        "system_count": TASK_COUNT * REPLICATES,
        "allocation": allocation,
        "history_steps": HISTORY_STEPS,
        "endpoint_step": ENDPOINT_STEP,
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "future_mask_digests": {
            name: digest([mask.tolist() for mask in future_masks(name)]) for name in TASKS
        },
        "typed_evidence_rules": {
            "contract_sha256": file_sha256(EVIDENCE_CONTRACT),
            "only_pass_authorizes": True,
            "abstention_does_not_authorize": True,
            "descriptive_float_does_not_gate": True,
        },
        "primary_conjunction": [
            "behavior",
            "typed_numerical_gauge",
            "positive_sentinel",
            "endpoint_camera",
            "prefix_camera",
            "selective_rescue",
            "independent_audit",
        ],
        "independent_audit_requirements": [
            "Recompute every source, protocol, camera, allocation, checkpoint, and result digest.",
            "Reconstruct all 64 behavior panels and endpoint response spectra from sealed checkpoints.",
            "Reconstruct all endpoint and prefix camera inputs and independently rescore the frozen camera.",
            "Recompile every typed numerical claim and replay one numerical case plus one positive sentinel per task.",
            "Independently recompute the complete rescue gate and replay one recipient intervention per passing task.",
            "Treat a main failure as terminal and never reinterpret an unrun component as a negative mechanism result.",
        ],
        "terminal_rules": [
            "No Phase1188 task, seed, checkpoint, response, donor, or future mask may alter the frozen camera.",
            "Held-out outcomes are absent until all 384 checkpoints and training-only metrics are sealed.",
            "All 64 systems are reported; no task, replicate, or low-scoring system may be deleted.",
            "A main pass does not authorize transfer until the independent audit passes.",
            "Regardless of result, this camera registry closes after one formal decision.",
            "A failure does not authorize another feature, threshold, task, seed, or reducer search.",
            "No Transformer or language-model transfer occurs in Phase1188.",
        ],
        "claim_scope": (
            "A pass is confined to the RoleSquare affine family under signed hidden-channel permutation. "
            "It does not establish semantic identity, language encoding, behavioral necessity, or universality."
        ),
        "source_hashes": source_hashes(),
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    copied = dict(protocol)
    stored = copied.pop("protocol_digest")
    if digest(copied) != stored:
        raise RuntimeError("protocol digest mismatch")
    if source_hashes() != protocol["source_hashes"]:
        raise RuntimeError("frozen source or development camera changed")
    return protocol


def train_and_seal() -> None:
    protocol = validate_protocol()
    if TRAINING_SEAL.exists():
        raise RuntimeError("training is already sealed")
    if SYSTEM_ROWS.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("holdout outcomes exist before training seal")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            set_seed(seed)
            data = make_data(operation, seed + 17)
            model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(MODULUS, WIDTH)).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=TRAINING["learning_rate"],
                weight_decay=TRAINING["weight_decay"],
            )
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
            for step in range(1, ENDPOINT_STEP + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(train_x).float()
                    loss = F.cross_entropy(logits, train_y)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"nonfinite loss: {task_name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                if step not in CHECKPOINT_STEPS:
                    continue
                model.eval()
                train_metrics = p1171.evaluate(model, data["train_x"], data["train_y"], device)
                path = checkpoint_path(task_name, replicate, seed, step)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        model, task_name, task_index, operation, replicate, seed, step
                    ),
                    path,
                )
                checkpoint_hashes[path.name] = file_sha256(path)
                rows.append(
                    {
                        "task_name": task_name,
                        "task_index": task_index,
                        "operation": operation,
                        "replicate": replicate,
                        "seed": seed,
                        "step": step,
                        "loss": float(loss.item()),
                        "train": train_metrics,
                        "train_pair_digest": digest(data["train_x"].tolist()),
                        "sealed_holdout_pair_digest": digest(data["holdout_x"].tolist()),
                        "train_label_digest": digest(data["train_y"].tolist()),
                        "sealed_holdout_label_digest": digest(data["holdout_y"].tolist()),
                        "holdout_evaluated_during_training": False,
                        "holdout_used_by_gradient": False,
                        "checkpoint_sha256": checkpoint_hashes[path.name],
                    }
                )
            print(canonical_json({"trained": task_name, "replicate": replicate}), flush=True)
            del model, optimizer, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()
    write_jsonl(TRAINING_METRICS, rows)
    seal = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": TASK_COUNT * REPLICATES,
        "checkpoint_count": len(rows),
        "training_metrics_sha256": file_sha256(TRAINING_METRICS),
        "checkpoint_hashes": checkpoint_hashes,
        "holdout_outcomes_absent_at_sealing": not SYSTEM_ROWS.exists() and not SUMMARY_PATH.exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "all_training_logits_exactly_finite": all(row["train"]["exact_all_finite"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = digest(seal)
    write_json(TRAINING_SEAL, seal)
    print(canonical_json({"seal_digest": seal["seal_digest"], "checkpoint_count": len(rows)}))


def verify_training_seal(protocol: dict[str, Any]) -> dict[str, Any]:
    seal = read_json(TRAINING_SEAL)
    copied = dict(seal)
    stored = copied.pop("seal_digest")
    if digest(copied) != stored or seal["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("training seal mismatch")
    if file_sha256(TRAINING_METRICS) != seal["training_metrics_sha256"]:
        raise RuntimeError("training metrics changed")
    for name, expected in seal["checkpoint_hashes"].items():
        if file_sha256(CHECKPOINT_ROOT / name) != expected:
            raise RuntimeError(f"checkpoint changed: {name}")
    return seal


def typed_bounded(value: float, threshold: float, comparator: str) -> dict[str, Any]:
    return p1187.compile_claim(
        {
            "claim_type": "bounded_float",
            "gating": True,
            "value": float(value),
            "threshold": float(threshold),
            "comparator": comparator,
            "dtype": "float64",
        },
        read_json(EVIDENCE_CONTRACT),
    )


def typed_universal(source: dict[str, Any]) -> dict[str, Any]:
    return p1187.compile_claim(
        {
            "claim_type": "universal_boolean",
            "gating": True,
            "agree_count": int(source["agree_count"]),
            "eligible_count": int(source["eligible_count"]),
            "abstained": bool(source["abstained"]),
        },
        read_json(EVIDENCE_CONTRACT),
    )


def typed_conjunction(claims: list[dict[str, Any]]) -> dict[str, Any]:
    return p1187.compile_claim(
        {
            "claim_type": "conjunction",
            "gating": True,
            "values": [bool(claim["authorizes"]) for claim in claims],
        },
        read_json(EVIDENCE_CONTRACT),
    )


@torch.inference_mode()
def numerical_transform_case(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    seed: int,
    feature_kind: str,
    device: torch.device,
    broken_output: bool = False,
) -> dict[str, Any]:
    transformed = p1183.gauge_model(model, seed, device, broken_output=broken_output)
    original32, _ = p1181.fp32_state(model, panel.x, device)
    changed32, _ = p1181.fp32_state(transformed, panel.x, device)
    _, original64 = p1183.cpu_hidden_and_logits(model, panel.x)
    _, changed64 = p1183.cpu_hidden_and_logits(transformed, panel.x)
    feature_fn = (
        p1183.algebraic_internal_features
        if feature_kind == "endpoint_algebraic"
        else canonical_channel_signature
    )
    feature = np.asarray(feature_fn(model, panel.x), dtype=np.float64)
    changed_feature = np.asarray(feature_fn(transformed, panel.x), dtype=np.float64)
    feature_error = float(np.max(np.abs(feature - changed_feature)))
    fp32 = p1185.forward_metrics(
        original32.detach().cpu().double().numpy(),
        changed32.detach().cpu().double().numpy(),
        "fp32",
    )
    fp64 = p1185.forward_metrics(original64, changed64, "fp64")
    decision = p1186.exact_decision_metrics(original32, changed32, panel.y)
    claims = [
        typed_bounded(feature_error, THRESHOLDS["algebraic_feature_error_max"], "<="),
        typed_bounded(fp64["absolute_max"], fp64["mixed_absolute_bound"], "<="),
        typed_bounded(fp64["scaled_max"], p1185.THRESHOLDS["fp64_scaled_error_max"], "<="),
        typed_bounded(fp64["rms_relative"], p1185.THRESHOLDS["fp64_scaled_error_max"], "<="),
        typed_bounded(fp32["absolute_max"], fp32["mixed_absolute_bound"], "<="),
        typed_bounded(fp32["scaled_max"], p1185.THRESHOLDS["fp32_scaled_error_max"], "<="),
        typed_bounded(fp32["rms_relative"], p1185.THRESHOLDS["fp32_scaled_error_max"], "<="),
        typed_universal(decision["decision"]),
        typed_universal(decision["margin_sign"]),
    ]
    gate = typed_conjunction(claims)
    result = {
        "feature_kind": feature_kind,
        "seed": seed,
        "broken_output": broken_output,
        "feature_error": feature_error,
        "fp32": fp32,
        "fp64": fp64,
        "exact_decision": decision,
        "typed_claims": claims,
        "typed_gate": gate,
    }
    del transformed
    return result


def qualified(row: dict[str, Any]) -> bool:
    behavior = row["behavior"]
    return bool(
        behavior["all_logits_finite"] == 1.0
        and behavior["train_accuracy"] >= THRESHOLDS["train_accuracy_min"]
        and behavior["holdout_accuracy"] >= THRESHOLDS["holdout_accuracy_min"]
    )


def science_qualified(row: dict[str, Any]) -> bool:
    return bool(qualified(row) and row["numerical"]["typed_gate"]["authorizes"])


def build_confirmation_record(endpoint: Path, case_index: int, device: torch.device) -> dict[str, Any]:
    row = build_record(endpoint, device)
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    panel = panel_from_payload(payload)
    endpoint_model = load_model(payload, device)
    prefix_payload = torch.load(history_path(endpoint, HISTORY_STEPS[-1]), map_location="cpu", weights_only=False)
    prefix_model = load_model(prefix_payload, device)
    endpoint_numerical = numerical_transform_case(
        endpoint_model,
        panel,
        11885000 + case_index * 2,
        "endpoint_algebraic",
        device,
    )
    prefix_numerical = numerical_transform_case(
        prefix_model,
        panel,
        11885001 + case_index * 2,
        "prefix_canonical",
        device,
    )
    row["numerical"] = {
        "endpoint": endpoint_numerical,
        "prefix": prefix_numerical,
        "typed_gate": typed_conjunction(
            [endpoint_numerical["typed_gate"], prefix_numerical["typed_gate"]]
        ),
    }
    del endpoint_model, prefix_model
    torch.cuda.empty_cache()
    return row


def positive_sentinel(endpoint: Path, case_index: int, device: torch.device) -> dict[str, Any]:
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    panel = panel_from_payload(payload)
    endpoint_model = load_model(payload, device)
    prefix_payload = torch.load(history_path(endpoint, HISTORY_STEPS[-1]), map_location="cpu", weights_only=False)
    prefix_model = load_model(prefix_payload, device)
    endpoint_case = numerical_transform_case(
        endpoint_model,
        panel,
        11889000 + case_index * 2,
        "endpoint_algebraic",
        device,
        broken_output=True,
    )
    prefix_case = numerical_transform_case(
        prefix_model,
        panel,
        11889001 + case_index * 2,
        "prefix_canonical",
        device,
        broken_output=True,
    )
    strength = max(
        endpoint_case["feature_error"],
        prefix_case["feature_error"],
        endpoint_case["fp32"]["scaled_max"],
        prefix_case["fp32"]["scaled_max"],
    )
    claim = typed_bounded(strength, THRESHOLDS["positive_sentinel_error_min"], ">=")
    result = {
        "task_name": str(payload["task_name"]),
        "checkpoint": endpoint.name,
        "strength": strength,
        "typed_gate": claim,
        "endpoint": endpoint_case,
        "prefix": prefix_case,
    }
    del endpoint_model, prefix_model
    torch.cuda.empty_cache()
    return result


def rescue_task(
    task_name: str,
    records: list[dict[str, Any]],
    path_map: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    bundles: list[dict[str, Any]] = []
    for record in records:
        payload = torch.load(path_map[record["checkpoint"]], map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        panel = panel_from_payload(payload)
        bundle = p1182.rescue_bundle(model, panel, device)
        bundle.update({"record": record, "model": model, "panel": panel})
        bundles.append(bundle)
    behavior = np.stack([bundle["behavior_vector"] for bundle in bundles])
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    masks = future_masks(task_name)
    rows: list[dict[str, Any]] = []
    for recipient_index, recipient in enumerate(bundles):
        candidates = [index for index in range(len(bundles)) if index != recipient_index]
        candidates.sort(key=lambda index: float(np.linalg.norm(behavior[recipient_index] - behavior[index])))
        pool = candidates[:4]
        recipient_ordered = np.sort(recipient["calibration_response"])
        distances = {
            index: float(
                np.linalg.norm(np.sort(bundles[index]["calibration_response"]) - recipient_ordered)
            )
            for index in pool
        }
        correct_index = min(pool, key=lambda index: distances[index])
        wrong_index = max(pool, key=lambda index: distances[index])
        recipient_order = np.argsort(recipient["calibration_response"])
        recipient_rank = np.empty(len(recipient_order), dtype=np.int64)
        recipient_rank[recipient_order] = np.arange(len(recipient_order))
        injured_channels = np.argsort(np.abs(recipient["calibration_response"]))[
            -INJURY_CHANNEL_COUNT:
        ]
        baseline = p1182.evaluate_hybrid(
            recipient["q_evaluation"],
            recipient["weight"],
            recipient["targets_evaluation"],
            masks,
            device,
        )
        injured_q = recipient["q_evaluation"].clone()
        injured_q[:, injured_channels] = 0.0
        injured = p1182.evaluate_hybrid(
            injured_q, recipient["weight"], recipient["targets_evaluation"], masks, device
        )
        rescues: dict[str, Any] = {}
        for label, donor_index in (("correct", correct_index), ("wrong", wrong_index)):
            donor = bundles[donor_index]
            _, donor_hidden = p1181.fp32_state(donor["model"], recipient["panel"].x, device)
            donor_q = donor_hidden.square()[recipient["evaluation_mask"].to(device)].cpu()
            donor_order = np.argsort(donor["calibration_response"])
            hybrid_q = recipient["q_evaluation"].clone()
            hybrid_weight = recipient["weight"].clone()
            for recipient_channel in injured_channels:
                donor_channel = donor_order[recipient_rank[recipient_channel]]
                hybrid_q[:, recipient_channel] = donor_q[:, donor_channel]
                hybrid_weight[:, recipient_channel] = donor["weight"][:, donor_channel]
            evaluated = p1182.evaluate_hybrid(
                hybrid_q,
                hybrid_weight,
                recipient["targets_evaluation"],
                masks,
                device,
            )
            evaluated["future_response_error"] = p1182.response_error(
                evaluated["future_response"], baseline["future_response"]
            )
            evaluated["donor_replicate"] = donor["record"]["replicate"]
            evaluated["calibration_response_distance"] = distances[donor_index]
            rescues[label] = evaluated
        injured["future_response_error"] = p1182.response_error(
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


def confirm() -> None:
    protocol = validate_protocol()
    verify_training_seal(protocol)
    if SUMMARY_PATH.exists():
        raise RuntimeError("Phase1188 confirmation already exists")
    device = torch.device("cuda")
    endpoints = sorted(CHECKPOINT_ROOT.glob("*step10000.pt"))
    if len(endpoints) != TASK_COUNT * REPLICATES:
        raise RuntimeError("endpoint count mismatch")
    rows: list[dict[str, Any]] = []
    for index, endpoint in enumerate(endpoints):
        rows.append(build_confirmation_record(endpoint, index, device))
        print(canonical_json({"confirmed": index + 1, "total": len(endpoints)}), flush=True)
    write_jsonl(SYSTEM_ROWS, rows)
    first_by_task: dict[str, Path] = {}
    for endpoint in endpoints:
        payload = torch.load(endpoint, map_location="cpu", weights_only=False)
        first_by_task.setdefault(str(payload["task_name"]), endpoint)
    positive_rows = [
        positive_sentinel(path, index, device)
        for index, (task_name, path) in enumerate(sorted(first_by_task.items()))
    ]
    write_jsonl(POSITIVE_ROWS, positive_rows)

    passing_tasks = sorted(
        task_name
        for task_name in TASKS
        if sum(row["task_name"] == task_name and science_qualified(row) for row in rows)
        >= THRESHOLDS["qualified_system_count_per_task_min"]
    )
    behavior_pass = len(passing_tasks) >= THRESHOLDS["qualified_task_count_min"]
    numerical_pass = all(row["numerical"]["typed_gate"]["authorizes"] for row in rows)
    positive_pass = all(row["typed_gate"]["authorizes"] for row in positive_rows)
    science_rows = [
        row for row in rows if row["task_name"] in passing_tasks and science_qualified(row)
    ]
    if science_rows:
        camera = load_camera()
        endpoint_score = score_stage(science_rows, "endpoint", camera["endpoint"])
        prefix_score = score_stage(science_rows, "prefix", camera["prefix"])
        endpoint_score["gate_pass"] = camera_gate("endpoint", endpoint_score)
        prefix_score["gate_pass"] = camera_gate("prefix", prefix_score)
    else:
        endpoint_score = {"gate_pass": False, "status": "no_qualified_rows"}
        prefix_score = {"gate_pass": False, "status": "no_qualified_rows"}

    path_map = {path.name: path for path in endpoints}
    rescue_tasks = [
        rescue_task(
            task_name,
            [row for row in science_rows if row["task_name"] == task_name],
            path_map,
            device,
        )
        for task_name in passing_tasks
        if sum(row["task_name"] == task_name for row in science_rows)
        >= THRESHOLDS["qualified_system_count_per_task_min"]
    ]
    write_json(RESCUE_RAW, {"tasks": rescue_tasks})
    rescue = (
        p1182.rescue_summary(rescue_tasks, "confirmation", THRESHOLDS)
        if rescue_tasks
        else {"gate_pass": False, "status": "no_qualified_tasks"}
    )
    component_values = [
        behavior_pass,
        numerical_pass,
        positive_pass,
        bool(endpoint_score["gate_pass"]),
        bool(prefix_score["gate_pass"]),
        bool(rescue["gate_pass"]),
    ]
    primary_claim = p1187.compile_claim(
        {"claim_type": "conjunction", "gating": True, "values": component_values},
        read_json(EVIDENCE_CONTRACT),
    )
    summary = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(rows),
        "task_count": TASK_COUNT,
        "behavior": {
            "qualified_system_count": sum(qualified(row) for row in rows),
            "science_qualified_system_count": sum(science_qualified(row) for row in rows),
            "passing_tasks": passing_tasks,
            "passing_task_count": len(passing_tasks),
            "gate_pass": behavior_pass,
        },
        "numerical": {
            "typed_system_pass_count": sum(
                row["numerical"]["typed_gate"]["authorizes"] for row in rows
            ),
            "system_count": len(rows),
            "gate_pass": numerical_pass,
        },
        "positive_sentinel": {
            "pass_count": sum(row["typed_gate"]["authorizes"] for row in positive_rows),
            "count": len(positive_rows),
            "minimum_strength": min(row["strength"] for row in positive_rows),
            "gate_pass": positive_pass,
        },
        "endpoint": endpoint_score,
        "prefix": prefix_score,
        "rescue": rescue,
        "primary_claim_before_audit": primary_claim,
        "main_pass_before_audit": primary_claim["authorizes"],
        "independent_audit_status": "required_not_run",
        "rows_sha256": file_sha256(SYSTEM_ROWS),
        "positive_rows_sha256": file_sha256(POSITIVE_ROWS),
        "rescue_raw_sha256": file_sha256(RESCUE_RAW),
    }
    summary["summary_digest"] = digest(summary)
    write_json(SUMMARY_PATH, summary)
    print(canonical_json(summary))


def finalize() -> None:
    protocol = validate_protocol()
    verify_training_seal(protocol)
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1188 already finalized")
    summary = read_json(SUMMARY_PATH)
    if not AUDIT_PATH.exists():
        raise RuntimeError("independent audit must finish before finalization")
    audit = read_json(AUDIT_PATH)
    copied_audit = dict(audit)
    stored_audit_digest = copied_audit.pop("audit_digest")
    if digest(copied_audit) != stored_audit_digest:
        raise RuntimeError("independent audit digest mismatch")
    if audit["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("independent audit protocol mismatch")
    if audit["summary_digest"] != summary["summary_digest"]:
        raise RuntimeError("independent audit summary mismatch")
    main_pass = bool(summary["main_pass_before_audit"])
    audit_pass = bool(audit["audit_pass"])
    joint_pass = bool(main_pass and audit_pass)
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "audit_digest": stored_audit_digest,
        "scientific_status": (
            "narrow_three_evidence_mechanism_confirmed"
            if joint_pass
            else "terminal_three_evidence_confirmation_failed_or_unaudited"
        ),
        "main_pass": main_pass,
        "independent_audit_pass": audit_pass,
        "joint_pass": joint_pass,
        "k165_status": (
            "E3_KT_narrow_RoleSquare_affine_family"
            if joint_pass
            else "E1_KT_discovery_candidate_not_confirmed"
        ),
        "registry": "closed_after_one_formal_decision",
        "claim_scope": protocol["claim_scope"],
        "phase1189_microtransformer_transfer_authorized": joint_pass,
        "auto_continue": joint_pass,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "build-development-seal",
            "preregister",
            "train-and-seal",
            "confirm",
            "finalize",
        ),
    )
    args = parser.parse_args()
    commands = {
        "build-development-seal": build_development_seal,
        "preregister": preregister,
        "train-and-seal": train_and_seal,
        "confirm": confirm,
        "finalize": finalize,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
