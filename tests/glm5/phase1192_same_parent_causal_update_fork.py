from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402
import phase1190_natural_sgd_quotient_transition as p1190  # noqa: E402
import phase1191_prefix_future_formation_identity as p1191  # noqa: E402


PHASE = 1192
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1192_same_parent_causal_update_fork_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1192_same_parent_causal_update_fork"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_ROW_ROOT = OUT_ROOT / "runs/formal/rows"
PARENT_ROOT = OUT_ROOT / "runs/formal/parents"
TRAINING_SEAL = OUT_ROOT / "runs/formal/seal.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

PARENT_STEP = 4_000
HORIZON = 50
DEVELOPMENT_REPLICATES = 4
FORMAL_REPLICATES = 8
TASK_SELECTION_SEED = 1_192_001_711
FORMAL_MODEL_SEED_BASE = 1_192_000_000
DEVELOPMENT_MODEL_SEED_BASE = 1_192_900_000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1.0
ANGLE_GRID = tuple(float(value) for value in np.linspace(math.pi / 3.0, 5.0 * math.pi / 3.0, 65))

CONTROL_THRESHOLDS = {
    "loss_gap_max": 1e-5,
    "update_norm_relative_error_max": 1e-6,
    "endpoint_norm_relative_error_max": 1e-6,
    "first_order_relative_error_max": 1e-4,
    "update_cosine_max": 0.80,
    "orthogonal_fraction_min": 0.50,
    "immediate_effect_norm_min": 2e-5,
    "horizon_effect_norm_min": 2e-5,
    "immediate_train_accuracy_min": 0.99,
    "horizon_holdout_accuracy_min": 0.80,
    "eligible_fraction_min": 0.95,
}
POSITIVE_THRESHOLDS = {
    "immediate_true_cosine_mean_min": 0.30,
    "immediate_advantage_mean_min": 0.20,
    "immediate_positive_fraction_min": 0.65,
    "horizon_true_cosine_mean_min": 0.15,
    "horizon_advantage_mean_min": 0.15,
    "horizon_positive_fraction_min": 0.65,
    "positive_task_count_per_split_min": 3,
}
NEGATIVE_THRESHOLDS = {
    "horizon_advantage_mean_max": 0.05,
    "horizon_positive_fraction_max": 0.60,
    "positive_task_count_per_split_max": 2,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_operations() -> tuple[tuple[int, int, int], ...]:
    excluded = set(p1171.OPERATION_SAMPLE)
    excluded.update(tuple(value) for value in p1190.TASKS.values())
    available = [operation for operation in p1171.eligible_operations() if operation not in excluded]
    return tuple(random.Random(TASK_SELECTION_SEED).sample(available, 12))


OPERATION_SAMPLE = selected_operations()
DEVELOPMENT_OPERATIONS = OPERATION_SAMPLE[:4]
FORMAL_OPERATIONS = OPERATION_SAMPLE[4:]
DEVELOPMENT_TASKS = {
    f"causal_dev_{index:02d}_a{operation[0]}_b{operation[1]}_g{operation[2]}": operation
    for index, operation in enumerate(DEVELOPMENT_OPERATIONS)
}
FORMAL_TASKS = {
    f"causal_affine_{index:02d}_a{operation[0]}_b{operation[1]}_g{operation[2]}": operation
    for index, operation in enumerate(FORMAL_OPERATIONS)
}


def model_seed(task_index: int, replicate: int, corpus: str) -> int:
    base = DEVELOPMENT_MODEL_SEED_BASE if corpus == "development" else FORMAL_MODEL_SEED_BASE
    return base + task_index * 100_003 + replicate * 1_009


def parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().float().reshape(-1) for parameter in parameters(model)])


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().float().reshape(-1) for parameter in parameters(model)])


@torch.no_grad()
def assign_parameters(model: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters(model):
        count = parameter.numel()
        parameter.copy_(vector[offset : offset + count].view_as(parameter).to(parameter.dtype))
        offset += count
    if offset != vector.numel():
        raise RuntimeError("parameter vector length mismatch")


def optimizer_for(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def training_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x).float(), y)


def training_step(
    model: torch.nn.Module, optimizer: torch.optim.AdamW, x: torch.Tensor, y: torch.Tensor
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x).float(), y)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("nonfinite training loss")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def clone_model(parent_state: dict[str, torch.Tensor], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
    model.load_state_dict(parent_state)
    return model


@torch.no_grad()
def select_control_update(
    probe: p1171.RoleSquareNetwork,
    parent_vector: torch.Tensor,
    real_update: torch.Tensor,
    gradient: torch.Tensor,
    target_loss: float,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    gradient_unit = gradient / gradient.norm().clamp_min(1e-12)
    basis = [gradient_unit]
    parent_orthogonal = parent_vector - torch.dot(parent_vector, gradient_unit) * gradient_unit
    if float(parent_orthogonal.norm().item()) > 1e-12:
        basis.append(parent_orthogonal / parent_orthogonal.norm())
    fixed = sum(torch.dot(real_update, vector) * vector for vector in basis)
    residual = real_update - fixed
    residual_norm = residual.norm()
    generator = torch.Generator(device=real_update.device).manual_seed(seed)
    random_direction = torch.randn(real_update.shape, generator=generator, device=real_update.device)
    for vector in basis:
        random_direction -= torch.dot(random_direction, vector) * vector
    if float(residual_norm.item()) > 1e-12:
        residual_unit = residual / residual_norm
        random_direction -= torch.dot(random_direction, residual_unit) * residual_unit
    random_direction /= random_direction.norm().clamp_min(1e-12)
    random_residual = random_direction * residual_norm

    candidates: list[tuple[float, float, torch.Tensor]] = []
    for phi in ANGLE_GRID:
        update = fixed + math.cos(phi) * residual + math.sin(phi) * random_residual
        assign_parameters(probe, parent_vector + update)
        loss = float(training_loss(probe, x, y).item())
        candidates.append((abs(loss - target_loss), phi, update.clone()))
    loss_gap, phi, selected = min(candidates, key=lambda item: (item[0], item[1]))
    assign_parameters(probe, parent_vector)
    real_norm = float(real_update.norm().item())
    selected_norm = float(selected.norm().item())
    real_endpoint_norm = float((parent_vector + real_update).norm().item())
    selected_endpoint_norm = float((parent_vector + selected).norm().item())
    update_cosine = float(torch.dot(real_update, selected).item() / max(real_norm * selected_norm, 1e-12))
    first_order_real = float(torch.dot(gradient, real_update).item())
    first_order_control = float(torch.dot(gradient, selected).item())
    return selected, {
        "loss_gap": loss_gap,
        "phi": phi,
        "update_norm_relative_error": abs(selected_norm - real_norm) / max(real_norm, 1e-12),
        "endpoint_norm_relative_error": abs(selected_endpoint_norm - real_endpoint_norm)
        / max(real_endpoint_norm, 1e-12),
        "first_order_relative_error": abs(first_order_control - first_order_real)
        / max(abs(first_order_real), 1e-12),
        "update_cosine": update_cosine,
        "orthogonal_fraction": float(residual_norm.item()) / max(real_norm, 1e-12),
        "real_update_norm": real_norm,
        "endpoint_parameter_norm": real_endpoint_norm,
    }


def quotient_pair(
    model: p1171.RoleSquareNetwork,
    operation: tuple[int, int, int],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    panel = p1189.panel_from_payload({"operation": operation, "seed": seed})
    model.eval()
    return (
        p1189.response_unit_shape(model, panel, panel.train_mask, device),
        p1189.response_unit_shape(model, panel, panel.holdout_mask, device),
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def train_parent(
    task_name: str,
    task_index: int,
    operation: tuple[int, int, int],
    replicate: int,
    corpus: str,
    device: torch.device,
) -> dict[str, Any]:
    seed = model_seed(task_index, replicate, corpus)
    set_seed(seed)
    data = p1171.make_data(operation, seed + 17)
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
    optimizer = optimizer_for(model)
    x = data["train_x"].to(device)
    y = data["train_y"].to(device)
    for _ in range(PARENT_STEP):
        training_step(model, optimizer, x, y)
    capsule = {
        "phase": PHASE,
        "corpus": corpus,
        "task_name": task_name,
        "task_index": task_index,
        "operation": operation,
        "replicate": replicate,
        "seed": seed,
        "parent_step": PARENT_STEP,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "train_pair_digest": digest(data["train_x"].tolist()),
        "sealed_holdout_pair_digest": digest(data["holdout_x"].tolist()),
    }
    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return capsule


def run_from_capsule(capsule: dict[str, Any], device: torch.device) -> dict[str, Any]:
    operation = tuple(int(value) for value in capsule["operation"])
    seed = int(capsule["seed"])
    data = p1171.make_data(operation, seed + 17)
    x = data["train_x"].to(device)
    y = data["train_y"].to(device)
    model = clone_model(capsule["model_state"], device)
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(capsule["optimizer_state"])
    parent_vector = flatten_parameters(model)
    parent_holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parent_loss = F.cross_entropy(model(x).float(), y)
    parent_loss.backward()
    gradient = flatten_gradients(model)
    optimizer.step()
    real_update = flatten_parameters(model) - parent_vector
    post_step_optimizer_state = copy.deepcopy(optimizer.state_dict())
    real_loss = float(training_loss(model, x, y).item())

    probe = clone_model(capsule["model_state"], device)
    control_update, control_metrics = select_control_update(
        probe, parent_vector, real_update, gradient, real_loss, x, y, seed + 991
    )
    control_model = clone_model(capsule["model_state"], device)
    assign_parameters(control_model, parent_vector + control_update)
    control_loss = float(training_loss(control_model, x, y).item())
    control_optimizer = optimizer_for(control_model)
    control_optimizer.load_state_dict(post_step_optimizer_state)

    real_immediate_train = p1171.evaluate(model, data["train_x"], data["train_y"], device)
    control_immediate_train = p1171.evaluate(control_model, data["train_x"], data["train_y"], device)
    immediate_real = quotient_pair(model, operation, seed, device)
    immediate_control = quotient_pair(control_model, operation, seed, device)
    immediate_calibration = immediate_real[0] - immediate_control[0]
    immediate_evaluation = immediate_real[1] - immediate_control[1]

    for _ in range(HORIZON - 1):
        training_step(model, optimizer, x, y)
        training_step(control_model, control_optimizer, x, y)
    horizon_real = quotient_pair(model, operation, seed, device)
    horizon_control = quotient_pair(control_model, operation, seed, device)
    horizon_calibration = horizon_real[0] - horizon_control[0]
    horizon_evaluation = horizon_real[1] - horizon_control[1]
    real_holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
    control_holdout = p1171.evaluate(control_model, data["holdout_x"], data["holdout_y"], device)

    row = {
        "corpus": str(capsule["corpus"]),
        "task_name": str(capsule["task_name"]),
        "task_index": int(capsule["task_index"]),
        "operation": list(operation),
        "replicate": int(capsule["replicate"]),
        "seed": seed,
        "trajectory_id": f"{capsule['task_name']}/r{int(capsule['replicate'])}",
        "parent_step": PARENT_STEP,
        "horizon": HORIZON,
        "parent_loss": float(parent_loss.item()),
        "real_immediate_loss": real_loss,
        "control_immediate_loss": control_loss,
        "parent_holdout_accuracy": parent_holdout["accuracy"],
        "real_immediate_train_accuracy": real_immediate_train["accuracy"],
        "control_immediate_train_accuracy": control_immediate_train["accuracy"],
        "real_horizon_holdout_accuracy": real_holdout["accuracy"],
        "control_horizon_holdout_accuracy": control_holdout["accuracy"],
        **control_metrics,
        "immediate_calibration": immediate_calibration.tolist(),
        "immediate_evaluation": immediate_evaluation.tolist(),
        "immediate_calibration_norm": float(np.linalg.norm(immediate_calibration)),
        "immediate_evaluation_norm": float(np.linalg.norm(immediate_evaluation)),
        "immediate_true_cosine": cosine(immediate_calibration, immediate_evaluation),
        "horizon_calibration": horizon_calibration.tolist(),
        "horizon_evaluation": horizon_evaluation.tolist(),
        "horizon_calibration_norm": float(np.linalg.norm(horizon_calibration)),
        "horizon_evaluation_norm": float(np.linalg.norm(horizon_evaluation)),
        "horizon_true_cosine": cosine(horizon_calibration, horizon_evaluation),
    }
    row["control_qualified"] = bool(
        row["loss_gap"] <= CONTROL_THRESHOLDS["loss_gap_max"]
        and row["update_norm_relative_error"] <= CONTROL_THRESHOLDS["update_norm_relative_error_max"]
        and row["endpoint_norm_relative_error"] <= CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
        and row["first_order_relative_error"] <= CONTROL_THRESHOLDS["first_order_relative_error_max"]
        and row["update_cosine"] <= CONTROL_THRESHOLDS["update_cosine_max"]
        and row["orthogonal_fraction"] >= CONTROL_THRESHOLDS["orthogonal_fraction_min"]
        and row["immediate_calibration_norm"] >= CONTROL_THRESHOLDS["immediate_effect_norm_min"]
        and row["horizon_calibration_norm"] >= CONTROL_THRESHOLDS["horizon_effect_norm_min"]
        and min(row["real_immediate_train_accuracy"], row["control_immediate_train_accuracy"])
        >= CONTROL_THRESHOLDS["immediate_train_accuracy_min"]
        and min(row["real_horizon_holdout_accuracy"], row["control_horizon_holdout_accuracy"])
        >= CONTROL_THRESHOLDS["horizon_holdout_accuracy_min"]
    )
    del model, optimizer, control_model, control_optimizer, probe
    gc.collect()
    torch.cuda.empty_cache()
    return row


def add_nulls(rows: list[dict[str, Any]], replicates: int) -> None:
    lookup = {(row["task_name"], row["replicate"]): row for row in rows}
    for row in rows:
        null = lookup[(row["task_name"], (row["replicate"] + 1) % replicates)]
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            null_evaluation = np.asarray(null[horizon + "_evaluation"], dtype=np.float64)
            null_cosine = cosine(calibration, null_evaluation)
            row[horizon + "_null_cosine"] = null_cosine
            row[horizon + "_advantage"] = row[horizon + "_true_cosine"] - null_cosine
        row["null_trajectory_id"] = null["trajectory_id"]


def task_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for task in sorted({row["task_name"] for row in rows}):
        selected = [row for row in rows if row["task_name"] == task]
        advantage = float(np.mean([row["horizon_advantage"] for row in selected]))
        result.append({"task_name": task, "horizon_advantage_mean": advantage, "positive": advantage > 0.0})
    return result


def summarize(
    rows: list[dict[str, Any]], split: str, expected_systems: int, expected_tasks: int
) -> dict[str, Any]:
    selected = rows if split == "development" else [row for row in rows if row["split"] == split]
    tasks = task_summaries(selected)
    result = {
        "split": split,
        "system_count": len(selected),
        "task_count": len(tasks),
        "eligible_system_count": sum(bool(row["control_qualified"]) for row in selected),
        "eligible_fraction": float(np.mean([row["control_qualified"] for row in selected])),
        "loss_gap_max": max(row["loss_gap"] for row in selected),
        "update_norm_relative_error_max": max(row["update_norm_relative_error"] for row in selected),
        "endpoint_norm_relative_error_max": max(row["endpoint_norm_relative_error"] for row in selected),
        "first_order_relative_error_max": max(row["first_order_relative_error"] for row in selected),
        "update_cosine_max": max(row["update_cosine"] for row in selected),
        "update_cosine_mean": float(np.mean([row["update_cosine"] for row in selected])),
        "orthogonal_fraction_min": min(row["orthogonal_fraction"] for row in selected),
        "immediate_effect_norm_min": min(row["immediate_calibration_norm"] for row in selected),
        "horizon_effect_norm_min": min(row["horizon_calibration_norm"] for row in selected),
        "immediate_train_accuracy_min": min(
            min(row["real_immediate_train_accuracy"], row["control_immediate_train_accuracy"])
            for row in selected
        ),
        "horizon_holdout_accuracy_min": min(
            min(row["real_horizon_holdout_accuracy"], row["control_horizon_holdout_accuracy"])
            for row in selected
        ),
        "immediate_true_cosine_mean": float(np.mean([row["immediate_true_cosine"] for row in selected])),
        "immediate_null_cosine_mean": float(np.mean([row["immediate_null_cosine"] for row in selected])),
        "immediate_advantage_mean": float(np.mean([row["immediate_advantage"] for row in selected])),
        "immediate_positive_fraction": float(np.mean([row["immediate_advantage"] > 0 for row in selected])),
        "horizon_true_cosine_mean": float(np.mean([row["horizon_true_cosine"] for row in selected])),
        "horizon_null_cosine_mean": float(np.mean([row["horizon_null_cosine"] for row in selected])),
        "horizon_advantage_mean": float(np.mean([row["horizon_advantage"] for row in selected])),
        "horizon_positive_fraction": float(np.mean([row["horizon_advantage"] > 0 for row in selected])),
        "positive_task_count": sum(bool(task["positive"]) for task in tasks),
        "task_summaries": tasks,
    }
    result["control_gate_pass"] = bool(
        len(selected) == expected_systems
        and len(tasks) == expected_tasks
        and result["eligible_fraction"] >= CONTROL_THRESHOLDS["eligible_fraction_min"]
        and result["loss_gap_max"] <= CONTROL_THRESHOLDS["loss_gap_max"]
        and result["update_norm_relative_error_max"] <= CONTROL_THRESHOLDS["update_norm_relative_error_max"]
        and result["endpoint_norm_relative_error_max"] <= CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
        and result["first_order_relative_error_max"] <= CONTROL_THRESHOLDS["first_order_relative_error_max"]
        and result["update_cosine_max"] <= CONTROL_THRESHOLDS["update_cosine_max"]
        and result["orthogonal_fraction_min"] >= CONTROL_THRESHOLDS["orthogonal_fraction_min"]
        and result["immediate_effect_norm_min"] >= CONTROL_THRESHOLDS["immediate_effect_norm_min"]
        and result["horizon_effect_norm_min"] >= CONTROL_THRESHOLDS["horizon_effect_norm_min"]
        and result["immediate_train_accuracy_min"] >= CONTROL_THRESHOLDS["immediate_train_accuracy_min"]
        and result["horizon_holdout_accuracy_min"] >= CONTROL_THRESHOLDS["horizon_holdout_accuracy_min"]
    )
    result["positive_gate_pass"] = bool(
        result["control_gate_pass"]
        and result["immediate_true_cosine_mean"] >= POSITIVE_THRESHOLDS["immediate_true_cosine_mean_min"]
        and result["immediate_advantage_mean"] >= POSITIVE_THRESHOLDS["immediate_advantage_mean_min"]
        and result["immediate_positive_fraction"] >= POSITIVE_THRESHOLDS["immediate_positive_fraction_min"]
        and result["horizon_true_cosine_mean"] >= POSITIVE_THRESHOLDS["horizon_true_cosine_mean_min"]
        and result["horizon_advantage_mean"] >= POSITIVE_THRESHOLDS["horizon_advantage_mean_min"]
        and result["horizon_positive_fraction"] >= POSITIVE_THRESHOLDS["horizon_positive_fraction_min"]
        and result["positive_task_count"] >= POSITIVE_THRESHOLDS["positive_task_count_per_split_min"]
    )
    result["negative_boundary_pass"] = bool(
        result["control_gate_pass"]
        and result["horizon_advantage_mean"] <= NEGATIVE_THRESHOLDS["horizon_advantage_mean_max"]
        and result["horizon_positive_fraction"] <= NEGATIVE_THRESHOLDS["horizon_positive_fraction_max"]
        and result["positive_task_count"] <= NEGATIVE_THRESHOLDS["positive_task_count_per_split_max"]
    )
    return result


def source_hashes() -> dict[str, str]:
    paths = [
        SCRIPT,
        AUDIT_SCRIPT,
        Path(p1171.__file__),
        Path(p1187.__file__),
        Path(p1189.__file__),
        Path(p1190.__file__),
        Path(p1191.__file__),
    ]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def develop() -> None:
    if DEVELOPMENT_ROWS.exists():
        raise RuntimeError("development already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = []
    for task_index, (task_name, operation) in enumerate(DEVELOPMENT_TASKS.items()):
        for replicate in range(DEVELOPMENT_REPLICATES):
            capsule = train_parent(task_name, task_index, operation, replicate, "development", device)
            rows.append(run_from_capsule(capsule, device))
            print(canonical_json({"development": len(rows), "total": 16}), flush=True)
    add_nulls(rows, DEVELOPMENT_REPLICATES)
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = summarize(rows, "development", 16, 4)
    summary.update(
        {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "formal_data_read": False,
            "control_thresholds": CONTROL_THRESHOLDS,
            "positive_thresholds": POSITIVE_THRESHOLDS,
            "negative_thresholds": NEGATIVE_THRESHOLDS,
        }
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(DEVELOPMENT_SUMMARY, summary)
    if not summary["positive_gate_pass"]:
        raise RuntimeError("development did not authorize formal preregistration")


def preregister() -> None:
    development = read_json(DEVELOPMENT_SUMMARY)
    upstream = read_json(p1191.FINAL_PATH)
    if not development["positive_gate_pass"] or not upstream["main_gate_complete"]:
        raise RuntimeError("development or upstream authorization failed")
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    protocol = {
        "phase": PHASE,
        "title": "Same-parent matched causal update fork",
        "created_at_utc": utc_now(),
        "scientific_question": (
            "From an identical trained parameter and AdamW state, do a real update and a direction-rotated "
            "control update matched in norm, endpoint norm, first-order loss effect, immediate empirical loss, "
            "and immediate train behavior causally select different quotient-response transitions that transfer "
            "from a calibration panel to an unseen evaluation panel?"
        ),
        "upstream": {
            "phase1191_final_sha256": file_sha256(p1191.FINAL_PATH),
            "phase1191_final_digest": upstream["final_digest"],
            "development_rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "development_summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "development_summary_digest": development["summary_digest"],
        },
        "tasks": {name: list(operation) for name, operation in FORMAL_TASKS.items()},
        "task_selection_seed": TASK_SELECTION_SEED,
        "replicates": FORMAL_REPLICATES,
        "parent_step": PARENT_STEP,
        "horizon": HORIZON,
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "precision": "bfloat16 autocast with FP32 parameters",
            "batching": "full batch",
        },
        "fork": {
            "same_parent_parameters": True,
            "same_parent_optimizer_state": True,
            "real_arm": "one natural AdamW update",
            "control_arm": (
                "rotate only the component orthogonal to both the current gradient and parent parameter vector; "
                "a frozen angle grid chooses the closest immediate empirical-loss match"
            ),
            "post_fork_optimizer_state": "both arms receive the identical optimizer state after the real update",
            "matched_nuisances": [
                "update L2 norm",
                "endpoint parameter L2 norm",
                "first-order gradient dot update",
                "immediate full-batch loss",
                "immediate train accuracy",
            ],
        },
        "null": "same task and horizon, cyclic next replicate evaluation contrast",
        "control_thresholds": CONTROL_THRESHOLDS,
        "positive_thresholds": POSITIVE_THRESHOLDS,
        "negative_thresholds": NEGATIVE_THRESHOLDS,
        "source_hashes": source_hashes(),
        "evidence_contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "decision": {
            "positive": "both splits pass the frozen positive causal-transfer gate",
            "negative_boundary": "both splits pass the frozen negative boundary with valid controls",
            "ambiguous": "neither joint decision passes",
        },
        "hard_stops": [
            "No alternative parent step, horizon, angle grid, response object, threshold, or null is searched.",
            "Only a joint discovery and confirmation positive authorizes a tiny-Transformer bridge.",
            "A failure or ambiguity closes this RoleSquare causal-fork registry.",
            "No frozen pretrained-language-model formation claim is authorized by any outcome.",
        ],
    }
    protocol["protocol_digest"] = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    write_json(PROTOCOL_PATH, protocol)


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    expected = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    if protocol["protocol_digest"] != expected:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source code changed after preregistration")
    if file_sha256(DEVELOPMENT_ROWS) != protocol["upstream"]["development_rows_sha256"]:
        raise RuntimeError("development rows changed")
    if file_sha256(DEVELOPMENT_SUMMARY) != protocol["upstream"]["development_summary_sha256"]:
        raise RuntimeError("development summary changed")
    if file_sha256(p1191.FINAL_PATH) != protocol["upstream"]["phase1191_final_sha256"]:
        raise RuntimeError("Phase1191 final changed")
    return protocol


def capsule_path(task_name: str, replicate: int) -> Path:
    return PARENT_ROOT / f"{task_name}_r{replicate}.pt"


def row_path(task_name: str, replicate: int) -> Path:
    return FORMAL_ROW_ROOT / f"{task_name}_r{replicate}.json"


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal run already sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    PARENT_ROOT.mkdir(parents=True, exist_ok=True)
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    total = len(FORMAL_TASKS) * FORMAL_REPLICATES
    completed = 0
    for task_index, (task_name, operation) in enumerate(FORMAL_TASKS.items()):
        for replicate in range(FORMAL_REPLICATES):
            parent_file = capsule_path(task_name, replicate)
            formal_row_file = row_path(task_name, replicate)
            if not parent_file.exists():
                capsule = train_parent(task_name, task_index, operation, replicate, "formal", device)
                torch.save(capsule, parent_file)
            else:
                capsule = torch.load(parent_file, map_location=device, weights_only=False)
            if not formal_row_file.exists():
                row = run_from_capsule(capsule, device)
                row["split"] = "discovery" if task_index < 4 else "confirmation"
                write_json(formal_row_file, row)
            completed += 1
            print(canonical_json({"completed": completed, "total": total, "task": task_name, "replicate": replicate}), flush=True)
    rows = [read_json(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))]
    if len(rows) != total:
        raise RuntimeError("formal row count mismatch")
    add_nulls(rows, FORMAL_REPLICATES)
    write_jsonl(RAW_ROWS, rows)
    parent_manifest = {path.name: file_sha256(path) for path in sorted(PARENT_ROOT.glob("*.pt"))}
    raw_row_manifest = {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))}
    seal = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "parent_count": len(parent_manifest),
        "row_count": len(rows),
        "parent_manifest": parent_manifest,
        "parent_manifest_digest": digest(parent_manifest),
        "raw_row_manifest": raw_row_manifest,
        "raw_row_manifest_digest": digest(raw_row_manifest),
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "seal_digest": None,
    }
    seal["seal_digest"] = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    write_json(TRAINING_SEAL, seal)


def verify_seal() -> dict[str, Any]:
    seal = read_json(TRAINING_SEAL)
    expected = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    if seal["seal_digest"] != expected:
        raise RuntimeError("formal seal digest mismatch")
    parent_manifest = {path.name: file_sha256(path) for path in sorted(PARENT_ROOT.glob("*.pt"))}
    row_manifest = {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))}
    if parent_manifest != seal["parent_manifest"] or row_manifest != seal["raw_row_manifest"]:
        raise RuntimeError("formal manifest changed")
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("formal analysis rows changed")
    return seal


def bounded(value: float, threshold: float, comparator: str) -> dict[str, Any]:
    return {
        "claim_type": "bounded_float",
        "gating": True,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "dtype": "float64",
    }


def compile_claims(summary: dict[str, Any]) -> dict[str, Any]:
    contract = read_json(p1187.CONTRACT_PATH)
    families: dict[str, dict[str, dict[str, Any]]] = {"positive": {}, "negative": {}}
    for split in ("discovery", "confirmation"):
        current = summary[split]
        families["positive"][split + ".controls"] = bounded(current["eligible_fraction"], CONTROL_THRESHOLDS["eligible_fraction_min"], ">=")
        families["positive"][split + ".immediate_advantage"] = bounded(current["immediate_advantage_mean"], POSITIVE_THRESHOLDS["immediate_advantage_mean_min"], ">=")
        families["positive"][split + ".horizon_advantage"] = bounded(current["horizon_advantage_mean"], POSITIVE_THRESHOLDS["horizon_advantage_mean_min"], ">=")
        families["positive"][split + ".horizon_fraction"] = bounded(current["horizon_positive_fraction"], POSITIVE_THRESHOLDS["horizon_positive_fraction_min"], ">=")
        families["positive"][split + ".tasks"] = bounded(current["positive_task_count"], POSITIVE_THRESHOLDS["positive_task_count_per_split_min"], ">=")
        families["negative"][split + ".controls"] = bounded(current["eligible_fraction"], CONTROL_THRESHOLDS["eligible_fraction_min"], ">=")
        families["negative"][split + ".horizon_advantage"] = bounded(current["horizon_advantage_mean"], NEGATIVE_THRESHOLDS["horizon_advantage_mean_max"], "<=")
        families["negative"][split + ".horizon_fraction"] = bounded(current["horizon_positive_fraction"], NEGATIVE_THRESHOLDS["horizon_positive_fraction_max"], "<=")
        families["negative"][split + ".tasks"] = bounded(current["positive_task_count"], NEGATIVE_THRESHOLDS["positive_task_count_per_split_max"], "<=")
    result: dict[str, Any] = {}
    for family, raw in families.items():
        compiled = {name: p1187.compile_claim(claim, contract) for name, claim in raw.items()}
        conjunction = p1187.compile_claim(
            {
                "claim_type": "conjunction",
                "gating": True,
                "values": [bool(claim["authorizes"]) for claim in compiled.values()],
            },
            contract,
        )
        result[family] = {
            "raw": raw,
            "compiled": compiled,
            "conjunction": conjunction,
            "gate_pass": bool(conjunction["authorizes"]),
        }
    return result


def analyze() -> None:
    protocol = verify_protocol()
    seal = verify_seal()
    rows = read_jsonl(RAW_ROWS)
    discovery = summarize(rows, "discovery", 32, 4)
    confirmation = summarize(rows, "confirmation", 32, 4)
    positive = bool(discovery["positive_gate_pass"] and confirmation["positive_gate_pass"])
    negative = bool(discovery["negative_boundary_pass"] and confirmation["negative_boundary_pass"])
    decision = "positive" if positive else ("negative_boundary" if negative else "ambiguous")
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "training_seal_digest": seal["seal_digest"],
        "rows_sha256": file_sha256(RAW_ROWS),
        "discovery": discovery,
        "confirmation": confirmation,
        "positive_gate_pass": positive,
        "negative_boundary_pass": negative,
        "decision": decision,
        "summary_digest": None,
    }
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, compile_claims(summary))


def finalize() -> None:
    protocol = verify_protocol()
    verify_seal()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH) if AUDIT_PATH.exists() else {}
    claim_key = "negative" if summary["decision"] == "negative_boundary" else "positive"
    typed_match = bool(claims[claim_key]["gate_pass"])
    complete = bool(summary["decision"] != "ambiguous" and typed_match and audit.get("gate_pass"))
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": (
            "same_parent_causal_transition_confirmed"
            if complete and summary["decision"] == "positive"
            else (
                "same_parent_causal_transition_negative_boundary"
                if complete and summary["decision"] == "negative_boundary"
                else ("awaiting_independent_audit" if summary["decision"] != "ambiguous" and typed_match else "ambiguous_or_failed")
            )
        ),
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(CLAIMS_PATH),
        "audit_digest": audit.get("audit_digest"),
        "decision": summary["decision"],
        "independent_audit_pass": bool(audit.get("gate_pass")),
        "main_gate_complete": complete,
        "evidence_grade": "E3_KT_free_network_causal" if complete else "no_upgrade",
        "authorized_next": {
            "tiny_transformer_bridge_preregistration": bool(complete and summary["decision"] == "positive"),
            "frozen_pretrained_lm_formation_scan": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "A same-parent, short-horizon causal update-direction effect on the RoleSquare quotient-response "
            "state, conditional on the frozen matching controls and task family. It does not identify a natural "
            "optimizer mechanism component and does not establish Transformer or language formation dynamics."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(FINAL_PATH, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "run-formal", "analyze", "finalize"))
    args = parser.parse_args()
    {
        "develop": develop,
        "preregister": preregister,
        "run-formal": run_formal,
        "analyze": analyze,
        "finalize": finalize,
    }[args.command]()


if __name__ == "__main__":
    main()
