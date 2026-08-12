from __future__ import annotations

import copy
import json
import math
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
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402


PARENT_STEP = 4000
HORIZON = 50
REPLICATES = 4
OPERATIONS = ((17, 29, 11), (23, 7, 41))
LR = 0.001
WEIGHT_DECAY = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def training_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x).float(), y)


def training_step(
    model: torch.nn.Module, optimizer: torch.optim.AdamW, x: torch.Tensor, y: torch.Tensor
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x).float(), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def clone_model(parent_state: dict[str, torch.Tensor], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
    model.load_state_dict(parent_state)
    return model


@torch.no_grad()
def select_control_update(
    parent_model: p1171.RoleSquareNetwork,
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
    parallel = sum(torch.dot(real_update, vector) * vector for vector in basis)
    residual = real_update - parallel
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
    for phi in np.linspace(math.pi / 3.0, 5.0 * math.pi / 3.0, 65):
        update = parallel + math.cos(float(phi)) * residual + math.sin(float(phi)) * random_residual
        assign_parameters(parent_model, parent_vector + update)
        loss = float(training_loss(parent_model, x, y).item())
        candidates.append((abs(loss - target_loss), float(phi), update.clone()))
    loss_gap, phi, selected = min(candidates, key=lambda item: (item[0], item[1]))
    assign_parameters(parent_model, parent_vector)
    real_norm = float(real_update.norm().item())
    update_cosine = float(
        torch.dot(real_update, selected).item()
        / max(real_norm * float(selected.norm().item()), 1e-12)
    )
    first_order_real = float(torch.dot(gradient, real_update).item())
    first_order_control = float(torch.dot(gradient, selected).item())
    return selected, {
        "loss_gap": loss_gap,
        "phi": phi,
        "update_norm_relative_error": abs(float(selected.norm().item()) - real_norm) / max(real_norm, 1e-12),
        "endpoint_norm_relative_error": abs(
            float((parent_vector + selected).norm().item())
            - float((parent_vector + real_update).norm().item())
        )
        / max(float((parent_vector + real_update).norm().item()), 1e-12),
        "first_order_relative_error": abs(first_order_control - first_order_real)
        / max(abs(first_order_real), 1e-12),
        "update_cosine": update_cosine,
        "orthogonal_fraction": float(residual_norm.item()) / max(real_norm, 1e-12),
    }


def quotient_pair(
    model: p1171.RoleSquareNetwork,
    operation: tuple[int, int, int],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    payload = {"operation": operation, "seed": seed}
    panel = p1189.panel_from_payload(payload)
    model.eval()
    calibration = p1189.response_unit_shape(model, panel, panel.train_mask, device)
    evaluation = p1189.response_unit_shape(model, panel, panel.holdout_mask, device)
    return calibration, evaluation


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def run_one(task_index: int, operation: tuple[int, int, int], replicate: int, device: torch.device) -> dict[str, Any]:
    seed = 1_192_000_000 + task_index * 100_003 + replicate * 1_009
    set_seed(seed)
    data = p1171.make_data(operation, seed + 17)
    x = data["train_x"].to(device)
    y = data["train_y"].to(device)
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig()).to(device)
    optimizer = optimizer_for(model)
    for _ in range(PARENT_STEP):
        training_step(model, optimizer, x, y)

    parent_state = copy.deepcopy(model.state_dict())
    parent_vector = flatten_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parent_loss = F.cross_entropy(model(x).float(), y)
    parent_loss.backward()
    gradient = flatten_gradients(model)
    optimizer.step()
    real_update = flatten_parameters(model) - parent_vector
    post_step_optimizer_state = copy.deepcopy(optimizer.state_dict())
    real_loss = float(training_loss(model, x, y).item())

    control_probe = clone_model(parent_state, device)
    control_update, control_metrics = select_control_update(
        control_probe,
        parent_vector,
        real_update,
        gradient,
        real_loss,
        x,
        y,
        seed + 991,
    )
    control_model = clone_model(parent_state, device)
    assign_parameters(control_model, parent_vector + control_update)
    control_loss = float(training_loss(control_model, x, y).item())
    control_optimizer = optimizer_for(control_model)
    control_optimizer.load_state_dict(post_step_optimizer_state)

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
    result = {
        "task_index": task_index,
        "operation": list(operation),
        "replicate": replicate,
        "seed": seed,
        "parent_loss": float(parent_loss.item()),
        "real_immediate_loss": real_loss,
        "control_immediate_loss": control_loss,
        **control_metrics,
        "immediate_calibration": immediate_calibration.tolist(),
        "immediate_evaluation": immediate_evaluation.tolist(),
        "immediate_calibration_norm": float(np.linalg.norm(immediate_calibration)),
        "immediate_evaluation_norm": float(np.linalg.norm(immediate_evaluation)),
        "immediate_cosine": cosine(immediate_calibration, immediate_evaluation),
        "horizon_calibration": horizon_calibration.tolist(),
        "horizon_evaluation": horizon_evaluation.tolist(),
        "horizon_calibration_norm": float(np.linalg.norm(horizon_calibration)),
        "horizon_evaluation_norm": float(np.linalg.norm(horizon_evaluation)),
        "horizon_cosine": cosine(horizon_calibration, horizon_evaluation),
        "real_holdout_accuracy": real_holdout["accuracy"],
        "control_holdout_accuracy": control_holdout["accuracy"],
    }
    del model, optimizer, control_model, control_optimizer, control_probe
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    rows = [
        run_one(task_index, operation, replicate, device)
        for task_index, operation in enumerate(OPERATIONS)
        for replicate in range(REPLICATES)
    ]
    lookup = {(row["task_index"], row["replicate"]): row for row in rows}
    for row in rows:
        null = lookup[(row["task_index"], (row["replicate"] + 1) % REPLICATES)]
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            null_evaluation = np.asarray(null[horizon + "_evaluation"], dtype=np.float64)
            row[horizon + "_null_cosine"] = cosine(calibration, null_evaluation)
            row[horizon + "_advantage"] = row[horizon + "_cosine"] - row[horizon + "_null_cosine"]
    summary = {
        "count": len(rows),
        "loss_gap_max": max(row["loss_gap"] for row in rows),
        "norm_error_max": max(row["update_norm_relative_error"] for row in rows),
        "endpoint_norm_error_max": max(row["endpoint_norm_relative_error"] for row in rows),
        "first_order_error_max": max(row["first_order_relative_error"] for row in rows),
        "update_cosine_max": max(row["update_cosine"] for row in rows),
        "update_cosine_mean": float(np.mean([row["update_cosine"] for row in rows])),
        "orthogonal_fraction_min": min(row["orthogonal_fraction"] for row in rows),
        "immediate_effect_norm_min": min(row["immediate_calibration_norm"] for row in rows),
        "horizon_effect_norm_min": min(row["horizon_calibration_norm"] for row in rows),
        "immediate_true_cosine_mean": float(np.mean([row["immediate_cosine"] for row in rows])),
        "immediate_advantage_mean": float(np.mean([row["immediate_advantage"] for row in rows])),
        "horizon_true_cosine_mean": float(np.mean([row["horizon_cosine"] for row in rows])),
        "horizon_advantage_mean": float(np.mean([row["horizon_advantage"] for row in rows])),
        "horizon_positive_fraction": float(np.mean([row["horizon_advantage"] > 0 for row in rows])),
        "real_holdout_accuracy_min": min(row["real_holdout_accuracy"] for row in rows),
        "control_holdout_accuracy_min": min(row["control_holdout_accuracy"] for row in rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
