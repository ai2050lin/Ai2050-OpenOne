"""Continuous sparse coalition rescue for natural AdamW update forks.

Phase 1195 freezes one non-negative weighted L1 solver on development tasks.
At a matched wrong-update fork, the solver uses only a calibration panel to
select continuous weights over layer-attention and layer-MLP update-difference
groups.  Exact nonlinear recovery is then measured on a disjoint evaluation
panel against wrong-component, wrong-time, wrong-task, sign-reversed, and
same-support random controls.  No top-k search is permitted.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402


PHASE = 1195
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1195_continuous_sparse_coalition_rescue_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1195_continuous_sparse_coalition_rescue"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_ROW_ROOT = OUT_ROOT / "runs/formal/rows"
REPLAY_ROOT = OUT_ROOT / "runs/formal/replay_capsules"
TRAINING_SEAL = OUT_ROOT / "runs/formal/seal.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

ARCHITECTURES = p1194.ARCHITECTURES
RESCUE_STAGE = 100
WRONG_TIME_STAGE = 25
MAX_STEP = RESCUE_STAGE
BATCH_SIZE = 64
BASIS_EPSILON = 0.25
REGULARIZATION = 0.05
SOLVER_ITERATIONS = 400
SUPPORT_EPSILON = 1e-3
CONTROL_PLANES = 16
CONTROL_REFINE_PLANES = 4
CONTROL_COSINE_TARGET = 0.88
DEVELOPMENT_REPLICATES = 4
FORMAL_REPLICATES = 4

DEVELOPMENT_TASKS = p1194.DEVELOPMENT_TASKS
FORMAL_TASKS = (
    {"name": "disc_affine_00", "split": "discovery", "family": "affine", "task_seed": 119_701},
    {"name": "disc_affine_01", "split": "discovery", "family": "affine", "task_seed": 119_707},
    {"name": "disc_bitmix_00", "split": "discovery", "family": "bitmix", "task_seed": 119_713},
    {"name": "disc_bitmix_01", "split": "discovery", "family": "bitmix", "task_seed": 119_719},
    {"name": "disc_random_00", "split": "discovery", "family": "random", "task_seed": 119_727},
    {"name": "disc_random_01", "split": "discovery", "family": "random", "task_seed": 119_733},
    {"name": "conf_affine_00", "split": "confirmation", "family": "affine", "task_seed": 119_803},
    {"name": "conf_affine_01", "split": "confirmation", "family": "affine", "task_seed": 119_809},
    {"name": "conf_bitmix_00", "split": "confirmation", "family": "bitmix", "task_seed": 119_817},
    {"name": "conf_bitmix_01", "split": "confirmation", "family": "bitmix", "task_seed": 119_823},
    {"name": "conf_random_00", "split": "confirmation", "family": "random", "task_seed": 119_831},
    {"name": "conf_random_01", "split": "confirmation", "family": "random", "task_seed": 119_839},
)

CONTROL_THRESHOLDS = {
    "control_error_min": 1e-5,
    "eligible_fraction_min": 0.95,
    "support_parameter_fraction_max": 0.50,
    "patch_update_fraction_max": 0.55,
    "support_parameter_fraction_mean_max": 0.30,
    "patch_update_fraction_mean_max": 0.35,
    "loss_gap_max": 1e-5,
    "update_norm_relative_error_max": 1e-6,
    "endpoint_norm_relative_error_max": 1e-6,
    "first_order_relative_error_max": 1e-4,
    "update_cosine_max": 0.90,
    "orthogonal_fraction_min": 0.40,
}

RESCUE_THRESHOLDS = {
    "correct_recovery_mean_min": 0.12,
    "advantage_mean_min": 0.07,
    "positive_fraction_min": 0.75,
    "architecture_recovery_min": 0.09,
    "architecture_advantage_min": 0.04,
    "architecture_positive_fraction_min": 2.0 / 3.0,
    "family_advantage_min": 0.03,
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def model_seed(task_index: int, architecture: str, replicate: int, corpus: str) -> int:
    base = 1_195_900_000 if corpus == "development" else 1_195_000_000
    return base + task_index * 100_003 + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def one_step_update(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    batch_indices: torch.Tensor,
) -> tuple[TinyCausalTransformer, torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    parent_vector = p1193.flatten_parameters(model)
    child = p1194.clone_model(model)
    child_optimizer = p1193.optimizer_for(child)
    child_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    child_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = p1193.training_loss(
            child, inputs[batch_indices], targets[batch_indices], candidates
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite event loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(child.parameters(), p1193.GRADIENT_CLIP_NORM)
    gradient = p1193.flatten_gradients(child)
    child_optimizer.step()
    update = p1193.flatten_parameters(child) - parent_vector
    optimizer_state = copy.deepcopy(child_optimizer.state_dict())
    for state in optimizer_state["state"].values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()
    return child, update, gradient, float(loss.detach().item()), optimizer_state


@torch.no_grad()
def select_precise_control_update(
    probe: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    real_update: torch.Tensor,
    gradient: torch.Tensor,
    target_endpoint_loss: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Refine the Phase 1193 norm/first-order matched update on its sealed arc."""
    gradient_unit = gradient / gradient.norm().clamp_min(1e-12)
    basis = [gradient_unit]
    parent_orthogonal = parent_vector - torch.dot(parent_vector, gradient_unit) * gradient_unit
    if float(parent_orthogonal.norm().item()) > 1e-12:
        basis.append(parent_orthogonal / parent_orthogonal.norm())
    fixed = sum(torch.dot(real_update, vector) * vector for vector in basis)
    residual = real_update - fixed
    residual_norm = residual.norm()
    generator = torch.Generator(device=real_update.device).manual_seed(seed)
    random_residuals = []
    residual_unit = residual / residual_norm.clamp_min(1e-12)
    for _ in range(CONTROL_PLANES):
        random_direction = torch.randn(
            real_update.shape, generator=generator, device=real_update.device
        )
        for vector in basis:
            random_direction -= torch.dot(random_direction, vector) * vector
        random_direction -= torch.dot(random_direction, residual_unit) * residual_unit
        for prior in random_residuals:
            prior_unit = prior / prior.norm().clamp_min(1e-12)
            random_direction -= torch.dot(random_direction, prior_unit) * prior_unit
        random_direction /= random_direction.norm().clamp_min(1e-12)
        random_residuals.append(random_direction * residual_norm)

    fixed_norm_sq = float(torch.dot(fixed, fixed).item())
    residual_norm_sq = float(torch.dot(residual, residual).item())
    update_norm_sq = max(fixed_norm_sq + residual_norm_sq, 1e-12)
    residual_cosine_limit = (
        CONTROL_COSINE_TARGET * update_norm_sq - fixed_norm_sq
    ) / max(residual_norm_sq, 1e-12)
    residual_cosine_limit = float(np.clip(residual_cosine_limit, -1.0, 1.0))
    low = math.acos(residual_cosine_limit)
    high = 2.0 * math.pi - low

    def score(angle: float, random_residual: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        update = fixed + math.cos(angle) * residual + math.sin(angle) * random_residual
        p1193.assign_parameters(probe, parent_vector + update)
        loss = float(p1193.training_loss(probe, inputs, targets, candidates).item())
        return abs(loss - target_endpoint_loss), loss, update.clone()

    coarse_angles = np.linspace(low, high, 65)
    plane_bests = []
    for plane_index, random_residual in enumerate(random_residuals):
        scored = [
            (float(angle), *score(float(angle), random_residual))
            for angle in coarse_angles
        ]
        best = min(scored, key=lambda item: (item[1], item[0]))
        plane_bests.append((best[1], plane_index, best))
    refined = []
    for _, plane_index, best in sorted(plane_bests)[:CONTROL_REFINE_PLANES]:
        random_residual = random_residuals[plane_index]
        center = best[0]
        span = float(coarse_angles[1] - coarse_angles[0])
        for _ in range(7):
            angles = np.linspace(max(low, center - span), min(high, center + span), 17)
            scored = [
                (float(angle), *score(float(angle), random_residual)) for angle in angles
            ]
            best = min(scored, key=lambda item: (item[1], item[0]))
            center = best[0]
            span /= 8.0
        refined.append((best[1], plane_index, best))
    _, plane_index, best = min(refined, key=lambda item: (item[0], item[1], item[2][0]))
    angle, loss_gap, selected_loss, selected = best
    p1193.assign_parameters(probe, parent_vector)
    real_norm = float(real_update.norm().item())
    selected_norm = float(selected.norm().item())
    real_endpoint = float((parent_vector + real_update).norm().item())
    selected_endpoint = float((parent_vector + selected).norm().item())
    first_order_real = float(torch.dot(gradient, real_update).item())
    first_order_control = float(torch.dot(gradient, selected).item())
    return selected, {
        "loss_gap": loss_gap,
        "target_endpoint_loss": target_endpoint_loss,
        "selected_endpoint_loss": selected_loss,
        "angle": angle,
        "plane_index": plane_index,
        "control_planes": CONTROL_PLANES,
        "update_norm_relative_error": abs(selected_norm - real_norm) / max(real_norm, 1e-12),
        "endpoint_norm_relative_error": abs(selected_endpoint - real_endpoint)
        / max(real_endpoint, 1e-12),
        "first_order_relative_error": abs(first_order_control - first_order_real)
        / max(abs(first_order_real), 1e-12),
        "update_cosine": float(
            torch.dot(real_update, selected).item() / max(real_norm * selected_norm, 1e-12)
        ),
        "orthogonal_fraction": float(residual_norm.item()) / max(real_norm, 1e-12),
        "real_update_norm": real_norm,
        "endpoint_parameter_norm": real_endpoint,
    }


def solve_sparse_coalition(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    control_update: torch.Tensor,
    difference: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
) -> dict[str, Any]:
    groups = p1194.component_masks(parent)
    control = p1194.clone_model(parent)
    p1193.assign_parameters(control, parent_vector + control_update)
    control_q = p1193.quotient_response(
        control, inputs[calibration], targets[calibration], candidates
    )
    real = p1194.clone_model(parent)
    p1193.assign_parameters(real, parent_vector + control_update + difference)
    real_q = p1193.quotient_response(real, inputs[calibration], targets[calibration], candidates)
    target = real_q - control_q

    columns: list[np.ndarray] = []
    components: list[torch.Tensor] = []
    parameter_fractions: list[float] = []
    for _, mask in groups:
        component = torch.where(mask, difference, torch.zeros_like(difference))
        probe = p1194.clone_model(parent)
        p1193.assign_parameters(
            probe, parent_vector + control_update + BASIS_EPSILON * component
        )
        response = p1193.quotient_response(
            probe, inputs[calibration], targets[calibration], candidates
        )
        columns.append((response - control_q) / BASIS_EPSILON)
        components.append(component)
        parameter_fractions.append(float(mask.float().mean().item()))
        del probe

    design = np.stack(columns, axis=1)
    target_norm_sq = max(float(np.dot(target, target)), 1e-12)
    lipschitz = float(np.linalg.norm(design, ord=2) ** 2 / target_norm_sq)
    step = 1.0 / max(lipschitz, 1e-9)
    weights = np.asarray(parameter_fractions, dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)
    alpha = np.zeros(len(groups), dtype=np.float64)
    for _ in range(SOLVER_ITERATIONS):
        gradient = design.T @ (design @ alpha - target) / target_norm_sq
        alpha = np.clip(
            alpha - step * gradient - step * REGULARIZATION * weights, 0.0, 1.0
        )

    patch = torch.zeros_like(difference)
    support = torch.zeros_like(difference, dtype=torch.bool)
    support_parameter_fraction = 0.0
    for coefficient, component, (_, mask) in zip(alpha, components, groups):
        patch += float(coefficient) * component
        if coefficient > SUPPORT_EPSILON:
            support |= mask
            support_parameter_fraction += float(mask.float().mean().item())
    fit = design @ alpha
    del control, real
    return {
        "patch": patch,
        "alpha": alpha,
        "support": support,
        "group_names": [name for name, _ in groups],
        "calibration_cosine": p1194.cosine(fit, target),
        "calibration_relative_error": float(
            np.linalg.norm(fit - target) / max(np.linalg.norm(target), 1e-12)
        ),
        "support_count": int(np.sum(alpha > SUPPORT_EPSILON)),
        "support_parameter_fraction": support_parameter_fraction,
        "coefficient_l1": float(np.sum(alpha)),
        "coefficient_max": float(np.max(alpha)),
        "patch_update_fraction": float(patch.norm() / difference.norm().clamp_min(1e-12)),
    }


def local_nulls(
    parent: TinyCausalTransformer,
    difference: torch.Tensor,
    alpha: np.ndarray,
    support: torch.Tensor,
    correct_patch: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    groups = p1194.component_masks(parent)
    shifted = np.roll(alpha, len(alpha) // 2)
    wrong_component = torch.zeros_like(difference)
    for coefficient, (_, mask) in zip(shifted, groups):
        wrong_component += float(coefficient) * torch.where(
            mask, difference, torch.zeros_like(difference)
        )
    target_norm = correct_patch.norm()
    wrong_component = p1194.scaled_like(wrong_component, target_norm)
    generator = torch.Generator(device=difference.device).manual_seed(seed)
    random_patch = torch.zeros_like(difference)
    random_values = torch.randn(
        int(support.sum().item()), generator=generator, device=difference.device
    )
    random_patch[support] = p1194.scaled_like(random_values, target_norm)
    return {
        "wrong_component": wrong_component,
        "negative": -correct_patch,
        "random": random_patch,
    }


def build_material(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    batch_indices: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    parent = p1194.clone_model(model)
    parent_vector = p1193.flatten_parameters(parent)
    real_child, real_update, gradient, real_loss, child_optimizer_state = one_step_update(
        model, optimizer, inputs, targets, candidates, batch_indices
    )
    control_probe = p1194.clone_model(parent)
    target_endpoint_loss = float(
        p1193.training_loss(
            real_child, inputs[batch_indices], targets[batch_indices], candidates
        ).item()
    )
    control_update, control_metrics = select_precise_control_update(
        control_probe,
        parent_vector,
        real_update,
        gradient,
        target_endpoint_loss,
        inputs[batch_indices],
        targets[batch_indices],
        candidates,
        seed + 17,
    )
    difference = real_update - control_update
    solution = solve_sparse_coalition(
        parent,
        parent_vector,
        control_update,
        difference,
        inputs,
        targets,
        candidates,
        calibration,
    )
    nulls = local_nulls(
        parent,
        difference,
        solution["alpha"],
        solution["support"],
        solution["patch"],
        seed + 43,
    )
    payload = {
        "parent_state": {key: value.detach().cpu() for key, value in parent.state_dict().items()},
        "parent_vector": parent_vector.detach().cpu(),
        "child_optimizer_state": child_optimizer_state,
        "real_update": real_update.detach().cpu(),
        "control_update": control_update.detach().cpu(),
        "difference": difference.detach().cpu(),
        "correct_patch": solution["patch"].detach().cpu(),
        "wrong_component_patch": nulls["wrong_component"].detach().cpu(),
        "negative_patch": nulls["negative"].detach().cpu(),
        "random_patch": nulls["random"].detach().cpu(),
        "alpha": solution["alpha"].tolist(),
        "group_names": solution["group_names"],
        "support_count": solution["support_count"],
        "support_parameter_fraction": solution["support_parameter_fraction"],
        "coefficient_l1": solution["coefficient_l1"],
        "coefficient_max": solution["coefficient_max"],
        "patch_update_fraction": solution["patch_update_fraction"],
        "calibration_cosine": solution["calibration_cosine"],
        "calibration_relative_error": solution["calibration_relative_error"],
        "real_child_q": p1193.quotient_response(
            real_child, inputs[evaluation], targets[evaluation], candidates
        ),
        "real_child_output": p1194.output_signature(real_child, inputs[evaluation], candidates),
        "real_child_accuracy": p1193.behavior(real_child, inputs, targets, candidates)["accuracy"],
        "event_loss": real_loss,
        "control_metrics": control_metrics,
    }
    del parent, real_child, control_probe
    return payload


def trajectory(
    task: dict[str, Any],
    task_index: int,
    architecture: str,
    replicate: int,
    corpus: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = model_seed(task_index, architecture, replicate, corpus)
    set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=batch_generator).to(device)
        for _ in range(MAX_STEP + 1)
    ]
    stage_material: dict[int, dict[str, Any]] = {}
    for step in range(MAX_STEP + 1):
        if step in (WRONG_TIME_STAGE, RESCUE_STAGE):
            stage_material[step] = build_material(
                model,
                optimizer,
                inputs,
                targets,
                candidates,
                calibration,
                evaluation,
                batches[step],
                seed + step * 1009,
            )
        if step < MAX_STEP:
            p1193.training_step(
                model, optimizer, inputs[batches[step]], targets[batches[step]], candidates
            )
    payload = stage_material[RESCUE_STAGE]
    wrong_time = p1194.scaled_like(
        stage_material[WRONG_TIME_STAGE]["correct_patch"].to(device),
        payload["correct_patch"].norm().to(device),
    ).cpu()
    trajectory_id = f"{task['name']}::{architecture}::r{replicate}"
    payload.update(
        {
            "wrong_time_patch": wrong_time,
            "task": dict(task),
            "task_index": task_index,
            "architecture": architecture,
            "replicate": replicate,
            "trajectory_id": trajectory_id,
            "model_seed": seed,
        }
    )
    row = {
        "trajectory_id": trajectory_id,
        "event_id": f"{trajectory_id}::s{RESCUE_STAGE}",
        "task_name": task["name"],
        "task_index": task_index,
        "task_seed": task["task_seed"],
        "family": task["family"],
        "split": task.get("split", "development"),
        "architecture": architecture,
        "replicate": replicate,
        "model_seed": seed,
        "stage": RESCUE_STAGE,
        "event_loss": payload["event_loss"],
        "real_child_accuracy": payload["real_child_accuracy"],
        "control_match": payload["control_metrics"],
        "alpha": payload["alpha"],
        "group_names": payload["group_names"],
        "support_count": payload["support_count"],
        "support_parameter_fraction": payload["support_parameter_fraction"],
        "coefficient_l1": payload["coefficient_l1"],
        "coefficient_max": payload["coefficient_max"],
        "patch_update_fraction": payload["patch_update_fraction"],
        "calibration_cosine": payload["calibration_cosine"],
        "calibration_relative_error": payload["calibration_relative_error"],
    }
    del model, optimizer, inputs, targets, candidates, batches, stage_material
    gc.collect()
    torch.cuda.empty_cache()
    return row, payload


def variant_metrics(
    payload: dict[str, Any], patch: torch.Tensor, device: torch.device
) -> dict[str, float]:
    task = payload["task"]
    inputs, targets, candidates, _, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    model.load_state_dict(payload["parent_state"])
    parent_vector = payload["parent_vector"].to(device)
    update = payload["control_update"].to(device) + patch.to(device)
    p1193.assign_parameters(model, parent_vector + update)
    response = p1193.quotient_response(
        model, inputs[evaluation], targets[evaluation], candidates
    )
    signature = p1194.output_signature(model, inputs[evaluation], candidates)
    behavior = p1193.behavior(model, inputs, targets, candidates)
    metrics = {
        "response_error": float(np.linalg.norm(response - np.asarray(payload["real_child_q"]))),
        "output_error": float(np.linalg.norm(signature - np.asarray(payload["real_child_output"]))),
        "accuracy": behavior["accuracy"],
    }
    del model, inputs, targets, candidates
    return metrics


def control_match_pass(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["loss_gap"] <= CONTROL_THRESHOLDS["loss_gap_max"]
        and metrics["update_norm_relative_error"]
        <= CONTROL_THRESHOLDS["update_norm_relative_error_max"]
        and metrics["endpoint_norm_relative_error"]
        <= CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
        and metrics["first_order_relative_error"]
        <= CONTROL_THRESHOLDS["first_order_relative_error_max"]
        and metrics["update_cosine"] <= CONTROL_THRESHOLDS["update_cosine_max"]
        and metrics["orthogonal_fraction"] >= CONTROL_THRESHOLDS["orthogonal_fraction_min"]
    )


def attach_metrics(
    rows: list[dict[str, Any]], payloads: list[dict[str, Any]], device: torch.device
) -> None:
    by_trajectory = {payload["trajectory_id"]: payload for payload in payloads}
    by_cell = {
        (
            payload["task"].get("split", "development"),
            payload["architecture"],
            payload["replicate"],
            payload["task_index"],
        ): payload
        for payload in payloads
    }
    split_indices: dict[str, list[int]] = {}
    for payload in payloads:
        split_indices.setdefault(payload["task"].get("split", "development"), []).append(
            payload["task_index"]
        )
    split_indices = {key: sorted(set(value)) for key, value in split_indices.items()}
    for row in rows:
        payload = by_trajectory[row["trajectory_id"]]
        indices = split_indices[row["split"]]
        next_index = indices[(indices.index(row["task_index"]) + 1) % len(indices)]
        wrong_task_payload = by_cell[
            (row["split"], row["architecture"], row["replicate"], next_index)
        ]
        correct_norm = payload["correct_patch"].norm()
        wrong_task = p1194.scaled_like(wrong_task_payload["correct_patch"], correct_norm)
        payload["wrong_task_patch"] = wrong_task
        payload["wrong_task_trajectory_id"] = wrong_task_payload["trajectory_id"]
        zero = torch.zeros_like(payload["correct_patch"])
        variants = {
            "control": zero,
            "correct": payload["correct_patch"],
            "wrong_component": payload["wrong_component_patch"],
            "wrong_time": payload["wrong_time_patch"],
            "wrong_task": wrong_task,
            "negative": payload["negative_patch"],
            "random": payload["random_patch"],
        }
        measured = {name: variant_metrics(payload, patch, device) for name, patch in variants.items()}
        control_error = measured["control"]["response_error"]
        for metrics in measured.values():
            metrics["response_recovery"] = (
                control_error - metrics["response_error"]
            ) / max(control_error, 1e-12)
        null_names = ("wrong_component", "wrong_time", "wrong_task", "negative", "random")
        null_recovery = max(measured[name]["response_recovery"] for name in null_names)
        correct_recovery = measured["correct"]["response_recovery"]
        eligible = bool(
            control_error >= CONTROL_THRESHOLDS["control_error_min"]
            and row["support_parameter_fraction"]
            <= CONTROL_THRESHOLDS["support_parameter_fraction_max"]
            and row["patch_update_fraction"] <= CONTROL_THRESHOLDS["patch_update_fraction_max"]
            and control_match_pass(row["control_match"])
        )
        row.update(
            {
                "wrong_task_trajectory_id": wrong_task_payload["trajectory_id"],
                "rescue_variants": measured,
                "rescue_control_error": control_error,
                "rescue_correct_recovery": correct_recovery,
                "rescue_null_recovery": null_recovery,
                "rescue_advantage": correct_recovery - null_recovery,
                "rescue_eligible": eligible,
            }
        )


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def rescue_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    eligible = [row for row in rows if row["rescue_eligible"]]
    return {
        "count": len(rows),
        "eligible_count": len(eligible),
        "correct_recovery_mean": mean(eligible, "rescue_correct_recovery"),
        "null_recovery_mean": mean(eligible, "rescue_null_recovery"),
        "advantage_mean": mean(eligible, "rescue_advantage"),
        "positive_fraction": float(np.mean([row["rescue_advantage"] > 0 for row in eligible]))
        if eligible
        else 0.0,
        "support_count_mean": mean(eligible, "support_count"),
        "support_parameter_fraction_mean": mean(eligible, "support_parameter_fraction"),
        "patch_update_fraction_mean": mean(eligible, "patch_update_fraction"),
        "calibration_cosine_mean": mean(eligible, "calibration_cosine"),
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    overall = rescue_group(selected)
    overall["eligible_fraction"] = overall["eligible_count"] / max(overall["count"], 1)
    by_architecture = {
        architecture: rescue_group(
            [row for row in selected if row["architecture"] == architecture]
        )
        for architecture in ARCHITECTURES
    }
    by_family = {
        family: rescue_group([row for row in selected if row["family"] == family])
        for family in ("affine", "bitmix", "random")
    }
    gate = bool(
        overall["eligible_fraction"] >= CONTROL_THRESHOLDS["eligible_fraction_min"]
        and overall["support_parameter_fraction_mean"]
        <= CONTROL_THRESHOLDS["support_parameter_fraction_mean_max"]
        and overall["patch_update_fraction_mean"]
        <= CONTROL_THRESHOLDS["patch_update_fraction_mean_max"]
        and overall["correct_recovery_mean"]
        >= RESCUE_THRESHOLDS["correct_recovery_mean_min"]
        and overall["advantage_mean"] >= RESCUE_THRESHOLDS["advantage_mean_min"]
        and overall["positive_fraction"] >= RESCUE_THRESHOLDS["positive_fraction_min"]
        and all(
            group["correct_recovery_mean"]
            >= RESCUE_THRESHOLDS["architecture_recovery_min"]
            and group["advantage_mean"]
            >= RESCUE_THRESHOLDS["architecture_advantage_min"]
            and group["positive_fraction"]
            >= RESCUE_THRESHOLDS["architecture_positive_fraction_min"]
            for group in by_architecture.values()
        )
        and all(
            group["advantage_mean"] >= RESCUE_THRESHOLDS["family_advantage_min"]
            for group in by_family.values()
        )
    )
    return {
        "split": split,
        "row_count": len(selected),
        "trajectory_count": len({row["trajectory_id"] for row in selected}),
        "rescue": overall,
        "rescue_by_architecture": by_architecture,
        "rescue_by_family": by_family,
        "rescue_gate_pass": gate,
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "phase1195": SCRIPT,
        "phase1195_audit": AUDIT_SCRIPT,
        "phase1194": p1194.SCRIPT,
        "phase1193": p1193.SCRIPT,
        "phase1146_model": ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py",
        "phase1159_data": ROOT / "tests/glm5/phase1159_free_transformer_causal_use_external_validity.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def run_corpus(
    tasks: tuple[dict[str, Any], ...],
    replicates: int,
    corpus: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for architecture in ARCHITECTURES:
            for replicate in range(replicates):
                row, payload = trajectory(
                    task, task_index, architecture, replicate, corpus, device
                )
                rows.append(row)
                payloads.append(payload)
                print(
                    canonical_json(
                        {
                            "corpus": corpus,
                            "task": task["name"],
                            "architecture": architecture,
                            "replicate": replicate,
                            "rows": len(rows),
                        }
                    ),
                    flush=True,
                )
    attach_metrics(rows, payloads, device)
    if corpus == "formal":
        replay_ids = {
            "disc_affine_00::compact::r0",
            "disc_affine_00::deep::r0",
            "conf_affine_00::compact::r0",
            "conf_affine_00::deep::r0",
        }
        REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            if payload["trajectory_id"] in replay_ids:
                torch.save(payload, REPLAY_ROOT / f"{payload['trajectory_id'].replace('::', '__')}.pt")
    del payloads
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def develop() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = run_corpus(
        DEVELOPMENT_TASKS,
        DEVELOPMENT_REPLICATES,
        "development",
        torch.device("cuda"),
    )
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = summarize(rows, "development")
    summary.update(
        {
            "phase": PHASE,
            "kind": "development_only",
            "created_at": utc_now(),
            "source_hashes": source_hashes(),
            "formal_tasks_seen": False,
        }
    )
    write_json(DEVELOPMENT_SUMMARY, summary)
    print(canonical_json({"development_gate_pass": summary["rescue_gate_pass"]}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal protocol or outcomes already exist")
    development = read_json(DEVELOPMENT_SUMMARY)
    if not development["rescue_gate_pass"]:
        raise RuntimeError("development rescue gate failed")
    protocol = {
        "phase": PHASE,
        "created_at": utc_now(),
        "question": "Can one sealed continuous sparse coalition, fit only on a calibration panel, selectively rescue a matched wrong natural AdamW update on a disjoint evaluation panel?",
        "scope": "Synthetic 32-class TinyTransformer update forks; not a natural-language encoding claim.",
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "formal_tasks": list(FORMAL_TASKS),
        "formal_replicates": FORMAL_REPLICATES,
        "rescue_stage": RESCUE_STAGE,
        "wrong_time_stage": WRONG_TIME_STAGE,
        "batch_size": BATCH_SIZE,
        "solver": {
            "basis_epsilon": BASIS_EPSILON,
            "regularization": REGULARIZATION,
            "iterations": SOLVER_ITERATIONS,
            "coefficient_domain": "0 <= alpha_g <= 1",
            "objective": "normalized calibration linear-response error plus weighted L1 coefficient cost",
            "support_epsilon": SUPPORT_EPSILON,
        },
        "control_generator": {
            "planes": CONTROL_PLANES,
            "refined_planes": CONTROL_REFINE_PLANES,
            "angle_interval": "event-specific arc satisfying update cosine <= 0.88",
            "update_cosine_target": CONTROL_COSINE_TARGET,
            "invariants": ["update norm", "endpoint parameter norm", "first-order gradient action"],
        },
        "nulls": [
            "same_norm_shifted_component_weights",
            "same_norm_wrong_time_coalition",
            "same_norm_wrong_task_coalition",
            "sign_reversed_correct_coalition",
            "same_support_same_norm_random",
        ],
        "control_thresholds": CONTROL_THRESHOLDS,
        "rescue_thresholds": RESCUE_THRESHOLDS,
        "continuation_rule": "Fixed-optimizer-state continuation is authorized only if the immediate sparse-rescue gate independently passes in discovery and confirmation.",
        "forbidden": [
            "change lambda, epsilon, or solver iterations after formal outcomes",
            "replace the continuous solver by post-hoc top-k search",
            "drop an architecture or task family after outcomes",
            "relax recovery, advantage, positivity, or complexity thresholds",
            "call immediate response recovery behavior recovery or future-learning recovery",
            "call sparse rescue an endogenous mechanism used by the network",
        ],
        "upstream": {
            "phase1194_final_sha256": file_sha256(p1194.FINAL_PATH),
            "development_rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "development_summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
        },
        "source_hashes": source_hashes(),
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    stored = protocol["protocol_digest"]
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != stored:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if file_sha256(p1194.FINAL_PATH) != protocol["upstream"]["phase1194_final_sha256"]:
        raise RuntimeError("Phase1194 final changed")
    return protocol


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = run_corpus(FORMAL_TASKS, FORMAL_REPLICATES, "formal", torch.device("cuda"))
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_json(FORMAL_ROW_ROOT / f"{row['event_id'].replace('::', '__')}.json", row)
    write_jsonl(RAW_ROWS, rows)
    row_manifest = {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))}
    replay_manifest = {path.name: file_sha256(path) for path in sorted(REPLAY_ROOT.glob("*.pt"))}
    seal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "row_manifest": row_manifest,
        "replay_manifest": replay_manifest,
    }
    seal["seal_digest"] = digest(seal)
    write_json(TRAINING_SEAL, seal)
    print(canonical_json({"row_count": len(rows), "seal_digest": seal["seal_digest"]}))


def analyze() -> None:
    verify_protocol()
    seal = read_json(TRAINING_SEAL)
    rows = read_jsonl(RAW_ROWS)
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("formal rows hash mismatch")
    discovery = summarize(rows, "discovery")
    confirmation = summarize(rows, "confirmation")
    positive = discovery["rescue_gate_pass"] and confirmation["rescue_gate_pass"]
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "discovery": discovery,
        "confirmation": confirmation,
        "rescue_decision": "positive" if positive else "not_confirmed",
        "overall_status": (
            "continuous_sparse_coalition_rescue_confirmed"
            if positive
            else "continuous_sparse_coalition_rescue_not_confirmed"
        ),
    }
    claims = {
        "continuous_sparse_coalition_rescue": {
            "type": "E3-KT" if positive else "E3-KT-scope-boundary",
            "accepted": True,
            "claim": (
                "One sealed calibration-only continuous sparse coalition selectively rescues the immediate quotient response of matched wrong natural AdamW updates across both architectures, all task families, and independent splits."
                if positive
                else "The sealed continuous sparse coalition did not satisfy the full immediate-rescue and complexity gate across both independent splits; this does not establish absence of denser, path-level, or nonlinear rescue."
            ),
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"rescue": summary["rescue_decision"], "status": summary["overall_status"]}))


def replay_capsule(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    task = payload["task"]
    inputs, targets, candidates, calibration, _ = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    parent = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    solution = solve_sparse_coalition(
        parent,
        payload["parent_vector"].to(device),
        payload["control_update"].to(device),
        payload["difference"].to(device),
        inputs,
        targets,
        candidates,
        calibration,
    )
    measured = {
        name: variant_metrics(payload, patch, device)
        for name, patch in {
            "control": torch.zeros_like(payload["correct_patch"]),
            "correct": solution["patch"].cpu(),
            "wrong_component": payload["wrong_component_patch"],
            "wrong_time": payload["wrong_time_patch"],
            "wrong_task": payload["wrong_task_patch"],
            "negative": payload["negative_patch"],
            "random": payload["random_patch"],
        }.items()
    }
    return {
        "trajectory_id": payload["trajectory_id"],
        "alpha_max_error": float(
            np.max(np.abs(solution["alpha"] - np.asarray(payload["alpha"])))
        ),
        "patch_relative_error": float(
            (solution["patch"].cpu() - payload["correct_patch"]).norm()
            / payload["correct_patch"].norm().clamp_min(1e-12)
        ),
        "measured": measured,
    }


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    positive = summary["rescue_decision"] == "positive"
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["overall_status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "fixed_optimizer_state_continuation": positive,
            "self_consistent_optimizer_continuation": False,
            "natural_language_encoding_claim": False,
        },
        "scope": {
            "confirmed": "immediate disjoint-panel quotient-response rescue only if both formal gates passed",
            "not_claimed": [
                "behavioral necessity",
                "future learning recovery",
                "self-consistent optimizer-state repair",
                "endogenous rescue used by the network",
                "natural-language encoding mechanism",
            ],
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "run-formal", "analyze", "finalize"))
    command = parser.parse_args().command
    {
        "develop": develop,
        "preregister": preregister,
        "run-formal": run_formal,
        "analyze": analyze,
        "finalize": finalize,
    }[command]()


if __name__ == "__main__":
    main()
