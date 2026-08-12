"""Development-only probe for a continuous sparse component rescue.

This script is deliberately barred from Phase 1195 formal task seeds.  It
uses Phase 1194 development tasks to choose one non-negative, complexity-
penalized coalition solver before any discovery or confirmation outcomes
exist.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402


STAGE = 100
EPSILON = 0.25
LAMBDAS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
ITERATIONS = 400
SUPPORT_EPSILON = 1e-3


def coalition_patch(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    control_update: torch.Tensor,
    difference: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, np.ndarray, dict[str, float]]:
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
    component_patches: list[torch.Tensor] = []
    parameter_weights: list[float] = []
    for _, mask in groups:
        component = torch.where(mask, difference, torch.zeros_like(difference))
        probe = p1194.clone_model(parent)
        p1193.assign_parameters(
            probe, parent_vector + control_update + EPSILON * component
        )
        response = p1193.quotient_response(
            probe, inputs[calibration], targets[calibration], candidates
        )
        columns.append((response - control_q) / EPSILON)
        component_patches.append(component)
        parameter_weights.append(float(mask.float().mean().item()))
        del probe

    design = np.stack(columns, axis=1)
    target_norm_sq = max(float(np.dot(target, target)), 1e-12)
    spectral = float(np.linalg.norm(design, ord=2) ** 2 / target_norm_sq)
    step = 1.0 / max(spectral, 1e-9)
    weights = np.asarray(parameter_weights, dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)
    alpha = np.zeros(len(groups), dtype=np.float64)
    for _ in range(ITERATIONS):
        gradient = design.T @ (design @ alpha - target) / target_norm_sq
        alpha = np.clip(alpha - step * gradient - step * regularization * weights, 0.0, 1.0)

    patch = torch.zeros_like(difference)
    support_parameter_fraction = 0.0
    for coefficient, component, (_, mask) in zip(alpha, component_patches, groups):
        patch += float(coefficient) * component
        if coefficient > SUPPORT_EPSILON:
            support_parameter_fraction += float(mask.float().mean().item())
    fit = design @ alpha
    return patch, alpha, {
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


def evaluate_patch(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    control_update: torch.Tensor,
    patch: torch.Tensor,
    real_q: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    evaluation: torch.Tensor,
) -> tuple[float, float]:
    model = p1194.clone_model(parent)
    p1193.assign_parameters(model, parent_vector + control_update + patch)
    response = p1193.quotient_response(
        model, inputs[evaluation], targets[evaluation], candidates
    )
    error = float(np.linalg.norm(response - real_q))
    accuracy = p1193.behavior(model, inputs, targets, candidates)["accuracy"]
    del model
    return error, accuracy


def local_null_patches(
    parent: TinyCausalTransformer,
    difference: torch.Tensor,
    alpha: np.ndarray,
    correct_patch: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    groups = p1194.component_masks(parent)
    shifted = np.roll(alpha, len(alpha) // 2)
    wrong_component = torch.zeros_like(difference)
    support = torch.zeros_like(difference, dtype=torch.bool)
    for coefficient, (_, mask) in zip(shifted, groups):
        wrong_component += float(coefficient) * torch.where(
            mask, difference, torch.zeros_like(difference)
        )
    for coefficient, (_, mask) in zip(alpha, groups):
        if coefficient > SUPPORT_EPSILON:
            support |= mask
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


def one_case(
    task: dict[str, object], task_index: int, architecture: str, device: torch.device
) -> list[dict[str, object]]:
    seed = p1194.model_seed(task_index, architecture, 0, "development") + 119_500
    p1194.set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(p1194.ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (p1194.BATCH_SIZE,), generator=generator).to(device)
        for _ in range(STAGE + 1)
    ]
    for step in range(STAGE):
        p1193.training_step(
            model, optimizer, inputs[batches[step]], targets[batches[step]], candidates
        )

    parent = p1194.clone_model(model)
    parent_vector = p1193.flatten_parameters(parent)
    real_child, real_update, gradient, real_loss = p1194.one_step_update(
        model, optimizer, inputs, targets, candidates, batches[STAGE]
    )
    control_probe = p1194.clone_model(parent)
    control_update, control_metrics = p1193.select_control_update(
        control_probe,
        parent_vector,
        real_update,
        gradient,
        real_loss,
        inputs[batches[STAGE]],
        targets[batches[STAGE]],
        candidates,
        seed + 700,
    )
    difference = real_update - control_update
    real_q = p1193.quotient_response(
        real_child, inputs[evaluation], targets[evaluation], candidates
    )
    zero = torch.zeros_like(difference)
    control_error, _ = evaluate_patch(
        parent,
        parent_vector,
        control_update,
        zero,
        real_q,
        inputs,
        targets,
        candidates,
        evaluation,
    )
    rows: list[dict[str, object]] = []
    for regularization in LAMBDAS:
        patch, alpha, diagnostics = coalition_patch(
            parent,
            parent_vector,
            control_update,
            difference,
            inputs,
            targets,
            candidates,
            calibration,
            regularization,
        )
        error, accuracy = evaluate_patch(
            parent,
            parent_vector,
            control_update,
            patch,
            real_q,
            inputs,
            targets,
            candidates,
            evaluation,
        )
        null_recoveries = {}
        for null_name, null_patch in local_null_patches(
            parent, difference, alpha, patch, seed + int(regularization * 10_000) + 900
        ).items():
            null_error, _ = evaluate_patch(
                parent,
                parent_vector,
                control_update,
                null_patch,
                real_q,
                inputs,
                targets,
                candidates,
                evaluation,
            )
            null_recoveries[null_name] = (control_error - null_error) / max(
                control_error, 1e-12
            )
        recovery = (control_error - error) / max(control_error, 1e-12)
        conservative_null = max(null_recoveries.values())
        rows.append(
            {
                "task": task["name"],
                "family": task["family"],
                "architecture": architecture,
                "regularization": regularization,
                "control_error": control_error,
                "evaluation_recovery": recovery,
                "null_recoveries": null_recoveries,
                "null_recovery": conservative_null,
                "rescue_advantage": recovery - conservative_null,
                "evaluation_error": error,
                "accuracy": accuracy,
                "alpha": alpha.tolist(),
                "control_match": control_metrics,
                **diagnostics,
            }
        )
    del model, optimizer, parent, real_child, control_probe
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(p1194.DEVELOPMENT_TASKS):
        for architecture in p1194.ARCHITECTURES:
            rows.extend(one_case(task, task_index, architecture, device))
            print(json.dumps({"task": task["name"], "architecture": architecture}), flush=True)
    summaries = {}
    for regularization in LAMBDAS:
        selected = [row for row in rows if row["regularization"] == regularization]
        summaries[str(regularization)] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "evaluation_recovery",
                "null_recovery",
                "rescue_advantage",
                "calibration_cosine",
                "calibration_relative_error",
                "support_count",
                "support_parameter_fraction",
                "coefficient_l1",
                "patch_update_fraction",
            )
        }
        summaries[str(regularization)]["positive_fraction"] = float(
            np.mean([float(row["rescue_advantage"]) > 0 for row in selected])
        )
        summaries[str(regularization)]["by_architecture"] = {
            architecture: float(
                np.mean(
                    [
                        float(row["evaluation_recovery"])
                        for row in selected
                        if row["architecture"] == architecture
                    ]
                )
            )
            for architecture in p1194.ARCHITECTURES
        }
    output = {"phase": 1195, "kind": "development_probe", "rows": rows, "summary": summaries}
    path = ROOT / "tests/glm5_temp/phase1195_sparse_rescue_development_probe.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
