"""Pilot for prospective prediction of natural TinyTransformer formation events."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1159_free_transformer_causal_use_external_validity as p1159  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402


DEVICE = torch.device("cuda")
STAGES = (25, 100, 300)
MAX_STEP = 320
BATCH_SIZE = 64
ARCHITECTURE = "compact"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_permutation(kind: str, seed: int) -> torch.Tensor:
    if kind == "affine":
        a = 3 + 2 * (seed % 14)
        b = (seed * 7 + 3) % 32
        return torch.tensor([(a * value + b) % 32 for value in range(32)], dtype=torch.long)
    if kind == "bitmix":
        orders = ((4, 2, 0, 3, 1), (1, 4, 2, 0, 3), (3, 0, 4, 1, 2))
        order = orders[seed % len(orders)]
        mask = (seed * 11 + 5) % 32
        values = []
        for value in range(32):
            output = sum(((value >> source) & 1) << target for target, source in enumerate(order))
            values.append(output ^ mask)
        return torch.tensor(values, dtype=torch.long)
    generator = np.random.default_rng(seed)
    return torch.tensor(generator.permutation(32), dtype=torch.long)


def make_data(seed: int, kind: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lexicon = p1159.make_lexicon(seed + 17)
    inputs_cpu, base_targets = p1159.all_training_examples(lexicon)
    targets = task_permutation(kind, seed)[base_targets]
    mask_values = []
    for template in range(6):
        for context in range(2):
            for row in range(4):
                for col in range(4):
                    mask_values.append((template + context + row + col) % 2 == 0)
    calibration = torch.tensor(mask_values, dtype=torch.bool, device=DEVICE)
    return (
        lexicon,
        inputs_cpu.to(DEVICE),
        targets.to(DEVICE),
        p1159.answer_ids(lexicon, DEVICE),
        calibration,
        ~calibration,
    )


def clone_model(model: TinyCausalTransformer) -> TinyCausalTransformer:
    clone = TinyCausalTransformer(model.config).to(DEVICE)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def group_norms(model: TinyCausalTransformer, vector: torch.Tensor) -> list[float]:
    values: list[float] = []
    offset = 0
    chunks: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        chunk = vector[offset : offset + count]
        offset += count
        group = "other"
        for layer in range(model.config.layers):
            if name.startswith(f"blocks.{layer}.attn"):
                group = f"layer{layer}.attn"
            elif name.startswith(f"blocks.{layer}.mlp"):
                group = f"layer{layer}.mlp"
        chunks.setdefault(group, []).append(chunk)
    for layer in range(model.config.layers):
        for component in ("attn", "mlp"):
            combined = torch.cat(chunks[f"layer{layer}.{component}"])
            values.append(float(torch.log1p(combined.norm()).item()))
    return values


def component_masks(model: TinyCausalTransformer, device: torch.device) -> list[torch.Tensor]:
    names_and_slices = []
    offset = 0
    total = sum(parameter.numel() for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        names_and_slices.append((name, offset, offset + parameter.numel()))
        offset += parameter.numel()
    masks = []
    for layer in range(model.config.layers):
        for component in ("attn", "mlp"):
            mask = torch.zeros(total, dtype=torch.bool, device=device)
            prefix = f"blocks.{layer}.{component}"
            for name, start, stop in names_and_slices:
                if name.startswith(prefix):
                    mask[start:stop] = True
            masks.append(mask)
    return masks


def event(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    cal: torch.Tensor,
    eva: torch.Tensor,
    batch_indices: torch.Tensor,
    stage: int,
) -> dict[str, Any]:
    parent = clone_model(model)
    child = clone_model(model)
    child_optimizer = p1193.optimizer_for(child)
    child_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parent_vector = p1193.flatten_parameters(parent)
    child_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = p1193.training_loss(
            child, inputs[batch_indices], targets[batch_indices], candidates
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(child.parameters(), p1193.GRADIENT_CLIP_NORM)
    gradient = p1193.flatten_gradients(child)
    child_optimizer.step()
    update = p1193.flatten_parameters(child) - parent_vector
    parent_q = p1193.quotient_response(parent, inputs[cal], targets[cal], candidates)
    epsilon = 0.125
    tangent_probe = clone_model(parent)
    p1193.assign_parameters(tangent_probe, parent_vector + epsilon * update)
    tangent_q = p1193.quotient_response(
        tangent_probe, inputs[cal], targets[cal], candidates
    )
    tangent_prediction = (tangent_q - parent_q) / epsilon
    generator = torch.Generator(device=DEVICE).manual_seed(stage * 1009 + len(inputs))
    random_update = torch.randn(update.shape, generator=generator, device=DEVICE)
    random_update *= update.norm() / random_update.norm().clamp_min(1e-12)
    random_probe = clone_model(parent)
    p1193.assign_parameters(random_probe, parent_vector + epsilon * random_update)
    random_prediction = (
        p1193.quotient_response(random_probe, inputs[cal], targets[cal], candidates) - parent_q
    ) / epsilon
    gradient_update = -gradient * (update.norm() / gradient.norm().clamp_min(1e-12))
    gradient_probe = clone_model(parent)
    p1193.assign_parameters(gradient_probe, parent_vector + epsilon * gradient_update)
    gradient_prediction = (
        p1193.quotient_response(gradient_probe, inputs[cal], targets[cal], candidates) - parent_q
    ) / epsilon
    child_cal = p1193.quotient_response(child, inputs[cal], targets[cal], candidates)
    child_eval = p1193.quotient_response(child, inputs[eva], targets[eva], candidates)
    target_cal = child_cal - parent_q
    parent_eval = p1193.quotient_response(parent, inputs[eva], targets[eva], candidates)
    target_eval = child_eval - parent_eval
    rescue_metrics: dict[str, float | int] = {}
    if stage == STAGES[1]:
        real_loss = float(
            p1193.training_loss(child, inputs[batch_indices], targets[batch_indices], candidates).item()
        )
        control_probe = clone_model(parent)
        control_update, _ = p1193.select_control_update(
            control_probe,
            parent_vector,
            update,
            gradient,
            real_loss,
            inputs[batch_indices],
            targets[batch_indices],
            candidates,
            stage * 1009 + 17,
        )
        masks = component_masks(parent, DEVICE)
        scores = []
        for mask in masks:
            group_probe = clone_model(parent)
            group_update = torch.where(mask, update, torch.zeros_like(update))
            p1193.assign_parameters(
                group_probe, parent_vector + epsilon * group_update
            )
            group_tangent = (
                p1193.quotient_response(
                    group_probe, inputs[cal], targets[cal], candidates
                )
                - parent_q
            ) / epsilon
            scores.append(float(np.linalg.norm(group_tangent)))
        difference = update - control_update
        ranking = list(np.argsort(np.asarray(scores))[::-1])
        control_model = clone_model(parent)
        p1193.assign_parameters(control_model, parent_vector + control_update)
        control_response = p1193.quotient_response(
            control_model, inputs[eva], targets[eva], candidates
        )
        control_error = float(np.linalg.norm(control_response - child_eval))
        rescue_metrics = {"control_error": control_error}
        for top_k in (1, 2, 3):
            selected_indices = ranking[:top_k]
            wrong_indices = ranking[-top_k:]
            selected_mask = torch.stack([masks[index] for index in selected_indices]).any(dim=0)
            wrong_mask = torch.stack([masks[index] for index in wrong_indices]).any(dim=0)
            correct_patch = torch.where(selected_mask, difference, torch.zeros_like(difference))
            wrong_patch = torch.where(wrong_mask, difference, torch.zeros_like(difference))
            random_generator = torch.Generator(device=DEVICE).manual_seed(
                stage * 2027 + top_k * 101 + 31
            )
            random_patch = torch.zeros_like(difference)
            random_values = torch.randn(
                int(selected_mask.sum().item()), generator=random_generator, device=DEVICE
            )
            selected_norm = correct_patch.norm()
            random_values *= selected_norm / random_values.norm().clamp_min(1e-12)
            random_patch[selected_mask] = random_values
            variants = {
                "correct": control_update + correct_patch,
                "wrong_component": control_update + wrong_patch,
                "random": control_update + random_patch,
            }
            errors = {}
            for name, variant in variants.items():
                variant_model = clone_model(parent)
                p1193.assign_parameters(variant_model, parent_vector + variant)
                response = p1193.quotient_response(
                    variant_model, inputs[eva], targets[eva], candidates
                )
                errors[name] = float(np.linalg.norm(response - child_eval))
            prefix = f"top{top_k}_"
            rescue_metrics.update(
                {
                    prefix + "patch_update_fraction": float(
                        selected_norm / difference.norm().clamp_min(1e-12)
                    ),
                    prefix + "patch_parameter_fraction": float(
                        selected_mask.float().mean().item()
                    ),
                    prefix + "correct_improvement": control_error - errors["correct"],
                    prefix + "wrong_component_improvement": control_error
                    - errors["wrong_component"],
                    prefix + "random_improvement": control_error - errors["random"],
                }
            )
    behavior = p1193.behavior(parent, inputs, targets, candidates)
    features = np.asarray(
        parent_q.tolist()
        + group_norms(parent, update)
        + [
            float(stage == STAGES[0]),
            float(stage == STAGES[1]),
            float(stage == STAGES[2]),
            float(loss.detach().item()),
            float(torch.log1p(gradient.norm()).item()),
            float(torch.log1p(update.norm()).item()),
            behavior["accuracy"],
        ],
        dtype=np.float64,
    )
    return {
        "stage": stage,
        "features": features,
        "parent_q": parent_q,
        "tangent_prediction": tangent_prediction,
        "random_prediction": random_prediction,
        "gradient_prediction": gradient_prediction,
        "target_cal": target_cal,
        "target_eval": target_eval,
        "target_norm": float(np.linalg.norm(target_eval)),
        **rescue_metrics,
    }


def trajectory(task_index: int, kind: str, replicate: int) -> list[dict[str, Any]]:
    seed = 1_194_000 + task_index * 10_003 + replicate * 1_009
    set_seed(seed)
    _, inputs, targets, candidates, cal, eva = make_data(seed, kind)
    model = TinyCausalTransformer(p1193.ARCHITECTURES[ARCHITECTURE]).to(DEVICE)
    optimizer = p1193.optimizer_for(model)
    generator = torch.Generator(device="cpu").manual_seed(seed + 71)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=generator, device="cpu").to(DEVICE)
        for _ in range(MAX_STEP + 1)
    ]
    rows = []
    for step in range(MAX_STEP + 1):
        if step in STAGES:
            row = event(model, optimizer, inputs, targets, candidates, cal, eva, batches[step], step)
            row.update({"task_index": task_index, "kind": kind, "replicate": replicate})
            rows.append(row)
        if step < MAX_STEP:
            p1193.training_step(
                model,
                optimizer,
                inputs[batches[step]],
                targets[batches[step]],
                candidates,
            )
    return rows


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = (x - model["mean"]) / model["scale"]
    return np.column_stack([np.ones(len(x)), normalized]) @ model["weights"]


def main() -> None:
    tasks = ("affine", "bitmix", "random", "affine", "bitmix", "random")
    rows: list[dict[str, Any]] = []
    for task_index, kind in enumerate(tasks):
        for replicate in range(2):
            rows.extend(trajectory(task_index, kind, replicate))
            print({"task": task_index, "replicate": replicate, "events": len(rows)}, flush=True)
    train = [row for row in rows if row["task_index"] < 3]
    test = [row for row in rows if row["task_index"] >= 3]
    x_train = np.stack([row["features"] for row in train])
    y_train = np.stack([row["target_cal"] for row in train])
    x_test = np.stack([row["features"] for row in test])
    y_test = np.stack([row["target_eval"] for row in test])
    full = fit_ridge(x_train, y_train)
    prediction = predict(full, x_test)
    parent_dim = len(train[0]["parent_q"])
    parent_only = fit_ridge(x_train[:, :parent_dim], y_train)
    parent_prediction = predict(parent_only, x_test[:, :parent_dim])
    stage_means = {
        stage: np.mean([row["target_cal"] for row in train if row["stage"] == stage], axis=0)
        for stage in STAGES
    }
    true_cosines = [cosine(prediction[index], y_test[index]) for index in range(len(test))]
    parent_cosines = [cosine(parent_prediction[index], y_test[index]) for index in range(len(test))]
    stage_cosines = [cosine(stage_means[row["stage"]], y_test[index]) for index, row in enumerate(test)]
    advantages = [
        true_cosines[index] - max(parent_cosines[index], stage_cosines[index])
        for index in range(len(test))
    ]
    tangent_cosines = [cosine(row["tangent_prediction"], row["target_eval"]) for row in test]
    random_cosines = [cosine(row["random_prediction"], row["target_eval"]) for row in test]
    gradient_cosines = [cosine(row["gradient_prediction"], row["target_eval"]) for row in test]
    tangent_advantages = [
        tangent_cosines[index] - max(random_cosines[index], gradient_cosines[index])
        for index in range(len(test))
    ]
    rescue_rows = [row for row in test if row["stage"] == STAGES[1]]
    rescue_by_k = {}
    for top_k in (1, 2, 3):
        prefix = f"top{top_k}_"
        advantages_k = [
            row[prefix + "correct_improvement"]
            - max(
                row[prefix + "wrong_component_improvement"],
                row[prefix + "random_improvement"],
            )
            for row in rescue_rows
        ]
        rescue_by_k[top_k] = {
            "correct": float(
                np.mean([row[prefix + "correct_improvement"] for row in rescue_rows])
            ),
            "null": float(
                np.mean(
                    [
                        max(
                            row[prefix + "wrong_component_improvement"],
                            row[prefix + "random_improvement"],
                        )
                        for row in rescue_rows
                    ]
                )
            ),
            "advantage": float(np.mean(advantages_k)),
            "positive_fraction": float(np.mean(np.asarray(advantages_k) > 0)),
            "update_fraction": float(
                np.mean([row[prefix + "patch_update_fraction"] for row in rescue_rows])
            ),
            "parameter_fraction": float(
                np.mean([row[prefix + "patch_parameter_fraction"] for row in rescue_rows])
            ),
        }
    print(
        {
            "events": len(rows),
            "test_events": len(test),
            "target_norm_min": min(row["target_norm"] for row in rows),
            "true_cosine": float(np.mean(true_cosines)),
            "parent_only_cosine": float(np.mean(parent_cosines)),
            "stage_mean_cosine": float(np.mean(stage_cosines)),
            "conservative_advantage": float(np.mean(advantages)),
            "positive_fraction": float(np.mean(np.asarray(advantages) > 0)),
            "tangent_cosine": float(np.mean(tangent_cosines)),
            "random_tangent_cosine": float(np.mean(random_cosines)),
            "gradient_tangent_cosine": float(np.mean(gradient_cosines)),
            "tangent_advantage": float(np.mean(tangent_advantages)),
            "tangent_positive_fraction": float(np.mean(np.asarray(tangent_advantages) > 0)),
            "tangent_by_stage": {
                stage: {
                    "cosine": float(
                        np.mean(
                            [
                                tangent_cosines[index]
                                for index, row in enumerate(test)
                                if row["stage"] == stage
                            ]
                        )
                    ),
                    "advantage": float(
                        np.mean(
                            [
                                tangent_advantages[index]
                                for index, row in enumerate(test)
                                if row["stage"] == stage
                            ]
                        )
                    ),
                }
                for stage in STAGES
            },
            "rescue_by_k": rescue_by_k,
            "by_stage": {
                stage: float(
                    np.mean([true_cosines[index] for index, row in enumerate(test) if row["stage"] == stage])
                )
                for stage in STAGES
            },
        }
    )


if __name__ == "__main__":
    main()
