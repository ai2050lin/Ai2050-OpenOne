"""Natural-minibatch prospective tangent prediction and minimal rescue.

Phase 1194 keeps the Phase 1193 quotient-response camera fixed.  At three
training stages it computes the exact AdamW update from the parent state and
uses a sealed small directional probe on the calibration panel to predict the
full-step response on a disjoint evaluation panel.  A second, independent
gate asks whether one preregistered layer/component group can rescue a matched
wrong update.  Prediction and rescue are never pooled into one success gate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
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

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402
import phase1159_free_transformer_causal_use_external_validity as p1159  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402


PHASE = 1194
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1194_natural_minibatch_tangent_and_minimal_rescue_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1194_natural_minibatch_tangent_and_minimal_rescue"
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

ARCHITECTURES = p1193.ARCHITECTURES
STAGES = (25, 100, 300)
RESCUE_STAGE = 100
WRONG_TIME_STAGE = 25
MAX_STEP = 300
BATCH_SIZE = 64
TANGENT_EPSILON = 0.125
DEVELOPMENT_REPLICATES = 1
FORMAL_REPLICATES = 3

DEVELOPMENT_TASKS = (
    {"name": "dev_affine_00", "family": "affine", "task_seed": 119_401},
    {"name": "dev_affine_01", "family": "affine", "task_seed": 119_409},
    {"name": "dev_bitmix_00", "family": "bitmix", "task_seed": 119_417},
    {"name": "dev_bitmix_01", "family": "bitmix", "task_seed": 119_423},
    {"name": "dev_random_00", "family": "random", "task_seed": 119_431},
    {"name": "dev_random_01", "family": "random", "task_seed": 119_437},
)

FORMAL_TASKS = (
    {"name": "disc_affine_00", "split": "discovery", "family": "affine", "task_seed": 119_501},
    {"name": "disc_affine_01", "split": "discovery", "family": "affine", "task_seed": 119_507},
    {"name": "disc_bitmix_00", "split": "discovery", "family": "bitmix", "task_seed": 119_513},
    {"name": "disc_bitmix_01", "split": "discovery", "family": "bitmix", "task_seed": 119_519},
    {"name": "disc_random_00", "split": "discovery", "family": "random", "task_seed": 119_527},
    {"name": "disc_random_01", "split": "discovery", "family": "random", "task_seed": 119_533},
    {"name": "conf_affine_00", "split": "confirmation", "family": "affine", "task_seed": 119_603},
    {"name": "conf_affine_01", "split": "confirmation", "family": "affine", "task_seed": 119_609},
    {"name": "conf_bitmix_00", "split": "confirmation", "family": "bitmix", "task_seed": 119_617},
    {"name": "conf_bitmix_01", "split": "confirmation", "family": "bitmix", "task_seed": 119_623},
    {"name": "conf_random_00", "split": "confirmation", "family": "random", "task_seed": 119_631},
    {"name": "conf_random_01", "split": "confirmation", "family": "random", "task_seed": 119_639},
)

CONTROL_THRESHOLDS = {
    "target_norm_min": 2e-4,
    "prediction_eligible_fraction_min": 0.95,
    "rescue_control_error_min": 1e-5,
    "rescue_eligible_fraction_min": 0.95,
    "patch_parameter_fraction_max": 0.15,
    "patch_update_fraction_max": 0.50,
}

PREDICTION_THRESHOLDS = {
    "true_cosine_mean_min": 0.70,
    "advantage_mean_min": 0.10,
    "positive_fraction_min": 0.75,
    "architecture_advantage_min": 0.08,
    "stage_advantage_min": 0.02,
    "family_advantage_min": 0.05,
}

RESCUE_THRESHOLDS = {
    "correct_recovery_mean_min": 0.10,
    "advantage_mean_min": 0.05,
    "positive_fraction_min": 0.75,
    "architecture_advantage_min": 0.03,
    "architecture_positive_fraction_min": 2.0 / 3.0,
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
    base = 1_194_900_000 if corpus == "development" else 1_194_000_000
    return base + task_index * 100_003 + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def task_permutation(family: str, seed: int) -> torch.Tensor:
    if family == "affine":
        multiplier = 3 + 2 * (seed % 14)
        offset = (seed * 7 + 3) % 32
        return torch.tensor([(multiplier * value + offset) % 32 for value in range(32)], dtype=torch.long)
    if family == "bitmix":
        orders = ((4, 2, 0, 3, 1), (1, 4, 2, 0, 3), (3, 0, 4, 1, 2), (2, 0, 4, 3, 1))
        order = orders[seed % len(orders)]
        mask = (seed * 11 + 5) % 32
        values = []
        for value in range(32):
            output = sum(((value >> source) & 1) << target for target, source in enumerate(order))
            values.append(output ^ mask)
        return torch.tensor(values, dtype=torch.long)
    generator = np.random.default_rng(seed)
    return torch.tensor(generator.permutation(32), dtype=torch.long)


def make_data(task_seed: int, family: str, device: torch.device) -> tuple[torch.Tensor, ...]:
    lexicon = p1159.make_lexicon(task_seed + 17)
    inputs_cpu, base_targets = p1159.all_training_examples(lexicon)
    targets_cpu = task_permutation(family, task_seed)[base_targets]
    calibration_values = []
    for template in range(len(p1159.TEMPLATES)):
        for context in range(p1159.CONTEXTS):
            for row in range(p1159.ROWS):
                for col in range(p1159.COLS):
                    calibration_values.append((template + context + row + col) % 2 == 0)
    calibration = torch.tensor(calibration_values, dtype=torch.bool, device=device)
    return (
        inputs_cpu.to(device),
        targets_cpu.to(device),
        p1159.answer_ids(lexicon, device),
        calibration,
        ~calibration,
    )


def clone_model(model: TinyCausalTransformer) -> TinyCausalTransformer:
    clone = TinyCausalTransformer(model.config).to(next(model.parameters()).device)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def component_masks(model: TinyCausalTransformer) -> list[tuple[str, torch.Tensor]]:
    device = next(model.parameters()).device
    total = sum(parameter.numel() for parameter in model.parameters())
    slices: list[tuple[str, int, int]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        slices.append((name, offset, offset + parameter.numel()))
        offset += parameter.numel()
    groups = []
    for layer in range(model.config.layers):
        for component in ("attn", "mlp"):
            mask = torch.zeros(total, dtype=torch.bool, device=device)
            prefix = f"blocks.{layer}.{component}"
            for name, start, stop in slices:
                if name.startswith(prefix):
                    mask[start:stop] = True
            groups.append((f"layer{layer}.{component}", mask))
    return groups


@torch.inference_mode()
def output_signature(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    candidates: torch.Tensor,
) -> np.ndarray:
    logits = model(inputs)[:, -1].float().index_select(-1, candidates)
    logits = logits - logits.mean(dim=-1, keepdim=True)
    values = logits.detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(values))
    return values.reshape(-1) / max(norm, 1e-12)


def scaled_like(vector: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return vector * (target_norm / vector.norm().clamp_min(1e-12))


def one_step_update(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    batch_indices: torch.Tensor,
) -> tuple[TinyCausalTransformer, torch.Tensor, torch.Tensor, float]:
    parent_vector = p1193.flatten_parameters(model)
    child = clone_model(model)
    child_optimizer = p1193.optimizer_for(child)
    child_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    child_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = p1193.training_loss(child, inputs[batch_indices], targets[batch_indices], candidates)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite event loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(child.parameters(), p1193.GRADIENT_CLIP_NORM)
    gradient = p1193.flatten_gradients(child)
    child_optimizer.step()
    update = p1193.flatten_parameters(child) - parent_vector
    return child, update, gradient, float(loss.detach().item())


def direction_prediction(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    direction: torch.Tensor,
    parent_q: np.ndarray,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> np.ndarray:
    probe = clone_model(parent)
    p1193.assign_parameters(probe, parent_vector + TANGENT_EPSILON * direction)
    prediction = (
        p1193.quotient_response(probe, inputs, targets, candidates) - parent_q
    ) / TANGENT_EPSILON
    del probe
    return prediction


def build_rescue_material(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    real_update: torch.Tensor,
    gradient: torch.Tensor,
    real_loss: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
    batch_indices: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    probe = clone_model(parent)
    control_update, control_metrics = p1193.select_control_update(
        probe,
        parent_vector,
        real_update,
        gradient,
        real_loss,
        inputs[batch_indices],
        targets[batch_indices],
        candidates,
        seed,
    )
    parent_q = p1193.quotient_response(parent, inputs[calibration], targets[calibration], candidates)
    groups = component_masks(parent)
    scores = []
    for _, mask in groups:
        group_direction = torch.where(mask, real_update, torch.zeros_like(real_update))
        tangent = direction_prediction(
            parent,
            parent_vector,
            group_direction,
            parent_q,
            inputs[calibration],
            targets[calibration],
            candidates,
        )
        scores.append(float(np.linalg.norm(tangent)))
    ranking = list(np.argsort(np.asarray(scores))[::-1])
    selected_name, selected_mask = groups[ranking[0]]
    wrong_name, wrong_mask = groups[ranking[-1]]
    difference = real_update - control_update
    correct_patch = torch.where(selected_mask, difference, torch.zeros_like(difference))
    patch_norm = correct_patch.norm()
    wrong_patch = torch.where(wrong_mask, difference, torch.zeros_like(difference))
    wrong_patch = scaled_like(wrong_patch, patch_norm)
    generator = torch.Generator(device=parent_vector.device).manual_seed(seed + 43)
    random_values = torch.randn(int(selected_mask.sum().item()), generator=generator, device=parent_vector.device)
    random_values = scaled_like(random_values, patch_norm)
    random_patch = torch.zeros_like(difference)
    random_patch[selected_mask] = random_values
    return {
        "control_update": control_update.detach().cpu(),
        "correct_patch": correct_patch.detach().cpu(),
        "wrong_component_patch": wrong_patch.detach().cpu(),
        "random_patch": random_patch.detach().cpu(),
        "selected_group": selected_name,
        "wrong_group": wrong_name,
        "selected_score": scores[ranking[0]],
        "wrong_score": scores[ranking[-1]],
        "patch_parameter_fraction": float(selected_mask.float().mean().item()),
        "patch_update_fraction": float(patch_norm / difference.norm().clamp_min(1e-12)),
        "control_metrics": control_metrics,
    }


def event(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    batch_indices: torch.Tensor,
    stage: int,
    event_seed: int,
    need_rescue_material: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    parent = clone_model(model)
    parent_vector = p1193.flatten_parameters(parent)
    child, update, gradient, event_loss = one_step_update(
        model, optimizer, inputs, targets, candidates, batch_indices
    )
    parent_cal = p1193.quotient_response(parent, inputs[calibration], targets[calibration], candidates)
    parent_eval = p1193.quotient_response(parent, inputs[evaluation], targets[evaluation], candidates)
    child_cal = p1193.quotient_response(child, inputs[calibration], targets[calibration], candidates)
    child_eval = p1193.quotient_response(child, inputs[evaluation], targets[evaluation], candidates)
    target_cal = child_cal - parent_cal
    target_eval = child_eval - parent_eval
    tangent_prediction = direction_prediction(
        parent,
        parent_vector,
        update,
        parent_cal,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    generator = torch.Generator(device=parent_vector.device).manual_seed(event_seed + 19)
    random_direction = torch.randn(update.shape, generator=generator, device=parent_vector.device)
    random_direction = scaled_like(random_direction, update.norm())
    random_prediction = direction_prediction(
        parent,
        parent_vector,
        random_direction,
        parent_cal,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    gradient_direction = scaled_like(-gradient, update.norm())
    gradient_prediction = direction_prediction(
        parent,
        parent_vector,
        gradient_direction,
        parent_cal,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    tangent_cosine = cosine(tangent_prediction, target_eval)
    random_cosine = cosine(random_prediction, target_eval)
    gradient_cosine = cosine(gradient_prediction, target_eval)
    conservative_null = max(random_cosine, gradient_cosine)
    parent_behavior = p1193.behavior(parent, inputs, targets, candidates)
    child_behavior = p1193.behavior(child, inputs, targets, candidates)
    row = {
        "stage": stage,
        "event_loss": event_loss,
        "parent_accuracy": parent_behavior["accuracy"],
        "child_accuracy": child_behavior["accuracy"],
        "update_norm": float(update.norm().item()),
        "gradient_norm": float(gradient.norm().item()),
        "target_calibration": target_cal.tolist(),
        "target_evaluation": target_eval.tolist(),
        "tangent_prediction": tangent_prediction.tolist(),
        "random_prediction": random_prediction.tolist(),
        "gradient_prediction": gradient_prediction.tolist(),
        "target_norm": float(np.linalg.norm(target_eval)),
        "tangent_cosine": tangent_cosine,
        "random_cosine": random_cosine,
        "gradient_cosine": gradient_cosine,
        "conservative_null_cosine": conservative_null,
        "tangent_advantage": tangent_cosine - conservative_null,
        "prediction_eligible": bool(
            np.isfinite(target_eval).all()
            and np.isfinite(tangent_prediction).all()
            and np.linalg.norm(target_eval) >= CONTROL_THRESHOLDS["target_norm_min"]
        ),
    }
    payload = None
    if need_rescue_material:
        material = build_rescue_material(
            parent,
            parent_vector,
            update,
            gradient,
            event_loss,
            inputs,
            targets,
            candidates,
            calibration,
            batch_indices,
            event_seed + 71,
        )
        payload = {
            "parent_state": {key: value.detach().cpu() for key, value in parent.state_dict().items()},
            "parent_vector": parent_vector.detach().cpu(),
            "real_child_q": child_eval,
            "real_child_output": output_signature(child, inputs[evaluation], candidates),
            **material,
        }
        row.update(
            {
                "selected_group": material["selected_group"],
                "wrong_group": material["wrong_group"],
                "selected_group_score": material["selected_score"],
                "wrong_group_score": material["wrong_score"],
                "patch_parameter_fraction": material["patch_parameter_fraction"],
                "patch_update_fraction": material["patch_update_fraction"],
                "control_match": material["control_metrics"],
            }
        )
    del parent, child
    return row, payload


def variant_metrics(
    payload: dict[str, Any],
    patch_cpu: torch.Tensor,
    architecture: str,
    task_seed: int,
    family: str,
    device: torch.device,
) -> dict[str, float]:
    inputs, targets, candidates, _, evaluation = make_data(task_seed, family, device)
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    model.load_state_dict(payload["parent_state"])
    parent_vector = payload["parent_vector"].to(device)
    update = payload["control_update"].to(device) + patch_cpu.to(device)
    p1193.assign_parameters(model, parent_vector + update)
    response = p1193.quotient_response(model, inputs[evaluation], targets[evaluation], candidates)
    signature = output_signature(model, inputs[evaluation], candidates)
    behavior = p1193.behavior(model, inputs, targets, candidates)
    response_error = float(np.linalg.norm(response - np.asarray(payload["real_child_q"])))
    output_error = float(np.linalg.norm(signature - np.asarray(payload["real_child_output"])))
    del model, inputs, targets, candidates
    return {
        "response_error": response_error,
        "output_error": output_error,
        "accuracy": behavior["accuracy"],
    }


def trajectory(
    task: dict[str, Any],
    task_index: int,
    architecture: str,
    replicate: int,
    corpus: str,
    device: torch.device,
    save_replay: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = model_seed(task_index, architecture, replicate, corpus)
    set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=batch_generator).to(device)
        for _ in range(MAX_STEP + 1)
    ]
    rows = []
    stage_payloads: dict[int, dict[str, Any]] = {}
    for step in range(MAX_STEP + 1):
        if step in STAGES:
            if save_replay:
                capsule = {
                    "task": dict(task),
                    "task_index": task_index,
                    "architecture": architecture,
                    "replicate": replicate,
                    "corpus": corpus,
                    "seed": seed,
                    "stage": step,
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                    "batch_indices": batches[step].detach().cpu(),
                }
                path = REPLAY_ROOT / f"{task['name']}__{architecture}__r{replicate}__s{step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(capsule, path)
            row, payload = event(
                model,
                optimizer,
                inputs,
                targets,
                candidates,
                calibration,
                evaluation,
                batches[step],
                step,
                seed + step * 1009,
                step in (WRONG_TIME_STAGE, RESCUE_STAGE),
            )
            row.update(
                {
                    "trajectory_id": f"{task['name']}::{architecture}::r{replicate}",
                    "event_id": f"{task['name']}::{architecture}::r{replicate}::s{step}",
                    "task_name": task["name"],
                    "task_index": task_index,
                    "task_seed": task["task_seed"],
                    "family": task["family"],
                    "split": task.get("split", "development"),
                    "architecture": architecture,
                    "replicate": replicate,
                    "model_seed": seed,
                    "response_dimension": len(row["target_evaluation"]),
                }
            )
            rows.append(row)
            if payload is not None:
                stage_payloads[step] = payload
        if step < MAX_STEP:
            p1193.training_step(model, optimizer, inputs[batches[step]], targets[batches[step]], candidates)
    rescue_payload = stage_payloads[RESCUE_STAGE]
    wrong_time = stage_payloads[WRONG_TIME_STAGE]["correct_patch"]
    wrong_time = scaled_like(wrong_time, rescue_payload["correct_patch"].norm())
    rescue_payload["wrong_time_patch"] = wrong_time
    rescue_payload.update(
        {
            "task": dict(task),
            "task_index": task_index,
            "architecture": architecture,
            "replicate": replicate,
            "trajectory_id": f"{task['name']}::{architecture}::r{replicate}",
        }
    )
    del model, optimizer, inputs, targets, candidates, batches, stage_payloads
    gc.collect()
    torch.cuda.empty_cache()
    return rows, rescue_payload


def attach_rescue_metrics(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    device: torch.device,
) -> None:
    payload_lookup = {payload["trajectory_id"]: payload for payload in payloads}
    by_cell = {
        (
            payload["task"].get("split", "development"),
            payload["architecture"],
            payload["replicate"],
            payload["task_index"],
        ): payload
        for payload in payloads
    }
    split_tasks: dict[str, list[int]] = {}
    for payload in payloads:
        split = payload["task"].get("split", "development")
        split_tasks.setdefault(split, []).append(payload["task_index"])
    split_tasks = {key: sorted(set(value)) for key, value in split_tasks.items()}
    for row in rows:
        if row["stage"] != RESCUE_STAGE:
            continue
        payload = payload_lookup[row["trajectory_id"]]
        split = row["split"]
        indices = split_tasks[split]
        next_task = indices[(indices.index(row["task_index"]) + 1) % len(indices)]
        wrong_task_payload = by_cell[(split, row["architecture"], row["replicate"], next_task)]
        wrong_task_patch = scaled_like(
            wrong_task_payload["correct_patch"], payload["correct_patch"].norm()
        )
        zero = torch.zeros_like(payload["correct_patch"])
        variants = {
            "control": zero,
            "correct": payload["correct_patch"],
            "wrong_component": payload["wrong_component_patch"],
            "wrong_time": payload["wrong_time_patch"],
            "wrong_task": wrong_task_patch,
            "random": payload["random_patch"],
        }
        measured = {
            name: variant_metrics(
                payload,
                patch,
                row["architecture"],
                row["task_seed"],
                row["family"],
                device,
            )
            for name, patch in variants.items()
        }
        control_error = measured["control"]["response_error"]
        for name in measured:
            measured[name]["response_recovery"] = (
                control_error - measured[name]["response_error"]
            ) / max(control_error, 1e-12)
        null_names = ("wrong_component", "wrong_time", "wrong_task", "random")
        null_recovery = max(measured[name]["response_recovery"] for name in null_names)
        correct_recovery = measured["correct"]["response_recovery"]
        row.update(
            {
                "wrong_task_trajectory_id": wrong_task_payload["trajectory_id"],
                "rescue_variants": measured,
                "rescue_control_error": control_error,
                "rescue_correct_recovery": correct_recovery,
                "rescue_null_recovery": null_recovery,
                "rescue_advantage": correct_recovery - null_recovery,
                "rescue_eligible": bool(
                    control_error >= CONTROL_THRESHOLDS["rescue_control_error_min"]
                    and row["patch_parameter_fraction"]
                    <= CONTROL_THRESHOLDS["patch_parameter_fraction_max"]
                    and row["patch_update_fraction"]
                    <= CONTROL_THRESHOLDS["patch_update_fraction_max"]
                ),
            }
        )


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def prediction_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    eligible = [row for row in rows if row["prediction_eligible"]]
    return {
        "count": len(rows),
        "eligible_count": len(eligible),
        "true_cosine_mean": mean(eligible, "tangent_cosine"),
        "random_cosine_mean": mean(eligible, "random_cosine"),
        "gradient_cosine_mean": mean(eligible, "gradient_cosine"),
        "null_cosine_mean": mean(eligible, "conservative_null_cosine"),
        "advantage_mean": mean(eligible, "tangent_advantage"),
        "positive_fraction": float(np.mean([row["tangent_advantage"] > 0 for row in eligible]))
        if eligible
        else 0.0,
        "target_norm_min": min((float(row["target_norm"]) for row in rows), default=0.0),
    }


def rescue_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    eligible = [row for row in rows if row.get("rescue_eligible", False)]
    return {
        "count": len(rows),
        "eligible_count": len(eligible),
        "correct_recovery_mean": mean(eligible, "rescue_correct_recovery"),
        "null_recovery_mean": mean(eligible, "rescue_null_recovery"),
        "advantage_mean": mean(eligible, "rescue_advantage"),
        "positive_fraction": float(np.mean([row["rescue_advantage"] > 0 for row in eligible]))
        if eligible
        else 0.0,
        "patch_parameter_fraction_mean": mean(eligible, "patch_parameter_fraction"),
        "patch_update_fraction_mean": mean(eligible, "patch_update_fraction"),
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    prediction = prediction_group(selected)
    prediction["eligible_fraction"] = prediction["eligible_count"] / max(prediction["count"], 1)
    by_architecture = {
        architecture: prediction_group([row for row in selected if row["architecture"] == architecture])
        for architecture in ARCHITECTURES
    }
    by_stage = {
        str(stage): prediction_group([row for row in selected if row["stage"] == stage])
        for stage in STAGES
    }
    by_family = {
        family: prediction_group([row for row in selected if row["family"] == family])
        for family in ("affine", "bitmix", "random")
    }
    prediction_gate = bool(
        prediction["eligible_fraction"] >= CONTROL_THRESHOLDS["prediction_eligible_fraction_min"]
        and prediction["true_cosine_mean"] >= PREDICTION_THRESHOLDS["true_cosine_mean_min"]
        and prediction["advantage_mean"] >= PREDICTION_THRESHOLDS["advantage_mean_min"]
        and prediction["positive_fraction"] >= PREDICTION_THRESHOLDS["positive_fraction_min"]
        and all(
            group["advantage_mean"] >= PREDICTION_THRESHOLDS["architecture_advantage_min"]
            for group in by_architecture.values()
        )
        and all(
            group["advantage_mean"] >= PREDICTION_THRESHOLDS["stage_advantage_min"]
            for group in by_stage.values()
        )
        and all(
            group["advantage_mean"] >= PREDICTION_THRESHOLDS["family_advantage_min"]
            for group in by_family.values()
        )
    )
    rescue_rows = [row for row in selected if row["stage"] == RESCUE_STAGE]
    rescue = rescue_group(rescue_rows)
    rescue["eligible_fraction"] = rescue["eligible_count"] / max(rescue["count"], 1)
    rescue_by_architecture = {
        architecture: rescue_group(
            [row for row in rescue_rows if row["architecture"] == architecture]
        )
        for architecture in ARCHITECTURES
    }
    rescue_gate = bool(
        rescue["eligible_fraction"] >= CONTROL_THRESHOLDS["rescue_eligible_fraction_min"]
        and rescue["correct_recovery_mean"] >= RESCUE_THRESHOLDS["correct_recovery_mean_min"]
        and rescue["advantage_mean"] >= RESCUE_THRESHOLDS["advantage_mean_min"]
        and rescue["positive_fraction"] >= RESCUE_THRESHOLDS["positive_fraction_min"]
        and all(
            group["advantage_mean"] >= RESCUE_THRESHOLDS["architecture_advantage_min"]
            and group["positive_fraction"]
            >= RESCUE_THRESHOLDS["architecture_positive_fraction_min"]
            for group in rescue_by_architecture.values()
        )
    )
    return {
        "split": split,
        "row_count": len(selected),
        "trajectory_count": len({row["trajectory_id"] for row in selected}),
        "prediction": prediction,
        "prediction_by_architecture": by_architecture,
        "prediction_by_stage": by_stage,
        "prediction_by_family": by_family,
        "prediction_gate_pass": prediction_gate,
        "rescue": rescue,
        "rescue_by_architecture": rescue_by_architecture,
        "rescue_gate_pass": rescue_gate,
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "phase1194": SCRIPT,
        "phase1194_audit": AUDIT_SCRIPT,
        "phase1193_camera": p1193.SCRIPT,
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
                save_replay = bool(
                    corpus == "formal"
                    and replicate == 0
                    and task["name"] in ("disc_affine_00", "conf_affine_00")
                )
                trajectory_rows, payload = trajectory(
                    task,
                    task_index,
                    architecture,
                    replicate,
                    corpus,
                    device,
                    save_replay,
                )
                rows.extend(trajectory_rows)
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
    attach_rescue_metrics(rows, payloads, device)
    del payloads
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def develop() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = run_corpus(DEVELOPMENT_TASKS, DEVELOPMENT_REPLICATES, "development", torch.device("cuda"))
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
    print(canonical_json({"prediction_gate_pass": summary["prediction_gate_pass"], "rescue_gate_pass": summary["rescue_gate_pass"]}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal protocol or outcomes already exist")
    development = read_json(DEVELOPMENT_SUMMARY)
    if not development["prediction_gate_pass"]:
        raise RuntimeError("development prediction gate failed")
    protocol = {
        "phase": PHASE,
        "created_at": utc_now(),
        "question": "Can a sealed local directional probe prospectively predict a full natural AdamW response transition, and can one preregistered component group nontrivially rescue a matched wrong update?",
        "scientific_separation": {
            "prediction": "Uses the exact pre-update parent, minibatch, optimizer state, and resulting deterministic AdamW direction; predicts a disjoint-panel functional response, not the update itself.",
            "rescue": "Independent gate. A prediction pass cannot compensate for rescue failure.",
            "scope": "Synthetic 32-class TinyTransformer training dynamics, not natural language pretraining.",
        },
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "formal_tasks": list(FORMAL_TASKS),
        "formal_replicates": FORMAL_REPLICATES,
        "stages": list(STAGES),
        "rescue_stage": RESCUE_STAGE,
        "wrong_time_stage": WRONG_TIME_STAGE,
        "batch_size": BATCH_SIZE,
        "tangent_epsilon": TANGENT_EPSILON,
        "prediction_nulls": ["same_norm_random_direction", "same_norm_negative_raw_gradient"],
        "rescue_selection": "top1 layer/component group by calibration-panel group-specific tangent norm",
        "rescue_nulls": ["same_norm_wrong_component", "same_norm_wrong_time", "same_norm_wrong_task", "same_mask_random"],
        "control_thresholds": CONTROL_THRESHOLDS,
        "prediction_thresholds": PREDICTION_THRESHOLDS,
        "rescue_thresholds": RESCUE_THRESHOLDS,
        "continuation_rule": "Self-consistent optimizer continuation is authorized only if prediction and rescue pass independently in both formal splits.",
        "forbidden": [
            "change epsilon after formal outcomes",
            "change stages or task families after formal outcomes",
            "select top-k greater than one after seeing rescue results",
            "pool architectures to hide an architecture failure",
            "treat tangent prediction as a learned global formation law",
            "treat failed rescue as proof that no distributed rescue exists",
        ],
        "upstream": {
            "phase1193_final_sha256": file_sha256(p1193.FINAL_PATH),
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
    if file_sha256(p1193.FINAL_PATH) != protocol["upstream"]["phase1193_final_sha256"]:
        raise RuntimeError("Phase1193 final changed")
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
    row_manifest = {
        path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))
    }
    replay_manifest = {
        path.name: file_sha256(path) for path in sorted(REPLAY_ROOT.glob("*.pt"))
    }
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
    print(canonical_json({"row_count": len(rows), "trajectory_count": seal["trajectory_count"], "seal_digest": seal["seal_digest"]}))


def compile_claims(summary: dict[str, Any]) -> dict[str, Any]:
    prediction_pass = summary["prediction_decision"] == "positive"
    rescue_pass = summary["rescue_decision"] == "positive"
    return {
        "prospective_tangent_prediction": {
            "type": "E3-KT" if prediction_pass else "E3-KT-negative-boundary",
            "accepted": True,
            "claim": (
                "A sealed calibration-panel local directional probe predicts the disjoint-panel full natural AdamW response transition across two TinyTransformer architectures, three task families, and early/middle/late stages."
                if prediction_pass
                else "The preregistered prospective tangent prediction did not confirm across both splits."
            ),
        },
        "minimal_component_rescue": {
            "type": "E3-KT" if rescue_pass else "E3-KT-scope-boundary",
            "accepted": True,
            "claim": (
                "A single prospectively selected layer/component patch nontrivially rescues the matched wrong update beyond all four null donors."
                if rescue_pass
                else "The top-1 component rescue did not independently confirm across both architectures and splits; this does not exclude distributed or multi-component rescue."
            ),
        },
    }


def analyze() -> None:
    verify_protocol()
    seal = read_json(TRAINING_SEAL)
    rows = read_jsonl(RAW_ROWS)
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("formal rows hash mismatch")
    discovery = summarize(rows, "discovery")
    confirmation = summarize(rows, "confirmation")
    prediction_positive = discovery["prediction_gate_pass"] and confirmation["prediction_gate_pass"]
    rescue_positive = discovery["rescue_gate_pass"] and confirmation["rescue_gate_pass"]
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "discovery": discovery,
        "confirmation": confirmation,
        "prediction_decision": "positive" if prediction_positive else "not_confirmed",
        "rescue_decision": "positive" if rescue_positive else "not_confirmed",
        "overall_status": (
            "prospective_tangent_and_minimal_rescue_confirmed"
            if prediction_positive and rescue_positive
            else (
                "prospective_tangent_confirmed_minimal_rescue_not_confirmed"
                if prediction_positive
                else "prospective_tangent_not_confirmed"
            )
        ),
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, compile_claims(summary))
    print(canonical_json({"prediction": summary["prediction_decision"], "rescue": summary["rescue_decision"], "status": summary["overall_status"]}))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    both = summary["prediction_decision"] == "positive" and summary["rescue_decision"] == "positive"
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["overall_status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "self_consistent_optimizer_continuation": both,
            "natural_language_formation_claim": False,
            "new_camera_search": False,
        },
        "scope": {
            "confirmed": "prospective local response law only if prediction gate passed",
            "not_claimed": [
                "a learned global formation operator",
                "why AdamW naturally selects its update",
                "natural-language encoding mechanism",
                "minimal rescue if its independent gate failed",
            ],
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def replay_capsule(path: Path, device: torch.device) -> dict[str, Any]:
    capsule = torch.load(path, map_location=device, weights_only=False)
    task = capsule["task"]
    inputs, targets, candidates, calibration, evaluation = make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[capsule["architecture"]]).to(device)
    model.load_state_dict(capsule["model_state"])
    optimizer = p1193.optimizer_for(model)
    optimizer.load_state_dict(capsule["optimizer_state"])
    row, _ = event(
        model,
        optimizer,
        inputs,
        targets,
        candidates,
        calibration,
        evaluation,
        capsule["batch_indices"].to(device),
        int(capsule["stage"]),
        int(capsule["seed"]) + int(capsule["stage"]) * 1009,
        False,
    )
    return row


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
