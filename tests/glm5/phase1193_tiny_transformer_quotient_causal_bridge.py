"""TinyTransformer quotient-response causal architecture bridge.

This phase transports the Phase 1192 same-parent matched update fork from a
RoleSquare network to causal Transformers with attention, residual streams,
LayerNorm, MLP branches, token embeddings, and autoregressive masking.  The
camera is a functional branch-ablation spectrum.  Head effects are sorted
within each layer, quotienting out the exact head-permutation gauge while
retaining layer order and component role.
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402
import phase1159_free_transformer_causal_use_external_validity as p1159  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1192_same_parent_causal_update_fork as p1192  # noqa: E402


PHASE = 1193
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1193_tiny_transformer_quotient_causal_bridge_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1193_tiny_transformer_quotient_causal_bridge"
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

ARCHITECTURES = {
    "compact": ModelConfig(
        layers=4, width=64, heads=4, mlp_width=128, max_length=5, vocab_size=48
    ),
    "deep": ModelConfig(
        layers=6, width=96, heads=4, mlp_width=192, max_length=5, vocab_size=48
    ),
}
DEVELOPMENT_TASKS = {
    "transformer_dev_00_a3_b1": (3, 1),
    "transformer_dev_01_a5_b7": (5, 7),
    "transformer_dev_02_a7_b11": (7, 11),
    "transformer_dev_03_a9_b13": (9, 13),
}
FORMAL_TASKS = {
    "transformer_affine_00_a11_b3": (11, 3),
    "transformer_affine_01_a13_b9": (13, 9),
    "transformer_affine_02_a15_b5": (15, 5),
    "transformer_affine_03_a17_b7": (17, 7),
    "transformer_affine_04_a19_b1": (19, 1),
    "transformer_affine_05_a21_b13": (21, 13),
    "transformer_affine_06_a23_b17": (23, 17),
    "transformer_affine_07_a25_b19": (25, 19),
}
PARENT_STEP = 500
HORIZON = 20
DEVELOPMENT_REPLICATES = 2
FORMAL_REPLICATES = 4
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
GRADIENT_CLIP_NORM = 1.0
DEVELOPMENT_MODEL_SEED_BASE = 1_193_900_000
FORMAL_MODEL_SEED_BASE = 1_193_000_000
ANGLE_GRID = tuple(float(value) for value in np.linspace(math.pi / 3.0, 5.0 * math.pi / 3.0, 65))

CONTROL_THRESHOLDS = {
    "gauge_logit_max_error_max": 1e-4,
    "gauge_response_distance_max": 1e-4,
    "sentinel_logit_change_min": 1e-2,
    "sentinel_response_distance_min": 1e-2,
    "loss_gap_max": 1e-5,
    "update_norm_relative_error_max": 1e-6,
    "endpoint_norm_relative_error_max": 1e-6,
    "first_order_relative_error_max": 1e-4,
    "update_cosine_max": 0.90,
    "orthogonal_fraction_min": 0.40,
    "immediate_effect_norm_min": 5e-5,
    "horizon_effect_norm_min": 5e-5,
    "parent_accuracy_min": 1.0,
    "immediate_accuracy_min": 1.0,
    "horizon_accuracy_min": 1.0,
    "eligible_fraction_min": 0.95,
}
POSITIVE_THRESHOLDS = {
    "immediate_true_cosine_mean_min": 0.80,
    "immediate_advantage_mean_min": 0.50,
    "immediate_positive_fraction_min": 0.75,
    "horizon_true_cosine_mean_min": 0.70,
    "horizon_advantage_mean_min": 0.50,
    "horizon_positive_fraction_min": 0.75,
    "positive_task_architecture_count_min": 6,
    "positive_groups_per_architecture_min": 3,
}
NEGATIVE_THRESHOLDS = {
    "horizon_advantage_mean_max": 0.10,
    "horizon_positive_fraction_max": 0.60,
    "positive_task_architecture_count_max": 4,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(task_index: int, architecture: str, replicate: int, corpus: str) -> int:
    base = DEVELOPMENT_MODEL_SEED_BASE if corpus == "development" else FORMAL_MODEL_SEED_BASE
    return base + task_index * 100_003 + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().float().reshape(-1) for parameter in model.parameters()])


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().float().reshape(-1) for parameter in model.parameters()])


@torch.no_grad()
def assign_parameters(model: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.copy_(vector[offset : offset + count].view_as(parameter).to(parameter.dtype))
        offset += count
    if offset != vector.numel():
        raise RuntimeError("parameter vector length mismatch")


def optimizer_for(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def make_dataset(
    affine: tuple[int, int], seed: int, device: torch.device
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lexicon = p1159.make_lexicon(seed + 11)
    inputs_cpu, base_targets_cpu = p1159.all_training_examples(lexicon)
    a, b = affine
    permutation = torch.tensor([(a * value + b) % p1159.N_CLASSES for value in range(p1159.N_CLASSES)])
    targets_cpu = permutation[base_targets_cpu]
    calibration_values: list[bool] = []
    for template_index in range(len(p1159.TEMPLATES)):
        for context in range(p1159.CONTEXTS):
            for row in range(p1159.ROWS):
                for col in range(p1159.COLS):
                    calibration_values.append((template_index + context + row + col) % 2 == 0)
    calibration = torch.tensor(calibration_values, dtype=torch.bool, device=device)
    return (
        lexicon,
        inputs_cpu.to(device),
        targets_cpu.to(device),
        p1159.answer_ids(lexicon, device),
        calibration,
        ~calibration,
    )


def training_loss(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)[:, -1].index_select(-1, candidates)
    return F.cross_entropy(logits.float(), targets)


def training_step(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = training_loss(model, inputs, targets, candidates)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite training loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
        raise RuntimeError("nonfinite gradient norm")
    optimizer.step()
    return float(loss.item())


@torch.inference_mode()
def behavior(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    logits = model(inputs)[:, -1].float().index_select(-1, candidates)
    probabilities = torch.softmax(logits, dim=-1)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    return {
        "accuracy": float((logits.argmax(dim=-1) == targets).float().mean().item()),
        "minimum_probability": float(target_probability.min().item()),
        "mean_probability": float(target_probability.mean().item()),
        "finite_fraction": float(torch.isfinite(logits).float().mean().item()),
    }


def logits_with_ablation(
    model: TinyCausalTransformer,
    input_ids: torch.Tensor,
    ablation: tuple[str, int, int] | None = None,
) -> torch.Tensor:
    hidden = model.embed(input_ids)
    for layer_index, block in enumerate(model.blocks):
        normed = block.attn_norm(hidden)
        if ablation is not None and ablation[0] == "head" and ablation[1] == layer_index:
            batch, length, width = normed.shape
            qkv = block.attn.qkv(normed).view(
                batch, length, 3, block.attn.heads, block.attn.head_dim
            )
            query, key, value = qkv.unbind(dim=2)
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
            )
            attended[:, ablation[2]] = 0.0
            attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
            attention_output = block.attn.out(attended)
        else:
            attention_output = block.attn(normed)
        hidden = hidden + attention_output
        if ablation is None or ablation[0] != "mlp" or ablation[1] != layer_index:
            hidden = hidden + block.mlp(block.mlp_norm(hidden))
    return model.lm_head(model.final_norm(hidden))


@torch.inference_mode()
def mean_margin(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    ablation: tuple[str, int, int] | None = None,
) -> float:
    logits = logits_with_ablation(model, inputs, ablation)[:, -1].float().index_select(-1, candidates)
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, targets[:, None], float("-inf"))
    return float((correct - torch.logsumexp(alternatives, dim=-1)).mean().item())


@torch.inference_mode()
def quotient_response(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> np.ndarray:
    model.eval()
    baseline = mean_margin(model, inputs, targets, candidates)
    response: list[float] = []
    for layer_index, block in enumerate(model.blocks):
        head_effects = [
            baseline
            - mean_margin(model, inputs, targets, candidates, ("head", layer_index, head_index))
            for head_index in range(block.attn.heads)
        ]
        response.extend(sorted(head_effects))
        response.append(
            baseline - mean_margin(model, inputs, targets, candidates, ("mlp", layer_index, -1))
        )
    vector = np.asarray(response, dtype=np.float64)
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


@torch.no_grad()
def permute_attention_heads(model: TinyCausalTransformer, compensate_output: bool) -> None:
    for block in model.blocks:
        heads = block.attn.heads
        head_dim = block.attn.head_dim
        permutation = torch.arange(heads - 1, -1, -1, device=block.attn.qkv.weight.device)
        qkv = block.attn.qkv.weight.view(3, heads, head_dim, -1).clone()
        block.attn.qkv.weight.copy_(qkv[:, permutation].reshape_as(block.attn.qkv.weight))
        if compensate_output:
            output = block.attn.out.weight.view(block.attn.out.weight.shape[0], heads, head_dim).clone()
            block.attn.out.weight.copy_(output[:, permutation].reshape_as(block.attn.out.weight))


def clone_model(
    config: ModelConfig, state: dict[str, torch.Tensor], device: torch.device
) -> TinyCausalTransformer:
    model = TinyCausalTransformer(config).to(device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def instrument_metrics(
    model: TinyCausalTransformer,
    config: ModelConfig,
    state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, float]:
    gauge = clone_model(config, state, inputs.device)
    permute_attention_heads(gauge, compensate_output=True)
    sentinel = clone_model(config, state, inputs.device)
    permute_attention_heads(sentinel, compensate_output=False)
    parent_logits = model(inputs)[:, -1].float()
    gauge_logits = gauge(inputs)[:, -1].float()
    sentinel_logits = sentinel(inputs)[:, -1].float()
    parent_response = quotient_response(model, inputs, targets, candidates)
    gauge_response = quotient_response(gauge, inputs, targets, candidates)
    sentinel_response = quotient_response(sentinel, inputs, targets, candidates)
    metrics = {
        "gauge_logit_max_error": float((parent_logits - gauge_logits).abs().max().item()),
        "gauge_response_distance": float(np.linalg.norm(parent_response - gauge_response)),
        "sentinel_logit_max_change": float((parent_logits - sentinel_logits).abs().max().item()),
        "sentinel_response_distance": float(np.linalg.norm(parent_response - sentinel_response)),
    }
    del gauge, sentinel
    return metrics


@torch.no_grad()
def select_control_update(
    probe: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    real_update: torch.Tensor,
    gradient: torch.Tensor,
    target_loss: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
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
    scored: list[tuple[float, float, torch.Tensor]] = []
    for angle in ANGLE_GRID:
        update = fixed + math.cos(angle) * residual + math.sin(angle) * random_residual
        assign_parameters(probe, parent_vector + update)
        loss = float(training_loss(probe, inputs, targets, candidates).item())
        scored.append((abs(loss - target_loss), angle, update.clone()))
    loss_gap, angle, selected = min(scored, key=lambda item: (item[0], item[1]))
    assign_parameters(probe, parent_vector)
    real_norm = float(real_update.norm().item())
    selected_norm = float(selected.norm().item())
    real_endpoint = float((parent_vector + real_update).norm().item())
    selected_endpoint = float((parent_vector + selected).norm().item())
    first_order_real = float(torch.dot(gradient, real_update).item())
    first_order_control = float(torch.dot(gradient, selected).item())
    return selected, {
        "loss_gap": loss_gap,
        "angle": angle,
        "update_norm_relative_error": abs(selected_norm - real_norm) / max(real_norm, 1e-12),
        "endpoint_norm_relative_error": abs(selected_endpoint - real_endpoint) / max(real_endpoint, 1e-12),
        "first_order_relative_error": abs(first_order_control - first_order_real)
        / max(abs(first_order_real), 1e-12),
        "update_cosine": float(
            torch.dot(real_update, selected).item() / max(real_norm * selected_norm, 1e-12)
        ),
        "orthogonal_fraction": float(residual_norm.item()) / max(real_norm, 1e-12),
        "real_update_norm": real_norm,
        "endpoint_parameter_norm": real_endpoint,
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def train_parent(
    task_name: str,
    task_index: int,
    affine: tuple[int, int],
    architecture: str,
    replicate: int,
    corpus: str,
    device: torch.device,
) -> dict[str, Any]:
    seed = model_seed(task_index, architecture, replicate, corpus)
    set_seed(seed)
    config = ARCHITECTURES[architecture]
    _, inputs, targets, candidates, _, _ = make_dataset(affine, seed, device)
    model = TinyCausalTransformer(config).to(device)
    optimizer = optimizer_for(model)
    for _ in range(PARENT_STEP):
        training_step(model, optimizer, inputs, targets, candidates)
    capsule = {
        "phase": PHASE,
        "corpus": corpus,
        "task_name": task_name,
        "task_index": task_index,
        "affine": list(affine),
        "architecture": architecture,
        "config": asdict(config),
        "replicate": replicate,
        "seed": seed,
        "parent_step": PARENT_STEP,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
    }
    del model, optimizer, inputs, targets, candidates
    gc.collect()
    torch.cuda.empty_cache()
    return capsule


def run_from_capsule(capsule: dict[str, Any], device: torch.device) -> dict[str, Any]:
    affine = tuple(int(value) for value in capsule["affine"])
    architecture = str(capsule["architecture"])
    config = ModelConfig(**capsule["config"])
    seed = int(capsule["seed"])
    _, inputs, targets, candidates, calibration_mask, evaluation_mask = make_dataset(affine, seed, device)
    model = clone_model(config, capsule["model_state"], device)
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(capsule["optimizer_state"])
    parent_behavior = behavior(model, inputs, targets, candidates)
    parent_vector = flatten_parameters(model)
    camera = instrument_metrics(
        model,
        config,
        capsule["model_state"],
        inputs[calibration_mask],
        targets[calibration_mask],
        candidates,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parent_loss = training_loss(model, inputs, targets, candidates)
    parent_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    gradient = flatten_gradients(model)
    optimizer.step()
    real_update = flatten_parameters(model) - parent_vector
    post_step_optimizer_state = copy.deepcopy(optimizer.state_dict())
    real_immediate_loss = float(training_loss(model, inputs, targets, candidates).item())

    probe = clone_model(config, capsule["model_state"], device)
    control_update, control_metrics = select_control_update(
        probe,
        parent_vector,
        real_update,
        gradient,
        real_immediate_loss,
        inputs,
        targets,
        candidates,
        seed + 991,
    )
    control_model = clone_model(config, capsule["model_state"], device)
    assign_parameters(control_model, parent_vector + control_update)
    control_immediate_loss = float(training_loss(control_model, inputs, targets, candidates).item())
    control_optimizer = optimizer_for(control_model)
    control_optimizer.load_state_dict(post_step_optimizer_state)

    real_immediate_behavior = behavior(model, inputs, targets, candidates)
    control_immediate_behavior = behavior(control_model, inputs, targets, candidates)
    immediate_calibration = quotient_response(
        model, inputs[calibration_mask], targets[calibration_mask], candidates
    ) - quotient_response(
        control_model, inputs[calibration_mask], targets[calibration_mask], candidates
    )
    immediate_evaluation = quotient_response(
        model, inputs[evaluation_mask], targets[evaluation_mask], candidates
    ) - quotient_response(
        control_model, inputs[evaluation_mask], targets[evaluation_mask], candidates
    )

    for _ in range(HORIZON - 1):
        training_step(model, optimizer, inputs, targets, candidates)
        training_step(control_model, control_optimizer, inputs, targets, candidates)
    horizon_calibration = quotient_response(
        model, inputs[calibration_mask], targets[calibration_mask], candidates
    ) - quotient_response(
        control_model, inputs[calibration_mask], targets[calibration_mask], candidates
    )
    horizon_evaluation = quotient_response(
        model, inputs[evaluation_mask], targets[evaluation_mask], candidates
    ) - quotient_response(
        control_model, inputs[evaluation_mask], targets[evaluation_mask], candidates
    )
    real_horizon_behavior = behavior(model, inputs, targets, candidates)
    control_horizon_behavior = behavior(control_model, inputs, targets, candidates)

    row = {
        "corpus": str(capsule["corpus"]),
        "task_name": str(capsule["task_name"]),
        "task_index": int(capsule["task_index"]),
        "affine": list(affine),
        "architecture": architecture,
        "config": asdict(config),
        "replicate": int(capsule["replicate"]),
        "seed": seed,
        "trajectory_id": f"{capsule['task_name']}/{architecture}/r{int(capsule['replicate'])}",
        "parent_step": PARENT_STEP,
        "horizon": HORIZON,
        "response_dimension": config.layers * (config.heads + 1),
        "parent_loss": float(parent_loss.detach().item()),
        "real_immediate_loss": real_immediate_loss,
        "control_immediate_loss": control_immediate_loss,
        "parent_accuracy": parent_behavior["accuracy"],
        "parent_minimum_probability": parent_behavior["minimum_probability"],
        "real_immediate_accuracy": real_immediate_behavior["accuracy"],
        "control_immediate_accuracy": control_immediate_behavior["accuracy"],
        "real_horizon_accuracy": real_horizon_behavior["accuracy"],
        "control_horizon_accuracy": control_horizon_behavior["accuracy"],
        **camera,
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
        row["gauge_logit_max_error"] <= CONTROL_THRESHOLDS["gauge_logit_max_error_max"]
        and row["gauge_response_distance"] <= CONTROL_THRESHOLDS["gauge_response_distance_max"]
        and row["sentinel_logit_max_change"] >= CONTROL_THRESHOLDS["sentinel_logit_change_min"]
        and row["sentinel_response_distance"] >= CONTROL_THRESHOLDS["sentinel_response_distance_min"]
        and row["loss_gap"] <= CONTROL_THRESHOLDS["loss_gap_max"]
        and row["update_norm_relative_error"] <= CONTROL_THRESHOLDS["update_norm_relative_error_max"]
        and row["endpoint_norm_relative_error"] <= CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
        and row["first_order_relative_error"] <= CONTROL_THRESHOLDS["first_order_relative_error_max"]
        and row["update_cosine"] <= CONTROL_THRESHOLDS["update_cosine_max"]
        and row["orthogonal_fraction"] >= CONTROL_THRESHOLDS["orthogonal_fraction_min"]
        and row["immediate_calibration_norm"] >= CONTROL_THRESHOLDS["immediate_effect_norm_min"]
        and row["horizon_calibration_norm"] >= CONTROL_THRESHOLDS["horizon_effect_norm_min"]
        and row["parent_accuracy"] >= CONTROL_THRESHOLDS["parent_accuracy_min"]
        and min(row["real_immediate_accuracy"], row["control_immediate_accuracy"])
        >= CONTROL_THRESHOLDS["immediate_accuracy_min"]
        and min(row["real_horizon_accuracy"], row["control_horizon_accuracy"])
        >= CONTROL_THRESHOLDS["horizon_accuracy_min"]
    )
    del model, optimizer, control_model, control_optimizer, probe
    gc.collect()
    torch.cuda.empty_cache()
    return row


def add_nulls(rows: list[dict[str, Any]], replicates: int) -> None:
    lookup = {
        (row["split"], row["task_index"], row["architecture"], row["replicate"]): row
        for row in rows
    }
    split_indices = {
        split: sorted({row["task_index"] for row in rows if row["split"] == split})
        for split in {row["split"] for row in rows}
    }
    for row in rows:
        key = (row["split"], row["task_index"], row["architecture"], (row["replicate"] + 1) % replicates)
        replicate_null = lookup[key]
        indices = split_indices[row["split"]]
        position = indices.index(row["task_index"])
        next_task_index = indices[(position + 1) % len(indices)]
        task_null = lookup[(row["split"], next_task_index, row["architecture"], row["replicate"])]
        row["replicate_null_trajectory_id"] = replicate_null["trajectory_id"]
        row["task_null_trajectory_id"] = task_null["trajectory_id"]
        for horizon in ("immediate", "horizon"):
            calibration = np.asarray(row[horizon + "_calibration"], dtype=np.float64)
            replicate_cosine = cosine(
                calibration, np.asarray(replicate_null[horizon + "_evaluation"], dtype=np.float64)
            )
            task_cosine = cosine(
                calibration, np.asarray(task_null[horizon + "_evaluation"], dtype=np.float64)
            )
            conservative_null = max(replicate_cosine, task_cosine)
            row[horizon + "_replicate_null_cosine"] = replicate_cosine
            row[horizon + "_task_null_cosine"] = task_cosine
            row[horizon + "_null_cosine"] = conservative_null
            row[horizon + "_advantage"] = row[horizon + "_true_cosine"] - conservative_null


def task_architecture_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    groups = sorted({(row["task_name"], row["architecture"]) for row in rows})
    for task_name, architecture in groups:
        selected = [
            row for row in rows if row["task_name"] == task_name and row["architecture"] == architecture
        ]
        advantage = float(np.mean([row["horizon_advantage"] for row in selected]))
        summaries.append(
            {
                "task_name": task_name,
                "architecture": architecture,
                "horizon_advantage_mean": advantage,
                "positive": advantage > 0.0,
            }
        )
    return summaries


def summarize(
    rows: list[dict[str, Any]], split: str, expected_systems: int, expected_tasks: int
) -> dict[str, Any]:
    selected = rows if split == "development" else [row for row in rows if row["split"] == split]
    groups = task_architecture_summaries(selected)
    architecture_positive_counts = {
        architecture: sum(group["positive"] for group in groups if group["architecture"] == architecture)
        for architecture in ARCHITECTURES
    }
    result = {
        "split": split,
        "system_count": len(selected),
        "task_count": len({row["task_name"] for row in selected}),
        "task_architecture_count": len(groups),
        "eligible_system_count": sum(row["control_qualified"] for row in selected),
        "eligible_fraction": float(np.mean([row["control_qualified"] for row in selected])),
        "gauge_logit_max_error_max": max(row["gauge_logit_max_error"] for row in selected),
        "gauge_response_distance_max": max(row["gauge_response_distance"] for row in selected),
        "sentinel_logit_change_min": min(row["sentinel_logit_max_change"] for row in selected),
        "sentinel_response_distance_min": min(row["sentinel_response_distance"] for row in selected),
        "loss_gap_max": max(row["loss_gap"] for row in selected),
        "update_norm_relative_error_max": max(row["update_norm_relative_error"] for row in selected),
        "endpoint_norm_relative_error_max": max(row["endpoint_norm_relative_error"] for row in selected),
        "first_order_relative_error_max": max(row["first_order_relative_error"] for row in selected),
        "update_cosine_max": max(row["update_cosine"] for row in selected),
        "update_cosine_mean": float(np.mean([row["update_cosine"] for row in selected])),
        "orthogonal_fraction_min": min(row["orthogonal_fraction"] for row in selected),
        "immediate_effect_norm_min": min(row["immediate_calibration_norm"] for row in selected),
        "horizon_effect_norm_min": min(row["horizon_calibration_norm"] for row in selected),
        "parent_accuracy_min": min(row["parent_accuracy"] for row in selected),
        "immediate_accuracy_min": min(
            min(row["real_immediate_accuracy"], row["control_immediate_accuracy"]) for row in selected
        ),
        "horizon_accuracy_min": min(
            min(row["real_horizon_accuracy"], row["control_horizon_accuracy"]) for row in selected
        ),
        "immediate_true_cosine_mean": float(np.mean([row["immediate_true_cosine"] for row in selected])),
        "immediate_null_cosine_mean": float(np.mean([row["immediate_null_cosine"] for row in selected])),
        "immediate_advantage_mean": float(np.mean([row["immediate_advantage"] for row in selected])),
        "immediate_positive_fraction": float(np.mean([row["immediate_advantage"] > 0 for row in selected])),
        "horizon_true_cosine_mean": float(np.mean([row["horizon_true_cosine"] for row in selected])),
        "horizon_null_cosine_mean": float(np.mean([row["horizon_null_cosine"] for row in selected])),
        "horizon_advantage_mean": float(np.mean([row["horizon_advantage"] for row in selected])),
        "horizon_positive_fraction": float(np.mean([row["horizon_advantage"] > 0 for row in selected])),
        "positive_task_architecture_count": sum(group["positive"] for group in groups),
        "architecture_positive_counts": architecture_positive_counts,
        "task_architecture_summaries": groups,
    }
    result["control_gate_pass"] = bool(
        len(selected) == expected_systems
        and result["task_count"] == expected_tasks
        and result["task_architecture_count"] == expected_tasks * len(ARCHITECTURES)
        and result["eligible_fraction"] >= CONTROL_THRESHOLDS["eligible_fraction_min"]
        and result["gauge_logit_max_error_max"] <= CONTROL_THRESHOLDS["gauge_logit_max_error_max"]
        and result["gauge_response_distance_max"] <= CONTROL_THRESHOLDS["gauge_response_distance_max"]
        and result["sentinel_logit_change_min"] >= CONTROL_THRESHOLDS["sentinel_logit_change_min"]
        and result["sentinel_response_distance_min"] >= CONTROL_THRESHOLDS["sentinel_response_distance_min"]
        and result["loss_gap_max"] <= CONTROL_THRESHOLDS["loss_gap_max"]
        and result["update_norm_relative_error_max"] <= CONTROL_THRESHOLDS["update_norm_relative_error_max"]
        and result["endpoint_norm_relative_error_max"] <= CONTROL_THRESHOLDS["endpoint_norm_relative_error_max"]
        and result["first_order_relative_error_max"] <= CONTROL_THRESHOLDS["first_order_relative_error_max"]
        and result["update_cosine_max"] <= CONTROL_THRESHOLDS["update_cosine_max"]
        and result["orthogonal_fraction_min"] >= CONTROL_THRESHOLDS["orthogonal_fraction_min"]
        and result["immediate_effect_norm_min"] >= CONTROL_THRESHOLDS["immediate_effect_norm_min"]
        and result["horizon_effect_norm_min"] >= CONTROL_THRESHOLDS["horizon_effect_norm_min"]
        and result["parent_accuracy_min"] >= CONTROL_THRESHOLDS["parent_accuracy_min"]
        and result["immediate_accuracy_min"] >= CONTROL_THRESHOLDS["immediate_accuracy_min"]
        and result["horizon_accuracy_min"] >= CONTROL_THRESHOLDS["horizon_accuracy_min"]
    )
    result["positive_gate_pass"] = bool(
        result["control_gate_pass"]
        and result["immediate_true_cosine_mean"] >= POSITIVE_THRESHOLDS["immediate_true_cosine_mean_min"]
        and result["immediate_advantage_mean"] >= POSITIVE_THRESHOLDS["immediate_advantage_mean_min"]
        and result["immediate_positive_fraction"] >= POSITIVE_THRESHOLDS["immediate_positive_fraction_min"]
        and result["horizon_true_cosine_mean"] >= POSITIVE_THRESHOLDS["horizon_true_cosine_mean_min"]
        and result["horizon_advantage_mean"] >= POSITIVE_THRESHOLDS["horizon_advantage_mean_min"]
        and result["horizon_positive_fraction"] >= POSITIVE_THRESHOLDS["horizon_positive_fraction_min"]
        and result["positive_task_architecture_count"]
        >= POSITIVE_THRESHOLDS["positive_task_architecture_count_min"]
        and all(
            count >= POSITIVE_THRESHOLDS["positive_groups_per_architecture_min"]
            for count in architecture_positive_counts.values()
        )
    )
    result["negative_boundary_pass"] = bool(
        result["control_gate_pass"]
        and result["horizon_advantage_mean"] <= NEGATIVE_THRESHOLDS["horizon_advantage_mean_max"]
        and result["horizon_positive_fraction"] <= NEGATIVE_THRESHOLDS["horizon_positive_fraction_max"]
        and result["positive_task_architecture_count"]
        <= NEGATIVE_THRESHOLDS["positive_task_architecture_count_max"]
    )
    return result


def source_hashes() -> dict[str, str]:
    actual = [SCRIPT, AUDIT_SCRIPT, Path(p1159.__file__), Path(p1187.__file__), Path(p1192.__file__)]
    architecture_source = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
    actual.append(architecture_source)
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in actual}


def develop() -> None:
    if DEVELOPMENT_ROWS.exists():
        raise RuntimeError("development already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    total = len(DEVELOPMENT_TASKS) * len(ARCHITECTURES) * DEVELOPMENT_REPLICATES
    for task_index, (task_name, affine) in enumerate(DEVELOPMENT_TASKS.items()):
        for architecture in ARCHITECTURES:
            for replicate in range(DEVELOPMENT_REPLICATES):
                capsule = train_parent(
                    task_name, task_index, affine, architecture, replicate, "development", device
                )
                row = run_from_capsule(capsule, device)
                row["split"] = "development"
                rows.append(row)
                print(canonical_json({"development": len(rows), "total": total}), flush=True)
    add_nulls(rows, DEVELOPMENT_REPLICATES)
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = summarize(rows, "development", total, len(DEVELOPMENT_TASKS))
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
    summary["summary_digest"] = digest(
        {key: value for key, value in summary.items() if key != "summary_digest"}
    )
    write_json(DEVELOPMENT_SUMMARY, summary)
    if not summary["positive_gate_pass"]:
        raise RuntimeError("development did not authorize formal preregistration")


def preregister() -> None:
    development = read_json(DEVELOPMENT_SUMMARY)
    upstream = read_json(p1192.FINAL_PATH)
    if not development["positive_gate_pass"]:
        raise RuntimeError("development gate failed")
    if not upstream["authorized_next"]["tiny_transformer_bridge_preregistration"]:
        raise RuntimeError("Phase1192 did not authorize the bridge")
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    protocol = {
        "phase": PHASE,
        "title": "TinyTransformer quotient-response causal architecture bridge",
        "created_at_utc": utc_now(),
        "scientific_question": (
            "Does the same-parent matched causal update-direction effect survive in freely trained causal "
            "Transformers when the response object is a head-permutation-quotiented attention/MLP branch "
            "ablation spectrum?"
        ),
        "upstream": {
            "phase1192_final_sha256": file_sha256(p1192.FINAL_PATH),
            "phase1192_final_digest": upstream["final_digest"],
            "development_rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "development_summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "development_summary_digest": development["summary_digest"],
        },
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "tasks": {name: list(affine) for name, affine in FORMAL_TASKS.items()},
        "replicates": FORMAL_REPLICATES,
        "parent_step": PARENT_STEP,
        "horizon": HORIZON,
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "precision": "BF16 autocast training with FP32 parameters; FP32 response camera",
            "batching": "full batch",
        },
        "camera": {
            "quantity": "drop in correct-answer log-odds margin under single branch ablation",
            "branches": "every attention head and each whole-layer MLP residual branch",
            "quotient": "sort head effects within each layer; retain layer order and component role",
            "normalization": "global centering followed by L2 unitization",
            "calibration": "compensated head reversal must preserve logits and quotient response",
            "positive_sentinel": "uncompensated QKV-head reversal must change logits and response",
        },
        "fork": {
            "same_parent_parameters_and_optimizer": True,
            "real_arm": "one natural AdamW update",
            "control_arm": "rotate the update residual orthogonal to gradient and parent parameter vector",
            "matched": [
                "update norm",
                "endpoint parameter norm",
                "gradient dot update",
                "immediate full-batch loss",
                "immediate behavior",
            ],
            "post_fork_optimizer_state": "identical post-real-step AdamW state in both arms",
        },
        "panels": "parity-balanced disjoint halves spanning all six token-order templates",
        "nulls": [
            "cyclic next replicate evaluation contrast within task and architecture",
            "cyclic next task evaluation contrast within split, architecture, and replicate",
            "the larger cosine is the frozen conservative null",
        ],
        "control_thresholds": CONTROL_THRESHOLDS,
        "positive_thresholds": POSITIVE_THRESHOLDS,
        "negative_thresholds": NEGATIVE_THRESHOLDS,
        "source_hashes": source_hashes(),
        "evidence_contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "hard_stops": [
            "No alternate architecture, parent step, horizon, camera, angle grid, null, or threshold is searched.",
            "Both architectures and both formal splits must satisfy the frozen positive gate.",
            "Any positive result is limited to this synthetic compositional Transformer family.",
            "No pretrained-language-model formation scan or theory closure is authorized.",
        ],
    }
    protocol["protocol_digest"] = digest(
        {key: value for key, value in protocol.items() if key != "protocol_digest"}
    )
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
    if file_sha256(p1192.FINAL_PATH) != protocol["upstream"]["phase1192_final_sha256"]:
        raise RuntimeError("Phase1192 final changed")
    return protocol


def capsule_path(task_name: str, architecture: str, replicate: int) -> Path:
    return PARENT_ROOT / f"{task_name}_{architecture}_r{replicate}.pt"


def row_path(task_name: str, architecture: str, replicate: int) -> Path:
    return FORMAL_ROW_ROOT / f"{task_name}_{architecture}_r{replicate}.json"


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal run already sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    PARENT_ROOT.mkdir(parents=True, exist_ok=True)
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    total = len(FORMAL_TASKS) * len(ARCHITECTURES) * FORMAL_REPLICATES
    completed = 0
    for task_index, (task_name, affine) in enumerate(FORMAL_TASKS.items()):
        split = "discovery" if task_index < 4 else "confirmation"
        split_task_index = task_index if split == "discovery" else task_index - 4
        for architecture in ARCHITECTURES:
            for replicate in range(FORMAL_REPLICATES):
                parent_file = capsule_path(task_name, architecture, replicate)
                formal_row_file = row_path(task_name, architecture, replicate)
                if not parent_file.exists():
                    capsule = train_parent(
                        task_name,
                        task_index,
                        affine,
                        architecture,
                        replicate,
                        "formal",
                        device,
                    )
                    torch.save(capsule, parent_file)
                else:
                    capsule = torch.load(parent_file, map_location=device, weights_only=False)
                if not formal_row_file.exists():
                    row = run_from_capsule(capsule, device)
                    row["split"] = split
                    row["split_task_index"] = split_task_index
                    write_json(formal_row_file, row)
                completed += 1
                print(
                    canonical_json(
                        {
                            "completed": completed,
                            "total": total,
                            "task": task_name,
                            "architecture": architecture,
                            "replicate": replicate,
                        }
                    ),
                    flush=True,
                )
    rows = [read_json(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))]
    if len(rows) != total:
        raise RuntimeError("formal row count mismatch")
    add_nulls(rows, FORMAL_REPLICATES)
    write_jsonl(RAW_ROWS, rows)
    parent_manifest = {path.name: file_sha256(path) for path in sorted(PARENT_ROOT.glob("*.pt"))}
    row_manifest = {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))}
    seal = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "parent_count": len(parent_manifest),
        "row_count": len(rows),
        "parent_manifest": parent_manifest,
        "raw_row_manifest": row_manifest,
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "seal_digest": None,
    }
    seal["seal_digest"] = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    write_json(TRAINING_SEAL, seal)


def verify_seal() -> dict[str, Any]:
    seal = read_json(TRAINING_SEAL)
    expected = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    if seal["seal_digest"] != expected:
        raise RuntimeError("seal digest mismatch")
    parent_manifest = {path.name: file_sha256(path) for path in sorted(PARENT_ROOT.glob("*.pt"))}
    row_manifest = {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))}
    if parent_manifest != seal["parent_manifest"] or row_manifest != seal["raw_row_manifest"]:
        raise RuntimeError("formal manifest changed")
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("analysis rows changed")
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
        positive = families["positive"]
        positive[split + ".controls"] = bounded(
            current["eligible_fraction"], CONTROL_THRESHOLDS["eligible_fraction_min"], ">="
        )
        positive[split + ".immediate_advantage"] = bounded(
            current["immediate_advantage_mean"],
            POSITIVE_THRESHOLDS["immediate_advantage_mean_min"],
            ">=",
        )
        positive[split + ".horizon_advantage"] = bounded(
            current["horizon_advantage_mean"],
            POSITIVE_THRESHOLDS["horizon_advantage_mean_min"],
            ">=",
        )
        positive[split + ".horizon_fraction"] = bounded(
            current["horizon_positive_fraction"],
            POSITIVE_THRESHOLDS["horizon_positive_fraction_min"],
            ">=",
        )
        positive[split + ".group_count"] = bounded(
            current["positive_task_architecture_count"],
            POSITIVE_THRESHOLDS["positive_task_architecture_count_min"],
            ">=",
        )
        for architecture in ARCHITECTURES:
            positive[f"{split}.{architecture}_groups"] = bounded(
                current["architecture_positive_counts"][architecture],
                POSITIVE_THRESHOLDS["positive_groups_per_architecture_min"],
                ">=",
            )
        negative = families["negative"]
        negative[split + ".controls"] = bounded(
            current["eligible_fraction"], CONTROL_THRESHOLDS["eligible_fraction_min"], ">="
        )
        negative[split + ".horizon_advantage"] = bounded(
            current["horizon_advantage_mean"], NEGATIVE_THRESHOLDS["horizon_advantage_mean_max"], "<="
        )
        negative[split + ".horizon_fraction"] = bounded(
            current["horizon_positive_fraction"],
            NEGATIVE_THRESHOLDS["horizon_positive_fraction_max"],
            "<=",
        )
        negative[split + ".group_count"] = bounded(
            current["positive_task_architecture_count"],
            NEGATIVE_THRESHOLDS["positive_task_architecture_count_max"],
            "<=",
        )
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
    summary["summary_digest"] = digest(
        {key: value for key, value in summary.items() if key != "summary_digest"}
    )
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, compile_claims(summary))


def finalize() -> None:
    protocol = verify_protocol()
    verify_seal()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH) if AUDIT_PATH.exists() else {}
    claim_key = "negative" if summary["decision"] == "negative_boundary" else "positive"
    complete = bool(
        summary["decision"] != "ambiguous"
        and claims[claim_key]["gate_pass"]
        and audit.get("gate_pass")
    )
    if complete and summary["decision"] == "positive":
        status = "tiny_transformer_causal_quotient_bridge_confirmed"
    elif complete and summary["decision"] == "negative_boundary":
        status = "tiny_transformer_causal_quotient_bridge_negative_boundary"
    elif summary["decision"] != "ambiguous" and claims[claim_key]["gate_pass"]:
        status = "awaiting_independent_audit"
    else:
        status = "ambiguous_or_failed"
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(CLAIMS_PATH),
        "audit_digest": audit.get("audit_digest"),
        "decision": summary["decision"],
        "independent_audit_pass": bool(audit.get("gate_pass")),
        "main_gate_complete": complete,
        "evidence_grade": "E3_KT_transformer_causal_bridge" if complete else "no_upgrade",
        "authorized_next": {
            "natural_transformer_formation_family_preregistration": bool(
                complete and summary["decision"] == "positive"
            ),
            "frozen_pretrained_lm_formation_scan": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "A same-parent, short-horizon causal update-direction effect on a head-permutation-quotiented "
            "attention/MLP branch-response state in two freely trained tiny causal-Transformer architectures. "
            "The result does not identify a natural optimizer component, a persistent identity, or a language-model mechanism."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(FINAL_PATH, final)


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
