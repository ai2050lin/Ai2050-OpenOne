"""Numerical pilot for the Phase 1193 TinyTransformer architecture bridge."""

from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402
import phase1159_free_transformer_causal_use_external_validity as p1159  # noqa: E402


DEVICE = torch.device("cuda")
CONFIG = ModelConfig(layers=4, width=64, heads=4, mlp_width=128, max_length=5, vocab_size=48)
ANGLES = tuple(float(value) for value in np.linspace(math.pi / 3.0, 5.0 * math.pi / 3.0, 65))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def flat_grads(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


@torch.no_grad()
def assign(model: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.copy_(vector[offset : offset + count].view_as(parameter))
        offset += count


def logits_for(
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
def margin(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    ablation: tuple[str, int, int] | None = None,
) -> float:
    logits = logits_for(model, inputs, ablation)[:, -1].float().index_select(-1, candidates)
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], float("-inf"))
    return float((correct - torch.logsumexp(masked, dim=-1)).mean().item())


@torch.inference_mode()
def quotient_response(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> np.ndarray:
    base = margin(model, inputs, targets, candidates)
    values: list[float] = []
    for layer_index, block in enumerate(model.blocks):
        heads = [
            base - margin(model, inputs, targets, candidates, ("head", layer_index, head))
            for head in range(block.attn.heads)
        ]
        values.extend(sorted(heads))
        values.append(base - margin(model, inputs, targets, candidates, ("mlp", layer_index, -1)))
    vector = np.asarray(values, dtype=np.float64)
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def loss_for(
    model: TinyCausalTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)[:, -1].index_select(-1, candidates)
    return F.cross_entropy(logits.float(), targets)


def train_step(
    model: TinyCausalTransformer,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_for(model, inputs, targets, candidates)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def select_control(
    probe: TinyCausalTransformer,
    parent: torch.Tensor,
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
    parent_orthogonal = parent - torch.dot(parent, gradient_unit) * gradient_unit
    if float(parent_orthogonal.norm()) > 1e-12:
        basis.append(parent_orthogonal / parent_orthogonal.norm())
    fixed = sum(torch.dot(real_update, vector) * vector for vector in basis)
    residual = real_update - fixed
    residual_norm = residual.norm()
    generator = torch.Generator(device=real_update.device).manual_seed(seed)
    random_direction = torch.randn(real_update.shape, generator=generator, device=real_update.device)
    for vector in basis:
        random_direction -= torch.dot(random_direction, vector) * vector
    residual_unit = residual / residual_norm.clamp_min(1e-12)
    random_direction -= torch.dot(random_direction, residual_unit) * residual_unit
    random_direction /= random_direction.norm().clamp_min(1e-12)
    random_residual = random_direction * residual_norm
    scored = []
    for angle in ANGLES:
        update = fixed + math.cos(angle) * residual + math.sin(angle) * random_residual
        assign(probe, parent + update)
        candidate_loss = float(loss_for(probe, inputs, targets, candidates).item())
        scored.append((abs(candidate_loss - target_loss), angle, update.clone()))
    gap, angle, selected = min(scored, key=lambda row: (row[0], row[1]))
    real_norm = float(real_update.norm())
    selected_norm = float(selected.norm())
    return selected, {
        "loss_gap": gap,
        "angle": angle,
        "update_norm_error": abs(selected_norm - real_norm) / real_norm,
        "endpoint_norm_error": abs(float((parent + selected).norm()) - float((parent + real_update).norm()))
        / float((parent + real_update).norm()),
        "first_order_error": abs(float(torch.dot(gradient, selected) - torch.dot(gradient, real_update)))
        / abs(float(torch.dot(gradient, real_update))),
        "update_cosine": float(torch.dot(real_update, selected) / (real_update.norm() * selected.norm())),
        "orthogonal_fraction": float(residual_norm) / real_norm,
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


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


def run(seed: int = 1193001, horizon: int = 20, affine_a: int = 5, affine_b: int = 7) -> dict[str, float]:
    set_seed(seed)
    lexicon = p1159.make_lexicon(seed + 11)
    inputs_cpu, base_targets_cpu = p1159.all_training_examples(lexicon)
    permutation = torch.tensor(
        [(affine_a * value + affine_b) % 32 for value in range(32)], dtype=torch.long
    )
    targets_cpu = permutation[base_targets_cpu]
    inputs = inputs_cpu.to(DEVICE)
    targets = targets_cpu.to(DEVICE)
    candidates = p1159.answer_ids(lexicon, DEVICE)
    model = TinyCausalTransformer(CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    for _ in range(500):
        train_step(model, optimizer, inputs, targets, candidates)
    with torch.inference_mode():
        parent_logits = model(inputs)[:, -1].index_select(-1, candidates)
        parent_accuracy = float((parent_logits.argmax(-1) == targets).float().mean())
    parent_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    parent_vector = flat_params(model)

    gauge = TinyCausalTransformer(CONFIG).to(DEVICE)
    gauge.load_state_dict(parent_state)
    permute_attention_heads(gauge, True)
    sentinel = TinyCausalTransformer(CONFIG).to(DEVICE)
    sentinel.load_state_dict(parent_state)
    permute_attention_heads(sentinel, False)
    gauge_mask = torch.arange(len(inputs), device=DEVICE) % 2 == 0
    parent_gauge_response = quotient_response(model, inputs[gauge_mask], targets[gauge_mask], candidates)
    gauge_response = quotient_response(gauge, inputs[gauge_mask], targets[gauge_mask], candidates)
    sentinel_response = quotient_response(sentinel, inputs[gauge_mask], targets[gauge_mask], candidates)
    with torch.inference_mode():
        parent_gauge_logits = model(inputs[gauge_mask])[:, -1]
        gauge_logits = gauge(inputs[gauge_mask])[:, -1]
        sentinel_logits = sentinel(inputs[gauge_mask])[:, -1]

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parent_loss = loss_for(model, inputs, targets, candidates)
    parent_loss.backward()
    gradient = flat_grads(model)
    optimizer.step()
    real_update = flat_params(model) - parent_vector
    post_state = copy.deepcopy(optimizer.state_dict())
    real_loss = float(loss_for(model, inputs, targets, candidates).item())
    probe = TinyCausalTransformer(CONFIG).to(DEVICE)
    probe.load_state_dict(parent_state)
    control_update, metrics = select_control(
        probe,
        parent_vector,
        real_update,
        gradient,
        real_loss,
        inputs,
        targets,
        candidates,
        seed + 991,
    )
    control = TinyCausalTransformer(CONFIG).to(DEVICE)
    control.load_state_dict(parent_state)
    assign(control, parent_vector + control_update)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.003, weight_decay=0.001)
    control_optimizer.load_state_dict(post_state)

    indices = torch.arange(len(inputs), device=DEVICE)
    cal = ((indices // 32) + (indices % 32) // 16 + (indices % 16) // 4 + indices % 4) % 2 == 0
    eva = ~cal
    immediate_real_cal = quotient_response(model, inputs[cal], targets[cal], candidates)
    immediate_control_cal = quotient_response(control, inputs[cal], targets[cal], candidates)
    immediate_real_eval = quotient_response(model, inputs[eva], targets[eva], candidates)
    immediate_control_eval = quotient_response(control, inputs[eva], targets[eva], candidates)
    immediate_cal = immediate_real_cal - immediate_control_cal
    immediate_eval = immediate_real_eval - immediate_control_eval
    for _ in range(horizon - 1):
        train_step(model, optimizer, inputs, targets, candidates)
        train_step(control, control_optimizer, inputs, targets, candidates)
    horizon_cal = quotient_response(model, inputs[cal], targets[cal], candidates) - quotient_response(
        control, inputs[cal], targets[cal], candidates
    )
    horizon_eval = quotient_response(model, inputs[eva], targets[eva], candidates) - quotient_response(
        control, inputs[eva], targets[eva], candidates
    )
    with torch.inference_mode():
        real_logits = model(inputs)[:, -1].index_select(-1, candidates)
        control_logits = control(inputs)[:, -1].index_select(-1, candidates)
    return {
        "parent_accuracy": parent_accuracy,
        "parent_loss": float(parent_loss),
        "gauge_logit_max_error": float((parent_gauge_logits - gauge_logits).abs().max()),
        "gauge_response_distance": float(np.linalg.norm(parent_gauge_response - gauge_response)),
        "sentinel_logit_max_change": float((parent_gauge_logits - sentinel_logits).abs().max()),
        "sentinel_response_distance": float(np.linalg.norm(parent_gauge_response - sentinel_response)),
        "real_loss": real_loss,
        "control_loss": float(loss_for(control, inputs, targets, candidates)),
        **metrics,
        "immediate_cal_norm": float(np.linalg.norm(immediate_cal)),
        "immediate_eval_norm": float(np.linalg.norm(immediate_eval)),
        "immediate_cosine": cosine(immediate_cal, immediate_eval),
        "immediate_calibration": immediate_cal.tolist(),
        "immediate_evaluation": immediate_eval.tolist(),
        "horizon_cal_norm": float(np.linalg.norm(horizon_cal)),
        "horizon_eval_norm": float(np.linalg.norm(horizon_eval)),
        "horizon_cosine": cosine(horizon_cal, horizon_eval),
        "horizon_calibration": horizon_cal.tolist(),
        "horizon_evaluation": horizon_eval.tolist(),
        "real_horizon_accuracy": float((real_logits.argmax(-1) == targets).float().mean()),
        "control_horizon_accuracy": float((control_logits.argmax(-1) == targets).float().mean()),
    }


if __name__ == "__main__":
    print(run())
