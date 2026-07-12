#!/usr/bin/env python3
"""Collect replayable component events at five Phase386 semantic coordinates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
    relative_error,
)
from phase371c_blind_vector_contrast import static_roles  # noqa: E402
from phase379_decision_aligned_trace import token_rank  # noqa: E402


PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
DEFAULT_OUT = PHASE_ROOT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "discovery", "calibration", "physical_holdout")
COMPONENTS = (
    "layer_input",
    "input_normalized_state",
    "attention_output",
    "post_attention_state",
    "post_attention_normalized_state",
    "mlp_output",
    "layer_output",
)
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)
MAX_COMPONENT_RELATIVE_ERROR = 0.01
MAX_PRODUCT_RELATIVE_ERROR = 0.01
FROZEN_DTYPE_BY_MODEL = {
    "qwen3": torch.float16,
    "glm4": torch.float16,
    "deepseek7b": torch.bfloat16,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = tensor.detach().contiguous()
    if dtype is not None:
        value = value.to(dtype)
    return value.cpu()


def weight_reference_id(model: str, layer_index: int, component: str) -> str:
    value = f"phase386-weight-ref:{model}:{layer_index}:{component}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_file(split: str) -> Path:
    name = (
        "phase386_instrument_audit_cases.jsonl"
        if split == "instrument_audit"
        else f"phase386_{split}_cases.jsonl"
    )
    return PHASE_ROOT / "protocol/private" / name


def split_authorized(split: str) -> None:
    freeze = read_json(PHASE_ROOT / "phase386_behavior_freeze_summary.json")
    if split == "instrument_audit":
        allowed = freeze["authorization"]["run_instrument_audit"]
    elif split == "discovery":
        gate = PHASE_ROOT / "phase386_instrument_audit_summary.json"
        allowed = gate.is_file() and read_json(gate)["authorization"][
            "discovery_collection"
        ]
    elif split == "calibration":
        gate = PHASE_ROOT / "phase386_discovery_relation_freeze.json"
        allowed = gate.is_file() and read_json(gate)["authorization"][
            "calibration_collection"
        ]
    else:
        gate = PHASE_ROOT / "phase386_physical_holdout_protocol.json"
        allowed = gate.is_file() and read_json(gate)["authorization"].get(
            "physical_holdout_collection", False
        )
    if not allowed:
        raise RuntimeError(f"Phase386 split is not authorized: {split}")


def build_generation_plan(
    loaded: Any, case: dict[str, Any]
) -> dict[str, Any]:
    static, base_length = static_roles(loaded.tokenizer, case)
    base = [
        int(value)
        for value in loaded.tokenizer(
            case["prompt"],
            add_special_tokens=bool(case["tokenization_add_special_tokens"]),
            truncation=True,
            max_length=256,
        )["input_ids"]
    ]
    if len(base) != base_length:
        raise RuntimeError(f"Base token mismatch for {case['blind_case_id']}")
    step = int(case["target_decision_step"])
    generated = [int(value) for value in case["generated_token_ids"]]
    if step < 0 or step + 1 >= len(generated):
        raise RuntimeError(f"Missing target/post token for {case['blind_case_id']}")
    initial_positions = [int(static[0]), int(static[1])]
    if min(initial_positions) < 0 or max(initial_positions) >= len(base):
        raise RuntimeError(
            f"Invalid prompt positions for {case['blind_case_id']}: "
            f"{initial_positions}/{len(base)}"
        )
    return {
        "base": base,
        "generated": generated,
        "target_decision_step": step,
        "source_position": initial_positions[0],
        "query_position": initial_positions[1],
    }


def replay_mlp_channels(
    product: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_size: int = 256,
) -> torch.Tensor:
    replay = torch.zeros(
        (*product.shape[:-1], weight.shape[0]),
        device=product.device,
        dtype=torch.float32,
    )
    for start in range(0, product.shape[-1], chunk_size):
        end = min(product.shape[-1], start + chunk_size)
        replay += F.linear(
            product[..., start:end].float(),
            weight[:, start:end].float(),
        )
    if bias is not None:
        replay += bias.float()
    return replay


@torch.inference_mode()
def capture_forward(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    spec: dict[str, Any],
    audit_neuron_replay: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    sequence = spec["sequence"]
    positions = spec["positions"]
    position_tensor = torch.tensor(
        positions, dtype=torch.long, device=loaded.input_device
    )
    captures.clear()
    input_ids = torch.tensor(
        [sequence], dtype=torch.long, device=loaded.input_device
    )
    output = loaded.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = output.logits[0, -1].detach().float()
    argmax_token = int(torch.argmax(logits).item())
    expected_token = spec["expected_next_token"]
    expected_rank = token_rank(logits, expected_token) if expected_token is not None else None
    transition_match = (
        argmax_token == expected_token and expected_rank == 1
        if expected_token is not None
        else None
    )
    forward_meta = {
        "forward_name": spec["forward_name"],
        "coordinate_names": spec["coordinate_names"],
        "sequence_length": len(sequence),
        "role_positions": positions,
        "expected_next_token_id_private": expected_token,
        "argmax_token_id_private": argmax_token,
        "expected_next_token_rank_private": expected_rank,
        "transition_required": spec["transition_required"],
        "transition_replay_match": transition_match,
        "full_vocabulary_logits": cpu(logits, torch.float32),
    }
    gate_maxima = {
        "attention_source": 0.0,
        "probability": 0.0,
        "mlp_product": 0.0,
        "mlp_direct": 0.0,
        "mlp_neuron": 0.0,
        "block": 0.0,
    }
    layer_frames: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        layer_input = captures[("layer_input", layer_index)]
        norm1 = captures[("norm1", layer_index)]
        attention_output = captures[("attention_output", layer_index)]
        probabilities = captures[("attention_probabilities", layer_index)]
        norm2 = captures[("norm2", layer_index)]
        down_input = captures[("down_proj_input", layer_index)]
        mlp_output = captures[("mlp_output", layer_index)]
        layer_output = captures[("layer_output", layer_index)]
        value_projection = captures[("value_projection", layer_index)]
        post_attention = layer_input + attention_output
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

        selected_input = layer_input.index_select(1, position_tensor)
        selected_norm1 = norm1.index_select(1, position_tensor)
        selected_attention = attention_output.index_select(1, position_tensor)
        selected_post = post_attention.index_select(1, position_tensor)
        selected_norm2 = norm2.index_select(1, position_tensor)
        selected_down = down_input.index_select(1, position_tensor)
        selected_mlp = mlp_output.index_select(1, position_tensor)
        selected_output = layer_output.index_select(1, position_tensor)
        selected_probs = probabilities.index_select(2, position_tensor)
        if loaded.key in {"qwen3", "deepseek7b"}:
            selected_gate_pre = captures[("gate_pre", layer_index)].index_select(
                1, position_tensor
            )
            selected_up = captures[("up", layer_index)].index_select(
                1, position_tensor
            )
            captured_product = layer.mlp.act_fn(selected_gate_pre) * selected_up
            adapter_kind = "separate_gate_up_silu"
        else:
            selected_gate_up = captures[("gate_up", layer_index)].index_select(
                1, position_tensor
            )
            selected_gate_pre, selected_up = selected_gate_up.chunk(2, dim=-1)
            captured_product = selected_up * layer.mlp.activation_fn(selected_gate_pre)
            adapter_kind = "fused_gate_up_silu"

        head_count = int(probabilities.shape[1])
        head_dim = int(o_proj.weight.shape[1] // head_count)
        if value_projection.shape[-1] % head_dim:
            raise RuntimeError(
                f"Invalid value width at {loaded.key}/L{layer_index}: "
                f"{value_projection.shape[-1]} % {head_dim}"
            )
        kv_head_count = int(value_projection.shape[-1] // head_dim)
        values = value_projection.view(
            1, len(sequence), kv_head_count, head_dim
        ).transpose(1, 2)
        repeated_values = values
        if kv_head_count != head_count:
            if head_count % kv_head_count:
                raise RuntimeError(
                    f"Invalid grouped-query heads at {loaded.key}/L{layer_index}"
                )
            repeated_values = values.repeat_interleave(
                head_count // kv_head_count, dim=1
            )
        weighted_heads = torch.einsum(
            "bhqs,bhsd->bhqd",
            selected_probs.float(),
            repeated_values.float(),
        )
        o_blocks = o_proj.weight.float().view(
            o_proj.weight.shape[0], head_count, head_dim
        )
        attention_replay = torch.einsum(
            "bhqd,ohd->bqo", weighted_heads, o_blocks
        )
        if o_proj.bias is not None:
            attention_replay += o_proj.bias.float()
        _, attention_error = relative_error(selected_attention, attention_replay)
        probability_error = float(
            (selected_probs.float().sum(dim=-1) - 1).abs().max().item()
        )
        _, product_error = relative_error(selected_down, captured_product)
        direct = F.linear(selected_down, down_proj.weight, down_proj.bias)
        _, direct_error = relative_error(selected_mlp, direct)
        if audit_neuron_replay:
            neuron = replay_mlp_channels(
                selected_down, down_proj.weight, down_proj.bias
            )
            _, neuron_error = relative_error(selected_mlp, neuron)
        else:
            neuron = None
            neuron_error = None
        block_replay = selected_input + selected_attention + selected_mlp
        _, block_error = relative_error(selected_output, block_replay)
        gates = {
            "attention_source": attention_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "probability": probability_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "mlp_product": product_error <= MAX_PRODUCT_RELATIVE_ERROR,
            "mlp_direct": direct_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "mlp_neuron": (
                neuron_error <= MAX_COMPONENT_RELATIVE_ERROR
                if neuron_error is not None
                else None
            ),
            "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
        }
        for key, value in {
            "attention_source": attention_error,
            "probability": probability_error,
            "mlp_product": product_error,
            "mlp_direct": direct_error,
            "mlp_neuron": neuron_error,
            "block": block_error,
        }.items():
            if value is not None:
                gate_maxima[key] = max(gate_maxima[key], value)
        layer_frames.append(
            {
                "forward_name": spec["forward_name"],
                "coordinate_names": spec["coordinate_names"],
                "role_positions": positions,
                "sequence_length": len(sequence),
                "component_vectors": {
                    "layer_input": cpu(selected_input),
                    "input_normalized_state": cpu(selected_norm1),
                    "attention_output": cpu(selected_attention),
                    "post_attention_state": cpu(selected_post),
                    "post_attention_normalized_state": cpu(selected_norm2),
                    "mlp_output": cpu(selected_mlp),
                    "layer_output": cpu(selected_output),
                },
                "attention": {
                    "value_states_all_sources": cpu(values, torch.float16),
                    "probabilities_receivers_all_sources": cpu(
                        selected_probs, torch.float16
                    ),
                    "head_count": head_count,
                    "key_value_head_count": kv_head_count,
                    "head_dim": head_dim,
                },
                "mlp": {
                    "adapter_kind": adapter_kind,
                    "gate_pre_at_coordinates": cpu(
                        selected_gate_pre, torch.float16
                    ),
                    "up_at_coordinates": cpu(selected_up, torch.float16),
                    "down_projection_input_product_at_coordinates": cpu(
                        selected_down, torch.float16
                    ),
                    "channel_count": int(selected_down.shape[-1]),
                },
                "quality": {
                    "errors": {
                        "attention_source": attention_error,
                        "probability": probability_error,
                        "mlp_product": product_error,
                        "mlp_direct": direct_error,
                        "mlp_neuron": neuron_error,
                        "block": block_error,
                    },
                    "gates": gates,
                    "neuron_replay_audited": audit_neuron_replay,
                    "required_gates_pass": all(
                        value for value in gates.values() if value is not None
                    ),
                },
            }
        )
        del (
            weighted_heads,
            attention_replay,
            direct,
            neuron,
            block_replay,
            captured_product,
        )
    del output, logits, input_ids, position_tensor
    captures.clear()
    return forward_meta, layer_frames, gate_maxima


@torch.inference_mode()
def capture_incremental_call(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    *,
    call_name: str,
    input_token_ids: list[int],
    total_sequence_length: int,
    local_positions: list[int],
    global_positions: list[int],
    coordinate_names: list[str],
    expected_next_token: int | None,
    transition_required: bool,
    past_key_values: Any,
    value_history: list[torch.Tensor | None],
    audit_neuron_replay: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any] | None],
    dict[str, float],
    Any,
    list[torch.Tensor],
]:
    if len(local_positions) != len(coordinate_names):
        raise ValueError("Local positions and coordinate names differ")
    if len(global_positions) != len(coordinate_names):
        raise ValueError("Global positions and coordinate names differ")
    captures.clear()
    input_ids = torch.tensor(
        [input_token_ids], dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.ones(
        (1, total_sequence_length),
        dtype=torch.long,
        device=loaded.input_device,
    )
    output = loaded.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = output.logits[0, -1].detach().float()
    argmax_token = int(torch.argmax(logits).item())
    expected_rank = (
        token_rank(logits, expected_next_token)
        if expected_next_token is not None
        else None
    )
    transition_match = (
        argmax_token == expected_next_token
        if expected_next_token is not None
        else None
    )
    call_meta = {
        "call_name": call_name,
        "input_token_count": len(input_token_ids),
        "total_sequence_length": total_sequence_length,
        "coordinate_names": coordinate_names,
        "global_positions": global_positions,
        "expected_next_token_id_private": expected_next_token,
        "argmax_token_id_private": argmax_token,
        "expected_next_token_rank_private": expected_rank,
        "transition_required": transition_required,
        "transition_replay_match": transition_match,
        "full_vocabulary_logits": cpu(logits, torch.float32),
    }
    gate_maxima = {
        "attention_source": 0.0,
        "probability": 0.0,
        "mlp_product": 0.0,
        "mlp_direct": 0.0,
        "mlp_neuron": 0.0,
        "block": 0.0,
    }
    position_tensor = (
        torch.tensor(local_positions, dtype=torch.long, device=loaded.input_device)
        if local_positions
        else None
    )
    next_history: list[torch.Tensor] = []
    layer_frames: list[dict[str, Any] | None] = []
    for layer_index, layer in enumerate(layers):
        probabilities = captures[("attention_probabilities", layer_index)]
        value_projection = captures[("value_projection", layer_index)]
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_count = int(probabilities.shape[1])
        head_dim = int(o_proj.weight.shape[1] // head_count)
        if value_projection.shape[-1] % head_dim:
            raise RuntimeError(
                f"Invalid value width at {loaded.key}/L{layer_index}: "
                f"{value_projection.shape[-1]} % {head_dim}"
            )
        kv_head_count = int(value_projection.shape[-1] // head_dim)
        current_values = value_projection.view(
            1, len(input_token_ids), kv_head_count, head_dim
        ).transpose(1, 2)
        previous_values = value_history[layer_index]
        all_values = (
            current_values
            if previous_values is None
            else torch.cat([previous_values, current_values], dim=2)
        )
        if all_values.shape[2] != total_sequence_length:
            raise RuntimeError(
                f"Value-history length mismatch at {loaded.key}/L{layer_index}: "
                f"{all_values.shape[2]} != {total_sequence_length}"
            )
        next_history.append(all_values.detach())
        if position_tensor is None:
            layer_frames.append(None)
            continue

        layer_input = captures[("layer_input", layer_index)]
        norm1 = captures[("norm1", layer_index)]
        attention_output = captures[("attention_output", layer_index)]
        norm2 = captures[("norm2", layer_index)]
        down_input = captures[("down_proj_input", layer_index)]
        mlp_output = captures[("mlp_output", layer_index)]
        layer_output = captures[("layer_output", layer_index)]
        post_attention = layer_input + attention_output
        down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

        selected_input = layer_input.index_select(1, position_tensor)
        selected_norm1 = norm1.index_select(1, position_tensor)
        selected_attention = attention_output.index_select(1, position_tensor)
        selected_post = post_attention.index_select(1, position_tensor)
        selected_norm2 = norm2.index_select(1, position_tensor)
        selected_down = down_input.index_select(1, position_tensor)
        selected_mlp = mlp_output.index_select(1, position_tensor)
        selected_output = layer_output.index_select(1, position_tensor)
        selected_probs = probabilities.index_select(2, position_tensor)
        if selected_probs.shape[-1] != all_values.shape[2]:
            raise RuntimeError(
                f"Attention/value source mismatch at {loaded.key}/L{layer_index}"
            )
        if loaded.key in {"qwen3", "deepseek7b"}:
            selected_gate_pre = captures[("gate_pre", layer_index)].index_select(
                1, position_tensor
            )
            selected_up = captures[("up", layer_index)].index_select(
                1, position_tensor
            )
            captured_product = layer.mlp.act_fn(selected_gate_pre) * selected_up
            adapter_kind = "separate_gate_up_silu"
        else:
            selected_gate_up = captures[("gate_up", layer_index)].index_select(
                1, position_tensor
            )
            selected_gate_pre, selected_up = selected_gate_up.chunk(2, dim=-1)
            captured_product = selected_up * layer.mlp.activation_fn(selected_gate_pre)
            adapter_kind = "fused_gate_up_silu"

        repeated_values = all_values
        if kv_head_count != head_count:
            if head_count % kv_head_count:
                raise RuntimeError(
                    f"Invalid grouped-query heads at {loaded.key}/L{layer_index}"
                )
            repeated_values = all_values.repeat_interleave(
                head_count // kv_head_count, dim=1
            )
        weighted_heads = torch.einsum(
            "bhqs,bhsd->bhqd",
            selected_probs.float(),
            repeated_values.float(),
        )
        o_blocks = o_proj.weight.float().view(
            o_proj.weight.shape[0], head_count, head_dim
        )
        attention_replay = torch.einsum(
            "bhqd,ohd->bqo", weighted_heads, o_blocks
        )
        if o_proj.bias is not None:
            attention_replay += o_proj.bias.float()
        _, attention_error = relative_error(selected_attention, attention_replay)
        probability_error = float(
            (selected_probs.float().sum(dim=-1) - 1).abs().max().item()
        )
        _, product_error = relative_error(selected_down, captured_product)
        direct = F.linear(selected_down, down_proj.weight, down_proj.bias)
        _, direct_error = relative_error(selected_mlp, direct)
        if audit_neuron_replay:
            neuron = replay_mlp_channels(
                selected_down, down_proj.weight, down_proj.bias
            )
            _, neuron_error = relative_error(selected_mlp, neuron)
        else:
            neuron = None
            neuron_error = None
        block_replay = selected_input + selected_attention + selected_mlp
        _, block_error = relative_error(selected_output, block_replay)
        gates = {
            "attention_source": attention_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "probability": probability_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "mlp_product": product_error <= MAX_PRODUCT_RELATIVE_ERROR,
            "mlp_direct": direct_error <= MAX_COMPONENT_RELATIVE_ERROR,
            "mlp_neuron": (
                neuron_error <= MAX_COMPONENT_RELATIVE_ERROR
                if neuron_error is not None
                else None
            ),
            "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
        }
        for key, value in {
            "attention_source": attention_error,
            "probability": probability_error,
            "mlp_product": product_error,
            "mlp_direct": direct_error,
            "mlp_neuron": neuron_error,
            "block": block_error,
        }.items():
            if value is not None:
                gate_maxima[key] = max(gate_maxima[key], value)
        layer_frames.append(
            {
                "call_name": call_name,
                "coordinate_names": coordinate_names,
                "global_positions": global_positions,
                "total_sequence_length": total_sequence_length,
                "component_vectors": {
                    "layer_input": cpu(selected_input),
                    "input_normalized_state": cpu(selected_norm1),
                    "attention_output": cpu(selected_attention),
                    "post_attention_state": cpu(selected_post),
                    "post_attention_normalized_state": cpu(selected_norm2),
                    "mlp_output": cpu(selected_mlp),
                    "layer_output": cpu(selected_output),
                },
                "attention": {
                    "coordinate_names": coordinate_names,
                    "global_positions": global_positions,
                    "value_states_all_sources": cpu(all_values, torch.float16),
                    "probabilities_receivers_all_sources": cpu(
                        selected_probs, torch.float16
                    ),
                    "head_count": head_count,
                    "key_value_head_count": kv_head_count,
                    "head_dim": head_dim,
                },
                "mlp": {
                    "adapter_kind": adapter_kind,
                    "gate_pre_at_coordinates": cpu(
                        selected_gate_pre, torch.float16
                    ),
                    "up_at_coordinates": cpu(selected_up, torch.float16),
                    "down_projection_input_product_at_coordinates": cpu(
                        selected_down, torch.float16
                    ),
                    "channel_count": int(selected_down.shape[-1]),
                },
                "quality": {
                    "errors": {
                        "attention_source": attention_error,
                        "probability": probability_error,
                        "mlp_product": product_error,
                        "mlp_direct": direct_error,
                        "mlp_neuron": neuron_error,
                        "block": block_error,
                    },
                    "gates": gates,
                    "neuron_replay_audited": audit_neuron_replay,
                    "required_gates_pass": all(
                        value for value in gates.values() if value is not None
                    ),
                },
            }
        )
        del (
            weighted_heads,
            attention_replay,
            direct,
            neuron,
            block_replay,
            captured_product,
        )
    next_past = output.past_key_values
    del output, logits, input_ids, attention_mask, position_tensor
    captures.clear()
    return call_meta, layer_frames, gate_maxima, next_past, next_history


def capture_case_incremental(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    case: dict[str, Any],
    audit_neuron_replay: bool,
) -> tuple[
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    dict[str, float],
    int,
]:
    plan = build_generation_plan(loaded, case)
    base = plan["base"]
    generated = plan["generated"]
    step = plan["target_decision_step"]
    past = None
    value_history: list[torch.Tensor | None] = [None] * len(layers)
    call_meta_rows: list[dict[str, Any]] = []
    frames_by_layer: list[list[dict[str, Any]]] = [
        [] for _ in range(len(layers))
    ]
    maxima = {
        "attention_source": 0.0,
        "probability": 0.0,
        "mlp_product": 0.0,
        "mlp_direct": 0.0,
        "mlp_neuron": 0.0,
        "block": 0.0,
    }

    def execute(
        *,
        call_name: str,
        input_token_ids: list[int],
        total_sequence_length: int,
        local_positions: list[int],
        global_positions: list[int],
        coordinate_names: list[str],
        expected_next_token: int | None,
        transition_required: bool,
    ) -> None:
        nonlocal past, value_history
        meta, layer_frames, call_maxima, past, value_history = (
            capture_incremental_call(
                loaded,
                layers,
                captures,
                call_name=call_name,
                input_token_ids=input_token_ids,
                total_sequence_length=total_sequence_length,
                local_positions=local_positions,
                global_positions=global_positions,
                coordinate_names=coordinate_names,
                expected_next_token=expected_next_token,
                transition_required=transition_required,
                past_key_values=past,
                value_history=value_history,
                audit_neuron_replay=audit_neuron_replay,
            )
        )
        call_meta_rows.append(meta)
        for layer_index, frame in enumerate(layer_frames):
            if frame is not None:
                frames_by_layer[layer_index].append(frame)
        for key, value in call_maxima.items():
            maxima[key] = max(maxima[key], value)

    initial_names = [COORDINATES[0], COORDINATES[1]]
    initial_positions = [plan["source_position"], plan["query_position"]]
    if step == 0:
        initial_names.append(COORDINATES[2])
        initial_positions.append(len(base) - 1)
    execute(
        call_name="prompt_forward",
        input_token_ids=base,
        total_sequence_length=len(base),
        local_positions=initial_positions,
        global_positions=initial_positions,
        coordinate_names=initial_names,
        expected_next_token=generated[0],
        transition_required=True,
    )
    for token_index in range(step):
        record_predecision = token_index == step - 1
        execute(
            call_name=f"prefix_token_{token_index}_forward",
            input_token_ids=[generated[token_index]],
            total_sequence_length=len(base) + token_index + 1,
            local_positions=[0] if record_predecision else [],
            global_positions=[len(base) + token_index] if record_predecision else [],
            coordinate_names=[COORDINATES[2]] if record_predecision else [],
            expected_next_token=generated[token_index + 1],
            transition_required=True,
        )
    execute(
        call_name="target_encoded_forward",
        input_token_ids=[generated[step]],
        total_sequence_length=len(base) + step + 1,
        local_positions=[0],
        global_positions=[len(base) + step],
        coordinate_names=[COORDINATES[3]],
        expected_next_token=generated[step + 1],
        transition_required=True,
    )
    post_expected = generated[step + 2] if step + 2 < len(generated) else None
    execute(
        call_name="post_decision_forward",
        input_token_ids=[generated[step + 1]],
        total_sequence_length=len(base) + step + 2,
        local_positions=[0],
        global_positions=[len(base) + step + 1],
        coordinate_names=[COORDINATES[4]],
        expected_next_token=post_expected,
        transition_required=post_expected is not None,
    )
    for layer_frames in frames_by_layer:
        flat = [
            name for frame in layer_frames for name in frame["coordinate_names"]
        ]
        if flat != list(COORDINATES):
            raise RuntimeError(
                f"Incremental coordinate order mismatch for {case['blind_case_id']}: {flat}"
            )
    model_call_count = len(call_meta_rows)
    del past, value_history
    return call_meta_rows, frames_by_layer, maxima, model_call_count


@torch.inference_mode()
def run_model(
    model: str,
    split: str,
    output_root: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    split_authorized(split)
    rows = read_jsonl(case_file(split))
    cases = [row for row in rows if row["private_execution_model"] == model]
    if not cases:
        raise RuntimeError(f"No Phase386 cases for {model}/{split}")
    if any(row.get("semantic_labels_available_to_collection", True) for row in cases):
        raise RuntimeError("Phase386 collection received semantic labels")
    loaded = None
    handles: list[Any] = []
    value_handles: list[Any] = []
    mlp_handles: list[Any] = []
    files: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    manifest_maxima = {
        "attention_source": 0.0,
        "probability": 0.0,
        "mlp_product": 0.0,
        "mlp_direct": 0.0,
        "mlp_neuron": 0.0,
        "block": 0.0,
    }
    try:
        loaded = load_probe_model(model)
        runtime_dtype = next(loaded.model.parameters()).dtype
        if runtime_dtype != FROZEN_DTYPE_BY_MODEL[model]:
            raise RuntimeError(
                f"Phase386 runtime dtype mismatch for {model}: "
                f"{runtime_dtype} != {FROZEN_DTYPE_BY_MODEL[model]}"
            )
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for layer_index, layer in enumerate(layers):
            value_proj = module_attr(layer.self_attn, ("v_proj", "value"))

            def value_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> None:
                captures[("value_projection", idx)] = output.detach()

            value_handles.append(value_proj.register_forward_hook(value_post))
            if model in {"qwen3", "deepseek7b"}:

                def gate_post(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    idx: int = layer_index,
                ) -> None:
                    captures[("gate_pre", idx)] = output.detach()

                def up_post(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    idx: int = layer_index,
                ) -> None:
                    captures[("up", idx)] = output.detach()

                mlp_handles.extend(
                    [
                        layer.mlp.gate_proj.register_forward_hook(gate_post),
                        layer.mlp.up_proj.register_forward_hook(up_post),
                    ]
                )
            else:

                def gate_up_post(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    idx: int = layer_index,
                ) -> None:
                    captures[("gate_up", idx)] = output.detach()

                mlp_handles.append(
                    layer.mlp.gate_up_proj.register_forward_hook(gate_up_post)
                )

        audit_neuron_replay = split == "instrument_audit"
        total_model_call_count = 0
        for case_index, case in enumerate(cases, 1):
            frame_meta, frames_by_layer, case_maxima, model_call_count = (
                capture_case_incremental(
                    loaded,
                    layers,
                    captures,
                    case,
                    audit_neuron_replay,
                )
            )
            total_model_call_count += model_call_count
            for key, value in case_maxima.items():
                manifest_maxima[key] = max(manifest_maxima[key], value)

            required_transition_pass = all(
                (not row["transition_required"]) or row["transition_replay_match"]
                for row in frame_meta
            )
            case_root = (
                output_root
                / split
                / "private/models"
                / model
                / case["blind_case_id"]
            )
            meta_payload = {
                "schema_version": "60.5.0",
                "phase_id": "Phase386-IncrementalEventCollection",
                "blind_case_id": case["blind_case_id"],
                "anonymous_model_id": case["anonymous_model_id"],
                "public_parallel_group_id": case[
                    "phase386_public_parallel_group_id"
                ],
                "anonymous_condition_slot": case["anonymous_condition_slot"],
                "semantic_coordinates": list(COORDINATES),
                "model_call_count": model_call_count,
                "recorded_semantic_coordinate_count": 5,
                "incremental_kv_cache_path": True,
                "independent_clock_times_claimed": False,
                "target_decision_step": int(case["target_decision_step"]),
                "runtime_dtype": str(runtime_dtype).replace("torch.", ""),
                "generation_calls": frame_meta,
                "required_transition_pass": required_transition_pass,
                "physical_holdout": split == "physical_holdout",
            }
            meta_path = case_root / "multitime_meta.pt"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meta_payload, meta_path)
            case_files = [meta_path]
            for layer_index, layer_frames in enumerate(frames_by_layer):
                flat_coordinates = [
                    name
                    for frame in layer_frames
                    for name in frame["coordinate_names"]
                ]
                if flat_coordinates != list(COORDINATES):
                    raise RuntimeError(
                        f"Coordinate order mismatch for {case['blind_case_id']}"
                    )
                payload = {
                    "schema_version": "60.5.0",
                    "phase_id": "Phase386-IncrementalEventCollection",
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_model_id": case["anonymous_model_id"],
                    "public_parallel_group_id": case[
                        "phase386_public_parallel_group_id"
                    ],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "layer_index": layer_index,
                    "coordinate_names": flat_coordinates,
                    "component_vectors": {
                        component: torch.cat(
                            [
                                frame["component_vectors"][component]
                                for frame in layer_frames
                            ],
                            dim=1,
                        )
                        for component in COMPONENTS
                    },
                    "attention": {
                        "frames": [frame["attention"] for frame in layer_frames],
                        "output_projection_weight_reference_id": weight_reference_id(
                            model, layer_index, "o_proj.weight"
                        ),
                        "lazy_exact_event_family": (
                            "semantic_coordinate x receiver x head x source_position; "
                            "reconstruct from probability, value state, and output "
                            "projection slice"
                        ),
                    },
                    "mlp": {
                        "adapter_kind": layer_frames[0]["mlp"]["adapter_kind"],
                        "gate_pre_at_coordinates": torch.cat(
                            [
                                frame["mlp"]["gate_pre_at_coordinates"]
                                for frame in layer_frames
                            ],
                            dim=1,
                        ),
                        "up_at_coordinates": torch.cat(
                            [
                                frame["mlp"]["up_at_coordinates"]
                                for frame in layer_frames
                            ],
                            dim=1,
                        ),
                        "down_projection_input_product_at_coordinates": torch.cat(
                            [
                                frame["mlp"][
                                    "down_projection_input_product_at_coordinates"
                                ]
                                for frame in layer_frames
                            ],
                            dim=1,
                        ),
                        "channel_count": layer_frames[0]["mlp"]["channel_count"],
                        "down_projection_weight_reference_id": weight_reference_id(
                            model, layer_index, "mlp.down_proj.weight"
                        ),
                        "lazy_exact_event_family": (
                            "semantic_coordinate x receiver x channel; reconstruct "
                            "from product scalar and down-projection column"
                        ),
                    },
                    "quality": {
                        "coordinate_frames": [
                            frame["quality"] for frame in layer_frames
                        ],
                        "all_required_gates_pass": all(
                            frame["quality"]["required_gates_pass"]
                            for frame in layer_frames
                        ),
                    },
                }
                path = case_root / f"layer_{layer_index:03d}.pt"
                torch.save(payload, path)
                case_files.append(path)
            case_bytes = sum(path.stat().st_size for path in case_files)
            all_layer_gates = all(
                all(
                    frame["quality"]["required_gates_pass"]
                    for frame in layer_frames
                )
                for layer_frames in frames_by_layer
            )
            case_rows.append(
                {
                    "blind_case_id": case["blind_case_id"],
                    "public_parallel_group_id": case[
                        "phase386_public_parallel_group_id"
                    ],
                    "anonymous_model_id": case["anonymous_model_id"],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "mechanism_id_private": case["mechanism_id"],
                    "contrast_condition_private": case["contrast_condition"],
                    "file_count": len(case_files),
                    "byte_count": case_bytes,
                    "required_transition_pass": required_transition_pass,
                    "all_layer_gates_pass": all_layer_gates,
                    "gate_maxima": case_maxima,
                    "model_call_count": model_call_count,
                }
            )
            for path in case_files:
                files.append(
                    {
                        "blind_case_id": case["blind_case_id"],
                        "relative_path": str(path.relative_to(output_root)),
                        "byte_count": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            print(
                f"[{model}/{split}] {case_index}/{len(cases)} "
                f"files={len(case_files)} bytes={case_bytes}",
                flush=True,
            )
            del frames_by_layer, frame_meta, meta_payload
            gc.collect()

        model_root = output_root / split / "models" / model
        manifest = {
            "schema_version": "60.5.0",
            "phase_id": "Phase386-IncrementalEventCollection",
            "created_at": now(),
            "model": model,
            "split": split,
            "runtime_dtype": str(runtime_dtype).replace("torch.", ""),
            "case_count": len(case_rows),
            "parallel_group_count": len(
                {row["public_parallel_group_id"] for row in case_rows}
            ),
            "layer_count": len(layers),
            "semantic_coordinate_count": len(COORDINATES),
            "model_call_count": total_model_call_count,
            "minimum_model_call_count_per_case": 3,
            "fixed_three_forward_pass_claimed": False,
            "incremental_kv_cache_path": True,
            "file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "required_transition_pass_count": sum(
                row["required_transition_pass"] for row in case_rows
            ),
            "all_case_gates_pass": all(
                row["all_layer_gates_pass"] for row in case_rows
            ),
            "gate_maxima": manifest_maxima,
            "case_rows": case_rows,
            "files": files,
            "neuron_replay_audited": audit_neuron_replay,
            "top_k_used": False,
            "pairwise_gram_materialized": False,
            "physical_holdout_opened": split == "physical_holdout",
            "valid": (
                all(row["required_transition_pass"] for row in case_rows)
                and all(row["all_layer_gates_pass"] for row in case_rows)
            ),
        }
        write_json(model_root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"files", "case_rows"}
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return manifest
    finally:
        for handle in [*handles, *value_handles, *mlp_handles]:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_model(args.model, args.split, args.output_root)
