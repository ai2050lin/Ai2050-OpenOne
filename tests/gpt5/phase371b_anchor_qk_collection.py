#!/usr/bin/env python3
"""Collect exact all-token Q/K conservation trees at three anchor layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase338_block_causal_screen import prompt_ids  # noqa: E402
from phase358_multiresolution_component_conservation import module_attr, relative_error  # noqa: E402


CASE_FILE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_collection_freeze/private/phase369_collection_execution_cases.jsonl"
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/anchor_qk_engineering"
MODELS = ("qwen3", "glm4", "deepseek7b")
GENERATION_TIME_COUNT = 3
PARTITION_COUNT = 8
COMPONENT_ERROR_GATE = 0.01
TREE_ERROR_GATE = 1e-5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def anchors(layer_count: int) -> tuple[int, int, int]:
    return 0, layer_count // 2, layer_count - 1


def contiguous_partitions(size: int, count: int = PARTITION_COUNT) -> list[tuple[int, int]]:
    return [
        (math.floor(index * size / count), math.floor((index + 1) * size / count))
        for index in range(count)
    ]


def weight_reference_id(model: str, layer_index: int, component: str) -> str:
    value = f"phase371b-weight-ref:{model}:{layer_index}:{component}"
    return hashlib.sha256(value.encode()).hexdigest()


def selected_case(model: str) -> dict[str, Any]:
    rows = [row for row in read_jsonl(CASE_FILE) if row["private_execution_model"] == model]
    if not rows:
        raise RuntimeError(f"No Phase369 execution cases found for {model}")
    return min(rows, key=lambda row: row["blind_case_id"])


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Expected tensor component, got {type(value).__name__}")


def install_anchor_hooks(
    layers: list[Any],
    anchor_layers: tuple[int, int, int],
    captures: dict[tuple[str, int], torch.Tensor],
) -> list[Any]:
    handles = []
    for layer_index in anchor_layers:
        layer = layers[layer_index]
        input_norm = module_attr(layer, ("input_layernorm", "input_layer_norm", "ln_1"))
        post_norm = module_attr(layer, ("post_attention_layernorm", "post_attention_layer_norm", "ln_2"))
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

        def layer_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("layer_input", idx)] = inputs[0].detach()

        def norm1_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("norm1", idx)] = component_tensor(output).detach()

        def o_proj_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("o_proj_input", idx)] = inputs[0].detach()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("attention_output", idx)] = component_tensor(output).detach()
            if isinstance(output, (tuple, list)) and len(output) > 1 and torch.is_tensor(output[1]):
                captures[("attention_probabilities", idx)] = output[1].detach()

        def norm2_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("norm2", idx)] = component_tensor(output).detach()

        def down_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("down_input", idx)] = inputs[0].detach()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("mlp_output", idx)] = component_tensor(output).detach()

        def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("layer_output", idx)] = component_tensor(output).detach()

        handles.extend([
            layer.register_forward_pre_hook(layer_pre),
            input_norm.register_forward_hook(norm1_post),
            o_proj.register_forward_pre_hook(o_proj_pre),
            layer.self_attn.register_forward_hook(attention_post),
            post_norm.register_forward_hook(norm2_post),
            down_proj.register_forward_pre_hook(down_pre),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(layer_post),
        ])
    return handles


def attention_module_for(model: str) -> Any:
    if model == "qwen3":
        import transformers.models.qwen3.modeling_qwen3 as modeling
    elif model == "glm4":
        import transformers.models.glm.modeling_glm as modeling
    else:
        import transformers.models.qwen2.modeling_qwen2 as modeling
    return modeling


@contextmanager
def capture_actual_qkv(
    model: str,
    anchor_layers: tuple[int, int, int],
    captures: dict[tuple[str, int], torch.Tensor],
) -> Iterator[None]:
    modeling = attention_module_for(model)
    original = modeling.eager_attention_forward
    anchor_set = set(anchor_layers)

    def wrapped(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_index = int(module.layer_idx)
        if layer_index in anchor_set:
            captures[("query", layer_index)] = query.detach()
            captures[("key", layer_index)] = key.detach()
            captures[("value", layer_index)] = value.detach()
            if attention_mask is not None:
                captures[("attention_mask", layer_index)] = attention_mask.detach()
            captures[("attention_scaling", layer_index)] = torch.tensor(float(scaling))
        return original(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )

    modeling.eager_attention_forward = wrapped
    try:
        yield
    finally:
        modeling.eager_attention_forward = original


def repeat_key_value(value: torch.Tensor, head_count: int) -> torch.Tensor:
    if value.shape[1] == head_count:
        return value
    if head_count % value.shape[1] != 0:
        raise RuntimeError(f"Cannot repeat {value.shape[1]} KV heads to {head_count} heads")
    return value.repeat_interleave(head_count // value.shape[1], dim=1)


def build_attention_tree(
    layer: Any,
    captures: dict[tuple[str, int], torch.Tensor],
    layer_index: int,
    materialize_derivatives: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    query_native = captures[("query", layer_index)]
    key_native = captures[("key", layer_index)]
    value_native = captures[("value", layer_index)]
    probabilities_native = captures[("attention_probabilities", layer_index)]
    mask = captures.get(("attention_mask", layer_index))
    scaling = float(captures[("attention_scaling", layer_index)].item())
    head_count = int(query_native.shape[1])
    repeated_key_native = repeat_key_value(key_native, head_count)
    scores = torch.matmul(query_native, repeated_key_native.transpose(2, 3)) * scaling
    if mask is not None:
        scores = scores + mask[..., : key_native.shape[-2]]
    replayed_probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_native.dtype)
    _, probability_error = relative_error(probabilities_native, replayed_probabilities)

    query = query_native.float()
    key = key_native.float()
    value = value_native.float()
    probabilities = probabilities_native.float()
    repeated_value = repeat_key_value(value, head_count)
    weighted = torch.matmul(probabilities, repeated_value)
    expected_o_input = captures[("o_proj_input", layer_index)].float()
    replayed_o_input = weighted.transpose(1, 2).contiguous().reshape_as(expected_o_input)
    _, o_input_error = relative_error(expected_o_input, replayed_o_input)
    o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
    head_dim = int(query.shape[-1])
    blocks = o_proj.weight.float().view(o_proj.weight.shape[0], head_count, head_dim)
    head_writes = torch.einsum("bhqd,ohd->bqho", weighted, blocks)
    direct = head_writes.sum(dim=2)
    if o_proj.bias is not None:
        direct = direct + o_proj.bias.float()
    expected_attention = captures[("attention_output", layer_index)].float()
    _, direct_error = relative_error(expected_attention, direct)
    partition_writes = torch.stack([
        head_writes[:, :, start:end].sum(dim=2)
        for start, end in contiguous_partitions(head_count)
    ])
    tree_parent = partition_writes.sum(dim=0)
    if o_proj.bias is not None:
        tree_parent = tree_parent + o_proj.bias.float()
    _, tree_error = relative_error(direct, tree_parent)
    state = {
        "query_states_all_positions": cpu(query_native),
        "key_states_all_positions": cpu(key_native),
        "value_states_all_positions": cpu(value_native),
        "probabilities_all_receivers_all_sources": cpu(probabilities_native),
        "head_partitions": contiguous_partitions(head_count),
        "head_count": head_count,
        "key_value_head_count": int(key.shape[1]),
        "head_dim": head_dim,
        "scaling": scaling,
    }
    if materialize_derivatives:
        state["head_writes_all_receivers"] = cpu(head_writes, torch.float16)
        state["head_partition_writes_all_receivers"] = cpu(partition_writes, torch.float16)
    return state, {
        "query_key_probability": probability_error,
        "attention_pre_projection": o_input_error,
        "attention_direct": direct_error,
        "attention_tree": tree_error,
    }


def build_mlp_tree(
    layer: Any,
    captures: dict[tuple[str, int], torch.Tensor],
    layer_index: int,
    materialize_derivatives: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    product = captures[("down_input", layer_index)].float()
    expected = captures[("mlp_output", layer_index)].float()
    down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))
    direct = F.linear(product, down_proj.weight.float(), down_proj.bias.float() if down_proj.bias is not None else None)
    _, direct_error = relative_error(expected, direct)
    partitions = contiguous_partitions(int(product.shape[-1]))
    partition_writes = torch.stack([
        F.linear(product[..., start:end], down_proj.weight[:, start:end].float())
        for start, end in partitions
    ])
    tree_parent = partition_writes.sum(dim=0)
    if down_proj.bias is not None:
        tree_parent = tree_parent + down_proj.bias.float()
    _, tree_error = relative_error(direct, tree_parent)
    state = {
        "down_projection_input_product_all_positions": cpu(product, torch.float16),
        "channel_partitions": partitions,
        "channel_count": int(product.shape[-1]),
    }
    if materialize_derivatives:
        state["partition_writes_all_receivers"] = cpu(partition_writes, torch.float16)
    return state, {"mlp_direct": direct_error, "mlp_tree": tree_error}


def use_sufficient_state_storage(payload: dict[str, Any]) -> dict[str, Any]:
    payload["attention"].pop("head_writes_all_receivers", None)
    payload["attention"].pop("head_partition_writes_all_receivers", None)
    payload["mlp"].pop("partition_writes_all_receivers", None)
    payload["derivation_contract"] = {
        "attention_head_writes": "derive_from_probabilities_value_states_and_o_projection_weight",
        "attention_partition_writes": "sum_heads_over_frozen_partitions",
        "mlp_single_neuron_write": "product_channel_times_down_projection_weight_column",
        "mlp_partition_writes": "sum_neurons_over_frozen_partitions",
        "materialized_derivatives_retained": False,
    }
    return payload


@torch.inference_mode()
def run_model(
    model: str,
    storage_mode: str = "materialized",
    output_root: Path = OUT,
) -> dict[str, Any]:
    if storage_mode not in {"materialized", "sufficient"}:
        raise ValueError(f"Unsupported storage mode: {storage_mode}")
    case = selected_case(model)
    loaded = None
    handles: list[Any] = []
    captures: dict[tuple[str, int], torch.Tensor] = {}
    files: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        anchor_layers = anchors(len(layers))
        handles = install_anchor_hooks(layers, anchor_layers, captures)
        base_ids = prompt_ids(loaded, case)
        sequence = list(base_ids)
        with capture_actual_qkv(model, anchor_layers, captures):
            for generation_time in range(GENERATION_TIME_COUNT):
                captures.clear()
                input_ids = torch.tensor([sequence], dtype=torch.long, device=loaded.input_device)
                output = loaded.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                next_token = int(output.logits[0, -1].argmax().item())
                for layer_index in anchor_layers:
                    layer = layers[layer_index]
                    attention, attention_errors = build_attention_tree(layer, captures, layer_index)
                    mlp, mlp_errors = build_mlp_tree(layer, captures, layer_index)
                    layer_input = captures[("layer_input", layer_index)].float()
                    attention_output = captures[("attention_output", layer_index)].float()
                    mlp_output = captures[("mlp_output", layer_index)].float()
                    expected_output = captures[("layer_output", layer_index)].float()
                    block_replay = layer_input + attention_output + mlp_output
                    _, block_error = relative_error(expected_output, block_replay)
                    errors = {**attention_errors, **mlp_errors, "block": block_error}
                    gates = {
                        "query_key_probability": errors["query_key_probability"] <= COMPONENT_ERROR_GATE,
                        "attention_pre_projection": errors["attention_pre_projection"] <= COMPONENT_ERROR_GATE,
                        "attention_direct": errors["attention_direct"] <= COMPONENT_ERROR_GATE,
                        "attention_tree": errors["attention_tree"] <= TREE_ERROR_GATE,
                        "mlp_direct": errors["mlp_direct"] <= COMPONENT_ERROR_GATE,
                        "mlp_tree": errors["mlp_tree"] <= TREE_ERROR_GATE,
                        "block": errors["block"] <= COMPONENT_ERROR_GATE,
                    }
                    payload = {
                        "schema_version": "47.2.0",
                        "phase_id": "Phase371B",
                        "blind_case_id": case["blind_case_id"],
                        "anonymous_model_id": case["anonymous_model_id"],
                        "anonymous_group_id": case["anonymous_group_id"],
                        "anonymous_condition_slot": case["anonymous_condition_slot"],
                        "generation_time": generation_time,
                        "layer_index": layer_index,
                        "sequence_length": len(sequence),
                        "component_vectors": {
                            "layer_input_all_positions": cpu(layer_input, torch.float16),
                            "input_normalized_state_all_positions": cpu(captures[("norm1", layer_index)], torch.float16),
                            "attention_output_all_positions": cpu(attention_output, torch.float16),
                            "post_attention_state_all_positions": cpu(layer_input + attention_output, torch.float16),
                            "post_attention_normalized_state_all_positions": cpu(captures[("norm2", layer_index)], torch.float16),
                            "mlp_output_all_positions": cpu(mlp_output, torch.float16),
                            "layer_output_all_positions": cpu(expected_output, torch.float16),
                        },
                        "attention": {
                            **attention,
                            "output_projection_weight_reference_id": weight_reference_id(model, layer_index, "o_proj.weight"),
                        },
                        "mlp": {
                            **mlp,
                            "down_projection_weight_reference_id": weight_reference_id(model, layer_index, "mlp.down_proj.weight"),
                            "single_neuron_write_materialization": "deferred_exact_product_times_weight_column",
                        },
                        "quality": {"errors": errors, "gates": gates, "all_gates_pass": all(gates.values())},
                        "claim_boundary": {
                            "engineering_replay_only": True,
                            "language_mechanism_claimed": False,
                            "semantic_labels_available": False,
                            "target_rank_or_margin_available": False,
                        },
                    }
                    if storage_mode == "sufficient":
                        payload = use_sufficient_state_storage(payload)
                    path = output_root / "private/models" / model / case["blind_case_id"] / f"time_{generation_time}" / f"layer_{layer_index:03d}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, path)
                    byte_count = path.stat().st_size
                    files.append({
                        "generation_time": generation_time,
                        "layer_index": layer_index,
                        "relative_path": str(path.relative_to(output_root)),
                        "byte_count": byte_count,
                        "sha256": sha256_file(path),
                        "all_gates_pass": all(gates.values()),
                    })
                    row_errors.append({
                        "generation_time": generation_time,
                        "layer_index": layer_index,
                        "sequence_length": len(sequence),
                        "errors": errors,
                        "all_gates_pass": all(gates.values()),
                    })
                    del payload, attention, mlp, layer_input, attention_output, mlp_output, expected_output
                sequence.append(next_token)
                del output, input_ids
                captures.clear()
                gc.collect()

        maxima = {
            key: max(row["errors"][key] for row in row_errors)
            for key in row_errors[0]["errors"]
        }
        manifest = {
            "schema_version": "47.2.0",
            "phase_id": "Phase371B",
            "created_at": now(),
            "model": model,
            "blind_case_id": case["blind_case_id"],
            "anchor_layers": list(anchor_layers),
            "generation_time_count": GENERATION_TIME_COUNT,
            "row_count": len(row_errors),
            "file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "storage_mode": storage_mode,
            "max_errors": maxima,
            "all_numeric_gates_pass": all(row["all_gates_pass"] for row in row_errors),
            "files": files,
            "rows": row_errors,
        }
        write_json(output_root / "models" / model / "manifest.json", manifest)
        print(json.dumps({key: value for key, value in manifest.items() if key not in {"files", "rows"}}, ensure_ascii=False, indent=2))
        return manifest
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--storage-mode", choices=("materialized", "sufficient"), default="materialized")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    run_model(args.model, storage_mode=args.storage_mode, output_root=args.output_root)
