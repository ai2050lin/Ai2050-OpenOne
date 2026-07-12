#!/usr/bin/env python3
"""Persist nine replayable anchors with attention source-edge conservation."""

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
from phase338_block_causal_screen import prompt_ids  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    MAX_ATTENTION_PROBABILITY_SUM_ERROR, MAX_COMPONENT_RELATIVE_ERROR,
    MLP_SHARD_COUNT, install_hooks, module_attr, relative_error,
)
from phase361_r0_r1_component_trace import fragment_end_position, norm_replay  # noqa: E402
from phase362_independent_case_bank import MODELS, OUT, ROUND  # noqa: E402


SCHEMA_VERSION = "39.0.0"
ROLE_NAMES = ("source", "query", "answer_start", "current_generation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cpu(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = tensor.detach().contiguous()
    if dtype is not None:
        value = value.to(dtype)
    return value.cpu()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_indices(loaded: Any, case: dict[str, Any], base_ids: list[int], sequence_length: int) -> list[int]:
    source, source_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], base_ids, case["source_fragment"], last=False,
    )
    query, query_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], base_ids, case["query_fragment"], last=True,
    )
    if not source_exact or not query_exact:
        raise RuntimeError(f"Exact anchor role mapping failed: {case['blind_case_id']}")
    return [source, query, len(base_ids) - 1, sequence_length - 1]


def co_shards(count: int) -> list[int]:
    if count == 2:
        return [0, 8]
    if count == 4:
        return [0, 4, 8, 12]
    if count == 16:
        return list(range(16))
    raise ValueError(count)


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    root = OUT / ROUND
    cases = {
        row["blind_case_id"]: row
        for row in read_jsonl(root / "private" / "phase362_execution_cases.jsonl")
        if row["model"] == model
    }
    anchors = [
        row for row in read_jsonl(root / "private" / "phase362_anchor_registry.jsonl")
        if row["model"] == model
    ]
    loaded = None
    handles: list[Any] = []
    value_handles: list[Any] = []
    files, time_summaries = [], []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for layer_index, layer in enumerate(layers):
            v_proj = module_attr(layer.self_attn, ("v_proj", "value"))

            def value_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
                captures[("value_projection", idx)] = output.detach()

            value_handles.append(v_proj.register_forward_hook(value_post))

        for anchor in anchors:
            case = cases[anchor["blind_case_id"]]
            base_ids = prompt_ids(loaded, case)
            sequence = list(base_ids)
            for generation_time in range(anchor["generation_time_count"]):
                captures.clear()
                input_ids = torch.tensor([sequence], dtype=torch.long, device=loaded.input_device)
                output = loaded.model(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, output_attentions=True, return_dict=True,
                )
                positions = role_indices(loaded, case, base_ids, len(sequence))
                position_tensor = torch.tensor(positions, device=loaded.input_device, dtype=torch.long)
                next_token_id = int(output.logits[0, -1].argmax().item())
                next_is_eos = next_token_id in set(loaded.tokenizer.all_special_ids)
                layer_gate_rows = []
                for layer_index, layer in enumerate(layers):
                    layer_input = captures[("layer_input", layer_index)]
                    norm1 = captures[("norm1", layer_index)]
                    o_input = captures[("o_proj_input", layer_index)]
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
                    input_norm = module_attr(layer, ("input_layernorm", "input_layer_norm", "ln_1"))
                    post_norm = module_attr(layer, ("post_attention_layernorm", "post_attention_layer_norm", "ln_2"))
                    head_count = int(getattr(layer.self_attn, "num_heads", 0) or loaded.model.config.num_attention_heads)
                    kv_head_count = int(getattr(loaded.model.config, "num_key_value_heads", head_count))
                    head_width = o_input.shape[-1] // head_count
                    values = value_projection.view(1, len(sequence), kv_head_count, head_width).transpose(1, 2)
                    if kv_head_count != head_count:
                        values = values.repeat_interleave(head_count // kv_head_count, dim=1)
                    selected_probs = probabilities.index_select(2, position_tensor).float()
                    selected_o_input = o_input.index_select(1, position_tensor)
                    projected_value_states, projected_head_outputs = [], []
                    edge_head_errors = []
                    for head_index in range(head_count):
                        start, end = head_index * head_width, (head_index + 1) * head_width
                        projected_values = F.linear(values[:, head_index].float(), o_proj.weight[:, start:end].float())
                        head_output = F.linear(
                            selected_o_input[..., start:end].float(), o_proj.weight[:, start:end].float(),
                        )
                        edge_sum = torch.einsum("bqs,bsh->bqh", selected_probs[:, head_index], projected_values)
                        _, edge_error = relative_error(head_output, edge_sum)
                        edge_head_errors.append(edge_error)
                        projected_value_states.append(cpu(projected_values, torch.float16))
                        projected_head_outputs.append(cpu(head_output, torch.float16))
                    projected_heads = torch.stack(projected_head_outputs)
                    attention_sum = projected_heads.float().sum(dim=0)
                    if o_proj.bias is not None:
                        attention_sum += cpu(o_proj.bias, torch.float32)
                    selected_attention = cpu(attention_output.index_select(1, position_tensor))
                    _, attention_error = relative_error(selected_attention, attention_sum)

                    selected_down = down_input.index_select(1, position_tensor)
                    channel_ids = torch.arange(down_input.shape[-1], device=down_input.device)
                    offset = int(hashlib.sha256(f"{model}:{layer_index}".encode()).hexdigest()[:8], 16) % MLP_SHARD_COUNT
                    shard_contributions, saved_activations, saved_channels = [], {}, {}
                    for shard_index in range(MLP_SHARD_COUNT):
                        channels = channel_ids[(channel_ids + offset) % MLP_SHARD_COUNT == shard_index]
                        contribution = F.linear(
                            selected_down.index_select(-1, channels).float(),
                            down_proj.weight.index_select(1, channels).float(),
                        )
                        shard_contributions.append(cpu(contribution, torch.float16))
                        if shard_index in co_shards(anchor["mlp_co_shard_count"]):
                            saved_activations[str(shard_index)] = cpu(selected_down.index_select(-1, channels))
                            saved_channels[str(shard_index)] = cpu(channels)
                    shard_tensor = torch.stack(shard_contributions)
                    mlp_sum = shard_tensor.float().sum(dim=0)
                    if down_proj.bias is not None:
                        mlp_sum += cpu(down_proj.bias, torch.float32)
                    selected_mlp = cpu(mlp_output.index_select(1, position_tensor))
                    _, mlp_error = relative_error(selected_mlp, mlp_sum)
                    _, block_error = relative_error(layer_output, post_attention + mlp_output)
                    _, norm1_error = relative_error(norm1, norm_replay(input_norm, layer_input))
                    _, norm2_error = relative_error(norm2, norm_replay(post_norm, post_attention))
                    probability_error = float((selected_probs.sum(dim=-1) - 1).abs().max().item())
                    gates = {
                        "edge": max(edge_head_errors) <= MAX_COMPONENT_RELATIVE_ERROR,
                        "attention": attention_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "mlp": mlp_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "input_norm": norm1_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "post_norm": norm2_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "probability": probability_error <= MAX_ATTENTION_PROBABILITY_SUM_ERROR,
                    }
                    layer_gate_rows.append(gates)
                    payload = {
                        "schema_version": SCHEMA_VERSION, "phase_id": "Phase362",
                        "anchor_id": anchor["anchor_id"], "anchor_type": anchor["anchor_type"],
                        "generation_time": generation_time, "layer_index": layer_index,
                        "role_names": ROLE_NAMES, "role_positions": positions,
                        "projected_value_states": torch.stack(projected_value_states),
                        "selected_attention_probabilities": cpu(selected_probs, torch.float16),
                        "projected_head_outputs": projected_heads,
                        "attention_output": selected_attention,
                        "attention_bias": cpu(o_proj.bias) if o_proj.bias is not None else None,
                        "mlp_shard_contributions": shard_tensor,
                        "mlp_output": selected_mlp,
                        "mlp_bias": cpu(down_proj.bias) if down_proj.bias is not None else None,
                        "saved_mlp_shard_activations": saved_activations,
                        "saved_mlp_shard_channel_ids": saved_channels,
                        "layer_input": cpu(layer_input.index_select(1, position_tensor)),
                        "input_norm_actual": cpu(norm1.index_select(1, position_tensor)),
                        "input_norm_replayed": cpu(norm_replay(input_norm, layer_input).index_select(1, position_tensor)),
                        "post_attention_state": cpu(post_attention.index_select(1, position_tensor)),
                        "post_norm_actual": cpu(norm2.index_select(1, position_tensor)),
                        "post_norm_replayed": cpu(norm_replay(post_norm, post_attention).index_select(1, position_tensor)),
                        "layer_output": cpu(layer_output.index_select(1, position_tensor)),
                        "gates": gates,
                    }
                    path = root / "sealed_anchors" / model / anchor["anchor_id"] / f"time_{generation_time}" / f"layer_{layer_index:03d}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, path)
                    files.append({
                        "anchor_id": anchor["anchor_id"], "generation_time": generation_time,
                        "layer_index": layer_index, "relative_path": str(path.relative_to(root)),
                        "byte_count": path.stat().st_size, "sha256": sha256_file(path),
                    })
                    del payload, projected_value_states, projected_head_outputs, shard_contributions
                time_summaries.append({
                    "anchor_id": anchor["anchor_id"], "anchor_type": anchor["anchor_type"],
                    "generation_time": generation_time, "sequence_length": len(sequence),
                    "next_token_is_special": next_is_eos,
                    "all_online_gates_pass": all(all(value.values()) for value in layer_gate_rows),
                })
                sequence.append(next_token_id)
                del output, input_ids
                captures.clear()
                gc.collect()
                print(f"[{model}] {anchor['anchor_type']} t={generation_time}", flush=True)
        manifest = {
            "schema_version": SCHEMA_VERSION, "phase_id": "Phase362", "created_at": now(),
            "model": model, "anchor_count": len(anchors),
            "anchor_time_count": len(time_summaries), "layer_file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "all_online_gates_pass": all(row["all_online_gates_pass"] for row in time_summaries),
            "time_summaries": time_summaries, "files": files,
        }
        model_root = root / "sealed_anchors" / model
        write_json(model_root / "manifest.json", manifest)
        return manifest
    finally:
        for handle in [*handles, *value_handles]:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    result = run_model(args.model)
    print(json.dumps({
        "model": result["model"], "anchor_count": result["anchor_count"],
        "anchor_time_count": result["anchor_time_count"],
        "layer_file_count": result["layer_file_count"],
        "total_byte_count": result["total_byte_count"],
        "all_online_gates_pass": result["all_online_gates_pass"],
    }, ensure_ascii=False, indent=2))
