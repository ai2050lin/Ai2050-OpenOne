#!/usr/bin/env python3
"""Record sealed R0 states and balanced R1 MLP shards for admitted mechanisms."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
from phase361_r0_r1_case_bank import MODELS, OUT, ROUND  # noqa: E402


SCHEMA_VERSION = "38.0.0"
ROLE_NAMES = ("source", "query", "answer_start", "current_generation")
STATE_NAMES = (
    "layer_input", "input_normalized_state", "attention_projection_input",
    "attention_output", "post_attention_residual_state",
    "post_attention_normalized_state", "mlp_down_projection_input",
    "mlp_output", "layer_output",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu()


def norm_replay(module: Any, value: torch.Tensor) -> torch.Tensor:
    if isinstance(module, torch.nn.LayerNorm):
        return F.layer_norm(
            value.float(), module.normalized_shape,
            module.weight.float() if module.weight is not None else None,
            module.bias.float() if module.bias is not None else None,
            module.eps,
        )
    epsilon = float(
        getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
    )
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    weight = getattr(module, "weight", None)
    if weight is not None:
        normalized = normalized * weight.float()
    bias = getattr(module, "bias", None)
    if bias is not None:
        normalized = normalized + bias.float()
    return normalized


def fragment_end_position(
    tokenizer: Any, prompt: str, prompt_token_ids: list[int], fragment: str, *, last: bool,
) -> tuple[int, bool]:
    start = prompt.rfind(fragment) if last else prompt.find(fragment)
    if start < 0:
        return -1, False
    boundary_end = min(len(prompt), start + len(fragment) + 1)
    prefix_ids = [
        int(value) for value in tokenizer(
            prompt[:boundary_end], add_special_tokens=False,
        )["input_ids"]
    ]
    for width in range(min(32, len(prefix_ids)), 3, -1):
        pattern = prefix_ids[-width:]
        matches = [
            index for index in range(len(prompt_token_ids) - width + 1)
            if prompt_token_ids[index : index + width] == pattern
        ]
        if matches:
            match = matches[-1] if last else matches[0]
            return match + width - 1, True
    return -1, False


def selected_positions(loaded: Any, case: dict[str, Any], ids: list[int]) -> tuple[list[int], dict[str, bool]]:
    source, source_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], ids, case["source_fragment"], last=False,
    )
    query, query_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], ids, case["query_fragment"], last=True,
    )
    if not source_exact or not query_exact:
        raise RuntimeError(
            f"Exact role mapping failed for {case['blind_case_id']}: "
            f"source={source_exact}, query={query_exact}"
        )
    answer = len(ids) - 1
    return [source, query, answer, answer], {
        "source": True, "query": True, "answer_start": True, "current_generation": True,
    }


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    root = OUT / ROUND
    cases = [
        row for row in read_jsonl(root / "private" / "phase361_execution_cases.jsonl")
        if row["model"] == model
    ]
    if len(cases) != 32:
        raise RuntimeError(f"Invalid R0/R1 model denominator for {model}: {len(cases)}")
    loaded = None
    handles: list[Any] = []
    ledger_rows: list[dict[str, Any]] = []
    case_manifests: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for case_index, case in enumerate(cases, 1):
            captures.clear()
            ids = prompt_ids(loaded, case)
            positions, role_exact = selected_positions(loaded, case, ids)
            position_tensor = torch.tensor(positions, device=loaded.input_device, dtype=torch.long)
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                use_cache=False, output_attentions=True, return_dict=True,
            )
            raw_layers = []
            all_case_gates = []
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
                post_attention = layer_input + attention_output
                input_norm_module = module_attr(layer, ("input_layernorm", "input_layer_norm", "ln_1"))
                post_norm_module = module_attr(layer, ("post_attention_layernorm", "post_attention_layer_norm", "ln_2"))
                o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
                down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

                selected = {
                    "layer_input": layer_input.index_select(1, position_tensor),
                    "input_normalized_state": norm1.index_select(1, position_tensor),
                    "attention_projection_input": o_input.index_select(1, position_tensor),
                    "attention_output": attention_output.index_select(1, position_tensor),
                    "post_attention_residual_state": post_attention.index_select(1, position_tensor),
                    "post_attention_normalized_state": norm2.index_select(1, position_tensor),
                    "mlp_down_projection_input": down_input.index_select(1, position_tensor),
                    "mlp_output": mlp_output.index_select(1, position_tensor),
                    "layer_output": layer_output.index_select(1, position_tensor),
                }
                head_count = int(getattr(layer.self_attn, "num_heads", 0) or loaded.model.config.num_attention_heads)
                head_width = o_input.shape[-1] // head_count
                head_norms, head_entropies = [], []
                attention_sum = torch.zeros_like(selected["attention_output"], dtype=torch.float32)
                selected_probabilities = probabilities.index_select(2, position_tensor).float().clamp_min(0)
                for head_index in range(head_count):
                    start, end = head_index * head_width, (head_index + 1) * head_width
                    contribution = F.linear(
                        selected["attention_projection_input"][..., start:end].float(),
                        o_proj.weight[:, start:end].float(),
                    )
                    attention_sum += contribution
                    head_norms.append([
                        round(float(torch.linalg.vector_norm(contribution[0, role]).item()), 7)
                        for role in range(len(ROLE_NAMES))
                    ])
                    probs = selected_probabilities[:, head_index]
                    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    head_entropies.append([round(float(value), 7) for value in entropy[0]])
                if o_proj.bias is not None:
                    attention_sum += o_proj.bias.float()
                _, attention_error = relative_error(selected["attention_output"], attention_sum)

                channel_count = int(down_input.shape[-1])
                channel_ids = torch.arange(channel_count, device=down_input.device)
                offset = int(hashlib.sha256(f"{model}:{layer_index}".encode()).hexdigest()[:8], 16) % MLP_SHARD_COUNT
                shard_norms, shard_channel_ids = [], []
                mlp_sum = torch.zeros_like(selected["mlp_output"], dtype=torch.float32)
                r1_activation = None
                r1_channels = None
                for shard_index in range(MLP_SHARD_COUNT):
                    channels = channel_ids[(channel_ids + offset) % MLP_SHARD_COUNT == shard_index]
                    contribution = F.linear(
                        selected["mlp_down_projection_input"].index_select(-1, channels).float(),
                        down_proj.weight.index_select(1, channels).float(),
                    )
                    mlp_sum += contribution
                    shard_norms.append([
                        round(float(torch.linalg.vector_norm(contribution[0, role]).item()), 7)
                        for role in range(len(ROLE_NAMES))
                    ])
                    shard_channel_ids.append(int(channels.numel()))
                    if shard_index == case["r1_mlp_shard_index"]:
                        r1_activation = selected["mlp_down_projection_input"].index_select(-1, channels)
                        r1_channels = channels
                if down_proj.bias is not None:
                    mlp_sum += down_proj.bias.float()
                _, mlp_error = relative_error(selected["mlp_output"], mlp_sum)
                _, block_error = relative_error(layer_output, post_attention + mlp_output)
                _, norm1_error = relative_error(norm1, norm_replay(input_norm_module, layer_input))
                _, norm2_error = relative_error(norm2, norm_replay(post_norm_module, post_attention))
                probability_error = float((selected_probabilities.sum(dim=-1) - 1).abs().max().item())
                gates = {
                    "attention": attention_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "mlp": mlp_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "input_norm": norm1_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "post_attention_norm": norm2_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "probability": probability_error <= MAX_ATTENTION_PROBABILITY_SUM_ERROR,
                }
                all_case_gates.append(all(gates.values()))
                ledger_rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": "Phase361",
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_model_id": case["anonymous_model_id"],
                    "split": "blind_discovery" if case["split"] == "physical_discovery" else "blind_calibration",
                    "layer_index": layer_index,
                    "relative_depth": round(layer_index / max(1, len(layers) - 1), 7),
                    "role_names": list(ROLE_NAMES), "role_position_exact": role_exact,
                    "component_norms": {
                        name: [round(float(torch.linalg.vector_norm(selected[name][0, role].float()).item()), 7) for role in range(len(ROLE_NAMES))]
                        for name in STATE_NAMES
                    },
                    "projected_head_norms": head_norms,
                    "attention_entropy": head_entropies,
                    "projected_mlp_shard_norms": shard_norms,
                    "mlp_shard_channel_counts": shard_channel_ids,
                    "r1_mlp_shard_index": case["r1_mlp_shard_index"],
                    "errors": {
                        "attention": round(attention_error, 9), "mlp": round(mlp_error, 9),
                        "block": round(block_error, 9), "input_norm": round(norm1_error, 9),
                        "post_attention_norm": round(norm2_error, 9),
                        "probability": round(probability_error, 9),
                    },
                    "gates": gates, "semantic_label_used": False,
                })
                raw_layers.append({
                    "layer_index": layer_index,
                    "role_positions": positions,
                    "role_states": {name: cpu(value) for name, value in selected.items()},
                    "attention_source_edges": cpu(selected_probabilities),
                    "r1_mlp_shard_index": case["r1_mlp_shard_index"],
                    "r1_mlp_channel_ids": cpu(r1_channels),
                    "r1_mlp_activations": cpu(r1_activation),
                })
            raw_path = root / "sealed" / model / f"{case['blind_case_id']}.pt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema_version": SCHEMA_VERSION, "phase_id": "Phase361",
                "blind_case_id": case["blind_case_id"],
                "anonymous_model_id": case["anonymous_model_id"],
                "prompt_token_count": len(ids), "role_names": ROLE_NAMES,
                "full_vocabulary_logits": cpu(output.logits[0, -1]),
                "layers": raw_layers,
            }, raw_path)
            case_manifests.append({
                "blind_case_id": case["blind_case_id"], "model": model,
                "layer_count": len(layers), "byte_count": raw_path.stat().st_size,
                "all_gates_pass": all(all_case_gates),
            })
            del output, input_ids, raw_layers
            captures.clear()
            gc.collect()
            print(f"[{model}] {case_index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        model_rows = [row for row in ledger_rows if row["anonymous_model_id"] == cases[0]["anonymous_model_id"]]
        write_jsonl(model_root / "phase361_r0_r1_ledger_rows.jsonl", model_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": "Phase361", "created_at": now(),
            "model": model, "case_count": len(cases), "layer_count": len(layers),
            "ledger_row_count": len(model_rows),
            "sealed_byte_count": sum(row["byte_count"] for row in case_manifests),
            "all_component_gates_pass": all(row["all_gates_pass"] for row in case_manifests),
            "valid": len(cases) == 32 and len(model_rows) == len(cases) * len(layers),
        }
        write_json(model_root / "complete.json", complete)
        return complete
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
    args = parser.parse_args()
    print(json.dumps(run_model(args.model), ensure_ascii=False, indent=2))
