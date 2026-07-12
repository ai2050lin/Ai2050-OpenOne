#!/usr/bin/env python3
"""Validate normalization, attention-head, and MLP-shard ledgers on format anchors."""

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
from phase334_natural_contrast_survey import component_tensor  # noqa: E402
from phase338_block_causal_screen import prompt_ids  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
OUT = ROOT / "tests/gpt5/result/phase358_multiresolution_full_trace"
ROUND_NAME = "format_development_component_conservation"
PHASE = "Phase358"
SCHEMA_VERSION = "34.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAT_CASES_PER_MODEL = 2
STAGE_CASE_COUNTS = {"format_development": 2, "blind_discovery": 3, "blind_calibration": 1}
MLP_SHARD_COUNT = 16
MAX_COMPONENT_RELATIVE_ERROR = 0.01
MAX_ATTENTION_PROBABILITY_SUM_ERROR = 0.01


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


def anchor_rank(case_id: str) -> str:
    return hashlib.sha256(f"phase358-format-v1:{case_id}".encode()).hexdigest()


def selected_cases(model: str, stage: str) -> list[dict[str, Any]]:
    discovery = [
        row for row in read_jsonl(SOURCE / "phase354_registered_cases.jsonl")
        if row["model"] == model and row["split"] == "physical_discovery"
    ]
    calibration = [
        row for row in read_jsonl(SOURCE / "phase354_registered_cases.jsonl")
        if row["model"] == model and row["split"] == "physical_calibration"
    ]
    discovery = sorted(discovery, key=lambda row: anchor_rank(row["case_id"]))
    calibration = sorted(calibration, key=lambda row: anchor_rank(row["case_id"]))
    if stage == "format_development":
        return discovery[:2]
    if stage == "blind_discovery":
        return discovery[2:5]
    if stage == "blind_calibration":
        return calibration[:1]
    raise ValueError(stage)


def module_attr(module: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot locate any of {names} on {type(module).__name__}")


def install_hooks(layers: list[Any], captures: dict[tuple[str, int], Any]) -> list[Any]:
    handles = []
    for layer_index, layer in enumerate(layers):
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

        def down_proj_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("down_proj_input", idx)] = inputs[0].detach()

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
            down_proj.register_forward_pre_hook(down_proj_pre),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(layer_post),
        ])
    return handles


def relative_error(actual: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float]:
    error = float(torch.linalg.vector_norm(actual.float() - reconstructed.float()).item())
    scale = float(torch.linalg.vector_norm(actual.float()).item())
    return error, error / max(scale, 1e-8)


@torch.inference_mode()
def run_model(model: str, stage: str = "format_development") -> dict[str, Any]:
    cases = selected_cases(model, stage)
    loaded = None
    handles: list[Any] = []
    layer_rows, head_rows, shard_rows = [], [], []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for case_index, case in enumerate(cases, 1):
            captures.clear()
            ids = prompt_ids(loaded, case)
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                use_cache=False, output_attentions=True, return_dict=True,
            )
            anchor_id = f"{stage}_{anchor_rank(case['case_id'])[:20]}"
            for layer_index, layer in enumerate(layers):
                layer_input = captures[("layer_input", layer_index)]
                norm1 = captures[("norm1", layer_index)]
                o_proj_input = captures[("o_proj_input", layer_index)]
                attention_output = captures[("attention_output", layer_index)]
                probabilities = captures.get(("attention_probabilities", layer_index))
                norm2 = captures[("norm2", layer_index)]
                down_input = captures[("down_proj_input", layer_index)]
                mlp_output = captures[("mlp_output", layer_index)]
                layer_output = captures[("layer_output", layer_index)]
                o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
                down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

                attention_sum = torch.zeros_like(attention_output, dtype=torch.float32)
                head_count = int(getattr(layer.self_attn, "num_heads", 0) or loaded.model.config.num_attention_heads)
                head_width = o_proj_input.shape[-1] // head_count
                for head_index in range(head_count):
                    start, end = head_index * head_width, (head_index + 1) * head_width
                    contribution = F.linear(
                        o_proj_input[..., start:end].float(),
                        o_proj.weight[:, start:end].float(),
                    )
                    attention_sum += contribution
                    if probabilities is not None:
                        probs = probabilities[:, head_index].float().clamp_min(0)
                        probability_sum_error = float((probs.sum(dim=-1) - 1).abs().max().item())
                        entropy = float((-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item())
                    else:
                        probability_sum_error = entropy = None
                    head_rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                        "model": model, "anchor_id": anchor_id, "layer_index": layer_index,
                        "head_index": head_index,
                        "projected_head_contribution_norm": round(float(torch.linalg.vector_norm(contribution).item()), 7),
                        "attention_entropy": round(entropy, 7) if entropy is not None and math.isfinite(entropy) else None,
                        "max_probability_sum_error": round(probability_sum_error, 9) if probability_sum_error is not None else None,
                        "probability_normalization_gate_pass": bool(
                            probability_sum_error is not None
                            and probability_sum_error <= MAX_ATTENTION_PROBABILITY_SUM_ERROR
                        ),
                        "selection_rule": "all_heads", "semantic_label_used": False,
                    })
                if o_proj.bias is not None:
                    attention_sum += o_proj.bias.float()
                attention_abs_error, attention_rel_error = relative_error(attention_output, attention_sum)

                mlp_sum = torch.zeros_like(mlp_output, dtype=torch.float32)
                channel_count = down_input.shape[-1]
                offset = int(hashlib.sha256(f"{model}:{layer_index}".encode()).hexdigest()[:8], 16) % MLP_SHARD_COUNT
                channel_ids = torch.arange(channel_count, device=down_input.device)
                for shard_index in range(MLP_SHARD_COUNT):
                    selected = channel_ids[(channel_ids + offset) % MLP_SHARD_COUNT == shard_index]
                    contribution = F.linear(
                        down_input.index_select(-1, selected).float(),
                        down_proj.weight.index_select(1, selected).float(),
                    )
                    mlp_sum += contribution
                    shard_rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                        "model": model, "anchor_id": anchor_id, "layer_index": layer_index,
                        "shard_index": shard_index, "channel_count": int(selected.numel()),
                        "projected_shard_contribution_norm": round(float(torch.linalg.vector_norm(contribution).item()), 7),
                        "selection_rule": "fixed_hash_partition_all_channels",
                        "semantic_label_used": False,
                    })
                if down_proj.bias is not None:
                    mlp_sum += down_proj.bias.float()
                mlp_abs_error, mlp_rel_error = relative_error(mlp_output, mlp_sum)

                reconstructed_native = layer_input + attention_output
                reconstructed_native = reconstructed_native + mlp_output
                block_abs_error, block_rel_error = relative_error(layer_output, reconstructed_native)
                norm_finite = bool(torch.isfinite(norm1).all() and torch.isfinite(norm2).all())
                layer_rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": model, "anchor_id": anchor_id, "layer_index": layer_index,
                    "token_position_count": len(ids), "execution_dtype": str(layer_output.dtype).replace("torch.", ""),
                    "input_normalization_finite": norm_finite,
                    "input_normalized_state_norm": round(float(torch.linalg.vector_norm(norm1.float()).item()), 7),
                    "post_attention_normalized_state_norm": round(float(torch.linalg.vector_norm(norm2.float()).item()), 7),
                    "attention_head_count": head_count, "mlp_channel_count": channel_count,
                    "mlp_shard_count": MLP_SHARD_COUNT,
                    "attention_absolute_reconstruction_error": round(attention_abs_error, 7),
                    "attention_relative_reconstruction_error": round(attention_rel_error, 9),
                    "attention_reconstruction_gate_pass": attention_rel_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "mlp_absolute_reconstruction_error": round(mlp_abs_error, 7),
                    "mlp_relative_reconstruction_error": round(mlp_rel_error, 9),
                    "mlp_reconstruction_gate_pass": mlp_rel_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "block_absolute_reconstruction_error": round(block_abs_error, 7),
                    "block_relative_reconstruction_error": round(block_rel_error, 9),
                    "block_reconstruction_gate_pass": block_rel_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "all_heads_recorded": True, "all_mlp_channels_partitioned": True,
                    "semantic_label_used": False, "causal_intervention": False,
                })
            del output, input_ids
            print(f"[{model}] {case_index}/{len(cases)}", flush=True)
        model_root = OUT / ROUND_NAME / "models" / model if stage == "format_development" else OUT / ROUND_NAME / "stages" / stage / "models" / model
        write_jsonl(model_root / "phase358_layer_rows.jsonl", layer_rows)
        write_jsonl(model_root / "phase358_attention_head_rows.jsonl", head_rows)
        write_jsonl(model_root / "phase358_mlp_shard_rows.jsonl", shard_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "stage": stage, "format_case_count": len(cases), "layer_count": len(layers),
            "layer_row_count": len(layer_rows), "attention_head_row_count": len(head_rows),
            "mlp_shard_row_count": len(shard_rows),
            "block_gate_pass": all(row["block_reconstruction_gate_pass"] for row in layer_rows),
            "attention_gate_pass": all(row["attention_reconstruction_gate_pass"] for row in layer_rows),
            "attention_probability_gate_pass": all(row["probability_normalization_gate_pass"] for row in head_rows),
            "mlp_gate_pass": all(row["mlp_reconstruction_gate_pass"] for row in layer_rows),
            "normalization_gate_pass": all(row["input_normalization_finite"] for row in layer_rows),
            "valid": len(cases) == STAGE_CASE_COUNTS[stage] and bool(layer_rows) and bool(head_rows) and bool(shard_rows),
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
    parser.add_argument("--stage", choices=tuple(STAGE_CASE_COUNTS), default="format_development")
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.stage), ensure_ascii=False, indent=2))
