#!/usr/bin/env python3
"""Persist one sealed, replayable full-vector component anchor per model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
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
    MAX_ATTENTION_PROBABILITY_SUM_ERROR,
    MAX_COMPONENT_RELATIVE_ERROR,
    MLP_SHARD_COUNT,
    install_hooks,
    module_attr,
    relative_error,
    selected_cases,
)


OUT = ROOT / "tests/gpt5/result/phase359_full_vector_anchor"
MODELS = ("qwen3", "glm4", "deepseek7b")
SCHEMA_VERSION = "35.0.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu()


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    budget = json.loads((OUT / "phase359_storage_budget.json").read_text(encoding="utf-8"))
    if budget["decision"] != "allow_one_full_vector_anchor_per_model":
        raise RuntimeError("Storage budget did not authorize anchor capture")
    case = selected_cases(model, "format_development")[0]
    case_digest = hashlib.sha256(f"phase359:{case['case_id']}".encode()).hexdigest()
    anchor_id = f"sealed_{case_digest[:24]}"
    model_root = OUT / "sealed_tensors" / model / anchor_id
    if model_root.exists():
        shutil.rmtree(model_root)
    model_root.mkdir(parents=True)

    loaded = None
    handles: list[Any] = []
    files: list[dict[str, Any]] = []
    layer_metrics: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        ids = prompt_ids(loaded, case)
        input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        del output, input_ids

        for layer_index, layer in enumerate(layers):
            layer_input = captures[("layer_input", layer_index)]
            norm1 = captures[("norm1", layer_index)]
            o_proj_input = captures[("o_proj_input", layer_index)]
            attention_output = captures[("attention_output", layer_index)]
            probabilities = captures[("attention_probabilities", layer_index)]
            norm2 = captures[("norm2", layer_index)]
            down_input = captures[("down_proj_input", layer_index)]
            mlp_output = captures[("mlp_output", layer_index)]
            layer_output = captures[("layer_output", layer_index)]
            o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
            down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))

            head_count = int(getattr(layer.self_attn, "num_heads", 0) or loaded.model.config.num_attention_heads)
            head_width = o_proj_input.shape[-1] // head_count
            head_contributions = []
            attention_sum = torch.zeros_like(attention_output, dtype=torch.float32)
            for head_index in range(head_count):
                start, end = head_index * head_width, (head_index + 1) * head_width
                contribution = F.linear(
                    o_proj_input[..., start:end].float(),
                    o_proj.weight[:, start:end].float(),
                )
                attention_sum += contribution
                head_contributions.append(cpu(contribution))
            attention_bias = o_proj.bias.float() if o_proj.bias is not None else None
            if attention_bias is not None:
                attention_sum += attention_bias

            channel_count = int(down_input.shape[-1])
            offset = int(hashlib.sha256(f"{model}:{layer_index}".encode()).hexdigest()[:8], 16) % MLP_SHARD_COUNT
            channel_ids = torch.arange(channel_count, device=down_input.device)
            shard_contributions = []
            shard_channel_ids = []
            mlp_sum = torch.zeros_like(mlp_output, dtype=torch.float32)
            for shard_index in range(MLP_SHARD_COUNT):
                selected = channel_ids[(channel_ids + offset) % MLP_SHARD_COUNT == shard_index]
                contribution = F.linear(
                    down_input.index_select(-1, selected).float(),
                    down_proj.weight.index_select(1, selected).float(),
                )
                mlp_sum += contribution
                shard_contributions.append(cpu(contribution))
                shard_channel_ids.append(cpu(selected))
            mlp_bias = down_proj.bias.float() if down_proj.bias is not None else None
            if mlp_bias is not None:
                mlp_sum += mlp_bias

            _, attention_relative_error = relative_error(attention_output, attention_sum)
            _, mlp_relative_error = relative_error(mlp_output, mlp_sum)
            _, block_relative_error = relative_error(
                layer_output, (layer_input + attention_output) + mlp_output
            )
            probability_error = float(
                (probabilities.float().sum(dim=-1) - 1).abs().max().item()
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase359",
                "model": model,
                "anchor_id": anchor_id,
                "layer_index": layer_index,
                "token_position_count": len(ids),
                "selection_rule": "first_pre_registered_format_anchor",
                "semantic_label_used": False,
                "causal_intervention": False,
                "layer_input": cpu(layer_input),
                "input_normalized_state": cpu(norm1),
                "attention_projection_input": cpu(o_proj_input),
                "attention_output": cpu(attention_output),
                "attention_probabilities": cpu(probabilities),
                "projected_head_contributions": torch.stack(head_contributions),
                "attention_projection_bias": cpu(attention_bias) if attention_bias is not None else None,
                "post_attention_normalized_state": cpu(norm2),
                "mlp_down_projection_input": cpu(down_input),
                "projected_mlp_shard_contributions": torch.stack(shard_contributions),
                "mlp_shard_channel_ids": shard_channel_ids,
                "mlp_down_projection_bias": cpu(mlp_bias) if mlp_bias is not None else None,
                "mlp_output": cpu(mlp_output),
                "layer_output": cpu(layer_output),
            }
            path = model_root / f"layer_{layer_index:03d}.pt"
            torch.save(payload, path)
            files.append({
                "layer_index": layer_index,
                "relative_path": str(path.relative_to(OUT)),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            layer_metrics.append({
                "layer_index": layer_index,
                "attention_relative_reconstruction_error": attention_relative_error,
                "mlp_relative_reconstruction_error": mlp_relative_error,
                "block_relative_reconstruction_error": block_relative_error,
                "attention_probability_sum_error": probability_error,
            })
            for name in (
                "layer_input", "norm1", "o_proj_input", "attention_output",
                "attention_probabilities", "norm2", "down_proj_input", "mlp_output", "layer_output",
            ):
                captures.pop((name, layer_index), None)
            del payload, head_contributions, shard_contributions, shard_channel_ids
            gc.collect()
            print(f"[{model}] layer {layer_index + 1}/{len(layers)} persisted", flush=True)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase359",
            "created_at": now(),
            "model": model,
            "anchor_id": anchor_id,
            "prompt_token_count": len(ids),
            "layer_count": len(layers),
            "layer_file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "sealed": True,
            "frontend_exported": False,
            "semantic_label_used": False,
            "causal_intervention": False,
            "files": files,
            "online_gates": {
                "attention_reconstruction_pass": all(
                    row["attention_relative_reconstruction_error"] <= MAX_COMPONENT_RELATIVE_ERROR
                    for row in layer_metrics
                ),
                "mlp_reconstruction_pass": all(
                    row["mlp_relative_reconstruction_error"] <= MAX_COMPONENT_RELATIVE_ERROR
                    for row in layer_metrics
                ),
                "block_reconstruction_pass": all(
                    row["block_relative_reconstruction_error"] <= MAX_COMPONENT_RELATIVE_ERROR
                    for row in layer_metrics
                ),
                "attention_probability_pass": all(
                    row["attention_probability_sum_error"] <= MAX_ATTENTION_PROBABILITY_SUM_ERROR
                    for row in layer_metrics
                ),
            },
            "online_layer_metrics": layer_metrics,
        }
        (model_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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
    args = parser.parse_args()
    result = run_model(args.model)
    print(json.dumps({
        "model": result["model"],
        "anchor_id": result["anchor_id"],
        "layer_count": result["layer_count"],
        "total_byte_count": result["total_byte_count"],
        "online_gates": result["online_gates"],
    }, ensure_ascii=False, indent=2))
