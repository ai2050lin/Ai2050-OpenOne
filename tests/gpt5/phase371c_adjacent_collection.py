#!/usr/bin/env python3
"""Collect the three missing adjacent layers for Phase371C local continuity replay."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase338_block_causal_screen import prompt_ids  # noqa: E402
from phase358_multiresolution_component_conservation import relative_error  # noqa: E402
from phase371b_anchor_qk_collection import (  # noqa: E402
    build_attention_tree, build_mlp_tree, capture_actual_qkv, cpu,
    install_anchor_hooks, sha256_file, use_sufficient_state_storage,
    weight_reference_id, write_json,
)
from phase371c_internal_collection import numeric_gates, selected_cases  # noqa: E402


BASE = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_internal_discovery"
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_adjacent_extension"
MODELS = ("qwen3", "glm4", "deepseek7b")
MIN_FREE_RESERVE_BYTES = 200 * 1024**3


def neighbor_contract(layer_count: int) -> tuple[tuple[int, int, int], dict[int, dict[str, Any]]]:
    middle = layer_count // 2
    last = layer_count - 1
    layers = (1, middle + 1, last - 1)
    metadata = {
        1: {"pair": "early", "role": "successor", "paired_existing_layer": 0},
        middle + 1: {"pair": "middle", "role": "successor", "paired_existing_layer": middle},
        last - 1: {"pair": "late", "role": "predecessor", "paired_existing_layer": last},
    }
    if len(set(layers)) != 3:
        raise RuntimeError(f"Adjacent layer contract collapsed for layer_count={layer_count}")
    return layers, metadata


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    cases = selected_cases(model)
    base_manifest = json.loads((BASE / "models" / model / "manifest.json").read_text(encoding="utf-8"))
    loaded = None
    handles: list[Any] = []
    captures: dict[tuple[str, int], torch.Tensor] = {}
    files = []
    case_rows = []
    max_errors: dict[str, float] = {}
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        selected_layers, layer_metadata = neighbor_contract(len(layers))
        handles = install_anchor_hooks(layers, selected_layers, captures)
        with capture_actual_qkv(model, selected_layers, captures):
            for case_index, case in enumerate(cases, 1):
                if shutil.disk_usage(ROOT).free < MIN_FREE_RESERVE_BYTES:
                    raise RuntimeError("Phase371C adjacent extension reached disk reserve")
                sequence = list(prompt_ids(loaded, case))
                case_files = []
                case_gate = True
                generation_match = True
                for generation_time in range(3):
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
                    base_time = torch.load(
                        BASE / "private/models" / model / case["blind_case_id"] / f"time_{generation_time}/time_meta.pt",
                        map_location="cpu", weights_only=True,
                    )
                    generation_match = generation_match and next_token == int(base_time["next_token_id_private"])
                    for layer_index in selected_layers:
                        layer = layers[layer_index]
                        attention, attention_errors = build_attention_tree(
                            layer, captures, layer_index, materialize_derivatives=False,
                        )
                        mlp, mlp_errors = build_mlp_tree(
                            layer, captures, layer_index, materialize_derivatives=False,
                        )
                        layer_input = captures[("layer_input", layer_index)].float()
                        attention_output = captures[("attention_output", layer_index)].float()
                        mlp_output = captures[("mlp_output", layer_index)].float()
                        expected_output = captures[("layer_output", layer_index)].float()
                        _, block_error = relative_error(
                            expected_output, layer_input + attention_output + mlp_output,
                        )
                        errors = {**attention_errors, **mlp_errors, "block": block_error}
                        gates = numeric_gates(errors)
                        case_gate = case_gate and all(gates.values())
                        for key, value in errors.items():
                            max_errors[key] = max(max_errors.get(key, 0.0), value)
                        payload = {
                            "schema_version": "47.13.0",
                            "phase_id": "Phase371C-Adj",
                            "blind_case_id": case["blind_case_id"],
                            "anonymous_model_id": case["anonymous_model_id"],
                            "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
                            "anonymous_group_id": case["anonymous_group_id"],
                            "anonymous_condition_slot": case["anonymous_condition_slot"],
                            "generation_time": generation_time,
                            "layer_index": layer_index,
                            "sequence_length": len(sequence),
                            "adjacent_contract": layer_metadata[layer_index],
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
                                "adjacent_measurement_only": True,
                                "language_mechanism_claimed": False,
                                "semantic_labels_available": False,
                            },
                        }
                        payload = use_sufficient_state_storage(payload)
                        path = OUT / "private/models" / model / case["blind_case_id"] / f"time_{generation_time}" / f"layer_{layer_index:03d}.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(payload, path)
                        file_row = {
                            "blind_case_id": case["blind_case_id"],
                            "generation_time": generation_time,
                            "layer_index": layer_index,
                            "kind": "adjacent_layer",
                            "relative_path": str(path.relative_to(OUT)),
                            "byte_count": path.stat().st_size,
                            "sha256": sha256_file(path),
                            "all_gates_pass": all(gates.values()),
                        }
                        files.append(file_row)
                        case_files.append(file_row)
                        del payload, attention, mlp, layer_input, attention_output, mlp_output, expected_output
                    sequence.append(next_token)
                    del output, input_ids, base_time
                    captures.clear()
                case_rows.append({
                    "blind_case_id": case["blind_case_id"],
                    "file_count": len(case_files),
                    "byte_count": sum(row["byte_count"] for row in case_files),
                    "all_numeric_gates_pass": case_gate,
                    "generation_tokens_match_base": generation_match,
                })
                print(f"[{model}] Phase371C adjacent {case_index}/{len(cases)} bytes={case_rows[-1]['byte_count']}", flush=True)
                gc.collect()
        manifest = {
            "schema_version": "47.13.0",
            "phase_id": "Phase371C-Adj",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "case_count": len(case_rows),
            "selected_layers": list(selected_layers),
            "generation_time_count": 3,
            "file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "max_errors": max_errors,
            "all_numeric_gates_pass": all(row["all_numeric_gates_pass"] for row in case_rows),
            "all_generation_tokens_match_base": all(row["generation_tokens_match_base"] for row in case_rows),
            "base_manifest_created_at": base_manifest["created_at"],
            "files": files,
            "case_rows": case_rows,
        }
        write_json(OUT / "models" / model / "manifest.json", manifest)
        print(json.dumps({key: value for key, value in manifest.items() if key not in {"files", "case_rows"}}, ensure_ascii=False, indent=2))
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
    run_model(args.model)
