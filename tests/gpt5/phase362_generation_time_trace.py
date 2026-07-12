#!/usr/bin/env python3
"""Record three natural generation times on 288 unseen calibration prompts."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase338_block_causal_screen import prompt_ids  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    MAX_ATTENTION_PROBABILITY_SUM_ERROR, MAX_COMPONENT_RELATIVE_ERROR,
    MLP_SHARD_COUNT, install_hooks, relative_error,
)
from phase361_r0_r1_component_trace import fragment_end_position, norm_replay, module_attr  # noqa: E402
from phase362_independent_case_bank import MODELS, OUT, ROUND  # noqa: E402


SCHEMA_VERSION = "39.1.0"
ROLE_NAMES = ("source", "query", "answer_start", "current_generation")
STATE_NAMES = (
    "layer_input", "input_normalized_state", "attention_projection_input",
    "attention_output", "post_attention_residual_state", "post_attention_normalized_state",
    "mlp_down_projection_input", "mlp_output", "layer_output",
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


def role_indices(loaded: Any, case: dict[str, Any], base_ids: list[int], sequence_length: int) -> list[int]:
    source, source_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], base_ids, case["source_fragment"], last=False,
    )
    query, query_exact = fragment_end_position(
        loaded.tokenizer, case["prompt"], base_ids, case["query_fragment"], last=True,
    )
    if not source_exact or not query_exact:
        raise RuntimeError(f"Exact Phase362 role mapping failed: {case['blind_case_id']}")
    return [source, query, len(base_ids) - 1, sequence_length - 1]


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    root = OUT / ROUND
    cases = [
        row for row in read_jsonl(root / "private" / "phase362_execution_cases.jsonl")
        if row["model"] == model and row["phase362_split"] == "independent_calibration"
    ]
    if len(cases) != 96:
        raise RuntimeError(f"Invalid Phase362 calibration denominator for {model}: {len(cases)}")
    ordered = sorted(cases, key=lambda row: hashlib.sha256(("phase362-r1:" + row["blind_case_id"]).encode()).hexdigest())
    shard_by_case = {row["blind_case_id"]: index % MLP_SHARD_COUNT for index, row in enumerate(ordered)}
    loaded = None
    handles: list[Any] = []
    ledger_rows, case_manifests = [], []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for case_index, case in enumerate(cases, 1):
            base_ids = prompt_ids(loaded, case)
            sequence = list(base_ids)
            raw_times, generated_token_ids = [], []
            all_gates = []
            for generation_time in range(3):
                captures.clear()
                input_ids = torch.tensor([sequence], dtype=torch.long, device=loaded.input_device)
                output = loaded.model(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, output_attentions=True, return_dict=True,
                )
                positions = role_indices(loaded, case, base_ids, len(sequence))
                position_tensor = torch.tensor(positions, device=loaded.input_device, dtype=torch.long)
                next_token_id = int(output.logits[0, -1].argmax().item())
                time_layers = []
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
                    input_norm = module_attr(layer, ("input_layernorm", "input_layer_norm", "ln_1"))
                    post_norm = module_attr(layer, ("post_attention_layernorm", "post_attention_layer_norm", "ln_2"))
                    _, block_error = relative_error(layer_output, post_attention + mlp_output)
                    _, norm1_error = relative_error(norm1, norm_replay(input_norm, layer_input))
                    _, norm2_error = relative_error(norm2, norm_replay(post_norm, post_attention))
                    selected_probs = probabilities.index_select(2, position_tensor).float().clamp_min(0)
                    probability_error = float((selected_probs.sum(dim=-1) - 1).abs().max().item())
                    entropy = -(
                        selected_probs.clamp_min(1e-12) * selected_probs.clamp_min(1e-12).log()
                    ).sum(dim=-1).mean(dim=1)[0]
                    channel_count = int(down_input.shape[-1])
                    offset = int(hashlib.sha256(f"{model}:{layer_index}".encode()).hexdigest()[:8], 16) % MLP_SHARD_COUNT
                    all_channels = torch.arange(channel_count, device=down_input.device)
                    shard_index = shard_by_case[case["blind_case_id"]]
                    shard_channels = all_channels[(all_channels + offset) % MLP_SHARD_COUNT == shard_index]
                    shard_activation = selected["mlp_down_projection_input"].index_select(-1, shard_channels)
                    gates = {
                        "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "input_norm": norm1_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "post_norm": norm2_error <= MAX_COMPONENT_RELATIVE_ERROR,
                        "probability": probability_error <= MAX_ATTENTION_PROBABILITY_SUM_ERROR,
                    }
                    all_gates.append(all(gates.values()))
                    ledger_rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": "Phase362",
                        "blind_case_id": case["blind_case_id"],
                        "anonymous_model_id": case["anonymous_model_id"],
                        "anonymous_group_id": case["phase362_group_id"],
                        "generation_time": generation_time, "layer_index": layer_index,
                        "relative_depth": round(layer_index / max(1, len(layers) - 1), 7),
                        "role_names": list(ROLE_NAMES),
                        "component_norms": {
                            name: [
                                round(float(torch.linalg.vector_norm(selected[name][0, role].float()).item()), 7)
                                for role in range(len(ROLE_NAMES))
                            ] for name in STATE_NAMES
                        },
                        "mean_attention_entropy": [round(float(value), 7) for value in entropy],
                        "r1_mlp_shard_index": shard_index,
                        "errors": {
                            "block": round(block_error, 9), "input_norm": round(norm1_error, 9),
                            "post_norm": round(norm2_error, 9), "probability": round(probability_error, 9),
                        },
                        "gates": gates, "semantic_label_used": False,
                    })
                    time_layers.append({
                        "layer_index": layer_index,
                        "role_positions": positions,
                        "role_states": {name: cpu(value) for name, value in selected.items()},
                        "r1_mlp_shard_index": shard_index,
                        "r1_mlp_channel_ids": cpu(shard_channels),
                        "r1_mlp_activations": cpu(shard_activation),
                    })
                raw_times.append({
                    "generation_time": generation_time, "sequence_length": len(sequence),
                    "full_vocabulary_logits": cpu(output.logits[0, -1],),
                    "layers": time_layers,
                })
                generated_token_ids.append(next_token_id)
                sequence.append(next_token_id)
                del output, input_ids, time_layers
                captures.clear()
            raw_path = root / "sealed_calibration" / model / f"{case['blind_case_id']}.pt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema_version": SCHEMA_VERSION, "phase_id": "Phase362",
                "blind_case_id": case["blind_case_id"],
                "anonymous_model_id": case["anonymous_model_id"],
                "role_names": ROLE_NAMES, "generated_token_ids": generated_token_ids,
                "times": raw_times,
            }, raw_path)
            case_manifests.append({
                "blind_case_id": case["blind_case_id"],
                "byte_count": raw_path.stat().st_size,
                "all_gates_pass": all(all_gates),
            })
            del raw_times
            gc.collect()
            if case_index % 8 == 0 or case_index == len(cases):
                print(f"[{model}] {case_index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        model_rows = [row for row in ledger_rows if row["anonymous_model_id"] == cases[0]["anonymous_model_id"]]
        write_jsonl(model_root / "phase362_generation_time_rows.jsonl", model_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": "Phase362", "created_at": now(),
            "model": model, "case_count": len(cases), "generation_time_count": 3,
            "layer_count": len(layers), "ledger_row_count": len(model_rows),
            "sealed_byte_count": sum(row["byte_count"] for row in case_manifests),
            "all_gates_pass": all(row["all_gates_pass"] for row in case_manifests),
            "valid": len(cases) == 96 and len(model_rows) == len(cases) * 3 * len(layers),
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
