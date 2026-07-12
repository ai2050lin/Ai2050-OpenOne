#!/usr/bin/env python3
"""Run two deterministic format forwards for one model and validate MLP write replay."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase365_dynamic_flow_instrumentation import (  # noqa: E402
    decompose_mlp_input, direct_mlp_output, relative_error,
    replay_mlp_from_neuron_writes,
)


OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/repeat_noise_format_gate"
MODELS = ("qwen3", "glm4", "deepseek7b")
PROMPT = "Trace calibration input: cedar amber cobalt seven."
REPEAT_COUNT = 2
MAX_MLP_REPLAY_RELATIVE_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def selected_layer_indices(layer_count: int) -> dict[str, int]:
    return {"early": 0, "middle": layer_count // 2, "late": layer_count - 1}


@torch.inference_mode()
def run_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise ValueError(model_key)
    loaded = None
    handles = []
    try:
        loaded = load_probe_model(model_key)
        layers = get_layers(loaded.model)
        selected = selected_layer_indices(len(layers))
        captures: dict[tuple[str, int], torch.Tensor] = {}
        for layer_index in selected.values():
            layer = layers[layer_index]

            def mlp_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
                captures[("mlp_input", idx)] = inputs[0].detach()

            def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
                captures[("mlp_output", idx)] = output.detach()

            handles.extend([
                layer.mlp.register_forward_pre_hook(mlp_pre),
                layer.mlp.register_forward_hook(mlp_post),
            ])

        encoded = loaded.tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
        first_snapshots: dict[str, torch.Tensor] = {}
        repeat_rows = []
        for repeat_index in range(REPEAT_COUNT):
            captures.clear()
            output = loaded.model(
                **encoded, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True,
            )
            final_logits = output.logits[:, -1, :].detach()
            layer_rows = []
            for role, layer_index in selected.items():
                mlp_input = captures[("mlp_input", layer_index)][:, -1:, :]
                actual = captures[("mlp_output", layer_index)][:, -1:, :]
                parts = decompose_mlp_input(model_key, layers[layer_index].mlp, mlp_input)
                direct = direct_mlp_output(parts)
                replayed = replay_mlp_from_neuron_writes(parts, chunk_size=128)
                direct_error = relative_error(actual, direct)
                neuron_error = relative_error(actual, replayed)
                tensors = {
                    f"{role}.mlp_input": mlp_input,
                    f"{role}.mlp_product": parts.product,
                    f"{role}.mlp_output": actual,
                }
                repeat_differences = {}
                for name, tensor in tensors.items():
                    if repeat_index == 0:
                        first_snapshots[name] = tensor.detach().cpu()
                    else:
                        first = first_snapshots[name].to(tensor.device)
                        repeat_differences[name] = {
                            "max_absolute": float((tensor.float() - first.float()).abs().max().item()),
                            "relative": relative_error(first, tensor),
                            "exact_equal": bool(torch.equal(first, tensor)),
                        }
                layer_rows.append({
                    "layer_role": role,
                    "layer_index": layer_index,
                    "adapter_kind": parts.adapter_kind,
                    "intermediate_size": int(parts.product.shape[-1]),
                    "direct_relative_error": direct_error,
                    "neuron_write_relative_error": neuron_error,
                    "replay_gate_pass": bool(
                        direct_error <= MAX_MLP_REPLAY_RELATIVE_ERROR
                        and neuron_error <= MAX_MLP_REPLAY_RELATIVE_ERROR
                    ),
                    "tensor_hashes": {name: tensor_sha256(tensor) for name, tensor in tensors.items()},
                    "repeat_differences": repeat_differences,
                })
            if repeat_index == 0:
                first_snapshots["final_logits"] = final_logits.detach().cpu()
                logits_difference = None
            else:
                first_logits = first_snapshots["final_logits"].to(final_logits.device)
                logits_difference = {
                    "max_absolute": float((final_logits.float() - first_logits.float()).abs().max().item()),
                    "relative": relative_error(first_logits, final_logits),
                    "exact_equal": bool(torch.equal(first_logits, final_logits)),
                }
            repeat_rows.append({
                "repeat_index": repeat_index,
                "final_logits_sha256": tensor_sha256(final_logits),
                "final_logits_repeat_difference": logits_difference,
                "layer_rows": layer_rows,
            })
            del output, final_logits
            gc.collect()

        second = repeat_rows[1]
        repeat_exact = bool(
            second["final_logits_repeat_difference"]["exact_equal"]
            and all(
                all(value["exact_equal"] for value in row["repeat_differences"].values())
                for row in second["layer_rows"]
            )
        )
        allocated, reserved = vram_gb()
        summary = {
            "schema_version": "42.2.0", "phase_id": "Phase365-A", "created_at": now(),
            "model": model_key,
            "denominator": {
                "fixed_prompt_count": 1, "repeat_count": REPEAT_COUNT,
                "selected_layer_count": len(selected), "token_count": int(encoded["input_ids"].shape[-1]),
            },
            "execution": {
                "use_cache": False, "output_attentions": False, "causal_intervention": False,
                "input_batch_size": 1, "model_execution_order_contract": list(MODELS),
            },
            "results": {
                "all_layer_replay_gates_pass": all(
                    row["replay_gate_pass"] for repeat in repeat_rows for row in repeat["layer_rows"]
                ),
                "repeat_exact_equal": repeat_exact,
                "max_direct_relative_error": max(
                    row["direct_relative_error"] for repeat in repeat_rows for row in repeat["layer_rows"]
                ),
                "max_neuron_write_relative_error": max(
                    row["neuron_write_relative_error"] for repeat in repeat_rows for row in repeat["layer_rows"]
                ),
                "max_repeat_relative_error": max(
                    [second["final_logits_repeat_difference"]["relative"]]
                    + [
                        value["relative"]
                        for row in second["layer_rows"] for value in row["repeat_differences"].values()
                    ]
                ),
            },
            "repeat_rows": repeat_rows,
            "vram_gb_before_release": {"allocated": allocated, "reserved": reserved},
            "valid": bool(
                repeat_exact
                and all(row["replay_gate_pass"] for repeat in repeat_rows for row in repeat["layer_rows"])
            ),
        }
        write_json(OUT / "models" / model_key / "complete.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
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
