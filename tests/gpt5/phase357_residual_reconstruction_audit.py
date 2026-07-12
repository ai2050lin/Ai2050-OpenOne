#!/usr/bin/env python3
"""Audit whether generic hooks reconstruct each transformer block update."""

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
from phase338_block_causal_screen import component_tensor, prompt_ids  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
OUT = ROOT / "tests/gpt5/result/phase357_residual_reconstruction_audit"
ROUND_NAME = "pre_registered_anchor_reconstruction"
PHASE = "Phase357"
SCHEMA_VERSION = "33.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
ANCHORS_PER_MODEL = 12
MAX_RELATIVE_INCREMENT_ERROR = 0.02
MAX_RELATIVE_OUTPUT_ERROR = 0.005


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
    return hashlib.sha256(f"phase357-anchor-v1:{case_id}".encode()).hexdigest()


def anchors(model: str) -> list[dict[str, Any]]:
    cases = [
        row for row in read_jsonl(SOURCE / "phase354_registered_cases.jsonl")
        if row["model"] == model and row["split"] == "physical_discovery"
    ]
    return sorted(cases, key=lambda row: anchor_rank(row["case_id"]))[:ANCHORS_PER_MODEL]


def install_hooks(layers: list[Any], captures: dict[tuple[str, int], torch.Tensor]) -> list[Any]:
    handles = []
    for layer_index, layer in enumerate(layers):
        def layer_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("input", idx)] = inputs[0].detach()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("attention", idx)] = component_tensor(output).detach()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("mlp", idx)] = component_tensor(output).detach()

        def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            captures[("output", idx)] = component_tensor(output).detach()

        handles.extend([
            layer.register_forward_pre_hook(layer_pre),
            layer.self_attn.register_forward_hook(attention_post),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(layer_post),
        ])
    return handles


@torch.inference_mode()
def run_model(model: str) -> dict[str, Any]:
    selected = anchors(model)
    loaded = None
    handles: list[Any] = []
    rows = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], torch.Tensor] = {}
        handles = install_hooks(layers, captures)
        for case_index, case in enumerate(selected, 1):
            captures.clear()
            ids = prompt_ids(loaded, case)
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                use_cache=False, return_dict=True,
            )
            for layer_index in range(len(layers)):
                layer_input_native = captures[("input", layer_index)]
                attention_native = captures[("attention", layer_index)]
                mlp_native = captures[("mlp", layer_index)]
                layer_output_native = captures[("output", layer_index)]
                shape_match = (
                    layer_input_native.shape == attention_native.shape
                    == mlp_native.shape == layer_output_native.shape
                )
                if shape_match:
                    reconstructed_native = layer_input_native + attention_native
                    reconstructed_native = reconstructed_native + mlp_native
                    layer_input = layer_input_native.float()
                    layer_output = layer_output_native.float()
                    observed_increment = layer_output - layer_input
                    error = layer_output - reconstructed_native.float()
                    error_norm = float(torch.linalg.vector_norm(error).item())
                    increment_norm = float(torch.linalg.vector_norm(observed_increment).item())
                    output_norm = float(torch.linalg.vector_norm(layer_output).item())
                    relative_increment_error = error_norm / max(increment_norm, 1e-8)
                    relative_output_error = error_norm / max(output_norm, 1e-8)
                    finite = all(math.isfinite(value) for value in (
                        error_norm, increment_norm, output_norm,
                        relative_increment_error, relative_output_error,
                    ))
                else:
                    error_norm = increment_norm = output_norm = None
                    relative_increment_error = relative_output_error = None
                    finite = False
                rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": model, "anchor_id": f"anchor_{anchor_rank(case['case_id'])[:20]}",
                    "layer_index": layer_index, "token_position_count": len(ids),
                    "shape_match": shape_match,
                    "execution_dtype": str(layer_output_native.dtype).replace("torch.", ""),
                    "native_precision_addition_replayed": True,
                    "absolute_reconstruction_error_norm": round(error_norm, 7) if finite else None,
                    "observed_increment_norm": round(increment_norm, 7) if finite else None,
                    "layer_output_norm": round(output_norm, 7) if finite else None,
                    "relative_increment_reconstruction_error": round(relative_increment_error, 9) if finite else None,
                    "relative_output_reconstruction_error": round(relative_output_error, 9) if finite else None,
                    "increment_gate_pass": bool(finite and relative_increment_error <= MAX_RELATIVE_INCREMENT_ERROR),
                    "output_gate_pass": bool(finite and relative_output_error <= MAX_RELATIVE_OUTPUT_ERROR),
                    "finite": finite, "target_direction_used": False,
                    "semantic_label_used_for_selection": False,
                })
            del output, input_ids
            print(f"[{model}] {case_index}/{len(selected)}", flush=True)
        model_root = OUT / ROUND_NAME / "models" / model
        write_jsonl(model_root / "phase357_reconstruction_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "anchor_case_count": len(selected), "layer_count": len(layers),
            "reconstruction_row_count": len(rows),
            "shape_mismatch_count": sum(not row["shape_match"] for row in rows),
            "nonfinite_row_count": sum(not row["finite"] for row in rows),
            "increment_gate_pass_count": sum(row["increment_gate_pass"] for row in rows),
            "output_gate_pass_count": sum(row["output_gate_pass"] for row in rows),
            "all_increment_gates_pass": all(row["increment_gate_pass"] for row in rows),
            "all_output_gates_pass": all(row["output_gate_pass"] for row in rows),
            "valid": len(selected) == ANCHORS_PER_MODEL and len(rows) == ANCHORS_PER_MODEL * len(layers) and all(row["finite"] for row in rows),
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
