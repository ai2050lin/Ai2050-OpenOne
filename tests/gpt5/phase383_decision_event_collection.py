#!/usr/bin/env python3
"""Collect exact replayable component events at the target semantic decision."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import replace
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
from phase365_dynamic_flow_instrumentation import (  # noqa: E402
    decompose_mlp_input,
    replay_mlp_from_neuron_writes,
)
from phase379_decision_aligned_trace import decision_input, token_rank  # noqa: E402


PHASE_ROOT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
DEFAULT_CASE_FILE = PHASE_ROOT / "protocol/private/phase383_execution_cases.jsonl"
DEFAULT_OUT = PHASE_ROOT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "discovery", "calibration", "physical_holdout")
ROLE_NAMES = ("source", "query", "answer_start", "current_generation")
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
    value = f"phase383-weight-ref:{model}:{layer_index}:{component}"
    return hashlib.sha256(value.encode()).hexdigest()


def split_authorized(split: str) -> None:
    protocol = read_json(PHASE_ROOT / "phase383_protocol.json")
    if split == "instrument_audit":
        allowed = protocol["authorization"]["instrument_audit_collection"]
    elif split == "discovery":
        gate = PHASE_ROOT / "phase383_instrument_audit_summary.json"
        allowed = gate.is_file() and read_json(gate)["authorization"][
            "discovery_collection"
        ]
    elif split == "calibration":
        gate = PHASE_ROOT / "phase383_discovery_map_freeze.json"
        allowed = gate.is_file() and read_json(gate)["authorization"][
            "calibration_collection"
        ]
    else:
        gate = PHASE_ROOT / "phase383_calibration_summary.json"
        allowed = gate.is_file() and read_json(gate)["authorization"].get(
            "physical_holdout_collection", False
        )
    if not allowed:
        raise RuntimeError(f"Phase383 split is not authorized: {split}")


def role_positions(
    loaded: Any, case: dict[str, Any], sequence: list[int]
) -> tuple[int, int, int, int]:
    _values, static = decision_input(loaded, case)
    base = loaded.tokenizer(
        case["prompt"],
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        truncation=True,
        max_length=256,
    )["input_ids"]
    positions = (int(static[0]), int(static[1]), len(base) - 1, len(sequence) - 1)
    if min(positions) < 0 or max(positions) >= len(sequence):
        raise RuntimeError(
            f"Invalid target-decision roles for {case['phase383_case_id']}: {positions}"
        )
    return positions


@torch.inference_mode()
def run_model(
    model: str,
    split: str,
    case_file: Path = DEFAULT_CASE_FILE,
    output_root: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    split_authorized(split)
    all_cases = read_jsonl(case_file)
    cases = [
        row
        for row in all_cases
        if row["private_execution_model"] == model
        and row["phase383_split"] == split
    ]
    if not cases:
        raise RuntimeError(f"No Phase383 cases for {model}/{split}")
    expected = sum(
        row["private_execution_model"] == model and row["phase383_split"] == split
        for row in all_cases
    )
    if len(cases) != expected:
        raise RuntimeError(f"Phase383 case selection mismatch for {model}/{split}")

    loaded = None
    handles: list[Any] = []
    value_handles: list[Any] = []
    mlp_internal_handles: list[Any] = []
    files: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    gate_maxima = {
        "attention_source": 0.0,
        "mlp_direct": 0.0,
        "mlp_neuron": 0.0,
        "mlp_product": 0.0,
        "block": 0.0,
        "probability": 0.0,
    }
    try:
        loaded = load_probe_model(model)
        runtime_dtype = next(loaded.model.parameters()).dtype
        if runtime_dtype != FROZEN_DTYPE_BY_MODEL[model]:
            raise RuntimeError(
                f"Phase383 runtime dtype mismatch for {model}: got {runtime_dtype}, "
                f"expected {FROZEN_DTYPE_BY_MODEL[model]}. Set PROBE_TORCH_DTYPE=float16."
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

                mlp_internal_handles.extend(
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

                mlp_internal_handles.append(
                    layer.mlp.gate_up_proj.register_forward_hook(gate_up_post)
                )

        for case_index, case in enumerate(cases, 1):
            sequence, _static = decision_input(loaded, case)
            positions = role_positions(loaded, case, sequence)
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
            target_step = int(case["target_decision_step"])
            target_token = int(case["generated_token_ids"][target_step])
            next_token = int(torch.argmax(logits).item())
            rank = token_rank(logits, target_token)
            baseline_match = next_token == target_token and rank == 1
            case_root = (
                output_root
                / split
                / "private/models"
                / model
                / case["phase383_case_id"]
                / "target_decision"
            )
            meta = {
                "schema_version": "57.1.0",
                "phase_id": "Phase383-DecisionEventCollection",
                "phase383_case_id": case["phase383_case_id"],
                "source_phase380_case_id": case["source_phase380_case_id"],
                "anonymous_model_id": case["anonymous_model_id"],
                "public_parallel_group_id": case[
                    "phase383_public_parallel_group_id"
                ],
                "anonymous_condition_slot": case["anonymous_condition_slot"],
                "semantic_time": "target_decision",
                "runtime_dtype": str(runtime_dtype).replace("torch.", ""),
                "target_decision_step": target_step,
                "sequence_length": len(sequence),
                "role_names": ROLE_NAMES,
                "role_positions": positions,
                "target_token_id_private": target_token,
                "next_token_id_private": next_token,
                "target_rank_private": rank,
                "baseline_replay_match": baseline_match,
                "full_vocabulary_logits": cpu(logits, torch.float32),
            }
            meta_path = case_root / "decision_meta.pt"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meta, meta_path)
            files.append(
                {
                    "phase383_case_id": case["phase383_case_id"],
                    "layer_index": None,
                    "kind": "decision_meta",
                    "relative_path": str(meta_path.relative_to(output_root)),
                    "byte_count": meta_path.stat().st_size,
                    "sha256": sha256_file(meta_path),
                }
            )
            case_file_count = 1
            case_bytes = meta_path.stat().st_size
            case_all_gates = baseline_match

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

                selected_input = layer_input.index_select(1, position_tensor)
                selected_norm1 = norm1.index_select(1, position_tensor)
                selected_attention = attention_output.index_select(1, position_tensor)
                selected_post = post_attention.index_select(1, position_tensor)
                selected_norm2 = norm2.index_select(1, position_tensor)
                selected_down = down_input.index_select(1, position_tensor)
                selected_mlp = mlp_output.index_select(1, position_tensor)
                selected_output = layer_output.index_select(1, position_tensor)
                selected_probs = probabilities.index_select(2, position_tensor)
                if model in {"qwen3", "deepseek7b"}:
                    selected_gate_pre = captures[("gate_pre", layer_index)].index_select(
                        1, position_tensor
                    )
                    selected_up = captures[("up", layer_index)].index_select(
                        1, position_tensor
                    )
                    captured_product = layer.mlp.act_fn(selected_gate_pre) * selected_up
                else:
                    selected_gate_up = captures[("gate_up", layer_index)].index_select(
                        1, position_tensor
                    )
                    selected_gate_pre, selected_up = selected_gate_up.chunk(2, dim=-1)
                    captured_product = selected_up * layer.mlp.activation_fn(
                        selected_gate_pre
                    )

                head_count = int(
                    getattr(layer.self_attn, "num_heads", 0)
                    or loaded.model.config.num_attention_heads
                )
                kv_head_count = int(
                    getattr(loaded.model.config, "num_key_value_heads", head_count)
                )
                head_dim = int(
                    getattr(loaded.model.config, "head_dim", 0)
                    or value_projection.shape[-1] // kv_head_count
                )
                values = value_projection.view(
                    1, len(sequence), kv_head_count, head_dim
                ).transpose(1, 2)
                repeated_values = values
                if kv_head_count != head_count:
                    repeated_values = values.repeat_interleave(
                        head_count // kv_head_count, dim=1
                    )
                source_sum = torch.zeros_like(selected_attention, dtype=torch.float32)
                for head_index in range(head_count):
                    start, end = head_index * head_dim, (head_index + 1) * head_dim
                    projected_values = F.linear(
                        repeated_values[:, head_index].float(),
                        o_proj.weight[:, start:end].float(),
                    )
                    source_sum += torch.einsum(
                        "bqs,bsh->bqh",
                        selected_probs[:, head_index].float(),
                        projected_values,
                    )
                if o_proj.bias is not None:
                    source_sum += o_proj.bias.float()
                _, attention_error = relative_error(selected_attention, source_sum)
                probability_error = float(
                    (selected_probs.float().sum(dim=-1) - 1).abs().max().item()
                )

                parts = decompose_mlp_input(model, layer.mlp, selected_norm2)
                _, product_error = relative_error(selected_down, captured_product)
                exact_parts = replace(parts, product=selected_down)
                direct = F.linear(
                    selected_down,
                    exact_parts.down_proj.weight,
                    exact_parts.down_proj.bias,
                )
                _, direct_error = relative_error(selected_mlp, direct)
                neuron_replay = replay_mlp_from_neuron_writes(
                    exact_parts, chunk_size=128
                )
                _, neuron_error = relative_error(selected_mlp, neuron_replay)
                reconstructed_block = selected_input + selected_attention + selected_mlp
                _, block_error = relative_error(selected_output, reconstructed_block)
                gates = {
                    "attention_source": attention_error
                    <= MAX_COMPONENT_RELATIVE_ERROR,
                    "probability": probability_error
                    <= MAX_COMPONENT_RELATIVE_ERROR,
                    "mlp_product": product_error <= MAX_PRODUCT_RELATIVE_ERROR,
                    "mlp_direct": direct_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "mlp_neuron": neuron_error <= MAX_COMPONENT_RELATIVE_ERROR,
                    "block": block_error <= MAX_COMPONENT_RELATIVE_ERROR,
                }
                case_all_gates = case_all_gates and all(gates.values())
                for key, value in {
                    "attention_source": attention_error,
                    "probability": probability_error,
                    "mlp_product": product_error,
                    "mlp_direct": direct_error,
                    "mlp_neuron": neuron_error,
                    "block": block_error,
                }.items():
                    gate_maxima[key] = max(gate_maxima[key], value)

                payload = {
                    "schema_version": "57.1.0",
                    "phase_id": "Phase383-DecisionEventCollection",
                    "phase383_case_id": case["phase383_case_id"],
                    "anonymous_model_id": case["anonymous_model_id"],
                    "public_parallel_group_id": case[
                        "phase383_public_parallel_group_id"
                    ],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "semantic_time": "target_decision",
                    "layer_index": layer_index,
                    "role_names": ROLE_NAMES,
                    "role_positions": positions,
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
                        "probabilities_role_receivers_all_sources": cpu(
                            selected_probs, torch.float16
                        ),
                        "output_projection_weight_reference_id": weight_reference_id(
                            model, layer_index, "o_proj.weight"
                        ),
                        "lazy_exact_event_family": (
                            "receiver x source_position x head; reconstruct from "
                            "probability, value state, and output projection slice"
                        ),
                        "head_count": head_count,
                        "key_value_head_count": kv_head_count,
                        "head_dim": head_dim,
                    },
                    "mlp": {
                        "adapter_kind": parts.adapter_kind,
                        "gate_pre_at_roles": cpu(selected_gate_pre, torch.float16),
                        "up_at_roles": cpu(selected_up, torch.float16),
                        "down_projection_input_product_at_roles": cpu(
                            selected_down, torch.float16
                        ),
                        "down_projection_weight_reference_id": weight_reference_id(
                            model, layer_index, "mlp.down_proj.weight"
                        ),
                        "lazy_exact_event_family": (
                            "receiver x channel; reconstruct from product scalar and "
                            "down projection column"
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
                        "all_gates_pass": all(gates.values()),
                    },
                }
                path = case_root / f"layer_{layer_index:03d}.pt"
                torch.save(payload, path)
                byte_count = path.stat().st_size
                files.append(
                    {
                        "phase383_case_id": case["phase383_case_id"],
                        "layer_index": layer_index,
                        "kind": "layer",
                        "relative_path": str(path.relative_to(output_root)),
                        "byte_count": byte_count,
                        "sha256": sha256_file(path),
                    }
                )
                case_file_count += 1
                case_bytes += byte_count
                del payload, source_sum, neuron_replay, direct, parts, exact_parts

            case_rows.append(
                {
                    "phase383_case_id": case["phase383_case_id"],
                    "public_parallel_group_id": case[
                        "phase383_public_parallel_group_id"
                    ],
                    "anonymous_model_id": case["anonymous_model_id"],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "mechanism_id_private": case["mechanism_id"],
                    "contrast_condition_private": case["contrast_condition"],
                    "file_count": case_file_count,
                    "byte_count": case_bytes,
                    "baseline_replay_match": baseline_match,
                    "all_gates_pass": case_all_gates,
                }
            )
            print(
                f"[{model}/{split}] {case_index}/{len(cases)} "
                f"files={case_file_count} bytes={case_bytes}",
                flush=True,
            )
            del output, input_ids, logits
            captures.clear()
            gc.collect()

        model_root = output_root / split / "models" / model
        manifest = {
            "schema_version": "57.1.0",
            "phase_id": "Phase383-DecisionEventCollection",
            "created_at": now(),
            "model": model,
            "split": split,
            "semantic_time": "target_decision",
            "runtime_dtype": str(runtime_dtype).replace("torch.", ""),
            "case_count": len(case_rows),
            "layer_count": len(layers),
            "file_count": len(files),
            "total_byte_count": sum(row["byte_count"] for row in files),
            "baseline_replay_match_count": sum(
                row["baseline_replay_match"] for row in case_rows
            ),
            "all_case_gates_pass": all(
                row["all_gates_pass"] for row in case_rows
            ),
            "gate_maxima": gate_maxima,
            "case_rows": case_rows,
            "files": files,
            "top_k_used": False,
            "exact_attention_head_source_events_replayable": True,
            "exact_mlp_channel_events_replayable": True,
            "valid": (
                len(case_rows) == expected
                and all(row["all_gates_pass"] for row in case_rows)
            ),
        }
        write_json(model_root / "manifest.json", manifest)
        compact = {
            key: value
            for key, value in manifest.items()
            if key not in {"files", "case_rows"}
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return manifest
    finally:
        for handle in [*handles, *value_handles, *mlp_internal_handles]:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_model(args.model, args.split, args.case_file, args.output_root)
