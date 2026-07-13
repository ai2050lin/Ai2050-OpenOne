#!/usr/bin/env python3
"""Validate Phase401 full-layer ledgers and same-shape token replay."""

from __future__ import annotations

import argparse
import gc
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
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
    relative_error,
)
from phase371b_anchor_qk_collection import (  # noqa: E402
    build_attention_tree,
    capture_actual_qkv,
)
from phase401_local_edge_protocol import MODELS, OUT  # noqa: E402


CASES = OUT / "trace/protocol/private/phase401_instrument_trace_cases.jsonl"


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase401 non-finite ledger scalar: {value}")
    return round(value, 10)


@torch.inference_mode()
def exact_replay(
    loaded: Any,
    prompt_ids: list[int],
    expected: list[int],
) -> tuple[int, int | None]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.input_device)
    output = loaded.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    past = output.past_key_values
    total_length = len(prompt_ids)
    matched = 0
    first_failure: int | None = None
    for index, expected_token in enumerate(expected):
        actual = int(output.logits[0, -1].argmax().item())
        if actual == expected_token and first_failure is None:
            matched += 1
        elif first_failure is None:
            first_failure = index
        if index + 1 >= len(expected):
            break
        total_length += 1
        token = torch.tensor(
            [[expected_token]], dtype=torch.long, device=loaded.input_device
        )
        output = loaded.model(
            input_ids=token,
            attention_mask=torch.ones(
                (1, total_length), dtype=torch.long, device=loaded.input_device
            ),
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past = output.past_key_values
    del output, past, ids
    return matched, first_failure


@torch.inference_mode()
def run(model: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase401_local_edge_protocol.json")
    gates = protocol["instrument_ledger_gates"]
    cases = [
        row for row in read_jsonl(CASES) if row["private_execution_model"] == model
    ]
    expected_count = 16 * len(
        read_json(OUT / "phase401_behavior_freeze_summary.json")["eligible_surfaces"]
    )
    if len(cases) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} Phase401 instrument cases for {model}, got {len(cases)}"
        )
    loaded = None
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        layer_ids = tuple(range(len(layers)))
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(model, layer_ids, captures):
            for case_index, case in enumerate(cases, 1):
                captures.clear()
                prompt = torch.tensor(
                    [case["prompt_token_ids_private"]],
                    dtype=torch.long,
                    device=loaded.input_device,
                )
                output = loaded.model(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                layer_errors: list[dict[str, float]] = []
                for layer_index, layer in enumerate(layers):
                    _state, attention_errors = build_attention_tree(
                        layer, captures, layer_index, materialize_derivatives=False
                    )
                    down_input = captures[("down_proj_input", layer_index)].float()
                    mlp_actual = captures[("mlp_output", layer_index)].float()
                    down_proj = module_attr(
                        layer.mlp, ("down_proj", "dense_4h_to_h")
                    )
                    mlp_replay = F.linear(
                        down_input,
                        down_proj.weight.float(),
                        down_proj.bias.float() if down_proj.bias is not None else None,
                    )
                    _, mlp_error = relative_error(mlp_actual, mlp_replay)
                    layer_input = captures[("layer_input", layer_index)].float()
                    attention = captures[("attention_output", layer_index)].float()
                    layer_output = captures[("layer_output", layer_index)].float()
                    block_replay = layer_input + attention + mlp_actual
                    _, block_error = relative_error(layer_output, block_replay)
                    continuity = 0.0
                    if layer_index + 1 < len(layers):
                        _, continuity = relative_error(
                            layer_output,
                            captures[("layer_input", layer_index + 1)].float(),
                        )
                    layer_errors.append(
                        {
                            "attention_probability": clean(
                                attention_errors["query_key_probability"]
                            ),
                            "attention_output": clean(
                                attention_errors["attention_direct"]
                            ),
                            "mlp_output": clean(mlp_error),
                            "block_output": clean(block_error),
                            "next_layer_continuity": clean(continuity),
                        }
                    )
                del output, prompt
                captures.clear()
                matched, first_failure = exact_replay(
                    loaded,
                    case["prompt_token_ids_private"],
                    case["exact_generated_replay_token_ids_private"],
                )
                maxima = {
                    key: max(item[key] for item in layer_errors)
                    for key in layer_errors[0]
                }
                ledger_pass = (
                    maxima["attention_probability"]
                    <= gates["attention_probability_replay_relative_error_max"]
                    and maxima["attention_output"]
                    <= gates["attention_output_replay_relative_error_max"]
                    and maxima["mlp_output"]
                    <= gates["mlp_output_replay_relative_error_max"]
                    and maxima["block_output"]
                    <= gates["block_output_replay_relative_error_max"]
                    and maxima["next_layer_continuity"]
                    <= gates["layer_output_to_next_input_relative_error_max"]
                )
                replay_pass = matched == len(
                    case["exact_generated_replay_token_ids_private"]
                )
                rows.append(
                    {
                        "schema_version": "75.6.0",
                        "phase_id": "Phase401-InstrumentLedger",
                        "created_at": now(),
                        "model": model,
                        "blind_case_id": case["blind_case_id"],
                        "public_parallel_group_id": case[
                            "phase401_public_parallel_group_id"
                        ],
                        "surface_private": case["task_surface_private"],
                        "condition_private": case["anonymous_condition_slot"],
                        "layer_count": len(layers),
                        "max_relative_errors": maxima,
                        "ledger_gate_pass": ledger_pass,
                        "expected_generated_token_count": len(
                            case["exact_generated_replay_token_ids_private"]
                        ),
                        "exact_replay_prefix_match_count": matched,
                        "exact_replay_first_failure_step": first_failure,
                        "exact_generated_replay_pass": replay_pass,
                        "quality_gate_pass": ledger_pass and replay_pass,
                        "causal_language_edge_claimed": False,
                    }
                )
                gc.collect()
                print(
                    f"[{model}] Phase401 instrument {case_index}/{len(cases)} "
                    f"ledger={ledger_pass} replay={replay_pass}",
                    flush=True,
                )
        maxima = {
            key: max(row["max_relative_errors"][key] for row in rows)
            for key in rows[0]["max_relative_errors"]
        }
        payload = {
            "schema_version": "75.6.0",
            "phase_id": "Phase401-InstrumentLedger",
            "created_at": now(),
            "model": model,
            "case_count": len(rows),
            "layer_count": len(layers),
            "max_relative_errors": maxima,
            "ledger_pass_case_count": sum(row["ledger_gate_pass"] for row in rows),
            "exact_replay_pass_case_count": sum(
                row["exact_generated_replay_pass"] for row in rows
            ),
            "quality_pass_case_count": sum(row["quality_gate_pass"] for row in rows),
            "valid": bool(rows) and all(row["quality_gate_pass"] for row in rows),
            "claim_boundary": {
                "ledger_conservation_is_language_mechanism": False,
                "exact_replay_is_batch_invariance": False,
            },
        }
        write_jsonl(OUT / "instrument/private" / model / "rows.jsonl", rows)
        write_json(OUT / "instrument" / model / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
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
    run(args.model)
