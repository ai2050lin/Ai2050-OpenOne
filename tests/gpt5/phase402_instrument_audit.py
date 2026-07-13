#!/usr/bin/env python3
"""Audit Phase402 empty-subset replay on every instrument case and layer."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
)
from phase371b_anchor_qk_collection import capture_actual_qkv  # noqa: E402
from phase399_dynamic_trace_collection import repeat_kv  # noqa: E402
from phase401_local_edge_collection import capture_case, to_device_layer  # noqa: E402
from phase402_multiparent_protocol import MODELS, OUT, PARENT_CATEGORIES  # noqa: E402


SOURCE = OUT / "trace/protocol/private/phase402_instrument_trace_cases.jsonl"


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
        raise RuntimeError(f"Phase402 non-finite instrument value: {value}")
    return round(value, 9)


@torch.inference_mode()
def replay_attention(layer: Any, state: dict[str, Any], role: str) -> torch.Tensor:
    receiver = state["receivers"][role]
    query = receiver["query"]
    head_count = int(query.shape[0])
    key = repeat_kv(state["key"], head_count)
    value = repeat_kv(state["value"], head_count)
    # Match the model's eager attention path: QK multiplication happens in the
    # native runtime dtype before the float32 softmax.  Promoting Q/K first can
    # change near-tied GLM4 rows enough to fail an otherwise empty intervention.
    scores = torch.einsum("hd,hsd->hs", query, key)
    scores = scores * float(state["scaling"])
    if receiver["mask"] is not None:
        scores = scores + receiver["mask"].to(scores.dtype).unsqueeze(0)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weighted = torch.einsum("hs,hsd->hd", probabilities, value)
    o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
    return o_proj(weighted.reshape(1, 1, -1))[0, 0]


@torch.inference_mode()
def run(model: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase402_multiparent_protocol.json")
    gate = protocol["instrument_gate"]
    cases = [
        row for row in read_jsonl(SOURCE) if row["private_execution_model"] == model
    ]
    freeze = read_json(OUT / "phase402_behavior_freeze_summary.json")
    expected = len(freeze["eligible_surfaces"]) * 16
    if len(cases) != expected:
        raise RuntimeError(f"Phase402 instrument {model}: {len(cases)} != {expected}")

    loaded = None
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(model, tuple(range(len(layers))), captures):
            for case_index, case in enumerate(cases, 1):
                collected = capture_case(loaded, layers, captures, case)
                partition_valid = True
                receiver = int(case["state_role_positions_private"]["query_end"][0])
                partition = case["parent_partitions_private"]["query_end"]
                flattened = [
                    position
                    for category in PARENT_CATEGORIES
                    for position in partition[category]
                ]
                partition_valid = (
                    sorted(flattened) == list(range(receiver + 1))
                    and len(flattened) == len(set(flattened))
                )
                for layer_index, layer in enumerate(layers):
                    state = to_device_layer(
                        collected["layers"][layer_index], loaded.input_device
                    )
                    replay = replay_attention(layer, state, "query_end").float()
                    natural = state["receivers"]["query_end"]["attention"].float()
                    error = float(
                        torch.linalg.vector_norm(replay - natural).item()
                        / max(float(torch.linalg.vector_norm(natural).item()), 1e-8)
                    )
                    rows.append(
                        {
                            "schema_version": "76.5.0",
                            "phase_id": "Phase402-InstrumentRow",
                            "model": model,
                            "surface_private": case["task_surface_private"],
                            "public_parallel_group_id": case[
                                "phase402_public_parallel_group_id"
                            ],
                            "condition_private": case[
                                "anonymous_condition_slot"
                            ],
                            "layer_index": layer_index,
                            "layer_count": len(layers),
                            "first_token_replay_match": collected[
                                "first_prediction_matches_frozen"
                            ],
                            "partition_conservation_pass": partition_valid,
                            "empty_subset_attention_relative_error": clean(error),
                            "pass": bool(
                                collected["first_prediction_matches_frozen"]
                                and partition_valid
                                and error
                                <= gate["empty_subset_attention_relative_error_max"]
                            ),
                        }
                    )
                    del state, replay, natural
                del collected
                if case_index % 8 == 0 or case_index == len(cases):
                    print(
                        f"[{model}/phase402-instrument] {case_index}/{len(cases)}",
                        flush=True,
                    )
        payload = {
            "schema_version": "76.5.0",
            "phase_id": "Phase402-InstrumentAuditModel",
            "created_at": now(),
            "model": model,
            "case_count": len(cases),
            "layer_count": len(layers),
            "row_count": len(rows),
            "first_token_replay_match_count": sum(
                row["first_token_replay_match"] for row in rows
            ),
            "partition_conservation_pass_count": sum(
                row["partition_conservation_pass"] for row in rows
            ),
            "max_empty_subset_attention_relative_error": max(
                row["empty_subset_attention_relative_error"] for row in rows
            ),
            "passing_row_count": sum(row["pass"] for row in rows),
            "valid": bool(rows) and all(row["pass"] for row in rows),
        }
        root = OUT / "instrument/private" / model
        write_jsonl(root / "rows.jsonl", rows)
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


def summarize() -> dict[str, Any]:
    model_rows = {
        model: read_json(OUT / "instrument" / model / "complete.json")
        for model in MODELS
    }
    valid = all(row["valid"] for row in model_rows.values())
    payload = {
        "schema_version": "76.5.0",
        "phase_id": "Phase402-InstrumentAudit",
        "created_at": now(),
        "models": model_rows,
        "valid": valid,
        "authorization": {
            "run_discovery": valid,
            "run_calibration": False,
            "run_physical_holdout": False,
        },
        "claim_boundary": {
            "empty_subset_replay_is_a_language_mechanism": False,
            "instrument_pass_is_a_parent_set": False,
        },
    }
    write_json(OUT / "phase402_instrument_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run(args.model)
    else:
        parser.error("use --model or --summarize")
