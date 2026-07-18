#!/usr/bin/env python3
"""Collect fresh prompt-end rows on Phase546 frozen upstream event axes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase545_natural_entry_physical_collect as base  # noqa: E402


PHASE544 = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
OUT_DIR = ROOT / "tests/gpt5/result/phase546_upstream_physical_prediction"
CASES_PATH = PHASE544 / "phase544_registered_cases.jsonl"
PAIRS_PATH = OUT_DIR / "phase546_registered_confirmation_pairs.jsonl"
EVENTS_PATH = OUT_DIR / "phase546_frozen_upstream_events.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase546_upstream_protocol.json"
AUDIT_PATH = OUT_DIR / "phase546_static_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify() -> dict[str, Any]:
    protocol = base.read_json(PROTOCOL_PATH)
    audit = base.read_json(AUDIT_PATH)
    if audit["status"] != "static_pass_no_fresh_hidden_state_read" or not audit["valid"]:
        raise RuntimeError("Phase546 static protocol did not pass")
    if protocol["registered_pairs_sha256"] != base.sha256_file(PAIRS_PATH):
        raise RuntimeError("Phase546 pair registry drift")
    if protocol["frozen_events_sha256"] != base.sha256_file(EVENTS_PATH):
        raise RuntimeError("Phase546 event registry drift")
    if protocol["claim_boundaries"]["sealed_split_read"]:
        raise RuntimeError("Phase546 must not read a sealed split")
    return protocol


def collect_prompt_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    row: dict[str, Any],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], dict[str, Any]]:
    encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    source_fragment, query_fragment = base.parse_direct_fragments(row["raw_prompt"])
    source_positions = base.fragment_positions(tokenizer, prompt_ids, source_fragment)
    query_positions = base.fragment_positions(tokenizer, prompt_ids, query_fragment)
    vectors, ledger_error = base.stage_vectors(
        model, device, layers, captures, prompt_ids, source_positions, query_positions
    )
    del encoded
    return vectors, {
        "source_position_count": len(source_positions),
        "query_position_count": len(query_positions),
        "component_ledger_relative_error": ledger_error,
    }


def combine_pair(
    loaded_model: Any,
    tokenizer: Any,
    layers: list[Any],
    pair: dict[str, Any],
    event: dict[str, Any],
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    vectors_a: dict[int, dict[str, dict[str, torch.Tensor]]],
    vectors_b: dict[int, dict[str, dict[str, torch.Tensor]]],
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
) -> list[dict[str, Any]]:
    token_a, token_b, divergence_index = base.first_divergent_ids(
        tokenizer, row_a["strict_expected"], row_b["strict_expected"]
    )
    embedding = loaded_model.get_output_embeddings()
    direction = (
        embedding.weight[token_a].float().detach().cpu()
        - embedding.weight[token_b].float().detach().cpu()
    )
    component = event["component"]
    role = event["role"]
    feature_key = f"{component}__{role}"
    rows = []
    for layer_index in range(len(layers)):
        left = vectors_a[layer_index][component][role]
        right = vectors_b[layer_index][component][role]
        delta = left - right
        rows.append({
            "schema_version": "phase546_upstream_pair_layer.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "physical_pair_id": pair["physical_pair_id"],
            "frozen_event_id": event["event_id"],
            "model": pair["model"],
            "family_id": pair["family_id"],
            "mechanism_id": pair["mechanism_id"],
            "split": pair["split"],
            "source_behavior_split": pair["source_behavior_split"],
            "pair_index": pair["pair_index"],
            "stage": "prompt_end",
            "component": component,
            "role": role,
            "layer": layer_index,
            "layer_count": len(layers),
            "relative_depth": base.clean(layer_index / max(1, len(layers) - 1)),
            "frozen_discovery_layer": event["layer"],
            "frozen_discovery_relative_depth": event["relative_depth"],
            "target_divergence_token_index": divergence_index,
            "target_divergence_token_a": token_a,
            "target_divergence_token_b": token_b,
            "feature_key": feature_key,
            "features": {
                "normalized_world_delta": base.normalized_delta(left, right),
                "world_cosine": base.cosine(left, right),
                "pair_direction_alignment": base.cosine(delta, direction),
                "world_a_norm": base.clean(float(torch.linalg.vector_norm(left).item())),
                "world_b_norm": base.clean(float(torch.linalg.vector_norm(right).item())),
            },
            "max_component_ledger_relative_error": base.clean(max(
                meta_a["component_ledger_relative_error"],
                meta_b["component_ledger_relative_error"],
            )),
            "source_position_count_a": meta_a["source_position_count"],
            "source_position_count_b": meta_b["source_position_count"],
            "query_position_count_a": meta_a["query_position_count"],
            "query_position_count_b": meta_b["query_position_count"],
            "physical": True,
            "observer_only": True,
            "predictive": False,
            "compute_edge": False,
            "causal": False,
            "single_neuron": False,
            "sealed": False,
        })
    return rows


def run_model(model_name: str, restart: bool) -> Path:
    verify()
    pairs = [row for row in base.read_jsonl(PAIRS_PATH) if row["model"] == model_name]
    output_path = OUT_DIR / f"phase546_{model_name}_pair_layer_rows.jsonl"
    summary_path = OUT_DIR / f"phase546_{model_name}_collection_summary.json"
    if not pairs:
        payload = {
            "schema_version": "phase546_collection_summary.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "status": "skipped_by_behavior_gate",
            "model": model_name,
            "registered_pair_count": 0,
            "pair_layer_row_count": 0,
            "cuda_loaded": False,
            "sealed_split_read": False,
        }
        base.write_json(summary_path, payload)
        print(summary_path)
        return summary_path
    if restart:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    completed: set[str] = set()
    if output_path.exists():
        for row in base.read_jsonl(output_path):
            completed.add(row["physical_pair_id"])
    pair_cases = {row["case_id"]: row for row in base.read_jsonl(CASES_PATH) if row["model"] == model_name}
    event_map = {
        row["event_id"]: row for row in base.read_jsonl(EVENTS_PATH) if row["model"] == model_name
    }
    model = None
    handles: list[Any] = []
    captures: dict[tuple[str, int], torch.Tensor] = {}
    started = time.monotonic()
    new_pairs = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase546 physical collection requires CUDA")
        model, tokenizer, device = base.load_model(model_name)
        layers = base.get_layers(model)
        handles = base.install_hooks(layers, captures)
        for pair in pairs:
            if pair["physical_pair_id"] in completed:
                continue
            event = event_map[pair["frozen_event_id"]]
            row_a = pair_cases[pair["world_a_case_id"]]
            row_b = pair_cases[pair["world_b_case_id"]]
            vectors_a, meta_a = collect_prompt_condition(
                model, tokenizer, device, layers, captures, row_a
            )
            vectors_b, meta_b = collect_prompt_condition(
                model, tokenizer, device, layers, captures, row_b
            )
            rows = combine_pair(
                model, tokenizer, layers, pair, event, row_a, row_b,
                vectors_a, vectors_b, meta_a, meta_b,
            )
            base.append_jsonl(output_path, rows)
            new_pairs += 1
            del vectors_a, vectors_b, rows
            gc.collect()
            if new_pairs == 1 or new_pairs % 8 == 0 or len(completed) + new_pairs == len(pairs):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} upstream pairs "
                    f"{len(completed) + new_pairs}/{len(pairs)}",
                    flush=True,
                )

        rows = base.read_jsonl(output_path)
        observed_pairs = {row["physical_pair_id"] for row in rows}
        if len(observed_pairs) != len(pairs):
            raise RuntimeError(f"Incomplete Phase546 {model_name}: {len(observed_pairs)}/{len(pairs)}")
        payload = {
            "schema_version": "phase546_collection_summary.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_pair_count": len(pairs),
            "completed_pair_count": len(observed_pairs),
            "pair_layer_row_count": len(rows),
            "layer_count": len(layers),
            "new_pairs_this_invocation": new_pairs,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "max_component_ledger_relative_error": max(
                row["max_component_ledger_relative_error"] for row in rows
            ),
            "output_path": str(output_path.relative_to(ROOT)),
            "output_sha256": base.sha256_file(output_path),
            "cuda_loaded": True,
            "prompt_end_only": True,
            "full_hidden_vectors_persisted": False,
            "head_channel_neuron_scan_executed": False,
            "sealed_split_read": False,
            "protocol_sha256": base.sha256_file(PROTOCOL_PATH),
        }
        base.write_json(summary_path, payload)
        print(summary_path)
        return summary_path
    finally:
        for handle in handles:
            handle.remove()
        captures.clear()
        if model is not None:
            base.release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run_model(args.model, args.restart)


if __name__ == "__main__":
    main()
