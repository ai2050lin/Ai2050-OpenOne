#!/usr/bin/env python3
"""Freeze Phase574 causal execution after natural endpoint replication."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase574_query_source_protocol as protocol  # noqa: E402
import phase574_query_source_trace as trace  # noqa: E402
import phase574_query_source_trace_protocol as trace_protocol  # noqa: E402


MODEL = "qwen3"
CAUSAL_PROTOCOL = protocol.OUT_DIR / "phase574_query_source_causal_protocol.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> Path:
    decision = read_json(trace.DECISION_PATH)
    if not decision["coarse_query_source_causal_authorized"]:
        raise RuntimeError("Phase574 natural trace did not authorize causal execution")
    frozen_trace = read_json(trace_protocol.TRACE_PROTOCOL)
    candidates = frozen_trace["causal_candidates"]
    wrong_depth = {}
    trace_layers = list(trace_protocol.LAYERS)
    half_turn = len(trace_layers) // 2
    for layer in trace_layers:
        index = trace_layers.index(layer)
        wrong_depth[str(layer)] = trace_layers[(index + half_turn) % len(trace_layers)]
    payload = {
        "schema_version": "phase574_query_source_causal_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "causal_splits": list(protocol.CAUSAL_SPLITS),
        "candidate_rows": candidates,
        "conditions": list(trace_protocol.CAUSAL_CONDITIONS),
        "recipient_variant": "base",
        "donor_variant_by_condition": {
            "relation_donor_replace": "relation_swap",
            "object_donor_replace": "object_swap",
            "order_donor_replace": "order_swap",
            "wrong_depth_relation_replace": "relation_swap",
            "wrong_position_relation_replace": "relation_swap",
            "channel_roll_relation_replace": "relation_swap",
        },
        "strict_object_control": {
            "recipient_and_object_donor_have_same_relation": True,
            "recipient_and_object_donor_have_different_object": True,
            "recipient_only_base_variant": True,
        },
        "wrong_depth_layer_mapping": wrong_depth,
        "wrong_position_receiver": "answer_boundary",
        "normal_receiver": "query_terminal",
        "downstream_layer": 24,
        "causal_behavior": {
            "candidate_worlds_each_split": 1024,
            "relation_minimum_each_split": 192,
            "control_screen_cap_each_split": 384,
            "all_axis_worlds_selected_each_split": 128,
            "noop_repeats": 2,
            "batch_size": 8,
            "selection_uses_internal_state": False,
        },
        "causal_world_batch_size": 4,
        "causal_discovery_evaluates_all_candidates": True,
        "causal_confirmation_evaluates_selected_candidate_only": True,
        "selection_rule": frozen_trace["causal_selection_rule"],
        "full_generation_gate": frozen_trace["full_generation_gate"],
        "generation_max_new_tokens": 4,
        "generation_world_batch_size": 4,
        "delete_then_restore_uses_two_separate_ordered_hooks": True,
        "right_padded_causal_logits_and_routing": True,
        "left_padded_full_generation": True,
        "explicit_position_ids_for_right_padded_runs": True,
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "sealed_split_read": False,
        "trace_protocol_sha256": sha256_file(trace_protocol.TRACE_PROTOCOL),
        "trace_rows_sha256": sha256_file(trace.ROWS_PATH),
        "trace_decision_sha256": sha256_file(trace.DECISION_PATH),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
    }
    if CAUSAL_PROTOCOL.exists():
        existing = read_json(CAUSAL_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase574 causal protocol drift")
    else:
        write_json(CAUSAL_PROTOCOL, payload)
    print(json.dumps({
        "candidate_count": len(candidates),
        "causal_splits": list(protocol.CAUSAL_SPLITS),
        "recipient_variant": "base",
        "wrong_depth_mapping": "ten-layer half-turn",
        "sealed_split_read": False,
    }, ensure_ascii=False, indent=2))
    return CAUSAL_PROTOCOL


if __name__ == "__main__":
    freeze()
