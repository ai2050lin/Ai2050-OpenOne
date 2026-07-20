#!/usr/bin/env python3
"""Freeze the Phase574 query-condition trace and coarse causal search."""

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


MODEL = "qwen3"
OUT_DIR = protocol.OUT_DIR
TRACE_PROTOCOL = OUT_DIR / "phase574_query_source_trace_protocol.json"
BEHAVIOR_REGISTRY = OUT_DIR / "phase574_qwen3_behavior_registry.json"
BEHAVIOR_SUMMARY = OUT_DIR / "phase574_qwen3_behavior_summary.json"
COMPONENTS = (
    "query_relation_value_message",
    "query_terminal_attention_output",
)
LAYERS = tuple(range(5, 25))
BANDS = tuple(protocol.DEPTH_BANDS)
CAUSAL_CANDIDATES = tuple(
    (component, band[0], band[1])
    for component in COMPONENTS
    for band in BANDS
)
CAUSAL_CONDITIONS = (
    "natural_baseline",
    "recipient_remove",
    "recipient_remove_restore",
    "relation_donor_replace",
    "object_donor_replace",
    "order_donor_replace",
    "wrong_depth_relation_replace",
    "wrong_position_relation_replace",
    "channel_roll_relation_replace",
)


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
    registry = read_json(BEHAVIOR_REGISTRY)
    summary = read_json(BEHAVIOR_SUMMARY)
    if not registry["authorized_for_query_source_trace"]:
        raise RuntimeError("Phase574 Qwen3 did not pass the frozen behavior gate")
    if not summary["authorized_for_query_source_trace"]:
        raise RuntimeError("Phase574 behavior summary/registry authorization drift")
    selected = registry["selected_base_case_ids_by_split"]
    if set(selected) != set(protocol.STRUCTURE_SPLITS):
        raise RuntimeError("Phase574 selected structure split drift")
    if any(len(selected[split]) != 128 for split in protocol.STRUCTURE_SPLITS):
        raise RuntimeError("Phase574 trace requires 128 worlds per structure split")

    candidate_rows = [
        {
            "candidate_id": f"{component}__L{start}_L{end}",
            "component": component,
            "band_start": start,
            "band_end": end,
            "patch_layers": list(range(start, end + 1)),
        }
        for component, start, end in CAUSAL_CANDIDATES
    ]
    payload = {
        "schema_version": "phase574_query_source_trace_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "structure_splits": list(protocol.STRUCTURE_SPLITS),
        "worlds_per_structure_split": 128,
        "selected_base_case_ids_by_split": selected,
        "trace_layers": list(LAYERS),
        "trace_components": list(COMPONENTS),
        "causal_candidates": candidate_rows,
        "causal_conditions": list(CAUSAL_CONDITIONS),
        "observer_coordinates": {
            "query_receiver": "query_terminal",
            "query_sources": ["query_relation", "query_object"],
            "downstream_receiver": "answer_boundary",
            "downstream_layer": 24,
            "downstream_sources": ["selected_fact", "other_relation_fact"],
            "qkv_projection_stage": "pre_rotary_projection",
        },
        "natural_trace_gate": {
            "full_attention_relation_relative_delta_minimum": 0.05,
            "full_attention_relation_world_rate_minimum_each_split": 0.75,
            "layer5_event_must_replicate": True,
            "layer24_relation_selection_pair_rate_minimum_each_split": 0.75,
            "layer24_object_selection_pair_rate_minimum_each_split": 0.75,
            "layer24_order_preservation_pair_rate_minimum_each_split": 0.75,
            "causal_prefix_maximum_relative_delta": 1e-5,
        },
        "causal_selection_rule": {
            "discovery_evaluates_all_eight_candidates": True,
            "primary_score": "relation_donor_route_switch_effect",
            "eligible_candidate_requires_relation_effect_positive_rate": 0.60,
            "eligible_candidate_requires_relation_effect_mean_minimum": 0.01,
            "eligible_candidate_requires_relation_vs_each_control_gap_minimum": 0.01,
            "eligible_candidate_requires_remove_recipient_margin_damage_mean": 0.01,
            "eligible_candidate_requires_restore_max_candidate_logit_delta": 1e-4,
            "tie_break": "lexicographic_candidate_id",
            "confirmation_reuses_exact_selected_candidate": True,
            "heldout_reuses_exact_selected_candidate": True,
            "pipeline_permutation_count": 1024,
            "pipeline_permutation_maximum_over_all_candidates": True,
            "maximum_smoothed_tail_fraction": 0.01,
        },
        "full_generation_gate": {
            "conditions": [
                "natural_baseline",
                "recipient_remove",
                "recipient_remove_restore",
                "relation_donor_replace",
                "object_donor_replace",
                "order_donor_replace",
            ],
            "relation_donor_target_win_rate_gain_minimum": 0.10,
            "restore_exact_semantic_mismatch_maximum": 0,
            "wrong_object_and_order_must_not_match_relation_gain": True,
        },
        "open_discovery_full_vector_world_cap": 32,
        "right_padding_and_explicit_position_ids_required": True,
        "full_short_generation_required": True,
        "delete_then_restore_separate_hooks_required": True,
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "phase574_protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "behavior_registry_sha256": sha256_file(BEHAVIOR_REGISTRY),
        "behavior_summary_sha256": sha256_file(BEHAVIOR_SUMMARY),
    }
    if TRACE_PROTOCOL.exists():
        existing = read_json(TRACE_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase574 query-source trace protocol drift")
    else:
        write_json(TRACE_PROTOCOL, payload)
    print(json.dumps({
        "model": MODEL,
        "structure_worlds": sum(len(ids) for ids in selected.values()),
        "trace_layers": len(LAYERS),
        "causal_candidate_count": len(candidate_rows),
        "sealed_split_read": False,
    }, ensure_ascii=False, indent=2))
    return TRACE_PROTOCOL


if __name__ == "__main__":
    freeze()
