#!/usr/bin/env python3
"""Freeze a minimal Phase575 causal test from replicated natural events."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase575_source_competition_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
NATURAL_ANALYSIS = OUT_DIR / "phase575_qwen3_natural_structure_analysis.json"
NATURAL_DECISION = OUT_DIR / "phase575_natural_structure_decision.json"
CAUSAL_PROTOCOL = OUT_DIR / "phase575_routing_causal_protocol.json"
DISCOVERY_CONDITIONS = (
    "natural_baseline",
    "q_relation_replace",
    "q_object_replace",
    "q_order_replace",
    "q_relation_wrong_depth_rescaled",
    "q_relation_delta_roll",
    "q_relation_score_restore",
    "wrong_receiver_q_relation_replace",
    "score_relation_replace",
    "score_object_replace",
    "score_order_replace",
    "score_equalize",
    "score_equalize_restore",
    "score_relation_weight_restore",
    "weight_relation_replace",
    "weight_object_replace",
    "weight_order_replace",
    "weight_relation_restore",
    "value_group_swap_positive_control",
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
    analysis = read_json(NATURAL_ANALYSIS)
    decision = read_json(NATURAL_DECISION)
    if not decision["causal_protocol_authorized"]:
        raise RuntimeError("Phase575 natural replication did not authorize causality")
    if decision["analysis_sha256"] != sha256_file(NATURAL_ANALYSIS):
        raise RuntimeError("Phase575 natural decision/analysis hash drift")
    if analysis["causal_splits_read"] or analysis["sealed_split_read"]:
        raise RuntimeError("Phase575 natural discovery crossed evidence boundaries")

    ranked = analysis["ranked_coordinates"]
    score = next(
        row
        for row in ranked
        if row["channel"] == "score"
        and row["receiver"] == "answer_boundary"
        and row["layer"] == 24
        and row["replicated_routing_event"]
    )
    weight = next(
        row
        for row in ranked
        if row["channel"] == "weight"
        and row["receiver"] == "answer_boundary"
        and row["layer"] == 24
        and row["replicated_routing_event"]
    )
    payload = {
        "schema_version": "phase575_routing_causal_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": "qwen3",
        "selection_was_frozen_after_natural_replication": True,
        "selected_coordinate": {
            "layer": 24,
            "receiver": "answer_boundary",
            "source_groups": [
                "anchor_base_selected",
                "anchor_base_other_relation",
            ],
            "score_natural_gate": score,
            "weight_natural_gate": weight,
            "reason": (
                "the only layer where raw score, normalized weight, and value-message "
                "events all independently replicated at the answer boundary"
            ),
        },
        "wrong_depth_control_layer": 23,
        "discovery_conditions": list(DISCOVERY_CONDITIONS),
        "causal_behavior": {
            "splits": list(protocol.CAUSAL_SPLITS),
            "relation_screen_worlds_each_split": 1024,
            "control_screen_cap_each_split": 384,
            "minimum_relation_qualified_each_split": 192,
            "selected_five_variant_worlds_each_split": 128,
            "two_exact_and_semantic_noop_repeats_required": True,
            "batch_size": protocol.FIXED_BATCH_SIZE,
            "max_new_tokens": 4,
        },
        "execution": {
            "world_batch_size": 8,
            "post_rotary_q_replace_preserves_receiver_k_and_v": True,
            "score_and_weight_replace_keep_recipient_values": True,
            "score_equalize_uses_two_fixed_source_groups": True,
            "restore_uses_direct_natural_tensor_overwrite": True,
            "right_padding_and_explicit_position_ids_required": True,
            "full_short_generation_after_open_confirmation": True,
            "full_short_generation_max_new_tokens": 4,
        },
        "discovery_gate": {
            "relation_route_effect_positive_rate_minimum": 0.80,
            "relation_route_effect_mean_minimum": 0.02,
            "relation_vs_object_effect_gap_minimum": 0.01,
            "relation_vs_order_effect_gap_minimum": 0.01,
            "restore_route_maximum_absolute_delta": 1e-5,
            "restore_candidate_logit_maximum_absolute_delta": 1e-4,
            "mediation_remaining_fraction_maximum": 0.20,
            "behavior_relation_logit_effect_positive_rate_minimum": 0.60,
            "behavior_relation_logit_effect_mean_minimum": 0.02,
            "pipeline_resample_count": 1024,
            "maximum_branch_smoothed_tail_fraction": 0.01,
        },
        "confirmation_rule": {
            "only_the_selected_discovery_branch_is_run": True,
            "same_direction_and_thresholds_required": True,
            "heldout_structure_was_already_required_before_freeze": True,
            "sealed_can_open_only_after_discovery_and_confirmation_behavior_pass": True,
        },
        "branch_definitions": {
            "query": {
                "relation": "q_relation_replace",
                "object": "q_object_replace",
                "order": "q_order_replace",
                "restore_or_mediator": "q_relation_score_restore",
            },
            "score": {
                "relation": "score_relation_replace",
                "object": "score_object_replace",
                "order": "score_order_replace",
                "restore_or_mediator": "score_relation_weight_restore",
            },
            "weight": {
                "relation": "weight_relation_replace",
                "object": "weight_object_replace",
                "order": "weight_order_replace",
                "restore_or_mediator": "weight_relation_restore",
            },
        },
        "positive_and_necessity_controls": [
            "value_group_swap_positive_control",
            "score_equalize",
            "score_equalize_restore",
        ],
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "natural_analysis_sha256": sha256_file(NATURAL_ANALYSIS),
        "natural_decision_sha256": sha256_file(NATURAL_DECISION),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "phase575_protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
    }
    if CAUSAL_PROTOCOL.exists():
        existing = read_json(CAUSAL_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase575 routing causal protocol drift")
    else:
        write_json(CAUSAL_PROTOCOL, payload)
    print(
        json.dumps(
            {
                "model": payload["model"],
                "selected_coordinate": payload["selected_coordinate"],
                "condition_count": len(DISCOVERY_CONDITIONS),
                "causal_splits_read": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return CAUSAL_PROTOCOL


if __name__ == "__main__":
    freeze()
