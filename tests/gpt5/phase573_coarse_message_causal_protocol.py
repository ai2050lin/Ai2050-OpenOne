#!/usr/bin/env python3
"""Freeze the Phase573 coarse semantic-fact message causal contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase573"
MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase573_natural_transition"
TRACE_DECISION = OUT_DIR / "phase573_natural_trace_decision.json"
TRACE_SUMMARY = OUT_DIR / "phase573_qwen3_natural_trace_summary.json"
FROZEN_PROTOCOL = OUT_DIR / "phase573_frozen_protocol.json"
OPEN_CASES = OUT_DIR / "phase573_open_cases.jsonl.gz"
CAUSAL_PROTOCOL = OUT_DIR / "phase573_coarse_message_causal_protocol.json"

CAUSAL_SPLITS = ("causal_discovery", "causal_confirmation")
CONDITIONS = (
    "same_case_restore",
    "selected_edge_remove",
    "nonselected_edge_remove",
    "paired_relation_selected_replace",
    "channel_roll_donor_replace",
    "wrong_depth_donor_replace",
    "wrong_position_donor_replace",
)
RELATION_SCREEN_MINIMUM = 192
CONTROL_SCREEN_CAP = 384
FINAL_WORLDS_PER_SPLIT = 128
NOOP_REPEATS = 2
BEHAVIOR_BATCH_SIZE = 8
CAUSAL_BATCH_WORLDS = 8
PERMUTATIONS = 1024


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


def freeze() -> dict[str, Any]:
    decision = read_json(TRACE_DECISION)
    summary = read_json(TRACE_SUMMARY)
    if not decision["coarse_message_causal_authorized"]:
        raise RuntimeError("Phase573 natural trace did not authorize coarse causal work")
    if not summary["causal_mask_prefix_audit_pass"]:
        raise RuntimeError("Phase573 natural trace failed its causal-prefix audit")
    candidate = decision["earliest_routing_event"]
    if candidate is None:
        raise RuntimeError("Phase573 has no frozen routing candidate")
    candidate_layer = int(candidate["layer"])
    receiver = candidate["receiver_role"]
    if candidate_layer != 24 or receiver != "answer_boundary":
        raise RuntimeError(f"Unexpected Phase573 routing candidate: {candidate}")
    payload = {
        "schema_version": "phase573_coarse_message_causal_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": MODEL,
        "causal_splits": list(CAUSAL_SPLITS),
        "candidate_layer": candidate_layer,
        "candidate_receiver": receiver,
        "candidate_source": "semantic selected fact object+relation+value token union",
        "wrong_depth_control_layer": 12,
        "wrong_position_control": "query_terminal",
        "conditions": list(CONDITIONS),
        "behavior_gate": {
            "minimum_relation_qualified_worlds_each_split": RELATION_SCREEN_MINIMUM,
            "control_screen_cap_each_split": CONTROL_SCREEN_CAP,
            "final_worlds_each_split": FINAL_WORLDS_PER_SPLIT,
            "both_noop_repeats_must_be_exactly_and_semantically_stable": True,
            "base_relation_object_order_must_all_be_correct": True,
        },
        "causal_gate": {
            "minimum_positive_effect_rate": 0.65,
            "minimum_mean_donor_switch_effect": 0.0,
            "minimum_mean_gap_vs_control": 0.05,
            "minimum_donor_candidate_win_rate": 0.10,
            "required_primary_conditions": [
                "selected_edge_remove", "paired_relation_selected_replace",
            ],
            "required_controls": [
                "nonselected_edge_remove", "channel_roll_donor_replace",
                "wrong_depth_donor_replace", "wrong_position_donor_replace",
            ],
            "discovery_and_confirmation_must_pass_independently": True,
        },
        "reconstruction_relative_error_max": 0.02,
        "permutations": PERMUTATIONS,
        "permutation_interpretation": (
            "fixed deterministic sign-flip audit of paired case effects; descriptive "
            "support only, not a substitute for confirmation"
        ),
        "behavior_batch_size": BEHAVIOR_BATCH_SIZE,
        "causal_batch_worlds": CAUSAL_BATCH_WORLDS,
        "observer_padding": "right",
        "position_ids": "cumulative attention mask",
        "post_softmax_value_contribution_intervention": True,
        "key_effect_identified": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "causal_splits_read_at_freeze": False,
        "sealed_split_read": False,
        "trace_decision_sha256": sha256_file(TRACE_DECISION),
        "trace_summary_sha256": sha256_file(TRACE_SUMMARY),
        "frozen_protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "open_cases_sha256": sha256_file(OPEN_CASES),
    }
    write_json(CAUSAL_PROTOCOL, payload)
    print(json.dumps({
        "candidate_layer": candidate_layer,
        "candidate_receiver": receiver,
        "causal_splits_read": False,
        "protocol": str(CAUSAL_PROTOCOL.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    freeze()
