#!/usr/bin/env python3
"""Freeze the one-shot Phase575 sealed validation before opening the seal."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase575_full_generation_protocol as generation_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
FULL_DECISION = OUT_DIR / "phase575_full_generation_decision.json"
FULL_SUMMARY = OUT_DIR / "phase575_qwen3_full_generation_summary.json"
SEALED_PROTOCOL = OUT_DIR / "phase575_sealed_validation_protocol.json"


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
    decision = read_json(FULL_DECISION)
    summary = read_json(FULL_SUMMARY)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    if decision["summary_sha256"] != sha256_file(FULL_SUMMARY):
        raise RuntimeError("Phase575 full-generation decision/summary hash drift")
    if not decision["sealed_validation_authorized"]:
        raise RuntimeError("Phase575 sealed validation is not authorized")
    if not summary["full_generation_gate_pass"]:
        raise RuntimeError("Phase575 full-generation gate did not pass")
    if commitment["sealed_split_read_for_analysis"]:
        raise RuntimeError("Phase575 sealed commitment was already opened")
    payload = {
        "schema_version": "phase575_sealed_validation_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": "qwen3",
        "split": "sealed",
        "one_shot": True,
        "selected_coordinate": {
            "layer": 24,
            "receiver": "answer_boundary",
            "branch": "score",
            "source_groups": [
                "anchor_base_selected",
                "anchor_base_other_relation",
            ],
        },
        "conditions": [
            "natural_baseline",
            "score_relation_replace",
            "score_object_replace",
            "score_order_replace",
            "score_relation_weight_restore",
        ],
        "behavior_qualification": {
            "candidate_world_count": 1024,
            "relation_screen_world_count": 1024,
            "control_screen_cap": 384,
            "minimum_relation_qualified": 192,
            "selected_five_variant_world_count": 128,
            "two_exact_and_semantic_noop_repeats_required": True,
            "batch_size": 8,
            "max_new_tokens": 4,
        },
        "causal_execution": {
            "world_batch_size": 8,
            "same_position_reexpression_of_pre_rotary_query_required": True,
            "recipient_keys_and_values_preserved": True,
            "direct_weight_restore_required": True,
        },
        "full_generation": {
            "world_count": 128,
            "execution_repeats": ["noop1", "noop2"],
            "max_new_tokens": 4,
            "do_sample": False,
            "use_cache": False,
        },
        "causal_gates": {
            "relation_route_effect_positive_rate_minimum": 0.80,
            "relation_route_effect_mean_minimum": 0.02,
            "relation_vs_object_effect_gap_minimum": 0.01,
            "relation_vs_order_effect_gap_minimum": 0.01,
            "restore_route_maximum_absolute_delta": 1e-5,
            "restore_candidate_logit_maximum_absolute_delta": 1e-4,
            "relation_logit_effect_positive_rate_minimum": 0.60,
            "relation_logit_effect_mean_minimum": 0.02,
            "pipeline_resample_count": 1024,
            "smoothed_tail_fraction_maximum": 0.01,
        },
        "generation_gates": read_json(generation_protocol.GENERATION_PROTOCOL)[
            "gates"
        ],
        "failure_policy": {
            "close_candidate_without_patch": True,
            "do_not_reopen_seal": True,
        },
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "sealed_split_read": False,
        "sealed_commitment_sha256": sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "committed_sealed_cases_sha256": commitment["sealed_cases_sha256"],
        "full_generation_summary_sha256": sha256_file(FULL_SUMMARY),
        "full_generation_decision_sha256": sha256_file(FULL_DECISION),
    }
    if SEALED_PROTOCOL.exists():
        existing = read_json(SEALED_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase575 sealed validation protocol drift")
    else:
        write_json(SEALED_PROTOCOL, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return SEALED_PROTOCOL


if __name__ == "__main__":
    freeze()
