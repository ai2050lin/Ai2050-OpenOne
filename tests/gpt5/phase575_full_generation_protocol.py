#!/usr/bin/env python3
"""Freeze Phase575 full-generation gates after open causal confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase575_routing_causal_protocol as causal_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
CONFIRMATION_DECISION = OUT_DIR / "phase575_routing_causal_confirmation_decision.json"
CONFIRMATION_SUMMARY = OUT_DIR / "phase575_qwen3_routing_causal_confirmation_summary.json"
GENERATION_PROTOCOL = OUT_DIR / "phase575_full_generation_protocol.json"


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
    decision = read_json(CONFIRMATION_DECISION)
    summary = read_json(CONFIRMATION_SUMMARY)
    if decision["summary_sha256"] != sha256_file(CONFIRMATION_SUMMARY):
        raise RuntimeError("Phase575 confirmation decision/summary hash drift")
    if not decision["full_short_generation_authorized"]:
        raise RuntimeError("Phase575 full short generation is not authorized")
    if summary["selected_branch"] != "score" or not summary[
        "open_confirmation_pass"
    ]:
        raise RuntimeError("Phase575 score branch did not pass open confirmation")
    payload = {
        "schema_version": "phase575_full_generation_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": "qwen3",
        "split": "causal_confirmation",
        "selected_branch": "score",
        "world_count": 128,
        "conditions": [
            "natural_baseline",
            "score_relation_replace",
            "score_object_replace",
            "score_order_replace",
            "score_relation_weight_restore",
        ],
        "execution_repeats": ["noop1", "noop2"],
        "world_batch_size": 8,
        "max_new_tokens": 4,
        "do_sample": False,
        "use_cache": False,
        "left_padding_and_explicit_attention_mask_required": True,
        "gates": {
            "natural_base_target_rate_minimum": 0.95,
            "relation_donor_target_rate_minimum": 0.40,
            "relation_donor_target_rate_gain_minimum": 0.30,
            "relation_vs_object_target_rate_gap_minimum": 0.20,
            "relation_vs_order_target_rate_gap_minimum": 0.20,
            "restore_exact_text_mismatch_maximum": 0,
            "restore_semantic_event_mismatch_maximum": 0,
            "repeat_exact_text_mismatch_maximum_each_condition": 0,
            "repeat_semantic_event_mismatch_maximum_each_condition": 0,
            "pipeline_resample_count": 1024,
            "smoothed_tail_fraction_maximum": 0.01,
        },
        "seal_policy": {
            "open_only_after_all_full_generation_gates_pass": True,
            "open_once": True,
            "same_frozen_score_operation_required": True,
        },
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "causal_splits_read": True,
        "sealed_split_read": False,
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "confirmation_summary_sha256": sha256_file(CONFIRMATION_SUMMARY),
        "confirmation_decision_sha256": sha256_file(CONFIRMATION_DECISION),
    }
    if GENERATION_PROTOCOL.exists():
        existing = read_json(GENERATION_PROTOCOL)
        ignored = {"created_at"}
        if {k: v for k, v in existing.items() if k not in ignored} != {
            k: v for k, v in payload.items() if k not in ignored
        }:
            raise RuntimeError("Phase575 full generation protocol drift")
    else:
        write_json(GENERATION_PROTOCOL, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return GENERATION_PROTOCOL


if __name__ == "__main__":
    freeze()
