#!/usr/bin/env python3
"""Freeze Phase573 sealed validation without reading sealed cases."""

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
CAUSAL_DECISION = OUT_DIR / "phase573_coarse_message_causal_decision.json"
CAUSAL_SUMMARY = OUT_DIR / "phase573_qwen3_coarse_message_causal_summary.json"
CAUSAL_PROTOCOL = OUT_DIR / "phase573_coarse_message_causal_protocol.json"
SEALED_COMMITMENT = OUT_DIR / "phase573_sealed_commitment.json"
SEALED_PROTOCOL = OUT_DIR / "phase573_sealed_validation_protocol.json"


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
    decision = read_json(CAUSAL_DECISION)
    summary = read_json(CAUSAL_SUMMARY)
    causal = read_json(CAUSAL_PROTOCOL)
    commitment = read_json(SEALED_COMMITMENT)
    if not decision["sealed_execution_authorized"]:
        raise RuntimeError("Phase573 causal decision did not authorize sealed execution")
    if not summary["coarse_message_causal_gate_pass"]:
        raise RuntimeError("Phase573 dual causal split did not pass")
    payload = {
        "schema_version": "phase573_sealed_validation_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": "sealed",
        "sealed_case_count_committed": commitment["sealed_case_count"],
        "sealed_cases_sha256_committed": commitment["sealed_cases_sha256"],
        "candidate_layer": causal["candidate_layer"],
        "candidate_receiver": causal["candidate_receiver"],
        "wrong_depth_control_layer": causal["wrong_depth_control_layer"],
        "wrong_position_control": causal["wrong_position_control"],
        "conditions": causal["conditions"],
        "behavior_gate": causal["behavior_gate"],
        "causal_gate": causal["causal_gate"],
        "reconstruction_relative_error_max": causal[
            "reconstruction_relative_error_max"
        ],
        "permutations": causal["permutations"],
        "behavior_batch_size": causal["behavior_batch_size"],
        "causal_batch_worlds": causal["causal_batch_worlds"],
        "no_threshold_or_candidate_update_after_opening": True,
        "sealed_cases_read_during_freeze": False,
        "causal_decision_sha256": sha256_file(CAUSAL_DECISION),
        "causal_summary_sha256": sha256_file(CAUSAL_SUMMARY),
        "causal_protocol_sha256": sha256_file(CAUSAL_PROTOCOL),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT),
    }
    write_json(SEALED_PROTOCOL, payload)
    print(json.dumps({
        "sealed_cases_read": False,
        "committed_case_count": commitment["sealed_case_count"],
        "candidate_layer": payload["candidate_layer"],
        "protocol": str(SEALED_PROTOCOL.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    freeze()
