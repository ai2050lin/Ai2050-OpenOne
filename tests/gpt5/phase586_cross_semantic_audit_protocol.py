#!/usr/bin/env python3
"""Freeze a blinded cross-model semantic audit of Phase585 open outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase585_object_swap_behavior as source_behavior
import phase585_object_swap_protocol as source


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase586"
MODELS = source.MODELS
JUDGE_REPEATS = ("judge1", "judge2")
FIXED_BATCH_SIZE = 32
MAX_NEW_TOKENS = 8
MIN_JUDGE_PARSE_RATE = 0.99
MIN_JUDGE_REPEAT_EXACT_RATE = 0.99
MIN_YES_VOTES = 2
MAX_NO_VOTES = 0

OUT_DIR = ROOT / "tests/gpt5/result/phase586_cross_semantic_audit"
PROTOCOL_PATH = OUT_DIR / "phase586_frozen_protocol.json"
DECISION_PATH = OUT_DIR / "phase586_cross_semantic_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def register() -> dict[str, Any]:
    source_artifacts: dict[str, Any] = {}
    for model in MODELS:
        paths = source_behavior.paths(model)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary["sealed_split_read"]:
            raise RuntimeError(f"Phase586 source {model} read sealed rows")
        if summary["rows_sha256"] != sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase586 source {model} row drift")
        source_artifacts[model] = {
            "rows_path": str(paths["rows"].relative_to(ROOT)),
            "rows_sha256": summary["rows_sha256"],
            "summary_path": str(paths["summary"].relative_to(ROOT)),
            "summary_sha256": sha256_file(paths["summary"]),
        }
    payload = {
        "schema_version": "phase586_cross_semantic_audit_protocol.v2",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Blinded cross-model audit of label-free natural responses",
        "source_phase": source.PHASE,
        "source_protocol_sha256": sha256_file(source.PROTOCOL_PATH),
        "source_artifacts": source_artifacts,
        "judge_models_in_required_execution_order": list(MODELS),
        "judge_repeats": list(JUDGE_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "judge_labels": ["YES", "NO", "AMBIGUOUS"],
        "interface_calibration_history": {
            "v1_status": "superseded_before_consensus_decision",
            "v1_artifact_directory": "tests/gpt5/result/phase586_cross_semantic_audit/v1_interface_calibration",
            "v1_failure": "deepseek7b emitted reasoning prefixes before the four-token limit",
            "v1_deepseek7b_parse_rate": 0.0010416666666666667,
            "v2_change": "all three judges rerun with an identical label-first instruction and eight-token limit",
            "scientific_cases_thresholds_and_consensus_rule_changed": False,
        },
        "judge_input_blinding": {
            "source_model_hidden": True,
            "split_hidden": True,
            "previous_alias_score_hidden": True,
            "internal_state_hidden": True,
            "raw_question_visible": True,
            "reference_fact_visible": True,
            "candidate_response_visible": True,
        },
        "consensus_gate": {
            "minimum_yes_votes": MIN_YES_VOTES,
            "maximum_no_votes": MAX_NO_VOTES,
            "all_judges_must_be_parseable_and_repeat_stable": True,
            "minimum_judge_parse_rate": MIN_JUDGE_PARSE_RATE,
            "minimum_judge_repeat_exact_rate": MIN_JUDGE_REPEAT_EXACT_RATE,
            "source_behavior_gate_reused_without_threshold_change": True,
        },
        "evidence_policy": {
            "retrospective_open_output_observer_calibration": True,
            "not_independent_behavior_confirmation": True,
            "not_internal_structure_evidence": True,
            "not_causal_evidence": True,
            "may_authorize_precommitted_sealed_behavior_confirmation": True,
            "cannot_directly_authorize_internal_trace": True,
            "sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
    }
    write_json(PROTOCOL_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
