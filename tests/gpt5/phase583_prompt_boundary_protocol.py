#!/usr/bin/env python3
"""Freeze a first-token logit observer at candidate-free prompt boundaries."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase581_typed_category_protocol as source


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase583"
MODELS = source.MODELS
OPEN_SPLITS = source.OPEN_SPLITS
RELATIONS = source.RELATIONS
NOOP_REPEATS = ("forward1", "forward2")
FIXED_BATCH_SIZE = 32
MIN_TARGET_WIN_RATE = 0.90
MIN_MEAN_MARGIN = 0.05
MAX_REPEAT_LOGIT_DELTA = 1e-6
MIN_STABLE_SURFACES_PER_OBJECT = 6
MIN_QUALIFIED_BY_RELATION_CATEGORY = source.MIN_QUALIFIED_BY_RELATION_CATEGORY

OUT_DIR = ROOT / "tests/gpt5/result/phase583_prompt_boundary"
PROTOCOL_PATH = OUT_DIR / "phase583_frozen_protocol.json"


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
    source_protocol = json.loads(source.PROTOCOL_PATH.read_text(encoding="utf-8"))
    source_audit = json.loads(source.AUDIT_PATH.read_text(encoding="utf-8"))
    if not source_audit["valid"]:
        raise RuntimeError("Phase583 requires valid Phase581 source")
    if source_protocol["open_cases_sha256"] != sha256_file(source.OPEN_CASES_PATH):
        raise RuntimeError("Phase583 source cases drift")
    payload = {
        "schema_version": "phase583_prompt_boundary_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Candidate-free prompt-boundary first-token category observer",
        "models_in_required_execution_order": list(MODELS),
        "source_phase": source.PHASE,
        "source_protocol_sha256": sha256_file(source.PROTOCOL_PATH),
        "source_cases_path": str(source.OPEN_CASES_PATH.relative_to(ROOT)),
        "source_cases_sha256": sha256_file(source.OPEN_CASES_PATH),
        "open_splits": list(OPEN_SPLITS),
        "relations": list(RELATIONS),
        "relation_categories": {
            key: list(value) for key, value in source.RELATION_CATEGORIES.items()
        },
        "observer_labels": list(source.CATEGORY_ALIASES),
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "score_definition": {
            "score": "raw logit of each label's first natural continuation token at the prompt boundary",
            "margin": "target first-token logit minus relation foil first-token logit",
            "target_win": "margin strictly greater than zero",
            "candidate_words_inserted_into_model_input": False,
            "teacher_forced_continuation_used": False,
            "prior_or_length_calibration_used": False,
        },
        "behavior_gate": {
            "minimum_target_win_rate_each_split_relation": MIN_TARGET_WIN_RATE,
            "minimum_mean_margin_each_split_relation": MIN_MEAN_MARGIN,
            "maximum_repeat_logit_delta": MAX_REPEAT_LOGIT_DELTA,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_relation_category_each_split": MIN_QUALIFIED_BY_RELATION_CATEGORY,
            "all_three_open_splits_must_pass_per_relation": True,
            "model_relation_specific_qualification": True,
        },
        "evidence_policy": {
            "prompt_boundary_observer_not_natural_generation": True,
            "prompt_boundary_observer_not_parametric_storage_localization": True,
            "all_depth_prompt_trace_may_be_collected_after_gate": True,
            "causal_intervention_authorized": False,
            "sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
    }
    write_json(PROTOCOL_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
