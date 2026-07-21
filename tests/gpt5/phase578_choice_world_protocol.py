#!/usr/bin/env python3
"""Freeze a world-level Phase578 gate over untouched Phase577 open splits."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase577_natural_choice_protocol as p577  # noqa: E402

PHASE = "Phase578"
MODELS = p577.MODELS
OPEN_SPLITS = p577.CAUSAL_SPLITS
SOURCE_CASES_PATH = p577.OPEN_CASES_PATH
OUT_DIR = ROOT / "tests/gpt5/result/phase578_choice_world"
PROTOCOL_PATH = OUT_DIR / "phase578_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase578_static_audit.json"
SEALED_REFERENCE_PATH = OUT_DIR / "phase578_sealed_reference.json"

MIN_STABLE_WORLD_RATE = 0.75
MIN_STABLE_WORLDS_PER_SPLIT = 168
MIN_STABLE_WORLDS_PER_RELATION = 84
MIN_DIVERSE_FRUITS = 8
MIN_DIVERSE_CONTROLS = 3
SELECTED_WORLDS_PER_SPLIT = 144
NATURAL_TRACE_WORLDS_PER_SPLIT = 72
CAUSAL_HOLDOUT_WORLDS_PER_SPLIT = 72


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source() -> Iterator[dict[str, Any]]:
    with gzip.open(SOURCE_CASES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def register() -> dict[str, Any]:
    rows = [row for row in iter_source() if row["split"] in OPEN_SPLITS]
    source_protocol = json.loads(p577.PROTOCOL_PATH.read_text(encoding="utf-8"))
    source_commitment = json.loads(
        p577.SEALED_COMMITMENT_PATH.read_text(encoding="utf-8")
    )
    audit = {
        "schema_version": "phase578_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "source_phase": p577.PHASE,
        "source_open_cases_sha256": sha256_file(SOURCE_CASES_PATH),
        "registered_case_count": len(rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "world_count": len({row["world_id"] for row in rows}),
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "unexpected_split_count": sum(row["split"] not in OPEN_SPLITS for row in rows),
        "sealed_row_count": sum(row["sealed"] for row in rows),
        "source_phase577_causal_splits_previously_executed": False,
        "source_sealed_split_read": source_protocol["evidence_policy"]["sealed_split_read"],
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == 896
        and set(audit["case_count_by_split"].values()) == {448}
        and audit["world_count"] == 448
        and all(
            audit[key] == 0
            for key in ("duplicate_case_id_count", "unexpected_split_count", "sealed_row_count")
        )
        and not audit["source_sealed_split_read"]
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    write_json(AUDIT_PATH, audit)
    write_json(SEALED_REFERENCE_PATH, {
        "schema_version": "phase578_sealed_reference.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "source_phase": p577.PHASE,
        "source_sealed_commitment_sha256": sha256_file(p577.SEALED_COMMITMENT_PATH),
        "source_sealed_cases_sha256": source_commitment["sealed_cases_sha256"],
        "source_sealed_case_count": source_commitment["sealed_case_count"],
        "sealed_rows_read": False,
        "may_open_only_after_open_natural_and_causal_confirmation": True,
    })
    frozen = {
        "schema_version": "phase578_choice_world_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "World-level stable natural fruit choice trajectory",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "source_phase": p577.PHASE,
        "source_case_path": str(SOURCE_CASES_PATH.relative_to(ROOT)),
        "source_case_sha256": sha256_file(SOURCE_CASES_PATH),
        "source_case_count": len(rows),
        "behavior_gate": {
            "minimum_stable_world_rate": MIN_STABLE_WORLD_RATE,
            "minimum_stable_worlds_per_split": MIN_STABLE_WORLDS_PER_SPLIT,
            "minimum_stable_worlds_per_relation": MIN_STABLE_WORLDS_PER_RELATION,
            "minimum_objects_with_both_relations": {
                "fruit": MIN_DIVERSE_FRUITS,
                "control": MIN_DIVERSE_CONTROLS,
            },
            "both_option_orders_and_two_exact_repeats_required_per_world": True,
        },
        "frozen_partition": {
            "selected_worlds_per_split": SELECTED_WORLDS_PER_SPLIT,
            "natural_trace_worlds_per_split": NATURAL_TRACE_WORLDS_PER_SPLIT,
            "causal_holdout_worlds_per_split": CAUSAL_HOLDOUT_WORLDS_PER_SPLIT,
            "partition_after_balanced_sort": "even ranks natural trace, odd ranks causal holdout",
        },
        "internal_policy": {
            "all_layers_observed_before_operator_definition": True,
            "natural_discovery_and_confirmation_splits_both_required": True,
            "causal_holdout_internal_state_unread_during_natural_discovery": True,
            "no_layer_head_channel_or_neuron_preselection": True,
        },
        "evidence_policy": {
            "phase577_behavior_results_motivated_world_level_gate": True,
            "phase577_thresholds_not_modified": True,
            "source_open_splits_previously_unexecuted": True,
            "source_sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
        "sealed_reference_path": str(SEALED_REFERENCE_PATH.relative_to(ROOT)),
        "sealed_reference_sha256": sha256_file(SEALED_REFERENCE_PATH),
    }
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
