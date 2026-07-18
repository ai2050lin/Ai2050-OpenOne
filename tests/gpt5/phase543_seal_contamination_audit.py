#!/usr/bin/env python3
"""Record and verify the Phase535 historical seal contamination correction."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/phase540_pair_answer_logodds_fresh_protocol.py"
CONTRACT = ROOT / "tests/gpt5/result/phase540_pair_answer_logodds_fresh_protocol/phase540_frozen_contract.json"
STATIC = ROOT / "tests/gpt5/result/phase540_pair_answer_logodds_fresh_protocol/phase540_static_audit.json"
OUT_DIR = ROOT / "tests/gpt5/result/phase543_seal_contamination_audit"
OUT_PATH = OUT_DIR / "phase543_seal_contamination_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    ast.parse(source_text)
    contract = read_json(CONTRACT)
    static = read_json(STATIC)
    expected_open = ["discovery", "entity_prediction", "relation_prediction"]
    correction_pass = (
        contract["phase535_prior_splits_read"] == expected_open
        and static["phase535_prior_splits_read"] == expected_open
        and contract["phase535_sealed_split_read_by_current_protocol"] is False
        and static["phase535_sealed_split_read_by_current_protocol"] is False
        and contract["historical_phase535_sealed_compromised_by_initial_phase540_audit"] is True
        and static["historical_phase535_sealed_compromised_by_initial_phase540_audit"] is True
        and contract["source_sha256"] == sha256_file(SOURCE)
    )
    payload = {
        "schema_version": "phase543_seal_contamination_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "historical_contamination_recorded_and_current_allowlist_verified" if correction_pass else "correction_failed",
        "incident": {
            "contaminated_split": "phase535.sealed",
            "incident_stage": "initial Phase540 static vocabulary-disjoint audit",
            "access_scope": "the complete JSONL file was parsed; only entity_names and relation_active were used by the audit",
            "model_outputs_or_hidden_states_read": False,
            "effect": "Phase535 historical seal is permanently invalid for future sealed claims",
            "phase540_current_sealed_affected": False,
        },
        "correction": {
            "phase535_prior_split_allowlist": expected_open,
            "current_source_sha256": sha256_file(SOURCE),
            "current_protocol_reads_phase535_sealed": False,
            "current_phase540_sealed_read": False,
            "correction_pass": correction_pass,
        },
        "evidence_boundary": {
            "global_any_sealed_split_read": True,
            "historical_phase535_sealed_read": True,
            "current_phase540_sealed_read": False,
            "pipeline_sealed": False,
            "closure_eligible": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)
    if not correction_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
