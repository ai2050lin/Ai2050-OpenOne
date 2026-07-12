#!/usr/bin/env python3
"""Freeze blind contrast code and exact expected denominator before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PROTOCOL = PHASE371 / "phase371c_blind_vector_contrast_protocol.json"
OUT = PHASE371 / "phase371c_blind_contrast_execution_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    script = ROOT / "tests/gpt5/phase371c_blind_vector_contrast.py"
    payload = {
        "schema_version": "47.17.0",
        "phase_id": "Phase371C-Contrast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": bool(protocol["authorization"]["implement_and_hash_blind_contrast_extractor"]),
        "frozen_hashes": {
            "extractor": sha256_file(script),
            "protocol": sha256_file(PROTOCOL),
            "collector_cases": sha256_file(
                PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
            ),
            "base_audit": sha256_file(PHASE371 / "phase371c_internal_collection_audit.json"),
            "adjacent_audit": sha256_file(PHASE371 / "phase371c_adjacent_extension_audit.json"),
        },
        "expected_denominator": {
            "route_rows": 66 * 6 * 3 * 3 * 4 * 21,
            "vocab_rows": 66 * 6 * 3,
            "semantic_condition_key_opened": False,
        },
        "authorization": {
            "execute_blind_contrast": True,
            "select_candidate_during_extraction": False,
            "open_semantics": False,
            "open_calibration": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
