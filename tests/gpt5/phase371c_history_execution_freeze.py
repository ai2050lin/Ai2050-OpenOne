#!/usr/bin/env python3
"""Freeze exact history evaluator code before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PROTOCOL = PHASE371 / "phase371c_history_residual_protocol.json"
OUT = PHASE371 / "phase371c_history_execution_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    script = ROOT / "tests/gpt5/phase371c_exact_history_residual.py"
    payload = {
        "schema_version": "47.22.0",
        "phase_id": "Phase371C-History",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": protocol["authorization"]["implement_and_hash_exact_history_evaluator"],
        "frozen_hashes": {
            "evaluator": sha256_file(script),
            "protocol": sha256_file(PROTOCOL),
            "model_candidates": sha256_file(
                PHASE371 / "phase371c_discovery_mapping/private/phase371c_provisional_model_candidates.jsonl"
            ),
            "group_gates": sha256_file(
                PHASE371 / "phase371c_discovery_mapping/private/phase371c_group_gate_rows.jsonl"
            ),
        },
        "authorization": {
            "execute_exact_history_projection": True,
            "model_execution": False,
            "open_calibration": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
