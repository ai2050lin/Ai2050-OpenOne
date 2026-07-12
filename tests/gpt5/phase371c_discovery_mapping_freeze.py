#!/usr/bin/env python3
"""Freeze semantic mapping code after the gate and blind-row hashes are sealed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
GATE = PHASE371 / "phase371c_semantic_discovery_gate.json"
AUDIT = PHASE371 / "phase371c_blind_vector_contrast/phase371c_blind_contrast_audit.json"
OUT = PHASE371 / "phase371c_discovery_mapping_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    script = ROOT / "tests/gpt5/phase371c_discovery_key_and_mapping.py"
    payload = {
        "schema_version": "47.20.0",
        "phase_id": "Phase371C-Discovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": audit["valid"] and gate["authorization"]["run_semantic_mapping_on_discovery_rows"],
        "frozen_hashes": {
            "mapping_script": sha256_file(script),
            "gate": sha256_file(GATE),
            "blind_audit": sha256_file(AUDIT),
            "blind_rows": audit["sealed_hashes"]["route_rows"],
        },
        "authorization": {
            "open_fresh_discovery_condition_semantics": True,
            "open_calibration_condition_semantics": False,
            "open_physical_condition_semantics": False,
            "full_candidate_before_history_gate": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
