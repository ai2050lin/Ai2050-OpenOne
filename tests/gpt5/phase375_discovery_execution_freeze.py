#!/usr/bin/env python3
"""Freeze Phase375 discovery code and inputs before semantic execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
FILES = {
    "protocol": OUT / "phase375_protocol.json",
    "blind_inventory_audit": OUT / "phase375_blind_inventory_audit.json",
    "discovery_analyzer": ROOT / "tests/gpt5/phase375_multivector_discovery.py",
    "condition_key": ROOT
    / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_discovery_mapping/private/phase371c_discovery_condition_key.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(FILES["blind_inventory_audit"].read_text(encoding="utf-8"))
    valid = bool(audit["authorization"]["run_discovery_subgraph_gate"])
    payload = {
        "schema_version": "48.2.1",
        "phase_id": "Phase375-DiscoveryFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "sealed_hashes": {name: sha256(path) for name, path in FILES.items()},
        "execution_contract": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "model_execution": False,
            "semantic_scope": "discovery_only",
            "calibration_opened": False,
            "physical_opened": False,
            "numeric_gates_mutable_after_execution": False,
        },
    }
    path = OUT / "phase375_discovery_execution_freeze.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
