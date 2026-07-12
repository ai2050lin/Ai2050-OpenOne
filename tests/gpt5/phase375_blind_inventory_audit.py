#!/usr/bin/env python3
"""Seal and audit the Phase375 blind finite-subgraph inventory."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
PROTOCOL = OUT / "phase375_protocol.json"
SUMMARY = OUT / "phase375_blind_inventory_summary.json"
INVENTORY = OUT / "private/phase375_blind_subgraph_inventory.jsonl"
EXTRACTOR = ROOT / "tests/gpt5/phase375_blind_subgraph_inventory.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(PROTOCOL)
    summary = read_json(SUMMARY)
    lines = [line for line in INVENTORY.read_text(encoding="utf-8").splitlines() if line]
    parsed = [json.loads(line) for line in lines]
    forbidden = {
        "family_id",
        "mechanism_id",
        "contrast_condition",
        "target",
        "distractors",
        "answer",
        "candidate_score",
    }
    forbidden_count = sum(sum(key in row for key in forbidden) for row in parsed)
    state_rows = sum(row["subgraph_kind"] == "state_graph" for row in parsed)
    formation_rows = sum(row["subgraph_kind"] == "formation_graph" for row in parsed)
    expected = summary["denominator"]
    valid = (
        protocol["authorization"]["freeze_inventory_hash_before_semantic_mapping"]
        and summary["valid"]
        and len(lines) == expected["total_inventory_row_count"]
        and state_rows == expected["state_graph_row_count"]
        and formation_rows == expected["formation_graph_row_count"]
        and forbidden_count == 0
        and all(not row["semantic_labels_available"] for row in parsed)
        and all(not row["candidate_selected"] for row in parsed)
    )
    audit = {
        "schema_version": "48.2.0",
        "phase_id": "Phase375-BlindInventoryAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "counts": {
            "inventory_row_count": len(lines),
            "state_graph_row_count": state_rows,
            "formation_graph_row_count": formation_rows,
            "forbidden_semantic_field_count": forbidden_count,
        },
        "sealed_hashes": {
            "protocol": sha256(PROTOCOL),
            "extractor": sha256(EXTRACTOR),
            "summary": sha256(SUMMARY),
            "inventory": sha256(INVENTORY),
        },
        "authorization": {
            "open_discovery_condition_key": valid,
            "run_discovery_subgraph_gate": valid,
            "open_calibration": False,
            "open_physical": False,
            "run_model_intervention": False,
        },
    }
    path = OUT / "phase375_blind_inventory_audit.json"
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
