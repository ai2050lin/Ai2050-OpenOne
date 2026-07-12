#!/usr/bin/env python3
"""Seal Phase376 intervention code before model execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
FILES = {
    "protocol": OUT / "phase376_intervention_protocol.json",
    "decision_alignment": OUT / "phase376_decision_time_alignment_summary.json",
    "intervention_script": ROOT / "tests/gpt5/phase376_decision_aligned_intervention.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(FILES["protocol"].read_text(encoding="utf-8"))
    alignment = json.loads(FILES["decision_alignment"].read_text(encoding="utf-8"))
    valid = bool(
        protocol["authorization"]["run_all_preregistered_discovery_interventions"]
        and alignment["results"]["decision_aligned_recollection_required"]
    )
    payload = {
        "schema_version": "49.1.1",
        "phase_id": "Phase376-InterventionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "sealed_hashes": {name: sha256(path) for name, path in FILES.items()},
        "execution_contract": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "device": "cuda",
            "one_model_at_a_time": True,
            "discovery_only": True,
            "calibration_opened": False,
            "physical_opened": False,
            "gates_mutable_after_execution": False,
        },
    }
    path = OUT / "phase376_intervention_execution_freeze.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
