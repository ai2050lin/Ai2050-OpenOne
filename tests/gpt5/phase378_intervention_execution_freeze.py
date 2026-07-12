#!/usr/bin/env python3
"""Seal Phase378 physical intervention code before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
FILES = {
    "protocol": OUT / "phase378_physical_protocol.json",
    "behavior_analysis": OUT / "phase378_physical_behavior_analysis_summary.json",
    "cases": OUT / "private/phase378_physical_intervention_cases.jsonl",
    "intervention_script": ROOT / "tests/gpt5/phase378_physical_intervention.py",
    "phase376_utility": ROOT / "tests/gpt5/phase376_decision_aligned_intervention.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    analysis = json.loads(FILES["behavior_analysis"].read_text(encoding="utf-8"))
    valid = bool(analysis["authorization"]["run_physical_interventions"])
    payload = {
        "schema_version": "51.2.1",
        "phase_id": "Phase378-InterventionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "sealed_hashes": {name: sha256(path) for name, path in FILES.items()},
        "execution_contract": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "device": "cuda",
            "one_model_at_a_time": True,
            "physical_only": True,
            "other_mechanisms_opened": False,
            "gates_mutable_after_execution": False,
        },
    }
    path = OUT / "phase378_intervention_execution_freeze.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
