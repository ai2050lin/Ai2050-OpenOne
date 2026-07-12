#!/usr/bin/env python3
"""Seal Phase377 calibration code before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration"
FILES = {
    "protocol": OUT / "phase377_calibration_protocol.json",
    "case_summary": OUT / "phase377_calibration_case_summary.json",
    "cases": OUT / "private/phase377_calibration_cases.jsonl",
    "intervention_script": ROOT / "tests/gpt5/phase377_calibration_intervention.py",
    "phase376_utility": ROOT / "tests/gpt5/phase376_decision_aligned_intervention.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(FILES["protocol"].read_text(encoding="utf-8"))
    cases = json.loads(FILES["case_summary"].read_text(encoding="utf-8"))
    valid = bool(
        protocol["authorization"]["run_calibration_interventions"] and cases["valid"]
    )
    payload = {
        "schema_version": "50.0.1",
        "phase_id": "Phase377-CalibrationFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "sealed_hashes": {name: sha256(path) for name, path in FILES.items()},
        "execution_contract": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "device": "cuda",
            "one_model_at_a_time": True,
            "calibration_only": True,
            "physical_opened": False,
            "gates_mutable_after_execution": False,
        },
    }
    path = OUT / "phase377_calibration_execution_freeze.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
