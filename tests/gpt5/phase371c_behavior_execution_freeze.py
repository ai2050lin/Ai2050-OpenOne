#!/usr/bin/env python3
"""Freeze behavior execution code hash after the Phase371C case contract passes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
AUDIT = PHASE371 / "phase371c_case_bank/phase371c_case_contract_audit.json"
OUT = PHASE371 / "phase371c_behavior_execution_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    behavior_script = ROOT / "tests/gpt5/phase371c_behavior_qualification.py"
    case_file = PHASE371 / "phase371c_case_bank/private/phase371c_nonphysical_execution_cases.jsonl"
    physical_file = PHASE371 / "phase371c_case_bank/sealed/private/phase371c_physical_execution_cases.jsonl"
    valid = bool(audit["valid"]) and len(case_file.read_text(encoding="utf-8").splitlines()) == 864
    payload = {
        "schema_version": "47.9.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "frozen_hashes": {
            "behavior_script": sha256_file(behavior_script),
            "nonphysical_execution_cases": sha256_file(case_file),
            "sealed_physical_execution_cases": sha256_file(physical_file),
        },
        "execution": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "case_count_per_model": 288,
            "max_new_tokens": 24,
            "equal_token_length_buckets": True,
            "physical_case_count_loaded": 0,
        },
        "authorization": {
            "run_nonphysical_behavior_qualification": valid,
            "internal_collection": False,
            "physical_execution": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
