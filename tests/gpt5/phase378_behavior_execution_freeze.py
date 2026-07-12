#!/usr/bin/env python3
"""Seal Phase378 physical behavior code before opening cases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
FILES = {
    "protocol": OUT / "phase378_physical_protocol.json",
    "behavior_script": ROOT / "tests/gpt5/phase378_physical_behavior.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(FILES["protocol"].read_text(encoding="utf-8"))
    valid = bool(protocol["authorization"]["open_two_mechanism_physical_behavior_cases"])
    payload = {
        "schema_version": "51.0.1",
        "phase_id": "Phase378-BehaviorFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "sealed_hashes": {name: sha256(path) for name, path in FILES.items()},
        "execution_contract": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "batch_sizes": {"qwen3": 8, "glm4": 2, "deepseek7b": 8},
            "device": "cuda",
            "one_model_at_a_time": True,
            "failed_groups_replaced": False,
        },
    }
    path = OUT / "phase378_behavior_execution_freeze.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
