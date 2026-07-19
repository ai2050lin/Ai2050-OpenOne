#!/usr/bin/env python3
"""Freeze Phase559 path behavior after, and only after, behavior authorization."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
SUMMARY_PATH = OUT_DIR / "phase559_behavior_summary.json"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
PATH_CONTRACT_PATH = OUT_DIR / "phase559_path_behavior_frozen_contract.json"
PATH_SPLITS = ("path_discovery", "path_confirmation", "unseen_recombination")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict[str, Any]:
    behavior = read_json(SUMMARY_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if behavior["authorized_models"] != ["qwen3"]:
        raise RuntimeError(
            "Phase559 path contract expected only Qwen3 authorization; no discretionary model selection"
        )
    row_counts = {
        split: int(protocol["split_world_counts"][split]) * 32 for split in PATH_SPLITS
    }
    payload = {
        "schema_version": "phase559_path_behavior_frozen_contract.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "parent_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "behavior_summary_sha256": sha256_file(SUMMARY_PATH),
        "authorized_models": ["qwen3"],
        "selected_splits": list(PATH_SPLITS),
        "row_counts": row_counts,
        "expected_rows_per_model": sum(row_counts.values()),
        "path_gate": {
            "world_all_32_rate_min_per_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "all_three_splits_required": True,
            "internal_anchor_requires_all_32_correct": True,
        },
        "evidence_policy": {
            "path_behavior_before_hidden_state_collection": True,
            "glm4_and_deepseek7b_remain_closed": True,
            "sealed_split_read": False,
        },
    }
    write_json(PATH_CONTRACT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    freeze()
