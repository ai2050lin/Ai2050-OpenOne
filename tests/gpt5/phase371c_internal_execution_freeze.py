#!/usr/bin/env python3
"""Freeze Phase371C internal discovery collector hashes and storage authorization."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
ANALYSIS = PHASE371 / "phase371c_behavior_analysis/phase371c_behavior_analysis_summary.json"
CASES = PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
B_REPAIR = PHASE371 / "phase371b_sufficient_state_summary.json"
OUT = PHASE371 / "phase371c_internal_execution_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    phase371b = json.loads(B_REPAIR.read_text(encoding="utf-8"))
    script = ROOT / "tests/gpt5/phase371c_internal_collection.py"
    helper = ROOT / "tests/gpt5/phase371b_anchor_qk_collection.py"
    per_model_bytes = {
        row["model"]: int(row["total_byte_count"])
        for row in phase371b["models"]
    }
    cases_per_model = int(analysis["internal_discovery"]["case_count_per_model"])
    projected = sum(value * cases_per_model for value in per_model_bytes.values())
    budget = 64 * 1024**3
    reserve = 200 * 1024**3
    free = int(shutil.disk_usage(ROOT).free)
    valid = (
        analysis["results"]["partial_discovery_cycle_authorized"]
        and len(CASES.read_text(encoding="utf-8").splitlines()) == 264
        and projected <= budget
        and free - projected >= reserve
    )
    payload = {
        "schema_version": "47.11.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "frozen_hashes": {
            "collector": sha256_file(script),
            "exact_tree_helper": sha256_file(helper),
            "collector_cases": sha256_file(CASES),
        },
        "denominator": {
            "eligible_mechanisms": analysis["results"]["eligible_mechanisms"],
            "parallel_group_count": analysis["internal_discovery"]["parallel_group_count"],
            "case_count": analysis["internal_discovery"]["case_count"],
            "case_count_per_model": cases_per_model,
            "generation_time_count": 3,
            "anchor_layer_count": 3,
        },
        "storage": {
            "projected_bytes_from_measured_phase371b_files": projected,
            "budget_bytes": budget,
            "free_disk_bytes": free,
            "minimum_reserve_bytes": reserve,
        },
        "execution": {
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "semantic_labels_available": False,
            "calibration_internal_states_opened": False,
            "physical_holdout_opened": False,
        },
        "authorization": {
            "run_internal_discovery_collection": valid,
            "run_calibration": False,
            "run_physical": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
