#!/usr/bin/env python3
"""Freeze the adjacent-layer extension required by the same-graph replay contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
AUDIT = PHASE371 / "phase371c_internal_collection_audit.json"
CASES = PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
OUT = PHASE371 / "phase371c_adjacent_extension_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    current = int(audit["storage"]["total_byte_count"])
    estimated_extension = current
    budget = int(audit["storage"]["budget_bytes"])
    free = int(shutil.disk_usage(ROOT).free)
    payload = {
        "schema_version": "47.13.0",
        "phase_id": "Phase371C-Adj",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "add_only_the_three_missing_neighbor_layers_needed_for_local_next_layer_replay",
        "trigger": {
            "base_ledger_valid": audit["valid"],
            "three_anchor_local_block_replay_available": True,
            "adjacent_next_layer_state_available": False,
            "semantic_candidate_search_started": False,
            "calibration_opened": False,
        },
        "layer_pairs": {
            "early": [0, 1],
            "middle": ["floor_half", "floor_half_plus_one"],
            "late": ["last_minus_one", "last"],
            "new_layers_collected_per_model": 3,
        },
        "denominator": {
            "case_count": 264,
            "case_count_per_model": 88,
            "generation_time_count": 3,
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "collector_case_hash": sha256_file(CASES),
            "collector_code_hash": sha256_file(ROOT / "tests/gpt5/phase371c_adjacent_collection.py"),
            "exact_tree_helper_hash": sha256_file(ROOT / "tests/gpt5/phase371b_anchor_qk_collection.py"),
        },
        "storage": {
            "current_base_bytes": current,
            "estimated_extension_bytes": estimated_extension,
            "estimated_combined_bytes": current + estimated_extension,
            "budget_bytes": budget,
            "free_disk_bytes": free,
            "minimum_reserve_bytes": 200 * 1024**3,
        },
        "gates": {
            "greedy_generation_tokens_must_match_base_ledger": True,
            "all_existing_numeric_gates_unchanged": True,
            "lossless_sufficient_state_only": True,
            "all_files_hash_and_shape_audited_before_path_extraction": True,
        },
        "claim_boundary": {
            "measurement_contract_repair_only": True,
            "language_path_claim": False,
            "calibration_or_physical_open": False,
        },
        "authorization": {
            "run_adjacent_collection": (
                audit["valid"]
                and current + estimated_extension <= budget
                and free - estimated_extension >= 200 * 1024**3
            ),
            "run_path_extraction_before_extension_audit": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
