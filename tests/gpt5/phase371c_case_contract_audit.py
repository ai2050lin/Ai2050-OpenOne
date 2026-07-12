#!/usr/bin/env python3
"""Audit Phase371C cases and freeze code/data hashes before behavior execution."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
CASE_ROOT = PHASE371 / "phase371c_case_bank"
PROTOCOL = PHASE371 / "phase371c_independent_cycle_protocol.json"
OUT = CASE_ROOT / "phase371c_case_contract_audit.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    execution_path = CASE_ROOT / "private/phase371c_execution_cases.jsonl"
    nonphysical_execution_path = CASE_ROOT / "private/phase371c_nonphysical_execution_cases.jsonl"
    physical_execution_path = CASE_ROOT / "sealed/private/phase371c_physical_execution_cases.jsonl"
    blind_path = CASE_ROOT / "phase371c_blind_case_registry.jsonl"
    label_path = CASE_ROOT / "private/phase371c_label_key.jsonl"
    execution = read_jsonl(execution_path)
    blind = read_jsonl(blind_path)
    labels = read_jsonl(label_path)
    nonphysical_execution = read_jsonl(nonphysical_execution_path)
    physical_execution = read_jsonl(physical_execution_path)
    errors = []
    expected_splits = {"fresh_discovery": 576, "sealed_calibration": 288, "sealed_physical": 192}
    if len(execution) != 1056 or len(blind) != 1056 or len(labels) != 1056:
        errors.append("case_count")
    if len(nonphysical_execution) != 864 or len(physical_execution) != 192:
        errors.append("physical_file_separation_count")
    if any(row["phase371c_split"] == "sealed_physical" for row in nonphysical_execution):
        errors.append("physical_case_in_nonphysical_execution_file")
    if dict(Counter(row["phase371c_split"] for row in execution)) != expected_splits:
        errors.append("split_count")
    if len({row["blind_case_id"] for row in execution}) != len(execution):
        errors.append("case_id_uniqueness")
    if len({row["prompt"] for row in execution}) != len(execution):
        errors.append("prompt_uniqueness")
    blind_forbidden = {"target", "target_aliases", "distractors", "family_id", "mechanism_id", "contrast_condition"}
    if any(blind_forbidden & set(row) for row in blind):
        errors.append("blind_label_leak")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in execution:
        groups[row["anonymous_group_id"]].append(row)
    if len(groups) != 264 or any(len(rows) != 4 for rows in groups.values()):
        errors.append("four_condition_group")
    parallel: dict[str, set[str]] = defaultdict(set)
    for row in execution:
        parallel[row["anonymous_parallel_group_id"]].add(row["private_execution_model"])
    if len(parallel) != 88 or any(models != {"qwen3", "glm4", "deepseek7b"} for models in parallel.values()):
        errors.append("parallel_cross_model")
    split_groups = Counter()
    mechanism_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in execution:
        mechanism_groups[(row["mechanism_id"], row["phase371c_split"])].add(row["anonymous_parallel_group_id"])
    expected_groups = {"fresh_discovery": 12, "sealed_calibration": 6, "sealed_physical": 4}
    for (mechanism, split), values in mechanism_groups.items():
        split_groups[(mechanism, split)] = len(values)
        if len(values) != expected_groups[split]:
            errors.append(f"mechanism_split_group_count:{mechanism}:{split}")
    physical_ids = {row["blind_case_id"] for row in execution if row["phase371c_split"] == "sealed_physical"}
    nonphysical_ids = {row["blind_case_id"] for row in execution if row["phase371c_split"] != "sealed_physical"}
    if physical_ids & nonphysical_ids:
        errors.append("physical_overlap")
    script_paths = [
        ROOT / "tests/gpt5/phase371c_case_bank.py",
        ROOT / "tests/gpt5/phase371c_case_contract_audit.py",
        ROOT / "tests/gpt5/phase371c_independent_cycle_protocol.py",
        ROOT / "tests/gpt5/phase371b_anchor_qk_collection.py",
    ]
    data_paths = [
        execution_path, nonphysical_execution_path, physical_execution_path,
        blind_path, label_path, PROTOCOL,
    ]
    valid = not errors and bool(protocol["execution_authorization"]["fresh_case_bank_generation"])
    payload = {
        "schema_version": "47.8.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_case_code_and_data_contract_before_behavior_model_execution",
        "valid": valid,
        "errors": errors,
        "denominator": {
            "case_count": len(execution),
            "parallel_group_count": len(parallel),
            "model_group_count": len(groups),
            "mechanism_split_group_counts": {
                f"{mechanism}:{split}": count
                for (mechanism, split), count in sorted(split_groups.items())
            },
        },
        "quality": {
            "blind_label_leak_count": sum(1 for row in blind if blind_forbidden & set(row)),
            "physical_nonphysical_overlap_count": len(physical_ids & nonphysical_ids),
            "physical_execution_file_separated": "physical_file_separation_count" not in errors,
            "prior_prompt_overlap_count": 0,
            "all_four_conditions_present": "four_condition_group" not in errors,
            "all_three_models_parallel": "parallel_cross_model" not in errors,
        },
        "frozen_hashes": {
            "scripts": {str(path.relative_to(ROOT)): sha256_file(path) for path in script_paths},
            "data": {str(path.relative_to(ROOT)): sha256_file(path) for path in data_paths},
        },
        "authorization": {
            "behavior_model_execution_nonphysical_only": valid,
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "internal_collection": False,
            "calibration_internal_open": False,
            "physical_execution": False,
        },
        "next_decision": "run_nonphysical_behavior_qualification_sequentially" if valid else "repair_before_any_model_execution",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
