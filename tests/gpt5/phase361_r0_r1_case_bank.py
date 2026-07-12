#!/usr/bin/env python3
"""Freeze balanced anonymous R0/R1 cases for the four admitted mechanisms."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace"
ROUND = "four_admitted_balanced_trace"
PHASE361 = ROOT / "tests/gpt5/result/phase361_contract_repair/seven_contract_repair"
PHASE354 = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
MODELS = ("qwen3", "glm4", "deepseek7b")
SCHEMA_VERSION = "38.0.0"
MLP_SHARD_COUNT = 16


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def choose_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["contrast_condition"][0])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: digest(f"phase361-r0-r1:{row['case_id']}"))
    selected = []
    for condition in ("A", "B"):
        selected.extend(groups[("physical_discovery", condition)][:2])
    for condition in ("C", "D"):
        selected.extend(groups[("physical_discovery", condition)][:1])
        selected.extend(groups[("physical_calibration", condition)][:1])
    if len(selected) != 8:
        raise RuntimeError(f"Balanced cell has {len(selected)} cases, expected 8")
    return selected


def main() -> None:
    behavior = read_json(PHASE361 / "phase361_behavior_summary.json")
    admitted = {(row["family_id"], row["mechanism_id"]) for row in behavior["admitted_mechanisms"]}
    original = read_jsonl(PHASE354 / "phase354_registered_cases.jsonl")
    repaired = [
        row for row in read_jsonl(PHASE361 / "phase361_registered_cases.jsonl")
        if row["mechanism_id"] == "number_agreement"
    ]
    candidates = [row for row in [*original, *repaired] if (row["family_id"], row["mechanism_id"]) in admitted]
    selected: list[dict[str, Any]] = []
    for model in MODELS:
        for family, mechanism in sorted(admitted):
            cell = [
                row for row in candidates
                if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism
            ]
            selected.extend(choose_balanced(cell))
    if len(selected) != 96:
        raise RuntimeError(f"Invalid R0/R1 denominator: {len(selected)}")

    execution_rows, blind_rows, private_rows = [], [], []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_model[row["model"]].append(row)
    for model in MODELS:
        ordered = sorted(by_model[model], key=lambda row: digest(f"phase361-shard:{row['case_id']}"))
        for index, row in enumerate(ordered):
            blind_case_id = f"blind_{digest('phase361-blind:' + row['case_id'])[:24]}"
            anonymous_model_id = f"am_{digest('phase361-model:' + model)[:12]}"
            shard = index % MLP_SHARD_COUNT
            execution_rows.append({
                **row,
                "blind_case_id": blind_case_id,
                "anonymous_model_id": anonymous_model_id,
                "r1_mlp_shard_index": shard,
                "r1_assignment_rule": "model_local_hash_order_round_robin_16",
            })
            blind_rows.append({
                "schema_version": SCHEMA_VERSION, "phase_id": "Phase361", "created_at": now(),
                "blind_case_id": blind_case_id, "anonymous_model_id": anonymous_model_id,
                "split": "blind_discovery" if row["split"] == "physical_discovery" else "blind_calibration",
                "contrast_bucket": row["contrast_condition"][0],
                "r1_mlp_shard_index": shard,
                "semantic_label_used_for_component_selection": False,
            })
            private_rows.append({
                "blind_case_id": blind_case_id, "model": model,
                "family_id": row["family_id"], "mechanism_id": row["mechanism_id"],
                "source_case_id": row["case_id"], "split": row["split"],
                "contrast_condition": row["contrast_condition"],
            })
    root = OUT / ROUND
    write_jsonl(root / "private" / "phase361_execution_cases.jsonl", execution_rows)
    write_jsonl(root / "private" / "phase361_label_key.jsonl", private_rows)
    write_jsonl(root / "phase361_blind_case_registry.jsonl", blind_rows)
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": "Phase361", "created_at": now(),
        "denominator": {
            "model_count": 3, "admitted_mechanism_count": len(admitted),
            "cases_per_model_mechanism": 8, "case_count": len(blind_rows),
            "blind_discovery_case_count": sum(row["split"] == "blind_discovery" for row in blind_rows),
            "blind_calibration_case_count": sum(row["split"] == "blind_calibration" for row in blind_rows),
            "condition_counts": {condition: sum(row["contrast_bucket"] == condition for row in blind_rows) for condition in "ABCD"},
            "r1_shard_counts": {str(shard): sum(row["r1_mlp_shard_index"] == shard for row in blind_rows) for shard in range(MLP_SHARD_COUNT)},
        },
        "quality": {
            "labels_sealed": True, "model_names_absent_from_blind_registry": True,
            "semantic_label_used_for_component_selection": False,
            "physical_heldout_included": False, "causal_sealed_included": False,
        },
        "entry_decision": "run_r0_r1_component_trace",
    }
    write_json(root / "phase361_r0_r1_case_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
