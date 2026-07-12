#!/usr/bin/env python3
"""Combine initial and replacement behavior-qualified Phase369 cohorts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase369"
SCHEMA_VERSION = "46.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
INITIAL = BASE / "behavior_qualification"
REPLACEMENT_RUN = INITIAL / "number_agreement_replacement"
REPLACEMENT_BANK = BASE / "raw_topology_preregister_number_agreement_replacement"
OUT = BASE / "behavior_qualification_final"
MINIMUM_GROUPS = {"fresh_discovery": 4, "fresh_calibration": 2}
MECHANISMS = ("relation_binding", "target_vs_wrong", "entity_recency", "number_agreement")


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


def qualify(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, tuple[str, str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    meta = {}
    for row in rows:
        parallel_id = row["anonymous_parallel_group_id"]
        grouped[(row["model"], parallel_id)].append(row)
        meta[parallel_id] = (row["phase369_split"], row["mechanism_id"], row["semantic_group_id"])
    model_ok = {
        key: len(group) == 4
        and all(row["strict_behavior_correct"] for row in group)
        and len({row["contrast_condition"] for row in group}) == 4
        for key, group in grouped.items()
    }
    qualified = {
        parallel_id for parallel_id in meta
        if all(model_ok.get((model, parallel_id), False) for model in MODELS)
    }
    return qualified, meta


def main() -> None:
    initial_rows = []
    replacement_rows = []
    for model in MODELS:
        initial_complete = read_json(INITIAL / "private/models" / model / "complete.json")
        replacement_complete = read_json(REPLACEMENT_RUN / "private/models" / model / "complete.json")
        if not initial_complete["valid"] or not replacement_complete["valid"]:
            raise RuntimeError(f"Incomplete Phase369 qualification for {model}")
        initial_rows.extend(read_jsonl(INITIAL / "private/models" / model / "phase369_behavior_rows.jsonl"))
        replacement_rows.extend(read_jsonl(REPLACEMENT_RUN / "private/models" / model / "phase369_behavior_rows.jsonl"))
    initial_qualified, initial_meta = qualify(initial_rows)
    replacement_qualified, replacement_meta = qualify(replacement_rows)
    initial_qualified = {
        key for key in initial_qualified if initial_meta[key][1] != "number_agreement"
    }
    replacement_qualified = {
        key for key in replacement_qualified if replacement_meta[key][1] == "number_agreement"
    }
    meta = {**initial_meta, **replacement_meta}
    qualified = initial_qualified | replacement_qualified
    counts = Counter((meta[key][0], meta[key][1]) for key in qualified)
    gates = [
        {
            "split": split,
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": counts[(split, mechanism)],
            "minimum_required": MINIMUM_GROUPS[split],
            "passed": counts[(split, mechanism)] >= MINIMUM_GROUPS[split],
        }
        for split in ("fresh_discovery", "fresh_calibration")
        for mechanism in MECHANISMS
    ]
    all_passed = all(gate["passed"] for gate in gates)

    all_rows = initial_rows + replacement_rows
    selected_rows = [row for row in all_rows if row["anonymous_parallel_group_id"] in qualified]
    selected_ids = {row["blind_case_id"] for row in selected_rows}
    blind_rows = read_jsonl(BASE / "raw_topology_preregister/phase369_blind_case_registry.jsonl")
    blind_rows.extend(read_jsonl(REPLACEMENT_BANK / "phase369_number_agreement_blind_cases.jsonl"))
    selected_blind = [row for row in blind_rows if row["blind_case_id"] in selected_ids]
    discovery = [row for row in selected_blind if row["phase369_split"] == "fresh_discovery"]
    calibration = [row for row in selected_blind if row["phase369_split"] == "fresh_calibration"]
    write_jsonl(OUT / "phase369_qualified_discovery_blind_cases.jsonl", discovery)
    write_jsonl(OUT / "phase369_qualified_calibration_blind_cases.jsonl", calibration)
    write_jsonl(OUT / "private/phase369_qualified_behavior_rows.jsonl", selected_rows)
    write_jsonl(
        OUT / "private/phase369_qualified_parallel_groups.jsonl",
        [
            {
                "anonymous_parallel_group_id": key,
                "phase369_split": meta[key][0],
                "mechanism_id": meta[key][1],
                "semantic_group_id": meta[key][2],
                "source_contract": (
                    "number_agreement_replacement" if key in replacement_qualified else "initial"
                ),
                "all_three_models_all_four_conditions_correct": True,
            }
            for key in sorted(qualified)
        ],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "freeze_final_cross_model_natural_behavior_denominator",
        "input": {
            "initial_nonphysical_case_count": len(initial_rows),
            "replacement_nonphysical_case_count": len(replacement_rows),
            "physical_holdout_case_count_loaded": 0,
        },
        "qualification": {
            "qualified_parallel_group_count": len(qualified),
            "qualified_case_count": len(selected_rows),
            "fresh_discovery_blind_case_count": len(discovery),
            "fresh_calibration_blind_case_count": len(calibration),
            "initial_number_agreement_groups_retired": True,
            "gates": gates,
            "all_gates_passed": all_passed,
        },
        "seals": {
            "semantic_labels_in_public_qualified_case_files": False,
            "physical_holdout_opened": False,
            "internal_trace_used_to_repair_case_contract": False,
        },
        "authorization": {
            "fresh_discovery_raw_collection": all_passed,
            "fresh_calibration_raw_collection": False,
            "physical_holdout_execution": False,
        },
        "next_decision": (
            "collect_fresh_discovery_raw_vectors_sequentially_qwen3_glm4_deepseek7b"
            if all_passed else "stop_before_internal_collection"
        ),
    }
    write_json(OUT / "phase369_behavior_qualification_final_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
