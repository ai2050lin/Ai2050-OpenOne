#!/usr/bin/env python3
"""Freeze cross-model behavior-qualified Phase369 discovery/calibration sets."""

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
PREREG = BASE / "raw_topology_preregister"
OUT = BASE / "behavior_qualification"
MECHANISMS = ("relation_binding", "target_vs_wrong", "entity_recency", "number_agreement")
MINIMUM_GROUPS = {"fresh_discovery": 4, "fresh_calibration": 2}


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


def main() -> None:
    protocol = read_json(PREREG / "phase369_protocol.json")
    if protocol["authorization"]["physical_holdout_execution"]:
        raise RuntimeError("Phase369 physical holdout must still be sealed")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        complete = read_json(OUT / "private/models" / model / "complete.json")
        if not complete["valid"] or complete["physical_holdout_case_count_loaded"] != 0:
            raise RuntimeError(f"Invalid behavior qualification run for {model}")
        rows.extend(read_jsonl(OUT / "private/models" / model / "phase369_behavior_rows.jsonl"))
    if len(rows) != 432:
        raise RuntimeError(f"Expected 432 behavior rows, got {len(rows)}")

    model_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_groups[(row["model"], row["anonymous_parallel_group_id"])].append(row)
    model_group_ok = {
        key: len(group) == 4
        and {row["contrast_condition"] for row in group} == {
            "A_target_lex_x", "B_control_lex_x", "C_target_lex_y", "D_control_lex_y"
        }
        and all(row["strict_behavior_correct"] for row in group)
        for key, group in model_groups.items()
    }
    parallel_meta = {}
    for row in rows:
        parallel_meta[row["anonymous_parallel_group_id"]] = (
            row["phase369_split"], row["family_id"], row["mechanism_id"], row["semantic_group_id"]
        )
    parallel_ok = {
        parallel_id: all(model_group_ok.get((model, parallel_id), False) for model in MODELS)
        for parallel_id in parallel_meta
    }
    qualified = {key for key, value in parallel_ok.items() if value}
    counts = Counter(
        (split, mechanism)
        for parallel_id in qualified
        for split, _family, mechanism, _semantic in [parallel_meta[parallel_id]]
    )
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
    all_gates_passed = all(gate["passed"] for gate in gates)

    blind_registry = read_jsonl(PREREG / "phase369_blind_case_registry.jsonl")
    selected_ids = {
        row["blind_case_id"] for row in rows
        if row["anonymous_parallel_group_id"] in qualified
    }
    selected_blind = [row for row in blind_registry if row["blind_case_id"] in selected_ids]
    by_split = defaultdict(list)
    for row in selected_blind:
        by_split[row["phase369_split"]].append(row)
    write_jsonl(OUT / "phase369_qualified_discovery_blind_cases.jsonl", by_split["fresh_discovery"])
    write_jsonl(OUT / "phase369_qualified_calibration_blind_cases.jsonl", by_split["fresh_calibration"])

    private_selected = [row for row in rows if row["anonymous_parallel_group_id"] in qualified]
    write_jsonl(OUT / "private/phase369_qualified_behavior_rows.jsonl", private_selected)
    write_jsonl(
        OUT / "private/phase369_qualified_parallel_groups.jsonl",
        [
            {
                "anonymous_parallel_group_id": parallel_id,
                "phase369_split": parallel_meta[parallel_id][0],
                "family_id": parallel_meta[parallel_id][1],
                "mechanism_id": parallel_meta[parallel_id][2],
                "semantic_group_id": parallel_meta[parallel_id][3],
                "all_three_models_all_four_conditions_correct": True,
            }
            for parallel_id in sorted(qualified)
        ],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "freeze_cross_model_natural_behavior_denominator_before_raw_internal_collection",
        "input": {
            "model_count": 3,
            "nonphysical_case_count": len(rows),
            "physical_holdout_case_count_loaded": 0,
            "parallel_group_count": len(parallel_meta),
        },
        "qualification": {
            "model_group_count": len(model_groups),
            "qualified_model_group_count": sum(model_group_ok.values()),
            "qualified_parallel_group_count": len(qualified),
            "qualified_case_count": len(private_selected),
            "fresh_discovery_blind_case_count": len(by_split["fresh_discovery"]),
            "fresh_calibration_blind_case_count": len(by_split["fresh_calibration"]),
            "requires_all_four_conditions_and_all_three_models": True,
            "gates": gates,
            "all_gates_passed": all_gates_passed,
        },
        "seals": {
            "semantic_labels_in_public_qualified_case_files": False,
            "physical_holdout_opened": False,
            "phase368_calibration_reused": False,
        },
        "authorization": {
            "fresh_discovery_raw_collection": all_gates_passed,
            "fresh_calibration_raw_collection": False,
            "physical_holdout_execution": False,
        },
        "next_decision": (
            "collect_fresh_discovery_raw_vectors_sequentially_qwen3_glm4_deepseek7b"
            if all_gates_passed else
            "stop_and_generate_an_independent_replacement_case_bank_without_opening_physical_holdout"
        ),
    }
    write_json(OUT / "phase369_behavior_qualification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
