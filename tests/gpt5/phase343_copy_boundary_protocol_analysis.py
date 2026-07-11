#!/usr/bin/env python3
"""Aggregate Phase343 task qualification without causal interpretation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase343_copy_boundary_protocol"
PHASE = "Phase343"
SCHEMA_VERSION = "19.0.0"
ROUND_DEFAULT = "copy_boundary_protocol_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "calibration", "heldout", "private_heldout")


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


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase343_registered_protocol.json")
    registered = read_jsonl(root / "phase343_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    tasks = {row["task_id"]: row["task_class"] for row in protocol["tasks"]}
    task_rows = []
    completions = []
    total_phrase = total_rollout = 0
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase343_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase343_rollout_rows.jsonl")
        completions.append(read_json(model_root / "complete.json"))
        total_phrase += len(phrase)
        total_rollout += len(rollout)
        for task_id, task_class in tasks.items():
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model, "task_id": task_id,
                "task_class": task_class, "execution_mode": "b1_left_cache0",
            }
            gates = []
            for split in SPLITS:
                p = [value for value in phrase if value["mechanism_id"] == task_id and value["split"] == split]
                r = [value for value in rollout if value["mechanism_id"] == task_id and value["split"] == split]
                accuracy = sum(value["answer_head_semantic_correct"] for value in r) / len(r)
                valid_rate = sum(value["score_valid"] for value in p) / len(p)
                gate = bool(
                    accuracy >= thresholds["split_baseline_accuracy_min"]
                    and valid_rate >= thresholds["split_phrase_valid_rate_min"]
                )
                gates.append(gate)
                row.update({
                    f"{split}_case_count": len(r),
                    f"{split}_baseline_accuracy": round(accuracy, 7),
                    f"{split}_phrase_valid_rate": round(valid_rate, 7),
                    f"{split}_gate_pass": gate,
                })
            for template in protocol["templates"]:
                values = [value for value in rollout if value["mechanism_id"] == task_id and value["template_id"] == template]
                row[f"{template}_accuracy"] = round(
                    sum(value["answer_head_semantic_correct"] for value in values) / len(values), 7
                )
            row["full_protocol_gate_pass"] = all(gates)
            row["internal_intervention"] = False
            task_rows.append(row)
    glm = [row for row in task_rows if row["model"] == "glm4"]
    class_counts = {
        task_class: sum(row["full_protocol_gate_pass"] for row in glm if row["task_class"] == task_class)
        for task_class in ("explicit_copy", "copy_neighbor", "noncopy_control")
    }
    entry = bool(
        class_counts["explicit_copy"] >= thresholds["glm4_explicit_copy_qualified_min"]
        and class_counts["copy_neighbor"] >= thresholds["glm4_copy_neighbor_qualified_min"]
        and class_counts["noncopy_control"] >= thresholds["glm4_noncopy_control_qualified_min"]
    )
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase343:{row['model']}:{row['task_id']}",
            "model": row["model"], "family_id": row["task_class"],
            "mechanism_id": row["task_id"], "protocol_gate_pass": row["full_protocol_gate_pass"],
            "mapping_status": "qualified_baseline_denominator" if row["full_protocol_gate_pass"] else "baseline_denominator_rejected",
            "internal_intervention": False, "single_unit_causal": False,
        }
        for row in task_rows
    ]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered), "registered_task_count": len(tasks),
            "phrase_row_count": total_phrase, "rollout_row_count": total_rollout,
            "invalid_phrase_row_count": sum(row["invalid_phrase_row_count"] for row in completions),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "execution_mode": "b1_left_cache0",
        },
        "results": {
            "qualified_model_task_count": sum(row["full_protocol_gate_pass"] for row in task_rows),
            "glm4_qualified_counts": class_counts,
            "copy_causal_boundary_entry_gate_open": entry,
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase343_copy_boundary_entry",
            "claim": "The fresh GLM4 matrix has enough qualified copy and noncopy tasks for causal boundary testing.",
            "status": "supported" if entry else "not_supported",
            "evidence_level": "L2_baseline_protocol_qualification",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase343_copy_mechanism", "claim": "Phase343 validates a copy mechanism.",
            "status": "not_supported", "evidence_level": "baseline_only",
        },
    ]
    write_jsonl(root / "phase343_task_protocol_summary.jsonl", task_rows)
    write_jsonl(root / "phase343_protocol_nodes.jsonl", nodes)
    write_jsonl(root / "phase343_claim_registry.jsonl", claims)
    write_json(root / "phase343_global_summary.json", summary)
    report = [
        "# Phase343 Copy-Boundary Protocol Qualification", "",
        f"- Registered cases: {len(registered)}", f"- Qualified model-task cells: {summary['results']['qualified_model_task_count']}/48",
        f"- GLM4 qualified counts: {json.dumps(class_counts, sort_keys=True)}",
        f"- Causal boundary entry: {entry}", "",
        "No internal intervention was run; qualification is not mechanism evidence.",
    ]
    (root / "phase343_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
