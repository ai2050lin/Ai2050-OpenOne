#!/usr/bin/env python3
"""Aggregate Phase345 protocol qualification and physical-trace entry gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase345_three_core_protocol"
PHASE = "Phase345"
SCHEMA_VERSION = "21.0.0"
ROUND_DEFAULT = "three_core_protocol_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "calibration", "heldout", "private_heldout")
FAMILIES = ("knowledge_network", "reasoning", "grammar", "protocol_control")


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
    protocol = read_json(root / "phase345_registered_protocol.json")
    registered = read_jsonl(root / "phase345_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    tasks = {row["task_id"]: row["task_class"] for row in protocol["tasks"]}
    task_rows = []
    completions = []
    total_phrase = total_rollout = 0
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase345_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase345_rollout_rows.jsonl")
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
            row["full_protocol_gate_pass"] = all(gates)
            row["internal_intervention"] = False
            task_rows.append(row)
    family_counts_by_model = {
        model: {
            family: sum(
                row["full_protocol_gate_pass"] for row in task_rows
                if row["model"] == model and row["task_class"] == family
            )
            for family in FAMILIES
        }
        for model in MODELS
    }
    entry_by_model = {
        model: bool(
            counts["knowledge_network"] >= thresholds["physical_trace_family_qualified_min"]
            and counts["reasoning"] >= thresholds["physical_trace_family_qualified_min"]
            and counts["grammar"] >= thresholds["physical_trace_family_qualified_min"]
            and counts["protocol_control"] >= thresholds["physical_trace_protocol_qualified_min"]
        )
        for model, counts in family_counts_by_model.items()
    }
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase345:{row['model']}:{row['task_id']}",
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
            "qualified_family_counts_by_model": family_counts_by_model,
            "physical_trace_entry_by_model": entry_by_model,
            "cross_model_physical_trace_entry_gate_open": all(entry_by_model.values()),
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase345_three_core_trace_entry",
            "claim": "All three models have enough qualified tasks in each core family for natural physical tracing.",
            "status": "supported" if summary["results"]["cross_model_physical_trace_entry_gate_open"] else "not_supported",
            "evidence_level": "L2_baseline_protocol_qualification",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase345_mcue", "claim": "Phase345 validates MCUE or an effective neuron set.",
            "status": "not_supported", "evidence_level": "baseline_only",
        },
    ]
    write_jsonl(root / "phase345_task_protocol_summary.jsonl", task_rows)
    write_jsonl(root / "phase345_protocol_nodes.jsonl", nodes)
    write_jsonl(root / "phase345_claim_registry.jsonl", claims)
    write_json(root / "phase345_global_summary.json", summary)
    report = [
        "# Phase345 Three-Core Protocol Qualification", "",
        f"- Registered cases: {len(registered)}", f"- Qualified cells: {summary['results']['qualified_model_task_count']}/36", "",
    ]
    for model in MODELS:
        report.append(f"- {model}: {json.dumps(family_counts_by_model[model], sort_keys=True)}, trace_entry={entry_by_model[model]}")
    report.extend(["", "No causal intervention, MCUE search, or neuron claim was executed."])
    (root / "phase345_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
