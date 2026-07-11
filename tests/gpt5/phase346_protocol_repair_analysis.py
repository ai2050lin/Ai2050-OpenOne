#!/usr/bin/env python3
"""Aggregate Phase346 repair gates and merge Phase345 trace eligibility."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase346_protocol_repair"
PHASE345 = ROOT / "tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification"
PHASE = "Phase346"
SCHEMA_VERSION = "22.0.0"
ROUND_DEFAULT = "three_core_protocol_repair"
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
    protocol = read_json(root / "phase346_registered_protocol.json")
    thresholds = protocol["thresholds"]
    rows = []
    completions = []
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase346_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase346_rollout_rows.jsonl")
        completions.append(read_json(model_root / "complete.json"))
        for task_id in protocol["tasks"]:
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model, "task_id": task_id,
                "task_class": "protocol_control", "execution_mode": "b1_left_cache0",
            }
            gates = []
            for split in SPLITS:
                p = [value for value in phrase if value["mechanism_id"] == task_id and value["split"] == split]
                r = [value for value in rollout if value["mechanism_id"] == task_id and value["split"] == split]
                accuracy = sum(value["answer_head_semantic_correct"] for value in r) / len(r)
                valid_rate = sum(value["score_valid"] for value in p) / len(p)
                gate = bool(accuracy >= thresholds["split_baseline_accuracy_min"] and valid_rate >= thresholds["split_phrase_valid_rate_min"])
                gates.append(gate)
                row.update({
                    f"{split}_baseline_accuracy": round(accuracy, 7),
                    f"{split}_phrase_valid_rate": round(valid_rate, 7),
                    f"{split}_gate_pass": gate,
                })
            row["full_protocol_gate_pass"] = all(gates)
            row["internal_intervention"] = False
            rows.append(row)
    phase345 = read_json(PHASE345 / "phase345_global_summary.json")
    core_counts = phase345["results"]["qualified_family_counts_by_model"]
    repaired_counts = {
        model: sum(row["full_protocol_gate_pass"] for row in rows if row["model"] == model)
        for model in MODELS
    }
    entry = {
        model: bool(
            core_counts[model]["knowledge_network"] >= 2
            and core_counts[model]["reasoning"] >= 2
            and core_counts[model]["grammar"] >= 2
            and core_counts[model]["protocol_control"] + repaired_counts[model] >= 2
            and repaired_counts[model] >= protocol["trace_entry_requires_repaired_task_pass_min"]
        )
        for model in MODELS
    }
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase346:{row['model']}:{row['task_id']}",
            "model": row["model"], "family_id": "protocol_control",
            "mechanism_id": row["task_id"], "protocol_gate_pass": row["full_protocol_gate_pass"],
            "mapping_status": "qualified_repair_denominator" if row["full_protocol_gate_pass"] else "repair_denominator_rejected",
            "internal_intervention": False, "single_unit_causal": False,
        }
        for row in rows
    ]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": protocol["registered_case_count"],
            "phrase_row_count": sum(row["phrase_row_count"] for row in completions),
            "rollout_row_count": sum(row["rollout_row_count"] for row in completions),
            "invalid_phrase_row_count": sum(row["invalid_phrase_row_count"] for row in completions),
            "all_model_completions_valid": all(row["valid"] for row in completions),
        },
        "results": {
            "repaired_task_pass_count_by_model": repaired_counts,
            "physical_trace_entry_by_model": entry,
            "cross_model_physical_trace_entry_gate_open": all(entry.values()),
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase346_task_protocol_summary.jsonl", rows)
    write_jsonl(root / "phase346_protocol_nodes.jsonl", nodes)
    write_json(root / "phase346_global_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
