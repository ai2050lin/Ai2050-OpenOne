#!/usr/bin/env python3
"""Analyze Phase556 behavior and authorize internal collection per model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
PROTOCOL_PATH = OUT_DIR / "phase556_frozen_protocol.json"
QUALIFICATION_PATH = OUT_DIR / "phase556_behavior_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase556_behavior_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = ("discovery", "independent_confirmation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rate(rows: list[dict[str, Any]], key: str = "semantic_correct") -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def analyze_model(model: str, protocol: dict[str, Any]) -> dict[str, Any]:
    rows_path = OUT_DIR / f"phase556_{model}_behavior_rows.jsonl"
    rows = read_jsonl(rows_path)
    if len(rows) != 3872:
        raise RuntimeError(f"Phase556 behavior incomplete for {model}: {len(rows)}")
    torch_dtypes = sorted({str(row.get("torch_dtype", "missing")) for row in rows})
    quantized_8bit_values = sorted({bool(row.get("quantized_8bit", False)) for row in rows})
    if torch_dtypes != ["torch.bfloat16"] or quantized_8bit_values != [False]:
        raise RuntimeError(
            f"Phase556 behavior execution drift for {model}: "
            f"dtype={torch_dtypes}, quantized_8bit={quantized_8bit_values}"
        )
    gates = protocol["behavior_gate"]
    split_reports: dict[str, Any] = {}
    controlled_gate = True
    relation_authorizations: dict[str, bool] = {}
    for split in OPEN_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        controlled = [row for row in split_rows if row["case_type"] == "controlled_factorial"]
        natural = [row for row in split_rows if row["case_type"] == "natural_knowledge"]
        anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in controlled:
            anchors[row["anchor_id"]].append(row)
        all16_count = sum(len(group) == 16 and all(row["semantic_correct"] for row in group) for group in anchors.values())
        all16_rate = all16_count / len(anchors)
        cell_rates = {
            cell: rate([row for row in controlled if row["factorial_cell"] == cell])
            for cell in protocol["factorial_cells"]
        }
        unrecoverable_rate = sum(row["semantic_event"] == "unrecoverable" for row in controlled) / len(controlled)
        split_controlled_pass = bool(
            all16_rate >= gates["controlled_all_16_anchor_rate_min_per_open_split"]
            and min(cell_rates.values()) >= gates["controlled_each_cell_accuracy_min_per_open_split"]
            and unrecoverable_rate <= gates["controlled_unrecoverable_rate_max_per_open_split"]
        )
        controlled_gate = controlled_gate and split_controlled_pass
        natural_relation_rates = {
            relation: rate([row for row in natural if row["natural_relation"] == relation])
            for relation in protocol["natural_relations"]
        }
        natural_surface_rates = {
            str(surface): rate([row for row in natural if row["surface_id"] == surface])
            for surface in range(4)
        }
        for relation, relation_rate in natural_relation_rates.items():
            relation_surface_min = min(
                rate([
                    row for row in natural
                    if row["natural_relation"] == relation and row["surface_id"] == surface
                ]) for surface in range(4)
            )
            passed = bool(
                relation_rate >= gates["natural_relation_accuracy_min_per_open_split"]
                and relation_surface_min >= gates["natural_surface_accuracy_min"]
            )
            relation_authorizations[relation] = relation_authorizations.get(relation, True) and passed
        split_reports[split] = {
            "controlled_case_count": len(controlled),
            "controlled_anchor_count": len(anchors),
            "controlled_all_16_correct_count": all16_count,
            "controlled_all_16_correct_rate": all16_rate,
            "controlled_cell_accuracy": cell_rates,
            "controlled_unrecoverable_rate": unrecoverable_rate,
            "controlled_gate_pass": split_controlled_pass,
            "natural_case_count": len(natural),
            "natural_accuracy": rate(natural),
            "natural_relation_accuracy": natural_relation_rates,
            "natural_surface_accuracy": natural_surface_rates,
        }
    return {
        "schema_version": "phase556_behavior_qualification.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "model": model,
        "torch_dtypes": torch_dtypes,
        "quantized_8bit": False,
        "open_case_count": len(rows),
        "semantic_accuracy": rate(rows),
        "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
        "event_counts": dict(Counter(row["semantic_event"] for row in rows)),
        "split_reports": split_reports,
        "controlled_factorial_gate_pass": controlled_gate,
        "natural_relation_authorizations": relation_authorizations,
        "authorized_natural_relations": sorted(key for key, value in relation_authorizations.items() if value),
        "internal_collection_authorized": controlled_gate,
        "sealed_split_read": False,
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    reports = [analyze_model(model, protocol) for model in MODELS]
    write_jsonl(QUALIFICATION_PATH, reports)
    summary = {
        "schema_version": "phase556_behavior_summary.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "model_reports": reports,
        "models_authorized_for_internal_collection": [
            row["model"] for row in reports if row["internal_collection_authorized"]
        ],
        "open_case_count": sum(row["open_case_count"] for row in reports),
        "registered_case_count_including_unread_sealed": protocol["registered_case_count"],
        "sealed_case_count_unread": protocol["sealed_case_count"],
        "sealed_split_read": False,
        "internal_collection_executed": False,
        "causal_intervention_executed": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
