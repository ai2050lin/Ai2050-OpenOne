#!/usr/bin/env python3
"""Analyze Phase557 behavior and authorize contextual and parametric ledgers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
PROTOCOL_PATH = OUT_DIR / "phase557_frozen_protocol.json"
SUMMARY_PATH = OUT_DIR / "phase557_behavior_summary.json"
QUALIFICATION_PATH = OUT_DIR / "phase557_behavior_qualification.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONTROLLED_SPLITS = (
    "behavior_discovery", "behavior_confirmation", "path_discovery",
    "path_confirmation", "unseen_recombination",
)
BEHAVIOR_GATE_SPLITS = CONTROLLED_SPLITS[:2]
NATURAL_GATE_SPLITS = BEHAVIOR_GATE_SPLITS
EXPECTED_MODEL_ROWS = 8064


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def rate(rows: list[dict[str, Any]], key: str = "semantic_correct") -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def controlled_split_report(rows: list[dict[str, Any]], split: str, protocol: dict[str, Any]) -> dict[str, Any]:
    controlled = [
        row for row in rows
        if row["split"] == split and row["case_type"] == "controlled_factorial"
    ]
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in controlled:
        anchors[row["anchor_id"]].append(row)
    world_all32 = sum(
        len(group) == 32 and all(row["semantic_correct"] for row in group)
        for group in anchors.values()
    )
    query_all16: dict[str, int] = {}
    for query in protocol["query_strata"]:
        query_all16[query] = sum(
            len(query_rows := [row for row in group if row["query_stratum"] == query]) == 16
            and all(row["semantic_correct"] for row in query_rows)
            for group in anchors.values()
        )
    cell_rates = {
        cell: rate([row for row in controlled if row["factorial_cell"] == cell])
        for cell in protocol["factorial_cells"]
    }
    unrecoverable = sum(row["semantic_event"] == "unrecoverable" for row in controlled) / len(controlled)
    gates = protocol["behavior_gate"]
    gate_pass = bool(
        world_all32 / len(anchors) >= gates["world_all_32_rate_min_per_behavior_split"]
        and min(count / len(anchors) for count in query_all16.values())
        >= gates["query_all_16_rate_min_per_behavior_split"]
        and min(cell_rates.values()) >= gates["each_cell_accuracy_min_per_behavior_split"]
        and unrecoverable <= gates["controlled_unrecoverable_rate_max_per_behavior_split"]
    )
    return {
        "controlled_case_count": len(controlled),
        "controlled_anchor_count": len(anchors),
        "world_all_32_correct_count": world_all32,
        "world_all_32_correct_rate": world_all32 / len(anchors),
        "query_all_16_correct_count": query_all16,
        "query_all_16_correct_rate": {
            query: count / len(anchors) for query, count in query_all16.items()
        },
        "cell_accuracy": cell_rates,
        "unrecoverable_rate": unrecoverable,
        "gate_pass": gate_pass,
        "all32_anchor_ids": sorted(
            anchor for anchor, group in anchors.items()
            if len(group) == 32 and all(row["semantic_correct"] for row in group)
        ),
    }


def natural_split_report(rows: list[dict[str, Any]], split: str, protocol: dict[str, Any]) -> dict[str, Any]:
    natural = [
        row for row in rows
        if row["split"] == split and row["case_type"] == "natural_parametric"
    ]
    if not natural:
        return {"case_count": 0, "accuracy": None, "relations": {}}
    relations = {}
    for relation in protocol["natural_relations"]:
        relation_rows = [row for row in natural if row["natural_relation"] == relation]
        surface_rates = {
            str(surface): rate([row for row in relation_rows if row["surface_id"] == surface])
            for surface in range(4)
        }
        fruit_rows = [row for row in relation_rows if row["is_fruit"]]
        relations[relation] = {
            "case_count": len(relation_rows),
            "accuracy": rate(relation_rows),
            "fruits_only_accuracy": rate(fruit_rows),
            "surface_accuracy": surface_rates,
        }
    return {"case_count": len(natural), "accuracy": rate(natural), "relations": relations}


def analyze_model(model: str, protocol: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(OUT_DIR / f"phase557_{model}_behavior_rows.jsonl")
    if len(rows) != EXPECTED_MODEL_ROWS:
        raise RuntimeError(f"Phase557 behavior incomplete for {model}: {len(rows)}")
    if {row.get("torch_dtype") for row in rows} != {"torch.bfloat16"}:
        raise RuntimeError(f"Phase557 behavior dtype drift for {model}")
    if {row.get("quantized_8bit") for row in rows} != {False}:
        raise RuntimeError(f"Phase557 behavior quantization drift for {model}")
    controlled_reports = {
        split: controlled_split_report(rows, split, protocol) for split in CONTROLLED_SPLITS
    }
    contextual_gate = all(controlled_reports[split]["gate_pass"] for split in BEHAVIOR_GATE_SPLITS)
    natural_reports = {
        split: natural_split_report(rows, split, protocol)
        for split in (*NATURAL_GATE_SPLITS, "unseen_recombination")
    }
    gates = protocol["behavior_gate"]
    natural_authorizations = {}
    for relation in protocol["natural_relations"]:
        relation_reports = [natural_reports[split]["relations"][relation] for split in NATURAL_GATE_SPLITS]
        natural_authorizations[relation] = all(
            report["accuracy"] >= gates["natural_relation_accuracy_min_per_behavior_split"]
            and min(report["surface_accuracy"].values()) >= gates["natural_surface_accuracy_min"]
            for report in relation_reports
        )
    return {
        "schema_version": "phase557_behavior_qualification.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "model": model,
        "open_case_count": len(rows),
        "semantic_accuracy": rate(rows),
        "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
        "event_counts": dict(Counter(row["semantic_event"] for row in rows)),
        "controlled_split_reports": controlled_reports,
        "natural_split_reports": natural_reports,
        "contextual_internal_collection_authorized": contextual_gate,
        "natural_relation_authorizations": natural_authorizations,
        "authorized_natural_relations": sorted(
            relation for relation, passed in natural_authorizations.items() if passed
        ),
        "path_discovery_all32_anchor_count": len(
            controlled_reports["path_discovery"]["all32_anchor_ids"]
        ),
        "path_confirmation_all32_anchor_count": len(
            controlled_reports["path_confirmation"]["all32_anchor_ids"]
        ),
        "unseen_all32_anchor_count": len(
            controlled_reports["unseen_recombination"]["all32_anchor_ids"]
        ),
        "torch_dtypes": ["torch.bfloat16"],
        "quantized_8bit": False,
        "sealed_split_read": False,
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    reports = [analyze_model(model, protocol) for model in MODELS]
    write_jsonl(QUALIFICATION_PATH, reports)
    summary = {
        "schema_version": "phase557_behavior_summary.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "model_reports": reports,
        "models_authorized_for_contextual_internal_collection": [
            row["model"] for row in reports if row["contextual_internal_collection_authorized"]
        ],
        "natural_authorizations": {
            row["model"]: row["authorized_natural_relations"] for row in reports
        },
        "open_case_count": sum(row["open_case_count"] for row in reports),
        "registered_case_count_including_unread_sealed": protocol["registered_case_count"],
        "sealed_case_count_unread": protocol["sealed_case_count"],
        "sealed_split_read": False,
        "internal_collection_executed": False,
        "source_recompute_intervention_executed": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "authorized_models": summary["models_authorized_for_contextual_internal_collection"],
        "natural_authorizations": summary["natural_authorizations"],
        "path_anchor_counts": {
            row["model"]: {
                "discovery": row["path_discovery_all32_anchor_count"],
                "confirmation": row["path_confirmation_all32_anchor_count"],
                "unseen": row["unseen_all32_anchor_count"],
            }
            for row in reports
        },
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
