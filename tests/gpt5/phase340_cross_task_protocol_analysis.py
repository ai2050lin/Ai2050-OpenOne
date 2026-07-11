#!/usr/bin/env python3
"""Aggregate Phase340 baseline protocol qualification gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase340_cross_task_protocol"
PHASE = "Phase340"
SCHEMA_VERSION = "16.0.0"
ROUND_DEFAULT = "fresh_cross_task_protocol_repair"
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
    protocol = read_json(root / "phase340_registered_protocol.json")
    registered = read_jsonl(root / "phase340_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    tasks = {row["task_id"]: row["task_class"] for row in protocol["tasks"]}
    task_rows: list[dict[str, Any]] = []
    all_phrase: list[dict[str, Any]] = []
    all_rollout: list[dict[str, Any]] = []
    completions = []
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase340_phrase_rows.jsonl")
        batch_rollout = read_jsonl(model_root / "phase340_rollout_rows.jsonl")
        diagnostic_path = model_root / "phase340_batch_invariance_rows.jsonl"
        diagnostic = read_jsonl(diagnostic_path) if diagnostic_path.exists() else []
        if diagnostic:
            rollout = [
                {
                    **row,
                    "answer_head_semantic_correct": row["batch1_correct"],
                    "answer_head_text": row["batch1_text"],
                }
                for row in diagnostic
            ]
            rollout_source = "single_case_diagnostic"
        else:
            rollout = batch_rollout
            rollout_source = "registered_batch6"
        completions.append(read_json(model_root / "complete.json"))
        all_phrase.extend(phrase)
        all_rollout.extend(batch_rollout)
        for task_id, task_class in tasks.items():
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model,
                "task_id": task_id, "task_class": task_class,
                "qualification_rollout_source": rollout_source,
            }
            split_passes = []
            for split in SPLITS:
                p = [value for value in phrase if value["mechanism_id"] == task_id and value["split"] == split]
                r = [value for value in rollout if value["mechanism_id"] == task_id and value["split"] == split]
                accuracy = sum(value["answer_head_semantic_correct"] for value in r) / len(r)
                valid_rate = sum(value["score_valid"] for value in p) / len(p)
                split_pass = bool(
                    accuracy >= thresholds["split_baseline_accuracy_min"]
                    and valid_rate >= thresholds["split_phrase_score_valid_rate_min"]
                )
                split_passes.append(split_pass)
                row.update({
                    f"{split}_case_count": len(r),
                    f"{split}_baseline_accuracy": round(accuracy, 7),
                    f"{split}_phrase_valid_rate": round(valid_rate, 7),
                    f"{split}_gate_pass": split_pass,
                })
            for template in protocol["templates"]:
                values = [
                    value for value in rollout
                    if value["mechanism_id"] == task_id and value["template_id"] == template
                ]
                row[f"{template}_all_split_accuracy"] = round(
                    sum(value["answer_head_semantic_correct"] for value in values) / len(values), 7
                )
            row["full_protocol_gate_pass"] = all(split_passes)
            row["internal_intervention"] = False
            task_rows.append(row)

    glm = {row["task_id"]: row["full_protocol_gate_pass"] for row in task_rows if row["model"] == "glm4"}
    relation_neighbor_count = sum(
        glm[task] for task in (
            "attribute_relation_binding", "part_relation_binding", "location_relation_binding"
        )
    )
    source_count = sum(glm[task] for task in ("identity_copy", "source_span_extraction"))
    cross_count = sum(glm[task] for task in ("singular_agreement", "direct_entailment", "answer_only_protocol"))
    entry_gate = bool(
        glm["material_relation_binding"]
        and relation_neighbor_count >= thresholds["glm4_relation_neighbor_qualified_min"]
        and source_count >= thresholds["glm4_source_control_qualified_min"]
        and cross_count >= thresholds["glm4_cross_control_qualified_min"]
    )
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase340:{row['model']}:{row['task_id']}",
            "model": row["model"], "family_id": row["task_class"],
            "mechanism_id": row["task_id"],
            "protocol_gate_pass": row["full_protocol_gate_pass"],
            "mapping_status": "qualified_baseline_denominator" if row["full_protocol_gate_pass"] else "baseline_denominator_rejected",
            "internal_intervention": False, "single_unit_causal": False,
        }
        for row in task_rows
    ]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered),
            "phrase_row_count": len(all_phrase), "rollout_row_count": len(all_rollout),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "invalid_phrase_row_count": sum(row["invalid_phrase_row_count"] for row in completions),
            "glm4_batch_invariance_diagnostic": (
                read_json(root / "models/glm4/phase340_batch_invariance_summary.json")
                if (root / "models/glm4/phase340_batch_invariance_summary.json").exists()
                else None
            ),
        },
        "results": {
            "full_protocol_task_gate_count": sum(row["full_protocol_gate_pass"] for row in task_rows),
            "glm4_material_gate_pass": glm["material_relation_binding"],
            "glm4_relation_neighbor_gate_count": relation_neighbor_count,
            "glm4_source_control_gate_count": source_count,
            "glm4_cross_control_gate_count": cross_count,
            "phase341_fresh_causal_boundary_entry_gate_open": entry_gate,
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase340_repaired_protocol_entry",
            "claim": "A fresh GLM4 task matrix is qualified for a new causal boundary audit.",
            "status": "supported" if entry_gate else "not_supported",
            "evidence_level": "L2_baseline_protocol_qualification",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase340_causal_mechanism",
            "claim": "Phase340 identifies an internal causal mechanism.",
            "status": "not_supported", "evidence_level": "baseline_only",
        },
    ]
    write_jsonl(root / "phase340_task_protocol_summary.jsonl", task_rows)
    write_jsonl(root / "phase340_protocol_nodes.jsonl", nodes)
    write_jsonl(root / "phase340_claim_registry.jsonl", claims)
    write_json(root / "phase340_global_summary.json", summary)
    report = [
        "# Phase340 Fresh Cross-Task Protocol Qualification", "",
        f"- Registered fresh cases: {len(registered)}",
        f"- Qualified model-task cells: {summary['results']['full_protocol_task_gate_count']}/27",
        f"- GLM4 material: {glm['material_relation_binding']}",
        f"- GLM4 relation neighbors: {relation_neighbor_count}/3",
        f"- GLM4 source controls: {source_count}/2",
        f"- GLM4 cross-family controls: {cross_count}/3",
        f"- Phase341 entry gate: {entry_gate}", "",
        "GLM4 qualification uses the single-case diagnostic because batch-6 generation failed invariance.",
        "No internal intervention was run; qualification is not causal evidence.",
    ]
    (root / "phase340_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
