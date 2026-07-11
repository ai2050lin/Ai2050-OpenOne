#!/usr/bin/env python3
"""Aggregate Phase337 protocol qualification without creating mechanism claims."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase337_protocol_qualification"
PHASE = "Phase337"
SCHEMA_VERSION = "13.0.0"
ROUND_DEFAULT = "material_relation_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("raw_completion", "native_chat", "answer_aligned_chat")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def target_in_text(text: str, target: str) -> bool:
    value = normalized(target)
    return bool(value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized(text)))


def qualify_row(row: dict[str, Any]) -> dict[str, Any]:
    answer_head = next(
        (line.strip() for line in row.get("answer_text", "").splitlines() if line.strip()), ""
    )
    head_correct = bool(row["answer_reached"] and target_in_text(answer_head, row["target"]))
    return {
        **row,
        "answer_head_text": answer_head,
        "answer_head_semantic_correct": head_correct,
        "baseline_capability": bool(
            row["answer_reached"] and head_correct and row["target_phrase_valid"]
        ),
    }


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase337_registered_protocol.json")
    registered = read_jsonl(root / "phase337_registered_cases.jsonl")
    rows: list[dict[str, Any]] = []
    completes = []
    for model in MODELS:
        model_root = root / "models" / model
        rows.extend(
            qualify_row(row)
            for row in read_jsonl(model_root / "phase337_qualification_rows.jsonl")
        )
        completes.append(read_json(model_root / "complete.json"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["interface"])].append(row)

    cell_rows = []
    for (model, interface), values in sorted(grouped.items()):
        n = len(values)
        answer_reached_count = sum(row["answer_reached"] for row in values)
        semantic_correct_count = sum(row["semantic_correct"] for row in values)
        answer_semantic_correct_count = sum(row["answer_semantic_correct"] for row in values)
        phrase_valid_count = sum(row["target_phrase_valid"] for row in values)
        capable_count = sum(row["baseline_capability"] for row in values)
        protocol_count = sum(row["protocol_followed"] for row in values)
        exhausted_count = sum(row["token_budget_exhausted"] for row in values)
        capable_cell = bool(
            capable_count >= int(protocol["cell_capable_case_min"])
            and answer_reached_count >= int(protocol["cell_answer_reached_min"])
        )
        cell_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "interface": interface,
            "registered_case_count": n,
            "answer_reached_count": answer_reached_count,
            "semantic_correct_count": semantic_correct_count,
            "answer_semantic_correct_count": answer_semantic_correct_count,
            "answer_head_semantic_correct_count": sum(
                row["answer_head_semantic_correct"] for row in values
            ),
            "semantic_correct_outside_answer_count": sum(
                row["semantic_correct_outside_answer"] for row in values
            ),
            "target_phrase_valid_count": phrase_valid_count,
            "baseline_capable_count": capable_count,
            "protocol_followed_count": protocol_count,
            "token_budget_exhausted_count": exhausted_count,
            "mean_initial_target_rank": round(mean(row["initial_target_rank"] for row in values), 7),
            "mean_target_phrase_margin": round(mean(row["target_phrase_margin"] for row in values), 7),
            "capable_cell": capable_cell,
            "evidence_level": "L2_protocol_qualification_cell",
            "mechanism_causal": False,
        })

    interface_rows = []
    for interface in INTERFACES:
        cells = [row for row in cell_rows if row["interface"] == interface]
        capable_models = [row["model"] for row in cells if row["capable_cell"]]
        interface_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "interface": interface,
            "capable_model_count": len(capable_models),
            "capable_models": capable_models,
            "minimum_capable_cases_across_models": min(row["baseline_capable_count"] for row in cells),
            "minimum_answer_reached_across_models": min(row["answer_reached_count"] for row in cells),
            "stage_gate_pass": len(capable_models) >= int(protocol["stage_gate_common_interface_model_min"]),
        })
    passing_interfaces = [row["interface"] for row in interface_rows if row["stage_gate_pass"]]
    preferred = None
    if passing_interfaces:
        preferred = max(
            (row for row in interface_rows if row["stage_gate_pass"]),
            key=lambda row: (
                row["capable_model_count"], row["minimum_capable_cases_across_models"],
                row["minimum_answer_reached_across_models"],
            ),
        )["interface"]

    invalid_metric_count = sum(
        not isinstance(row["target_phrase_margin"], (int, float))
        or not isinstance(row["initial_target_rank"], int)
        for row in rows
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered),
            "executed_case_count": len(rows),
            "model_count": len(MODELS),
            "interface_count": len(INTERFACES),
            "objects_per_model_interface": 12,
            "all_models_complete": all(row["valid"] for row in completes),
            "invalid_metric_count": invalid_metric_count,
        },
        "results": {
            "capable_cell_count": sum(row["capable_cell"] for row in cell_rows),
            "passing_interface_count": len(passing_interfaces),
            "passing_interfaces": passing_interfaces,
            "preferred_interface_for_next_stage": preferred,
            "protocol_qualification_gate_pass": bool(passing_interfaces),
            "mechanism_causal_count": 0,
            "single_unit_causal_count": 0,
        },
        "claim_boundary": (
            "A pass only qualifies a baseline denominator for later block-level causal tests."
        ),
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase337_cell_summary.jsonl", cell_rows)
    write_jsonl(root / "phase337_interface_gate_summary.jsonl", interface_rows)
    write_jsonl(root / "phase337_qualified_rows.jsonl", rows)
    write_json(root / "phase337_global_summary.json", summary)
    report = [
        "# Phase337 Protocol Qualification",
        "",
        f"- Registered/executed: {len(registered)}/{len(rows)}",
        f"- Capable model-interface cells: {summary['results']['capable_cell_count']}/9",
        f"- Passing interfaces: {', '.join(passing_interfaces) if passing_interfaces else 'none'}",
        f"- Preferred next-stage interface: {preferred or 'none'}",
        f"- Protocol qualification gate: {summary['results']['protocol_qualification_gate_pass']}",
        "- Mechanism causal claims: 0",
        "- Single-unit causal claims: 0",
        "",
        "## Cell Results",
        "",
    ]
    for row in cell_rows:
        report.append(
            f"- {row['model']} / {row['interface']}: capable "
            f"{row['baseline_capable_count']}/12, answer reached {row['answer_reached_count']}/12, "
            f"answer-head correct {row['answer_head_semantic_correct_count']}/12, "
            f"semantic anywhere {row['semantic_correct_count']}/12, phrase valid "
            f"{row['target_phrase_valid_count']}/12, gate {row['capable_cell']}"
        )
    report.extend([
        "",
        "This phase measures protocol eligibility only. It does not capture activations, intervene on the model, "
        "or establish a language mechanism.",
    ])
    (root / "phase337_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
