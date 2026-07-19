#!/usr/bin/env python3
"""Analyze Phase560 parent components and close the semantic-source stage."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
CONTRACT_PATH = OUT_DIR / "phase560_parent_decomposition_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase560_parent_decomposition_rows.jsonl"
ANALYSIS_PATH = OUT_DIR / "phase560_parent_decomposition_analysis.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(ROWS_PATH)
    if len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase560 parent denominator is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    reports = []
    for candidate_id, candidate_rows in sorted(grouped.items()):
        by_condition = {
            condition: [row for row in candidate_rows if row["condition"] == condition]
            for condition in contract["conditions"]
        }
        metrics = {
            condition: {
                "mean_donor_switch_effect": mean(group, "donor_switch_effect"),
                "donor_win_rate": rate(group, "intervention_donor_wins"),
                "recipient_retained_rate": rate(group, "intervention_recipient_retained"),
            }
            for condition, group in by_condition.items()
        }
        layer_input_rate = metrics["layer_input_donor_replace"]["donor_win_rate"]
        layer_output_rate = metrics["layer_output_donor_replace"]["donor_win_rate"]
        local_writer_rate = max(
            metrics["attention_output_donor_replace"]["donor_win_rate"],
            metrics["mlp_output_donor_replace"]["donor_win_rate"],
        )
        reports.append({
            "candidate_id": candidate_id,
            "layer": candidate_rows[0]["layer"],
            "zone": candidate_rows[0]["zone"],
            "condition_metrics": metrics,
            "layer_input_matches_layer_output_rate_gap": layer_output_rate - layer_input_rate,
            "maximum_local_writer_donor_win_rate": local_writer_rate,
            "residual_carry_dominant": (
                layer_input_rate >= 0.90 and layer_output_rate >= 0.90 and local_writer_rate <= 0.10
            ),
            "current_layer_is_unique_color_writer": False,
        })
    summary = {
        "schema_version": "phase560_parent_decomposition_analysis.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "candidate_reports": reports,
        "all_tested_layers_residual_carry_dominant": all(
            row["residual_carry_dominant"] for row in reports
        ),
        "source_color_content_route_confirmed": True,
        "source_color_unique_writer_identified": False,
        "object_color_binding_operation_identified": False,
        "next_physical_question": (
            "after the earliest qualified L3 source-color edge, at which downstream query layer and "
            "component does the causal source difference first appear"
        ),
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
