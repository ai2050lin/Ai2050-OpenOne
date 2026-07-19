#!/usr/bin/env python3
"""Analyze discovery/confirmation Phase557 coarse parent-block interventions."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
MODELS = ("qwen3", "glm4")
STAGES = ("parent_discovery", "parent_confirmation")
PARENTS = ("layer_input", "attention_output", "mlp_output")
GATES = {
    "same_case_max_abs_candidate_logit_delta": 0.05,
    "parent_donor_switch_effect_median_min": 0.50,
    "parent_donor_win_rate_min": 0.50,
    "parent_minus_roll_mean_effect_min": 0.25,
    "parent_delete_recipient_retention_rate_max": 0.75,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["donor_switch_effect"]) for row in rows]
    deltas = [
        max(
            abs(float(value) - float(row["baseline_scores"][word]))
            for word, value in row["intervention_scores"].items()
        )
        for row in rows
    ]
    return {
        "row_count": len(rows),
        "switch_effect_mean": sum(effects) / len(effects),
        "switch_effect_median": float(statistics.median(effects)),
        "donor_win_rate": sum(row["intervention_donor_wins"] for row in rows) / len(rows),
        "recipient_retention_rate": sum(row["intervention_recipient_retained"] for row in rows) / len(rows),
        "candidate_logit_delta_max": max(deltas),
    }


def analyze_stage(model: str, stage: str) -> dict[str, Any]:
    path = (
        OUT_DIR / "natural_color_parent_blocks" / model / stage
        / "phase557_natural_color_parent_rows.jsonl"
    )
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    candidates = []
    for candidate_id, candidate_rows in sorted(grouped.items()):
        conditions = {
            condition: report([row for row in candidate_rows if row["condition"] == condition])
            for condition in sorted({row["condition"] for row in candidate_rows})
        }
        expected = conditions["same_case_restore"]["row_count"]
        if any(value["row_count"] != expected for value in conditions.values()):
            raise RuntimeError(f"Phase557 parent denominator drift: {candidate_id}/{stage}")
        implementation_valid = (
            conditions["same_case_restore"]["candidate_logit_delta_max"]
            <= GATES["same_case_max_abs_candidate_logit_delta"]
        )
        parent_reports = {}
        for parent in PARENTS:
            donor = conditions[f"{parent}_donor_replace"]
            delete = conditions[f"{parent}_delete"]
            roll = conditions[f"{parent}_roll"]
            specificity = donor["switch_effect_mean"] - roll["switch_effect_mean"]
            transfer_pass = bool(
                donor["switch_effect_median"] >= GATES["parent_donor_switch_effect_median_min"]
                and donor["donor_win_rate"] >= GATES["parent_donor_win_rate_min"]
                and specificity >= GATES["parent_minus_roll_mean_effect_min"]
            )
            necessity_pass = (
                delete["recipient_retention_rate"]
                <= GATES["parent_delete_recipient_retention_rate_max"]
            )
            parent_reports[parent] = {
                "donor_replace": donor,
                "delete": delete,
                "roll_control": roll,
                "donor_minus_roll_mean_effect": specificity,
                "transfer_pass": transfer_pass,
                "necessity_pass": necessity_pass,
                "qualified_parent_block": bool(implementation_valid and transfer_pass and necessity_pass),
            }
        candidates.append({
            "candidate_id": candidate_id,
            "model": model,
            "stage": stage,
            "layer": int(candidate_rows[0]["layer"]),
            "pair_count_per_condition": expected,
            "implementation_valid": implementation_valid,
            "layer_output_reference": conditions["layer_output_donor_replace"],
            "parent_reports": parent_reports,
            "qualified_parent_blocks": sorted([
                parent for parent, value in parent_reports.items() if value["qualified_parent_block"]
            ]),
        })
    return {
        "model": model,
        "stage": stage,
        "row_count": len(rows),
        "candidate_reports": candidates,
    }


def main() -> None:
    stage_reports = {
        stage: [analyze_stage(model, stage) for model in MODELS]
        for stage in STAGES
    }
    discovery = {
        (candidate["model"], candidate["candidate_id"], parent)
        for model_report in stage_reports["parent_discovery"]
        for candidate in model_report["candidate_reports"]
        for parent in candidate["qualified_parent_blocks"]
    }
    confirmation = {
        (candidate["model"], candidate["candidate_id"], parent)
        for model_report in stage_reports["parent_confirmation"]
        for candidate in model_report["candidate_reports"]
        for parent in candidate["qualified_parent_blocks"]
    }
    replicated = sorted(discovery & confirmation)
    replicated_rows = [{
        "schema_version": "phase557_replicated_natural_color_parent_block.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "model": model,
        "candidate_id": candidate_id,
        "parent_component": parent,
        "source_position": "object_source_end",
        "discovery_stage": "behavior_confirmation_surfaces_0_1",
        "confirmation_stage": "unseen_recombination_surfaces_2_3",
        "sealed": False,
    } for model, candidate_id, parent in replicated]
    replicated_writer_rows = [
        row for row in replicated_rows if row["parent_component"] in ("attention_output", "mlp_output")
    ]
    replicated_carry_rows = [
        row for row in replicated_rows if row["parent_component"] == "layer_input"
    ]
    summary = {
        "schema_version": "phase557_natural_color_parent_analysis.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "frozen_gates": GATES,
        "stage_reports": stage_reports,
        "discovery_qualified_parent_count": len(discovery),
        "confirmation_qualified_parent_count": len(confirmation),
        "replicated_parent_block_count": len(replicated_rows),
        "replicated_parent_blocks": replicated_rows,
        "replicated_writer_parent_count": len(replicated_writer_rows),
        "replicated_residual_carry_parent_count": len(replicated_carry_rows),
        "upstream_layer_input_trace_required": bool(replicated_carry_rows),
        "fine_grained_parameter_scan_authorized": bool(replicated_writer_rows),
        "fine_grained_scan_executed": False,
        "sealed_split_read": False,
        "closure_claim": False,
    }
    write_json(OUT_DIR / "phase557_natural_color_parent_analysis.json", summary)
    path = OUT_DIR / "phase557_replicated_natural_color_parent_blocks.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in replicated_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "replicated_parent_block_count": len(replicated_rows),
        "replicated_parent_blocks": [
            {"model": row["model"], "candidate_id": row["candidate_id"], "parent": row["parent_component"]}
            for row in replicated_rows
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
