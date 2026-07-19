#!/usr/bin/env python3
"""Analyze Phase557 upstream layer-input source tracing."""

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
STAGES = ("trace_discovery", "trace_confirmation")
GATES = {
    "same_case_max_abs_candidate_logit_delta": 0.05,
    "donor_switch_effect_median_min": 0.50,
    "donor_win_rate_min": 0.50,
    "donor_minus_best_control_mean_effect_min": 0.25,
    "delete_recipient_retention_rate_max": 0.75,
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


def condition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        OUT_DIR / "natural_color_upstream_trace" / model / stage
        / "phase557_natural_color_upstream_rows.jsonl"
    )
    rows = read_jsonl(path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)
    layer_reports = []
    for layer, layer_rows in sorted(grouped.items()):
        conditions = {
            condition: condition_report([row for row in layer_rows if row["condition"] == condition])
            for condition in sorted({row["condition"] for row in layer_rows})
        }
        expected = conditions["same_case_restore"]["row_count"]
        if any(value["row_count"] != expected for value in conditions.values()):
            raise RuntimeError(f"Phase557 upstream denominator drift: {model}/{stage}/L{layer}")
        same_valid = (
            conditions["same_case_restore"]["candidate_logit_delta_max"]
            <= GATES["same_case_max_abs_candidate_logit_delta"]
        )
        donor = conditions["layer_input_donor_replace"]
        delete = conditions["layer_input_delete"]
        controls = [
            conditions["relation_position_donor_replace"],
            conditions["channel_roll_donor_replace"],
        ]
        specificity = donor["switch_effect_mean"] - max(
            value["switch_effect_mean"] for value in controls
        )
        transfer_pass = bool(
            donor["switch_effect_median"] >= GATES["donor_switch_effect_median_min"]
            and donor["donor_win_rate"] >= GATES["donor_win_rate_min"]
            and specificity >= GATES["donor_minus_best_control_mean_effect_min"]
        )
        necessity_pass = (
            delete["recipient_retention_rate"] <= GATES["delete_recipient_retention_rate_max"]
        )
        layer_reports.append({
            "model": model,
            "stage": stage,
            "layer": layer,
            "pair_count_per_condition": expected,
            "conditions": conditions,
            "same_case_valid": same_valid,
            "donor_minus_best_control_mean_effect": specificity,
            "transfer_pass": transfer_pass,
            "necessity_pass": necessity_pass,
            "qualified_layer_input_edge": bool(same_valid and transfer_pass and necessity_pass),
        })
    return {
        "model": model,
        "stage": stage,
        "row_count": len(rows),
        "layer_reports": layer_reports,
        "qualified_layers": [
            row["layer"] for row in layer_reports if row["qualified_layer_input_edge"]
        ],
    }


def main() -> None:
    stage_reports = {
        stage: [analyze_stage(model, stage) for model in MODELS]
        for stage in STAGES
    }
    discovery = {
        (report["model"], layer)
        for report in stage_reports["trace_discovery"] for layer in report["qualified_layers"]
    }
    confirmation = {
        (report["model"], layer)
        for report in stage_reports["trace_confirmation"] for layer in report["qualified_layers"]
    }
    replicated = sorted(discovery & confirmation)
    model_reports = []
    for model in MODELS:
        layers = [layer for candidate_model, layer in replicated if candidate_model == model]
        model_reports.append({
            "model": model,
            "replicated_layer_input_edges": layers,
            "earliest_replicated_layer_input_edge": min(layers) if layers else None,
            "embedding_boundary_reached": 0 in layers,
        })
    summary = {
        "schema_version": "phase557_natural_color_upstream_analysis.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "frozen_gates": GATES,
        "stage_reports": stage_reports,
        "replicated_layer_input_edge_count": len(replicated),
        "model_reports": model_reports,
        "embedding_boundary_reached_models": [
            row["model"] for row in model_reports if row["embedding_boundary_reached"]
        ],
        "interpretation": (
            "A layer-input donor edge transports full lexical object identity. Reaching L0 does not "
            "show that color is stored in the embedding; it bounds the causal route at lexical input."
        ),
        "fine_grained_parameter_scan_authorized": False,
        "next_required_object": "conditional object-to-color transformation and query integration",
        "sealed_split_read": False,
        "closure_claim": False,
    }
    write_json(OUT_DIR / "phase557_natural_color_upstream_analysis.json", summary)
    print(json.dumps({
        "replicated_layer_input_edge_count": len(replicated),
        "model_reports": model_reports,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
