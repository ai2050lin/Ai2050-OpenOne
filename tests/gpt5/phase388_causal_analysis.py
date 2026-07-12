#!/usr/bin/env python3
"""Summarize Phase388 source K/V interventions without a composite score."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def scenario(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["scenario_rows"] if item["intervention"] == name)


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def model_summary(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    main = [scenario(row, "donor_source_kv") for row in rows]
    wrong_source = [scenario(row, "donor_wrong_source_kv") for row in rows]
    wrong_layer = [
        scenario(row, "donor_source_kv_at_terminal_control_depth") for row in rows
    ]
    key_only = [scenario(row, "donor_source_k_only") for row in rows]
    value_only = [scenario(row, "donor_source_v_only") for row in rows]

    main_query = [item["query_projection_toward_donor"] for item in main]
    main_margin = [item["donor_direction_margin_shift"] for item in main]
    wrong_source_query = [
        item["query_projection_toward_donor"] for item in wrong_source
    ]
    wrong_layer_query = [item["query_projection_toward_donor"] for item in wrong_layer]
    wrong_source_margin = [item["donor_direction_margin_shift"] for item in wrong_source]
    wrong_layer_margin = [item["donor_direction_margin_shift"] for item in wrong_layer]
    query_control_dominance = [
        main_query[index] > wrong_source_query[index]
        and main_query[index] > wrong_layer_query[index]
        for index in range(len(rows))
    ]
    margin_control_dominance = [
        main_margin[index] > wrong_source_margin[index]
        and main_margin[index] > wrong_layer_margin[index]
        for index in range(len(rows))
    ]
    generation_switches = [
        bool(row["main_generation"]["donor_target_strict_switch"]) for row in rows
    ]
    donor_present = [
        bool(row["main_generation"]["donor_target_present"]) for row in rows
    ]
    recipient_present = [
        bool(row["main_generation"]["recipient_target_present"]) for row in rows
    ]

    query_gate = (
        median(main_query) > 0.0
        and rate([value > 0.0 for value in main_query]) >= 0.75
        and rate(query_control_dominance) >= 0.75
    )
    margin_gate = (
        median(main_margin) > 0.0
        and rate([value > 0.0 for value in main_margin]) >= 0.75
        and rate(margin_control_dominance) >= 0.75
    )
    behavior_gate = rate(generation_switches) >= 0.5

    return {
        "schema_version": "62.5.0",
        "phase_id": "Phase388-CausalAnalysis",
        "model": model,
        "direction_count": len(rows),
        "query_state": {
            "main_median_projection": median(main_query),
            "main_positive_rate": rate([value > 0.0 for value in main_query]),
            "wrong_source_median_projection": median(wrong_source_query),
            "wrong_layer_median_projection": median(wrong_layer_query),
            "main_exceeds_both_controls_rate": rate(query_control_dominance),
            "key_only_median_projection": median(
                item["query_projection_toward_donor"] for item in key_only
            ),
            "value_only_median_projection": median(
                item["query_projection_toward_donor"] for item in value_only
            ),
            "gate_pass": query_gate,
        },
        "target_margin": {
            "main_median_shift": median(main_margin),
            "main_positive_rate": rate([value > 0.0 for value in main_margin]),
            "wrong_source_median_shift": median(wrong_source_margin),
            "wrong_layer_median_shift": median(wrong_layer_margin),
            "main_exceeds_both_controls_rate": rate(margin_control_dominance),
            "key_only_median_shift": median(
                item["donor_direction_margin_shift"] for item in key_only
            ),
            "value_only_median_shift": median(
                item["donor_direction_margin_shift"] for item in value_only
            ),
            "gate_pass": margin_gate,
        },
        "generation": {
            "donor_target_present_count": sum(donor_present),
            "recipient_target_present_count": sum(recipient_present),
            "strict_donor_target_switch_count": sum(generation_switches),
            "strict_donor_target_switch_rate": rate(generation_switches),
            "gate_pass": behavior_gate,
        },
        "all_three_outcomes_pass": query_gate and margin_gate and behavior_gate,
    }


def main() -> None:
    rows_by_model = {
        model: read_jsonl(
            P388 / "collection/causal_test" / model / "direction_rows.jsonl"
        )
        for model in MODELS
    }
    if any(len(rows) != 32 for rows in rows_by_model.values()):
        raise RuntimeError(
            f"Phase388 causal denominator mismatch: "
            f"{ {model: len(rows) for model, rows in rows_by_model.items()} }"
        )
    model_rows = [model_summary(model, rows_by_model[model]) for model in MODELS]
    write_jsonl(P388 / "phase388_model_causal_rows.jsonl", model_rows)
    all_rows = [row for rows in rows_by_model.values() for row in rows]
    patch_failures = []
    for row in all_rows:
        for item in row["scenario_rows"]:
            audit = item["patch_audit"]
            if any(
                audit[key] != 0.0
                for key in (
                    "key_max_patch_error",
                    "value_max_patch_error",
                    "key_max_outside_error",
                    "value_max_outside_error",
                )
            ):
                patch_failures.append(f"{row['direction_id']}:{item['intervention']}")

    outcome_counts = Counter()
    for row in model_rows:
        outcome_counts["query"] += int(row["query_state"]["gate_pass"])
        outcome_counts["margin"] += int(row["target_margin"]["gate_pass"])
        outcome_counts["behavior"] += int(row["generation"]["gate_pass"])
        outcome_counts["all"] += int(row["all_three_outcomes_pass"])

    positive_registration_allowed = False
    summary = {
        "schema_version": "62.5.0",
        "phase_id": "Phase388-CausalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "causal_test_group_count": 16,
            "models": list(MODELS),
            "direction_count": len(all_rows),
            "scenario_count": len(all_rows) * 7,
            "main_patched_generation_count": len(all_rows),
            "candidate_group_replacement_count": 0,
            "phase386_holdout_reuse_count": 0,
        },
        "results": {
            "models_passing_query_state_gate": outcome_counts["query"],
            "models_passing_target_margin_gate": outcome_counts["margin"],
            "models_passing_behavior_gate": outcome_counts["behavior"],
            "models_passing_all_three_outcomes": outcome_counts["all"],
            "strict_donor_target_switch_count": sum(
                row["generation"]["strict_donor_target_switch_count"]
                for row in model_rows
            ),
            "patch_failure_count": len(patch_failures),
            "causal_source_kv_transport_path_established": False,
            "single_neuron_path_established": False,
            "language_encoding_closed": False,
        },
        "gate_registration_boundary": {
            "qualitative_outcomes_frozen_before_collection": True,
            "numeric_rate_thresholds_written_before_full_collection": False,
            "positive_causal_registration_allowed": positive_registration_allowed,
            "reason": (
                "The full result may reject this coarse intervention, but no positive "
                "causal path may be registered from post-collection numeric gates."
            ),
        },
        "model_results": model_rows,
        "decision": (
            "Close the coarse single-position full-K/V transfer as a cross-model "
            "causal language path. Preserve component-wise perturbation measurements "
            "as negative/diagnostic evidence."
        ),
        "authorization": {
            "register_direct_causal_edge": False,
            "register_single_neuron_path": False,
            "reuse_current_causal_denominator": False,
            "analyze_head_and_source_resolved_receiver_contributions": True,
            "run_unbounded_neuron_scan": False,
        },
        "next_stage": {
            "phase": 389,
            "objective": (
                "decompose the measured receiver effect by query head and source role "
                "before any new intervention"
            ),
            "model_run_required_initially": False,
            "reason": (
                "The current aggregate query state can hide head-wise opposition; the "
                "saved exact Phase386 attention events should first determine whether "
                "a stable receiver head/source route exists."
            ),
        },
    }
    write_json(P388 / "phase388_causal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
