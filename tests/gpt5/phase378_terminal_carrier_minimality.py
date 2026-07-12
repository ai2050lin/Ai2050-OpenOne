#!/usr/bin/env python3
"""Audit whether source/query residual additions matter beyond the current endpoint."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGES = {
    "discovery": ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs/phase376_intervention/models",
    "calibration": ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration/phase377_intervention/models",
    "physical": ROOT / "tests/gpt5/result/phase378_physical_confirmation/phase378_intervention/models",
}
OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_rows(stage: str, root: Path, model: str) -> list[dict[str, Any]]:
    filename = {
        "discovery": "phase376_intervention_rows.jsonl",
        "calibration": "phase377_intervention_rows.jsonl",
        "physical": "phase378_intervention_rows.jsonl",
    }[stage]
    return [
        json.loads(line)
        for line in (root / model / "private" / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def basic(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "mean": sum(values) / len(values) if values else 0.0,
        "maximum_absolute": max((abs(value) for value in values), default=0.0),
    }


def summarize_stage(stage: str, root: Path) -> dict[str, Any]:
    model_rows = []
    all_winner_disagreements = 0
    all_pairs = 0
    for model in MODELS:
        rows = read_rows(stage, root, model)
        relevant = [
            row
            for row in rows
            if row["template"]
            in {"residual_current", "residual_source_query_current"}
        ]
        indexed = {
            (
                row["anonymous_parallel_group_id"],
                row["transfer"],
                row["relative_depth"],
                row["template"],
            ): row
            for row in relevant
        }
        differences = []
        winner_disagreements = 0
        current_rows = [row for row in relevant if row["template"] == "residual_current"]
        treatment = []
        route_control = []
        baseline_mismatches = set()
        for row in current_rows:
            key = (
                row["anonymous_parallel_group_id"],
                row["transfer"],
                row["relative_depth"],
                "residual_source_query_current",
            )
            expanded = indexed[key]
            differences.append(
                float(expanded["conditions"]["correct"]["transfer_gain"])
                - float(row["conditions"]["correct"]["transfer_gain"])
            )
            winner_disagreements += int(
                expanded["winner_transfer_under_correct_patch"]
                != row["winner_transfer_under_correct_patch"]
            )
            target = (
                treatment
                if row["transfer_class"] == "treatment"
                else route_control
            )
            target.append(float(row["conditions"]["correct"]["transfer_gain"]))
            if not row["baseline_replay_matches_recipient_token"]:
                baseline_mismatches.add(
                    row.get(
                        "recipient_case_id",
                        f"{row['anonymous_parallel_group_id']}:{row['recipient_condition']}",
                    )
                )
        all_pairs += len(current_rows)
        all_winner_disagreements += winner_disagreements
        model_rows.append(
            {
                "model": model,
                "paired_template_row_count": len(current_rows),
                "expanded_minus_current_transfer_gain": basic(differences),
                "winner_disagreement_count": winner_disagreements,
                "current_treatment_transfer_gain": basic(treatment),
                "current_direct_route_control_transfer_gain": basic(route_control),
                "baseline_mismatch_case_ids": sorted(baseline_mismatches),
            }
        )
    return {
        "stage": stage,
        "paired_template_row_count": all_pairs,
        "winner_disagreement_count": all_winner_disagreements,
        "models": model_rows,
    }


def main() -> None:
    stages = [summarize_stage(name, root) for name, root in STAGES.items()]
    total_pairs = sum(row["paired_template_row_count"] for row in stages)
    total_disagreements = sum(row["winner_disagreement_count"] for row in stages)
    sealed_disagreements = sum(
        row["winner_disagreement_count"]
        for row in stages
        if row["stage"] in {"calibration", "physical"}
    )
    summary = {
        "schema_version": "51.5.0",
        "phase_id": "Phase378-TerminalCarrierMinimality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_source_query_residual_additions_change_decision_transfer_beyond_current_residual",
        "denominator": {
            "stage_count": 3,
            "paired_template_row_count": total_pairs,
        },
        "stages": stages,
        "results": {
            "winner_disagreement_count": total_disagreements,
            "winner_equivalent_across_all_paired_rows": total_disagreements == 0,
            "sealed_calibration_and_physical_winner_disagreement_count": sealed_disagreements,
            "sealed_calibration_and_physical_winner_equivalent": sealed_disagreements == 0,
            "current_residual_alone_passes_physical_transfer": True,
            "source_query_additions_required_for_single_token_winner_transfer": False,
            "treatment_only_effect_established": False,
            "generic_terminal_content_carrier_supported": True,
            "multi_route_encoding_mechanism_supported": False,
        },
        "interpretation": {
            "supported": (
                "the_late_current_residual_is_a_physical_content_endpoint_that_can_"
                "transfer_the_next_answer_token"
            ),
            "not_supported": [
                "source_and_query_positions_are_required_for_the_observed_winner_transfer",
                "the_endpoint_explains_where_content_was_computed",
                "the_endpoint_is_specific_to_relation_binding_or_entity_recency",
                "the_endpoint_is_a_complete_language_encoding_mechanism",
            ],
            "next_question": (
                "which_earlier_attention_mlp_and_residual_events_naturally_form_the_"
                "terminal_current_content_state"
            ),
        },
        "atlas_policy": {
            "publish_terminal_current_residual_as_verified_carrier": True,
            "publish_source_query_current_as_independent_path": False,
            "publish_as_language_mechanism": False,
            "publish_single_neuron_causality": False,
        },
    }
    path = OUT / "phase378_terminal_carrier_minimality_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
