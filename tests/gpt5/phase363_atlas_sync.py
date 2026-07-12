#!/usr/bin/env python3
"""Publish the strict Phase363 result without exposing sealed tensors or cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "tests/gpt5/result/phase363_temporal_hypotheses/strict_temporal_innovation_formulas"
P361 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2/phase361_contract_repair_summary.json"
TARGETS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    hypothesis = read_json(ROUND / "phase363_hypothesis_summary.json")
    formulas = read_jsonl(ROUND / "phase363_all_formula_rows.jsonl")
    contract = read_json(P361)
    positive_all = [row for row in formulas if all(value["gain"] > 0 for value in row["per_model"].values())]
    summary = {
        "schema_version": "40.1.0",
        "phase_id": "Phase363",
        "created_at": now(),
        "phase362_assessment": {
            "overall_direction_correct": True,
            "correct_parts": [
                "component_conservation_next_layer_prediction_and_generation_time_prediction_are_distinct_gates",
                "old_candidates_require_new_frozen_mapping_before_new_claims",
                "independent_groups_not_layers_or_conditions_are_the_analysis_units",
                "physical_confirmation_must_remain_sealed_until_discovery_formulas_pass",
            ],
            "required_corrections": [
                "phase361_candidates_did_not_predefine_temporal_competition_or_divergence_mappings",
                "current_trace_identifies_token_competition_not_full_phrase_or_semantic_competition",
                "positive_gain_without_exceeding_natural_variation_is_not_a_survivor",
                "one_global_project_progress_percentage_is_not_scientifically_valid",
            ],
        },
        "denominator": {
            "model_count": 3,
            "discovery_prompt_count": 288,
            "sealed_confirmation_prompt_count": 96,
            **hypothesis["denominator"],
            "time_innovation_formula_count": sum(row["target_type"] == "time_innovation" for row in formulas),
            "competition_change_formula_count": sum(row["target_type"] == "competition_change" for row in formulas),
        },
        "results": {
            **hypothesis["results"],
            "positive_gain_all_models_formula_count": len(positive_all),
            "positive_gain_all_models_formula_ids": [row["formula_id"] for row in positive_all],
            "cross_model_temporal_state_candidate_count": 0,
            "cross_model_competition_state_candidate_count": 0,
            "physical_confirmation_executed_case_count": 0,
        },
        "objective_coverage": {
            "registered_family_coverage": {"numerator": 9, "denominator": 9},
            "registered_mechanism_coverage": {"numerator": 18, "denominator": 18},
            "blind_discovery_admission_coverage": {
                "numerator": contract["results"]["total_blind_discovery_admitted_count"], "denominator": 18,
            },
            "strict_temporal_formula_coverage": {"numerator": 0, "denominator": len(formulas)},
            "physical_heldout_mechanism_coverage": {"numerator": 0, "denominator": 18},
            "causal_sealed_mechanism_coverage": {"numerator": 0, "denominator": 18},
            "strict_mechanism_closure": {"numerator": 0, "denominator": 72},
        },
        "claim_boundary": {
            **hypothesis["claim_boundary"],
            "physical_confirmation_read": hypothesis["quality"]["physical_confirmation_read"],
            "phase361_phase362_candidate_route_closed": True,
            "local_deepseek_signals_are_cross_model_mechanisms": False,
            "language_encoding_closed": False,
            "intelligence_theory_closed": False,
            "single_global_progress_percentage_valid": False,
        },
        "decision": "close_phase361_phase362_temporal_candidate_route_keep_96_confirmation_cases_sealed",
        "next_stage": {
            "name": "dynamic_trajectory_object_preregistration",
            "priority": "return_to_blind_cartography_instead_of_patching_the_seven_closed_candidates",
            "work_packages": [
                "define_dynamic_path_objects_from_replayable_component_and_attention_edge_ledgers",
                "freeze_path_matching_without_behavior_labels_or_confirmation_cases",
                "test whether path_objects_recur_across_admitted_mechanisms_and_models",
                "only_then_register_new_competition_or_causal_predictions",
            ],
            "new_model_execution_authorized": False,
        },
    }
    write_json(ROUND / "phase363_global_summary.json", summary)
    updated_at = now()
    public_files = (
        "phase363_global_summary.json",
        "phase363_hypothesis_summary.json",
        "phase363_all_formula_rows.jsonl",
    )
    for target in TARGETS:
        write_json(target / "phase363_global_summary.json", summary)
        write_json(target / "phase363_hypothesis_summary.json", hypothesis)
        (target / "phase363_all_formula_rows.jsonl").write_text(
            (ROUND / "phase363_all_formula_rows.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
        )
        manifest_path = target / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["updated_at"] = updated_at
        manifest["phase363"] = {
            "status": "temporal_candidate_route_closed_confirmation_remains_sealed",
            "discovery_prompt_count": 288,
            "sealed_confirmation_prompt_count": 96,
            "tested_formula_count": len(formulas),
            "frozen_formula_count": hypothesis["denominator"]["frozen_formula_count"],
            "temporal_state_candidate_count": 0,
            "competition_state_candidate_count": 0,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
            "raw_tensors_frontend_exported": False,
            "single_global_progress_percentage_valid": False,
            "files": list(public_files),
        }
        write_json(manifest_path, manifest)
        progress_path = target / "progress.json"
        progress = read_json(progress_path)
        progress["last_phase"] = "Phase363"
        progress["updated_at"] = updated_at
        progress["objective_denominator_progress"] = summary["objective_coverage"]
        progress["single_global_progress_percentage_valid"] = False
        progress["phase363_decision"] = summary["decision"]
        write_json(progress_path, progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
