#!/usr/bin/env python3
"""Build the frozen Phase406 stage summary from formal result artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase406_conditioned_sequence_state"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read(OUT / "phase406_conditioned_sequence_protocol.json")
    analysis = read(OUT / "phase406_discovery_analysis.json")
    diagnostic = read(OUT / "phase406_discovery_failure_diagnostic.json")
    upper = read(OUT / "phase406_lexical_upper_bound_audit.json")
    horizon = read(OUT / "phase406_horizon_extension_diagnostic.json")
    parity = read(OUT / "diagnostics/phase406_glm_batch_parity.json")

    group_passes = {
        f"{row['model']}:{row['family_id']}": row["group_pass_count"]
        for row in analysis["model_family_rows"]
    }
    payload = {
        "schema_version": "80.7.0",
        "phase_id": "Phase406-ConditionedSequenceStage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "build_exact_state_condition_short_sequence_tables_and_gate_any_operator_or_physical_mapping",
        "assessment": {
            "state_condition_response_object_is_well_formed": True,
            "semantic_answer_contract_amended_and_all_models_restarted": True,
            "short_sequence_recovers_semantics_beyond_first_token": True,
            "stable_crossmodel_conditioned_state_observed": False,
            "leave_one_interface_transfer_crossmodel_validated": False,
            "direct_state_operator_test_authorized": False,
            "physical_mapping_authorized": False,
            "causal_or_neuron_work_authorized": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "frozen_all_split_case_count": protocol["denominator"][
                "case_count_all_models_all_splits"
            ],
            "formal_discovery_case_count": analysis["case_count"],
            "formal_model_family_cell_count": len(analysis["model_family_rows"]),
            "formal_group_count": 72,
            "calibration_case_count_consumed": 0,
            "behavioral_holdout_case_count_consumed": 0,
            "physical_holdout_case_count_consumed": 0,
            "horizon_diagnostic_selected_failure_case_count": horizon["case_count"],
        },
        "results": {
            "first_step_candidate_correct_count": analysis[
                "first_step_candidate_correct_count"
            ],
            "first_step_global_top_target_count": analysis[
                "first_step_global_top_is_target_count"
            ],
            "conservative_semantic_parse_count": analysis["semantic_parse_count"],
            "H12_short_sequence_semantic_correct_count": analysis[
                "short_sequence_semantic_correct_count"
            ],
            "H12_stop_or_sentence_boundary_after_semantic_count": analysis[
                "sequence_stop_or_boundary_after_semantic_count"
            ],
            "first_vocab_wrong_H12_sequence_correct_count": diagnostic[
                "first_vocab_wrong_sequence_correct_count"
            ],
            "candidate_correct_H12_sequence_wrong_count": diagnostic[
                "candidate_correct_sequence_wrong_count"
            ],
            "surface_disagreement_unit_count": diagnostic[
                "surface_disagreement_unit_count"
            ],
            "surface_unit_count": diagnostic["surface_unit_count"],
            "nonfinite_any_step_case_count": diagnostic[
                "nonfinite_any_generated_step_case_count"
            ],
            "formal_group_pass_count": sum(group_passes.values()),
            "formal_group_count": 72,
            "model_family_group_pass_counts": group_passes,
            "crossmodel_candidate_family_count": len(
                analysis["crossmodel_candidate_families"]
            ),
            "crossmodel_candidate_families": analysis[
                "crossmodel_candidate_families"
            ],
            "lexical_upper_bound_newly_credited_count": upper[
                "newly_credited_case_count"
            ],
            "lexical_upper_bound_crossmodel_candidate_family_count": len(
                upper["crossmodel_upper_bound_candidate_families"]
            ),
            "horizon_correct_count_by_H_on_formal_failures": horizon[
                "correct_count_by_horizon"
            ],
            "horizon_newly_recovered_after_H12": horizon[
                "newly_recovered_after_H12_by_horizon"
            ],
            "glm_batch_parity_case_count": parity["case_count"],
            "glm_batch4_nonfinite_count": parity["batch4_nonfinite_count"],
            "glm_batch1_nonfinite_count": parity["batch1_nonfinite_count"],
            "glm_finite_argmax_mismatch_count": parity[
                "finite_argmax_mismatch_count"
            ],
            "validated_conditioned_state_family_count": 0,
            "validated_direct_operator_count": 0,
            "new_physical_path_count": 0,
            "new_neuron_node_count": 0,
        },
        "hard_limits": [
            "six_conditions_are_a_finite_panel_not_all_legal_futures",
            "only_answer_first_and_natural_completion_interfaces_are_tested",
            "generation_history_is_fixed_empty_and_not_generalized",
            "semantic_parser_is_a_conservative_observer_not_internal_state",
            "H12_has_only_one_eos_event_across_5760_formal_cases",
            "H48_post_discovery_diagnostic_is_not_an_independent_candidate_gate",
            "deepseek_failure_subset_rebatching_changes_13_H12_outcomes",
            "glm_has_prompt_dependent_nonfinite_fp16_logits_at_batch1_and_batch4",
            "all_nine_model_family_cells_fail_the_registered_group_gate",
            "optimistic_exact_target_upper_bound_still_has_zero_crossmodel_candidates",
        ],
        "authorization": {
            "show_protocol_and_response_ledger": True,
            "show_short_sequence_recovery_as_observation": True,
            "show_conditioned_state_as_validated": False,
            "show_semantic_transition_as_internal_operator": False,
            "show_physical_state_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_calibration": False,
            "run_behavioral_holdout": False,
            "run_direct_operator": False,
            "run_physical_mapping": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "next_stage": {
            "objective": "freeze_a_fresh_long_horizon_condition_contract_with_batch1_parity_audit_and_identity_aware_semantic_slots",
            "automatic_model_execution_authorized": False,
            "reason": "the_next_protocol_changes_H_batch_parity_and_condition_interfaces_and_requires_a_fresh_denominator",
            "required_changes": [
                "freeze_H48_as_primary_before_new_outputs_are_seen",
                "reserve_fresh_semantic_groups_instead_of_reusing_phase406_discovery",
                "require_batch1_parity_on_a_frozen_subset_before_batched_execution",
                "retain_registered_identity_aliases_and_unique_grammar_slots",
                "record_horizon_ladder_but_never_choose_the_best_H_post_hoc",
                "require_crossmodel_calibration_before_any_operator_or_physical work",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    path = OUT / "phase406_conditioned_sequence_stage_summary.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
