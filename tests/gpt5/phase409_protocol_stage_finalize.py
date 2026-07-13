#!/usr/bin/env python3
"""Freeze the Phase409 protocol-only decision and its execution boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase409_dynamic_response_protocol"


def read(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    protocol = read("phase409_dynamic_response_protocol.json")
    qualification = read("phase409_protocol_qualification.json")
    agreement = read("phase409_rule_engine_agreement.json")
    prompt_audit = read("phase409_prompt_hash_audit.json")
    denominator = protocol["denominator"]
    machine_gate = bool(
        qualification["machine_protocol_gate_pass"]
        and agreement["valid"]
        and prompt_audit["valid"]
    )
    payload = {
        "schema_version": "83.1.0",
        "phase_id": "Phase409-DynamicResponseProtocolStage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": protocol["objective"],
        "assessment": {
            "phase408_failure_boundary_preserved": True,
            "schema_83_protocol_registry_frozen": True,
            "machine_protocol_gate_pass": machine_gate,
            "dual_rule_engine_agreement": agreement["valid"],
            "independent_human_rule_review_completed": False,
            "incremental_collector_token_equivalence_completed": False,
            "model_execution_performed": False,
            "activation_or_physical_mapping_performed": False,
            "causal_intervention_performed": False,
            "head_channel_or_neuron_scan_performed": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "query_contract_count": qualification["query_contract_count"],
            "rule_engine_scenario_count": agreement["scenario_count"],
            "abstract_case_count": denominator[
                "abstract_case_count_all_registered_splits"
            ],
            "future_model_rendered_prompt_count": denominator[
                "future_model_rendered_case_count_all_models"
            ],
            "discovery_abstract_case_count": denominator[
                "discovery_abstract_case_count"
            ],
            "future_discovery_model_case_count": denominator[
                "future_discovery_case_count_all_models"
            ],
            "gate_eligible_abstract_case_count": denominator[
                "gate_eligible_abstract_case_count"
            ],
            "conflict_diagnostic_abstract_case_count": denominator[
                "conflict_diagnostic_abstract_case_count"
            ],
            "future_qualification_case_count_per_model": denominator[
                "sealed_qualification_case_count_per_future_model"
            ],
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "dual_rule_engine_agreement_count": agreement["scenario_count"],
            "dual_rule_engine_disagreement_count": agreement["disagreement_count"],
            "prompt_hash_overlap_with_phase403_408_count": prompt_audit[
                "previous_phase_prompt_overlap_count"
            ],
            "within_phase_prompt_duplicate_count": prompt_audit[
                "within_phase_model_prompt_duplicate_count"
            ],
            "query_contract_joint_signature_failure_count": qualification[
                "joint_query_signature_failure_count"
            ],
            "grammar_sentence_bare_be_alias_count": qualification[
                "grammar_sentence_bare_be_alias_count"
            ],
            "knowledge_single_query_individually_injective": False,
            "knowledge_three_query_joint_signature_injective": True,
            "dynamic_event_parser_synthetic_test_count": 1,
            "model_behavior_result_count": 0,
            "new_internal_operator_count": 0,
            "new_physical_path_count": 0,
            "new_head_channel_or_neuron_count": 0,
        },
        "hard_limits": [
            "dual_software_solvers_share_one_registered_semantic_spec_and_are_not_independent_human_reviewers",
            "no_model_weight_was_loaded_and_no_behavioral_observation_was_collected",
            "dynamic_response_automaton_is_an_external_parser_not_a_model_internal_state_machine",
            "history_h3_is_a_two_state_conflict_diagnostic_and_not_a_unique_accuracy_target",
            "single_entity_queries_identify_six_knowledge_permutations_only_as_a_three_role_joint_signature",
            "incremental_generation_collector_has_not_yet_matched_the_frozen_reference_token_by_token",
            "no_activation_attention_mlp_head_channel_or_neuron_data_was_collected",
            "small_models_may_use_coarser_or_model_specific_execution_structures",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "show_protocol_registry": True,
            "show_machine_rule_agreement": True,
            "show_as_behavioral_model_evidence": False,
            "show_as_internal_language_state": False,
            "show_as_physical_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_model_qualification_next": False,
            "run_formal_discovery_next": False,
            "run_physical_mapping_next": False,
            "run_causal_intervention_next": False,
            "run_neuron_scan_next": False,
        },
        "next_stage": {
            "phase_id": "Phase409A",
            "objective": "independent_rule_review_and_incremental_collector_token_equivalence_only",
            "automatic_model_execution_authorized": False,
            "required_before_model_qualification": [
                "independent_human_rule_review",
                "old_and_incremental_collector_exact_token_equivalence_on_sealed_cases",
                "schema_83_parser_boundary_and_censoring_review",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    path = OUT / "phase409_protocol_stage_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
