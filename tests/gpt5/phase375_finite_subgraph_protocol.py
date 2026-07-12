#!/usr/bin/env python3
"""Freeze the Phase375 finite exact-subgraph protocol before extraction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"

INPUTS = {
    "collector_cases": PHASE371
    / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl",
    "base_audit": PHASE371 / "phase371c_internal_collection_audit.json",
    "adjacent_audit": PHASE371 / "phase371c_adjacent_extension_audit.json",
    "blind_contrast_audit": PHASE371
    / "phase371c_blind_vector_contrast/phase371c_blind_contrast_audit.json",
    "history_summary": PHASE371
    / "phase371c_exact_history_residual/phase371c_exact_history_residual_summary.json",
}

ROLE_ORDER = ("source_end", "query_end", "answer_start", "current_generation")

# These are fixed physical templates, not a combinatorial subset search. Every
# vector remains an exact d_model vector; summaries are diagnostics only.
STATE_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "receiver_transition": [
        {"role": "current_generation", "route": "layer_input"},
        {"role": "current_generation", "route": "attention_merge"},
        {"role": "current_generation", "route": "mlp_merge"},
    ],
    "source_query_receiver_outputs": [
        {"role": "source_end", "route": "layer_output"},
        {"role": "query_end", "route": "layer_output"},
        {"role": "current_generation", "route": "layer_output"},
    ],
    "binding_transition": [
        {"role": "source_end", "route": "layer_output"},
        {"role": "query_end", "route": "layer_output"},
        {"role": "current_generation", "route": "layer_input"},
        {"role": "current_generation", "route": "attention_merge"},
        {"role": "current_generation", "route": "mlp_merge"},
    ],
    "four_role_transition": [
        {"role": "source_end", "route": "layer_output"},
        {"role": "query_end", "route": "layer_output"},
        {"role": "answer_start", "route": "layer_output"},
        {"role": "current_generation", "route": "layer_input"},
        {"role": "current_generation", "route": "attention_merge"},
        {"role": "current_generation", "route": "mlp_merge"},
    ],
}

FORMATION_TEMPLATES = {
    "attention_children": [f"attention_partition_{index}" for index in range(8)],
    "mlp_children": [f"mlp_partition_{index}" for index in range(8)],
    "joint_attention_mlp_children": [
        *(f"attention_partition_{index}" for index in range(8)),
        *(f"mlp_partition_{index}" for index in range(8)),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    protocol = {
        "schema_version": "48.0.0",
        "phase_id": "Phase375-Protocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test_finite_exact_multi_vector_subgraphs_without_reusing_lossy_"
            "summaries_as_state"
        ),
        "attachment_audit": {
            "supported": [
                "lossy_scalar_and_single_route_readouts_are_not_sufficient",
                "exact_conservation_ledgers_are_a_valid_next_foundation",
                "finite_multi_route_subgraphs_are_a_warranted_next_unit",
            ],
            "corrections": [
                "three_local_layer_pairs_are_not_a_complete_model_replay",
                "lossless_means_reconstructable_for_measured_operations_only",
                "no_single_scientific_progress_percentage_is_defined",
                "gram_rank_and_cancellation_are_navigation_diagnostics_not_state",
                "local_conservation_replay_is_not_future_prediction",
                "local_mechanism_mediation_does_not_require_global_markov_sufficiency",
            ],
        },
        "frozen_scope": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "model_execution_order": ["qwen3", "glm4", "deepseek7b"],
            "mechanisms": ["relation_binding", "entity_recency"],
            "discovery_parallel_groups_per_model": 22,
            "conditions_per_group": 4,
            "discovery_case_count": 264,
            "generation_times": [0, 1, 2],
            "relative_depth_pairs": ["early", "middle", "late"],
            "target_role": "current_generation",
            "calibration_opened": False,
            "physical_holdout_opened": False,
            "new_prompt_generation": False,
        },
        "object_separation": {
            "formation_graph": (
                "exact_children_explain_how_a_parent_write_is_formed_but_are_not_"
                "separately_readable_after_linear_merge"
            ),
            "state_graph": (
                "exact_boundary_and_transition_vectors_that_are_present_on_the_"
                "measured_residual_computation"
            ),
            "formation_children_eligible_as_sufficient_state": False,
            "state_template_definitions": STATE_TEMPLATES,
            "formation_template_definitions": FORMATION_TEMPLATES,
            "arbitrary_head_or_neuron_subsets_allowed": False,
        },
        "blind_inventory": {
            "semantic_key_available": False,
            "all_cases_times_depths_and_templates_retained": True,
            "top_k_allowed": False,
            "task_score_selection_allowed": False,
            "exact_vector_duplication_allowed": False,
        },
        "discovery_readout": {
            "semantic_pairs": ["A_B", "C_D"],
            "current_generation_time": 1,
            "future_generation_time": 2,
            "past_generation_time": 0,
            "future_target": "adjacent_receiver_layer_output_difference_at_current_generation",
            "state_readout": "orthonormal_span_of_exact_template_vector_differences",
            "single_route_baseline": "best_individual_vector_in_same_template",
            "matched_controls": [
                "same_template_previous_time",
                "same_template_next_relative_depth",
                "same_template_cyclic_role_map",
                "same_template_next_independent_group",
            ],
            "vocabulary_context_readout": (
                "label_free_full_vocab_difference_persistence_is_a_context_gate_"
                "not_subgraph_mediation_evidence"
            ),
            "trained_probe_allowed": False,
            "posthoc_coordinate_rotation_allowed": False,
        },
        "frozen_numeric_gates": {
            "minimum_exact_basis_rank": 2,
            "maximum_current_projection_error": 0.75,
            "minimum_error_margin_vs_best_single": 0.01,
            "minimum_error_margin_vs_past": 0.02,
            "minimum_error_margin_vs_wrong_depth": 0.02,
            "minimum_error_margin_vs_wrong_role": 0.02,
            "minimum_error_margin_vs_wrong_group": 0.02,
            "maximum_history_gain": 0.01,
            "minimum_vocab_persistence_margin_vs_past": 0.01,
            "minimum_vocab_persistence_margin_vs_wrong_group": 0.01,
            "minimum_independent_groups_per_model_mechanism_template": 8,
            "lexical_pair_conjunction_required": True,
        },
        "cross_model_gate": {
            "canonical_identity": ["mechanism", "relative_depth", "template"],
            "heterogeneous_level2_requires_glm4": True,
            "same_layer_head_or_neuron_indices_required": False,
            "level3_requires_all_models": True,
        },
        "causal_authorization": {
            "requires_heterogeneous_level2": True,
            "discovery_only": True,
            "single_neuron_scan": False,
            "calibration_before_discovery_causal_success": False,
            "physical_before_calibration_success": False,
        },
        "stop_rules": {
            "no_crossmodel_candidate": "stop_before_model_intervention",
            "only_common_flow_not_vocab_context": "register_architecture_subgraph_only",
            "single_model_only": "register_model_specific_predictive_subgraph_only",
            "causal_failure": "do_not_register_language_mechanism",
        },
        "claim_boundary": {
            "multi_vector_projection_is_causal": False,
            "local_conservation_is_global_state_sufficiency": False,
            "candidate_is_language_path": False,
            "candidate_is_encoding_mechanism": False,
        },
        "input_hashes": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in INPUTS.items()
        },
        "authorization": {
            "build_blind_finite_subgraph_inventory": True,
            "freeze_inventory_hash_before_semantic_mapping": True,
            "run_discovery_mapping_after_freeze": True,
            "run_causal_replay_before_crossmodel_gate": False,
        },
    }
    write_json(OUT / "phase375_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
