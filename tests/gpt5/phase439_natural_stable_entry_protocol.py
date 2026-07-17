#!/usr/bin/env python3
"""Freeze the Phase439/441 natural-stable-entry protocol as machine-readable JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase439_natural_stable_entry"
OUT_PATH = OUT_DIR / "phase439_protocol_freeze.json"


TASK_LIBRARY = {
    "knowledge_network": [
        "context_single_attribute_read",
        "parametric_single_fact_read",
        "category_attribute_inheritance",
        "parametric_context_consistent",
        "parametric_context_conflict",
    ],
    "single_step_reasoning": [
        "set_inclusion_one_step",
        "size_comparison_one_step",
        "conditional_implication_one_step",
        "relation_transitive_one_step",
        "one_step_exclusion",
    ],
    "syntax_system": [
        "subject_verb_number_agreement",
        "pronoun_number_agreement",
        "active_passive_role_conversion",
        "relative_clause_role",
        "sentence_boundary_closure_choice",
    ],
}


SHORT_ANSWER_INTERFACES = [
    "direct_short_answer",
    "restricted_choice",
    "single_field",
    "natural_short_sentence",
]


SURFACE_TRANSFORMS = [
    "synonym_rewrite",
    "order_swap",
    "distance_change",
    "boundary_rewrite",
    "length_change",
    "label_or_structure_order_change",
    "query_expression_rewrite",
]


SPLITS = [
    "interface_calibration",
    "task_discovery",
    "surface_orbit_holdout",
    "physical_window_freeze",
    "physical_prediction_holdout",
    "sealed_physical_holdout",
]


SHORTCUT_BASELINES = [
    "position",
    "nearest_item",
    "token_frequency",
    "majority_class",
    "surface_template",
    "answer_length",
]


def build_protocol() -> dict:
    return {
        "schema_version": "phase439_natural_stable_entry_protocol.v2",
        "phase_id": "Phase439-441-NaturalStableEntry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "theory_name": "语言是动态模式网络",
        "method_frame": "条件物理状态图谱",
        "status": "protocol_v2_frozen_no_cuda_run",
        "models": ["qwen3", "glm4", "deepseek7b"],
        "task_library": TASK_LIBRARY,
        "interfaces": SHORT_ANSWER_INTERFACES,
        "surface_transforms": SURFACE_TRANSFORMS,
        "splits": SPLITS,
        "split_rules": {
            "entity_relation_rule_template_disjoint_across_splits": True,
            "task_discovery_selects_at_most_one_task_per_model_ability": True,
            "no_posthoc_task_addition": True,
            "no_threshold_patch_after_failure": True,
            "physical_prediction_holdout_is_independent_from_window_freeze": True,
            "sealed_physical_holdout_read_only_after_all_gates_pass": True,
        },
        "selection_rule": {
            "order": [
                "pass_hard_behavior_gates",
                "maximize_worst_surface_condition_accuracy",
                "minimize_surface_orbit_accuracy_range",
                "prefer_simplest_interface",
                "deterministic_task_id_tiebreak",
            ],
            "select_at_most_one_task_per_model_ability": True,
            "forbid_selecting_by_physical_metrics": True,
        },
        "behavior_gates": {
            "semantic_lcb_95_min": 0.85,
            "other_ucb_95_max": 0.05,
            "surface_orbit_max_gap": 0.05,
            "orbit_group_consistency_lcb_95_min": 0.80,
            "orbit_group_consistency_formula": "C_orbit=(1/N)*sum_i prod_g 1[Y(gx_i)=Y_i_star]",
            "must_beat_shortcut_baselines": SHORTCUT_BASELINES,
        },
        "sample_freeze_plan": {
            "base_semantic_groups_per_task": 384,
            "base_semantic_groups_per_split": 64,
            "requires_static_contract_before_cuda": True,
            "manifest_path": "tests/gpt5/result/phase439_natural_stable_entry/phase441_task_split_manifest.json",
        },
        "semantic_orbit_contract": {
            "every_transform_requires_semantic_preservation_proof": True,
            "every_transform_requires_position_role_mapping": True,
            "answer_aliases_are_frozen_before_model_run": True,
            "forbid_semantic_changing_transform_in_invariance_orbit": True,
            "equivariance_transforms_must_register_node_mapping": True,
        },
        "physical_scope": {
            "run_only_after_behavior_and_surface_orbit_pass": True,
            "record_compact_all_layer_ledger": True,
            "record_residual_qkv_attention_output_mlp_net_abs_coherence_random_projection": True,
            "no_causal_intervention": True,
            "no_head_channel_neuron_scan": True,
        },
        "orbit_metrics": {
            "quality_formula": "Q_orbit=(D_between-D_within)/(D_between+D_within+epsilon)",
            "requires_layer_position_time_normalization": True,
            "requires_matched_between_orbit_controls": [
                "token_length",
                "token_frequency",
                "sentence_form",
                "position",
                "interface",
                "ability_family",
            ],
            "requires_blind_window_selection": True,
            "requires_permutation_null": True,
            "permutation_null": "shuffle_semantic_orbit_labels_within_matched_controls",
        },
        "equivariance_metric": {
            "formula": "E_equiv=||A(gx)-P_g A(x) P_g^T||_F/(||A(x)||_F+epsilon)",
            "forbid_posthoc_full_hidden_linear_fit": True,
            "requires_pre_registered_node_mapping": True,
        },
        "physical_prediction_gate": {
            "split": "physical_prediction_holdout",
            "targets": ["complete_semantic_event", "ability_family", "orbit_group"],
            "minimum_improvements": ["DeltaCE>0", "DeltaBalancedAccuracy>0", "DeltaMacroF1>0"],
            "must_exceed_permutation_null_95_percentile": True,
            "baselines": SHORTCUT_BASELINES + ["interface", "recent_item"],
        },
        "transport_observables": {
            "knowledge": "W_target_fact_to_query - W_distractor_to_query",
            "reasoning": "W_required_premise_to_query - W_irrelevant_premise_to_query",
            "grammar": "W_controller_to_slot - W_distractor_to_slot",
        },
        "sealed_authorization_gate": [
            "G_behavior",
            "G_orbit",
            "G_equivariance",
            "G_transport",
            "G_prediction",
            "G_specificity",
        ],
        "stop_rules": [
            "all_tasks_fail_behavior_entry_close_current_small_model_natural_entry",
            "surface_orbit_fail_record_task_or_interface_overfit",
            "blind_geometry_follows_vocabulary_or_template_close_semantic_state_candidate",
            "transport_follows_position_or_distance_record_surface_routing",
            "physical_prediction_fail_do_not_read_sealed",
            "single_model_only_pass_record_model_specific_route",
            "sealed_fail_close_model_task_physical_candidate",
            "no_new_tasks_interfaces_thresholds_or_windows_after_failure",
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(build_protocol(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
