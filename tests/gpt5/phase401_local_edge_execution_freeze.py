#!/usr/bin/env python3
"""Freeze Phase401 donor pairs, controls, aggregation, and selection rules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    instrument = read_json(OUT / "phase401_instrument_audit.json")
    if not instrument["authorization"]["run_discovery_local_edges"]:
        raise RuntimeError("Phase401 local-edge discovery is not authorized")
    protocol = read_json(OUT / "phase401_local_edge_protocol.json")
    payload = {
        "schema_version": "75.8.0",
        "phase_id": "Phase401-LocalEdgeExecutionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_direct_parent_replacement_before_discovery_vectors",
        "discovery_denominator": {
            "eligible_surfaces": read_json(
                OUT / "phase401_behavior_freeze_summary.json"
            )["eligible_surfaces"],
            "groups_per_surface": 8,
            "conditions_per_group": 16,
            "directed_true_pairs_per_group": 16,
            "all_layers": True,
            "models": protocol["models_in_execution_order"],
        },
        "true_pair": {
            "recipient": "axis_Rr_Oo_Qq",
            "donor": "axis_R(1-r)_Oo_Qq",
            "both_directions": True,
            "expected_target_change_required": True,
        },
        "source_roles_replaced_together": [
            "source_entity_a",
            "source_entity_b",
            "source_value_a",
            "source_value_b",
            "source_structure",
        ],
        "position_alignment": {
            "source_roles": "all_frozen_role_positions_in_token_order",
            "receiver_position": "first_frozen_position_for_receiver_role",
            "mismatched_source_role_token_counts": "pair_not_informative",
        },
        "controls": {
            "wrong_source_order_matched_same_target": {
                "donor": "axis_Rr_O(1-o)_Qq",
                "receiver": "query_end",
                "depth": "same",
                "role_map": "identity",
            },
            "wrong_receiver_role": {
                "donor": "axis_R(1-r)_Oo_Qq",
                "receiver": "query_entity",
                "depth": "same",
                "role_map": "identity",
            },
            "wrong_semantic_time": {
                "donor": "axis_R(1-r)_Oo_Qq",
                "receiver": "answer_anchor",
                "depth": "same",
                "role_map": "identity",
            },
            "wrong_depth_quarter_model_shift": {
                "donor": "axis_R(1-r)_Oo_Qq",
                "receiver": "query_end",
                "depth": "plus_quarter_else_minus_quarter",
                "role_map": "identity",
            },
            "source_role_permutation": {
                "donor": "axis_R(1-r)_Oo_Qq",
                "receiver": "query_end",
                "depth": "same",
                "role_map": {
                    "source_entity_a": "source_entity_b",
                    "source_entity_b": "source_entity_a",
                    "source_value_a": "source_value_b",
                    "source_value_b": "source_value_a",
                    "source_structure": "source_structure",
                },
            },
            "same_content_wrong_structure": {
                "donor": "axis_R(1-r)_O(1-o)_Qq",
                "receiver": "query_end",
                "depth": "same",
                "role_map": "identity",
            },
            "deterministic_random_natural_donor": {
                "donor": "other_axis_Rr_Oo_Qq",
                "receiver": "query_end",
                "depth": "same",
                "role_map": "identity",
            },
            "same_absolute_mass_sign_permuted": {
                "donor": "axis_R(1-r)_Oo_Qq",
                "receiver": "query_end",
                "depth": "same",
                "operation": (
                    "recipient_attention_plus_absolute_true_attention_delta_times_"
                    "deterministic_alternating_sign"
                ),
            },
        },
        "local_chain": [
            "source_KV_replacement_to_query_attention",
            "recipient_layer_input_plus_counterfactual_attention",
            "real_post_attention_norm_and_MLP_recomputation",
            "counterfactual_post_attention_plus_counterfactual_MLP",
        ],
        "pair_gate": protocol["discovery_local_edge_gate"],
        "minimum_informative_baseline_relative_norm": protocol[
            "intervention_pair_contract"
        ]["minimum_informative_baseline_relative_norm"],
        "semantic_gate": {
            **protocol["semantic_prediction_gate"],
            "natural_competition_gap_min": 0.01,
            "recipient_and_donor_target_ids_must_differ": True,
        },
        "group_layer_gate": {
            "informative_pair_rate_min": protocol["discovery_local_edge_gate"]
            ["informative_pair_rate_min"],
            "direct_attention_pair_pass_rate_min": protocol[
                "discovery_local_edge_gate"
            ]["pair_pass_rate_min"],
            "direct_attention_median_recovery_min": protocol[
                "discovery_local_edge_gate"
            ]["median_state_recovery_min"],
            "semantic_informative_pair_rate_min": protocol[
                "discovery_local_edge_gate"
            ]["informative_pair_rate_min"],
            "semantic_positive_shift_rate_min": protocol["semantic_prediction_gate"]
            ["pair_positive_shift_rate_min"],
            "semantic_median_recovery_min": protocol["semantic_prediction_gate"]
            ["median_normalized_competition_recovery_min"],
            "all_fields_required": True,
        },
        "model_surface_layer_gate": {
            "qualified_group_rate_min": protocol["discovery_local_edge_gate"]
            ["qualified_discovery_group_rate_min"],
            "true_minus_each_control_median_recovery_min": protocol[
                "counterfactual_controls"
            ]["true_minus_each_control_median_recovery_min"],
            "true_minus_each_control_pair_pass_rate_min": protocol[
                "counterfactual_controls"
            ]["true_minus_each_control_pair_pass_rate_min"],
            "true_minus_each_control_semantic_recovery_min": protocol[
                "semantic_prediction_gate"
            ]["improvement_over_each_control_min"],
            "all_eight_controls_required_separately": True,
        },
        "selection_rule": {
            "eligible_layer": (
                "all_group_layer_fields_and_all_control_separations_pass"
            ),
            "selected_layer": "earliest_eligible_layer",
            "if_none": "no_candidate_for_model_surface",
            "relative_depth_zones": {
                "early": [0.0, 0.3333333333],
                "middle": [0.3333333333, 0.6666666667],
                "late": [0.6666666667, 1.0],
            },
            "crossmodel_candidate": (
                "same_surface_same_edge_all_three_models_and_same_relative_depth_zone"
            ),
            "no_posthoc_layer_rotation_mapping_or_threshold_change": True,
        },
        "storage_contract": {
            "per_pair_rows_private": True,
            "per_group_layer_control_rows_private": True,
            "public_outputs_only_aggregate_nonsemantic_ids": True,
            "raw_qkv_vectors_persisted": False,
            "one_group_processed_at_a_time": True,
        },
        "authorization": {
            "discovery_collection": True,
            "calibration_collection": False,
            "physical_holdout": False,
            "head_channel_neuron_scan": False,
        },
        "claim_boundary": {
            "architecture_residual_addition_is_language_edge": False,
            "direct_child_recovery_without_controls_is_functional_edge": False,
            "logit_lens_shift_is_generated_answer_change": False,
        },
    }
    write_json(OUT / "phase401_local_edge_execution_freeze.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
