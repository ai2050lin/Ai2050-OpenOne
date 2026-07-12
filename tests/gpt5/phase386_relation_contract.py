#!/usr/bin/env python3
"""Freeze Phase386 relation extraction and calibration gates before analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"


def main() -> None:
    audit = json.loads(
        (OUT / "phase386_discovery_collection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not audit["authorization"]["discovery_relation_extraction"]:
        raise RuntimeError("Phase386 discovery relation extraction is not authorized")
    payload = {
        "schema_version": "60.7.0",
        "phase_id": "Phase386-RelationContract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_discovery_relation_values_read": True,
        "denominator": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "mechanisms": [
                "relation_binding",
                "entity_recency",
                "field_extraction",
            ],
            "discovery_groups_per_mechanism": 8,
            "calibration_groups_per_mechanism": 4,
            "conditions": [
                "A_operation_lex_x",
                "B_control_lex_x",
                "C_operation_lex_y",
                "D_control_lex_y",
            ],
            "semantic_coordinates": [
                "source_encoded",
                "query_integrated",
                "pre_decision",
                "target_encoded",
                "post_decision_next_token",
            ],
            "depth_bin_count": 6,
        },
        "vector_families": {
            "layer_input": "exact residual stream before the block",
            "attention_output": "exact residual write after output projection",
            "mlp_output": "exact residual write after down projection",
            "layer_output": "exact residual stream after the block",
            "attention_head_state": (
                "all heads and all source positions summed only by the model's "
                "native attention rule, before output projection"
            ),
            "mlp_channel_product": (
                "all MLP channel coefficients before down projection; no top-k"
            ),
        },
        "contrast": {
            "lex_x": "delta_x = vector(A_operation_lex_x) - vector(B_control_lex_x)",
            "lex_y": "delta_y = vector(C_operation_lex_y) - vector(D_control_lex_y)",
            "lexical_replication": "cosine(delta_x, delta_y)",
            "zero_norm_cosine": 0.0,
            "pairwise_gram_materialized": False,
            "top_k_used": False,
        },
        "relations": {
            "axis": "adjacent_semantic_coordinate_at_fixed_exact_layer",
            "transitions": [
                ["source_encoded", "query_integrated"],
                ["query_integrated", "pre_decision"],
                ["pre_decision", "target_encoded"],
                ["target_encoded", "post_decision_next_token"],
            ],
            "causal_claim_allowed": False,
            "descriptive_physical_relation_claim_allowed": True,
        },
        "discovery_gate_per_model_exact_layer": {
            "complete_group_count": 8,
            "nonzero_effect_group_rate_min": 0.875,
            "median_relation_cosine_lex_x_min": 0.15,
            "median_relation_cosine_lex_y_min": 0.15,
            "positive_relation_group_rate_lex_x_min": 0.75,
            "positive_relation_group_rate_lex_y_min": 0.75,
            "median_source_lexical_replication_min": 0.10,
            "median_target_lexical_replication_min": 0.10,
        },
        "crossmodel_freeze": {
            "all_three_models_must_have_a_passing_exact_layer_in_same_depth_bin": True,
            "model_layer_selection": "earliest_passing_exact_layer_in_depth_bin",
            "weighted_or_composite_score_used": False,
            "physical_holdout_used": False,
        },
        "calibration_prediction": {
            "method": (
                "for each calibration source contrast, choose the discovery source "
                "contrast with highest cosine and copy its paired target contrast"
            ),
            "fitted_linear_operator": False,
            "calibration_relation_median_min": 0.10,
            "calibration_positive_group_rate_min": 0.75,
            "prediction_control_margin_min": 0.10,
            "controls": {
                "shuffled_pair": "rotate discovery target pairing by one group",
                "wrong_time": {
                    "source_encoded->query_integrated": "post_decision_next_token",
                    "query_integrated->pre_decision": "source_encoded",
                    "pre_decision->target_encoded": "query_integrated",
                    "target_encoded->post_decision_next_token": "source_encoded",
                },
                "wrong_depth": (
                    "same target coordinate at a layer displaced by max(2, "
                    "floor(layer_count/5)), clipped away from the candidate layer"
                ),
            },
            "predictive_relation_path_gate": (
                "relation replication and all three prediction-control advantages "
                "must pass separately for both lexical variants in all three models"
            ),
        },
        "claim_boundary": {
            "relation_similarity_is_causality": False,
            "nearest_neighbor_prediction_is_mechanistic_closure": False,
            "passing_parent_relation_proves_single_neuron_path": False,
            "small_model_architecture_may_bias_distribution": True,
        },
        "authorization": {
            "run_discovery_relation_extraction": True,
            "run_calibration_before_candidate_freeze": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
        },
    }
    path = OUT / "phase386_relation_contract.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
