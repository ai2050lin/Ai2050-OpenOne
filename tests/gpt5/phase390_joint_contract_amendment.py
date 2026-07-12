#!/usr/bin/env python3
"""Freeze Phase390 joint-state metrics after instrumentation and before discovery analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"


def main() -> None:
    audit = json.loads(
        (OUT / "phase390_instrument_audit_summary.json").read_text(encoding="utf-8")
    )
    if not audit["results"]["all_three_model_instruments_valid"]:
        raise RuntimeError("Phase390 instrument audit must pass before contract amendment")
    payload = {
        "schema_version": "64.5.0",
        "phase_id": "Phase390-JointContractAmendment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timing": "after_engineering_instrument_audit_before_discovery_analysis",
        "reason": (
            "Permuting labels before an unlabelled vector sum leaves the sum exactly "
            "unchanged. Replace mathematically inert permutation controls with explicit "
            "best-single, other-prefix, leave-one-role-out, and wrong-group controls."
        ),
        "retired_controls": {
            "source_role_permutation_of_unlabelled_sum": "exactly_invariant",
            "attention_head_permutation_of_unlabelled_sum": "exactly_invariant",
            "same_size_random_set_when_all_prefix_positions_are_selected": "same_set",
        },
        "replacement_objects": {
            "semantic_source_joint": (
                "union of entities, attributes_items, relations, query_keywords, "
                "and query_window; excludes other_causal_prefix"
            ),
            "other_prefix_baseline": "other_causal_prefix alone",
            "single_role_baseline": "best one preregistered semantic role",
            "single_head_baseline": "best one physical head after output projection",
            "leave_one_role_out_diagnostic": "semantic joint minus exactly one role",
            "wrong_group_control": "next sealed parallel group under deterministic order",
            "wrong_time_control": "post_decision_next_token terminal state",
        },
        "frozen_candidate_lattice": {
            "receiver_coordinates": ["query_integrated", "pre_decision"],
            "relative_depth_anchor_count": 8,
            "window_lengths": [1, 2, 4],
            "window_mapping": "round(anchor_fraction * (layer_count - window_length))",
            "duplicate_actual_windows_deduplicated": True,
            "crossmodel_match_key": [
                "receiver_coordinate",
                "relative_depth_anchor_index",
                "window_length",
            ],
        },
        "frozen_vectors": {
            "semantic_attention_joint": "sum of exact projected head-by-source writes",
            "full_block_window": "sum of exact attention_output plus mlp_output",
            "terminal_target": "final_layer target_encoded layer_output operation_minus_control",
            "wrong_time_target": (
                "final_layer post_decision_next_token layer_output operation_minus_control"
            ),
            "lexical_contrasts": ["A_operation_lex_x-B_control_lex_x", "C_operation_lex_y-D_control_lex_y"],
            "learned_predictor": None,
            "learned_basis": None,
        },
        "frozen_gates": {
            "median_min_correct_alignment": 0.10,
            "median_lexical_replication": 0.10,
            "median_correct_minus_wrong_group": 0.05,
            "median_correct_minus_wrong_time": 0.05,
            "median_multi_source_advantage_over_best_role": 0.05,
            "median_multi_source_advantage_over_other_prefix": 0.05,
            "median_multi_head_advantage_over_best_head": 0.05,
            "median_cross_layer_advantage_over_best_single_layer": 0.05,
            "minimum_discovery_group_support": 8,
            "minimum_calibration_group_support": 4,
            "minimum_physical_group_support": 4,
            "all_three_models_required_for_shared_candidate": True,
        },
        "interpretation": {
            "semantic_joint_pass": "predictive joint formation candidate only",
            "exact_attention_additivity": "not evidence of nonlinear synergy",
            "crosslayer_window_pass": "predictive accumulated write candidate only",
            "causal_path_before_parent_boundary_replay": False,
            "single_neuron_scan_authorized": False,
        },
    }
    (OUT / "phase390_joint_contract_amendment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
