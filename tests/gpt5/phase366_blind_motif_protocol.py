#!/usr/bin/env python3
"""Freeze a label-blind motif protocol before scaling the dynamic-flow collection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
P363_SPLIT = ROOT / "tests/gpt5/result/phase363_temporal_hypotheses/strict_temporal_innovation_formulas/phase363_formula_split_rows.jsonl"
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_motif_protocol"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    execution = [
        row for row in read_jsonl(P362 / "private" / "phase362_execution_cases.jsonl")
        if row["phase362_split"] == "independent_calibration"
    ]
    split_rows = read_jsonl(P363_SPLIT)
    split_by_group = {(row["model"], row["group_id"]): row["split"] for row in split_rows}
    blind_groups = []
    private_key = []
    seen = set()
    for row in sorted(execution, key=lambda item: (item["model"], item["phase362_group_id"])):
        key = (row["model"], row["phase362_group_id"])
        if key in seen:
            continue
        seen.add(key)
        scientific_split = {
            "formula_train": "blind_motif_discovery",
            "formula_validation": "blind_motif_calibration",
        }[split_by_group[key]]
        anonymous_group = "motif_" + hashlib.sha256(f"phase366:{row['anonymous_model_id']}:{row['phase362_group_id']}".encode()).hexdigest()[:20]
        blind_groups.append({
            "anonymous_model_id": row["anonymous_model_id"],
            "anonymous_group_id": anonymous_group,
            "source_group_id_private_ref": "private_" + hashlib.sha256(row["phase362_group_id"].encode()).hexdigest()[:16],
            "scientific_split": scientific_split,
            "condition_count": 4,
            "generation_time_count": 3,
        })
        private_key.append({
            "anonymous_model_id": row["anonymous_model_id"],
            "anonymous_group_id": anonymous_group,
            "model": row["model"], "source_group_id": row["phase362_group_id"],
            "family_id": row["family_id"], "mechanism_id": row["mechanism_id"],
            "scientific_split": scientific_split,
        })
    if len(blind_groups) != 72:
        raise RuntimeError(f"Expected 72 groups, got {len(blind_groups)}")

    protocol = {
        "schema_version": "43.0.0", "phase_id": "Phase366", "created_at": now(),
        "objective": "discover_recurrent_typed_flow_motifs_without_semantic_or_target_labels",
        "denominator": {
            "model_count": 3, "admitted_mechanism_count_private": 4,
            "independent_group_count": 72, "case_count": 288,
            "condition_count_per_group": 4, "generation_time_count": 3,
            "blind_discovery_group_count": 48, "blind_calibration_group_count": 24,
            "physical_confirmation_case_count": 96,
        },
        "event_descriptor": {
            "allowed": [
                "event_type", "edge_type", "relative_layer", "generation_time",
                "source_role_alias", "receiver_role_alias", "raw_vector_reference",
                "vector_norm", "signed_alignment_to_local_write", "local_layer_order",
                "branch_and_merge_degree",
            ],
            "forbidden": [
                "real_model_name", "family", "mechanism", "condition_semantics",
                "correct_answer", "target_token", "target_margin", "target_rank",
                "historical_candidate_score",
            ],
            "raw_vectors_remain_authoritative": True,
        },
        "condition_handling": {
            "average_four_conditions_before_discovery": False,
            "anonymous_condition_slots_retained": True,
            "condition_semantics_revealed_during_discovery": False,
            "direct_graph_subtraction": False,
            "typed_event_alignment_before_any_contrast": True,
            "unmatched_events_retained": True,
        },
        "common_backbone": {
            "raw_graph_deleted_after_residualization": False,
            "raw_and_residual_views_both_retained": True,
            "backbone_may_use_semantic_labels": False,
            "common_motif_classified_only_after_recurrence": True,
        },
        "motif_enumeration": {
            "top_k_component_selection": False,
            "fixed_mad_only_threshold": False,
            "path_lengths": [2, 3, 4, 6, 8],
            "path_length_persistence_counts_as_independent_replication": False,
            "typed_contiguous_edges_required": True,
            "time_order_must_be_preserved": True,
            "shuffled_time_control_required": True,
            "matched_size_random_bundle_control_required": True,
        },
        "threshold_custodian": {
            "fixed_execution_repeat_floor": 0.0,
            "reconstruction_floor_source": "phase365_engineering_collection_max_errors",
            "same_condition_template_floor_required": True,
            "condition_effect_may_be_counted_as_noise": False,
            "threshold_values_frozen_before_motif_scoring": True,
        },
        "prediction_gate": {
            "targets_during_blind_stage": ["next_typed_events", "next_layer_local_write"],
            "target_specific_competition_added_only_after_motif_freeze": True,
            "strong_baselines": [
                "model_relative_layer_role_common_transition", "time_order_shuffle",
                "matched_size_random_bundle", "current_bundle_persistence",
            ],
            "independent_group_is_analysis_unit": True,
        },
        "cross_model_equivalence": {
            "same_layer_number_required": False,
            "same_head_or_channel_id_required": False,
            "compare": [
                "typed_event_order", "source_receiver_roles", "relative_depth",
                "branch_merge_structure", "future_transition_signature",
            ],
        },
        "gates": {
            "engineering_96_cases_can_produce_language_claim": False,
            "all_288_cases_must_be_collected_before_scientific_scoring": True,
            "blind_motif_must_replicate_in_calibration_groups": True,
            "labels_revealed_only_after_motif_and_threshold_freeze": True,
            "physical_confirmation_opened": False,
            "causal_intervention": False,
        },
        "stop_rules": [
            "if_only_common_architecture_motifs_recur_record_backbone_only",
            "if_motifs_do_not_predict_future_events_do_not_reveal_function_labels",
            "if_calibration_fails_do_not_change_thresholds_or_path_lengths",
            "if_model_specific_only_do_not_enter_cross_model_theory",
            "never_open_physical_confirmation_to_repair_motif_discovery",
        ],
        "authorization": {
            "collect_remaining_192_dynamic_bundles": True,
            "scientific_motif_scoring_now": False,
            "condition_or_target_label_reveal_now": False,
            "physical_confirmation_now": False,
        },
        "next_decision": "collect_remaining_192_cases_with_frozen_phase365_schema_before_blind_scoring",
    }
    write_jsonl(OUT / "phase366_blind_group_registry.jsonl", blind_groups)
    write_jsonl(OUT / "private" / "phase366_group_label_key.jsonl", private_key)
    write_json(OUT / "phase366_blind_motif_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
