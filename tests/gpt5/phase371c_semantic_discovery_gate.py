#!/usr/bin/env python3
"""Freeze semantic discovery gates after blind rows are sealed and before mapping."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
AUDIT = PHASE371 / "phase371c_blind_vector_contrast/phase371c_blind_contrast_audit.json"
OUT = PHASE371 / "phase371c_semantic_discovery_gate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "47.19.0",
        "phase_id": "Phase371C-Discovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_route_replication_and_control_gates_before_mapping_anonymous_slots_to_A_B_C_D",
        "authorization_basis": {
            "blind_rows_valid": audit["valid"],
            "blind_route_hash": audit["sealed_hashes"]["route_rows"],
            "blind_vocab_hash": audit["sealed_hashes"]["vocab_rows"],
            "audit_hash": sha256_file(AUDIT),
        },
        "row_gate": {
            "exact_difference_norm_strictly_positive": True,
            "signed_cosine_to_source_output_difference_strictly_positive": True,
            "child_parent_inner_product_share_strictly_positive": True,
            "adjacent_persistence_must_exceed_wrong_depth": True,
            "adjacent_persistence_must_exceed_wrong_role": True,
            "adjacent_persistence_must_exceed_time_shuffle": True,
            "post_hoc_numeric_margin": 0.0,
            "weighted_scalar_score": False,
        },
        "within_group_gate": {
            "A_B_and_C_D_rows_both_must_pass": True,
            "mean_A_B_C_D_adjacent_persistence_must_exceed_other_four_pairings": True,
            "both_lexical_slots_required": True,
            "single_pair_success_sufficient": False,
        },
        "replication_gate": {
            "minimum_independent_discovery_groups_per_model_mechanism": 8,
            "registered_eligible_discovery_groups_per_mechanism": 11,
            "same_canonical_time_depth_role_route_required_within_model": True,
            "absolute_layer_number_required_cross_model": False,
            "normalized_partition_index_used": True,
        },
        "cross_model_gate": {
            "level1": "one_model_passes_group_replication_gate",
            "level2": "glm4_and_at_least_one_of_qwen3_or_deepseek7b_pass_same_canonical_route",
            "level3": "all_three_models_pass_same_canonical_route",
            "qwen3_plus_deepseek7b_without_glm4": "architecture_family_only",
        },
        "unresolved_full_gate": {
            "exact_history_residual_projection_required": True,
            "history_residual_available_in_blind_index": False,
            "exact_vector_replay_confirmation_required": True,
            "provisional_route_can_open_calibration": False,
        },
        "candidate_statuses": {
            "row_pass": "navigation_only",
            "group_replicated": "provisional_exact_confirmation_required",
            "level2_or_level3_before_history_gate": "provisional_cross_model_only",
            "calibratable_language_path": "requires_history_and_exact_replay_gates_not_yet_available",
        },
        "authorization": {
            "create_discovery_only_condition_key": audit["valid"],
            "run_semantic_mapping_on_discovery_rows": audit["valid"],
            "select_full_language_path": False,
            "open_internal_calibration": False,
            "open_physical": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
