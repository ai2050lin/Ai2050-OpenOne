#!/usr/bin/env python3
"""Freeze the Phase391 local direct-parent graph contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P390 = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
OUT = ROOT / "tests/gpt5/result/phase391_local_parent_graph"


def main() -> None:
    phase390 = json.loads(
        (P390 / "phase390_discovery_candidate_freeze.json").read_text(encoding="utf-8")
    )
    if phase390["denominator"]["passing_crossmodel_candidate_count"] != 0:
        raise RuntimeError("Phase391 fallback requires the Phase390 global gate to be closed")
    payload = {
        "schema_version": "65.0.0",
        "phase_id": "Phase391-LocalParentContract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "map replicated direct-parent source-role and head contributions without "
            "assuming that a residual direction is preserved across downstream nonlinear layers"
        ),
        "evidence_reuse": {
            "discovery_source": "Phase390 discovery exact ledgers",
            "discovery_status": "algorithm_development_only",
            "calibration_source": "Phase390 sealed calibration groups, internal data unopened",
            "physical_source": "Phase390 sealed physical groups, internal data unopened",
            "behavior_denominator_changed": False,
            "physical_holdout_opened_before_contract": False,
        },
        "direct_parent_identity": (
            "delta_attention_output equals delta_semantic_source_write plus "
            "delta_other_causal_prefix_write within reconstruction tolerance"
        ),
        "semantic_source_roles": [
            "entities",
            "attributes_items",
            "relations",
            "query_keywords",
            "query_window",
        ],
        "receivers": ["query_integrated", "pre_decision"],
        "relative_depth_anchor_count": 8,
        "frozen_metrics": {
            "share": "dot(child,parent)/(dot(parent,parent)+epsilon)",
            "semantic_share": "share(sum_semantic_roles, delta_attention_output)",
            "other_share": "share(other_causal_prefix, delta_attention_output)",
            "fixed_role_share": "share(one fixed role, delta_attention_output)",
            "fixed_head_share": "share(one fixed projected head, delta_attention_output)",
            "lexical_replication": "cosine(delta_semantic_x, delta_semantic_y)",
            "attention_mlp_compensation": "cosine(delta_attention, delta_mlp)",
            "event_cancellation": (
                "norm(delta_attention+delta_mlp)/(norm(delta_attention)+norm(delta_mlp)+epsilon)"
            ),
        },
        "discovery_selection": {
            "best_role": "one role maximizing median minimum x/y share across all discovery groups",
            "best_head": "one head maximizing median minimum x/y share across all discovery groups",
            "participating_role_threshold": 0.05,
            "participating_head_threshold": 0.02,
            "posthoc_per_group_best_role_or_head_forbidden": True,
        },
        "frozen_gates": {
            "median_semantic_share": 0.10,
            "median_semantic_minus_other_share": 0.05,
            "median_lexical_replication": 0.10,
            "median_joint_advantage_over_fixed_role": 0.05,
            "median_joint_advantage_over_fixed_head": 0.05,
            "minimum_participating_roles": 2,
            "minimum_participating_heads": 2,
            "minimum_discovery_support_groups": 8,
            "minimum_calibration_support_groups": 4,
            "minimum_physical_support_groups": 4,
            "all_three_models_required_for_shared_layout": True,
        },
        "calibration_and_physical": {
            "receiver_frozen": True,
            "relative_depth_anchor_frozen": True,
            "model_specific_layer_frozen": True,
            "model_specific_best_role_frozen": True,
            "model_specific_best_head_frozen": True,
            "thresholds_frozen": True,
            "failed_candidate_replacement_allowed": False,
        },
        "claim_boundary": {
            "local_parent_share_is_terminal_prediction": False,
            "local_parent_share_is_causal_language_function": False,
            "multiple_positive_contributors_prove_nonlinear_synergy": False,
            "physical_replication_authorizes_neuron_scan": False,
            "language_encoding_closed": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase391_protocol.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
