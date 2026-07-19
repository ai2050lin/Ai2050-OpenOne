#!/usr/bin/env python3
"""Freeze the independent Phase563 multi-position attention reader test."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE559 = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PHASE560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
ANCHORS_PATH = PHASE559 / "phase559_path_anchor_registry.json"
REGISTRY_PATH = OUT_DIR / "phase563_multiposition_reader_candidate_registry.json"
CONTRACT_PATH = OUT_DIR / "phase563_multiposition_reader_frozen_contract.json"
USED_CONTRACTS = (
    PHASE560 / "phase560_semantic_color_unseen_frozen_contract.json",
    PHASE560 / "phase560_parent_decomposition_frozen_contract.json",
    OUT_DIR / "phase561_source_to_query_trace_frozen_contract.json",
)
CONDITIONS = (
    "same_case_restore",
    "paired_contrast_neutralize",
    "correct_paired_donor_replace",
    "wrong_depth_donor_replace",
    "wrong_position_donor_replace",
    "channel_roll_donor_replace",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    anchor_registry = read_json(ANCHORS_PATH)
    eligible = {
        row["anchor_id"]
        for row in anchor_registry["anchors"]
        if row["split"] == "unseen_recombination"
        and row["reserved_for_unseen_validation"]
    }
    used: set[str] = set()
    for path in USED_CONTRACTS:
        used.update(read_json(path)["selected_anchor_ids"])
    selected = sorted(eligible - used)
    if len(selected) != 15 or selected != [
        f"phase559_unseen_recombination_{index:03d}" for index in range(81, 96)
    ]:
        raise RuntimeError("Phase563 untouched unseen denominator drift")

    role_blocks = {
        "query_answer_roles": [
            "query_relation_end",
            "query_object_end",
            "answer_boundary",
        ],
        "all_semantic_roles": [
            "source_object_end",
            "source_color_end",
            "source_fact_end",
            "nontarget_fact_end",
            "query_relation_end",
            "query_object_end",
            "answer_boundary",
        ],
    }
    candidates = []
    for layer, wrong_depth in ((4, 22), (10, 28)):
        for block_name, semantic_roles in role_blocks.items():
            candidates.append({
                "candidate_id": f"qwen3__multiposition_reader__attention_L{layer}__{block_name}",
                "model": "qwen3",
                "layer": layer,
                "wrong_depth_control_layer": wrong_depth,
                "component": "attention_output",
                "position_block": block_name,
                "semantic_positions": semantic_roles,
                "wrong_position_transform": "one_token_left_per_role",
                "selection_source": "phase561_trace_and_phase562_single_position_rejection",
                "candidate_is_compute_edge": False,
            })
    recipient_case_count = len(selected) * 32
    registry = {
        "schema_version": "phase563_multiposition_reader_candidate_registry.v1",
        "phase_id": "Phase563",
        "created_at": now(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_data_disjoint_from_validation": True,
        "candidate_family_frozen_before_model_execution": True,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    contract = {
        "schema_version": "phase563_multiposition_reader_frozen_contract.v1",
        "phase_id": "Phase563",
        "created_at": now(),
        "model": "qwen3",
        "split": "unseen_recombination",
        "selected_anchor_ids": selected,
        "selected_anchor_count": len(selected),
        "prior_unseen_anchor_overlap_count": 0,
        "recipient_case_count": recipient_case_count,
        "candidate_count": len(candidates),
        "conditions": list(CONDITIONS),
        "expected_intervention_rows": recipient_case_count * len(candidates) * len(CONDITIONS),
        "validation_gate": {
            "same_case_max_absolute_switch_effect": 0.0001,
            "correct_donor_win_rate_min": 0.70,
            "minimum_factorial_cell_donor_win_rate": 0.50,
            "correct_donor_mean_switch_effect_min": 1.0,
            "paired_neutralize_mean_switch_effect_min": 0.50,
            "correct_minus_channel_roll_mean_switch_effect_min": 0.50,
            "correct_minus_wrong_position_mean_switch_effect_min": 0.50,
        },
        "evidence_policy": {
            "passing_block_is_sufficiency_not_compute_edge": True,
            "failure_closes_only_tested_attention_output_blocks": True,
            "full_sequence_or_full_residual_block_not_tested": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    write_json(REGISTRY_PATH, registry)
    write_json(CONTRACT_PATH, contract)
    print(json.dumps({
        "selected_anchor_count": len(selected),
        "recipient_case_count": recipient_case_count,
        "candidate_count": len(candidates),
        "expected_intervention_rows": contract["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
