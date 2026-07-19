#!/usr/bin/env python3
"""Freeze Phase565 full-residual multi-position operator validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
PARENT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PROTOCOL_PATH = PARENT_DIR / "phase564_frozen_protocol.json"
ANCHOR_REGISTRY_PATH = PARENT_DIR / "phase564_edge_anchor_registry.json"
EDGE_ANALYSIS_PATH = PARENT_DIR / "phase564_source_edge_confirmation_analysis.json"
CONTRACT_PATH = OUT_DIR / "phase565_residual_operator_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase565_residual_operator_candidates.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    anchors = read_json(ANCHOR_REGISTRY_PATH)
    edge_analysis = read_json(EDGE_ANALYSIS_PATH)
    if edge_analysis["confirmation_passing_candidate_count"] != 0:
        raise RuntimeError("Phase565 is allowed only after the frozen Phase564 edge stop")
    selected = sorted(
        row["anchor_id"] for row in anchors["anchors"]
        if row["model"] == "qwen3"
        and row["split"] == "edge_unseen"
        and row["authorized_for_internal_collection"]
    )
    if len(selected) != 62:
        raise RuntimeError("Phase565 expected the 62 untouched all-correct unseen worlds")
    semantic_roles = [
        "source_object_end", "source_color_end", "source_fact_end", "nontarget_fact_end",
        "query_relation_end", "query_object_end", "answer_boundary",
    ]
    candidates = []
    for layer in (4, 7, 10):
        wrong_depth = layer - 3 if layer >= 7 else layer + 3
        candidates.extend((
            {
                "candidate_id": f"phase565_L{layer}_semantic7_residual_block",
                "layer": layer,
                "wrong_depth_control_layer": wrong_depth,
                "position_block": "semantic7",
                "semantic_positions": semantic_roles,
                "component": "layer_output_residual",
                "scope": "typed_semantic_positions_only",
                "compute_edge": False,
            },
            {
                "candidate_id": f"phase565_L{layer}_full_sequence_residual_block",
                "layer": layer,
                "wrong_depth_control_layer": wrong_depth,
                "position_block": "full_sequence",
                "semantic_positions": [],
                "component": "layer_output_residual",
                "scope": "all_nonpadding_sequence_positions",
                "compute_edge": False,
            },
        ))
    registry = {
        "schema_version": "phase565_residual_operator_candidates.v1",
        "phase_id": "Phase565",
        "created_at": now(),
        "candidate_family_frozen_before_model_execution": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(CANDIDATES_PATH, registry)
    conditions = [
        "same_case_restore",
        "paired_contrast_neutralize",
        "paired_donor_residual_replace",
        "wrong_depth_donor_replace",
        "wrong_position_donor_replace",
        "channel_roll_donor_replace",
    ]
    case_count = len(selected) * 32
    contract = {
        "schema_version": "phase565_residual_operator_frozen_contract.v1",
        "phase_id": "Phase565",
        "created_at": now(),
        "model": "qwen3",
        "split": "edge_unseen",
        "selected_anchor_ids": selected,
        "world_count": len(selected),
        "recipient_case_count": case_count,
        "candidate_count": len(candidates),
        "conditions": conditions,
        "expected_intervention_rows": case_count * len(candidates) * len(conditions),
        "parent_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_anchor_registry_sha256": sha256_file(ANCHOR_REGISTRY_PATH),
        "parent_edge_analysis_sha256": sha256_file(EDGE_ANALYSIS_PATH),
        "candidate_registry_sha256": sha256_file(CANDIDATES_PATH),
        "effect_baseline": "same_case_restore_from_same_fused_batch_shape",
        "operator_gate": {
            "same_restore_max_abs_effect": 0.05,
            "paired_donor_win_rate_min": 0.80,
            "minimum_factorial_cell_donor_win_rate": 0.60,
            "paired_donor_mean_effect_min": 1.0,
            "paired_donor_must_exceed_midpoint_wrong_depth_wrong_position_and_roll": True,
        },
        "evidence_policy": {
            "distributed_state_sufficiency_only": True,
            "full_sequence_success_is_not_a_compute_edge": True,
            "natural_necessity_tested": False,
            "source_edge_tested_in_parent_phase": True,
            "fine_scan_executed": False,
            "sealed_split_read": False,
        },
    }
    write_json(CONTRACT_PATH, contract)
    print(json.dumps({
        "world_count": len(selected),
        "case_count": case_count,
        "candidate_count": len(candidates),
        "condition_count": len(conditions),
        "expected_rows": contract["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))
    return contract


if __name__ == "__main__":
    freeze()
