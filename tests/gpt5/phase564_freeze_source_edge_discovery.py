#!/usr/bin/env python3
"""Freeze Phase564 aggregate source-edge discovery before intervention."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PROTOCOL_PATH = OUT_DIR / "phase564_frozen_protocol.json"
BEHAVIOR_SUMMARY_PATH = OUT_DIR / "phase564_edge_behavior_summary.json"
ANCHOR_REGISTRY_PATH = OUT_DIR / "phase564_edge_anchor_registry.json"
CONTRACT_PATH = OUT_DIR / "phase564_source_edge_discovery_frozen_contract.json"
CANDIDATE_REGISTRY_PATH = OUT_DIR / "phase564_source_edge_discovery_candidates.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
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
    behavior = read_json(BEHAVIOR_SUMMARY_PATH)
    anchors = read_json(ANCHOR_REGISTRY_PATH)
    if behavior["authorized_models"] != ["qwen3"] or anchors["authorized_models"] != ["qwen3"]:
        raise RuntimeError("Phase564 source-edge discovery requires Qwen3-only authorization")
    selected = sorted(
        row["anchor_id"] for row in anchors["anchors"]
        if row["model"] == "qwen3"
        and row["split"] == "edge_discovery"
        and row["authorized_for_internal_collection"]
    )
    expected_worlds = int(protocol["split_world_counts"]["edge_discovery"])
    if len(selected) != expected_worlds:
        raise RuntimeError("Phase564 edge-discovery all-correct world denominator drift")
    design = protocol["edge_design"]
    candidates = [
        {
            "candidate_id": f"phase564_L{layer}_{target_role}_from_source_color",
            "layer": int(layer),
            "target_role": target_role,
            "source_role": design["source_role"],
            "component": "aggregate_attention_source_contribution",
            "head_scope": "all_heads_summed",
            "selection_split": "edge_discovery",
            "compute_edge": False,
        }
        for layer in design["candidate_layers"]
        for target_role in design["target_roles"]
    ]
    registry = {
        "schema_version": "phase564_source_edge_discovery_candidates.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "candidate_family_frozen_before_intervention": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(CANDIDATE_REGISTRY_PATH, registry)
    conditions = list(design["discovery_conditions"])
    case_count = len(selected) * 32
    contract = {
        "schema_version": "phase564_source_edge_discovery_frozen_contract.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "model": "qwen3",
        "split": "edge_discovery",
        "selected_anchor_ids": selected,
        "world_count": len(selected),
        "recipient_case_count": case_count,
        "candidate_count": len(candidates),
        "conditions": conditions,
        "expected_intervention_rows": case_count * len(candidates) * len(conditions),
        "parent_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_behavior_summary_sha256": sha256_file(BEHAVIOR_SUMMARY_PATH),
        "parent_anchor_registry_sha256": sha256_file(ANCHOR_REGISTRY_PATH),
        "candidate_registry_sha256": sha256_file(CANDIDATE_REGISTRY_PATH),
        "reconstruction_relative_error_max": 0.01,
        "effect_baseline": "same_case_restore_from_same_fused_batch_shape",
        "selection_policy": {
            "rank_by": [
                "paired_donor_win_rate", "paired_donor_mean_effect", "removal_mean_damage"
            ],
            "maximum_confirmation_candidates": design["maximum_frozen_confirmation_candidates"],
            "confirmation_data_used_for_selection": False,
        },
        "evidence_policy": {
            "post_softmax_value_contribution_only": True,
            "key_or_attention_weight_mechanism_claimed": False,
            "same_case_restore_required": True,
            "sealed_split_read": False,
            "fine_scan_executed": False,
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
