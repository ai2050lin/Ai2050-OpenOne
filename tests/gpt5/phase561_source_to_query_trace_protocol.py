#!/usr/bin/env python3
"""Freeze a fresh causal propagation trace from Phase560's earliest source edge."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PHASE560_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = PARENT_DIR / "phase559_path_anchor_registry.json"
UNSEEN_CONTRACT = PHASE560_DIR / "phase560_semantic_color_unseen_frozen_contract.json"
PARENT_CONTRACT = PHASE560_DIR / "phase560_parent_decomposition_frozen_contract.json"
EDGES_PATH = PHASE560_DIR / "phase560_coarse_source_color_edges.jsonl"
PARENT_ANALYSIS = PHASE560_DIR / "phase560_parent_decomposition_analysis.json"
CONTRACT_PATH = OUT_DIR / "phase561_source_to_query_trace_frozen_contract.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def freeze() -> dict[str, Any]:
    edges = read_jsonl(EDGES_PATH)
    parent_analysis = read_json(PARENT_ANALYSIS)
    early = [row for row in edges if int(row["source_layer"]) == 3]
    if len(early) != 1 or not parent_analysis["all_tested_layers_residual_carry_dominant"]:
        raise RuntimeError("Phase561 requires the qualified residual-carry L3 source edge")
    used = set(read_json(UNSEEN_CONTRACT)["selected_anchor_ids"])
    used.update(read_json(PARENT_CONTRACT)["selected_anchor_ids"])
    anchor_registry = read_json(ANCHORS_PATH)
    valid = {
        row["anchor_id"] for row in anchor_registry["anchors"]
        if row["split"] == "unseen_recombination" and row["reserved_for_unseen_validation"]
    }
    path_rows = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == "unseen_recombination" and row["anchor_id"] in valid
    ]
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        worlds[row["anchor_id"]].append(row)
    groups: dict[str, list[str]] = defaultdict(list)
    for anchor_id, rows in worlds.items():
        group = f"{rows[0]['color_regime']}::{rows[0]['color_a']}|{rows[0]['color_b']}"
        groups[group].append(anchor_id)
    selected = sorted(
        next(anchor for anchor in sorted(groups[group]) if anchor not in used)
        for group in sorted(groups)
    )
    if len(selected) != 20 or set(selected) & used:
        raise RuntimeError("Phase561 trace anchors are not independent")
    contract = {
        "schema_version": "phase561_source_to_query_trace_frozen_contract.v1",
        "phase_id": "Phase561",
        "created_at": now(),
        "model": "qwen3",
        "split": "unseen_recombination",
        "selected_anchor_ids": selected,
        "selected_anchor_count": len(selected),
        "prior_phase560_anchor_overlap_count": 0,
        "case_count": len(selected) * 32,
        "counterfactual_pair_count": len(selected) * 16,
        "source_intervention": {
            "layer": 3,
            "component": "layer_output",
            "semantic_position": "source_color_end",
            "replacement": "paired_binding_donor",
        },
        "traced_positions": ["query_object_end", "answer_boundary"],
        "traced_components": ["layer_input", "attention_output", "mlp_output", "layer_output"],
        "analysis_thresholds": {
            "causal_to_natural_norm_ratio_for_onset": 0.05,
            "causal_projection_to_natural_for_onset": 0.05,
            "source_patch_donor_win_rate_min": 0.90,
        },
        "evidence_policy": {
            "trace_is_intervention_conditioned_observation": True,
            "reader_compute_edge_not_claimed_without_reader_intervention": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    write_json(CONTRACT_PATH, contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return contract


if __name__ == "__main__":
    freeze()
