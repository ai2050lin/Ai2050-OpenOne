#!/usr/bin/env python3
"""Analyze Phase561 causal onset and freeze independent reader candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PHASE560_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
ROWS_PATH = OUT_DIR / "phase561_source_to_query_trace_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase561_source_to_query_trace_execution_summary.json"
CONTRACT_PATH = OUT_DIR / "phase561_source_to_query_trace_frozen_contract.json"
ANALYSIS_PATH = OUT_DIR / "phase561_source_to_query_trace_analysis.json"
READER_REGISTRY = OUT_DIR / "phase562_reader_candidate_registry.json"
READER_CONTRACT = OUT_DIR / "phase562_reader_validation_frozen_contract.json"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = PARENT_DIR / "phase559_path_anchor_registry.json"
PHASE559_SCREEN = PARENT_DIR / "phase559_causal_screen_frozen_contract.json"
PHASE560_SCREEN = PHASE560_DIR / "phase560_semantic_color_screen_frozen_contract.json"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": row["layer"],
        "component": row["component"],
        "semantic_position": row["semantic_position"],
        "mean_causal_to_natural_norm_ratio": row["mean_causal_to_natural_norm_ratio"],
        "mean_causal_projection_to_natural": row["mean_causal_projection_to_natural"],
        "mean_causal_natural_direction_cosine": row["mean_causal_natural_direction_cosine"],
        "minimum_factorial_cell_causal_to_natural_norm_ratio": row[
            "minimum_factorial_cell_causal_to_natural_norm_ratio"
        ],
    }


def main() -> None:
    rows = read_jsonl(ROWS_PATH)
    execution = read_json(SUMMARY_PATH)
    contract = read_json(CONTRACT_PATH)
    if len(rows) != 288 or execution["status"] != "complete":
        raise RuntimeError("Phase561 trace is incomplete")
    threshold = contract["analysis_thresholds"]
    reports = {}
    for position in ("query_object_end", "answer_boundary"):
        position_rows = [
            row for row in rows
            if row["semantic_position"] == position and row["component"] == "attention_output"
        ]
        onset = next(
            row for row in sorted(position_rows, key=lambda row: int(row["layer"]))
            if row["mean_causal_to_natural_norm_ratio"] >= threshold[
                "causal_to_natural_norm_ratio_for_onset"
            ]
            and row["mean_causal_projection_to_natural"] >= threshold[
                "causal_projection_to_natural_for_onset"
            ]
        )
        stable = next(
            row for row in sorted(position_rows, key=lambda row: int(row["layer"]))
            if row["mean_causal_projection_to_natural"] >= 0.40
            and row["minimum_factorial_cell_causal_to_natural_norm_ratio"] >= 0.80
        )
        peak = max(position_rows, key=lambda row: row["mean_causal_projection_to_natural"])
        reports[position] = {
            "first_causal_onset": compact(onset),
            "first_stable_integration": compact(stable),
            "peak_projection": compact(peak),
            "trajectory": [compact(row) for row in sorted(position_rows, key=lambda row: int(row["layer"]))],
        }
    pre_source_rows = [
        row for row in rows
        if int(row["layer"]) <= 3 and row["mean_causal_delta_norm"] != 0.0
    ]
    candidates = [
        {
            "candidate_id": "qwen3__reader_query_onset__attention_L4",
            "model": "qwen3",
            "boundary": "query_reader_onset",
            "semantic_position": "query_object_end",
            "wrong_position_control": "source_color_end",
            "component": "attention_output",
            "zone": "early",
            "layer": 4,
            "wrong_depth_control_layer": 22,
            "selection_source": "phase561_first_causal_onset",
            "candidate_is_compute_edge": False,
        },
        {
            "candidate_id": "qwen3__reader_answer_onset__attention_L4",
            "model": "qwen3",
            "boundary": "answer_reader_onset",
            "semantic_position": "answer_boundary",
            "wrong_position_control": "query_object_end",
            "component": "attention_output",
            "zone": "early",
            "layer": 4,
            "wrong_depth_control_layer": 22,
            "selection_source": "phase561_first_causal_onset",
            "candidate_is_compute_edge": False,
        },
        {
            "candidate_id": "qwen3__reader_answer_stable__attention_L10",
            "model": "qwen3",
            "boundary": "answer_reader_stable",
            "semantic_position": "answer_boundary",
            "wrong_position_control": "query_object_end",
            "component": "attention_output",
            "zone": "middle",
            "layer": 10,
            "wrong_depth_control_layer": 28,
            "selection_source": "phase561_first_stable_integration",
            "candidate_is_compute_edge": False,
        },
    ]
    used = set(read_json(PHASE559_SCREEN)["selected_anchor_ids"])
    used.update(read_json(PHASE560_SCREEN)["selected_anchor_ids"])
    anchor_registry = read_json(ANCHORS_PATH)
    eligible = {
        row["anchor_id"] for row in anchor_registry["anchors"]
        if row["split"] == "path_confirmation" and row["authorized_for_internal_collection"]
    }
    selected = sorted(eligible - used)
    if len(selected) != 16:
        raise RuntimeError("Phase562 confirmation remainder drift")
    reader_registry = {
        "schema_version": "phase562_reader_candidate_registry.v1",
        "phase_id": "Phase562",
        "created_at": now(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_data_disjoint_from_validation": True,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    reader_contract = {
        "schema_version": "phase562_reader_validation_frozen_contract.v1",
        "phase_id": "Phase562",
        "created_at": now(),
        "model": "qwen3",
        "split": "path_confirmation",
        "selected_anchor_ids": selected,
        "selected_anchor_count": len(selected),
        "prior_confirmation_anchor_overlap_count": 0,
        "recipient_case_count": len(selected) * 32,
        "candidate_count": len(candidates),
        "conditions": list(CONDITIONS),
        "expected_intervention_rows": len(selected) * 32 * len(candidates) * len(CONDITIONS),
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
            "trajectory_onset_does_not_prejudge_reader_validation": True,
            "failed_single_position_reader_closes_static_reader_candidate": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    analysis = {
        "schema_version": "phase561_source_to_query_trace_analysis.v1",
        "phase_id": "Phase561",
        "created_at": now(),
        "source_patch_donor_win_rate": execution["source_patch_donor_win_rate"],
        "zero_effect_through_source_layer": not pre_source_rows,
        "position_reports": reports,
        "first_query_causal_onset": reports["query_object_end"]["first_causal_onset"],
        "reader_compute_edge_confirmed": False,
        "phase562_candidate_registry_path": str(READER_REGISTRY.relative_to(ROOT)),
        "sealed_split_read": False,
    }
    write_json(READER_REGISTRY, reader_registry)
    write_json(READER_CONTRACT, reader_contract)
    write_json(ANALYSIS_PATH, analysis)
    print(json.dumps({
        "source_patch_donor_win_rate": execution["source_patch_donor_win_rate"],
        "zero_effect_through_source_layer": not pre_source_rows,
        "query_onset": reports["query_object_end"]["first_causal_onset"],
        "answer_onset": reports["answer_boundary"]["first_causal_onset"],
        "answer_stable": reports["answer_boundary"]["first_stable_integration"],
        "phase562_validation_anchor_count": len(selected),
        "phase562_expected_rows": reader_contract["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
