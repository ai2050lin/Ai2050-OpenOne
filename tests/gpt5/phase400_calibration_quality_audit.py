#!/usr/bin/env python3
"""Close Phase400 validation safely when a frozen replay gate fails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase400_partial_order_common import MODELS, OUT, now, read_json, read_jsonl, write_json


def shard_roots(model: str) -> list[Path]:
    root = OUT / "dynamic_trace/calibration/private/models" / model / "shards"
    paths = sorted(path for path in root.glob("shard_*") if path.is_dir())
    if not paths:
        raise RuntimeError(f"No Phase400 calibration shards for {model}")
    return paths


def main() -> None:
    protocol = read_json(OUT / "phase400_dynamic_trace_protocol.json")
    diagnostic_path = (
        OUT
        / "dynamic_trace/calibration/private/phase400_failed_replay_diagnostic.json"
    )
    diagnostic = read_json(diagnostic_path) if diagnostic_path.is_file() else None
    audits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    shard_completes: list[dict[str, Any]] = []
    for model in MODELS:
        for root in shard_roots(model):
            complete = read_json(root / "complete.json")
            if complete["model"] != model or complete["stage"] != "calibration":
                raise RuntimeError(f"Mismatched Phase400 shard identity: {root}")
            shard_completes.append(complete)
            audits.extend(read_jsonl(root / "group_audit_rows.jsonl"))
            predictions.extend(read_jsonl(root / "case_prediction_rows.jsonl"))
    expected_cases = protocol["denominator"]["split_case_counts"]["calibration"]
    expected_groups = expected_cases // 16
    if len(predictions) != expected_cases or len(audits) != expected_groups:
        raise RuntimeError(
            f"Phase400 calibration audit denominator mismatch: "
            f"{len(predictions)}/{expected_cases} cases, {len(audits)}/{expected_groups} groups"
        )
    failed_groups = [row for row in audits if not row["quality_gate_pass"]]
    failed_cases = [
        row
        for row in predictions
        if not (
            row["first_answer_replay_match"]
            and row.get("target_completion_replay_match", True)
            and row.get("post_target_replay_match", True)
        )
    ]
    # Prediction rows expose only the first replay flag. Group audits carry all three counts.
    result = {
        "schema_version": "74.8.0",
        "phase_id": "Phase400-CalibrationCollectionQualityAudit",
        "created_at": now(),
        "status": "calibration_collection_invalid_frozen_replay_gate_failed",
        "denominator": {
            "model_count": len(MODELS),
            "shard_count": len(shard_completes),
            "case_count": len(predictions),
            "group_model_cell_count": len(audits),
            "quality_group_model_cell_count": sum(
                row["quality_gate_pass"] for row in audits
            ),
            "first_answer_replay_match_count": sum(
                row["first_answer_replay_match_count"] for row in audits
            ),
            "target_completion_replay_match_count": sum(
                row["target_completion_replay_match_count"] for row in audits
            ),
            "post_target_replay_match_count": sum(
                row["post_target_replay_match_count"] for row in audits
            ),
        },
        "max_errors": {
            "block_relative_error": max(row["max_block_relative_error"] for row in audits),
            "attention_replay_relative_error": max(
                row["max_attention_replay_relative_error"] for row in audits
            ),
            "probability_sum_absolute_error": max(
                row["max_probability_sum_error"] for row in audits
            ),
        },
        "failed_group_model_cells": [
            {
                "model": row["model"],
                "surface": row["surface_private"],
                "public_parallel_group_id": row[
                    "phase400_public_parallel_group_id"
                ],
                "first_answer_replay_match_count": row[
                    "first_answer_replay_match_count"
                ],
                "target_completion_replay_match_count": row[
                    "target_completion_replay_match_count"
                ],
                "post_target_replay_match_count": row[
                    "post_target_replay_match_count"
                ],
            }
            for row in failed_groups
        ],
        "diagnosis": {
            "numeric_conservation_gate_failed": any(
                row["max_block_relative_error"]
                > protocol["quality_gates"]["block_relative_error_max"]
                or row["max_attention_replay_relative_error"]
                > protocol["quality_gates"][
                    "attention_role_replay_relative_error_max"
                ]
                or row["max_probability_sum_error"]
                > protocol["quality_gates"][
                    "attention_probability_sum_absolute_error_max"
                ]
                for row in failed_groups
            ),
            "exact_first_answer_replay_gate_failed": any(
                row["first_answer_replay_match_count"] < row["case_count"]
                for row in failed_groups
            ),
            "exact_semantic_target_completion_gate_failed": any(
                row["target_completion_replay_match_count"] < row["case_count"]
                for row in failed_groups
            ),
            "exact_post_target_replay_gate_failed": any(
                row["post_target_replay_match_count"] < row["case_count"]
                for row in failed_groups
            ),
            "parent_capture_hooks_changed_current_single_case_top1": (
                not diagnostic["plain_vs_hooked_equal"] if diagnostic else None
            ),
            "current_single_case_plain_vs_hooked_invariance": (
                diagnostic["plain_vs_hooked_equal"] if diagnostic else None
            ),
            "batch_size_1_vs_8_first_token_invariance": (
                diagnostic["batch_size_1_vs_8_equal"] if diagnostic else None
            ),
            "cached_behavior_reproduced_at_original_batch_size": (
                diagnostic["cached_behavior_reproduced_now"] if diagnostic else None
            ),
            "full_behavior_to_trace_execution_invariance_established": False,
            "format_token_instability_observed": True,
        },
        "authorization": {
            "analyze_calibration_event_graph": False,
            "open_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "head_channel_or_neuron_scan": False,
        },
        "non_override_audit": {
            "failed_group_removed": False,
            "replacement_group_added": False,
            "quality_threshold_relaxed": False,
            "semantic_equivalence_used_to_override_exact_replay": False,
            "failed_attempt_hidden_by_rerun": False,
        },
        "claim_boundary": {
            "phase400_discovery_graph_is_calibration_confirmed": False,
            "calibration_failure_disproves_dynamic_events": False,
            "calibration_failure_is_zero_causal_effect": False,
        },
    }
    private = {
        "schema_version": "phase400.calibration_failure_private.v1",
        "created_at": result["created_at"],
        "failed_case_rows": failed_cases,
        "private_only": True,
    }
    calibration = {
        "schema_version": "74.8.0",
        "phase_id": "Phase400-PartialOrder-calibration-Validation",
        "created_at": result["created_at"],
        "stage": "calibration",
        "status": "not_analyzed_invalid_collection",
        "quality_audit": result,
        "cells": [],
        "crossmodel_surfaces": [],
        "results": {
            "validated_model_surface_cell_count": 0,
            "crossmodel_isomorphism_surface_count": 0,
            "prediction_pass_cell_count": 0,
        },
        "authorization": result["authorization"],
    }
    physical = {
        "schema_version": "74.8.0",
        "phase_id": "Phase400-PartialOrder-physical_holdout-Blocked",
        "created_at": result["created_at"],
        "stage": "physical_holdout",
        "status": "not_opened_due_to_calibration_collection_failure",
        "case_count_consumed": 0,
        "physical_holdout_remains_unopened": True,
        "authorization": {
            "run_joint_causal_intervention": False,
            "head_channel_or_neuron_scan": False,
        },
    }
    write_json(OUT / "phase400_calibration_collection_quality_audit.json", result)
    write_json(
        OUT / "dynamic_trace/calibration/private/phase400_failed_case_audit.json",
        private,
    )
    write_json(OUT / "phase400_partial_order_calibration.json", calibration)
    write_json(OUT / "phase400_partial_order_physical.json", physical)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
