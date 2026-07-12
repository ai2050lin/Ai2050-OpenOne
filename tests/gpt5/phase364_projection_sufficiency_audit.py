#!/usr/bin/env python3
"""Audit observation sufficiency and freeze the next instrumentation boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
P362 = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
P361 = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
P363 = ROOT / "tests/gpt5/result/phase363_temporal_hypotheses/strict_temporal_innovation_formulas"
OUT = ROOT / "tests/gpt5/result/phase364_projection_sufficiency_audit/offline_projection_audit"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_value(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    _prefix, component, role = candidate["feature_id"].split("::", 2)
    return float(row["component_norms"][component][row["role_names"].index(role)])


def depth_match(row: dict[str, Any], depth: str) -> bool:
    value = float(row["relative_depth"])
    return (depth == "early" and value < 1 / 3) or (depth == "middle" and 1 / 3 <= value < 2 / 3) or (depth == "late" and value >= 2 / 3)


def main() -> None:
    candidate_source = P361 / "phase361_frozen_predictive_candidates.jsonl"
    audit_rows = read_jsonl(P362 / "phase362_frozen_candidate_audit_rows.jsonl")
    candidates = [row for row in audit_rows if row["b3_beats_all_alternatives_all_models"]]
    if len(candidates) != 7:
        raise RuntimeError(f"Expected seven Phase362 candidates, got {len(candidates)}")

    projection_rows = []
    for model in MODELS:
        ledger = read_jsonl(P362 / "models" / model / "phase362_generation_time_rows.jsonl")
        by_case_time: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in ledger:
            by_case_time[(row["blind_case_id"], int(row["generation_time"]))].append(row)
        for time in range(3):
            case_ids = sorted(case_id for case_id, candidate_time in by_case_time if candidate_time == time)
            matrix = np.array([
                [
                    np.mean([
                        feature_value(row, candidate)
                        for row in by_case_time[(case_id, time)]
                        if depth_match(row, candidate["depth_bin"])
                    ])
                    for candidate in candidates
                ]
                for case_id in case_ids
            ], dtype=np.float64)
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            singular = np.linalg.svd(centered, compute_uv=False)
            tolerance = max(centered.shape) * np.finfo(np.float64).eps * (float(singular.max()) if len(singular) else 0.0)
            duplicate_pairs = []
            for left in range(len(candidates)):
                for right in range(left + 1, len(candidates)):
                    equal = matrix[:, left] == matrix[:, right]
                    if bool(equal.any()):
                        duplicate_pairs.append({
                            "left_candidate_id": candidates[left]["candidate_id"],
                            "right_candidate_id": candidates[right]["candidate_id"],
                            "exact_equal_case_count": int(equal.sum()),
                            "exact_equal_fraction": round(float(equal.mean()), 9),
                            "all_cases_exact_equal": bool(equal.all()),
                        })
            projection_rows.append({
                "model": model,
                "generation_time": time,
                "case_count": len(case_ids),
                "input_feature_count": len(candidates),
                "centered_numeric_rank": int((singular > tolerance).sum()),
                "singular_values": singular.tolist(),
                "exact_duplicate_pairs": duplicate_pairs,
                "all_case_duplicate_pair_count": sum(row["all_cases_exact_equal"] for row in duplicate_pairs),
            })

    anchor_rows = []
    for model in MODELS:
        manifest_path = P362 / "sealed_anchors" / model / "manifest.json"
        manifest = read_json(manifest_path)
        first = manifest["files"][0]
        payload = torch.load(P362 / first["relative_path"], map_location="cpu", weights_only=True)
        keys = set(payload)
        vector_fields = {
            key: list(payload[key].shape)
            for key in (
                "layer_input", "input_norm_actual", "attention_output", "post_attention_state",
                "post_norm_actual", "mlp_output", "layer_output",
            ) if key in payload
        }
        attention_source_replayable = {
            "projected_value_states", "selected_attention_probabilities", "projected_head_outputs",
        }.issubset(keys)
        mlp_shard_replayable = "mlp_shard_contributions" in keys
        per_neuron_self_contained = {
            "saved_mlp_shard_activations", "saved_mlp_shard_channel_ids", "mlp_down_weight_columns",
        }.issubset(keys)
        anchor_rows.append({
            "model": model,
            "anchor_count": manifest["anchor_count"],
            "anchor_time_count": manifest["anchor_time_count"],
            "layer_file_count": manifest["layer_file_count"],
            "total_byte_count": manifest["total_byte_count"],
            "all_online_gates_pass": manifest["all_online_gates_pass"],
            "sample_relative_path": first["relative_path"],
            "sample_hidden_size": int(payload["layer_input"].shape[-1]),
            "saved_vector_fields": vector_fields,
            "attention_source_edge_offline_replayable": attention_source_replayable,
            "mlp_shard_write_offline_replayable": mlp_shard_replayable,
            "mlp_single_neuron_write_self_contained": per_neuron_self_contained,
            "missing_for_single_neuron_replay": [] if per_neuron_self_contained else ["mlp_down_weight_columns_or_frozen_weight_reference"],
            "dynamic_flow_bundle_schema_present": False,
        })
        del payload

    strict = read_json(P363 / "phase363_hypothesis_summary.json")
    observation_rows = [
        {
            "level": "P0_component_norm_skeleton",
            "availability": "all_288_discovery_cases_three_times",
            "reversible_to_parent": False,
            "future_state_sufficiency_identified": False,
            "structurally_noninjective": True,
            "reason": "vector_norm_and_depth_average_are_many_to_one",
        },
        {
            "level": "P1_component_full_vectors",
            "availability": "nine_anchor_cases_only",
            "reversible_to_parent": True,
            "future_state_sufficiency_identified": False,
            "structurally_noninjective": False,
            "reason": "format_replayable_but_denominator_is_engineering_only",
        },
        {
            "level": "P2_typed_source_and_channel_writes",
            "availability": "attention_source_and_mlp_shard_writes_on_nine_anchors",
            "reversible_to_parent": True,
            "future_state_sufficiency_identified": False,
            "structurally_noninjective": False,
            "reason": "single_neuron_mlp_writes_not_self_contained_and_bulk_cases_lack_source_edges",
        },
        {
            "level": "P3_dynamic_flow_bundles",
            "availability": "not_implemented",
            "reversible_to_parent": False,
            "future_state_sufficiency_identified": False,
            "structurally_noninjective": None,
            "reason": "cross_layer_cross_position_cross_time_bundle_schema_missing",
        },
    ]
    protocol = {
        "schema_version": "41.0.0",
        "phase_id": "Phase364-A",
        "created_at": now(),
        "observation_levels": [row["level"] for row in observation_rows],
        "phase365_fixed_engineering_denominator": {
            "model_count": 3,
            "admitted_mechanism_count": 4,
            "group_count_per_model_mechanism": 2,
            "condition_count_per_group": 4,
            "total_case_count": 96,
            "generation_time_count": 3,
        },
        "required_fields_before_execution": {
            "attention": ["source_position", "receiver_position", "head", "probability", "projected_value", "residual_write"],
            "mlp": ["channel_id", "gate_value", "up_value", "gate_up_product", "down_projection_reference", "residual_write"],
            "state": ["layer_input", "normalized_inputs", "post_attention_state", "layer_output"],
            "time": ["generation_time", "generated_prefix", "role_positions"],
            "quality": ["dtype_add_order", "component_conservation", "source_conservation", "shard_balance"],
        },
        "execution_order": ["qwen3", "glm4", "deepseek7b"],
        "semantic_labels_allowed_during_discovery": False,
        "top_k_selection_allowed": False,
        "new_model_execution_authorized": False,
        "authorization_blockers": [
            "model_specific_mlp_single_neuron_write_adapter_missing",
            "dynamic_flow_bundle_schema_not_implemented",
            "repeat_noise_threshold_for_collision_gate_missing",
        ],
    }
    all_time_zero_duplicate = [
        pair
        for row in projection_rows if row["generation_time"] == 0
        for pair in row["exact_duplicate_pairs"] if pair["all_cases_exact_equal"]
    ]
    summary = {
        "schema_version": "41.0.0",
        "phase_id": "Phase364-A",
        "created_at": now(),
        "denominator": {
            "model_count": 3,
            "discovery_case_count": 288,
            "generation_time_count": 3,
            "projection_matrix_count": len(projection_rows),
            "candidate_feature_count": len(candidates),
            "anchor_count": sum(row["anchor_count"] for row in anchor_rows),
            "anchor_layer_file_count": sum(row["layer_file_count"] for row in anchor_rows),
            "strict_formula_count_reused": strict["denominator"]["tested_formula_count"],
        },
        "frozen_inputs": {
            "phase361_candidate_sha256": sha256_file(candidate_source),
            "physical_confirmation_read": False,
            "new_model_execution_count": 0,
        },
        "results": {
            "p0_structurally_noninjective": True,
            "p0_time_zero_all_case_duplicate_pair_record_count": len(all_time_zero_duplicate),
            "p0_min_centered_numeric_rank": min(row["centered_numeric_rank"] for row in projection_rows),
            "p0_max_centered_numeric_rank": max(row["centered_numeric_rank"] for row in projection_rows),
            "strict_formula_survivor_count": strict["denominator"]["frozen_formula_count"],
            "attention_source_edge_anchor_model_count": sum(row["attention_source_edge_offline_replayable"] for row in anchor_rows),
            "mlp_shard_write_anchor_model_count": sum(row["mlp_shard_write_offline_replayable"] for row in anchor_rows),
            "mlp_single_neuron_self_contained_model_count": sum(row["mlp_single_neuron_write_self_contained"] for row in anchor_rows),
            "dynamic_flow_bundle_schema_model_count": sum(row["dynamic_flow_bundle_schema_present"] for row in anchor_rows),
        },
        "claim_boundary": {
            "p0_is_proven_sufficient_state": False,
            "p0_is_proven_insufficient_for_every_possible_nonlinear_mapping": False,
            "p0_cannot_be_assumed_sufficient": True,
            "p1_p2_language_mechanism_tested": False,
            "collision_rate_claim_available_without_rerun_noise": False,
            "language_encoding_closed": False,
            "intelligence_theory_closed": False,
        },
        "decision": "freeze_p0_as_lossy_skeleton_and_block_new_execution_until_p2_p3_instrumentation_is_complete",
        "next_action": "implement_and_unit_test_model_specific_mlp_write_adapters_and_dynamic_bundle_schema_without_running_models",
    }
    write_jsonl(OUT / "phase364_projection_rank_rows.jsonl", projection_rows)
    write_jsonl(OUT / "phase364_anchor_capability_rows.jsonl", anchor_rows)
    write_jsonl(OUT / "phase364_observation_level_rows.jsonl", observation_rows)
    write_json(OUT / "phase365_instrumentation_protocol.json", protocol)
    write_json(OUT / "phase364_projection_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
