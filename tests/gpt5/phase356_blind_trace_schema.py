#!/usr/bin/env python3
"""Build a label-blind, loss-audited coarse trace skeleton from Phase354."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"
OUT = ROOT / "tests/gpt5/result/phase356_blind_neural_path_cartography"
ROUND_NAME = "coarse_trace_feasibility"
PHASE = "Phase356"
SCHEMA_VERSION = "32.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
BLIND_SALT = "phase356-blind-case-v1"
BANNED_DISCOVERY_KEYS = {
    "case_id", "semantic_case_id", "contract_group_id", "model", "family_id",
    "mechanism_id", "target", "target_aliases", "distractors", "operation_demanded",
    "contrast_condition", "lexical_set", "template_id", "signed_target_cosine",
    "signed_best_competitor_cosine", "signed_competition_margin", "actual_token_id",
    "expected_token_id", "token_matches_expected", "rollout_semantic_correct",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def blind_id(case_id: str) -> str:
    digest = hashlib.sha256(f"{BLIND_SALT}:{case_id}".encode()).hexdigest()[:24]
    return f"blind_{digest}"


def build() -> dict[str, Any]:
    out = OUT / ROUND_NAME
    cases = read_jsonl(SOURCE / "phase354_registered_cases.jsonl")
    opaque = {row["case_id"]: blind_id(row["case_id"]) for row in cases}
    label_rows = [{
        "blind_case_id": opaque[row["case_id"]],
        "source_case_id": row["case_id"],
        "model": row["model"], "family_id": row["family_id"],
        "mechanism_id": row["mechanism_id"], "split": row["split"],
        "template_id": row["template_id"], "contrast_condition": row["contrast_condition"],
        "operation_demanded": row["operation_demanded"], "item_index": row["item_index"],
    } for row in cases]
    write_jsonl(out / "sealed_labels" / "phase356_private_label_key.jsonl", label_rows)

    aggregates: dict[tuple[Any, ...], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
    )
    total_rows = free_rows = teacher_rows = nonfinite_rows = 0
    for model in MODELS:
        path = SOURCE / "models" / model / "phase354_semantic_time_rows.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                if row["trajectory_mode"] != "free_rollout":
                    teacher_rows += 1
                    continue
                free_rows += 1
                if not row["finite"] or row["component_l2_norm"] is None:
                    nonfinite_rows += 1
                    continue
                key = (
                    opaque[row["case_id"]],
                    "blind_discovery" if row["split"] == "physical_discovery" else "blind_calibration",
                    row["semantic_step"], row["semantic_step_count"], row["semantic_time_rho"],
                    row["component"], row["depth_bin"], row["position_role"],
                )
                value = float(row["component_l2_norm"])
                bucket = aggregates[key]
                bucket["count"] += 1
                bucket["sum"] += value
                bucket["min"] = min(bucket["min"], value)
                bucket["max"] = max(bucket["max"], value)

    skeleton = []
    for key, bucket in aggregates.items():
        blind_case, split, step, step_count, rho, component, depth, role = key
        count = int(bucket["count"])
        skeleton.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "blind_case_id": blind_case, "blind_split": split,
            "trajectory_mode": "free_rollout", "generation_step": step,
            "generation_step_count": step_count, "generation_progress": rho,
            "step_role": "start" if step == 0 else "continuation",
            "component": component, "relative_depth": depth, "position_role": role,
            "mean_component_l2_norm": round(bucket["sum"] / count, 7),
            "min_component_l2_norm": round(bucket["min"], 7),
            "max_component_l2_norm": round(bucket["max"], 7),
            "source_trace_row_count": count, "finite": True,
        })
    skeleton.sort(key=lambda row: (
        row["blind_case_id"], row["generation_step"], row["component"],
        row["position_role"], row["relative_depth"],
    ))
    write_jsonl(out / "phase356_blind_skeleton_rows.jsonl", skeleton)

    leaked_keys = sorted({key for row in skeleton for key in row if key in BANNED_DISCOVERY_KEYS})
    consumed_rows = sum(row["source_trace_row_count"] for row in skeleton)
    schema = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
        "trajectory_object": "coarse_label_blind_free_rollout_skeleton",
        "resolutions": {
            "r0_coarse_skeleton": {"status": "available", "fields": sorted(skeleton[0])},
            "r1_balanced_neuron_shards": {"status": "not_executed"},
            "r2_pre_registered_full_neuron_anchors": {"status": "not_executed"},
            "r3_causal_deep_audit": {"status": "forbidden_in_phase356"},
        },
        "discovery_forbidden_fields": sorted(BANNED_DISCOVERY_KEYS),
        "source_measurements_used": ["component_l2_norm"],
        "source_measurements_intentionally_excluded": [
            "target_direction", "competitor_direction", "token_identity", "behavior_label",
            "family_label", "mechanism_label", "model_identity",
        ],
    }
    write_json(out / "phase356_trace_schema.json", schema)
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(cases), "source_raw_trace_row_count": total_rows,
            "source_free_rollout_row_count": free_rows, "source_teacher_forced_row_count": teacher_rows,
            "blind_skeleton_row_count": len(skeleton), "consumed_free_rollout_row_count": consumed_rows,
            "nonfinite_free_rollout_row_count": nonfinite_rows,
        },
        "quality": {
            "source_row_conservation_valid": consumed_rows + nonfinite_rows == free_rows,
            "label_leakage_key_count": len(leaked_keys), "leaked_keys": leaked_keys,
            "blind_schema_valid": not leaked_keys,
            "exact_residual_reconstruction_available": False,
            "raw_vector_reconstruction_available": False,
            "attention_connection_weights_available": False,
            "normalization_state_available": False,
            "full_neuron_coverage_available": False,
            "balanced_neuron_sharding_executed": False,
            "full_neuron_anchor_executed": False,
        },
        "claim_boundary": {
            "full_trace_atlas_completed": False,
            "coarse_skeleton_is_reconstructable_full_computation": False,
            "coarse_skeleton_is_valid_for_blind_feasibility_pilot": not leaked_keys and consumed_rows > 0,
            "teacher_forced_used_for_blind_discovery": False,
            "physical_heldout_revealed": False,
            "causal_intervention_executed": False,
            "single_unit_causal_count": 0,
        },
        "next_decision": "run_blind_coarse_motif_feasibility" if not leaked_keys else "repair_label_leakage",
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_json(out / "phase356_schema_quality_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
