#!/usr/bin/env python3
"""Independent zero-output and result audit for Phase1206."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import phase1206_qwen3_object_attribute_causal_transfer as run


EXPECTED_CONDITION_IDS = (
    "active_target_full",
    "active_target_half",
    "matched_null_target",
    "surface_only_target",
    "semantic_neighbor_target",
    "random_target_r0",
    "random_target_r1",
    "random_target_r2",
    "random_target_r3",
    "zero_target",
    "active_answer_prefix",
    "active_query_value",
    "active_preband_generation",
)
EXPECTED_PRIMARY_CONTROLS = (
    "matched_null_target",
    "surface_only_target",
    "semantic_neighbor_target",
    "random_target_r0",
    "random_target_r1",
    "random_target_r2",
    "random_target_r3",
    "zero_target",
)
EXPECTED_THRESHOLDS = {
    "finite_fraction": 1.0,
    "baseline_behavior_accuracy": 1.0,
    "full_donor_behavior_accuracy": 1.0,
    "positive_donor_margin_shift_fraction": 0.95,
    "donor_choice_fraction": 0.80,
    "minimum_median_transfer_fraction": 0.50,
    "active_beats_all_primary_controls_fraction": 0.75,
    "minimum_median_active_minus_max_control_shift": 0.10,
    "minimum_each_direction_donor_choice_fraction": 0.75,
    "zero_patch_max_abs_logit_drift": 1e-4,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate(value: dict[str, Any], key: str) -> None:
    if digest({name: item for name, item in value.items() if name != key}) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def source_hashes() -> dict[str, str]:
    return {
        "main": run.sha256_file(Path(run.__file__).resolve()),
        "audit": run.sha256_file(Path(__file__).resolve()),
        "runner": run.sha256_file(run.RUNNER_SCRIPT),
    }


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preexecution(write_output: bool) -> dict[str, Any]:
    if write_output and run.PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1206 preexecution audit already exists")
    protocol = run.read_json(run.PROTOCOL_PATH)
    validate(protocol, "protocol_digest")
    final1205 = run.read_json(run.SOURCE1205_FINAL)
    audit1205 = run.read_json(run.SOURCE1205_AUDIT)
    validate(final1205, "final_digest")
    validate(audit1205, "audit_digest")
    pairs = run.read_jsonl(run.SOURCE_PAIR_MANIFEST)
    active = [row for row in pairs if row["panel"] == "active"]
    checks: list[dict[str, Any]] = []
    add(checks, "phase", protocol.get("phase") == 1206)
    add(checks, "schema", protocol.get("schema_version") == "phase1206.qwen3_object_attribute_causal_transfer.v1")
    add(checks, "source_hashes", protocol.get("source_hashes") == source_hashes())
    add(checks, "phase1205_final", final1205["final_digest"] == run.EXPECTED_PHASE1205_FINAL_DIGEST)
    add(checks, "phase1205_audit", audit1205["audit_digest"] == run.EXPECTED_PHASE1205_AUDIT_DIGEST and audit1205["gate_pass"] is True)
    add(checks, "phase1205_hidden_pass", final1205["hidden_specificity_gate"] is True)
    add(checks, "frozen_depth_25", final1205["selected_depth"] == protocol["target"]["depth"] == 25)
    add(checks, "frozen_role", protocol["target"]["role"] == "generation_boundary")
    add(checks, "not_refit", protocol["target"]["not_refit_in_phase1206"] is True)
    add(checks, "qwen_only", protocol["scope"]["model"] == "qwen3" and protocol["scope"]["model_specific_only"] is True)
    add(checks, "no_necessity_claim", protocol["scope"]["causal_necessity_claim"] is False)
    add(checks, "no_natural_claim", protocol["scope"]["natural_use_claim"] is False)
    add(checks, "no_cross_claim", protocol["scope"]["cross_model_claim"] is False)
    add(checks, "no_closure_claim", protocol["scope"]["mechanism_closure_claim"] is False)
    add(checks, "strict_fp16", protocol["model"]["precision"] == "FP16" and protocol["model"]["quantization"] == "none" and protocol["model"]["placement"] == "full_cuda")
    add(checks, "condition_ids", tuple(item["id"] for item in protocol["conditions"]) == EXPECTED_CONDITION_IDS)
    add(checks, "primary_controls", tuple(protocol["primary_controls"]) == EXPECTED_PRIMARY_CONTROLS)
    add(checks, "thresholds", protocol["primary_gate"]["thresholds"] == EXPECTED_THRESHOLDS)
    add(checks, "role_depth_controls_descriptive", protocol["primary_gate"]["role_and_depth_controls_are_descriptive"] is True)
    add(checks, "pair_file_hash", protocol["upstream"]["pair_manifest_file_sha256"] == run.sha256_file(run.SOURCE_PAIR_MANIFEST))
    add(checks, "pair_semantic_digest", protocol["upstream"]["pair_manifest_digest"] == digest(pairs))
    add(checks, "pair_count_2016", len(pairs) == 2016)
    add(checks, "active_count_504", len(active) == 504)
    add(checks, "quartets", len(active) * 4 == len(pairs) and all({row["panel"] for row in pairs if row["group_id"] == item["group_id"]} == set(run.phase1205.PANELS) for item in active))
    add(checks, "expected_records", protocol["counts"]["expected_intervention_records"] == 504 * 2 * len(EXPECTED_CONDITION_IDS))
    add(checks, "no_vector_output", not run.VECTOR_PATH.exists())
    add(checks, "no_raw_output", not run.RAW_PATH.exists())
    add(checks, "no_run_summary", not run.RUN_SUMMARY_PATH.exists())
    add(checks, "no_verdict", not run.VERDICT_PATH.exists())
    add(checks, "no_final", not run.FINAL_PATH.exists())
    output: dict[str, Any] = {
        "phase": 1206,
        "audit_stage": "preexecution",
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "model_outputs_observed": 0,
        "authorization": {
            "qwen3_causal_transfer_run": all(item["pass"] for item in checks),
            "head_or_neuron_search": False,
            "cross_model_claim": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write_output:
        if not output["gate_pass"]:
            raise RuntimeError([item["name"] for item in checks if not item["pass"]])
        write(run.PREAUDIT_PATH, output)
    return output


def median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.median(np.asarray(items, dtype=np.float64))) if items else 0.0


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    base = labels.index(str(row["recipient_gold"]))
    donor = labels.index(str(row["donor_gold"]))
    before = np.asarray(row["recipient_unhooked_scores"], dtype=np.float64)
    full = np.asarray(row["donor_unhooked_scores"], dtype=np.float64)
    after = np.asarray(row["patched_scores"], dtype=np.float64)
    before_margin = float(before[donor] - before[base])
    full_margin = float(full[donor] - full[base])
    after_margin = float(after[donor] - after[base])
    shift = after_margin - before_margin
    full_shift = full_margin - before_margin
    return {
        **row,
        "recipient_donor_margin": before_margin,
        "full_donor_margin": full_margin,
        "patched_donor_margin": after_margin,
        "donor_margin_shift": shift,
        "full_unhooked_margin_shift": full_shift,
        "transfer_fraction": shift / (full_shift + run.EPSILON),
        "positive_shift": shift > 0,
        "donor_choice": row["patched_prediction"] == row["donor_gold"],
        "recipient_correct": row["recipient_unhooked_prediction"] == row["recipient_gold"],
        "donor_correct": row["donor_unhooked_prediction"] == row["donor_gold"],
    }


def split_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    lookup = {(row["group_id"], row["recipient_state"], row["condition"]): row for row in members}
    targets = [row for row in members if row["condition"] == "active_target_full"]
    advantages: list[float] = []
    for row in targets:
        control = max(float(lookup[(row["group_id"], row["recipient_state"], name)]["donor_margin_shift"]) for name in EXPECTED_PRIMARY_CONTROLS)
        advantages.append(float(row["donor_margin_shift"]) - control)
    directions = {}
    for state in (0, 1):
        subset = [row for row in targets if int(row["recipient_state"]) == state]
        directions[f"state{state}_to_state{1-state}"] = sum(bool(row["donor_choice"]) for row in subset) / max(len(subset), 1)
    result = {
        "split": split,
        "target_record_count": len(targets),
        "finite_fraction": sum(bool(row["recipient_unhooked_finite"] and row["donor_unhooked_finite"] and row["patched_finite"]) for row in targets) / max(len(targets), 1),
        "baseline_behavior_accuracy": sum(bool(row["recipient_correct"]) for row in targets) / max(len(targets), 1),
        "full_donor_behavior_accuracy": sum(bool(row["donor_correct"]) for row in targets) / max(len(targets), 1),
        "positive_donor_margin_shift_fraction": sum(bool(row["positive_shift"]) for row in targets) / max(len(targets), 1),
        "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in targets) / max(len(targets), 1),
        "median_donor_margin_shift": median(row["donor_margin_shift"] for row in targets),
        "median_transfer_fraction": median(row["transfer_fraction"] for row in targets),
        "active_beats_all_primary_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_active_minus_max_control_shift": median(advantages),
        "direction_donor_choice_fraction": directions,
    }
    threshold = EXPECTED_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= threshold["finite_fraction"]
        and result["baseline_behavior_accuracy"] >= threshold["baseline_behavior_accuracy"]
        and result["full_donor_behavior_accuracy"] >= threshold["full_donor_behavior_accuracy"]
        and result["positive_donor_margin_shift_fraction"] >= threshold["positive_donor_margin_shift_fraction"]
        and result["donor_choice_fraction"] >= threshold["donor_choice_fraction"]
        and result["median_transfer_fraction"] >= threshold["minimum_median_transfer_fraction"]
        and result["active_beats_all_primary_controls_fraction"] >= threshold["active_beats_all_primary_controls_fraction"]
        and result["median_active_minus_max_control_shift"] >= threshold["minimum_median_active_minus_max_control_shift"]
        and min(directions.values()) >= threshold["minimum_each_direction_donor_choice_fraction"]
    )
    return result


def summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for condition in EXPECTED_CONDITION_IDS:
        members = [row for row in rows if row["condition"] == condition]
        output[condition] = {
            "count": len(members),
            "median_donor_margin_shift": median(row["donor_margin_shift"] for row in members),
            "median_transfer_fraction": median(row["transfer_fraction"] for row in members),
            "positive_shift_fraction": sum(bool(row["positive_shift"]) for row in members) / max(len(members), 1),
            "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in members) / max(len(members), 1),
            "median_delta_l2": median(row["delta_l2"] for row in members),
        }
    return output


def result(write_output: bool) -> dict[str, Any]:
    if write_output and run.RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1206 result audit already exists")
    protocol = run.verify_protocol()
    preaudit = run.read_json(run.PREAUDIT_PATH)
    summary = run.read_json(run.RUN_SUMMARY_PATH)
    verdict = run.read_json(run.VERDICT_PATH)
    validate(preaudit, "audit_digest")
    validate(summary, "summary_digest")
    validate(verdict, "verdict_digest")
    raw = run.read_jsonl_gz(run.RAW_PATH)
    rows = [enrich(row) for row in raw]
    checks: list[dict[str, Any]] = []
    add(checks, "preexecution", preaudit["gate_pass"] is True)
    add(checks, "protocol_links", summary["protocol_digest"] == verdict["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == source_hashes())
    add(checks, "raw_file_hash", summary["raw_file_sha256"] == run.sha256_file(run.RAW_PATH))
    add(checks, "raw_digest", summary["raw_digest"] == digest(raw))
    add(checks, "vector_file_hash", summary["vector_file_sha256"] == run.sha256_file(run.VECTOR_PATH))
    add(checks, "record_count", len(raw) == summary["record_count"] == protocol["counts"]["expected_intervention_records"])
    unique_ids = {row["record_id"] for row in raw}
    add(checks, "record_ids_unique", len(unique_ids) == len(raw))
    expected_per_condition = protocol["counts"]["active_pairs"] * 2
    add(checks, "condition_completeness", all(sum(row["condition"] == name for row in raw) == expected_per_condition for name in EXPECTED_CONDITION_IDS))
    add(checks, "finite_raw_scores", all(row["recipient_unhooked_finite"] and row["donor_unhooked_finite"] and row["patched_finite"] and np.isfinite(row["recipient_unhooked_scores"]).all() and np.isfinite(row["donor_unhooked_scores"]).all() and np.isfinite(row["patched_scores"]).all() for row in raw))
    add(checks, "baseline_predictions_correct", all(row["recipient_unhooked_prediction"] == row["recipient_gold"] and row["donor_unhooked_prediction"] == row["donor_gold"] for row in raw))
    precision = summary["precision_audit"]
    add(checks, "strict_fp16", precision["has_fp16_parameters"] and not precision["has_bf16_parameters"] and not precision["has_quantized_modules"] and set(precision["parameter_dtypes"]) == {"float16"})
    placement = summary["placement"]
    add(checks, "full_cuda", placement["placement"] == "full_cuda" and placement["devices"] == ["cuda:0"] and placement["quantization"] == "none")
    with np.load(run.VECTOR_PATH, allow_pickle=False) as arrays:
        expected_vector_shape = [2016, 2, 2560]
        vector_names = ("d24_generation_boundary", "d25_generation_boundary", "d25_answer_prefix", "d25_query_value")
        add(checks, "vector_shapes", all(list(arrays[name].shape) == expected_vector_shape for name in vector_names))
        add(checks, "baseline_array_shapes", list(arrays["baseline_scores"].shape) == [2016, 2, 3] and list(arrays["baseline_finite"].shape) == [2016, 2])
        add(checks, "arrays_finite", all(np.isfinite(arrays[name]).all() for name in (*vector_names, "baseline_scores")))
        add(checks, "baseline_finite_array", bool(arrays["baseline_finite"].all()))
    recomputed_splits = {split: split_metrics(rows, split) for split in run.SPLITS}
    recomputed_summary = summaries(rows)
    zero = [row for row in rows if row["condition"] == "zero_target"]
    zero_drift = max(abs(float(patched) - float(base)) for row in zero for patched, base in zip(row["patched_scores"], row["recipient_unhooked_scores"]))
    identity_pass = zero_drift <= EXPECTED_THRESHOLDS["zero_patch_max_abs_logit_drift"]
    gate = bool(identity_pass and all(value["pass"] for value in recomputed_splits.values()))
    add(checks, "split_metrics", verdict["split_metrics"] == recomputed_splits)
    add(checks, "condition_summary", verdict["condition_summary"] == recomputed_summary)
    add(checks, "zero_drift", math.isclose(verdict["zero_patch_max_abs_logit_drift"], zero_drift, rel_tol=0.0, abs_tol=1e-12))
    add(checks, "identity_pass", verdict["zero_patch_identity_pass"] is identity_pass)
    add(checks, "causal_gate", verdict["causal_transfer_gate"] is gate)
    expected_status = "qwen3_controlled_causal_transfer_qualified" if gate else "qwen3_controlled_causal_transfer_not_qualified"
    add(checks, "status", verdict["status"] == expected_status)
    add(checks, "claim_qwen_only", verdict["claim_boundary"]["qwen3_model_specific"] is True)
    add(checks, "no_necessity", verdict["claim_boundary"]["causal_necessity"] is False)
    add(checks, "no_natural", verdict["claim_boundary"]["natural_use"] is False)
    add(checks, "no_cross", verdict["claim_boundary"]["cross_model"] is False)
    add(checks, "no_closure", verdict["claim_boundary"]["mechanism_closure"] is False)
    output: dict[str, Any] = {
        "phase": 1206,
        "audit_stage": "result",
        "protocol_digest": protocol["protocol_digest"],
        "run_summary_digest": summary["summary_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "independent_recomputation": {
            "split_metrics": recomputed_splits,
            "zero_patch_max_abs_logit_drift": zero_drift,
            "zero_patch_identity_pass": identity_pass,
            "causal_transfer_gate": gate,
        },
        "claim_boundary": {
            "qwen3_controlled_causal_transfer_only": True,
            "necessity": False,
            "natural_use": False,
            "cross_model": False,
            "mechanism_closure": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write_output:
        if not output["gate_pass"]:
            raise RuntimeError([item["name"] for item in checks if not item["pass"]])
        write(run.RESULT_AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preexecution", "result"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = preexecution(args.write) if args.stage == "preexecution" else result(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
