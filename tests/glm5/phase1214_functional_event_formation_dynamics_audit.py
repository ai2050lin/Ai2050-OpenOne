#!/usr/bin/env python3
"""Independent preregistration and CUDA result audit for Phase 1214."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1213_free_transformer_behavioral_quotient_event as p1213  # noqa: E402
import phase1214_functional_event_formation_dynamics as p1214  # noqa: E402


def write_audit(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["audit_digest"] = p1214.digest(value)
    p1214.write_json(path, value)
    return value


def close(left: Any, right: Any, tolerance: float = 1.0e-7) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def preaudit() -> dict[str, Any]:
    protocol = p1214.verify_protocol()
    old_tasks = {
        (task["name"], tuple(task["coefficients"]), task["bias"], task["lexicon_seed"])
        for tasks in p1213.TASKS.values()
        for task in tasks
    }
    new_tasks = {
        (task["name"], tuple(task["coefficients"]), task["bias"], task["lexicon_seed"])
        for tasks in p1214.TASKS.values()
        for task in tasks
    }
    old_architectures = {
        tuple(p1213.asdict(config).values())
        for values in p1213.ARCHITECTURES.values()
        for config in values.values()
    }
    new_architectures = {
        tuple(p1214.asdict(config).values())
        for values in p1214.ARCHITECTURES.values()
        for config in values.values()
    }
    all_seeds = [
        p1214.model_seed(split, task_index, architecture_index, replicate)
        for split in p1214.TASKS
        for task_index in range(len(p1214.TASKS[split]))
        for architecture_index in range(len(p1214.ARCHITECTURES[split]))
        for replicate in range(p1214.REPLICATES)
    ]
    checkpoint_steps = protocol["checkpoint_steps"]
    checks: dict[str, bool] = {
        "protocol_digest_valid": True,
        "source_gate_all_passed": all(protocol["source_gate"].values()),
        "script_hashes_frozen": protocol["script_hashes"] == p1214.script_hashes(),
        "phase1213_tasks_disjoint": old_tasks.isdisjoint(new_tasks),
        "phase1213_architectures_disjoint": old_architectures.isdisjoint(new_architectures),
        "formal_seeds_unique": len(all_seeds) == len(set(all_seeds)) == 24,
        "discovery_confirmation_seed_disjoint": set(all_seeds[:12]).isdisjoint(all_seeds[12:]),
        "formal_run_count_24": protocol["formal_run_count"] == 24,
        "twelve_runs_per_split": all(
            len(p1214.TASKS[split]) * len(p1214.ARCHITECTURES[split]) * p1214.REPLICATES == 12
            for split in p1214.TASKS
        ),
        "three_tasks_per_split": all(len(value) == 3 for value in p1214.TASKS.values()),
        "two_architectures_per_split": all(len(value) == 2 for value in p1214.ARCHITECTURES.values()),
        "fixed_horizon_2400": p1214.TRAINING["maximum_steps"] == 2400,
        "no_early_stopping": p1214.TRAINING["no_early_stopping"] is True,
        "checkpoint_grid_includes_zero": checkpoint_steps[0] == 0,
        "checkpoint_grid_includes_endpoint": checkpoint_steps[-1] == 2400,
        "checkpoint_grid_count_25": len(checkpoint_steps) == 25,
        "checkpoint_grid_uniform_100": all(right - left == 100 for left, right in zip(checkpoint_steps, checkpoint_steps[1:])),
        "two_consecutive_passes": p1214.TRAINING["required_consecutive_passes"] == 2,
        "stability_threshold_frozen": p1214.TRAINING["post_formation_stability_min"] == 0.80,
        "strict_behavior_accuracy": p1214.THRESHOLDS["behavior_accuracy_min"] == 1.0,
        "strict_behavior_probability": p1214.THRESHOLDS["behavior_minimum_probability_min"] == 0.95,
        "camera_validation_frozen": p1214.THRESHOLDS["camera_validation_accuracy_min"] == 0.95,
        "camera_holdout_frozen": p1214.THRESHOLDS["camera_holdout_accuracy_min"] == 0.95,
        "random_camera_control": p1214.THRESHOLDS["random_camera_accuracy_max"] == 0.25,
        "same_patch_gate": p1214.THRESHOLDS["patch_same_preservation_min"] == 0.98,
        "wrong_patch_gate": p1214.THRESHOLDS["patch_wrong_transfer_min"] == 0.90,
        "behavior_breadth_gate": p1214.THRESHOLDS["behavior_models_per_split_min"] == 8,
        "conditional_event_gate": p1214.THRESHOLDS["conditional_event_fraction_min"] == 0.80,
        "coupling_window_frozen": p1214.THRESHOLDS["temporal_coupling_window_steps"] == 200,
        "right_censoring_explicit": "right censoring" in protocol["scientific_object"]["censoring"],
        "event_unauthorized_when_behavior_censored": "unauthorized" in protocol["scientific_object"]["censoring"],
        "target_is_external_contract": "input-output contract" in protocol["scientific_object"]["target"],
        "parameter_proxy_labeled_descriptive": "descriptive" in protocol["normalization"]["parameter_token_proxy"],
        "event_is_seven_part_conjunction": len(protocol["event_conjunction"]) == 7,
        "independent_audit_required": protocol["claim_gate"]["independent_result_audit_required"] is True,
        "phase1213_continuation_forbidden": "continue any Phase1213 formal run" in protocol["forbidden"],
        "horizon_extension_forbidden": "extend the 2400-step horizon after seeing outcomes" in protocol["forbidden"],
        "censored_selection_forbidden": "select only behavior-qualified runs and claim a population law" in protocol["forbidden"],
        "llm_transfer_forbidden": any("Qwen3" in value for value in protocol["forbidden"]),
        "necessity_not_claimable": any("necessity" in value for value in protocol["forbidden"]),
        "balanced_task_splits": all(
            len(p1214.split_combinations(task)[0]) == 384 and len(p1214.split_combinations(task)[1]) == 128
            for tasks in p1214.TASKS.values()
            for task in tasks
        ),
        "all_contracts_have_512_classes": all(
            len(set(p1214.expected_signatures(task).values())) == 512
            for tasks in p1214.TASKS.values()
            for task in tasks
        ),
        "no_formal_metrics_before_preaudit": not any(p1214.OUT_ROOT.glob("runs/**/metrics.json")),
        "no_final_before_preaudit": not p1214.FINAL_PATH.exists(),
        "cuda_available": torch.cuda.is_available(),
    }
    result = {
        "phase": p1214.PHASE,
        "kind": "independent_zero_output_preaudit",
        "created_at": p1214.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    return write_audit(p1214.PREAUDIT_PATH, result)


def find_checkpoint(row: dict[str, Any], step: int) -> Path:
    manifest = {int(value["step"]): value for value in row["checkpoint_manifest"]}
    return ROOT / manifest[int(step)]["path"]


def recompute_behavior_checkpoint(
    row: dict[str, Any],
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    model = p1214.load_checkpoint(find_checkpoint(row, step), device)
    train, holdout = p1214.split_combinations(row["task"])
    result = {
        "train": p1214.evaluate_behavior(model, row["task"], train),
        "holdout": p1214.evaluate_behavior(model, row["task"], holdout),
    }
    result["behavior_pass"] = p1214.behavior_pass(result["train"]) and p1214.behavior_pass(result["holdout"])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def recompute_event_checkpoint(
    row: dict[str, Any],
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    model = p1214.load_checkpoint(find_checkpoint(row, step), device)
    result = p1214.scan_checkpoint(model, row["task"], row["initial_camera_controls"], step)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final_audit() -> dict[str, Any]:
    protocol = p1214.verify_protocol()
    preaudit_value = p1214.read_json(p1214.PREAUDIT_PATH)
    p1214.validate_digest(preaudit_value, "audit_digest")
    final = p1214.read_json(p1214.FINAL_PATH)
    p1214.validate_digest(final, "final_digest")
    expected_steps = protocol["checkpoint_steps"]
    checks: dict[str, bool] = {
        "protocol_digest_matches": final["protocol_digest"] == protocol["protocol_digest"],
        "preaudit_digest_matches": final["preaudit_digest"] == preaudit_value["audit_digest"],
        "preaudit_passed": preaudit_value["all_checks_passed"] is True,
        "manifest_has_24_runs": len(final["run_manifest"]) == 24,
        "manifest_run_ids_unique": len({value["run_id"] for value in final["run_manifest"]}) == 24,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for result audit")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    per_run: dict[str, dict[str, bool]] = {}
    initial_control_sentinels: set[tuple[str, str]] = set()
    for manifest in final["run_manifest"]:
        metrics_path = ROOT / manifest["metrics"]
        row = p1214.read_json(metrics_path)
        p1214.validate_digest(row, "metrics_digest")
        row_checks: dict[str, bool] = {
            "metrics_hash": p1214.sha256_file(metrics_path) == manifest["metrics_sha256"],
            "metrics_digest": row["metrics_digest"] == manifest["metrics_digest"],
            "checkpoint_count_25": len(row["checkpoint_manifest"]) == manifest["checkpoint_count"] == 25,
            "trajectory_count_25": len(row["trajectory"]) == 25,
            "trajectory_steps_frozen": [value["step"] for value in row["trajectory"]] == expected_steps,
            "checkpoint_steps_frozen": [value["step"] for value in row["checkpoint_manifest"]] == expected_steps,
            "contract_512": row["functional_contract"]["class_count"] == 512,
            "parameter_count_positive": row["parameter_count"] > 0,
        }
        for checkpoint in row["checkpoint_manifest"]:
            path = ROOT / checkpoint["path"]
            if not path.exists() or p1214.sha256_file(path) != checkpoint["sha256"]:
                row_checks["all_checkpoint_hashes"] = False
                break
        else:
            row_checks["all_checkpoint_hashes"] = True

        recomputed_formation = p1214.summarize_trajectory(row["trajectory"], int(row["parameter_count"]))
        row_checks["formation_summary_recomputed"] = close(recomputed_formation, row["formation"])
        endpoint_stored = row["trajectory"][-1]
        endpoint = recompute_behavior_checkpoint(row, int(p1214.TRAINING["maximum_steps"]), device)
        row_checks["endpoint_train_recomputed"] = close(endpoint["train"], endpoint_stored["train_behavior"])
        row_checks["endpoint_holdout_recomputed"] = close(endpoint["holdout"], endpoint_stored["holdout_behavior"])
        row_checks["endpoint_behavior_flag_recomputed"] = endpoint["behavior_pass"] is endpoint_stored["behavior_pass"]

        tau_b = row["formation"]["tau_B"]
        if tau_b["status"] == "observed":
            for offset, label in ((0, "tauB"), (int(p1214.TRAINING["evaluation_interval"]), "tauB_next")):
                step = int(tau_b["step"]) + offset
                stored = row["trajectory"][expected_steps.index(step)]
                observed = recompute_behavior_checkpoint(row, step, device)
                row_checks[f"{label}_behavior_pass"] = observed["behavior_pass"] is True
                row_checks[f"{label}_behavior_metrics"] = close(observed["holdout"], stored["holdout_behavior"])
        else:
            row_checks["tauB_censoring_consistent"] = row["formation"]["tau_E"]["status"] == "not_authorized_behavior_right_censored"

        tau_e = row["formation"]["tau_E"]
        if tau_e["status"] == "observed":
            for offset, label in ((0, "tauE"), (int(p1214.TRAINING["evaluation_interval"]), "tauE_next")):
                step = int(tau_e["step"]) + offset
                stored = row["trajectory"][expected_steps.index(step)]
                observed = recompute_event_checkpoint(row, step, device)
                row_checks[f"{label}_event_pass"] = observed["event_pass"] is True
                row_checks[f"{label}_event_layer"] = observed["earliest_event_layer"] == stored["earliest_event_layer"]
                selected = int(observed["earliest_event_layer"])
                row_checks[f"{label}_selected_metrics"] = close(observed["layers"][selected], stored["layers"][selected])

        sentinel_key = (row["split"], row["architecture"])
        if sentinel_key not in initial_control_sentinels:
            initial_control_sentinels.add(sentinel_key)
            initial_model = p1214.load_checkpoint(find_checkpoint(row, 0), device)
            initial_controls = p1214.initial_camera_controls(initial_model, row["task"])
            row_checks["initial_camera_control_recomputed"] = close(initial_controls, row["initial_camera_controls"])
            del initial_model
            gc.collect()
            torch.cuda.empty_cache()

        per_run[row["run_id"]] = row_checks
        checks[f"run_{row['run_id']}"] = all(row_checks.values())
        rows.append(row)

    recomputed_summaries = {
        split: p1214.group_summary(split, [row for row in rows if row["split"] == split])
        for split in ("discovery", "confirmation")
    }
    checks["discovery_summary_recomputed"] = close(recomputed_summaries["discovery"], final["summaries"]["discovery"])
    checks["confirmation_summary_recomputed"] = close(recomputed_summaries["confirmation"], final["summaries"]["confirmation"])
    gate = all(value["formation_dynamics_gate"] for value in recomputed_summaries.values())
    expected_claim = "experimental_gate_passed_pending_independent_audit" if gate else "not_confirmed"
    checks["claim_matches_recomputation"] = final["claims"]["formation_dynamics"] == expected_claim
    checks["auto_continue_matches_gate"] = final["auto_continue"] is gate
    checks["no_llm_claim"] = final["claims"]["natural_language_transfer"] == "not_tested"
    result = {
        "phase": p1214.PHASE,
        "kind": "independent_checkpoint_trajectory_and_result_audit",
        "created_at": p1214.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "per_run_checks": per_run,
        "check_count": len(checks) + sum(len(value) for value in per_run.values()),
        "passed_count": sum(checks.values()) + sum(sum(value.values()) for value in per_run.values()),
        "all_checks_passed": all(checks.values()) and all(all(value.values()) for value in per_run.values()),
        "recomputed_summaries": recomputed_summaries,
        "authorized_claim": "C1214_E3_KT" if gate else "no_new_mechanism_claim",
    }
    return write_audit(p1214.OUT_ROOT / "audit/independent_result_audit.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preaudit", "final"))
    args = parser.parse_args()
    value = preaudit() if args.command == "preaudit" else final_audit()
    print(
        json.dumps(
            {
                key: value[key]
                for key in ("kind", "check_count", "passed_count", "all_checks_passed", "audit_digest")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
