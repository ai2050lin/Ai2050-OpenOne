#!/usr/bin/env python3
"""Independent protocol and checkpoint audit for Phase 1213."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1213_free_transformer_behavioral_quotient_event as p1213  # noqa: E402
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


def close(left: float, right: float, tolerance: float = 2.0e-5) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))


def write_audit(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["audit_digest"] = p1213.digest(value)
    p1213.write_json(path, value)
    return value


def preaudit() -> dict[str, Any]:
    protocol = p1213.verify_protocol()
    checks: dict[str, bool] = {}
    checks["phase_correct"] = protocol["phase"] == p1213.PHASE
    checks["source_final_frozen"] = protocol["source_phase1212_final_digest"] == p1213.EXPECTED_1212_FINAL
    checks["source_audit_frozen"] = protocol["source_phase1212_audit_digest"] == p1213.EXPECTED_1212_AUDIT
    checks["source_gate_passed"] = all(protocol["source_gate"].values())
    checks["script_hashes_frozen"] = protocol["script_hashes"] == p1213.script_hashes()
    checks["no_run_metrics_before_preaudit"] = not any((p1213.OUT_ROOT / "runs").glob("**/metrics.json"))
    checks["disjoint_task_names"] = not (
        {value["name"] for value in p1213.TASKS["discovery"]}
        & {value["name"] for value in p1213.TASKS["confirmation"]}
    )
    checks["disjoint_lexicon_seeds"] = not (
        {value["lexicon_seed"] for value in p1213.TASKS["discovery"]}
        & {value["lexicon_seed"] for value in p1213.TASKS["confirmation"]}
    )
    discovery_shapes = {(value.layers, value.width) for value in p1213.ARCHITECTURES["discovery"].values()}
    confirmation_shapes = {(value.layers, value.width) for value in p1213.ARCHITECTURES["confirmation"].values()}
    checks["disjoint_architecture_shapes"] = not (discovery_shapes & confirmation_shapes)
    checks["confirmation_has_new_depths"] = not (
        {value.layers for value in p1213.ARCHITECTURES["discovery"].values()}
        & {value.layers for value in p1213.ARCHITECTURES["confirmation"].values()}
    )
    checks["confirmation_has_new_widths"] = not (
        {value.width for value in p1213.ARCHITECTURES["discovery"].values()}
        & {value.width for value in p1213.ARCHITECTURES["confirmation"].values()}
    )
    checks["three_tasks_per_split"] = all(len(values) == 3 for values in p1213.TASKS.values())
    checks["two_replicates"] = p1213.REPLICATES == 2
    checks["six_nuisance_templates"] = len(p1213.TEMPLATES) == 6
    checks["target_has_512_states"] = len(p1213.ALL_COMBINATIONS) == 512
    for split, tasks in p1213.TASKS.items():
        for task in tasks:
            train, holdout = p1213.split_combinations(task)
            checks[f"{split}_{task['name']}_split_384_128"] = len(train) == 384 and len(holdout) == 128
            lexicon = p1213.make_lexicon(task)
            checks[f"{split}_{task['name']}_answer_permutation"] = sorted(lexicon["answer_permutation"]) == list(range(8))
            checks[f"{split}_{task['name']}_token_roles_disjoint"] = len(
                set(lexicon["roles"].values())
                | set(lexicon["values"])
                | set(lexicon["queries"].values())
                | set(lexicon["answers"])
            ) == 23
            sample = holdout[0]
            if sample[0] == sample[1]:
                sample = next(value for value in holdout if value[0] != value[1])
            left = p1213.encode(sample, 0, "row", lexicon)
            swapped = (sample[1], sample[0], sample[2])
            right = p1213.encode(swapped, 0, "row", lexicon)
            checks[f"{split}_{task['name']}_swap_same_bag"] = sorted(left) == sorted(right) and left != right
    checks["future_horizon_frozen"] = p1213.FUTURE_STEPS == 32
    checks["ridge_frozen"] = close(p1213.RIDGE, 1.0e-3, 0.0)
    checks["target_is_behavioral"] = "actual endpoint" in protocol["object_contract"]["target"]
    checks["hidden_not_target"] = "hidden coordinate" in protocol["object_contract"]["not_target"]
    checks["posthoc_coarsening_forbidden"] = "define functional classes from hidden results" in protocol["forbidden"]
    checks["global_minimality_forbidden"] = "claim global minimality from the typed registry" in protocol["forbidden"]
    checks["probe_registry_frozen"] = protocol["probe_registry"][-1] == "primitive_triple" and len(protocol["probe_registry"]) == 9
    checks["formal_run_count_24"] = (
        sum(len(p1213.TASKS[split]) * len(p1213.ARCHITECTURES[split]) * p1213.REPLICATES for split in p1213.TASKS)
        == 24
    )
    result = {
        "phase": p1213.PHASE,
        "kind": "independent_zero_output_preaudit",
        "created_at": p1213.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    return write_audit(p1213.PREAUDIT_PATH, result)


def load_model(path: Path, device: torch.device) -> TinyCausalTransformer:
    value = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyCausalTransformer(ModelConfig(**value["config"]))
    model.load_state_dict(value["state_dict"])
    return model.to(device)


def final_audit() -> dict[str, Any]:
    protocol = p1213.verify_protocol()
    preaudit_value = p1213.read_json(p1213.PREAUDIT_PATH)
    p1213.validate_digest(preaudit_value, "audit_digest")
    final = p1213.read_json(p1213.FINAL_PATH)
    p1213.validate_digest(final, "final_digest")
    checks: dict[str, bool] = {
        "protocol_digest_matches": final["protocol_digest"] == protocol["protocol_digest"],
        "preaudit_digest_matches": final["preaudit_digest"] == preaudit_value["audit_digest"],
        "preaudit_passed": preaudit_value["all_checks_passed"] is True,
        "manifest_has_24_runs": len(final["run_manifest"]) == 24,
        "manifest_run_ids_unique": len({value["run_id"] for value in final["run_manifest"]}) == 24,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint audit")
    device = torch.device("cuda")
    recomputed_rows: dict[str, list[dict[str, Any]]] = {"discovery": [], "confirmation": []}
    per_run: dict[str, dict[str, bool]] = {}
    for manifest in final["run_manifest"]:
        metrics_path = ROOT / manifest["metrics"]
        metrics = p1213.read_json(metrics_path)
        p1213.validate_digest(metrics, "metrics_digest")
        task = metrics["task"]
        config = ModelConfig(**metrics["config"])
        endpoint_path = ROOT / manifest["endpoint"]
        future_path = ROOT / manifest["future"]
        row_checks = {
            "metrics_hash": p1213.sha256_file(metrics_path) == manifest["metrics_sha256"],
            "metrics_digest": metrics["metrics_digest"] == manifest["metrics_digest"],
            "endpoint_hash": p1213.sha256_file(endpoint_path) == manifest["endpoint_sha256"],
            "future_hash": p1213.sha256_file(future_path) == manifest["future_sha256"],
        }
        endpoint = load_model(endpoint_path, device)
        future = load_model(future_path, device)
        p1213.set_seed(int(metrics["seed"]))
        initial = TinyCausalTransformer(config).to(device)
        train, holdout = p1213.split_combinations(task)
        endpoint_train = p1213.evaluate_behavior(endpoint, task, train)
        endpoint_holdout = p1213.evaluate_behavior(endpoint, task, holdout)
        future_train = p1213.evaluate_behavior(future, task, train)
        future_holdout = p1213.evaluate_behavior(future, task, holdout)
        row_checks["endpoint_train_accuracy"] = close(endpoint_train["accuracy"], metrics["training"]["train"]["accuracy"])
        row_checks["endpoint_holdout_accuracy"] = close(endpoint_holdout["accuracy"], metrics["training"]["holdout"]["accuracy"])
        row_checks["endpoint_holdout_min_probability"] = close(
            endpoint_holdout["minimum_probability"], metrics["training"]["holdout"]["minimum_probability"]
        )
        row_checks["future_train_accuracy"] = close(future_train["accuracy"], metrics["future_behavior"]["train"]["accuracy"])
        row_checks["future_holdout_accuracy"] = close(future_holdout["accuracy"], metrics["future_behavior"]["holdout"]["accuracy"])
        endpoint_signatures, endpoint_signature_metrics = p1213.signature_map(endpoint, task)
        future_signatures, future_signature_metrics = p1213.signature_map(future, task)
        stability = float(np.mean([endpoint_signatures[value] == future_signatures[value] for value in p1213.ALL_COMBINATIONS]))
        analysis = metrics["analysis"]
        row_checks["endpoint_signature_digest"] = endpoint_signature_metrics["signature_digest"] == analysis["endpoint_signature"]["signature_digest"]
        row_checks["future_signature_digest"] = future_signature_metrics["signature_digest"] == analysis["future_signature"]["signature_digest"]
        row_checks["future_signature_stability"] = close(stability, analysis["future_signature_stability"])
        row_checks["bag_control"] = p1213.bag_control(endpoint_signatures) == analysis["bag_control"]
        row_checks["probe_registry"] = p1213.probe_registry(endpoint_signatures) == analysis["probe_registry"]
        selected = analysis["selected_layer"]
        if selected is not None:
            camera, weights = p1213.camera_for_layer(endpoint, initial, task, endpoint_signatures, int(selected))
            row_checks["selected_validation_camera"] = close(
                camera["validation"]["combined_accuracy"],
                analysis["layers"][int(selected)]["validation"]["combined_accuracy"],
            )
            row_checks["selected_random_camera"] = close(
                camera["initial_validation"]["combined_accuracy"],
                analysis["layers"][int(selected)]["initial_validation"]["combined_accuracy"],
            )
            endpoint_features, endpoint_combinations = p1213.collect_response_features(endpoint, task, holdout, (4, 5), int(selected))
            heldout_camera = p1213.decoder_accuracy(endpoint_features, endpoint_combinations, endpoint_signatures, weights)
            future_features, future_combinations = p1213.collect_response_features(future, task, holdout, (4, 5), int(selected))
            future_camera = p1213.decoder_accuracy(future_features, future_combinations, endpoint_signatures, weights)
            row_checks["heldout_camera"] = close(heldout_camera["combined_accuracy"], analysis["heldout_camera"]["combined_accuracy"])
            row_checks["future_camera"] = close(future_camera["combined_accuracy"], analysis["future_camera"]["combined_accuracy"])
            patch = p1213.query_patch_metrics(endpoint, task, holdout, (4, 5), (0, 1), int(selected))
            row_checks["patch_same"] = close(patch["same_preservation"], analysis["heldout_patch"]["same_preservation"])
            row_checks["patch_transfer"] = close(patch["wrong_transfer"], analysis["heldout_patch"]["wrong_transfer"])
            distance = p1213.full_state_distance(endpoint, task, holdout, int(selected))
            row_checks["state_min_distance"] = close(distance["minimum_rms_distance"], analysis["state_distance"]["minimum_rms_distance"])
            row_checks["decoder_digest"] = p1213.digest([value.tolist() for value in weights]) == analysis["decoder_digest"]
            earlier = analysis["layers"][: int(selected)]
            row_checks["selected_is_earliest_stored_event"] = all(
                value["patch"] is None
                or value["patch"]["same_preservation"] < p1213.THRESHOLDS["patch_same_preservation_min"]
                or value["patch"]["wrong_transfer"] < p1213.THRESHOLDS["patch_wrong_transfer_min"]
                for value in earlier
            )
        else:
            row_checks["no_event_record_consistent"] = analysis["event_qualified"] is False
        per_run[metrics["run_id"]] = row_checks
        checks[f"run_{metrics['run_id']}"] = all(row_checks.values())
        recomputed_rows[metrics["split"]].append(metrics)
        del endpoint, future, initial
        gc.collect()
        torch.cuda.empty_cache()
    recomputed_summaries = {
        split: p1213.group_summary(split, rows) for split, rows in recomputed_rows.items()
    }
    checks["discovery_summary_recomputed"] = recomputed_summaries["discovery"] == final["summaries"]["discovery"]
    checks["confirmation_summary_recomputed"] = recomputed_summaries["confirmation"] == final["summaries"]["confirmation"]
    confirmed = bool(
        recomputed_summaries["discovery"]["behavior_gate"]
        and recomputed_summaries["confirmation"]["behavior_gate"]
        and recomputed_summaries["discovery"]["event_gate"]
        and recomputed_summaries["confirmation"]["event_gate"]
    )
    checks["claim_matches_recomputation"] = (
        final["claims"]["free_behavioral_quotient"] == ("confirmed" if confirmed else "not_confirmed")
    )
    result = {
        "phase": p1213.PHASE,
        "kind": "independent_checkpoint_and_result_audit",
        "created_at": p1213.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "per_run_checks": per_run,
        "check_count": len(checks) + sum(len(value) for value in per_run.values()),
        "passed_count": sum(checks.values()) + sum(sum(value.values()) for value in per_run.values()),
        "all_checks_passed": all(checks.values()) and all(all(value.values()) for value in per_run.values()),
        "recomputed_summaries": recomputed_summaries,
    }
    return write_audit(p1213.OUT_ROOT / "audit/independent_audit.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preaudit", "final"))
    args = parser.parse_args()
    value = preaudit() if args.command == "preaudit" else final_audit()
    print(json.dumps({key: value[key] for key in ("kind", "check_count", "passed_count", "all_checks_passed", "audit_digest")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
