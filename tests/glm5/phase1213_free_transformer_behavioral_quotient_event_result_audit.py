#!/usr/bin/env python3
"""Post-result type-safe audit for Phase 1213.

The preregistered audit v1 attempted to read hidden-analysis fields from runs
that correctly stopped at the behavior gate.  This separate auditor preserves
the frozen v1 file and fixes only that sum-type dispatch: failed behavior runs
are audited as untested, while qualified runs receive the full checkpoint
recomputation.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1213_free_transformer_behavioral_quotient_event as p1213  # noqa: E402
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


SCRIPT = Path(__file__).resolve()


def close(left: float, right: float, tolerance: float = 2.0e-5) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))


def load_model(path: Path, device: torch.device) -> TinyCausalTransformer:
    value = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyCausalTransformer(ModelConfig(**value["config"]))
    model.load_state_dict(value["state_dict"])
    return model.to(device)


def main() -> None:
    protocol = p1213.verify_protocol()
    preaudit = p1213.read_json(p1213.PREAUDIT_PATH)
    p1213.validate_digest(preaudit, "audit_digest")
    final = p1213.read_json(p1213.FINAL_PATH)
    p1213.validate_digest(final, "final_digest")
    checks: dict[str, bool] = {
        "protocol_digest_matches": final["protocol_digest"] == protocol["protocol_digest"],
        "preaudit_digest_matches": final["preaudit_digest"] == preaudit["audit_digest"],
        "preaudit_47_of_47": preaudit["all_checks_passed"] and preaudit["check_count"] == 47,
        "manifest_has_24_runs": len(final["run_manifest"]) == 24,
        "manifest_ids_unique": len({value["run_id"] for value in final["run_manifest"]}) == 24,
        "preregistered_audit_v1_retained": p1213.sha256_file(p1213.AUDIT_SCRIPT) == protocol["script_hashes"]["audit"],
    }
    device = torch.device("cuda")
    per_run: dict[str, dict[str, bool]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"discovery": [], "confirmation": []}
    for manifest in final["run_manifest"]:
        metrics_path = ROOT / manifest["metrics"]
        metrics = p1213.read_json(metrics_path)
        p1213.validate_digest(metrics, "metrics_digest")
        task = metrics["task"]
        config = ModelConfig(**metrics["config"])
        endpoint_path = ROOT / manifest["endpoint"]
        future_path = ROOT / manifest["future"]
        row: dict[str, bool] = {
            "metrics_hash": p1213.sha256_file(metrics_path) == manifest["metrics_sha256"],
            "metrics_digest": metrics["metrics_digest"] == manifest["metrics_digest"],
            "endpoint_hash": p1213.sha256_file(endpoint_path) == manifest["endpoint_sha256"],
            "future_hash": p1213.sha256_file(future_path) == manifest["future_sha256"],
        }
        endpoint = load_model(endpoint_path, device)
        future = load_model(future_path, device)
        train, holdout = p1213.split_combinations(task)
        endpoint_train = p1213.evaluate_behavior(endpoint, task, train)
        endpoint_holdout = p1213.evaluate_behavior(endpoint, task, holdout)
        future_train = p1213.evaluate_behavior(future, task, train)
        future_holdout = p1213.evaluate_behavior(future, task, holdout)
        row["endpoint_train_accuracy"] = close(endpoint_train["accuracy"], metrics["training"]["train"]["accuracy"])
        row["endpoint_holdout_accuracy"] = close(endpoint_holdout["accuracy"], metrics["training"]["holdout"]["accuracy"])
        row["endpoint_holdout_min_probability"] = close(
            endpoint_holdout["minimum_probability"], metrics["training"]["holdout"]["minimum_probability"]
        )
        row["future_train_accuracy"] = close(future_train["accuracy"], metrics["future_behavior"]["train"]["accuracy"])
        row["future_holdout_accuracy"] = close(future_holdout["accuracy"], metrics["future_behavior"]["holdout"]["accuracy"])
        analysis = metrics["analysis"]
        if not metrics["training"]["qualified"]:
            row["behavior_failure_typed_as_untested"] = (
                analysis.get("not_tested_reason") == "behavior_gate_failed"
                and analysis.get("event_qualified") is False
                and "endpoint_signature" not in analysis
            )
        else:
            p1213.set_seed(int(metrics["seed"]))
            initial = TinyCausalTransformer(config).to(device)
            endpoint_signatures, endpoint_signature_metrics = p1213.signature_map(endpoint, task)
            future_signatures, future_signature_metrics = p1213.signature_map(future, task)
            stability = float(
                np.mean([endpoint_signatures[value] == future_signatures[value] for value in p1213.ALL_COMBINATIONS])
            )
            row["endpoint_signature"] = endpoint_signature_metrics["signature_digest"] == analysis["endpoint_signature"]["signature_digest"]
            row["future_signature"] = future_signature_metrics["signature_digest"] == analysis["future_signature"]["signature_digest"]
            row["future_stability"] = close(stability, analysis["future_signature_stability"])
            row["bag_control"] = p1213.bag_control(endpoint_signatures) == analysis["bag_control"]
            row["probe_registry"] = p1213.probe_registry(endpoint_signatures) == analysis["probe_registry"]
            selected = int(analysis["selected_layer"])
            camera, weights = p1213.camera_for_layer(endpoint, initial, task, endpoint_signatures, selected)
            row["selected_camera"] = close(
                camera["validation"]["combined_accuracy"], analysis["layers"][selected]["validation"]["combined_accuracy"]
            )
            row["random_camera"] = close(
                camera["initial_validation"]["combined_accuracy"],
                analysis["layers"][selected]["initial_validation"]["combined_accuracy"],
            )
            endpoint_features, endpoint_combinations = p1213.collect_response_features(endpoint, task, holdout, (4, 5), selected)
            future_features, future_combinations = p1213.collect_response_features(future, task, holdout, (4, 5), selected)
            heldout_camera = p1213.decoder_accuracy(endpoint_features, endpoint_combinations, endpoint_signatures, weights)
            future_camera = p1213.decoder_accuracy(future_features, future_combinations, endpoint_signatures, weights)
            row["heldout_camera"] = close(heldout_camera["combined_accuracy"], analysis["heldout_camera"]["combined_accuracy"])
            row["future_camera"] = close(future_camera["combined_accuracy"], analysis["future_camera"]["combined_accuracy"])
            patch = p1213.query_patch_metrics(endpoint, task, holdout, (4, 5), (0, 1), selected)
            row["patch_same"] = close(patch["same_preservation"], analysis["heldout_patch"]["same_preservation"])
            row["patch_transfer"] = close(patch["wrong_transfer"], analysis["heldout_patch"]["wrong_transfer"])
            distance = p1213.full_state_distance(endpoint, task, holdout, selected)
            row["state_distance"] = close(distance["minimum_rms_distance"], analysis["state_distance"]["minimum_rms_distance"])
            row["decoder_digest"] = p1213.digest([value.tolist() for value in weights]) == analysis["decoder_digest"]
            row["event_recomputed_qualified"] = bool(
                heldout_camera["combined_accuracy"] >= p1213.THRESHOLDS["camera_holdout_accuracy_min"]
                and future_camera["combined_accuracy"] >= p1213.THRESHOLDS["camera_future_accuracy_min"]
                and patch["same_preservation"] >= p1213.THRESHOLDS["patch_same_preservation_min"]
                and patch["wrong_transfer"] >= p1213.THRESHOLDS["patch_wrong_transfer_min"]
                and distance["minimum_rms_distance"] >= p1213.THRESHOLDS["state_rms_distance_min"]
            ) == analysis["event_qualified"]
            del initial
        per_run[metrics["run_id"]] = row
        checks[f"run_{metrics['run_id']}"] = all(row.values())
        rows_by_split[metrics["split"]].append(metrics)
        del endpoint, future
        gc.collect()
        torch.cuda.empty_cache()
    summaries = {split: p1213.group_summary(split, rows) for split, rows in rows_by_split.items()}
    checks["discovery_summary"] = summaries["discovery"] == final["summaries"]["discovery"]
    checks["confirmation_summary"] = summaries["confirmation"] == final["summaries"]["confirmation"]
    confirmed = bool(
        summaries["discovery"]["behavior_gate"]
        and summaries["confirmation"]["behavior_gate"]
        and summaries["discovery"]["event_gate"]
        and summaries["confirmation"]["event_gate"]
    )
    checks["negative_claim_matches"] = (
        final["claims"]["free_behavioral_quotient"] == ("confirmed" if confirmed else "not_confirmed")
    )
    result = {
        "phase": p1213.PHASE,
        "kind": "post_result_type_safe_independent_audit",
        "created_at": p1213.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "auditor_sha256": p1213.sha256_file(SCRIPT),
        "preregistered_audit_v1_status": "failed_with_KeyError_on_behavior-gated_sum_type",
        "repair_scope": "dispatch only; no threshold, target, run, or metric change",
        "checks": checks,
        "per_run_checks": per_run,
        "check_count": len(checks) + sum(len(value) for value in per_run.values()),
        "passed_count": sum(checks.values()) + sum(sum(value.values()) for value in per_run.values()),
        "all_checks_passed": all(checks.values()) and all(all(value.values()) for value in per_run.values()),
        "recomputed_summaries": summaries,
    }
    result["audit_digest"] = p1213.digest(result)
    p1213.write_json(p1213.OUT_ROOT / "audit/independent_result_audit.json", result)
    print(p1213.canonical({
        "check_count": result["check_count"],
        "passed_count": result["passed_count"],
        "all_checks_passed": result["all_checks_passed"],
        "audit_digest": result["audit_digest"],
    }))


if __name__ == "__main__":
    main()
