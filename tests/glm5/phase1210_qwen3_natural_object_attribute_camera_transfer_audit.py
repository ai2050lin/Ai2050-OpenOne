#!/usr/bin/env python3
"""Independent protocol, replay, and result audit for Phase1210."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1210_qwen3_natural_object_attribute_camera_transfer as main


FINAL_AUDIT_PATH = main.OUT_ROOT / "audit/independent_result_audit.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def finish(checks: list[dict[str, Any]], stage: str, path: Path) -> dict[str, Any]:
    value = {
        "phase": main.PHASE,
        "stage": stage,
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    value["audit_digest"] = main.digest(value)
    main.write_json(path, value)
    return value


def recompute_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    panel_accuracy = {
        panel: float(np.mean([row["correct"] for row in rows if row["panel"] == panel]))
        for panel in main.PANELS
    }
    finite_fraction = float(np.mean([row["finite"] for row in rows]))
    gate = bool(
        finite_fraction >= main.THRESHOLDS["finite_fraction_min"]
        and panel_accuracy["active"] >= main.THRESHOLDS["active_accuracy_min"]
        and all(
            panel_accuracy[panel] >= main.THRESHOLDS["control_accuracy_min"]
            for panel in main.PANELS if panel != "active"
        )
    )
    return {
        "case_count": len(rows),
        "finite_fraction": finite_fraction,
        "panel_accuracy": panel_accuracy,
        "gate": gate,
    }


def compare_behavior(stored: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> dict[str, Any]:
    stored_by_id = {row["case_id"]: row for row in stored}
    replayed_by_id = {row["case_id"]: row for row in replayed}
    shared = sorted(set(stored_by_id) & set(replayed_by_id))
    score_error = max(
        (
            float(np.max(np.abs(
                np.asarray(stored_by_id[key]["candidate_scores"], dtype=np.float64)
                - np.asarray(replayed_by_id[key]["candidate_scores"], dtype=np.float64)
            )))
            for key in shared
        ),
        default=float("inf"),
    )
    return {
        "same_ids": len(shared) == len(stored) == len(replayed),
        "same_predictions": all(stored_by_id[key]["prediction"] == replayed_by_id[key]["prediction"] for key in shared),
        "same_correctness": all(stored_by_id[key]["correct"] == replayed_by_id[key]["correct"] for key in shared),
        "max_score_abs_error": score_error,
    }


def numeric_max_error(left: Any, right: Any) -> float:
    errors: list[float] = []

    def walk(a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(float("inf"))
                return
            for key in sorted(a):
                walk(a[key], b[key])
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(float("inf"))
                return
            for x, y in zip(a, b):
                walk(x, y)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            errors.append(abs(float(a) - float(b)))
        elif a != b:
            errors.append(float("inf"))

    walk(left, right)
    return max(errors, default=0.0)


def preaudit() -> dict[str, Any]:
    if main.PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1210 preaudit already exists")
    checks: list[dict[str, Any]] = []
    protocol = main.read_json(main.PROTOCOL_PATH)
    clean = dict(protocol)
    stored_digest = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored_digest)
    add(checks, "source_hashes", protocol["source_hashes"] == main.source_hashes())
    add(checks, "main_hash", protocol["source_hashes"]["main"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["source_hashes"]["audit"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "all_protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add(checks, "phase1209_final", protocol["source_phase1209_final_digest"] == main.EXPECTED_1209_FINAL)
    add(checks, "phase1209_audit", protocol["source_phase1209_audit_digest"] == main.EXPECTED_1209_AUDIT)
    add(checks, "qwen3_fp16_full_cuda", protocol["model"]["name"] == "qwen3" and protocol["model"]["precision"] == "FP16" and protocol["model"]["placement"] == "full_cuda")
    add(checks, "six_attributes", tuple(protocol["attributes"]) == main.ATTRIBUTES)
    add(checks, "four_panels", tuple(protocol["panels"]) == main.PANELS)
    add(checks, "twenty_five_candidate_events", len(protocol["event_registry"]) == 25)
    add(checks, "four_frozen_events", protocol["top_events_per_attribute"] == 4)
    add(checks, "prediction_keys", tuple(protocol["prediction_keys"]) == main.PREDICTION_KEYS)
    add(checks, "thresholds_frozen", protocol["thresholds"] == main.THRESHOLDS)
    add(checks, "camera_thresholds_frozen", protocol["camera_thresholds"] == main.camera.CAMERA_THRESHOLDS)

    groups = main.read_jsonl_gz(main.MATERIAL_PATH)
    tokenizer = main.tokenizer_instance()
    rebuilt = main.build_material(tokenizer)
    summary = main.material_summary(groups)
    add(checks, "material_file_hash", protocol["material_file_sha256"] == main.sha256_file(main.MATERIAL_PATH))
    add(checks, "material_digest", protocol["material"]["material_digest"] == main.digest(groups))
    add(checks, "material_exact_rebuild", main.digest(groups) == main.digest(rebuilt))
    add(checks, "material_summary", protocol["material"] == summary)
    add(checks, "group_count", len(groups) == 144)
    add(checks, "split_balance", all(len([row for row in groups if row["split"] == split]) == 72 for split in ("discovery", "confirmation")))
    add(checks, "attribute_balance", all(sum(row["split"] == split and row["attribute"] == attribute for row in groups) == 12 for split in ("discovery", "confirmation") for attribute in main.ATTRIBUTES))
    add(checks, "single_token_candidates", summary["all_candidates_single_token"])
    add(checks, "exact_length_balance", summary["all_group_lengths_exact"])
    add(checks, "split_entities_disjoint", summary["discovery_confirmation_entities_disjoint"])
    add(checks, "split_values_disjoint", summary["discovery_confirmation_values_disjoint"])
    add(checks, "split_templates_disjoint", summary["discovery_confirmation_templates_disjoint"])
    add(checks, "phase1202_entities_excluded", summary["new_entities_exclude_phase1202"])
    add(checks, "phase1202_values_excluded", summary["new_values_exclude_phase1202"])
    add(checks, "prediction_before_holdout", protocol["execution_order"].index("measure discovery low-order intervention table and seal quotient/holdout predictions") < protocol["execution_order"].index("measure discovery triple/all-event holdouts and score"))
    add(checks, "no_automatic_search", any("No rescue, head, neuron, or Phase1211 search is automatic" in row for row in protocol["hard_stops"]))
    forbidden_outputs = [
        main.OUT_ROOT / "runs",
        main.FROZEN_SITES_PATH,
        main.OUT_ROOT / "analysis/discovery_predictions.jsonl.gz",
        main.OUT_ROOT / "analysis/discovery_prediction_manifest.json",
        main.OUT_ROOT / "analysis/discovery_score.json",
        main.OUT_ROOT / "analysis/confirmation_score.json",
        main.FINAL_PATH,
        FINAL_AUDIT_PATH,
    ]
    add(checks, "zero_formal_outputs", not any(path.exists() for path in forbidden_outputs))
    add(checks, "cuda_available", torch.cuda.is_available())
    result = finish(checks, "independent zero-output protocol and material audit", main.PREAUDIT_PATH)
    result["protocol_digest"] = protocol["protocol_digest"]
    result["audit_digest"] = main.digest({key: value for key, value in result.items() if key != "audit_digest"})
    main.write_json(main.PREAUDIT_PATH, result)
    return result


def add_split_artifact_checks(checks: list[dict[str, Any]], split: str) -> dict[str, Any]:
    raw = main.read_jsonl_gz(main.split_path(split, "behavior_rows.jsonl.gz"))
    summary = main.read_json(main.split_path(split, "behavior_summary.json"))
    main.validate_digest(summary, "summary_digest")
    recomputed = recompute_behavior(raw)
    add(checks, f"{split}_behavior_count", recomputed["case_count"] == 576)
    add(checks, f"{split}_behavior_metrics", numeric_max_error(recomputed, {key: summary[key] for key in recomputed}) <= 1.0e-12)
    add(checks, f"{split}_behavior_protocol", summary["protocol_digest"] == main.verify_protocol()["protocol_digest"])
    add(checks, f"{split}_precision", summary["precision"]["gate"] is True)
    return summary


def add_scored_split_checks(checks: list[dict[str, Any]], split: str) -> dict[str, Any]:
    low = main.read_jsonl_gz(main.split_path(split, "low_order_camera_inputs.jsonl.gz"))
    predictions = main.read_jsonl_gz(main.OUT_ROOT / "analysis" / f"{split}_predictions.jsonl.gz")
    manifest = main.read_json(main.OUT_ROOT / "analysis" / f"{split}_prediction_manifest.json")
    holdout = main.read_jsonl_gz(main.split_path(split, "holdout_responses.jsonl.gz"))
    score = main.read_json(main.OUT_ROOT / "analysis" / f"{split}_score.json")
    main.validate_digest(manifest, "manifest_digest")
    main.validate_digest(score, "score_digest")
    expected_predictions = [main.camera_prediction(row) for row in low]
    add(checks, f"{split}_six_camera_units", len(low) == len(predictions) == len(holdout) == 6)
    add(checks, f"{split}_prediction_recompute", main.digest(predictions) == main.digest(expected_predictions))
    add(checks, f"{split}_prediction_manifest", manifest["prediction_digest"] == main.digest(predictions))
    add(checks, f"{split}_holdout_absent_at_prediction", manifest["holdout_absent_at_prediction"] is True)
    add(checks, f"{split}_prediction_precedes_holdout", (main.OUT_ROOT / "analysis" / f"{split}_prediction_manifest.json").stat().st_mtime_ns <= main.split_path(split, "holdout_responses.jsonl.gz").stat().st_mtime_ns)
    by_id = {row["system_id"]: row["responses"] for row in holdout}
    errors = [
        abs(float(row["predicted_holdout_responses"][key]) - float(by_id[row["system_id"]][key]))
        for row in predictions for key in main.PREDICTION_KEYS
    ]
    recomputed_metrics = {
        "unit_count": len(predictions),
        "nonabstain_count": sum(not row["abstain"] for row in predictions),
        "nonabstain_fraction": sum(not row["abstain"] for row in predictions) / max(len(predictions), 1),
        "holdout_mae": float(np.mean(errors)),
        "holdout_max_abs_error": float(max(errors)),
        "camera_decision_distribution": dict(sorted(__import__("collections").Counter(row["camera_decision"] for row in predictions).items())),
        "matched_null_max_damage": float(max(row["matched_null_max_drift"] for row in low)),
        "carrier_max_damage": float(max(row["carrier_control_max_drift"] for row in low)),
    }
    add(checks, f"{split}_score_metrics", numeric_max_error(recomputed_metrics, score["metrics"]) <= 1.0e-12)
    expected_checks = {
        "behavior": main.read_json(main.split_path(split, "behavior_summary.json"))["gate"] is True,
        "sites": main.read_json(main.FROZEN_SITES_PATH)["measurement_authorized"] is True,
        "nonabstain_breadth": recomputed_metrics["nonabstain_count"] >= main.THRESHOLDS["nonabstain_attributes_min"],
        "matched_null": recomputed_metrics["matched_null_max_damage"] <= main.THRESHOLDS["matched_null_max_damage_max"],
        "carrier": recomputed_metrics["carrier_max_damage"] <= main.THRESHOLDS["carrier_max_damage_max"],
        "holdout_mae": recomputed_metrics["holdout_mae"] <= main.THRESHOLDS["holdout_mae_max"],
        "holdout_max": recomputed_metrics["holdout_max_abs_error"] <= main.THRESHOLDS["holdout_max_abs_error_max"],
    }
    add(checks, f"{split}_score_gate_logic", score["checks"] == expected_checks and score["gate"] == all(expected_checks.values()))
    return score


def final_audit(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1210 result audit requires CUDA")
    checks: list[dict[str, Any]] = []
    protocol = main.verify_protocol()
    pre = main.read_json(main.PREAUDIT_PATH)
    main.validate_digest(pre, "audit_digest")
    add(checks, "preaudit_passed", pre["all_checks_passed"] and pre["protocol_digest"] == protocol["protocol_digest"])
    discovery_behavior = add_split_artifact_checks(checks, "discovery")

    sites = main.read_json(main.FROZEN_SITES_PATH) if main.FROZEN_SITES_PATH.exists() else None
    if sites is not None:
        main.validate_digest(sites, "site_digest")
        registry = {row["event_id"] for row in main.event_registry()}
        add(checks, "sites_discovery_only", sites["selection_split"] == "discovery")
        add(checks, "sites_candidate_count", sites["candidate_count"] == 25)
        add(checks, "sites_four_each", all(len(rows) == 4 and len({row["event_id"] for row in rows}) == 4 for rows in sites["selected"].values()))
        add(checks, "sites_in_registry", all(row["event_id"] in registry for rows in sites["selected"].values() for row in rows))
        expected_site_gate = all(sites["checks"].values())
        add(checks, "site_gate_logic", sites["measurement_authorized"] == expected_site_gate)
    else:
        add(checks, "sites_absent_only_after_behavior_stop", discovery_behavior["gate"] is False)

    discovery_score_path = main.OUT_ROOT / "analysis/discovery_score.json"
    discovery_score = add_scored_split_checks(checks, "discovery") if discovery_score_path.exists() else None
    confirmation_behavior_path = main.split_path("confirmation", "behavior_summary.json")
    confirmation_behavior = add_split_artifact_checks(checks, "confirmation") if confirmation_behavior_path.exists() else None
    confirmation_score_path = main.OUT_ROOT / "analysis/confirmation_score.json"
    confirmation_score = add_scored_split_checks(checks, "confirmation") if confirmation_score_path.exists() else None

    if discovery_score is None:
        add(checks, "no_confirmation_without_discovery_score", confirmation_behavior is None and confirmation_score is None)
    elif discovery_score["confirmation_authorized"]:
        add(checks, "confirmation_executed_when_authorized", confirmation_behavior is not None and confirmation_score is not None)
    else:
        add(checks, "confirmation_stopped_when_denied", confirmation_behavior is None and confirmation_score is None)

    final = main.read_json(main.FINAL_PATH)
    main.validate_digest(final, "final_digest")
    expected_pass = bool(confirmation_score is not None and confirmation_score["gate"])
    add(checks, "final_protocol", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "final_pass_logic", final["naturalized_qwen3_external_validity"] == expected_pass)
    add(checks, "claim_boundary", "Qwen3-only" in final["claim_boundary"] and "does not establish" in final["claim_boundary"])
    add(checks, "rescue_untested", final["rescue_status"].startswith("untested"))
    add(checks, "auto_continue_false", final["auto_continue"] is False)

    model, _tokenizer, replay_device, _load = main.load_fp16(main.MODEL)
    layers = main.get_layers(model)
    try:
        precision = main.precision_gate(model)
        add(checks, "replay_fp16_full_cuda", precision["gate"])
        for split in ("discovery", "confirmation"):
            path = main.split_path(split, "behavior_rows.jsonl.gz")
            if not path.exists():
                continue
            stored_rows = main.read_jsonl_gz(path)
            replay_rows, _summary = main.behavior_rows(main.split_groups(split), model, replay_device)
            replay = compare_behavior(stored_rows, replay_rows)
            add(checks, f"{split}_behavior_cuda_replay", replay["same_ids"] and replay["same_predictions"] and replay["same_correctness"] and replay["max_score_abs_error"] <= 1.0e-4, replay)

        if discovery_score is not None and sites is not None:
            replay_index = int(protocol["protocol_digest"][:8], 16) % len(main.ATTRIBUTES)
            replay_attribute = main.ATTRIBUTES[replay_index]
            factor_groups = [row for row in main.split_groups("discovery") if row["attribute"] == replay_attribute]
            events = [
                {key: value for key, value in row.items() if key in {"event_id", "depth", "role", "component"}}
                for row in sites["selected"][replay_attribute]
            ]
            stored_low = {
                row["factor"]: row for row in main.read_jsonl_gz(main.split_path("discovery", "low_order_camera_inputs.jsonl.gz"))
            }[replay_attribute]
            stored_holdout = {
                row["system_id"]: row for row in main.read_jsonl_gz(main.split_path("discovery", "holdout_responses.jsonl.gz"))
            }[f"p1210:discovery:{replay_attribute}"]
            replay_low = main.low_order_row(model, layers, "discovery", replay_attribute, factor_groups, events, replay_device)
            replay_holdout = main.holdout_row(model, layers, "discovery", replay_attribute, factor_groups, events, replay_device)
            low_error = numeric_max_error(stored_low, replay_low)
            holdout_error = numeric_max_error(stored_holdout, replay_holdout)
            add(checks, "sealed_attribute_response_cuda_replay", low_error <= 1.0e-6 and holdout_error <= 1.0e-6, {"attribute": replay_attribute, "low_order_max_error": low_error, "holdout_max_error": holdout_error})
    finally:
        main.release_fp16(model)

    return finish(checks, "independent formula, seal-order, CUDA behavior, and response replay audit", FINAL_AUDIT_PATH)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preaudit", "final"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = preaudit() if args.command == "preaudit" else final_audit(torch.device(args.device))
    print(json.dumps({
        "stage": result["stage"],
        "all_checks_passed": result["all_checks_passed"],
        "passed_count": result["passed_count"],
        "check_count": result["check_count"],
        "audit_digest": result["audit_digest"],
    }, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
