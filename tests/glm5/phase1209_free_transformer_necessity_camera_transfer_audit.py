#!/usr/bin/env python3
"""Independent protocol and artifact audit for Phase1209."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1209_free_transformer_necessity_camera_transfer as main


PRE_PATH = main.OUT_ROOT / "protocol/independent_preaudit.json"
FINAL_PATH = main.OUT_ROOT / "audit/independent_audit.json"


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


def preaudit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = main.read_json(main.OUT_ROOT / "protocol/preregistration.json")
    clean = dict(protocol)
    stored = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored)
    add(checks, "main_hash", protocol["scripts"]["main_sha256"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["scripts"]["audit_sha256"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "all_protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add(checks, "source_final_digest", len(protocol["source_phase1208_final_digest"]) == 64)
    add(checks, "source_audit_digest", len(protocol["source_phase1208_audit_digest"]) == 64)
    add(checks, "four_sites", protocol["top_sites"] == 4)
    add(checks, "two_architectures", len(protocol["architectures"]) == 2)
    add(checks, "four_replicates", protocol["replicates"] == 4)
    add(checks, "three_factors", len(protocol["factors"]) == 3)
    add(checks, "holdout_three", len(protocol["overlap_holdout_keys"]) == 3)
    add(checks, "confirmation_absent", not (main.OUT_ROOT / "runs/confirmation").exists())
    add(checks, "formal_runs_absent", not (main.OUT_ROOT / "runs").exists())
    add(checks, "cuda_available", torch.cuda.is_available())
    add(checks, "pretrained_forbidden", protocol["checks"]["pretrained_model_scan_forbidden"])
    return finish(checks, "zero-output protocol audit", PRE_PATH)


def recompute_measurements(split: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sites = main.read_json(main.OUT_ROOT / "analysis/frozen_sites.json")
    main.validate_digest(sites, "site_digest")
    public = main.read_jsonl_gz(main.OUT_ROOT / f"runs/{split}/public_models.jsonl.gz")
    low: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in public:
        model, payload, _ = main.load_model(split, row, device)
        config = main.base.ModelConfig(**payload["config"])
        for factor in main.FACTORS:
            selected = sites["factor_sites"][factor]["site_indices"]
            low.append(main.low_order_row(
                model, config, payload["lexicon"], split, factor, row["model_id"], selected, device,
            ))
            holdout.append(main.heldout_row(
                model, config, payload["lexicon"], split, factor, row["model_id"], selected, device,
            ))
        del model
        torch.cuda.empty_cache()
    return low, holdout


def final_audit(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1209 audit requires CUDA")
    checks: list[dict[str, Any]] = []
    protocol = main.verify_protocol()
    pre = main.read_json(PRE_PATH)
    main.validate_digest(pre, "audit_digest")
    add(checks, "preaudit_passed", pre["all_checks_passed"])

    discovery_root = main.OUT_ROOT / "runs/discovery"
    public = main.read_jsonl_gz(discovery_root / "public_models.jsonl.gz")
    truth = main.read_jsonl_gz(discovery_root / "sealed_model_truth.jsonl.gz")
    training = main.read_jsonl_gz(discovery_root / "training_metrics.jsonl.gz")
    scout = main.read_jsonl_gz(discovery_root / "scout_profiles.jsonl.gz")
    summary = main.read_json(discovery_root / "training_summary.json")
    main.validate_digest(summary, "summary_digest")
    expected_models = len(main.ARCHITECTURES) * main.REPLICATES
    add(checks, "discovery_model_count", len(public) == expected_models)
    add(checks, "discovery_truth_count", len(truth) == expected_models)
    add(checks, "discovery_training_count", len(training) == expected_models)
    add(checks, "discovery_scout_count", len(scout) == expected_models * len(main.FACTORS))
    add(checks, "discovery_joined_ids", {row["model_id"] for row in public} == {row["model_id"] for row in truth} == {row["model_id"] for row in training})
    add(checks, "checkpoint_hashes", all(
        row["checkpoint_sha256"] == main.sha256_file(main.checkpoint_path("discovery", row["model_id"]))
        for row in public
    ))
    add(checks, "all_models_qualified", summary["all_models_qualified"] and all(row["qualified"] for row in training))
    add(checks, "behavior_accuracy", summary["behavior_accuracy_min"] >= main.THRESHOLDS["behavior_accuracy_min"])
    add(checks, "behavior_probability", summary["behavior_min_probability_min"] >= main.THRESHOLDS["behavior_min_probability_min"])
    add(checks, "summary_public_digest", summary["public_digest"] == main.digest(public))
    add(checks, "summary_truth_digest", summary["sealed_digest"] == main.digest(truth))
    add(checks, "summary_scout_digest", summary["scout_digest"] == main.digest(scout))

    frozen = main.read_json(main.OUT_ROOT / "analysis/frozen_sites.json")
    main.validate_digest(frozen, "site_digest")
    add(checks, "sites_authorized", frozen["measurement_authorized"])
    add(checks, "four_unique_sites_per_factor", all(
        len(set(row["site_indices"])) == main.TOP_SITES for row in frozen["factor_sites"].values()
    ))
    add(checks, "site_checks_all", all(all(row.values()) for row in frozen["checks"].values()), frozen["checks"])

    stored_low = main.read_jsonl_gz(discovery_root / "low_order_camera_inputs.jsonl.gz")
    stored_holdout = main.read_jsonl_gz(discovery_root / "sealed_holdout_responses.jsonl.gz")
    low, holdout = recompute_measurements("discovery", device)
    add(checks, "low_order_count", len(stored_low) == expected_models * len(main.FACTORS))
    add(checks, "holdout_count", len(stored_holdout) == expected_models * len(main.FACTORS))
    add(checks, "low_order_exact_recompute", main.digest(stored_low) == main.digest(low))
    add(checks, "holdout_exact_recompute", main.digest(stored_holdout) == main.digest(holdout))
    add(checks, "low_holdout_ids", {row["system_id"] for row in stored_low} == {row["system_id"] for row in stored_holdout})

    discovery = main.read_json(main.OUT_ROOT / "analysis/discovery_score.json")
    main.validate_digest(discovery, "score_digest")
    predictions = [main.camera_prediction(row) for row in stored_low]
    errors = main.prediction_errors(predictions, stored_holdout)
    add(checks, "discovery_prediction_digest", discovery["predictions_digest"] == main.digest(predictions))
    add(checks, "discovery_mae_recompute", abs(discovery["metrics"]["holdout_mae"] - float(np.mean(errors))) <= 1.0e-12)
    add(checks, "discovery_max_recompute", abs(discovery["metrics"]["holdout_max_abs_error"] - float(max(errors))) <= 1.0e-12)
    add(checks, "discovery_gate_logic", discovery["confirmation_authorized"] == all(discovery["checks"].values()))

    confirmation_root = main.OUT_ROOT / "runs/confirmation"
    if discovery["confirmation_authorized"]:
        add(checks, "confirmation_present_when_authorized", confirmation_root.exists())
        confirmation_public = main.read_jsonl_gz(confirmation_root / "public_models.jsonl.gz")
        add(checks, "confirmation_model_count", len(confirmation_public) == expected_models)
        manifest = main.read_json(main.OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
        main.validate_digest(manifest, "manifest_digest")
        add(checks, "holdout_absent_at_prediction", manifest["holdout_absent_at_prediction"])
        confirmation_predictions = main.read_jsonl_gz(main.OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
        add(checks, "confirmation_prediction_digest", manifest["prediction_digest"] == main.digest(confirmation_predictions))
        confirmation_score = main.read_json(main.OUT_ROOT / "analysis/confirmation_score.json")
        main.validate_digest(confirmation_score, "score_digest")
        add(checks, "confirmation_gate_logic", confirmation_score["external_validity_gate"] == all(confirmation_score["checks"].values()))
    else:
        add(checks, "confirmation_absent_after_discovery_stop", not confirmation_root.exists())
        add(checks, "confirmation_predictions_absent", not (main.OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz").exists())
        add(checks, "confirmation_score_absent", not (main.OUT_ROOT / "analysis/confirmation_score.json").exists())

    final = main.read_json(main.OUT_ROOT / "analysis/final.json")
    main.validate_digest(final, "final_digest")
    add(checks, "final_protocol_digest", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "final_discovery_digest", final["discovery_score_digest"] == discovery["score_digest"])
    add(checks, "final_external_validity_logic", final["learned_micro_transformer_external_validity"] is bool(final["confirmation"] is not None and main.read_json(main.OUT_ROOT / "analysis/confirmation_score.json")["external_validity_gate"]) if final["confirmation"] is not None else final["learned_micro_transformer_external_validity"] is False)
    add(checks, "claim_boundary_present", "does not show" in final["claim_boundary"])
    add(checks, "pretrained_transfer_denied", "pretrained-model transfer remains denied" in final["authorized_next"])
    add(checks, "auto_continue_false", final["auto_continue"] is False)
    return finish(checks, "independent checkpoint replay and gate audit", FINAL_PATH)


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
