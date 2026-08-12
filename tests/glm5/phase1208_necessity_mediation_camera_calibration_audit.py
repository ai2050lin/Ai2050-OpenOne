#!/usr/bin/env python3
"""Independent protocol and result audit for Phase1208."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1208_necessity_mediation_camera_calibration as main


PREAUDIT_PATH = main.OUT_ROOT / "protocol/independent_preaudit.json"
FINAL_AUDIT_PATH = main.OUT_ROOT / "audit/independent_audit.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def finish(checks: list[dict[str, Any]], stage: str, path: Path) -> dict[str, Any]:
    result = {
        "phase": main.PHASE,
        "stage": stage,
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    result["audit_digest"] = main.digest(result)
    main.write_json(path, result)
    return result


def preaudit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = main.read_json(main.OUT_ROOT / "protocol/preregistration.json")
    clean = dict(protocol)
    stored = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored)
    add(checks, "main_hash", protocol["scripts"]["main_sha256"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["scripts"]["audit_sha256"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "source_final_digest_present", len(protocol["source_phase1207_final_digest"]) == 64)
    add(checks, "source_audit_digest_present", len(protocol["source_phase1207_audit_digest"]) == 64)
    add(checks, "all_internal_checks", all(protocol["checks"].values()), protocol["checks"])
    add(
        checks,
        "split_widths_disjoint",
        set(protocol["splits"]["discovery"]["widths"]).isdisjoint(protocol["splits"]["confirmation"]["widths"]),
    )
    add(checks, "balanced_subtype_count", protocol["systems_per_subtype"] == main.SYSTEMS_PER_SUBTYPE)
    add(checks, "six_quotient_classes", len(protocol["quotient_labels"]) == 6)
    add(checks, "two_unidentifiable_subtypes", len(main.UNKNOWN_SUBTYPES) == 2)
    add(checks, "confirmation_truth_forbidden", protocol["checks"]["confirmation_truth_forbidden_during_predict"])
    add(checks, "holdout_forbidden", protocol["checks"]["holdout_responses_forbidden_during_predict"])
    add(checks, "abstention_required", protocol["checks"]["latent_subtype_abstention_required"])
    add(checks, "qwen_retuning_forbidden", protocol["checks"]["qwen3_retuning_forbidden"])
    add(checks, "cuda_available", torch.cuda.is_available())
    add(checks, "formal_outputs_absent", not (main.OUT_ROOT / "runs").exists() and not (main.OUT_ROOT / "analysis").exists())
    add(checks, "heldout_interventions_six", len(protocol["heldout_interventions"]) == 6)
    add(checks, "hard_stop_scope", any("not evidence" in item for item in protocol["hard_stops"]))
    return finish(checks, "zero-output independent protocol audit", PREAUDIT_PATH)


def recompute_split(split: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for spec in main.all_specs(split):
        p_row, h_row, t_row = main.response_record(spec, device)
        public.append(p_row)
        holdout.append(h_row)
        truth.append(t_row)
    return public, holdout, truth


def final_audit(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal independent audit requires CUDA")
    checks: list[dict[str, Any]] = []
    protocol = main.verify_protocol()
    pre = main.read_json(PREAUDIT_PATH)
    main.validate_digest(pre, "audit_digest")
    add(checks, "preaudit_passed", pre["all_checks_passed"])

    stored: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    recomputed: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    split_ids: dict[str, set[str]] = {}
    for split in main.SPLITS:
        root = main.OUT_ROOT / f"runs/{split}"
        public = main.read_jsonl_gz(root / "public_camera_inputs.jsonl.gz")
        holdout = main.read_jsonl_gz(root / "sealed_holdout_responses.jsonl.gz")
        truth = main.read_jsonl_gz(root / "sealed_truth.jsonl.gz")
        stored[split] = (public, holdout, truth)
        rep_public, rep_holdout, rep_truth = recompute_split(split, device)
        recomputed[split] = (rep_public, rep_holdout, rep_truth)
        expected = main.SYSTEMS_PER_SUBTYPE * len(main.LATENT_SUBTYPES)
        add(checks, f"{split}_public_count", len(public) == expected, len(public))
        add(checks, f"{split}_holdout_count", len(holdout) == expected, len(holdout))
        add(checks, f"{split}_truth_count", len(truth) == expected, len(truth))
        ids = {row["system_id"] for row in public}
        add(checks, f"{split}_unique_ids", len(ids) == expected)
        add(checks, f"{split}_joined_ids", ids == {row["system_id"] for row in holdout} == {row["system_id"] for row in truth})
        add(checks, f"{split}_public_exact_regeneration", main.digest(public) == main.digest(rep_public))
        add(checks, f"{split}_holdout_exact_regeneration", main.digest(holdout) == main.digest(rep_holdout))
        add(checks, f"{split}_truth_exact_regeneration", main.digest(truth) == main.digest(rep_truth))
        add(
            checks,
            f"{split}_public_schema_sealed",
            all(
                key not in row
                for row in public
                for key in ("latent_subtype", "quotient_label", "slot_by_role", "latent_bit")
            ),
        )
        add(
            checks,
            f"{split}_subtype_balance",
            set(Counter(row["latent_subtype"] for row in truth).values()) == {main.SYSTEMS_PER_SUBTYPE},
        )
        add(checks, f"{split}_clean_exact", min(row["baseline_accuracy"] for row in public) == 1.0)
        add(checks, f"{split}_null_identity", max(row["matched_null_max_drift"] for row in public) <= main.GATES["matched_null_drift_max"])
        add(checks, f"{split}_carrier_identity", max(row["carrier_control_max_drift"] for row in public) <= main.GATES["carrier_control_drift_max"])
        summary = main.read_json(root / "summary.json")
        main.validate_digest(summary, "summary_digest")
        add(checks, f"{split}_summary_public_digest", summary["public_digest"] == main.digest(public))
        add(checks, f"{split}_summary_holdout_digest", summary["holdout_digest"] == main.digest(holdout))
        add(checks, f"{split}_summary_truth_digest", summary["truth_digest"] == main.digest(truth))
        split_ids[split] = ids

    add(checks, "split_ids_disjoint", split_ids["discovery"].isdisjoint(split_ids["confirmation"]))
    add(
        checks,
        "split_widths_realized_disjoint",
        {row["task_width"] for row in stored["discovery"][0]}.isdisjoint(
            {row["task_width"] for row in stored["confirmation"][0]}
        ),
    )

    fit = main.read_json(main.OUT_ROOT / "analysis/fit.json")
    main.validate_digest(fit, "fit_digest")
    discovery_predictions = [main.classify_camera(row) for row in stored["discovery"][0]]
    discovery_metrics = main.score_predictions(discovery_predictions, stored["discovery"][2])
    add(checks, "fit_metrics_exact", fit["metrics"] == discovery_metrics)
    add(checks, "fit_authorized", fit["confirmation_authorized"])
    add(checks, "fit_checks_all", all(fit["checks"].values()), fit["checks"])

    manifest = main.read_json(main.OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
    main.validate_digest(manifest, "manifest_digest")
    predictions = main.read_jsonl_gz(main.OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
    add(checks, "prediction_digest", manifest["prediction_digest"] == main.digest(predictions))
    add(checks, "prediction_count", len(predictions) == len(stored["confirmation"][0]))
    add(checks, "prediction_truth_flag_false", manifest["truth_read"] is False)
    add(checks, "prediction_holdout_flag_false", manifest["holdout_response_read"] is False)
    add(
        checks,
        "prediction_precedes_score",
        (main.OUT_ROOT / "analysis/confirmation_prediction_manifest.json").stat().st_mtime_ns
        <= (main.OUT_ROOT / "analysis/confirmation_score.json").stat().st_mtime_ns,
    )

    score = main.read_json(main.OUT_ROOT / "analysis/confirmation_score.json")
    main.validate_digest(score, "score_digest")
    confirmation_metrics = main.score_predictions(predictions, stored["confirmation"][2])
    add(checks, "confirmation_metrics_exact", score["metrics"] == confirmation_metrics)
    add(checks, "confirmation_camera_gate", score["camera_gate"])
    add(checks, "confirmation_checks_all", all(score["checks"].values()), score["checks"])
    add(checks, "quotient_accuracy_exact", confirmation_metrics["accuracy"] == 1.0)
    add(checks, "structure_accuracy_exact", confirmation_metrics["structure_accuracy"] == 1.0)
    add(checks, "abstention_exact", confirmation_metrics["abstention_or_subtype_accuracy"] == 1.0)
    add(checks, "heldout_prediction_exact", score["heldout_intervention_max_abs_error"] <= main.GATES["holdout_max_abs_error_max"])
    add(checks, "gauge_invariant_accuracy", score["gauge_accuracy_gap"] == 0.0)
    add(checks, "scalar_camera_incomplete", score["phase1207_contrast_scalar_camera_accuracy"] < 1.0)
    add(checks, "single_operator_false_negative", score["phase1207_operator_hidden_necessity_sensitivity"] < 1.0)

    truth_by_id = {row["system_id"]: row for row in stored["confirmation"][2]}
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stored["confirmation"][0]:
        rows_by_label[truth_by_id[row["system_id"]]["quotient_label"]].append(row)
    add(
        checks,
        "necessary_contrast_detected",
        min(row["phase1207_contrast"]["behavior_damage"] for row in rows_by_label["necessary_single"]) >= 0.99,
    )
    add(
        checks,
        "small_energy_necessary_detected",
        min(row["phase1207_contrast"]["behavior_damage"] for row in rows_by_label["small_energy_necessary"]) >= 0.99,
    )
    add(
        checks,
        "late_sufficiency_without_behavior_necessity",
        max(row["phase1207_contrast"]["behavior_damage"] for row in rows_by_label["late_sufficient_nonnecessary"]) <= 1.0e-7
        and min(row["full_hidden_donor"]["donor_choice"] for row in rows_by_label["late_sufficient_nonnecessary"]) >= 0.99,
    )
    add(
        checks,
        "redundancy_single_false_negative_joint_positive",
        max(
            max(item["behavior_damage"] for item in row["single_ablation"])
            for row in rows_by_label["redundant_double"]
        ) <= 1.0e-7
        and min(
            max(item["behavior_damage"] for item in row["pair_ablation"])
            for row in rows_by_label["redundant_double"]
        ) >= 0.99,
    )
    add(
        checks,
        "context_gate_stratified",
        min(
            max(abs(item["context_behavior_damage"][0] - item["context_behavior_damage"][1]) for item in row["single_ablation"])
            for row in rows_by_label["context_gate"]
        ) >= 0.99,
    )
    add(
        checks,
        "unknown_hidden_interventions_null",
        max(
            max(item["behavior_damage"] for item in row["pair_ablation"])
            for row in rows_by_label["unidentifiable_equivalence"]
        ) <= 1.0e-7,
    )

    final = main.read_json(main.OUT_ROOT / "analysis/final.json")
    main.validate_digest(final, "final_digest")
    add(checks, "final_camera_calibrated", final["known_truth_camera_calibrated"])
    add(checks, "final_matches_score", final["known_truth_camera_calibrated"] == score["camera_gate"])
    add(checks, "final_claim_boundary", "does not identify" in final["claim_boundary"])
    add(checks, "final_k188_scope", final["new_k_item"]["level"] == "E3-KT")
    add(checks, "next_is_learned_transfer_not_qwen", "learned micro-Transformer" in final["authorized_next"])
    add(checks, "protocol_digest_propagated", final["protocol_digest"] == protocol["protocol_digest"])
    return finish(checks, "independent exact regeneration and result audit", FINAL_AUDIT_PATH)


def main_cli() -> None:
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
    main_cli()
