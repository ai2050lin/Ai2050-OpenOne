#!/usr/bin/env python3
"""Independent protocol and result audit for Phase1226."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import phase1226_known_truth_temporal_coalition_camera as main


PREAUDIT_PATH = main.OUT_ROOT / "protocol/independent_preaudit.json"
FINAL_AUDIT_PATH = main.OUT_ROOT / "audit/independent_result_audit.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def finish(checks: list[dict[str, Any]], stage: str, path: Path) -> dict[str, Any]:
    result = {
        "phase": main.PHASE,
        "stage": stage,
        "checks": checks,
        "passed_count": sum(row["passed"] for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    result["audit_digest"] = main.digest(result)
    main.write_json(path, result)
    return result


def preaudit() -> dict[str, Any]:
    protocol = main.read_json(main.OUT_ROOT / "protocol/preregistration.json")
    checks: list[dict[str, Any]] = []
    clean = dict(protocol)
    stored = clean.pop("protocol_digest")
    add(checks, "protocol_digest", main.digest(clean) == stored)
    add(checks, "main_hash", protocol["scripts"]["main_sha256"] == main.sha256_file(main.SCRIPT))
    add(checks, "audit_hash", protocol["scripts"]["audit_sha256"] == main.sha256_file(Path(__file__).resolve()))
    add(checks, "all_internal_checks", all(protocol["checks"].values()), protocol["checks"])
    add(checks, "phase1225_stop_preserved", protocol["checks"]["phase1225_auto_continue_false_preserved"])
    add(checks, "new_authorization_typed", protocol["authorization_type"].startswith("new explicit user turn"))
    add(checks, "fixed_cuda_fp16", protocol["numerical_type_eta"] == main.NUMERICAL_TYPE)
    add(checks, "fixed_batch_geometry", protocol["numerical_type_eta"]["batch_size"] == main.BATCH_SIZE)
    add(checks, "three_mechanisms", tuple(protocol["mechanism_classes"]) == main.MECHANISMS)
    add(checks, "seven_coalitions", len(protocol["slot_coalitions"]) == 7)
    add(checks, "two_time_regimes", tuple(protocol["temporal_regimes"]) == main.TEMPORAL_REGIMES)
    add(
        checks,
        "tasks_disjoint",
        set(map(tuple, protocol["splits"]["discovery"]["task_coefficients"])).isdisjoint(
            map(tuple, protocol["splits"]["confirmation"]["task_coefficients"])
        ),
    )
    add(checks, "balanced_system_count", protocol["systems_per_latent"] == main.SYSTEMS_PER_LATENT)
    add(checks, "truth_forbidden", protocol["checks"]["confirmation_truth_forbidden_during_prediction"])
    add(checks, "holdout_forbidden", protocol["checks"]["confirmation_holdout_forbidden_during_prediction"])
    add(checks, "abstention_required", protocol["checks"]["latent_abstention_required"])
    add(checks, "qwen_forbidden", protocol["checks"]["qwen3_execution_forbidden"])
    add(checks, "cuda_available", torch.cuda.is_available())
    add(checks, "formal_outputs_absent", not (main.OUT_ROOT / "runs").exists() and not (main.OUT_ROOT / "analysis").exists())
    add(checks, "hard_stop_present", any("not a Qwen3" in item for item in protocol["hard_stops"]))
    return finish(checks, "zero-output independent protocol audit", PREAUDIT_PATH)


def regenerate_split(split: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    with torch.inference_mode():
        for spec in main.all_specs(split):
            p_row, h_row, t_row = main.response_record(spec, device)
            public.append(p_row)
            holdout.append(h_row)
            truth.append(t_row)
    return public, holdout, truth


def independent_signature_checks(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, bool]:
    truth_by_id = {row["system_id"]: row for row in truth}
    outcomes = {
        "boundary_has_singleton_single": True,
        "joint_has_pair_single": True,
        "sustained_has_no_full_single": True,
        "sustained_has_pair_sustained": True,
    }
    for row in public:
        expected = truth_by_id[row["system_id"]]["mechanism_class"]
        single = [item for item in row["correct_donor_responses"] if item["regime"] == "single"]
        sustained = [item for item in row["correct_donor_responses"] if item["regime"] == "sustained"]
        one = max(item["donor_fraction"] for item in single if len(item["slots"]) == 1)
        two = max(item["donor_fraction"] for item in single if len(item["slots"]) == 2)
        full_single = next(item["donor_fraction"] for item in single if len(item["slots"]) == 3)
        pair_sustained = max(item["donor_fraction"] for item in sustained if len(item["slots"]) == 2)
        if expected == "boundary_store":
            outcomes["boundary_has_singleton_single"] &= one == 1.0
        elif expected == "source_query_joint":
            outcomes["joint_has_pair_single"] &= one == 0.0 and two == 1.0
        elif expected == "sustained_recompute":
            outcomes["sustained_has_no_full_single"] &= full_single == 0.25
            outcomes["sustained_has_pair_sustained"] &= pair_sustained == 1.0
    return outcomes


def result_audit(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1226 result audit requires CUDA")
    protocol = main.verify_protocol()
    preaudit_doc = main.read_json(PREAUDIT_PATH)
    main.validate_digest(preaudit_doc, "audit_digest")
    checks: list[dict[str, Any]] = []
    add(checks, "preaudit_passed", preaudit_doc["all_checks_passed"], preaudit_doc["audit_digest"])

    stored: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    split_ids: dict[str, set[str]] = {}
    expected_count = main.SYSTEMS_PER_LATENT * len(main.MECHANISMS) * len(main.LATENT_VARIANTS)
    for split in main.SPLITS:
        root = main.OUT_ROOT / f"runs/{split}"
        public = main.read_jsonl_gz(root / "public_camera_inputs.jsonl.gz")
        holdout = main.read_jsonl_gz(root / "sealed_holdout_responses.jsonl.gz")
        truth = main.read_jsonl_gz(root / "sealed_truth.jsonl.gz")
        stored[split] = (public, holdout, truth)
        rep_public, rep_holdout, rep_truth = regenerate_split(split, device)
        ids = {row["system_id"] for row in public}
        split_ids[split] = ids
        add(checks, f"{split}_counts", len(public) == len(holdout) == len(truth) == expected_count, len(public))
        add(checks, f"{split}_unique_ids", len(ids) == expected_count)
        add(checks, f"{split}_joined_ids", ids == {row["system_id"] for row in holdout} == {row["system_id"] for row in truth})
        add(checks, f"{split}_public_exact_regeneration", main.digest(public) == main.digest(rep_public))
        add(checks, f"{split}_holdout_exact_regeneration", main.digest(holdout) == main.digest(rep_holdout))
        add(checks, f"{split}_truth_exact_regeneration", main.digest(truth) == main.digest(rep_truth))
        add(checks, f"{split}_truth_sealed", all(
            key not in row for row in public for key in
            ("mechanism_class", "slot_by_role", "minimal_sufficient_slots", "required_temporal_regime")
        ))
        add(checks, f"{split}_balanced_mechanisms", set(Counter(row["mechanism_class"] for row in truth).values()) == {main.SYSTEMS_PER_LATENT * 2})
        add(checks, f"{split}_balanced_latents", set(Counter(row["latent_variant"] for row in truth).values()) == {main.SYSTEMS_PER_LATENT * 3})
        signatures = independent_signature_checks(public, truth)
        add(checks, f"{split}_independent_signatures", all(signatures.values()), signatures)
        summary = main.read_json(root / "summary.json")
        main.validate_digest(summary, "summary_digest")
        add(checks, f"{split}_summary_public_digest", summary["public_digest"] == main.digest(public))
        add(checks, f"{split}_summary_holdout_digest", summary["holdout_digest"] == main.digest(holdout))
        add(checks, f"{split}_summary_truth_digest", summary["truth_digest"] == main.digest(truth))
        add(checks, f"{split}_fixed_geometry", all(
            row["width"] == main.WIDTH and row["rollout_steps"] == main.ROLL_OUT_STEPS for row in public
        ))
        add(checks, f"{split}_baseline_exact", min(row["baseline_recipient_accuracy"] for row in public) == 1.0)
        add(checks, f"{split}_donor_exact", min(row["donor_clean_accuracy"] for row in public) == 1.0)
        add(checks, f"{split}_null_exact", max(row["baseline_donor_fraction"] for row in public) == 0.0)

    add(checks, "split_ids_disjoint", split_ids["discovery"].isdisjoint(split_ids["confirmation"]))

    discovery_score = main.read_json(main.OUT_ROOT / "analysis/discovery_score.json")
    main.validate_digest(discovery_score, "score_digest")
    discovery_predictions = [main.infer_camera(row) for row in stored["discovery"][0]]
    add(checks, "discovery_structure_exact", discovery_score["structure_metrics"] == main.score_structure(discovery_predictions, stored["discovery"][2]))
    add(checks, "discovery_holdout_exact", discovery_score["heldout_metrics"] == main.heldout_error(discovery_predictions, stored["discovery"][1]))
    add(checks, "discovery_authorized", discovery_score["confirmation_authorized"])

    manifest = main.read_json(main.OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
    main.validate_digest(manifest, "manifest_digest")
    predictions = main.read_jsonl_gz(main.OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
    add(checks, "prediction_digest", manifest["prediction_digest"] == main.digest(predictions))
    add(checks, "prediction_count", len(predictions) == expected_count)
    add(checks, "truth_not_read_at_prediction", manifest["truth_read"] is False)
    add(checks, "holdout_not_read_at_prediction", manifest["holdout_response_read"] is False)
    add(
        checks,
        "prediction_precedes_reveal",
        (main.OUT_ROOT / "analysis/confirmation_prediction_manifest.json").stat().st_mtime_ns
        <= (main.OUT_ROOT / "analysis/confirmation_score.json").stat().st_mtime_ns,
    )

    confirmation_score = main.read_json(main.OUT_ROOT / "analysis/confirmation_score.json")
    main.validate_digest(confirmation_score, "score_digest")
    structure = main.score_structure(predictions, stored["confirmation"][2])
    holdout_metrics = main.heldout_error(predictions, stored["confirmation"][1])
    add(checks, "confirmation_structure_exact", confirmation_score["structure_metrics"] == structure)
    add(checks, "confirmation_holdout_exact", confirmation_score["heldout_metrics"] == holdout_metrics)
    add(checks, "confirmation_gate", confirmation_score["camera_gate"])
    add(checks, "confirmation_all_checks", all(confirmation_score["checks"].values()), confirmation_score["checks"])
    add(checks, "class_accuracy_exact", structure["class_accuracy"] == 1.0)
    add(checks, "structure_accuracy_exact", structure["structure_accuracy"] == 1.0)
    add(checks, "abstention_exact", structure["abstention_accuracy"] == 1.0)
    add(checks, "heldout_prediction_exact", holdout_metrics["max_abs_error"] == 0.0)
    add(checks, "metadata_null_at_chance", confirmation_score["metadata_null_accuracy"] == 1.0 / 3.0)
    add(checks, "leaky_sentinel_exact", confirmation_score["leaky_sentinel_accuracy"] == 1.0)

    final = main.read_json(main.OUT_ROOT / "analysis/final.json")
    main.validate_digest(final, "final_digest")
    add(checks, "final_status", final["status"] == "known_truth_temporal_coalition_camera_passed")
    add(checks, "final_camera_gate", final["result"]["camera_gate"] is True)
    add(checks, "k203_scope", final["k_item"]["identifier"] == "K203" and "known-truth" in final["k_item"]["scope"])
    add(checks, "qwen_not_auto_started", final["authorization"]["qwen3_execution_now"] is False)
    add(checks, "phase1225_c1_still_failed", protocol["checks"]["phase1225_contracts_frozen"])
    return finish(checks, "independent exact-regeneration result audit", FINAL_AUDIT_PATH)


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "result"))
    args = parser.parse_args()
    result = preaudit() if args.stage == "pre" else result_audit(torch.device("cuda"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()
