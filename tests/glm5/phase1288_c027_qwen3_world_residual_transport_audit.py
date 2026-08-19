#!/usr/bin/env python3
"""Pre-run and final replay audits for Phase1288 C027."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
from phase1288_c027_qwen3_world_residual_transport import (  # noqa: E402
    AUDITOR, CAMPAIGN, COMPLETE, ENVIRONMENT, FINAL, GENERATIONS, INPUT_AUDIT,
    INPUT_FINAL, INPUT_MATERIAL, INPUT_PROTOCOL, PREAUDIT, PROTOCOL, RAW,
    RUN_SUMMARY, SCRIPT, SELECTION_DECISION, behavior_summary, build_signatures,
    canonical_json, digest, discovery_centers, evaluate_selected, file_sha256,
    fit_and_select, generation_summary, read_json, read_jsonl, reliability_summary,
)


FINAL_AUDIT = ROOT / "tests/glm5/result/phase1288_c027_qwen3_world_residual_transport/audit/independent_final_audit.json"


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def protocol_checks(require_incomplete: bool) -> dict[str, bool]:
    protocol = read_json(PROTOCOL)
    parent_final = read_json(INPUT_FINAL)
    parent_audit = read_json(INPUT_AUDIT)
    timeless = {key: value for key, value in protocol.items() if key not in ("created_at_utc", "protocol_digest")}
    return {
        "phase_campaign": protocol["phase"] == 1288 and protocol["campaign"] == CAMPAIGN,
        "protocol_digest": protocol["protocol_digest"] == digest(timeless),
        "main_source_hash": protocol["source_hashes"]["main"] == file_sha256(SCRIPT),
        "auditor_source_hash": protocol["source_hashes"]["auditor"] == file_sha256(AUDITOR),
        "parent_protocol_hash": protocol["dependencies"]["phase1287_protocol"] == file_sha256(INPUT_PROTOCOL),
        "parent_material_hash": protocol["dependencies"]["phase1287_material"] == file_sha256(INPUT_MATERIAL),
        "parent_final_hash": protocol["dependencies"]["phase1287_final"] == file_sha256(INPUT_FINAL),
        "parent_audit_hash": protocol["dependencies"]["phase1287_audit"] == file_sha256(INPUT_AUDIT),
        "parent_authorization": parent_final["authorization"] == "phase1288_qwen3_world_residual_behavior_after_audit",
        "parent_audit_passed": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "phase1288_qwen3_world_residual_behavior",
        "formal_budget_one": protocol["formal_run_budget"] == 1,
        "model_precision": protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization",
        "counts_frozen": protocol["counts"]["worlds"] == 162 and protocol["counts"]["scored_sequences"] == 46656,
        "selection_before_confirmation": protocol["unblinding_order"][2].startswith("write the selected-family artifact"),
        "complete_marker_absent": (not COMPLETE.exists()) if require_incomplete else True,
    }


def preaudit() -> None:
    checks = protocol_checks(require_incomplete=True)
    result = {
        "phase": 1288,
        "campaign": CAMPAIGN,
        "audit_stage": "pre_weight",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "run_once" if all(checks.values()) else "stop_before_weights",
    }
    atomic_write(PREAUDIT, result)
    print(canonical_json({"stage": "preaudit", "passed": result["passed"], "total": result["total"], "authorization": result["authorization"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def final_audit() -> None:
    protocol = read_json(PROTOCOL)
    final = read_json(FINAL)
    summary = read_json(RUN_SUMMARY)
    complete = read_json(COMPLETE)
    selection_artifact = read_json(SELECTION_DECISION)
    rows = read_jsonl(INPUT_MATERIAL)
    raw = read_jsonl(RAW)
    generations = read_jsonl(GENERATIONS)

    signatures = build_signatures(raw, rows)
    centers_mean = {
        kind: discovery_centers(rows, signatures["mean_log_prob"], kind)
        for kind in ("active", "lexical", "role")
    }
    selection = fit_and_select(
        rows, signatures["mean_log_prob"], centers_mean["active"], protocol["content_feature_order"],
        protocol["thresholds"]["selection_simplicity_tolerance"], "mean_log_prob",
    )
    behavior = behavior_summary(raw, rows, signatures, protocol["thresholds"])
    generation = generation_summary(generations, protocol["thresholds"])
    reliability = reliability_summary(
        rows, signatures["mean_log_prob"], centers_mean["active"], protocol["thresholds"], "active",
    )
    mapping = evaluate_selected(
        rows, signatures["mean_log_prob"], centers_mean, protocol["content_feature_order"],
        selection["selected_family"], protocol["thresholds"], "mean_log_prob",
    )
    centers_total = {
        kind: discovery_centers(rows, signatures["total_log_prob"], kind)
        for kind in ("active", "lexical", "role")
    }
    total_mapping = evaluate_selected(
        rows, signatures["total_log_prob"], centers_total, protocol["content_feature_order"],
        selection["selected_family"], protocol["thresholds"], "total_log_prob_same_family",
    )
    total_pass = min(
        value["risk_gain_over_zero"] for value in total_mapping["confirmation"].values()
    ) > protocol["thresholds"]["total_account_transport_gain_min"]
    ledgers = {
        "behavior": behavior["passed"],
        "generation": generation["passed"],
        "residual_reliability": reliability["passed"],
        "transport": mapping["mapping_passed"],
        "specificity": mapping["specificity_passed"],
        "total_account": total_pass,
    }
    all_passed = all(ledgers.values())

    checks = {
        **protocol_checks(require_incomplete=False),
        "preaudit_passed": read_json(PREAUDIT)["all_checks_passed"],
        "raw_row_count": len(raw) == protocol["counts"]["contexts"],
        "generation_row_count": len(generations) == protocol["counts"]["confirmation_generations"],
        "raw_all_finite": all(row["finite"] for row in raw),
        "selection_artifact_preconfirmation_flag": selection_artifact["confirmation_metrics_read"] is False,
        "selection_digest": selection_artifact["decision_digest"] == digest({key: value for key, value in selection_artifact.items() if key != "decision_digest"}),
        "selection_recomputed": digest(selection) == digest({
            key: value for key, value in selection_artifact.items()
            if key not in ("phase", "campaign", "written_at_utc", "confirmation_metrics_read", "decision_digest")
        }),
        "selected_family_match": final["selection"]["selected_family"] == selection["selected_family"],
        "behavior_recomputed": digest(final["behavior"]) == digest(behavior),
        "generation_recomputed": digest(final["generation"]) == digest(generation),
        "reliability_recomputed": digest(final["reliability"]) == digest(reliability),
        "mapping_recomputed": digest(final["mapping"]) == digest(mapping),
        "total_mapping_recomputed": digest(summary["total_log_prob_sensitivity"]) == digest(total_mapping),
        "ledgers_recomputed": final["ledgers"] == ledgers,
        "verdict_recomputed": final["verdict"] == ("qwen3_world_residual_transport_qualified" if all_passed else "qwen3_world_residual_transport_gate_failed"),
        "authorization_recomputed": final["authorization"] == ("phase1289_qwen3_hidden_world_residual_path" if all_passed else "close_c027_without_hidden"),
        "precision_fp16": set(final["precision_audit"]["parameter_dtypes"]) == {"float16"},
        "precision_no_quantization": not final["precision_audit"]["has_quantized_modules"] and not final["precision_audit"]["has_bf16_parameters"],
        "raw_hash": complete["raw_sha256"] == file_sha256(RAW),
        "generation_hash": complete["generation_sha256"] == file_sha256(GENERATIONS),
        "selection_hash": complete["selection_decision_sha256"] == file_sha256(SELECTION_DECISION),
        "summary_hash": complete["run_summary_sha256"] == file_sha256(RUN_SUMMARY),
        "final_hash": complete["final_sha256"] == file_sha256(FINAL),
        "selection_written_before_final": SELECTION_DECISION.stat().st_mtime_ns <= FINAL.stat().st_mtime_ns,
    }
    result = {
        "phase": 1288,
        "campaign": CAMPAIGN,
        "audit_stage": "final_independent_replay",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_ledgers": ledgers,
        "recomputed_selected_family": selection["selected_family"],
        "authorization": final["authorization"] if all(checks.values()) else "audit_failure_stop",
    }
    atomic_write(FINAL_AUDIT, result)
    print(canonical_json({
        "stage": "final", "passed": result["passed"], "total": result["total"],
        "ledgers": ledgers, "authorization": result["authorization"],
    }))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    arguments = parser.parse_args()
    preaudit() if arguments.stage == "preaudit" else final_audit()
