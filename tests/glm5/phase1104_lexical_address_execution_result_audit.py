#!/usr/bin/env python3
"""Audit the complete Phase1104 result chain."""

from __future__ import annotations

import json
from pathlib import Path

import phase1104_lexical_address_execution_protocol as protocol


def main() -> None:
    checks: dict[str, bool] = {}
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    causal = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json"
    )
    final = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    checks["protocol_digest"] = protocol.digest({
        key: value for key, value in prereg.items() if key != "protocol_digest"
    }) == prereg["protocol_digest"]
    checks["protocol_audit_passed"] = protocol_audit["all_checks_passed"]
    checks["behavior_digest"] = protocol.digest({
        key: value for key, value in behavior.items()
        if key != "authorization_digest"
    }) == behavior["authorization_digest"]
    checks["causal_digest"] = protocol.digest({
        key: value for key, value in causal.items()
        if key != "causal_authorization_digest"
    }) == causal["causal_authorization_digest"]
    checks["final_digest"] = protocol.digest({
        key: value for key, value in final.items()
        if key != "final_summary_digest"
    }) == final["final_summary_digest"]
    checks["phase1103_unchanged_flag"] = (
        behavior["phase1103_decision_unchanged"]
        and final["phase1103_frozen_decision_unchanged"]
    )
    checks["all_behavior_models_present"] = set(behavior["models"]) == set(
        protocol.MODELS
    )
    checks["behavior_summary_chain"] = all(
        protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )["summary_digest"] == behavior["models"][model]["summary_digest"]
        for model in protocol.MODELS
    )
    checks["selected_pairs_are_behavior_passing"] = all(
        set(row["causal_selected_pairs"]).issubset(
            row["model_specific_passing_pairs"]
        )
        and len(row["causal_selected_pairs"])
        <= protocol.MAX_CAUSAL_PAIRS_PER_MODEL
        for row in behavior["models"].values()
    )
    checks["causal_files_match_authorization"] = all(
        (protocol.OUT_ROOT / "causal" / model / "summary.json").exists()
        for model in protocol.MODELS
    )
    checks["causal_pair_scope"] = True
    checks["causal_record_schema"] = True
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "causal" / model / "summary.json"
        )
        selected = set(behavior["models"][model]["causal_selected_pairs"])
        checks["causal_pair_scope"] &= set(summary["eligible_pairs"]) == selected
        detail_path = protocol.OUT_ROOT / "causal" / model / "patch_detail.jsonl"
        if summary.get("skipped"):
            checks["causal_record_schema"] &= not detail_path.exists()
            continue
        rows = protocol.read_jsonl(detail_path)
        checks["causal_pair_scope"] &= all(
            row["relation_pair"] in selected for row in rows
        )
        checks["causal_record_schema"] &= all(
            row["schema_version"] == "phase1104_causal_patch_record.v1"
            and row["source_regime"] != row["target_regime"]
            and (
                row["delta_origin_regime"] == row["target_regime"]
                if row["patch_kind"] == "within_regime_raw"
                else row["delta_origin_regime"] == row["source_regime"]
            )
            for row in rows
        )
    checks["confirmation_not_used_for_depth_selection"] = all(
        row["selection_used_only_qualification"]
        for row in causal.get("model_cells", {}).values()
    )
    checks["cross_model_claim_requires_two_models"] = all(
        len(row["passing_models"])
        >= protocol.THRESHOLDS["minimum_models_for_cross_model_upgrade"]
        for row in causal.get("cross_model_confirmed_cells", [])
    )
    checks["automatic_next_recorded"] = (
        final["automatic_next_required"]
        and bool(final["automatic_next_task"])
    )
    result = {
        "schema_version": "phase1104_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "audit" / "result_audit.json", result
    )
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1104 result audit failed: {failed}")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
