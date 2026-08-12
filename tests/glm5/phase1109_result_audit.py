#!/usr/bin/env python3
"""Audit Phase1109 artifacts and fail-closed interpretation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import phase1109_attention_routing_protocol as protocol


def recompute_digest(value: dict, field: str) -> str:
    copied = dict(value)
    copied.pop(field, None)
    return protocol.digest(copied)


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(root / "protocol" / "audit.json")
    final = protocol.read_json(root / "analysis" / "final_summary.json")
    decisions = protocol.read_json(root / "analysis" / "model_decisions.json")
    cross = protocol.read_json(root / "analysis" / "cross_model_topology.json")
    denial = protocol.read_json(root / "atlas" / "deepseek7b" / "denial.json")
    thresholds = prereg["thresholds"]
    checks = {}
    checks["preregistration_digest"] = (
        recompute_digest(prereg, "protocol_digest") == prereg["protocol_digest"]
    )
    checks["protocol_audit_digest"] = (
        recompute_digest(protocol_audit, "audit_digest") == protocol_audit["audit_digest"]
        and protocol_audit["all_checks_passed"]
    )
    checks["final_summary_digest"] = (
        recompute_digest(final, "final_summary_digest") == final["final_summary_digest"]
    )
    checks["cross_model_digest"] = (
        recompute_digest(cross, "cross_model_digest") == cross["cross_model_digest"]
    )
    checks["authorized_models_exact"] = tuple(final["authorized_models"]) == protocol.AUTHORIZED_MODELS
    checks["denied_model_exact"] = (
        tuple(final["denied_models"]) == protocol.DENIED_MODELS
        and denial["model"] == "deepseek7b"
        and denial["hidden_access"] is False
    )
    checks["source_authorization_bound"] = (
        final["source_behavior_authorization_digest"]
        == prereg["source"]["behavior_authorization_digest"]
        == denial["source_behavior_authorization_digest"]
    )
    summaries = {}
    arrays = {}
    units = {}
    for model in protocol.AUTHORIZED_MODELS:
        atlas = root / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        arrays[model] = np.load(atlas / "attention_routing_fields.npz")
        units[model] = protocol.read_json(atlas / "units.json")
    checks["model_summary_digests"] = all(
        recompute_digest(summaries[model], "summary_digest")
        == summaries[model]["summary_digest"]
        == final["model_summary_digests"][model]
        for model in protocol.AUTHORIZED_MODELS
    )
    checks["precision_and_instrument"] = all(
        summaries[model]["all_checks_passed"]
        and summaries[model]["precision"]["has_fp16_parameters"]
        and not summaries[model]["precision"]["has_bf16_parameters"]
        and not summaries[model]["precision"]["has_quantized_modules"]
        and summaries[model]["observed_attention_mass_finite_fraction"]
        >= thresholds["minimum_attention_finite_fraction"]
        and summaries[model]["deterministic_identity_maximum_error"]
        <= thresholds["maximum_deterministic_identity_error"]
        and summaries[model]["pre_selector_identity_maximum_error"]
        <= thresholds["maximum_pre_selector_identity_error"]
        for model in protocol.AUTHORIZED_MODELS
    )
    checks["atlas_shapes"] = all(
        arrays[model]["key_follow"].shape
        == (
            96, 2, 2, 2,
            summaries[model]["layer_count"],
            summaries[model]["head_count"],
            len(protocol.QUERY_ROLES),
        )
        and arrays[model]["key_total"].shape == arrays[model]["key_follow"].shape
        and arrays[model]["record_follow"].shape == arrays[model]["key_follow"].shape
        and arrays[model]["record_total"].shape == arrays[model]["key_follow"].shape
        and len(units[model]) == 96
        for model in protocol.AUTHORIZED_MODELS
    )
    checks["atlas_values_finite"] = all(
        all(np.isfinite(arrays[model][name]).all() for name in arrays[model].files)
        for model in protocol.AUTHORIZED_MODELS
    )
    checks["qualification_selection_frozen"] = all(
        0 < len(decisions[model]["selection"]["selected_events"])
        <= protocol.MAX_SELECTED_EVENTS
        and all(row["eligible"] for row in decisions[model]["selection"]["selected_events"])
        for model in protocol.AUTHORIZED_MODELS
    )
    checks["attention_address_pass_preserved"] = (
        final["prospective_predictions"]["P4"] is True
        and final["prospective_predictions"]["P5"] is True
        and all(decisions[model]["P4_attention_address_confirmation"] for model in protocol.AUTHORIZED_MODELS)
        and all(decisions[model]["P5_pair_breadth"] for model in protocol.AUTHORIZED_MODELS)
    )
    checks["execution_modulation_failure_preserved"] = (
        final["prospective_predictions"]["P6"] is False
        and all(not decisions[model]["P6_execution_modulation"] for model in protocol.AUTHORIZED_MODELS)
        and all(
            row["execution_modulation_mean"] < thresholds["minimum_execution_modulation"]
            for model in protocol.AUTHORIZED_MODELS
            for row in decisions[model]["confirmation"].values()
        )
    )
    cosine = cross["cosine"]
    mae = cross["mae"]
    checks["cross_model_failure_preserved"] = (
        final["prospective_predictions"]["P7"] is False
        and cross["passed"] is False
        and (
            cosine is None
            or cosine < thresholds["minimum_cross_model_curve_cosine"]
            or mae > thresholds["maximum_cross_model_curve_mae"]
        )
    )
    checks["causal_hard_stop"] = (
        final["prospective_predictions"]["P8"] is False
        and final["causal_staircase_authorized"] is False
        and final["component_head_qkv_neuron_localization_authorized"] is False
        and final["automatic_next_required"] is False
        and final["evidence"]["causal_edge"] == "not_added"
    )
    checks["canonical_theory_name_stable"] = (
        final["canonical_theory_name_unchanged"]
        == "conditional output-field closure theory"
    )
    result = {
        "schema_version": "phase1109_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_summary_digest": final["final_summary_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    audit_root = root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    protocol.write_json(audit_root / "result_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
