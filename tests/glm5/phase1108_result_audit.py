#!/usr/bin/env python3
"""Audit Phase1108 artifacts, digests, axes, and stop decision."""

from __future__ import annotations

import json

import numpy as np

import phase1108_exact_key_event_protocol as protocol


def digest_without(row: dict, key: str) -> str:
    value = dict(row)
    value.pop(key, None)
    return protocol.digest(value)


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    model_summaries = {
        model: protocol.read_json(protocol.OUT_ROOT / "atlas" / model / "summary.json")
        for model in behavior["authorized_models"]
    }
    checks = {}
    checks["preregistration_digest"] = (
        digest_without(prereg, "protocol_digest") == prereg["protocol_digest"]
    )
    checks["protocol_audit_digest"] = (
        digest_without(protocol_audit, "audit_digest") == protocol_audit["audit_digest"]
        and protocol_audit["all_checks_passed"]
    )
    checks["behavior_authorization_digest"] = (
        digest_without(behavior, "authorization_digest")
        == behavior["authorization_digest"]
    )
    checks["final_summary_digest"] = (
        digest_without(final, "final_summary_digest")
        == final["final_summary_digest"]
    )
    checks["source_digests_bound"] = (
        prereg["source_phase1104_protocol_digest"]
        == protocol.read_json(protocol.SOURCE_PREREG)["protocol_digest"]
        and prereg["source_phase1104_result_audit_digest"]
        == protocol.read_json(protocol.SOURCE_AUDIT)["audit_digest"]
        and prereg["source_phase1104_final_summary_digest"]
        == protocol.read_json(protocol.SOURCE_FINAL)["final_summary_digest"]
    )
    checks["authorized_models_exact"] = set(behavior["authorized_models"]) == {
        "qwen3", "glm4"
    }
    checks["cross_model_pairs_exact"] = (
        tuple(behavior["cross_model_pairs"]) == protocol.RELATION_PAIRS
    )
    checks["deepseek_hidden_excluded"] = not (
        protocol.OUT_ROOT / "atlas" / "deepseek7b" / "summary.json"
    ).exists()
    checks["model_summary_digests"] = all(
        digest_without(summary, "summary_digest") == summary["summary_digest"]
        and final["atlas_summary_digests"][model] == summary["summary_digest"]
        for model, summary in model_summaries.items()
    )
    checks["precision_and_numeric_audits"] = all(
        summary["precision"]["has_fp16_parameters"]
        and not summary["precision"]["has_bf16_parameters"]
        and not summary["precision"]["has_quantized_modules"]
        and summary["candidate_finite_fraction"] >= 0.95
        and summary["hidden_finite_fraction"] >= 0.97
        and summary["identity_maximum_error"] <= 1e-8
        and summary["pre_query_maximum_error"] <= 1e-8
        for summary in model_summaries.values()
    )
    axes_ok = True
    counts_ok = True
    for model, summary in model_summaries.items():
        arrays = np.load(
            protocol.OUT_ROOT / "atlas" / model / "signed_event_fields.npz"
        )
        expected = (
            len(summary["relation_pairs"]),
            len(summary["surfaces"]),
            len(summary["splits"]),
            summary["event_count"],
            len(summary["roles"]),
            len(summary["fields"]),
            protocol.SIGNED_PROJECTION_REPLICATES,
            protocol.SIGNED_PROJECTION_DIM,
        )
        axes_ok &= arrays["direction_sum"].shape == expected
        axes_ok &= arrays["direction_count"].shape == expected[:-1]
        axes_ok &= arrays["relative_sum"].shape == expected[:-2]
        axes_ok &= arrays["relative_count"].shape == expected[:-2]
        counts_ok &= bool(np.all(arrays["direction_count"] >= 0))
        counts_ok &= bool(np.all(arrays["relative_count"] >= 0))
        counts_ok &= bool(np.isfinite(arrays["direction_sum"]).all())
        counts_ok &= bool(np.isfinite(arrays["relative_sum"]).all())
    checks["atlas_axes"] = axes_ok
    checks["atlas_counts_and_values"] = counts_ok
    checks["qualification_selection_frozen"] = (
        final["qualification_selected_event"]["event_index"] == 21
        and final["qualification_selected_event"]["role"] == "selector_end"
    )
    predictions = final["prospective_predictions"]
    checks["prediction_vector_exact"] = predictions == {
        "P1": True,
        "P2": True,
        "P3": True,
        "P4": False,
        "P5": False,
        "P6": True,
        "P7": False,
    }
    checks["negative_gate_preserved"] = (
        not final["causal_staircase_authorized"]
        and not final["component_or_neuron_localization_authorized"]
        and not final["automatic_next_required"]
    )
    checks["posthoc_does_not_upgrade"] = (
        final["posthoc_relation_label_retrieval"]["evidence_status"].startswith(
            "posthoc descriptor only"
        )
        and not final["component_or_neuron_localization_authorized"]
    )
    result = {
        "schema_version": "phase1108_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_summary_digest": final["final_summary_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
