#!/usr/bin/env python3
"""Audit all Phase1084 artifacts without changing frozen thresholds."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1084_two_entity_attribute_protocol as protocol


def check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def main() -> None:
    failures: list[str] = []
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    check(static["all_checks_passed"], "static_protocol", failures)
    check(prereg["protocol_revision"] == 1, "protocol_revision", failures)
    check(prereg["precision"] == "fp16", "fp16_preregistered", failures)
    check(prereg["quantization"] == "none", "no_quantization", failures)
    check(prereg["case_count_per_model"] == 12288, "case_count", failures)
    check(len(prereg["states"]) == 32, "complete_factor_cross", failures)
    check(
        prereg["capture_scope"]["relative_depth_min"] == 1.0 / 3.0,
        "middle_depth_min",
        failures,
    )
    check(
        prereg["capture_scope"]["relative_depth_max"] == 2.0 / 3.0,
        "middle_depth_max",
        failures,
    )

    behavior_stop = (
        protocol.OUT_ROOT / "analysis" / "behavior_stop_summary.json"
    )
    model_audit: dict[str, Any] = {}
    for model in protocol.MODELS:
        pilot = protocol.read_json(protocol.OUT_ROOT / "pilot" / f"{model}.json")
        if behavior_stop.is_file():
            precision = pilot["precision"]
            model_checks = {
                "pilot_protocol_digest": pilot["protocol_digest"] == prereg["protocol_digest"],
                "pilot_case_digest": pilot["case_digest"] == prereg["model_case_digests"][model],
                "pilot_candidate_finite": pilot["candidate_finite_fraction"] >= 0.95,
                "fp16": bool(precision["has_fp16_parameters"]),
                "no_bf16": not precision["has_bf16_parameters"],
                "no_quantized_modules": not precision["has_quantized_modules"],
            }
            for name, passed in model_checks.items():
                check(bool(passed), f"{model}.{name}", failures)
            model_audit[model] = {
                "checks": model_checks,
                "pilot_result_digest": pilot["result_digest"],
                "passing_operations": pilot["passing_operations"],
            }
            continue
        root = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(root / "summary.json")
        metrics = protocol.read_jsonl(root / "response_metrics.jsonl")
        precision = summary["precision"]
        model_checks = {
            "pilot_protocol_digest": pilot["protocol_digest"] == prereg["protocol_digest"],
            "pilot_case_digest": pilot["case_digest"] == prereg["model_case_digests"][model],
            "pilot_behavior_gate_recorded": isinstance(
                pilot["model_behavior_gate_passed"], bool
            ),
            "pilot_candidate_finite": pilot["candidate_finite_fraction"] >= 0.95,
            "atlas_protocol_digest": summary["protocol_digest"] == prereg["protocol_digest"],
            "case_count": summary["case_count"] == prereg["case_count_per_model"],
            "unit_count": summary["unit_count"] == prereg["unit_count_per_model"],
            "metrics_nonempty": bool(metrics),
            "middle_scope_only": bool(metrics) and all(
                protocol.TARGET_RELATIVE_DEPTH_MIN
                <= float(row["relative_depth"])
                <= protocol.TARGET_RELATIVE_DEPTH_MAX
                for row in metrics
            ),
            "capture_roles_exact": bool(metrics) and {
                row["role"] for row in metrics
            } == set(protocol.CAPTURE_ROLES),
            "fp16": bool(precision["has_fp16_parameters"]),
            "no_bf16": not precision["has_bf16_parameters"],
            "no_quantized_modules": not precision["has_quantized_modules"],
            "identity_repeat": float(summary["identity_maximum"]) <= 1e-8,
            "pre_query_zero": float(summary["pre_query_global_max_abs"])
            <= protocol.EVIDENCE_THRESHOLDS["pre_query_tolerance"],
        }
        for name, passed in model_checks.items():
            check(bool(passed), f"{model}.{name}", failures)
        model_audit[model] = {
            "checks": model_checks,
            "summary_digest": summary["summary_digest"],
            "pilot_result_digest": pilot["result_digest"],
            "metric_row_count": len(metrics),
        }

    analysis_root = protocol.OUT_ROOT / "analysis"
    if behavior_stop.is_file():
        required = (
            "behavior_audit.json",
            "prediction_audit.json",
            "automatic_next.json",
            "behavior_stop_summary.json",
        )
        for filename in required:
            check((analysis_root / filename).is_file(), f"analysis.{filename}", failures)
        final = protocol.read_json(behavior_stop)
        automatic = protocol.read_json(analysis_root / "automatic_next.json")
        check(final["protocol_digest"] == prereg["protocol_digest"], "final_digest", failures)
        check(
            final["status"] == "stopped_before_hidden_scan_behavior_gate",
            "behavior_stop_status",
            failures,
        )
        check(not final["predictions"]["behavior_gate_passed"], "P2_failed", failures)
        check(not automatic["hidden_scan_authorized"], "hidden_stop_enforced", failures)
        check(not automatic["full_atlas_authorized"], "full_stop_enforced", failures)
        result = {
            "schema_version": "phase1084_result_audit.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "status": "behavior_stop_audited",
            "models": model_audit,
            "required_analysis_files": list(required),
            "failure_count": len(failures),
            "failures": failures,
            "all_checks_passed": not failures,
        }
        result["audit_digest"] = protocol.digest(result)
        protocol.write_json(analysis_root / "result_audit.json", result)
        print({
            "phase": protocol.PHASE,
            "status": result["status"],
            "all_checks_passed": result["all_checks_passed"],
            "failure_count": result["failure_count"],
            "audit_digest": result["audit_digest"],
        })
        if failures:
            raise SystemExit(1)
        return

    required = (
        "exact_assignments.json",
        "behavior_audit.json",
        "factor_ratio_audit.json",
        "integrity_audit.json",
        "operation_world_decomposition.json",
        "prediction_audit.json",
        "operation_evidence.json",
        "heldout_world_audit.json",
        "cross_world_advantage.json",
        "cross_model_audit.json",
        "automatic_next.json",
        "top_regions.jsonl",
        "final_summary.json",
    )
    for filename in required:
        check((analysis_root / filename).is_file(), f"analysis.{filename}", failures)
    final = protocol.read_json(analysis_root / "final_summary.json")
    automatic = protocol.read_json(analysis_root / "automatic_next.json")
    check(final["protocol_digest"] == prereg["protocol_digest"], "final_digest", failures)
    check(final["case_count_total"] == 36864, "final_case_count", failures)
    check(final["unit_count_total"] == 1152, "final_unit_count", failures)
    check(final["status"] == "complete_descriptive_noncausal", "status", failures)
    check(not automatic["local_causal_authorized"], "causal_forbidden", failures)
    if automatic["full_atlas_authorized"]:
        predictions = final["predictions"]["predictions"]
        check(
            all(predictions[name]["passed"] for name in (
                "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9"
            )),
            "full_atlas_gate_consistency",
            failures,
        )

    result = {
        "schema_version": "phase1084_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": model_audit,
        "required_analysis_files": list(required),
        "failure_count": len(failures),
        "failures": failures,
        "all_checks_passed": not failures,
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(analysis_root / "result_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "failure_count": result["failure_count"],
        "audit_digest": result["audit_digest"],
    })
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
