#!/usr/bin/env python3
"""Audit Phase1082 revision-2 artifacts without changing frozen thresholds."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1082_semantic_output_operation_world_protocol as protocol


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
    check(prereg["protocol_revision"] == 2, "protocol_revision", failures)
    check(prereg["precision"] == "fp16", "fp16_preregistered", failures)
    check(prereg["quantization"] == "none", "no_quantization", failures)
    check(
        prereg["case_count_per_model"] == 12288,
        "case_count_preregistered",
        failures,
    )
    check(
        len(prereg["cells"]) == 32 and len(prereg["states"]) == 32,
        "complete_factor_cross",
        failures,
    )

    model_audit: dict[str, Any] = {}
    for model in protocol.MODELS:
        root = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(root / "summary.json")
        candidate = protocol.read_jsonl(root / "candidate_behavior.jsonl")
        generated = protocol.read_jsonl(root / "natural_generation.jsonl")
        metrics = protocol.read_jsonl(root / "response_metrics.jsonl")
        repeats = protocol.read_jsonl(root / "split_direction_repeat.jsonl")
        precision = summary["precision"]
        model_checks = {
            "protocol_digest": summary["protocol_digest"]
            == prereg["protocol_digest"],
            "case_count": summary["case_count"]
            == prereg["case_count_per_model"],
            "unit_count": summary["unit_count"]
            == prereg["unit_count_per_model"],
            "candidate_rows": len(candidate)
            == prereg["case_count_per_model"],
            "generation_rows": len(generated)
            == len(protocol.CELLS) * len(protocol.SPLITS)
            * protocol.GENERATION_UNITS_PER_FAMILY_SPLIT,
            "metrics_nonempty": bool(metrics),
            "repeat_rows_nonempty": bool(repeats),
            "fp16": precision["has_fp16_parameters"],
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
            "candidate_row_count": len(candidate),
            "generation_row_count": len(generated),
            "metric_row_count": len(metrics),
            "repeat_row_count": len(repeats),
            "summary_digest": summary["summary_digest"],
        }

    analysis_root = protocol.OUT_ROOT / "analysis"
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
        "posthoc_control_diagnostic.json",
        "top_regions.jsonl",
        "final_summary.json",
    )
    for filename in required:
        check((analysis_root / filename).is_file(), f"analysis.{filename}", failures)
    final = protocol.read_json(analysis_root / "final_summary.json")
    check(final["protocol_digest"] == prereg["protocol_digest"], "final_digest", failures)
    check(final["case_count_total"] == 36864, "final_case_count", failures)
    check(final["unit_count_total"] == 1152, "final_unit_count", failures)
    check(
        final["status"] == "complete_descriptive_noncausal",
        "noncausal_status",
        failures,
    )
    check(
        not final["automatic_next"]["local_causal_authorized"]
        or final["predictions"]["passed_count"] == 9,
        "causal_gate_consistency",
        failures,
    )

    result = {
        "schema_version": "phase1082_result_audit.v2",
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
