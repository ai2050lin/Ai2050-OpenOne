#!/usr/bin/env python3
"""Audit Phase1096 provenance, result files, shapes, and frozen decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1096_comparison_dynamics_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    checks = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_consistent": (
            protocol_audit["protocol_digest"] == prereg["protocol_digest"]
            == behavior["protocol_digest"] == final["protocol_digest"]
        ),
        "behavior_digest_consistent": (
            final["behavior_authorization_digest"] == behavior["authorization_digest"]
        ),
        "all_models_present": set(final["ledgers"]) == set(protocol.MODELS),
        "all_gates_present": set(final["gates"]) == {f"P{index}" for index in range(1, 10)},
        "automatic_next_matches_execution_language_gates": (
            bool(final["automatic_next_required"])
            == bool(final["gates"]["P6"]["passed"] and final["gates"]["P7"]["passed"])
        ),
        "causal_matches_p9": bool(final["causal_authorized"]) == bool(final["gates"]["P9"]["passed"]),
    }
    model_records = {}
    expected_prefix = (
        len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS)
    )
    for model_name in protocol.MODELS:
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(atlas_root / "summary.json")
        with np.load(atlas_root / "three_ledger_fields.npz") as payload:
            shape = tuple(payload["direction_sum"].shape)
            count_shape = tuple(payload["direction_count"].shape)
            relative_shape = tuple(payload["relative_sum"].shape)
            finite_arrays = all(np.isfinite(payload[key]).all() for key in payload.files)
        expected_shape = (
            *expected_prefix,
            int(summary["event_count"]), len(protocol.CAPTURE_ROLES),
            len(protocol.SIGNED_FIELDS), protocol.SIGNED_PROJECTION_REPLICATES,
            protocol.SIGNED_PROJECTION_DIM,
        )
        model_checks = {
            "protocol_digest": summary["protocol_digest"] == prereg["protocol_digest"],
            "case_digest": summary["case_digest"] == prereg["model_case_digests"][model_name],
            "direction_shape": shape == expected_shape,
            "count_shape": count_shape == expected_shape[:-1],
            "relative_shape": relative_shape == expected_shape[:-2],
            "finite_arrays": finite_arrays,
            "fp16_no_quantization": (
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"]["has_bf16_parameters"]
                and not summary["precision"]["has_quantized_modules"]
            ),
            "identity_exact": summary["identity_maximum"] == 0.0,
            "pre_task_zero": summary["pre_task_control_execution_maximum"] <= prereg["evidence_thresholds"]["pre_task_tolerance"],
            "scan_behavior_readout_consistent": (
                abs(
                    float(summary["candidate_accuracy"])
                    - float(behavior["models"][model_name]["overall_candidate_accuracy"])
                )
                <= 0.02
            ),
        }
        model_records[model_name] = {
            "checks": model_checks,
            "all_checks_passed": all(model_checks.values()),
            "summary_digest": summary["summary_digest"],
            "scan_candidate_accuracy": summary["candidate_accuracy"],
            "behavior_candidate_accuracy": behavior["models"][model_name]["overall_candidate_accuracy"],
        }
    checks["all_model_results_passed"] = all(
        row["all_checks_passed"] for row in model_records.values()
    )
    result = {
        "schema_version": "phase1096_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_summary_digest": final["summary_digest"],
        "checks": checks,
        "models": model_records,
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "result_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "audit_digest": result["audit_digest"],
    })


if __name__ == "__main__":
    main()
