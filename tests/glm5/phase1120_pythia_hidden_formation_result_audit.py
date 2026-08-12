#!/usr/bin/env python3
"""Recompute and audit all frozen Phase1120 results."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path
from typing import Any

import numpy as np

import phase1120_pythia_hidden_formation_finalize as finalize_module
import phase1120_pythia_hidden_formation_protocol as protocol


SCRIPT_NAMES = (
    "phase1120_pythia_hidden_formation_protocol.py",
    "phase1120_pythia_hidden_formation_behavior.py",
    "phase1120_pythia_hidden_formation_finalize.py",
    "phase1120_pythia_hidden_formation_result_audit.py",
)


def audit() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    final_before = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    checks: dict[str, bool] = {}

    prereg_core = {key: value for key, value in prereg.items() if key != "protocol_digest"}
    checks["protocol_digest"] = protocol.digest(prereg_core) == prereg["protocol_digest"]
    checks["protocol_audit"] = protocol_audit["all_checks_passed"]
    checks["protocol_audit_digest"] = protocol.digest(
        {key: value for key, value in protocol_audit.items() if key != "audit_digest"}
    ) == protocol_audit["audit_digest"]
    checks["case_digest"] = protocol.digest(
        list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    ) == prereg["case_digest"]
    projection_path = protocol.OUT_ROOT / "protocol" / "projection_matrix.npy"
    checks["projection_digest"] = protocol.file_sha256(projection_path) == prereg["projection"]["sha256"]
    projection = np.load(projection_path, allow_pickle=False)
    checks["projection_shape"] = projection.shape == (protocol.HIDDEN_SIZE, protocol.PROJECTION_DIM)
    checks["projection_finite"] = bool(np.isfinite(projection).all())
    checks["checkpoint_set"] = prereg["checkpoints"] == list(protocol.CHECKPOINTS)
    checks["eligible_layers"] = prereg["eligible_layer_indices"] == list(protocol.ELIGIBLE_LAYER_INDICES)
    checks["terminal_excluded"] = protocol.HIDDEN_STATE_COUNT - 1 not in prereg["eligible_layer_indices"]

    expected_shapes = {
        "case_indices": (prereg["case_count"],),
        "true_z": (prereg["case_count"], prereg["hidden_state_count"]),
        "control_z": (prereg["case_count"], prereg["hidden_state_count"]),
        "state_projection": (prereg["case_count"], prereg["hidden_state_count"], prereg["projection"]["dimension"]),
        "final_selected_logit_error": (prereg["case_count"], 4),
    }
    for checkpoint in protocol.CHECKPOINTS:
        root = protocol.OUT_ROOT / "hidden" / checkpoint
        summary = protocol.read_json(root / "summary.json")
        summary_core = {key: value for key, value in summary.items() if key != "summary_digest"}
        artifact_path = protocol.OUT_ROOT / summary["artifact"]
        checks[f"{checkpoint}_summary_digest"] = protocol.digest(summary_core) == summary["summary_digest"]
        checks[f"{checkpoint}_artifact_digest"] = protocol.file_sha256(artifact_path) == summary["artifact_sha256"]
        checks[f"{checkpoint}_protocol_digest"] = summary["protocol_digest"] == prereg["protocol_digest"]
        checks[f"{checkpoint}_parameter_probe"] = (
            summary["parameter_probe"]["digest"] == summary["expected_parameter_probe_digest"]
        )
        checks[f"{checkpoint}_precision"] = (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        )
        checks[f"{checkpoint}_finite"] = summary["finite_fraction"] >= prereg["thresholds"]["minimum_finite_fraction"]
        checks[f"{checkpoint}_logit_reproduction"] = (
            summary["maximum_final_selected_logit_error"]
            <= prereg["thresholds"]["maximum_final_logit_reproduction_error"]
        )
        with np.load(artifact_path, allow_pickle=False) as data:
            checks[f"{checkpoint}_array_names"] = set(data.files) == set(expected_shapes)
            checks[f"{checkpoint}_array_shapes"] = all(data[name].shape == shape for name, shape in expected_shapes.items())
            checks[f"{checkpoint}_arrays_finite"] = all(np.isfinite(data[name]).all() for name in expected_shapes)
            checks[f"{checkpoint}_case_order"] = np.array_equal(
                data["case_indices"], np.arange(prereg["case_count"], dtype=np.int32)
            )

    final_core = {key: value for key, value in final_before.items() if key != "final_digest"}
    checks["final_digest"] = protocol.digest(final_core) == final_before["final_digest"]
    checks["instrument_passed"] = bool(final_before["instrument_passed"])
    checks["output_reproduction_reported"] = all(
        key in final_before["output_reproduction"]
        for key in (
            "source_direction_accuracy",
            "reproduced_direction_accuracy",
            "source_control_advantage",
            "reproduced_control_advantage",
            "aggregate_exact_match",
            "interpretation",
        )
    )
    checks["readout_layer_eligible"] = int(final_before["readout_selection"]["selected"]["layer_index"]) in protocol.ELIGIBLE_LAYER_INDICES
    checks["geometry_layer_eligible"] = int(final_before["geometry_selection"]["selected"]["layer_index"]) in protocol.ELIGIBLE_LAYER_INDICES
    checks["readout_confirmation_split_set"] = {
        value["split"] for value in final_before["readout_confirmation_gates"]
    } == {"independent_confirmation", "heldout"}
    checks["geometry_confirmation_split_set"] = {
        value["split"] for value in final_before["geometry_confirmation_gates"]
    } == {"independent_confirmation", "heldout"}
    checks["prediction_p3_consistency"] = (
        final_before["prospective_predictions"]["P3"] == ("pass" if final_before["readout_event_passed"] else "fail")
    )
    checks["prediction_p4_consistency"] = (
        final_before["prospective_predictions"]["P4"] == ("pass" if final_before["geometry_event_passed"] else "fail")
    )
    checks["joint_event_consistency"] = final_before["joint_event_passed"] == (
        final_before["instrument_passed"]
        and final_before["readout_event_passed"]
        and final_before["geometry_event_passed"]
    )
    checks["no_same_phase_component_scan"] = not final_before["automatic_continuation"]["run_component_or_causal_in_phase1120"]

    rerun = finalize_module.finalize()
    checks["deterministic_recompute"] = rerun["final_digest"] == final_before["final_digest"]
    for script_name in SCRIPT_NAMES:
        script_path = protocol.ROOT / "tests" / "glm5" / script_name
        try:
            py_compile.compile(str(script_path), doraise=True)
            checks[f"compile_{script_name}"] = True
        except py_compile.PyCompileError:
            checks[f"compile_{script_name}"] = False

    audit_core = {
        "schema_version": "phase1120_pythia_hidden_formation_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final_before["final_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    result = dict(audit_core)
    result["audit_digest"] = protocol.digest(audit_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    if not result["all_checks_passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Phase1120 result audit failed: {failed}")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
