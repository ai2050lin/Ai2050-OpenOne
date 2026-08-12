#!/usr/bin/env python3
"""Strict integrity audit for all formal Phase1071 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1071_exposure_pattern_protocol as protocol


SCRIPT_NAMES = (
    "phase1071_behavior_calibration_protocol.py",
    "phase1071_behavior_calibration_scan.py",
    "phase1071_behavior_calibration_finalize.py",
    "phase1071_exposure_pattern_protocol.py",
    "phase1071_exposure_pattern_scan.py",
    "phase1071_finalize.py",
    "phase1071_posthoc_diagnostics.py",
    "phase1071_result_audit.py",
)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value: Any) -> bool:
    return value is None or math.isfinite(float(value))


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    calibration_prereg = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    calibration_selection = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "analysis"
        / "prompt_selection.json"
    )
    automatic_next = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    posthoc = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "posthoc_diagnostics.json"
    )
    relation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_evidence.jsonl"
    )
    model_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "model_gates.jsonl"
    )

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["protocol_audit_passed"] = bool(
        protocol.read_json(
            protocol.OUT_ROOT / "protocol" / "audit.json"
        )["all_checks_passed"]
    )
    checks["calibration_protocol_audit_passed"] = bool(
        protocol.read_json(
            protocol.CALIBRATION_ROOT
            / "protocol"
            / "audit.json"
        )["all_checks_passed"]
    )
    checks["calibration_digest_linked"] = (
        prereg["calibration_protocol_digest"]
        == calibration_prereg["protocol_digest"]
        == calibration_selection["protocol_digest"]
    )
    checks["selected_style_linked"] = (
        int(prereg["selected_prompt_style"])
        == int(calibration_selection["selected_prompt_style"])
    )
    checks["model_gate_row_count"] = (
        len(model_rows) == len(protocol.MODELS)
    )
    checks["relation_evidence_row_count"] = (
        len(relation_rows)
        == len(protocol.MODELS) * len(protocol.RELATION_NAMES)
    )

    model_details = {}
    for model in protocol.MODELS:
        calibration_summary = protocol.read_json(
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "summary.json"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        candidate_path = (
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "candidate_behavior.jsonl"
        )
        natural_path = (
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "natural_generation_audit.jsonl"
        )
        response_path = (
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        readout_path = (
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "local_readout_metrics.jsonl"
        )
        calibration_candidate = (
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "candidate_behavior.jsonl"
        )
        calibration_natural = (
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "natural_generation_audit.jsonl"
        )
        expected_response = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * len(protocol.QUERY_TYPES)
            * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
            * 2
        )
        expected_readout = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * len(protocol.QUERY_TYPES)
            * int(summary["event_count"])
            * 2
        )
        local_checks = {
            "protocol_digest": (
                summary["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "calibration_digest": (
                calibration_summary["protocol_digest"]
                == calibration_prereg["protocol_digest"]
            ),
            "formal_fp16_no_quant": (
                bool(summary["precision"]["has_fp16_parameters"])
                and not summary["precision"]["has_bf16_parameters"]
                and not summary["precision"]["has_quantized_modules"]
            ),
            "case_count": (
                line_count(candidate_path)
                == prereg["case_count_per_model"]
                == summary["case_count"]
            ),
            "natural_count": (
                line_count(natural_path)
                == prereg["natural_audit_per_model"]
            ),
            "response_count": (
                line_count(response_path) == expected_response
            ),
            "readout_count": (
                line_count(readout_path) == expected_readout
            ),
            "calibration_candidate_count": (
                line_count(calibration_candidate)
                == calibration_prereg["case_count_per_model"]
            ),
            "calibration_natural_count": (
                line_count(calibration_natural)
                == calibration_prereg["case_count_per_model"]
            ),
        }
        checks[f"{model}_artifacts"] = all(local_checks.values())
        model_details[model] = {
            "checks": local_checks,
            "summary": {
                "candidate_finite_rate": summary[
                    "candidate_finite_rate"
                ],
                "residual_metric_finite_rate": summary[
                    "residual_metric_finite_rate"
                ],
                "internal_readout_finite_rate": summary[
                    "internal_readout_finite_rate"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            "expected_response_rows": expected_response,
            "expected_readout_rows": expected_readout,
        }
    details["models"] = model_details

    hard_negative_values = [
        split["hard_negative_process_did_max"]
        for row in relation_rows
        for split in (row["discovery"], row["confirmation"])
    ]
    checks["hard_negative_values_finite"] = all(
        finite_or_none(value) and value is not None
        for value in hard_negative_values
    )
    checks["hard_negative_gate_consistent"] = all(
        bool(split["checks"]["hard_negative_control"])
        == (
            split["hard_negative_process_did_max"] is not None
            and float(split["hard_negative_process_did_max"])
            <= prereg["gates"]["hard_negative_process_did_max"]
        )
        for row in relation_rows
        for split in (row["discovery"], row["confirmation"])
    )
    checks["automatic_decision_consistent"] = (
        automatic_next["selected_models"]
        == [
            row["model"]
            for row in model_rows
            if row["process_model_gate_passed"]
        ]
        and bool(automatic_next["should_continue_automatically"])
        == (
            len(automatic_next["selected_models"])
            >= prereg["gates"]["minimum_repeated_models"]
        )
    )
    checks["posthoc_preserves_frozen_decision"] = (
        posthoc["protocol_digest"] == prereg["protocol_digest"]
        and posthoc["frozen_automatic_decision_unchanged"]
        == automatic_next
    )
    tensor_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".npy"}
    tensor_artifacts = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in tensor_suffixes
    ]
    checks["no_raw_tensor_artifacts"] = not tensor_artifacts
    details["raw_tensor_artifacts"] = tensor_artifacts

    script_hashes = {
        name: sha256(ROOT / "tests" / "glm5" / name)
        for name in SCRIPT_NAMES
    }
    details["script_sha256"] = script_hashes
    result = {
        "schema_version": "phase1071_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "calibration_protocol_digest": calibration_prereg[
            "protocol_digest"
        ],
        "checks": checks,
        "details": details,
        "all_integrity_checks_passed": all(checks.values()),
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "all_integrity_checks_passed": result[
            "all_integrity_checks_passed"
        ],
        "failed_checks": [
            key for key, value in checks.items() if not value
        ],
    }), flush=True)
    if not result["all_integrity_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
