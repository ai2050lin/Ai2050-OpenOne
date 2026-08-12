#!/usr/bin/env python3
"""Audit Phase1074 protocol, outputs, precision, and decisions."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1074_polarity_dynamics_protocol as protocol


SCRIPT_NAMES = (
    "phase1074_polarity_dynamics_protocol.py",
    "phase1074_behavior_scan.py",
    "phase1074_behavior_finalize.py",
    "phase1074_dynamics_scan.py",
    "phase1074_finalize.py",
    "phase1074_run_sequential.py",
    "phase1074_posthoc_numeric_audit.py",
    "phase1074_result_audit.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    behavior_decision = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "automatic_next.json"
    )
    manifest = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "run_manifest.json"
    )
    posthoc_path = (
        protocol.OUT_ROOT
        / "analysis"
        / "glm4_nonfinite_posthoc_summary.json"
    )
    posthoc = (
        protocol.read_json(posthoc_path)
        if posthoc_path.exists()
        else None
    )

    model_details = {}
    behavior_counts_valid = True
    precision_valid = True
    dynamics_counts_valid = True
    routing_shapes_valid = True
    dynamics_ran = bool(
        behavior_decision["should_run_internal_dynamics"]
    )
    for model in protocol.MODELS:
        behavior_summary = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        candidate_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "candidate_behavior.jsonl"
        )
        natural_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "natural_generation.jsonl"
        )
        expected_natural = (
            len(protocol.RELATIONS)
            * len(protocol.TASKS)
            * len(protocol.PATHS)
            * len(protocol.LAYOUTS)
            * len(protocol.SPLITS)
            * 2
        )
        behavior_valid = bool(
            len(candidate_rows) == prereg["case_count_per_model"]
            and len(natural_rows) == expected_natural
            and behavior_summary["protocol_digest"]
            == prereg["protocol_digest"]
        )
        behavior_counts_valid = (
            behavior_counts_valid and behavior_valid
        )
        precision = behavior_summary["precision"]
        model_precision_valid = bool(
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
        )
        precision_valid = precision_valid and model_precision_valid
        detail = {
            "behavior_case_count": len(candidate_rows),
            "natural_case_count": len(natural_rows),
            "behavior_counts_valid": behavior_valid,
            "fp16_no_quantization_valid": model_precision_valid,
        }
        if dynamics_ran:
            dynamics_summary = protocol.read_json(
                protocol.OUT_ROOT
                / "dynamics"
                / model
                / "summary.json"
            )
            residual_rows = protocol.read_jsonl(
                protocol.OUT_ROOT
                / "dynamics"
                / model
                / "residual_unit_metrics.jsonl"
            )
            expected_residual = (
                prereg["internal_unit_count_per_model"]
                * (int(dynamics_summary["n_layers"]) + 1)
                * len(protocol.CAPTURE_ROLES)
            )
            residual_valid = len(residual_rows) == expected_residual
            dynamics_counts_valid = (
                dynamics_counts_valid and residual_valid
            )
            archive = np.load(
                protocol.OUT_ROOT
                / "dynamics"
                / model
                / "routing_aggregates.npz",
                allow_pickle=False,
            )
            expected_shape = (
                len(protocol.RELATIONS),
                len(protocol.SPLITS),
                len(protocol.PATHS),
                len(protocol.LAYOUTS),
                int(dynamics_summary["n_layers"]),
                int(dynamics_summary["n_heads"]),
                len(protocol.ATTENTION_DESTINATIONS),
                len(protocol.SOURCE_PAIRS),
                2,
                2,
            )
            shape_valid = bool(
                archive["sums"].shape == expected_shape
                and archive["counts"].shape == expected_shape
                and archive["positive_counts"].shape
                == expected_shape
            )
            routing_shapes_valid = (
                routing_shapes_valid and shape_valid
            )
            dynamics_precision = dynamics_summary["precision"]
            dynamics_precision_valid = bool(
                dynamics_precision["has_fp16_parameters"]
                and not dynamics_precision["has_bf16_parameters"]
                and not dynamics_precision[
                    "has_quantized_modules"
                ]
            )
            precision_valid = (
                precision_valid and dynamics_precision_valid
            )
            detail.update({
                "residual_metric_row_count": len(residual_rows),
                "expected_residual_metric_row_count": (
                    expected_residual
                ),
                "dynamics_counts_valid": residual_valid,
                "routing_shape": list(archive["sums"].shape),
                "routing_shape_valid": shape_valid,
                "dynamics_fp16_no_quantization_valid": (
                    dynamics_precision_valid
                ),
            })
        model_details[model] = detail

    expected_stages = (
        len(protocol.MODELS) * (2 if dynamics_ran else 1)
    )
    stage_pairs = [
        (row["stage"], row["model"])
        for row in manifest["model_stages"]
    ]
    stages_valid = bool(
        len(stage_pairs) == expected_stages
        and len(set(stage_pairs)) == expected_stages
        and all("completed_at_utc" in row for row in manifest[
            "model_stages"
        ])
    )
    decision_consistent = bool(
        automatic["selected_model_count"]
        == len(automatic["selected_models"])
        and automatic["should_continue_automatically"]
        == (
            automatic["selected_model_count"]
            >= prereg["gates"]["minimum_dynamic_models"]
        )
    )
    raw_tensor_files = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".pt", ".pth", ".bin", ".safetensors"}
    ]
    checks = {
        "protocol_audit_passed": bool(
            protocol_audit["all_checks_passed"]
        ),
        "protocol_digest_linked": (
            protocol_audit["protocol_digest"]
            == prereg["protocol_digest"]
            == behavior_decision["protocol_digest"]
            == automatic["protocol_digest"]
        ),
        "source_phase1073_decision_preserved": (
            prereg["source_phase1073_decision"]["route"]
            == "stop_at_late_query_operation_selection"
            and not prereg["source_phase1073_decision"][
                "should_continue_automatically"
            ]
        ),
        "behavior_row_counts_valid": behavior_counts_valid,
        "all_models_fp16_no_quantization": precision_valid,
        "dynamics_row_counts_valid": dynamics_counts_valid,
        "routing_aggregate_shapes_valid": routing_shapes_valid,
        "sequential_manifest_model_stages_complete": stages_valid,
        "manifest_declares_no_model_overlap": (
            manifest["concurrent_model_processes"] is False
        ),
        "automatic_decision_consistent": decision_consistent,
        "posthoc_did_not_change_automatic_decision": (
            posthoc is None
            or (
                posthoc["protocol_digest"]
                == prereg["protocol_digest"]
                and posthoc["automatic_decision_unchanged"]
            )
        ),
        "no_raw_tensor_artifacts": not raw_tensor_files,
    }
    files = [
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    ]
    payload = {
        "schema_version": "phase1074_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
        "failed_checks": [
            key for key, value in checks.items() if not value
        ],
        "dynamics_ran": dynamics_ran,
        "model_details": model_details,
        "script_sha256": {
            name: sha256(ROOT / "tests" / "glm5" / name)
            for name in SCRIPT_NAMES
        },
        "raw_tensor_files": raw_tensor_files,
        "posthoc_numeric_audit": posthoc,
        "file_count_before_this_audit": len(files),
        "total_bytes_before_this_audit": sum(
            path.stat().st_size for path in files
        ),
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json",
        payload,
    )
    if not payload["all_integrity_checks_passed"]:
        raise RuntimeError(
            f"Phase1074 integrity audit failed: "
            f"{payload['failed_checks']}"
        )
    print(
        f"Phase1074 integrity audit passed: "
        f"{len(files)} files"
    )


if __name__ == "__main__":
    main()
