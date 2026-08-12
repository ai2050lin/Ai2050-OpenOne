#!/usr/bin/env python3
"""Strict integrity audit for all Phase1073 artifacts."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1073_late_query_protocol as protocol


SCRIPT_NAMES = (
    "phase1073_late_query_protocol.py",
    "phase1073_behavior_calibration_protocol.py",
    "phase1073_behavior_calibration_scan.py",
    "phase1073_behavior_calibration_finalize.py",
    "phase1073_late_query_scan.py",
    "phase1073_finalize.py",
    "phase1073_posthoc_diagnostics.py",
    "phase1073_run_sequential.py",
    "phase1073_result_audit.py",
)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    calibration_prereg = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    calibration_audit = protocol.read_json(
        protocol.CALIBRATION_ROOT / "protocol" / "audit.json"
    )
    calibration_summary = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "analysis"
        / "calibration_summary.json"
    )
    atlas_summary = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "atlas_summary.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    posthoc = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "posthoc_diagnostics.json"
    )
    run_manifest = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "run_manifest.json"
    )
    source_automatic = protocol.read_json(
        protocol.SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    source_audit = protocol.read_json(
        protocol.SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )

    checks: dict[str, bool] = {}
    checks["protocol_audit_passed"] = bool(
        audit["all_checks_passed"]
    )
    checks["calibration_protocol_audit_passed"] = bool(
        calibration_audit["all_checks_passed"]
    )
    checks["protocol_digest_linked"] = bool(
        audit["protocol_digest"]
        == prereg["protocol_digest"]
        == atlas_summary["protocol_digest"]
        == posthoc["protocol_digest"]
    )
    checks["calibration_digest_linked"] = bool(
        calibration_audit["protocol_digest"]
        == calibration_prereg["protocol_digest"]
        == calibration_summary["protocol_digest"]
        == prereg["calibration_protocol_digest"]
        == atlas_summary["calibration_protocol_digest"]
    )
    checks["source_phase1072_preserved"] = bool(
        not source_automatic["should_continue_automatically"]
        and source_automatic["route"]
        == "stop_at_bidirectional_pattern_specificity"
        and source_audit["all_integrity_checks_passed"]
    )

    expected_formal = next(iter(prereg["case_counts"].values()))
    expected_calibration = next(
        iter(calibration_prereg["case_counts"].values())
    )
    expected_natural = (
        len(protocol.RELATION_NAMES)
        * protocol.NATURAL_AUDIT_PER_CONDITION
    )
    record_counts_valid = True
    precision_valid = True
    skeletons_valid_all = True
    model_details = {}
    for model in protocol.MODELS:
        formal_cases_path = (
            protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        calibration_cases_path = (
            protocol.CALIBRATION_ROOT
            / "protocol"
            / f"cases.{model}.jsonl"
        )
        formal_cases = protocol.read_jsonl(formal_cases_path)
        calibration_cases = protocol.read_jsonl(
            calibration_cases_path
        )
        formal_skeletons = {
            row["prompt_skeleton_sha256"] for row in formal_cases
        }
        calibration_skeletons = {
            row["prompt_skeleton_sha256"]
            for row in calibration_cases
        }
        skeleton_valid = calibration_skeletons <= formal_skeletons
        skeletons_valid_all = skeletons_valid_all and skeleton_valid

        calibration_atlas = (
            protocol.CALIBRATION_ROOT / "atlas" / model
        )
        formal_atlas = protocol.OUT_ROOT / "atlas" / model
        calibration_model = protocol.read_json(
            calibration_atlas / "summary.json"
        )
        formal_model = protocol.read_json(
            formal_atlas / "summary.json"
        )
        precision = formal_model["precision"]
        model_precision = bool(
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
            and formal_model["placement"]["quantization"] == "none"
        )
        precision_valid = precision_valid and model_precision
        counts = {
            "formal_cases": line_count(formal_cases_path),
            "calibration_cases": line_count(calibration_cases_path),
            "calibration_candidate": line_count(
                calibration_atlas / "candidate_behavior.jsonl"
            ),
            "calibration_natural": line_count(
                calibration_atlas / "natural_generation_audit.jsonl"
            ),
            "formal_candidate": line_count(
                formal_atlas / "candidate_behavior.jsonl"
            ),
            "formal_natural": line_count(
                formal_atlas / "natural_generation_audit.jsonl"
            ),
            "response_metrics": line_count(
                formal_atlas / "response_metrics.jsonl"
            ),
        }
        expected_response = (
            len(protocol.OPERATION_CONDITIONS)
            * len(protocol.SPLITS)
            * len(protocol.QUERY_TYPES)
            * int(formal_model["event_count"])
            * len(protocol.CAPTURE_ROLES)
            * 2
        )
        model_counts = bool(
            counts["formal_cases"] == expected_formal
            and counts["calibration_cases"] == expected_calibration
            and counts["calibration_candidate"] == expected_calibration
            and counts["calibration_natural"] == expected_calibration
            and counts["formal_candidate"] == expected_formal
            and counts["formal_natural"] == expected_natural
            and counts["response_metrics"] == expected_response
            and int(formal_model["case_count"]) == expected_formal
            and int(calibration_model["case_count"])
            == expected_calibration
        )
        record_counts_valid = record_counts_valid and model_counts
        model_details[model] = {
            "counts": counts,
            "expected_response_metrics": expected_response,
            "counts_valid": model_counts,
            "prompt_skeleton_linkage_valid": skeleton_valid,
            "fp16_no_quantization_valid": model_precision,
            "elapsed_seconds": {
                "calibration": calibration_model["elapsed_seconds"],
                "formal": formal_model["elapsed_seconds"],
            },
        }

    checks["all_calibration_skeletons_link_to_formal"] = (
        skeletons_valid_all
    )
    checks["all_model_record_counts_valid"] = record_counts_valid
    checks["all_models_fp16_no_quantization"] = precision_valid

    condition_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "condition_evidence.jsonl"
    )
    operation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "operation_evidence.jsonl"
    )
    relation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_gates.jsonl"
    )
    model_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "model_gates.jsonl"
    )
    cross_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "cross_model_profiles.jsonl"
    )
    checks["analysis_row_counts_valid"] = bool(
        len(condition_rows)
        == len(protocol.MODELS) * len(protocol.RELATION_NAMES)
        and len(operation_rows)
        == len(protocol.MODELS) * len(protocol.OPERATION_CONDITIONS)
        and len(relation_rows)
        == len(protocol.MODELS) * len(protocol.BASE_RELATIONS)
        and len(model_rows) == len(protocol.MODELS)
        and len(cross_rows)
        == len(protocol.MODELS) * (len(protocol.MODELS) - 1) // 2
    )
    checks["prebranch_gate_consistent"] = all(
        bool(row["checks"]["pre_branch_hard_control"])
        == (
            row["pre_branch_operation_contrast_max"] is not None
            and float(row["pre_branch_operation_contrast_max"])
            <= protocol.GATES["pre_branch_operation_contrast_max"]
        )
        for row in relation_rows
    )
    selected_models = [
        row["model"] for row in model_rows if row["model_gate_passed"]
    ]
    expected_continue = bool(
        len(selected_models)
        >= protocol.GATES["minimum_repeated_models"]
    )
    checks["automatic_decision_consistent"] = bool(
        automatic["selected_models"] == selected_models
        and automatic["repeated_model_count"] == len(selected_models)
        and automatic["should_continue_automatically"]
        == expected_continue
        and automatic["next_phase"]
        == (1074 if expected_continue else None)
    )
    checks["posthoc_did_not_change_automatic_decision"] = bool(
        posthoc["frozen_automatic_next"] == automatic
    )

    required_model_stages = [
        *(f"calibration_scan_{model}" for model in protocol.MODELS),
        *(f"formal_scan_{model}" for model in protocol.MODELS),
    ]
    stage_map = {
        row["label"]: row for row in run_manifest["stages"]
    }
    checks["sequential_manifest_model_stages_complete"] = all(
        label in stage_map
        and stage_map[label]["return_code"] == 0
        and stage_map[label]["completed_at_utc"] is not None
        for label in required_model_stages
    )
    chronological = True
    for prefix in ("calibration_scan", "formal_scan"):
        stages = [
            stage_map[f"{prefix}_{model}"]
            for model in protocol.MODELS
        ]
        for left, right in zip(stages, stages[1:]):
            chronological = chronological and (
                parse_time(left["completed_at_utc"])
                <= parse_time(right["started_at_utc"])
            )
    checks["model_scans_do_not_overlap"] = chronological
    checks["manifest_declares_fp16_no_quantization"] = bool(
        run_manifest["strictly_sequential"]
        and run_manifest["model_order"] == list(protocol.MODELS)
        and run_manifest["precision"] == "fp16"
        and run_manifest["quantization"] == "none"
    )

    tensor_suffixes = {
        ".pt", ".pth", ".npy", ".npz", ".safetensors", ".bin"
    }
    tensor_files = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in tensor_suffixes
    ]
    checks["no_raw_tensor_artifacts"] = not tensor_files

    script_hashes = {
        name: sha256(ROOT / "tests" / "glm5" / name)
        for name in SCRIPT_NAMES
    }
    files = [
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    ]
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema_version": "phase1073_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "calibration_protocol_digest": prereg[
            "calibration_protocol_digest"
        ],
        "checks": checks,
        "all_integrity_checks_passed": not failed,
        "failed_checks": failed,
        "model_details": model_details,
        "script_sha256": script_hashes,
        "raw_tensor_files": tensor_files,
        "file_count_before_this_audit": len(files),
        "total_bytes_before_this_audit": sum(
            path.stat().st_size for path in files
        ),
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json",
        result,
    )
    print(
        "Phase1073 integrity: "
        f"{result['all_integrity_checks_passed']} failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
