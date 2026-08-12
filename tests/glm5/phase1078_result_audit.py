#!/usr/bin/env python3
"""Audit Phase1078 result integrity and descriptive-evidence boundaries."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1078_shared_shell_pattern_atlas_protocol as protocol


def result_file(
    path: Path,
    *,
    required: bool = True,
) -> dict[str, Any]:
    sha256 = None
    if path.exists():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        sha256 = hasher.hexdigest()
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "required": required,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256,
    }


def main() -> None:
    prereg_path = (
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit_path = (
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    prereg = protocol.read_json(prereg_path)
    protocol_audit = protocol.read_json(protocol_audit_path)
    checks = {
        "protocol_audit_passed": protocol_audit[
            "all_checks_passed"
        ],
        "model_order_frozen": prereg["sequential_model_order"]
        == list(protocol.MODELS),
        "precision_fp16": prereg["precision"] == "fp16",
        "quantization_none": prereg["quantization"] == "none",
        "all_finite_is_primary": prereg["conditionings"][0]
        == "all_finite",
        "three_factor_state_count": len(prereg["states"]) == 8,
        "predictions_frozen": prereg["prospective_predictions"]
        == protocol.PROSPECTIVE_PREDICTIONS,
        "automatic_next_frozen_false": not prereg[
            "automatic_next"
        ]["continue"],
    }
    files = [
        result_file(prereg_path),
        result_file(protocol_audit_path),
    ]
    model_audits = {}
    for model_name in protocol.MODELS:
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        summary_path = atlas_root / "summary.json"
        case_path = (
            protocol.OUT_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        behavior_path = atlas_root / "candidate_behavior.jsonl"
        natural_path = atlas_root / "natural_generation_audit.jsonl"
        metrics_path = atlas_root / "response_metrics.jsonl"
        split_path = atlas_root / "split_direction_repeat.jsonl"
        directions_path = (
            atlas_root
            / "residual_mean_truth_directions.fp16.npz"
        )
        summary = protocol.read_json(summary_path)
        cases = protocol.read_jsonl(case_path)
        behavior = protocol.read_jsonl(behavior_path)
        natural = protocol.read_jsonl(natural_path)
        metrics = protocol.read_jsonl(metrics_path)
        split_rows = protocol.read_jsonl(split_path)
        arrays = np.load(directions_path)
        expected_metric_count = (
            len(protocol.CONDITIONINGS)
            * len(protocol.FAMILIES)
            * len(protocol.SPLITS)
            * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
        )
        all_lookup = {
            (
                row["family"],
                row["split"],
                row["event_id"],
                row["role"],
            ): row
            for row in metrics
            if row["conditioning"] == "all_finite"
        }
        conditioned_lookup = {
            (
                row["family"],
                row["split"],
                row["event_id"],
                row["role"],
            ): row
            for row in metrics
            if row["conditioning"] == "behavior_complete"
        }
        count_fields = (
            "truth_observation_count",
            "surface_observation_count",
            "shell_observation_count",
            "truth_surface_interaction_count",
            "truth_shell_interaction_count",
        )
        count_monotonic = all(
            int(conditioned_lookup[key][field])
            <= int(value[field])
            for key, value in all_lookup.items()
            for field in count_fields
        )
        numeric_fields = (
            "truth_direction_consistency",
            "mean_truth_relative_magnitude",
            "mean_surface_relative_magnitude",
            "mean_shell_relative_magnitude",
            "mean_truth_surface_interaction",
            "mean_truth_shell_interaction",
            "mean_truth_cross_surface_cosine",
            "mean_truth_cross_shell_cosine",
        )
        metrics_finite_or_missing = all(
            value is None or np.isfinite(float(value))
            for row in metrics
            for value in (
                row.get(field) for field in numeric_fields
            )
        )
        precision = summary["precision"]
        hidden_truth_total = (
            4
            * int(summary["unit_count"])
            * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
        )
        local_checks = {
            "protocol_digest_matches": summary["protocol_digest"]
            == prereg["protocol_digest"],
            "case_digest_matches": protocol.digest(cases)
            == prereg["model_case_digests"][model_name],
            "case_count": summary["case_count"]
            == prereg["case_count_per_model"],
            "unit_count": summary["unit_count"]
            == prereg["unit_count_per_model"],
            "candidate_record_count": len(behavior)
            == prereg["case_count_per_model"],
            "natural_record_count": len(natural)
            == (
                len(protocol.FAMILIES)
                * len(protocol.SPLITS)
                * prereg[
                    "natural_audit_cases_per_family_split"
                ]
            ),
            "metric_record_count": len(metrics)
            == expected_metric_count,
            "split_direction_record_count": len(split_rows)
            == expected_metric_count // len(protocol.SPLITS),
            "hidden_nonfinite_accounted": (
                0
                <= int(summary[
                    "nonfinite_hidden_truth_role_count"
                ])
                <= hidden_truth_total
            ),
            "metrics_finite_or_missing": metrics_finite_or_missing,
            "identity_repeat_exact": summary["identity_maximum"] <= 1e-6,
            "has_fp16_parameters": precision[
                "has_fp16_parameters"
            ],
            "has_no_bf16_parameters": not precision[
                "has_bf16_parameters"
            ],
            "has_no_quantized_modules": not precision[
                "has_quantized_modules"
            ],
            "all_finite_not_smaller_than_behavior_ledger": (
                count_monotonic
            ),
            "both_conditionings_present": {
                row["conditioning"] for row in metrics
            } == set(protocol.CONDITIONINGS),
            "all_families_retained": all(
                value["atlas_retained_regardless_of_behavior"]
                for value in summary["families"].values()
            ),
            "residual_direction_shape": arrays[
                "mean_directions"
            ].shape[:3] == (
                len(protocol.CONDITIONINGS),
                len(protocol.FAMILIES),
                len(protocol.SPLITS),
            ),
        }
        model_audits[model_name] = {
            "checks": local_checks,
            "all_checks_passed": all(local_checks.values()),
            "case_count": summary["case_count"],
            "unit_count": summary["unit_count"],
            "event_count": summary["event_count"],
            "candidate_finite_coverage": (
                1.0
                - summary["nonfinite_candidate_count"]
                / summary["case_count"]
            ),
            "hidden_truth_finite_coverage": (
                1.0
                - summary["nonfinite_hidden_truth_role_count"]
                / hidden_truth_total
            ),
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for path in (
            case_path,
            summary_path,
            behavior_path,
            natural_path,
            metrics_path,
            split_path,
            directions_path,
        ):
            files.append(result_file(path))

    analysis_root = protocol.OUT_ROOT / "analysis"
    final_path = analysis_root / "final_summary.json"
    evidence_path = analysis_root / "family_evidence_ledger.jsonl"
    regions_path = analysis_root / "consensus_regions.jsonl"
    assignments_path = (
        analysis_root / "exact_permutation_assignment.json"
    )
    controls_path = analysis_root / "factor_control_ratios.json"
    alignment_path = analysis_root / "generic_truth_alignment.json"
    prediction_path = analysis_root / "prediction_audit.json"
    automatic_path = analysis_root / "automatic_next.json"
    final = protocol.read_json(final_path)
    evidence = protocol.read_jsonl(evidence_path)
    assignments = protocol.read_json(assignments_path)
    prediction = protocol.read_json(prediction_path)
    automatic = protocol.read_json(automatic_path)
    final_checks = {
        "final_protocol_digest_matches": final["protocol_digest"]
        == prereg["protocol_digest"],
        "all_models_present": final["models"] == list(protocol.MODELS),
        "all_families_have_evidence": {
            row["family"] for row in evidence
        } == set(protocol.FAMILIES),
        "no_causal_level_assigned": all(
            row["highest_evidence_level"] != "L5"
            and row["causal_status"] == "not_tested"
            for row in evidence
        ),
        "all_families_retained_in_final": all(
            row["retained_in_atlas"] for row in evidence
        ),
        "exact_permutation_count": all(
            row["exact_permutation_count"]
            == math_factorial(len(protocol.FAMILIES))
            for row in assignments["rows"]
        ),
        "prediction_keys_match_preregistration": set(
            prediction["predictions"]
        ) == set(prereg["prospective_predictions"]),
        "automatic_next_false": not automatic["continue"],
        "new_mathematics_not_predeclared": not final[
            "mathematical_status"
        ]["new_mathematics_needed_now"],
    }
    checks.update(final_checks)
    checks["all_model_audits_passed"] = all(
        row["all_checks_passed"]
        for row in model_audits.values()
    )
    for path in (
        final_path,
        evidence_path,
        regions_path,
        assignments_path,
        controls_path,
        alignment_path,
        prediction_path,
        automatic_path,
    ):
        files.append(result_file(path))
    checks["all_required_files_present"] = all(
        row["exists"] for row in files if row["required"]
    )

    payload = {
        "schema_version": "phase1078_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "model_audits": model_audits,
        "files": files,
        "all_checks_passed": (
            all(checks.values())
            and all(
                row["all_checks_passed"]
                for row in model_audits.values()
            )
        ),
    }
    payload["audit_digest"] = protocol.digest(payload)
    protocol.write_json(
        analysis_root / "integrity_audit.json",
        payload,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "all_checks_passed": payload["all_checks_passed"],
        "failed_checks": [
            key for key, value in checks.items() if not value
        ],
        "failed_models": [
            model
            for model, row in model_audits.items()
            if not row["all_checks_passed"]
        ],
        "audit_digest": payload["audit_digest"],
    }, ensure_ascii=False), flush=True)
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def math_factorial(value: int) -> int:
    result = 1
    for item in range(2, value + 1):
        result *= item
    return result


if __name__ == "__main__":
    main()
