#!/usr/bin/env python3
"""Audit Phase1077 result integrity and the nonblocking evidence contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1077_nonblocking_pattern_atlas_protocol as protocol


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
            atlas_root / "residual_mean_directions.fp16.npz"
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
        count_monotonic = all(
            int(conditioned_lookup[key]["semantic_pair_count"])
            <= int(value["semantic_pair_count"])
            for key, value in all_lookup.items()
        )
        recorded_nonfinite_candidates = sum(
            1 for row in behavior if not row["finite_candidate"]
        )
        hidden_observation_total = (
            2
            * int(summary["unit_count"])
            * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
        )
        numeric_metric_fields = (
            "semantic_direction_consistency",
            "mean_semantic_relative_magnitude",
            "mean_lexical_relative_magnitude",
            "mean_semantic_cross_surface_cosine",
            "mean_interaction_relative_magnitude",
        )
        metrics_finite_or_missing = all(
            value is None or np.isfinite(float(value))
            for row in metrics
            for value in (
                row.get(field) for field in numeric_metric_fields
            )
        )
        precision = summary["precision"]
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
            "candidate_nonfinite_accounted": summary[
                "nonfinite_candidate_count"
            ] == recorded_nonfinite_candidates,
            "hidden_nonfinite_accounted": (
                0
                <= int(summary["nonfinite_hidden_role_count"])
                <= hidden_observation_total
                and metrics_finite_or_missing
            ),
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
            "aggregate_direction_shape": arrays[
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
            "hidden_semantic_observation_finite_coverage": (
                1.0
                - summary["nonfinite_hidden_role_count"]
                / hidden_observation_total
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

    final_path = (
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    evidence_path = (
        protocol.OUT_ROOT
        / "analysis"
        / "family_evidence_ledger.jsonl"
    )
    regions_path = (
        protocol.OUT_ROOT / "analysis" / "consensus_regions.jsonl"
    )
    reuse_path = (
        protocol.OUT_ROOT / "analysis" / "reuse_matrix.json"
    )
    automatic_path = (
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    retrieval_path = (
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_family_retrieval.json"
    )
    final = protocol.read_json(final_path)
    evidence = protocol.read_jsonl(evidence_path)
    automatic = protocol.read_json(automatic_path)
    retrieval = protocol.read_json(retrieval_path)
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
        "automatic_next_false": not automatic["continue"],
        "new_mathematics_not_predeclared": not final[
            "mathematical_status"
        ]["new_mathematics_needed_now"],
        "posthoc_retrieval_not_an_evidence_gate": not retrieval[
            "preregistered_evidence_gate"
        ],
    }
    checks.update(final_checks)
    checks["all_model_audits_passed"] = all(
        row["all_checks_passed"] for row in model_audits.values()
    )
    for path in (
        final_path,
        evidence_path,
        regions_path,
        reuse_path,
        automatic_path,
        retrieval_path,
    ):
        files.append(result_file(path))
    checks["all_required_files_present"] = all(
        row["exists"] for row in files if row["required"]
    )

    payload = {
        "schema_version": "phase1077_integrity_audit.v1",
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
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json",
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


if __name__ == "__main__":
    main()
