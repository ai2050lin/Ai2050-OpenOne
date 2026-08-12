#!/usr/bin/env python3
"""Audit Phase1080 artifacts and seal the automatic-next decision."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1080_natural_relevance_atlas_protocol as protocol


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def finite_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and math.isfinite(float(value))
    )


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    final_path = protocol.OUT_ROOT / "analysis" / "final_summary.json"
    final = protocol.read_json(final_path)
    prediction = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "prediction_audit.json"
    )
    assignments = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "exact_assignments.json"
    )
    automatic_path = protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    automatic = protocol.read_json(automatic_path)

    required = [
        protocol.OUT_ROOT / "protocol" / "preregistration.json",
        protocol.OUT_ROOT / "protocol" / "audit.json",
        protocol.OUT_ROOT / "analysis" / "final_summary.json",
        protocol.OUT_ROOT / "analysis" / "exact_assignments.json",
        protocol.OUT_ROOT / "analysis" / "factor_ratios.json",
        protocol.OUT_ROOT / "analysis" / "heldout_prediction.json",
        protocol.OUT_ROOT / "analysis" / "prediction_audit.json",
        protocol.OUT_ROOT / "analysis" / "family_evidence_ledger.jsonl",
        protocol.OUT_ROOT / "analysis" / "top_regions.jsonl",
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
    ]
    for model in protocol.MODELS:
        required.extend([
            protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl",
            protocol.OUT_ROOT / "protocol" / f"audit.{model}.json",
            protocol.OUT_ROOT / "atlas" / model / "summary.json",
            protocol.OUT_ROOT / "atlas" / model / "candidate_behavior.jsonl",
            protocol.OUT_ROOT / "atlas" / model / "natural_generation.jsonl",
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl",
            protocol.OUT_ROOT / "atlas" / model / "split_direction_repeat.jsonl",
        ])

    vector_fields = (
        "relevance", "presence", "total",
        "infer_answer", "decoy_answer", "direct_answer",
    )
    scalar_fields = (
        "surface", "shell", "relevance_answer_interaction",
        "relevance_cross_answer", "relevance_cross_surface",
        "relevance_cross_shell",
    )
    metric_columns = tuple(
        value
        for field in vector_fields
        for value in (
            f"mean_{field}_relative_magnitude",
            f"{field}_direction_consistency",
        )
    ) + tuple(f"mean_{field}" for field in scalar_fields)

    model_audits: dict[str, Any] = {}
    for model in protocol.MODELS:
        cases = protocol.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        case_audit = protocol.read_json(
            protocol.OUT_ROOT / "protocol" / f"audit.{model}.json"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        behavior = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "candidate_behavior.jsonl"
        )
        generations = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "natural_generation.jsonl"
        )
        metrics = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        split_rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "split_direction_repeat.jsonl"
        )
        expected_metric_count = (
            len(protocol.CONDITIONINGS) * len(protocol.FAMILIES)
            * len(protocol.SPLITS) * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
        )
        expected_split_count = (
            len(protocol.CONDITIONINGS) * len(protocol.FAMILIES)
            * int(summary["event_count"]) * len(protocol.CAPTURE_ROLES)
        )
        expected_generation_count = (
            len(protocol.FAMILIES) * len(protocol.SPLITS)
            * len(protocol.BRANCHES)
            * int(prereg["generation_units_per_family_split_branch"])
        )
        checks = {
            "protocol_digest_matches": (
                summary["protocol_digest"] == prereg["protocol_digest"]
            ),
            "case_digest_matches": (
                case_audit["case_digest"]
                == prereg["model_case_digests"][model]
                == protocol.digest(cases)
            ),
            "case_count": (
                len(cases) == len(behavior)
                == int(prereg["case_count_per_model"])
                == int(summary["case_count"])
            ),
            "unit_count": (
                len({row["unit_id"] for row in cases})
                == int(prereg["unit_count_per_model"])
                == int(summary["unit_count"])
            ),
            "generation_count": len(generations) == expected_generation_count,
            "generation_branch_coverage": {
                row["branch"] for row in generations
            } == set(protocol.BRANCHES),
            "metric_count": len(metrics) == expected_metric_count,
            "split_direction_count": len(split_rows) == expected_split_count,
            "metrics_finite_or_missing": all(
                finite_or_none(row[column])
                for row in metrics for column in metric_columns
            ),
            "model_protocol_audit_passed": case_audit["all_checks_passed"],
            "fp16_present": summary["precision"]["has_fp16_parameters"],
            "bf16_absent": not summary["precision"]["has_bf16_parameters"],
            "quantization_absent": not summary["precision"][
                "has_quantized_modules"
            ],
            "pre_branch_causal_order_zero": (
                float(summary["pre_branch_global_max_abs"])
                <= float(prereg["evidence_thresholds"]["pre_branch_tolerance"])
            ),
            "identity_repeat_exact": float(summary["identity_maximum"]) == 0.0,
            "all_families_retained": set(summary["behavior_summary"])
            == set(protocol.FAMILIES),
        }
        model_audits[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "case_count": len(cases),
            "unit_count": summary["unit_count"],
            "event_count": summary["event_count"],
            "nonfinite_candidate_count": summary["nonfinite_candidate_count"],
            "nonfinite_hidden_magnitude_role_count": summary[
                "nonfinite_hidden_magnitude_role_count"
            ],
            "elapsed_seconds": summary["elapsed_seconds"],
        }

    global_checks = {
        "all_required_files_present": all(path.exists() for path in required),
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_digest_matches": final["protocol_digest"]
        == prereg["protocol_digest"],
        "model_order_frozen": tuple(final["models"]) == protocol.MODELS,
        "case_count_total": int(final["case_count_total"])
        == int(prereg["case_count_per_model"]) * len(protocol.MODELS),
        "unit_count_total": int(final["unit_count_total"])
        == int(prereg["unit_count_per_model"]) * len(protocol.MODELS),
        "all_models_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
        "all_families_have_evidence": set(final["family_evidence"])
        == set(protocol.FAMILIES),
        "no_causal_level_assigned": all(
            row["highest_evidence_level"] != "L5"
            and row["causal_status"] == "not_tested"
            for row in final["family_evidence"].values()
        ),
        "prediction_keys_match": set(prediction["predictions"])
        == set(protocol.PROSPECTIVE_PREDICTIONS),
        "exact_permutation_count": all(
            int(row["permutation_count"])
            == math.factorial(len(protocol.BASE_FAMILIES))
            for row in assignments["rows"]
        ),
        "primary_population_finite": str(prereg["primary_population"])
        .startswith("All finite preregistered observations"),
        "new_mathematics_not_predeclared": final["mathematical_status"][
            "new_mathematics_needed_now"
        ] is False,
    }
    all_checks = all(global_checks.values()) and all(
        row["all_checks_passed"] for row in model_audits.values()
    )
    empirical_continue = bool(automatic["continue"])
    automatic["integrity_audit_pending"] = False
    automatic["integrity_audit_passed"] = all_checks
    automatic["continue"] = empirical_continue and all_checks
    if not all_checks:
        automatic["reason"] = (
            "Integrity audit failed. Automatic continuation is forbidden."
        )
    elif automatic["continue"]:
        automatic["reason"] = (
            "The frozen empirical gate and integrity audit passed."
        )
    else:
        automatic["reason"] = (
            "Integrity passed, but the frozen empirical gate failed. Preserve the descriptive atlas and do not select components or neurons."
        )
    automatic.pop("decision_digest", None)
    automatic["decision_digest"] = protocol.digest(automatic)
    protocol.write_json(automatic_path, automatic)

    final["automatic_next"] = automatic
    final.pop("summary_digest", None)
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(final_path, final)

    file_rows = [{
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    } for path in required]
    audit = {
        "schema_version": "phase1080_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": global_checks,
        "model_audits": model_audits,
        "automatic_next": automatic,
        "files": file_rows,
        "all_checks_passed": all_checks,
    }
    audit["audit_digest"] = protocol.digest(audit)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json", audit
    )
    if not all_checks:
        raise RuntimeError(json.dumps({
            "global": global_checks,
            "models": model_audits,
        }, indent=2))
    print({
        "phase": protocol.PHASE,
        "status": "audit_passed",
        "automatic_continue": automatic["continue"],
        "audit_digest": audit["audit_digest"],
    })


if __name__ == "__main__":
    main()
