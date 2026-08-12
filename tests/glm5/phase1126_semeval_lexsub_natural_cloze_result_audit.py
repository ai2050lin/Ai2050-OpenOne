#!/usr/bin/env python3
"""Recompute structural and digest checks for Phase1126 outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1126_semeval_lexsub_natural_cloze_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    units = protocol.read_jsonl(protocol.OUT_ROOT / "analysis" / "interaction_units.jsonl")

    prereg_without_digest = dict(prereg)
    stored_protocol_digest = prereg_without_digest.pop("protocol_digest")
    final_without_digest = dict(final)
    stored_final_digest = final_without_digest.pop("final_digest")
    model_detail_checks = {}
    model_precision_checks = {}
    model_case_counts = {}
    for model_name in protocol.MODELS:
        summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model_name / "summary.json")
        details = protocol.read_jsonl(protocol.OUT_ROOT / "behavior" / model_name / "scores.jsonl")
        model_detail_checks[model_name] = protocol.digest(details) == summary["detail_digest"]
        precision = summary["precision"]
        model_precision_checks[model_name] = (
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
        )
        model_case_counts[model_name] = len(details)

    expected_units = len(protocol.BEHAVIOR_PARTITIONS) * protocol.PANELS_PER_PARTITION * len(protocol.REPLICAS) * len(protocol.MODELS)
    checks = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_valid": protocol.digest(prereg_without_digest) == stored_protocol_digest,
        "protocol_links_match": protocol_audit["protocol_digest"] == stored_protocol_digest == final["protocol_digest"],
        "source_hashes_valid": all(
            protocol.file_sha256(protocol.SOURCE_ROOT / name) == spec["sha256"]
            for name, spec in protocol.SOURCE_SPECS.items()
        ),
        "material_digest_valid": protocol.digest(protocol.read_json(protocol.OUT_ROOT / "protocol" / "selected_panels.json")["panels"]) == prereg["material_digest"],
        "all_model_details_valid": all(model_detail_checks.values()),
        "all_models_are_fp16_unquantized": all(model_precision_checks.values()),
        "all_case_counts_match": all(
            count == protocol_audit["expected_cases_per_model"]
            for count in model_case_counts.values()
        ),
        "unit_count_matches": len(units) == expected_units,
        "unit_digest_valid": protocol.digest(units) == final["unit_digest"],
        "final_digest_valid": protocol.digest(final_without_digest) == stored_final_digest,
        "authorized_model_count_consistent": (
            len(final["authorized_models"]) >= prereg["thresholds"]["models_required"]
        ) == final["predictions"]["P2_cross_resource_behavior"],
        "auto_continue_consistent": bool(final["auto_continue"]["value"]) == bool(final["predictions"]["P2_cross_resource_behavior"]),
        "holdout_not_scored": all(row["partition"] in protocol.BEHAVIOR_PARTITIONS for row in units),
        "no_hidden_artifacts_present": not (protocol.OUT_ROOT / "hidden").exists(),
    }
    audit = {
        "schema_version": "phase1126_semeval_lexsub_natural_cloze_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": stored_protocol_digest,
        "final_digest": stored_final_digest,
        "audit_digest": "",
    }
    audit["audit_digest"] = protocol.digest({key: value for key, value in audit.items() if key != "audit_digest"})
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
