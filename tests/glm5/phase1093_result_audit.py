#!/usr/bin/env python3
"""Audit Phase1093 digests, lineage, frozen gates, and causal status."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1093_independent_relation_protocol as protocol


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_digest(path: Path, key: str) -> tuple[bool, str]:
    row = protocol.read_json(path)
    expected = str(row.pop(key))
    actual = protocol.digest(row)
    return expected == actual, expected


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    static = protocol.read_json(root / "protocol" / "audit.json")
    behavior = protocol.read_json(root / "analysis" / "behavior_authorization.json")
    summary = protocol.read_json(root / "analysis" / "final_summary.json")
    automatic = protocol.read_json(root / "analysis" / "automatic_next.json")

    digest_specs = {
        "protocol/preregistration.json": "protocol_digest",
        "protocol/audit.json": "audit_digest",
        "analysis/behavior_authorization.json": "summary_digest",
        "analysis/within_language_identity.json": "within_digest",
        "analysis/cross_language_identity_geometry.json": "cross_language_digest",
        "analysis/heldout_world_geometry.json": "heldout_digest",
        "analysis/cross_model_geometry.json": "cross_model_digest",
        "analysis/cross_phase_geometry.json": "cross_phase_digest",
        "analysis/dictionary_alignment.json": "alignment_digest",
        "analysis/shared_field.json": "shared_field_digest",
        "analysis/control_audit.json": "control_digest",
        "analysis/decomposition.json": "decomposition_digest",
        "analysis/physical_map.json": "physical_map_digest",
        "analysis/size_physical_map.json": "size_map_digest",
        "analysis/projection_audit.json": "projection_digest",
        "analysis/numeric_audit.json": "numeric_digest",
        "analysis/posthoc_template_robustness.json": "posthoc_digest",
        "analysis/posthoc_incidence_audit.json": "incidence_audit_digest",
        "analysis/behavior_failure_diagnostics.json": "diagnostic_digest",
        "analysis/automatic_next.json": "automatic_next_digest",
        "analysis/final_summary.json": "summary_digest",
    }
    digest_checks = {}
    digest_values = {}
    for relative, key in digest_specs.items():
        passed, value = verify_digest(root / relative, key)
        digest_checks[relative] = passed
        digest_values[relative] = value

    model_rows = {}
    for model_name in protocol.MODELS:
        cases = protocol.read_jsonl(
            root / "protocol" / f"cases.{model_name}.jsonl"
        )
        audit = protocol.read_json(
            root / "protocol" / f"audit.{model_name}.json"
        )
        pilot = protocol.read_json(root / "pilot" / f"{model_name}.json")
        atlas = protocol.read_json(root / "atlas" / model_name / "summary.json")
        npz = root / "atlas" / model_name / "signed_fields.npz"
        model_rows[model_name] = {
            "case_count": len(cases),
            "case_digest_matches": protocol.digest(cases) == audit["case_digest"],
            "pilot_result_digest_matches": (
                verify_digest(root / "pilot" / f"{model_name}.json", "result_digest")[0]
            ),
            "atlas_summary_digest_matches": (
                verify_digest(
                    root / "atlas" / model_name / "summary.json", "summary_digest"
                )[0]
            ),
            "npz_sha256": sha256_file(npz),
            "summary_npz_sha256_matches": (
                sha256_file(npz) == summary["models"][model_name]["npz_sha256"]
            ),
            "protocol_digest_matches": (
                pilot["protocol_digest"] == prereg["protocol_digest"]
                and atlas["protocol_digest"] == prereg["protocol_digest"]
            ),
        }

    checks = {
        "all_json_digests_valid": all(digest_checks.values()),
        "all_static_audits_passed": bool(static["all_checks_passed"]),
        "protocol_lineage_matches": (
            summary["source_phase1092_summary_digest"]
            == prereg["source_phase1092_summary_digest"]
        ),
        "all_case_counts_match": all(
            row["case_count"] == prereg["case_count_per_model"]
            for row in model_rows.values()
        ),
        "all_model_case_digests_match": all(
            row["case_digest_matches"] for row in model_rows.values()
        ),
        "all_model_result_digests_match": all(
            row["pilot_result_digest_matches"]
            and row["atlas_summary_digest_matches"]
            for row in model_rows.values()
        ),
        "all_model_npz_hashes_match": all(
            row["summary_npz_sha256_matches"] for row in model_rows.values()
        ),
        "all_protocol_digests_match": all(
            row["protocol_digest_matches"] for row in model_rows.values()
        ),
        "prediction_partition_complete": (
            set(summary["passed_predictions"] + summary["failed_predictions"])
            == {f"P{index}" for index in range(1, 11)}
        ),
        "causal_not_authorized": not bool(automatic["local_causal_authorized"]),
        "behavior_digest_matches": (
            summary["behavior_authorization_digest"] == behavior["summary_digest"]
        ),
    }
    result = {
        "schema_version": "phase1093_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "digest_checks": digest_checks,
        "digest_values": digest_values,
        "models": model_rows,
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(root / "analysis" / "result_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(f"Phase1093 result audit failed: {checks}")
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": True,
        "audit_digest": result["audit_digest"],
    })


if __name__ == "__main__":
    main()
