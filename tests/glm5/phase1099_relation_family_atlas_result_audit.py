#!/usr/bin/env python3
"""Audit Phase1099 artifacts and cross-file invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1099_relation_family_atlas_protocol as protocol
import phase1099_relation_family_atlas_finalize as finalize


def valid_digest(row: dict, key: str) -> bool:
    expected = row.get(key)
    copy = dict(row)
    copy.pop(key, None)
    return expected == protocol.digest(copy)


def main() -> None:
    checks = {}
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    checks["preregistration_digest"] = valid_digest(prereg, "protocol_digest")
    checks["protocol_audit_digest"] = valid_digest(protocol_audit, "audit_digest")
    checks["protocol_audit_passed"] = bool(protocol_audit["all_checks_passed"])
    checks["authorization_digest"] = valid_digest(authorization, "authorization_digest")
    checks["hidden_scan_authorized"] = bool(authorization["hidden_scan_authorized"])
    checks["final_summary_digest"] = valid_digest(final, "summary_digest")
    checks["family_count"] = len(protocol.FAMILIES) == 5
    checks["relation_count"] = len(protocol.RELATIONS) == 30
    checks["family_balance"] = all(sum(protocol.RELATION_FAMILY[r] == family for r in protocol.RELATIONS) == 6 for family in protocol.FAMILIES)
    checks["holdout_balance"] = all(
        sum(protocol.RELATION_FAMILY[r] == family and protocol.RELATION_SPLIT[r] == split for r in protocol.RELATIONS) == 3
        for family in protocol.FAMILIES for split in protocol.RELATION_SPLITS
    )
    synthetic = np.arange(len(protocol.RELATIONS) ** 2, dtype=np.float64).reshape(len(protocol.RELATIONS), len(protocol.RELATIONS))
    synthetic_block = finalize.family_block(synthetic, protocol.RELATION_SPLITS[0])
    aggregation_checks = []
    for left, left_family in enumerate(protocol.FAMILIES):
        left_indices = [
            index for index, relation in enumerate(protocol.RELATIONS)
            if protocol.RELATION_FAMILY[relation] == left_family
            and protocol.RELATION_SPLIT[relation] == protocol.RELATION_SPLITS[0]
        ]
        for right, right_family in enumerate(protocol.FAMILIES):
            right_indices = [
                index for index, relation in enumerate(protocol.RELATIONS)
                if protocol.RELATION_FAMILY[relation] == right_family
                and protocol.RELATION_SPLIT[relation] == protocol.RELATION_SPLITS[0]
            ]
            expected_values = [
                synthetic[i, j] for i in left_indices for j in right_indices
                if left != right or i != j
            ]
            aggregation_checks.append(np.isclose(synthetic_block[left, right], np.mean(expected_values)))
    checks["family_block_aggregation"] = bool(all(aggregation_checks))
    for model_name in protocol.MODELS:
        behavior = protocol.read_json(protocol.OUT_ROOT / "behavior" / model_name / "summary.json")
        checks[f"{model_name}_behavior_digest"] = valid_digest(behavior, "summary_digest")
        checks[f"{model_name}_behavior_protocol"] = behavior["protocol_digest"] == prereg["protocol_digest"]
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(atlas_root / "summary.json")
        index = protocol.read_jsonl(atlas_root / "superunit_index.jsonl")
        checks[f"{model_name}_atlas_digest"] = valid_digest(summary, "summary_digest")
        checks[f"{model_name}_atlas_protocol"] = summary["protocol_digest"] == prereg["protocol_digest"]
        checks[f"{model_name}_superunit_count"] = len(index) == prereg["superunit_count_per_model"]
        checks[f"{model_name}_event_count"] = summary["event_count"] == len(summary["events"])
        checks[f"{model_name}_primary_excludes_output"] = bool(summary["primary_signature_excludes_output_interaction"])
        with np.load(atlas_root / "relative_relation_geometry.npz") as data:
            expected_prefix = (
                prereg["superunit_count_per_model"],
                summary["event_count"],
                len(protocol.FIELDS),
                len(protocol.CAPTURE_ROLES),
            )
            checks[f"{model_name}_gram_shape"] = data["relation_gram"].shape == expected_prefix + (len(protocol.RELATIONS), len(protocol.RELATIONS))
            checks[f"{model_name}_energy_shape"] = data["shared_energy"].shape == expected_prefix
            checks[f"{model_name}_centered_norm_shape"] = data["centered_norm"].shape == expected_prefix + (len(protocol.RELATIONS),)
            closure = np.abs(data["shared_energy"].astype(np.float64) + data["differential_energy"].astype(np.float64) - 1.0)
            valid = np.isfinite(closure) & (data["shared_energy"] + data["differential_energy"] > 0)
            checks[f"{model_name}_energy_closure"] = bool(np.nanmax(closure[valid]) < 1e-4)
        checks[f"{model_name}_final_digest_link"] = final["models"][model_name]["summary_digest"] == summary["summary_digest"]
    checks["final_protocol_link"] = final["protocol_digest"] == prereg["protocol_digest"]
    checks["final_authorization_link"] = final["authorization_digest"] == authorization["authorization_digest"]
    checks["automatic_rule_consistent"] = final["automatic_next_required"] == all(final["gates"].values())
    diagnostic = protocol.read_json(protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json")
    checks["failure_diagnostic_digest"] = valid_digest(diagnostic, "diagnostic_digest")
    checks["failure_diagnostic_final_link"] = diagnostic["final_summary_digest"] == final["summary_digest"]
    checks["failure_diagnostic_is_post_hoc"] = diagnostic["evidence_status"].startswith("post_hoc_descriptive_only")
    result = {
        "schema_version": "phase1099_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    print({"phase": protocol.PHASE, "passed_checks": result["passed_checks"], "check_count": result["check_count"], "all_checks_passed": result["all_checks_passed"], "audit_digest": result["audit_digest"]})
    if not result["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase1099 result audit failed: {failed}")


if __name__ == "__main__":
    main()
