#!/usr/bin/env python3
"""Audit Phase1098 artifacts, hashes, array schemas, and arithmetic closure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1098_relative_relation_geometry_protocol as protocol


def main() -> None:
    checks: dict[str, bool] = {}
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    block = protocol.read_json(protocol.SOURCE_BLOCK_AUDIT)
    diagnostic = protocol.read_json(protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json")
    checks["protocol_digest"] = prereg["protocol_digest"] == protocol.digest({key: value for key, value in prereg.items() if key != "protocol_digest"})
    checks["protocol_audit_passed"] = bool(audit["all_checks_passed"])
    checks["authorization_digest"] = authorization["authorization_digest"] == protocol.digest({key: value for key, value in authorization.items() if key != "authorization_digest"})
    checks["hidden_scan_authorized"] = bool(authorization["hidden_scan_authorized"])
    checks["block_audit_phase"] = block.get("phase") == 1098 and block.get("source_phase") == 1097
    checks["diagnostic_digest"] = diagnostic["diagnostic_digest"] == protocol.digest({key: value for key, value in diagnostic.items() if key != "diagnostic_digest"})
    checks["diagnostic_is_posthoc"] = diagnostic.get("status") == "posthoc_descriptive_only"
    checks["final_protocol_digest"] = final["protocol_digest"] == prereg["protocol_digest"]
    checks["final_summary_digest"] = final["summary_digest"] == protocol.digest({key: value for key, value in final.items() if key != "summary_digest"})
    checks["final_links_diagnostic"] = final["posthoc_failure_diagnostic"]["diagnostic_digest"] == diagnostic["diagnostic_digest"]
    for model in protocol.MODELS:
        behavior = protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
        summary = protocol.read_json(protocol.OUT_ROOT / "atlas" / model / "summary.json")
        index = protocol.read_jsonl(protocol.OUT_ROOT / "atlas" / model / "superunit_index.jsonl")
        checks[f"{model}_behavior_digest"] = behavior["summary_digest"] == protocol.digest({key: value for key, value in behavior.items() if key != "summary_digest"})
        checks[f"{model}_atlas_digest"] = summary["summary_digest"] == protocol.digest({key: value for key, value in summary.items() if key != "summary_digest"})
        checks[f"{model}_protocol_digest"] = summary["protocol_digest"] == prereg["protocol_digest"]
        checks[f"{model}_fp16"] = (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        )
        checks[f"{model}_superunit_count"] = len(index) == prereg["superunit_count_per_model"] == summary["superunit_count"]
        checks[f"{model}_output_excluded"] = bool(summary["primary_signature_excludes_output_interaction"])
        with np.load(protocol.OUT_ROOT / "atlas" / model / "relative_relation_geometry.npz") as data:
            expected_prefix = (prereg["superunit_count_per_model"], summary["event_count"], len(protocol.FIELDS), len(protocol.CAPTURE_ROLES))
            checks[f"{model}_gram_shape"] = data["relation_gram"].shape == expected_prefix + (len(protocol.RELATIONS), len(protocol.RELATIONS))
            checks[f"{model}_energy_shape"] = data["shared_energy"].shape == expected_prefix
            checks[f"{model}_differential_shape"] = data["differential_energy"].shape == expected_prefix
            checks[f"{model}_centered_norm_shape"] = data["centered_norm"].shape == expected_prefix + (len(protocol.RELATIONS),)
            checks[f"{model}_output_shape"] = data["output_interaction"].shape == (prereg["superunit_count_per_model"], len(protocol.RELATIONS), len(protocol.FIELDS))
            closure = np.abs(data["shared_energy"].astype(np.float64) + data["differential_energy"].astype(np.float64) - 1.0)
            energy_sum = data["shared_energy"].astype(np.float64) + data["differential_energy"].astype(np.float64)
            nonzero_energy = energy_sum > 0.5
            checks[f"{model}_energy_closure"] = bool(nonzero_energy.any() and np.nanmax(closure[nonzero_energy]) <= 2e-5)
            diagonal = np.diagonal(data["relation_gram"], axis1=-2, axis2=-1)
            finite_diagonal = diagonal[np.isfinite(diagonal)]
            checks[f"{model}_gram_unit_diagonal"] = bool(finite_diagonal.size and np.max(np.abs(finite_diagonal - 1.0)) <= 2e-4)
            primary = summary["fields"].index(protocol.PRIMARY_FIELD)
            query_end = summary["roles"].index("query_end")
            answer_boundary = summary["roles"].index("answer_boundary")
            dynamic_geometry = data["relation_gram"][:, :, primary, [query_end, answer_boundary], :, :]
            checks[f"{model}_finite_geometry"] = bool(np.isfinite(dynamic_geometry).mean() >= 0.95)
    result = {
        "schema_version": "phase1098_result_audit.v1",
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
        raise SystemExit([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
