#!/usr/bin/env python3
"""Audit Phase1100 source extraction, formulas, and frozen decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1099_relation_family_atlas_protocol as source_protocol
import phase1100_relation_graph_inheritance_finalize as finalize
import phase1100_relation_graph_inheritance_protocol as protocol


def main() -> None:
    preregistration = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    diagnostic = protocol.read_json(protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json")
    checks: dict[str, bool] = {
        "preregistration_digest": preregistration["protocol_digest"] == protocol.digest({key: value for key, value in preregistration.items() if key != "protocol_digest"}),
        "protocol_audit_digest": protocol_audit["audit_digest"] == protocol.digest({key: value for key, value in protocol_audit.items() if key != "audit_digest"}),
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "final_digest": final["final_digest"] == protocol.digest({key: value for key, value in final.items() if key != "final_digest"}),
        "final_protocol_link": final["protocol_digest"] == preregistration["protocol_digest"],
        "formal_models_exact": set(protocol.FORMAL_MODELS) == {"qwen3", "glm4"},
        "family_permutation_count": final["registered_family_permutations"] == 119,
        "within_family_permutation_count": final["registered_within_family_permutations"] == 7775,
        "automatic_rule_consistent": final["automatic_next_required"] == all(final["gates"].values()),
        "primary_excludes_outputs": bool(final["gates"]["P7"]),
        "diagnostic_digest": diagnostic["diagnostic_digest"] == protocol.digest({key: value for key, value in diagnostic.items() if key != "diagnostic_digest"}),
        "diagnostic_final_link": diagnostic["final_digest"] == final["final_digest"],
        "diagnostic_is_post_hoc": diagnostic["post_hoc_only"] and diagnostic["cannot_upgrade_registered_gates"],
    }
    for model in protocol.MODELS:
        source_root = protocol.OUT_ROOT / "source" / model
        summary = protocol.read_json(source_root / "summary.json")
        checks[f"{model}_summary_digest"] = summary["summary_digest"] == protocol.digest({key: value for key, value in summary.items() if key != "summary_digest"})
        checks[f"{model}_protocol_link"] = summary["protocol_digest"] == preregistration["protocol_digest"]
        checks[f"{model}_fp16"] = summary["precision"]["has_fp16_parameters"] and not summary["precision"]["has_bf16_parameters"] and not summary["precision"]["has_quantized_modules"]
        checks[f"{model}_finite"] = summary["source_finite_fraction"] >= protocol.THRESHOLDS["minimum_source_finite_fraction"]
        with np.load(source_root / "lexical_source.npz", allow_pickle=False) as archive:
            checks[f"{model}_source_keys"] = set(archive.files) == {protocol.PRIMARY_SOURCE, protocol.ALTERNATIVE_SOURCE, protocol.FORM_SOURCE}
            checks[f"{model}_input_shape"] = archive[protocol.PRIMARY_SOURCE].shape[:2] == (len(protocol.SURFACES), len(source_protocol.RELATIONS))
            checks[f"{model}_output_shape"] = archive[protocol.ALTERNATIVE_SOURCE].shape[:2] == (len(protocol.SURFACES), len(source_protocol.RELATIONS))
            checks[f"{model}_form_shape"] = archive[protocol.FORM_SOURCE].shape == (len(protocol.SURFACES), len(source_protocol.RELATIONS), 7)
            checks[f"{model}_arrays_finite"] = all(np.isfinite(archive[key]).all() for key in archive.files)
        checks[f"{model}_surface_count"] = len(final["models"][model]["surfaces"]) == len(protocol.SURFACES)
        checks[f"{model}_confirmation_count"] = all(len(row["confirmation_records"]) == 3 for row in final["models"][model]["surfaces"].values())

    synthetic = np.asarray([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]], dtype=np.float64)
    graph = finalize.centered_gram(synthetic)
    checks["synthetic_gram_diagonal"] = bool(np.allclose(np.diag(graph), 1.0))
    vector, fraction = finalize.graph_vector(graph)
    checks["synthetic_graph_vector"] = bool(np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector), 1.0) and fraction == 1.0)
    checks["cross_model_curve_count"] = len(final["cross_model_functional_trajectories"]) == 6
    checks["gate_keys_exact"] = set(final["gates"]) == set(protocol.GATES)
    checks["diagnostic_cell_count"] = diagnostic["all_model_summary"]["cell_count"] == len(protocol.MODELS) * len(protocol.SURFACES) * 4

    result = {
        "schema_version": "phase1100_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    if not result["all_checks_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1100 result audit failed: {failed}")
    print(json.dumps({"phase": protocol.PHASE, "passed_checks": result["passed_checks"], "audit_digest": result["audit_digest"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
