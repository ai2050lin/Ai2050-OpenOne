#!/usr/bin/env python3
"""Independently audit Phase1097 artifacts, hashes, shapes, and gate logic."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1097_conditional_transition_protocol as protocol


def digest_without(record: dict, key: str) -> str:
    copy = dict(record)
    copy.pop(key, None)
    return protocol.digest(copy)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    diagnostic = protocol.read_json(protocol.OUT_ROOT / "analysis" / "transition_diagnostic.json")
    checks: dict[str, bool] = {}
    checks["protocol_audit_passed"] = bool(protocol_audit["all_checks_passed"])
    checks["preregistration_digest"] = prereg["protocol_digest"] == digest_without(prereg, "protocol_digest")
    checks["behavior_digest"] = behavior["authorization_digest"] == digest_without(behavior, "authorization_digest")
    checks["final_digest"] = final["summary_digest"] == digest_without(final, "summary_digest")
    checks["diagnostic_digest"] = diagnostic["diagnostic_digest"] == digest_without(diagnostic, "diagnostic_digest")
    checks["protocol_chain"] = final["protocol_digest"] == prereg["protocol_digest"] == behavior["protocol_digest"]
    checks["behavior_authorized"] = bool(behavior["hidden_scan_authorized"])

    expected_unit_shape = (
        prereg["unit_count_per_model"], len(protocol.FIELDS),
        len(protocol.CAPTURE_ROLES), len(protocol.DEPTH_ANCHORS),
    )
    artifact_digests = {}
    for model_name in protocol.MODELS:
        cases_path = protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        cases = protocol.read_jsonl(cases_path)
        checks[f"{model_name}_case_count"] = len(cases) == prereg["case_count_per_model"]
        checks[f"{model_name}_case_digest"] = protocol.digest(cases) == prereg["model_case_digests"][model_name]
        behavior_path = protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
        behavior_summary = protocol.read_json(behavior_path)
        checks[f"{model_name}_behavior_digest"] = behavior_summary["summary_digest"] == digest_without(behavior_summary, "summary_digest")
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_summary = protocol.read_json(atlas_root / "summary.json")
        checks[f"{model_name}_atlas_digest"] = atlas_summary["summary_digest"] == digest_without(atlas_summary, "summary_digest")
        checks[f"{model_name}_protocol_digest_chain"] = atlas_summary["protocol_digest"] == prereg["protocol_digest"]
        with np.load(atlas_root / "unit_transition_invariants.npz") as unit_data:
            checks[f"{model_name}_unit_amplitude_shape"] = unit_data["amplitude"].shape == expected_unit_shape
            checks[f"{model_name}_unit_gram_shape"] = unit_data["depth_gram"].shape == expected_unit_shape + (len(protocol.DEPTH_ANCHORS),)
            checks[f"{model_name}_unit_margin_shape"] = unit_data["local_margin"].shape == expected_unit_shape
            checks[f"{model_name}_unit_count"] = unit_data["amplitude"].shape[0] == atlas_summary["unit_count"]
        with np.load(atlas_root / "transition_aggregates.npz") as aggregate:
            required = {
                "amplitude_sum", "amplitude_count", "local_margin_sum", "local_margin_count",
                "gram_sum", "gram_count", "panel_alignment_sum", "panel_alignment_count",
                "ledger_alignment_sum", "ledger_alignment_count", "physical_sum", "physical_count",
            }
            checks[f"{model_name}_aggregate_fields"] = required == set(aggregate.files)
            checks[f"{model_name}_positive_counts"] = int(np.sum(aggregate["amplitude_count"])) > 0 and int(np.sum(aggregate["physical_count"])) > 0
        checks[f"{model_name}_gate_recompute_p4"] = (
            final["models"][model_name]["split_repeat"]["passed"]
            == (final["models"][model_name]["split_repeat"]["passing_records"] >= prereg["evidence_thresholds"]["minimum_split_records"])
        )
        checks[f"{model_name}_gate_recompute_p6"] = (
            final["models"][model_name]["behavior_anchor"]["passed"]
            == (final["models"][model_name]["behavior_anchor"]["passing_cells"] >= prereg["evidence_thresholds"]["minimum_behavior_anchor_cells"])
        )
        checks[f"{model_name}_gate_recompute_p7"] = (
            final["models"][model_name]["panel_convergence"]["passed"]
            == (final["models"][model_name]["panel_convergence"]["passing_cells"] >= prereg["evidence_thresholds"]["minimum_behavior_anchor_cells"])
        )
        artifact_digests[model_name] = {
            "cases_sha256": file_sha256(cases_path),
            "behavior_summary_sha256": file_sha256(behavior_path),
            "atlas_summary_sha256": file_sha256(atlas_root / "summary.json"),
            "aggregate_npz_sha256": file_sha256(atlas_root / "transition_aggregates.npz"),
            "unit_invariants_npz_sha256": file_sha256(atlas_root / "unit_transition_invariants.npz"),
        }

    checks["automatic_next_logic"] = (
        final["automatic_next_required"]
        == all(final["gates"][key] for key in (
            "P5_heldout_relation_transition",
            "P6_behavior_anchored_execution",
            "P7_early_late_panel_convergence",
            "P8_cross_language_transition",
        ))
    )
    checks["no_causal_upgrade"] = not final["causal_localization_authorized"] and not final["gates"]["P9_causal_localization_authorized"]
    checks["automatic_next_not_required"] = final["automatic_next_required"] is False
    result = {
        "schema_version": "phase1097_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_summary_digest": final["summary_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "artifact_digests": artifact_digests,
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "check_count": len(checks),
        "failed_checks": [key for key, value in checks.items() if not value],
        "all_checks_passed": result["all_checks_passed"],
        "audit_digest": result["audit_digest"],
    })


if __name__ == "__main__":
    main()
