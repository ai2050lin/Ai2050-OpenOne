#!/usr/bin/env python3
"""Audit Phase1095 frozen artifacts and provenance."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1095_query_antisymmetric_protocol as protocol


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    source_prereg = protocol.read_json(
        protocol.SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_behavior = protocol.read_json(
        protocol.SOURCE_ROOT / "analysis" / "behavior_authorization.json"
    )
    model_checks = {}
    for model in protocol.MODELS:
        summary_path = protocol.OUT_ROOT / "atlas" / model / "summary.json"
        npz_path = protocol.OUT_ROOT / "atlas" / model / "signed_fields.npz"
        summary = protocol.read_json(summary_path)
        checks = {
            "summary_protocol_digest": summary["protocol_digest"] == prereg["protocol_digest"],
            "summary_phase": int(summary["phase"]) == protocol.PHASE,
            "fp16": bool(summary["precision"]["has_fp16_parameters"]),
            "not_quantized": not bool(summary["precision"]["has_quantized_modules"]),
            "hidden_finite": float(summary["hidden_finite_fraction_lower_bound"])
            >= float(protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]),
            "pre_query_zero": float(summary["pre_query_global_max_abs"])
            <= float(protocol.EVIDENCE_THRESHOLDS["pre_query_tolerance"]),
            "npz_exists_nonempty": npz_path.exists() and npz_path.stat().st_size > 0,
            "summary_finite": finite_tree(summary),
        }
        model_checks[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "summary_sha256": sha256_file(summary_path),
            "npz_sha256": sha256_file(npz_path),
        }
    checks = {
        "static_audit_passed": bool(static["all_checks_passed"]),
        "behavior_authorized": bool(behavior["hidden_scan_authorized"]),
        "source_protocol_digest_matches": prereg["source_phase1094_protocol_digest"]
        == source_prereg["protocol_digest"],
        "source_behavior_digest_matches": prereg["source_phase1094_behavior_digest"]
        == source_behavior["summary_digest"],
        "all_model_artifacts_pass": all(row["all_checks_passed"] for row in model_checks.values()),
        "final_protocol_digest_matches": final["protocol_digest"] == prereg["protocol_digest"],
        "all_predictions_boolean": all(
            isinstance(row["passed"], bool) for row in final["predictions"].values()
        ),
        "final_tree_finite": finite_tree(final),
        "causal_not_authorized": final["causal_authorized"] is False,
    }
    audit = {
        "schema_version": "phase1095_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "models": model_checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = protocol.digest(audit)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "result_audit.json", audit)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": audit["all_checks_passed"],
        "audit_digest": audit["audit_digest"],
    })


if __name__ == "__main__":
    main()
