#!/usr/bin/env python3
"""Audit Phase1106 protocol, behavior summaries, and frozen decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1106_causal_evidence_semantic_replication_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    source_authorization = protocol.read_json(protocol.SOURCE_AUTHORIZATION)
    checks = {
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_digest": authorization["protocol_digest"] == prereg["protocol_digest"],
        "source_authorization_unchanged": prereg["source_authorization_digest"] == source_authorization["authorization_digest"],
        "unique_prior_candidate": protocol.source_semantic_candidates() == [protocol.RELATION_PAIR],
        "all_models_present": set(authorization["models"]) == set(protocol.MODELS),
        "behavior_summary_chain": all(row["summary_digest"] == protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")["summary_digest"] for model, row in authorization["models"].items()),
        "hidden_state_not_accessed": not authorization["hidden_state_accessed_in_phase1106"],
        "cross_model_threshold_enforced": (not authorization["cross_model_semantic_replication"] or len(authorization["passing_models"]) >= protocol.THRESHOLDS["minimum_models_for_shared_replication"]),
        "automatic_next_recorded": authorization["automatic_next_required"] and bool(authorization["automatic_next_task"]),
    }
    result = {
        "schema_version": "phase1106_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    print(json.dumps(result))
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1106 result audit failed")


if __name__ == "__main__":
    main()
