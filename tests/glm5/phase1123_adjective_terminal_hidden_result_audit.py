#!/usr/bin/env python3
"""Recompute and audit the frozen Phase1123 result."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1123_adjective_terminal_hidden_finalize as finalize
import phase1123_adjective_terminal_hidden_protocol as protocol


def main() -> None:
    existing = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    recomputed = finalize.finalize()
    checks = {
        "final_digest_recomputed": existing["final_digest"] == recomputed["final_digest"],
        "protocol_digest_stable": existing["protocol_digest"] == recomputed["protocol_digest"],
        "model_results_stable": existing["models"] == recomputed["models"],
        "cross_model_results_stable": existing["cross_model"] == recomputed["cross_model"],
        "qualification_stable": existing["qualified_models"] == recomputed["qualified_models"],
        "prediction_vector_stable": existing["predictions"] == recomputed["predictions"],
        "authorization_stable": existing["automatic_continuation"] == recomputed["automatic_continuation"],
    }
    core = {
        "schema_version": "phase1123_adjective_terminal_hidden_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "final_digest": recomputed["final_digest"],
    }
    audit = dict(core)
    audit["audit_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1123 result audit failed")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
