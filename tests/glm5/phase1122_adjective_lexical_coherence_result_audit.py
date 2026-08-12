#!/usr/bin/env python3
"""Recompute and audit the frozen Phase1122 result."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1122_adjective_lexical_coherence_analyze as analyze
import phase1122_adjective_lexical_coherence_protocol as protocol


def main() -> None:
    existing = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    recomputed = analyze.compute()
    checks = {
        "final_digest_recomputed": existing["final_digest"] == recomputed["final_digest"],
        "protocol_digest_stable": existing["protocol_digest"] == recomputed["protocol_digest"],
        "source_final_digest_stable": existing["source_final_digest"] == recomputed["source_final_digest"],
        "lexical_interactions_digest_stable": existing["lexical_interactions_digest"] == recomputed["lexical_interactions_digest"],
        "gate_vector_stable": existing["gates"] == recomputed["gates"],
        "model_metrics_stable": existing["models"] == recomputed["models"],
        "null_summary_stable": existing["null_summary"] == recomputed["null_summary"],
    }
    core = {
        "schema_version": "phase1122_lexical_coherence_result_audit.v1",
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
        raise RuntimeError("Phase1122 result audit failed")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
