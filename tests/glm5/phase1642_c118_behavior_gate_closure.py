#!/usr/bin/env python3
"""Close C118 at the frozen behavior gate and authorize a fresh repair contract."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1640_c118_identifiable_default_override_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    audit = core.load(OUT / "audit/independent_capture_audit.json")
    if not audit["all_checks_passed"] or capture["behavior_gate_passed"]:
        raise RuntimeError("C118 does not satisfy the registered closure branch")
    closure = {
        "phase": 1642,
        "campaign": "C118",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_gate_failed_hidden_state_adjudication_not_authorized",
        "headline": capture["behavior"],
        "gate_checks": capture["behavior_gate_checks"],
        "strict_conclusion": "Qwen3 executes conflicting item-specific exceptions and both output vocabularies well, but does not reliably ignore an exception naming another item and inherit the general policy under this prompt contract.",
        "not_tested": ["override HiddenState field", "coordinate assignment", "common/residual intervention", "relation-specific route"],
        "raw_archive_status": "captured during the unified behavior/field forward pass but sealed from scientific HiddenState analysis because the behavior gate failed",
        "new_puzzles": {"K310-BOUNDARY": "default inheritance and conflicting exception override are behaviorally separable; C118 passes override but fails default inheritance"},
        "problems": ["controlled synthetic English", "model over-applies a salient exception to the wrong item", "machine naturalness only", "one Qwen3", "captured hidden states are not behavior-qualified evidence"],
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": "C119 fresh behavior-interface repair using an explicit exception-applicability statement; preserve factors, partitions, thresholds and claim boundary; do not inspect C118 hidden states",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "capture_audit": audit["all_checks_passed"],
        "gate_failed": not capture["behavior_gate_passed"],
        "default_failed": not capture["behavior_gate_checks"]["default"],
        "override_passed": capture["behavior_gate_checks"]["override"],
        "not_tested": len(closure["not_tested"]) == 4,
        "authorization": closure["next_authorization"].startswith("C119"),
    }
    report = {"phase": 1642, "campaign": "C118", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": closure["next_authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"closure": closure, "audit": report}, indent=2))


if __name__ == "__main__":
    main()
