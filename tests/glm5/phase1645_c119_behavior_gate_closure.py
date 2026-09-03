#!/usr/bin/env python3
"""Close C119 after the explicit-applicability repair fails default inheritance."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1643_c119_identifiable_default_override_campaign"
C118 = TESTS / "result/phase1640_c118_identifiable_default_override_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


if __name__ == "__main__":
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    c118 = core.load(C118 / "analysis/closure.json")
    audit = core.load(OUT / "audit/independent_capture_audit.json")
    if not audit["all_checks_passed"] or capture["behavior_gate_passed"]:
        raise RuntimeError("C119 closure branch mismatch")
    closure = {
        "phase": 1645, "campaign": "C119", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "explicit_applicability_repair_failed_default_inheritance_gate",
        "headline": capture["behavior"], "gate_checks": capture["behavior_gate_checks"],
        "paired_c118_default_inheritance": c118["headline"]["default_inheritance"],
        "paired_change": capture["behavior"]["default_inheritance"] - c118["headline"]["default_inheritance"],
        "strict_conclusion": "An explicit statement that the exception does or does not apply to the queried item did not qualify default inheritance. Conflicting hit exceptions remained perfect, so this route separates salient explicit exception reading from reliable fallback to a general policy.",
        "not_tested": ["override HiddenState field", "coordinate assignment", "common/residual intervention", "relation-specific route"],
        "raw_archive_status": "captured in the unified forward pass but sealed from HiddenState adjudication after the behavior failure",
        "new_puzzles": {"K311-BOUNDARY": "the C119 applicability-interface repair does not recover default inheritance; explicit conflicting exception reading remains much stronger"},
        "problems": ["controlled synthetic English", "paired prompt intervention changes wording but not model", "default language may be weaker than explicit inspection language", "machine naturalness only", "one Qwen3", "no hidden-state claim"],
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": "C120 matched-output controlled comparison-family observation campaign; no further default-interface tuning in this major stage",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"capture_audit": audit["all_checks_passed"], "gate_failed": not capture["behavior_gate_passed"], "default_failed": not capture["behavior_gate_checks"]["default"], "override_passed": capture["behavior_gate_checks"]["override"], "paired": closure["paired_change"] < 0, "sealed": "sealed" in closure["raw_archive_status"], "authorization": closure["next_authorization"].startswith("C120")}
    report = {"phase": 1645, "campaign": "C119", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": closure["next_authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"closure": closure, "audit": report}, indent=2))
