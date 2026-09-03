#!/usr/bin/env python3
"""Phase1474: close C081 and the explicit-label interaction route."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1474_c081_route_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1474 exists")
    stages = {
        1469: RESULT / "phase1469_c080_balanced_interaction_contract",
        1470: RESULT / "phase1470_c080_explicit_behavior",
        1471: RESULT / "phase1471_c080_behavior_gate_closure",
        1472: RESULT / "phase1472_c081_validated_interface_contract",
        1473: RESULT / "phase1473_c081_behavior",
    }
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in stages.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in stages.items()}
    c080 = core.load(stages[1470] / "analysis/behavior_summary.json")
    c081 = core.load(stages[1473] / "analysis/behavior_summary.json")
    scripts = []
    for phase in range(1469, 1474):
        for path in sorted(TESTS.glob(f"phase{phase}_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    checks = {
        "audits": all(value["all_checks_passed"] for value in audits.values()),
        "c080_failed": not c080["behavior_qualified"] and finals[1470]["authorization"] == "close_c080_explicit_at_behavior_gate",
        "c081_failed": not c081["behavior_qualified"] and finals[1473]["authorization"] == "close_c081_and_explicit_interaction_route_at_behavior_gate",
        "negative_cases": c080["error_counts"]["truth"]["false"] == c081["error_counts"]["truth"]["false"] == 0,
        "positive_failure": c080["error_counts"]["truth"]["true"] == 607 and c081["error_counts"]["truth"]["true"] == 121,
        "no_hidden": not c080["hidden_state_accessed"] and not c081["hidden_state_accessed"],
        "scripts_compile": len(scripts) == 10,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1474,
        "campaign": "C081",
        "status": "explicit_label_balanced_interaction_route_closed_before_hidden_access",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "answer": "two separately frozen dual-surface contracts failed because positive equality execution remained interface-sensitive; the hidden equality-interaction object remains untested",
        "route_closed": [
            "additional explicit-label prompt rescue",
            "C080/C081 Hidden-State capture",
            "label withdrawal conditioned on the explicit interaction gate",
        ],
        "campaign_continues_with": {
            "name": "C082 coordinate-resolved retrospective atlas of the already legal C079 raw field",
            "scope": "exploratory observation only because all C079 splits have already been opened",
            "forbidden_claim": "no C079 atlas pattern may be called independently confirmed",
        },
        "authorization": "preregister_c082_c079_coordinate_resolved_exploratory_atlas",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
