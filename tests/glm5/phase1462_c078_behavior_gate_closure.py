#!/usr/bin/env python3
"""Phase1462: close C078 after its over-granular behavior conjunction failed."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1460_c078_colon_label_contract"
BEHAVIOR = TESTS / "result/phase1461_c078_behavior"
OUT = TESTS / "result/phase1462_c078_behavior_gate_closure"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1462 exists")
    ca = core.load(CONTRACT / "audit/independent_final_audit.json")
    ba = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    bf = core.load(BEHAVIOR / "analysis/final.json")
    summary = core.load(BEHAVIOR / "analysis/behavior_summary.json")
    failing = []
    for family, relations in summary["family_relation_surface"].items():
        for relation, surfaces in relations.items():
            for surface, result in surfaces.items():
                if not result["qualified"]:
                    failing.append({"family": family, "relation": relation, "surface": surface, "checks": result["checks"], "metrics": result["metrics"]})
    scripts = []
    for path in sorted(TESTS.glob("phase146[01]_c078_*.py")):
        py_compile.compile(str(path), doraise=True)
        scripts.append(path.name)
    checks = {
        "contract_audit": ca["all_checks_passed"],
        "behavior_audit": ba["all_checks_passed"],
        "behavior_failed": not summary["behavior_qualified"],
        "close_authorized": bf["authorization"] == "close_c078_at_behavior_gate",
        "aggregate_passed": summary["checks"]["all_surfaces"] and summary["checks"]["eligible_total"] and summary["checks"]["eligible_splits"] and summary["checks"]["eligible_relations"],
        "granular_only_failure": not summary["checks"]["all_family_relation_surface"] and all(summary["checks"][key] for key in summary["checks"] if key != "all_family_relation_surface"),
        "eligible_exact": summary["eligible_count"] == 204 and summary["eligible_partition_counts"] == {"response_discovery": 67, "confirmation": 67, "lockbox": 70},
        "errors": sum(sum(values.values()) for values in summary["error_surface_truth_counts"].values()) == 13,
        "hidden": summary["hidden_state_accessed"] is False,
        "no_capture": not (TESTS / "result/phase1462_c078_discovery_full_field_capture").exists(),
        "scripts_compile": len(scripts) == 4,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1462,
        "campaign": "C078",
        "status": "closed_at_behavior_gate_after_sparse_subcell_conjunction_failure",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "failing_family_relation_surface_count": len(failing),
        "failing_family_relation_surface": failing,
        "retained": ["both global surfaces passed 0.98 BA", "204 complete factorial sets were behavior-correct", "all split and relation eligible-set breadth gates passed", "numeric execution was healthy"],
        "rejected": ["C078 hidden-state access", "post-hoc removal of the granular conjunction", "internal relation conclusions"],
        "authorization": "preregister_c079_aggregate_eligible_observation_campaign_on_fresh_material",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "failing_family_relation_surface"}, indent=2))


if __name__ == "__main__":
    main()
