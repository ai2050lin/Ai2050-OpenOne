#!/usr/bin/env python3
"""Phase1389: close C061 after the frozen behavior gate failure."""
from pathlib import Path
import json, py_compile, sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
BEHAVIOR = TESTS / "result/phase1388_c061_qwen_behavior_qualification"
OUT = TESTS / "result/phase1389_c061_behavior_gate_closure"


def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError("Phase1389 already exists")
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    summary = core.load(BEHAVIOR / "analysis/qwen3_behavior_summary.json")
    if final["authorization"] != "close_c061_behavior_unqualified_before_hidden_access" or not audit["all_checks_passed"]:
        raise RuntimeError("C061 closure not authorized")
    scripts = sorted(TESTS.glob("phase138[7-9]_c061_*.py"))
    for script in scripts: py_compile.compile(str(script), doraise=True)
    hidden_artifacts = list((TESTS / "result").glob("phase13*_c061*/*/qwen3_*hidden*"))
    checks = {
        "behavior_failed": not summary["behavior_qualified"],
        "status_healthy": summary["status"]["accuracy"] == 1.0,
        "numeric_healthy": summary["numeric_same_shape_max_abs_diff"] == 0.0,
        "no_selected_hidden_cases": summary["selected_pair_count"] == 0,
        "hidden_state_not_accessed": not hidden_artifacts,
        "audit_passed": audit["all_checks_passed"],
        "scripts_compile": True,
    }
    result = {"phase": 1389, "campaign": "C061", "status": "closed_at_behavior_gate",
              "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()),
              "formal_result": {"active_accuracy": summary["active"]["accuracy"],
                                "sport_accuracy": summary["active"]["target_family"]["sport"],
                                "true_accuracy": summary["active"]["truth"]["true"],
                                "quartet_all_fraction": summary["active"]["quartet_all_fraction"],
                                "eligible_cell_min": summary["eligible_cell_min"]},
              "claim_boundary": "material/interface behavior qualification failure; no hidden-state or mechanism test",
              "authorization": "preregister_c062_route_factorized_behavior_and_hidden_campaign",
              "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
