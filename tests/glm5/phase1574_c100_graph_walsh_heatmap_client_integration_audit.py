#!/usr/bin/env python3
"""Independent audit for Phase1574."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1574_c100_graph_walsh_heatmap_client_integration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1574_c100_graph_walsh_heatmap_client_integration.py"), doraise=True)
    report = core.load(OUT / "analysis/client_integration.json")
    final = core.load(OUT / "analysis/final.json")
    c101 = core.load(OUT / "protocol/c101_requirements.json")
    checks = {
        "producer": all(report["checks"].values()) and report["passed"] == report["total"],
        "commands": report["external_commands"]["targeted_eslint"] == "passed" and report["external_commands"]["vite_production_build"].startswith("passed"),
        "files": len(report["files"]) == 6 and all(len(value["sha256"]) == 64 for value in report["files"].values()),
        "authorization": final["authorization"] == "append_phase1571_1574_major_stage_memo",
        "c101_frozen": c101["status"] == "requirements_frozen_not_started"
        and c101["confirmation_arm"]["primary_state"] == 24
        and len(c101["breadth_observation_arm"]["pattern_families"]) == 4,
    }
    result = {"phase": 1574, "campaign": "C100", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
