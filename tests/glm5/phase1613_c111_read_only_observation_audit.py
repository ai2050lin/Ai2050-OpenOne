#!/usr/bin/env python3
"""Independent audit for the C111 read-only observation."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1613_c111_read_only_observation.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "analysis/observation_report.json")
    audit = core.load(OUT / "audit/internal_observation_audit.json")
    pairs = core.rows(OUT / "analysis/pair_value_role_geometry.jsonl")
    summary = core.rows(OUT / "analysis/pair_value_role_geometry_summary.jsonl")
    trajectory = core.rows(OUT / "analysis/cross_archive_role_state_trajectory.jsonl")
    matrix = core.rows(OUT / "analysis/state19_role_cosine_matrix.jsonl")
    checks = {
        "producer": core.sha(producer) == audit["producer_sha256"],
        "internal": audit["all_checks_passed"],
        "hashes": core.sha(OUT / "analysis/pair_value_role_geometry.jsonl") == audit["pair_sha256"] and core.sha(OUT / "analysis/cross_archive_role_state_trajectory.jsonl") == audit["trajectory_sha256"],
        "pairs": len(pairs) == 192 and len({row["pair_id"] for row in pairs}) == 192,
        "summary": len(summary) == 8 and all(row["pairs"] == 24 for row in summary),
        "trajectory": len(trajectory) == 518,
        "matrix": len(matrix) == 98,
        "missingness": len(report["planned_missingness"]) == 4,
        "boundary": "no new model behavior" in report["interpretation_boundary"],
        "authorization": report["authorization"] == "run_phase1614_c111_synthesis_heatmap_and_closure",
    }
    result = {"phase": 1613, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_observation_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
