#!/usr/bin/env python3
"""Independent audit for Phase1517 diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1517_c088_full_dimensional_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/full_dimensional_diagnostic_summary.json")
    coordinates = core.rows(OUT / "analysis/coordinate_partition_pairs.jsonl")
    formation = core.rows(OUT / "analysis/formation_trajectory.jsonl")
    interaction = core.rows(OUT / "analysis/interaction_geometry.jsonl")
    behavior = core.rows(OUT / "analysis/behavior_truth_code_partition.jsonl")
    checks = {
        "authorization": final["authorization"] == "run_phase1518_c088_major_stage_closure",
        "row_counts": [len(coordinates), len(formation), len(interaction), len(behavior)] == [18, 148, 4, 16],
        "effects": sorted(set(row["effect"] for row in coordinates)) == ["code", "semantic", "semantic_code"],
        "partitions": sorted(set(row["partition"] for row in formation)) == sorted(["response_discovery", "confirmation", "lockbox", "fresh_external"]),
        "behavior_recompute": abs(sum(row["accuracy"] * row["count"] for row in behavior if row["codebook"] == "standard") / sum(row["count"] for row in behavior if row["codebook"] == "standard") - summary["behavior_boundary"]["standard_accuracy"]) < 1e-12,
        "scope": "not a localized semantic circuit" in summary["claim_boundary"],
        "no_projection": final["checks"]["no_reduced_projection"],
    }
    audit = {
        "phase": 1517,
        "campaign": "C088",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
