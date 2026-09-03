#!/usr/bin/env python3
"""Independent audit for Phase1620 / C113 field adjudication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/field_adjudication.json")
    unit = np.load(OUT / "analysis/unit_truth_role_state.float32.npy", mmap_mode="r")
    mean = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    results = core.rows(OUT / "analysis/field_prediction_results.jsonl")
    trajectory = core.rows(OUT / "analysis/role_state_trajectory.jsonl")
    checks = {
        "producer": report["producer_sha256"] == core.sha(TESTS / "phase1620_c113_field_adjudication.py"),
        "unit": unit.shape == (24, 7, 37, 2560) and unit.dtype == np.float32 and core.sha(OUT / "analysis/unit_truth_role_state.float32.npy") == report["unit_sha256"],
        "mean": mean.shape == (2, 2, 7, 37, 2560) and mean.dtype == np.float32 and core.sha(OUT / "analysis/mean_truth_role_state.float32.npy") == report["mean_sha256"],
        "finite": bool(np.isfinite(unit).all() and np.isfinite(mean).all()),
        "results": len(results) == 2 and all(set(row["gates"]) == {"cross_partition", "reference", "support_overlap"} for row in results),
        "trajectory": len(trajectory) == 518 and sum(row["state_kind"] == "embedding" for row in trajectory) == 14,
        "hashes": core.sha(OUT / "analysis/field_prediction_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/role_state_trajectory.jsonl") == report["trajectory_sha256"],
        "authorization": report["authorization"] == "execute_phase1621_c113_coordinate_and_role_interventions_regardless_of_field_gate",
    }
    audit = {"phase": 1620, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": report["authorization"]}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/independent_field_adjudication_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
