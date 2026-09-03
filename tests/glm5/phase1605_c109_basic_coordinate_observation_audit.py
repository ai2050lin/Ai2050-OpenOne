#!/usr/bin/env python3
"""Independent audit of the C109 basic coordinate observation."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1605_c109_basic_coordinate_observation.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "audit/basic_observation_internal_audit.json")
    summary = core.load(OUT / "analysis/basic_observation_summary.json")
    trajectory = core.rows(OUT / "analysis/role_state_truth_trajectory.jsonl")
    pair_rows = core.rows(OUT / "analysis/c108_pair_support_energy.jsonl")
    unit_truth = np.load(OUT / "analysis/unit_truth_role_state.float32.npy", mmap_mode="r")
    mean_truth = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "internal": report["all_checks_passed"],
        "unit_shape": unit_truth.shape == (24, 7, 37, 2560),
        "mean_shape": mean_truth.shape == (2, 2, 7, 37, 2560),
        "trajectory": len(trajectory) == 518 and {row["state"] for row in trajectory} == set(range(37)),
        "coordinates": all(0.0 <= row["cross_partition_topk_overlap"] <= 1.0 and 0.0 <= row["prospective_frozen_support_energy_fraction"] <= 1.000001 for row in trajectory),
        "pairs": len(pair_rows) == 192 and all(row["target_support_energy"] >= 0.0 and row["same_k_wrong_support_energy"] >= 0.0 for row in pair_rows),
        "candidate": len(summary["candidate_query_anchor_state19"]) == 2,
        "boundary": "already exposed" in summary["interpretation_rules"]["no_independent_confirmation"],
        "authorization": summary["authorization"] == "run_phase1606_c109_heatmap_synthesis_and_closure",
    }
    result = {"phase": 1605, "campaign": "C109", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_basic_observation_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
