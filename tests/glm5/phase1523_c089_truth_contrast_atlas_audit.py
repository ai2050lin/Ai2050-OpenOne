#!/usr/bin/env python3
"""Independent audit for Phase1523."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1523_c089_truth_contrast_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    summary = core.load(OUT / "analysis/truth_contrast_atlas_summary.json")
    final = core.load(OUT / "analysis/final.json")
    group = np.load(OUT / "atlas/group_truth_contrast.float16.npy", mmap_mode="r")
    mean = np.load(OUT / "atlas/partition_family_truth_contrast_mean.float32.npy", mmap_mode="r")
    counts = np.load(OUT / "atlas/partition_family_counts.int32.npy")
    py_compile.compile(str(TESTS / "phase1523_c089_truth_contrast_atlas.py"), doraise=True)
    recomputed = np.asarray(group[[row["group_index"] for row in core.rows(OUT / "atlas/group_truth_contrast_index.jsonl") if row["partition"] == "response_discovery" and row["family"] == "synonym"]], dtype=np.float32).mean(axis=0)
    checks = {
        "status": final["status"] == "counterbalanced_truth_contrast_atlas_complete_unqualified",
        "shapes": list(group.shape) == [45, 2, 37, 4, 2560] and list(mean.shape) == [3, 3, 2, 37, 4, 2560],
        "counts": bool(np.all(counts == 5)),
        "state0": float(np.max(np.abs(np.asarray(group[:, :, 0], dtype=np.float32)))) == 0.0,
        "recompute": float(np.max(np.abs(recomputed - np.asarray(mean[0, 0])))) <= 2e-3,
        "hashes": core.sha(OUT / "atlas/group_truth_contrast.float16.npy") == summary["files"]["group"]["sha256"] and core.sha(OUT / "atlas/partition_family_truth_contrast_mean.float32.npy") == summary["files"]["mean"]["sha256"],
        "finite": bool(np.isfinite(np.asarray(mean)).all()),
        "unqualified": summary["behavior_qualified_families"] == [] and "descriptive" in summary["evidence_scope"],
        "authorization": final["authorization"] == "run_phase1524_c089_discovery_observation_freeze",
        "summary": all(summary["checks"].values()),
    }
    result = {"phase": 1523, "campaign": "C089", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
