#!/usr/bin/env python3
"""Independent audit for Phase1531."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1531_c090_canonical_truth_contrast_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/canonical_truth_contrast_atlas.json")
    group = np.load(OUT / "atlas/canonical_group_truth_contrast.float16.npy", mmap_mode="r")
    mean = np.load(OUT / "atlas/canonical_partition_family_truth_mean.float32.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1531_c090_canonical_truth_contrast_atlas.py"), doraise=True)
    checks = {
        "status": final["status"] == "canonical_truth_contrast_atlas_complete",
        "shapes": list(group.shape) == [45, 2, 37, 4, 2560] and list(mean.shape) == [3, 3, 2, 37, 4, 2560],
        "source": float(np.max(np.abs(np.asarray(group[:, :, :, 0], dtype=np.float32)))) == 0.0,
        "finite": bool(np.isfinite(np.asarray(mean)).all()),
        "hashes": core.sha(OUT / "atlas/canonical_group_truth_contrast.float16.npy") == summary["files"]["group"]["sha256"] and core.sha(OUT / "atlas/canonical_partition_family_truth_mean.float32.npy") == summary["files"]["mean"]["sha256"],
        "unqualified": summary["behavior_qualified_families"] == [],
        "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1532_c090_discovery_observation_freeze",
    }
    result = {"phase": 1531, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
