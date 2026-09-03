#!/usr/bin/env python3
"""Independent audit for Phase1499."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1499_c086_four_factor_atlas"
CAPTURE = TESTS / "result/phase1498_c086_all_case_field_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    summary = core.load(OUT / "analysis/four_factor_atlas_summary.json")
    full = np.load(OUT / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    key = np.load(OUT / "atlas/stratum_key_effect_mean.float32.npy", mmap_mode="r")
    counts = np.load(OUT / "atlas/stratum_sample_counts.int32.npy")
    field = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1499_c086_four_factor_atlas.py"), doraise=True)
    checks = {
        "shapes": list(full.shape) == [15, 6, 3, 2, 37, 7, 2560]
        and list(key.shape) == [4, 3, 6, 3, 2, 37, 7, 2560],
        "hashes": core.sha(OUT / "atlas/all_four_factor_contrast_mean.float32.npy")
        == summary["files"]["full"]["sha256"]
        and core.sha(OUT / "atlas/stratum_key_effect_mean.float32.npy")
        == summary["files"]["key"]["sha256"],
        "counts": bool(np.all(counts[0] == 12))
        and bool(np.all(counts[2] == 12))
        and bool(np.all(counts[[1, 3]] == 0)),
        "key_views": float(np.max(np.abs(np.asarray(key[0]) - np.asarray(key[2])))) == 0.0,
        "finite_samples": bool(np.isfinite(np.asarray(full[:, :, :, :, 35, 6])).all())
        and bool(np.isfinite(np.asarray(field[[0, 6911], :, :, :])).all()),
        "runtime": all(summary["checks"].values()),
    }
    result = {
        "phase": 1499,
        "campaign": "C086",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
