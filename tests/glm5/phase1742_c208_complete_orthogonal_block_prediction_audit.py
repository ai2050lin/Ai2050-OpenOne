#!/usr/bin/env python3
"""Independent audit for C208."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C208


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    calibration = np.load(OUT / "protocol/calibration_patterns.float32.npy")
    holdout = np.load(OUT / "protocol/holdout_patterns.float32.npy")
    checks = {"final": final["all_checks_passed"], "orthogonal_complete": calibration.shape == (32, 32) and bool(np.allclose(calibration @ calibration.T, 32 * np.eye(32))), "holdout_unseen": all(not any(np.array_equal(row, known) or np.array_equal(row, -known) for known in calibration) for row in holdout), "calibration_shape": np.load(OUT / "raw/calibration_effects.float16.npy", mmap_mode="r").shape == (18, 32, 2, 2, 6, common.DIM), "holdout_shape": np.load(OUT / "raw/holdout_effects.float16.npy", mmap_mode="r").shape == (18, 8, 2, 2, 6, common.DIM), "producer_hash": core.sha(Path(__file__).with_name("phase1742_c208_complete_orthogonal_block_prediction.py")) == protocol["producer_sha256"]}
    report = {"phase": 1742, "campaign": "C208", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
