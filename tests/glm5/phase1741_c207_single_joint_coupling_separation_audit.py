#!/usr/bin/env python3
"""Independent audit for C207."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C207


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    writes = np.load(OUT / "raw/actual_single_writes.float32.npy")
    checks = {"final": final["all_checks_passed"], "shape": np.load(OUT / "raw/summed_single_effects.float16.npy", mmap_mode="r").shape == (18, 2, 2, common.WIDTH, common.DIM), "all_single_coordinates": writes.shape == (18, 32, 2), "same_dose_registered": "exactly matching" in protocol["single_coordinate_delta"], "finite": bool(np.isfinite(writes).all()), "producer_hash": core.sha(Path(__file__).with_name("phase1741_c207_single_joint_coupling_separation.py")) == protocol["producer_sha256"]}
    report = {"phase": 1741, "campaign": "C207", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
