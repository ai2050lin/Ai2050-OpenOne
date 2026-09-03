#!/usr/bin/env python3
"""Independent audit for C195 signed role/checkpoint trajectories."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1729_c195_signed_role_checkpoint_trajectory"; PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c195_signed_operator_trajectory.json"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/signed_trajectory.json"); producer = Path(__file__).with_name("phase1729_c195_signed_role_checkpoint_trajectory.py")
    raw = np.load(OUT / "raw/signed_q23_q24_q25.float16.npy", mmap_mode="r"); baseline = np.load(OUT / "raw/baseline_role_states.float16.npy", mmap_mode="r"); payload = core.load(PUBLIC)
    sample = np.asarray(raw[[0, 55, 111], :, :, :, ::257], dtype=np.float32)
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"], "raw_shape": list(raw.shape) == [112, 64, 2, 6, 2560],
        "baseline_shape": list(baseline.shape) == [112, 4, 6, 2560], "sample_finite": bool(np.isfinite(sample).all()),
        "groups": report["groups"] == 56 and len(report["group_rows"]) == 56, "asset": len(payload["dimensions"]) == 2560 and len(payload["rows"]) == 696,
        "asset_hash": core.sha(PUBLIC) == final["asset"]["sha256"], "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {"phase": 1729, "campaign": "C195", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
