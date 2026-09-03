#!/usr/bin/env python3
"""Independent audit for C196 multi-dose orthogonal identification."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1730_c196_multidose_orthogonal_identification"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/multidose_identification.json"); producer = Path(__file__).with_name("phase1730_c196_multidose_orthogonal_identification.py")
    actual = np.load(OUT / "raw/orthogonal_actual.float16.npy", mmap_mode="r"); predicted = np.load(OUT / "raw/orthogonal_predicted.float16.npy", mmap_mode="r"); patterns = np.load(OUT / "protocol/hadamard_patterns.float32.npy")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "shape": list(actual.shape) == [14, 3, 16, 2, 6, 2560] and list(predicted.shape) == list(actual.shape), "orthogonal": bool(np.allclose(patterns @ patterns.T, 64 * np.eye(16))), "finite_sample": bool(np.isfinite(np.asarray(actual[:, :, :, :, :, ::263], dtype=np.float32)).all()), "three_doses": len(report["dose_rows"]) == 3, "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1730, "campaign": "C196", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
