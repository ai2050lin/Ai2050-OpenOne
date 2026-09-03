#!/usr/bin/env python3
"""Independent audit for C173."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1707_c173_role_specific_full_coordinate_response"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/role_specific_coordinate_lock.json")
    final = core.load(OUT / "analysis/final.json")
    discovery = np.load(OUT / "raw/discovery_full_response.float16.npy", mmap_mode="r")
    validation = np.load(OUT / "raw/validation_response.float16.npy", mmap_mode="r")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "full_scan": list(discovery.shape) == [2, 8, 2560, 6, 2560],
        "validation": list(validation.shape) == [2, 3, 16, 64, 6, 2560],
        "selection_blind": lock["confirmation_and_fresh_unread"],
        "role_locks": all(len(lock["roles"][r]["coordinates"]) == 64 for r in ("primary", "query")),
        "signed_metrics": "signed_nrmse" in protocol["primary_metrics"],
        "hash": core.sha(Path(__file__).with_name("phase1707_c173_role_specific_full_coordinate_response.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1707, "campaign": "C173", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
