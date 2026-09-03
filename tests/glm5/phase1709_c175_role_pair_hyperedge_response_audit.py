#!/usr/bin/env python3
"""Independent audit for C175."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1709_c175_role_pair_hyperedge_response"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/pair_interaction.float16.npy", mmap_mode="r")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "shape": list(raw.shape) == [2, 2, 8, 8, 6, 2560],
        "pairs": all(len(v) == 8 for v in protocol["pairs"].values()),
        "fresh": protocol["partitions"] == ["discovery", "fresh"],
        "hash": core.sha(Path(__file__).with_name("phase1709_c175_role_pair_hyperedge_response.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1709, "campaign": "C175", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
