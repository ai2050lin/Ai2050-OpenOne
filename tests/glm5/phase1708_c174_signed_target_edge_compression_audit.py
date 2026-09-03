#!/usr/bin/env python3
"""Independent audit for C174."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1708_c174_signed_target_edge_compression"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    masks = np.load(OUT / "analysis/discovery_80pct_edge_masks.bool.npy", mmap_mode="r")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "adaptive_counts": len(protocol["energy_fractions"]) == 4,
        "mask": list(masks.shape) == [2, 3, 4, 64, 6, 2560] and masks.dtype == np.bool_,
        "all_branches": set(final["headline"]["summary_80pct"]) == {"primary", "query"},
        "hash": core.sha(Path(__file__).with_name("phase1708_c174_signed_target_edge_compression.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1708, "campaign": "C174", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
