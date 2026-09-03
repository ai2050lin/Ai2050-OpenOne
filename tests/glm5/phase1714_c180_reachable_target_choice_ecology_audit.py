#!/usr/bin/env python3
"""Independent audit for C180."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1714_c180_reachable_target_choice_ecology"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    final = core.load(OUT / "analysis/final.json")
    response = np.load(OUT / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "behavior_before_hidden": protocol["hidden_policy"].startswith("no HiddenState"),
        "balanced": protocol["zero_models"]["always_A"] == 0.5,
        "eligible": len(lock["eligible_families"]) > 0,
        "response": list(response.shape) == [3, 3 * len(lock["eligible_families"]), 64, 6, 2560],
        "all_tokens": (OUT / "raw/anchor_all_token_all_checkpoint.bf16.npy").exists(),
        "hash": core.sha(Path(__file__).with_name("phase1714_c180_reachable_target_choice_ecology.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1714, "campaign": "C180", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
