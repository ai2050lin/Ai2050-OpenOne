#!/usr/bin/env python3
"""Independent audit for C206."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C206


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    parent_protocol = core.load(common.C205 / "protocol/preregistration.json")
    effects = np.load(OUT / "raw/joint_effects.float16.npy", mmap_mode="r")
    baseline = np.load(OUT / "raw/baseline_full.float16.npy", mmap_mode="r")
    checks = {
        "final": final["all_checks_passed"],
        "shape": effects.shape == (36, 6, 2, 2, common.WIDTH, common.DIM),
        "baseline": baseline.shape == (36, 4, common.WIDTH, common.DIM),
        "odd_even_separate": parent_protocol["response_split"]["odd"] != parent_protocol["response_split"]["even"],
        "full_token": "all tokens" in protocol["saved"],
        "repeat_floor": core.load(OUT / "analysis/capture.json")["repeat_hidden_max_abs"] == 0.0,
        "producer_hash": core.sha(Path(__file__).with_name("phase1740_c206_full_sequence_response_regimes.py")) == protocol["producer_sha256"],
    }
    report = {"phase": 1740, "campaign": "C206", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
