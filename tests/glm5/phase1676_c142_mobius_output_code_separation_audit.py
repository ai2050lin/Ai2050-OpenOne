#!/usr/bin/env python3
"""Independent audit for C142."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1676_c142_mobius_output_code_separation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def final() -> None:
    freeze = core.load(OUT / "protocol/frozen_nominees.json")
    result = core.load(OUT / "analysis/confirmation.json")
    discovery = np.load(OUT / "analysis/discovery_mobius.float32.npy", mmap_mode="r")
    confirmation = np.load(OUT / "analysis/confirmation_mobius.float32.npy", mmap_mode="r")
    checks = {
        "internal": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "discovery_shape": list(discovery.shape) == [5, 4, 7, 6, 38, 2560],
        "confirmation_shape": list(confirmation.shape) == [5, 4, 7, 6, 38, 2560],
        "nominees": sum(len([k for k in arm if k != "output_code"]) for arm in freeze["nominees"].values()) == 35,
        "code_nominees": sum("output_code" in arm for arm in freeze["nominees"].values()) == 5,
        "results": result["total_semantic_nominees"] == 35,
        "source_hash": core.sha(TESTS / "result/phase1675_c141_multifamily_full_coordinate_atlas/raw/qwen3_six_role_field.bf16.npy") == freeze["source_hashes"]["C141_role"],
        "boundary": "no natural operator" in result["claim_boundary"],
    }
    report = {"phase": 1676, "campaign": "C142", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_replication_count": result["passing_semantic_nominees"], "authorization": "start_C143"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    final()
