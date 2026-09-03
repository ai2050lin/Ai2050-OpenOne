#!/usr/bin/env python3
"""Independent audit for C103 existing-data atlas."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1588_c103_code_residualized_role_state_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1588_c103_code_residualized_role_state_atlas.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    graph = np.load(OUT / "raw/graph_residual_vectors.float32.npy", mmap_mode="r")
    breadth = np.load(OUT / "raw/breadth_residual_vectors.float32.npy", mmap_mode="r")
    atlas = core.rows(OUT / "analysis/role_state_atlas.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "scope": protocol["no_gate"] and "not causal purification" in protocol["claim_boundary"][1],
        "rows": len(atlas) == final["rows"] == 4 * 6 * 37 + 4 * 7 * 37,
        "shapes": graph.shape == (4, 6, 37, 3, 2560) and breadth.shape == (4, 7, 37, 3, 2560),
        "hashes": core.sha(OUT / "raw/graph_residual_vectors.float32.npy") == final["graph_residual_sha256"] and core.sha(OUT / "raw/breadth_residual_vectors.float32.npy") == final["breadth_residual_sha256"],
        "finite": bool(np.isfinite(graph).all() and np.isfinite(breadth).all()),
        "candidates": len(final["candidates"]) == 8 and all(row["upstream"] and row["state"] <= 24 for row in final["candidates"]),
        "claim": "post-hoc" in final["claim_boundary"],
    }
    result = {"phase": 1588, "campaign": "C103", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
