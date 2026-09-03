#!/usr/bin/env python3
"""Independent audit for Phase1514."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1514_c088_factorial_field_atlas"
CAPTURE = RESULT / "phase1513_c088_unified_forward_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1514_c088_factorial_field_atlas as atlas


def main() -> None:
    summary = core.load(OUT / "analysis/factorial_field_atlas_summary.json")
    effect = np.load(OUT / "atlas/group_factorial_effect.float16.npy", mmap_mode="r")
    aggregate = np.load(OUT / "atlas/partition_factorial_effect_mean.float32.npy", mmap_mode="r")
    source = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    source_index = {row["case_id"]: row for row in core.rows(CAPTURE / "raw/all_role_field_index.jsonl")}
    groups = core.rows(CAPTURE / "material/stratified_composition_sets.jsonl")
    py_compile.compile(str(TESTS / "phase1514_c088_factorial_field_atlas.py"), doraise=True)
    exact = True
    for gi in (0, 71, 143, 215, 247):
        group = groups[gi]
        for ui, surface in enumerate(("a_code", "b_code")):
            rows = [source_index[group[f"{surface}_{codebook}_{semantic}"]] for codebook in ("standard", "reversed") for semantic in ("same", "different")]
            block = np.asarray(source[[row["row_index"] for row in rows]], dtype=np.float32)
            expected = np.tensordot(atlas.weights(rows), block, axes=(1, 0)).astype(np.float16)
            exact = exact and bool(np.array_equal(expected, np.asarray(effect[gi, ui])))
    checks = {
        "hashes": core.sha(OUT / "atlas/group_factorial_effect.float16.npy") == summary["files"]["group"]["sha256"] and core.sha(OUT / "atlas/partition_factorial_effect_mean.float32.npy") == summary["files"]["aggregate"]["sha256"],
        "shapes": list(effect.shape) == [248, 2, 3, 37, 4, 2560] and list(aggregate.shape) == [4, 2, 3, 37, 4, 2560],
        "exact_recompute": exact,
        "causal_zero": summary["checks"]["source_causal_zero"] and summary["checks"]["candidate_code_causal_zero"],
        "factor_zero": summary["checks"]["state0_semantic_counterbalance"] and summary["checks"]["state0_interaction_zero"],
        "summary": all(summary["checks"].values()),
    }
    result = {"phase": 1514, "campaign": "C088", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
