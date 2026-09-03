#!/usr/bin/env python3
"""Independent audit for Phase1486."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1486_c084_factorial_surface_atlas"
DISCOVERY = RESULT / "phase1465_c079_discovery_full_field_capture"
PRIOR = RESULT / "phase1476_c082_coordinate_atlas/atlas/mean_effect.float32.npy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


EFFECTS = ["relation", "entity", "object", "relation_entity", "relation_object", "entity_object", "relation_entity_object"]
ORDERS = {"relation": 1, "entity": 1, "object": 1, "relation_entity": 2, "relation_object": 2, "entity_object": 2, "relation_entity_object": 3}


def signs(row: dict) -> dict[str, int]:
    r = 1 if row["relation_match"] else -1
    e = 1 if row["entity_match"] else -1
    o = 1 if row["object_match"] else -1
    return {"relation": r, "entity": e, "object": o, "relation_entity": r * e, "relation_object": r * o, "entity_object": e * o, "relation_entity_object": r * e * o}


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/factorial_atlas_summary.json")
    contract = core.load(RESULT / "phase1484_c084_batch_deep_mining_contract/protocol/preregistration.json")
    py_compile.compile(str(TESTS / "phase1486_c084_factorial_surface_atlas.py"), doraise=True)
    atlas = np.load(OUT / "atlas/factorial_contrast_mean.float32.npy", mmap_mode="r")
    prior = np.load(PRIOR, mmap_mode="r")
    index = core.rows(DISCOVERY / "raw/discovery_role_field_index.jsonl")
    field = np.load(DISCOVERY / "raw/discovery_role_field.float16.npy", mmap_mode="r")
    rows = [row for row in index if row["partition"] == "response_discovery" and row["record_relation_id"] == "join" and row["surface"] == "a_colon"]
    keys = sorted({(row["family"], row["index"]) for row in rows})
    lookup = {(row["family"], row["index"], row["cell"]): row for row in rows}
    cells = ["111", "110", "101", "100", "011", "010", "001", "000"]
    total = np.zeros((7, 37, 9, 2560), dtype=np.float64)
    for family, index_value in keys:
        selected = [lookup[(family, index_value, cell)] for cell in cells]
        block = np.asarray(field[[row["row_index"] for row in selected]], dtype=np.float32)
        weight = np.asarray([[signs(row)[effect] * (2 ** ORDERS[effect]) / 8.0 for row in selected] for effect in EFFECTS], dtype=np.float32)
        total += np.tensordot(weight, block, axes=(1, 0))
    recomputed = (total / len(keys)).astype(np.float32)
    checks = {
        "status": final["status"] == "factorial_surface_atlas_complete" and all(final["output_checks"].values()),
        "shape": list(atlas.shape) == [7, 6, 3, 2, 37, 9, 2560],
        "relation_prior": float(np.max(np.abs(np.asarray(atlas[0], dtype=np.float32) - np.asarray(prior, dtype=np.float32)))) <= 1e-5,
        "selected_panel": float(np.max(np.abs(recomputed - np.asarray(atlas[:, 0, 0, 0], dtype=np.float32)))) <= 1e-5,
        "hash": core.sha(OUT / "atlas/factorial_contrast_mean.float32.npy") == summary["factorial_atlas"]["sha256"],
        "counts": int(np.sum(np.load(OUT / "atlas/sample_counts.int32.npy"))) == 414,
        "rows": len(core.rows(OUT / "analysis/layer_factorial_metrics.jsonl")) == 1998 and len(core.rows(OUT / "analysis/boundary_state35_factorial_panels.jsonl")) == 252,
        "no_model": not final["model_run"],
        "scope": summary["interpretation_boundary"].startswith("orthogonal contrasts"),
        "contract": contract["factorial_branch"]["contrasts"] == EFFECTS,
    }
    result = {"phase": 1486, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
