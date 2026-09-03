#!/usr/bin/env python3
"""Independent audit for Phase1476."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1476_c082_coordinate_atlas"
CONTRACT = RESULT / "phase1475_c082_coordinate_atlas_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1476_c082_coordinate_atlas as phase


def recompute_panel(relation: str, split: str, surface: str, protocol: dict) -> tuple[np.ndarray, np.ndarray, int]:
    fields, indexes, lookups = phase.load_sources()
    c079 = core.load(RESULT / "phase1463_c079_aggregate_observation_contract/protocol/preregistration.json")
    keys = phase.panel_set_keys(indexes[split], split, relation)
    effects = []
    for family, index in keys:
        ids = phase.row_ids_for(lookups[split], family, index, relation, surface, list(c079["cells"]))
        effects.append(phase.sample_effect(fields[split], ids))
    stack = np.stack(effects, axis=0)
    mean = np.mean(stack, axis=0, dtype=np.float64).astype(np.float32)
    positive = np.sum(stack > 0, axis=0)
    negative = np.sum(stack < 0, axis=0)
    consistency = np.where(mean > 0, positive / len(stack), np.where(mean < 0, negative / len(stack), (len(stack) - positive - negative) / len(stack))).astype(np.float16)
    return mean, consistency, len(stack)


def main() -> None:
    metadata = core.load(OUT / "analysis/atlas_metadata.json")
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    means = np.load(OUT / "atlas/mean_effect.float32.npy", mmap_mode="r")
    signs = np.load(OUT / "atlas/sign_consistency.float16.npy", mmap_mode="r")
    counts = np.load(OUT / "atlas/sample_counts.int32.npy")
    layer_rows = core.rows(OUT / "analysis/layer_role_metrics.jsonl")
    onset_rows = core.rows(OUT / "analysis/onset_metrics.jsonl")
    panel_rows = core.rows(OUT / "analysis/panel_stability.jsonl")
    py_compile.compile(str(TESTS / "phase1476_c082_coordinate_atlas.py"), doraise=True)
    panels = [("join", "response_discovery", "a_colon"), ("select", "confirmation", "b_colon"), ("praise", "lockbox", "a_colon")]
    panel_checks = []
    for relation, split, surface in panels:
        mean, consistency, count = recompute_panel(relation, split, surface, protocol)
        index = (protocol["axes"]["relations"].index(relation), protocol["axes"]["splits"].index(split), protocol["axes"]["surfaces"].index(surface))
        panel_checks.append(bool(np.array_equal(mean, means[index]) and np.array_equal(consistency, signs[index]) and count == counts[index]))
    files = [OUT / "atlas/mean_effect.float32.npy", OUT / "atlas/sign_consistency.float16.npy", OUT / "atlas/sample_counts.int32.npy"]
    checks = {
        "metadata": final["atlas_complete"] and all(metadata["output_checks"].values()),
        "shape": means.shape == signs.shape == (6, 3, 2, 37, 9, 2560) and counts.shape == (6, 3, 2),
        "hashes": all(metadata["files"][path.name]["sha256"] == core.sha(path) for path in files),
        "finite": bool(np.isfinite(means).all() and np.isfinite(signs).all()),
        "rows": len(layer_rows) == 11988 and len(onset_rows) == 324 and len(panel_rows) == 1998,
        "fixed_panels": all(panel_checks),
        "upstream_zero": all(row["all_panels_exact_zero"] for row in panel_rows if row["role"].startswith("record_")),
        "authorization": final["authorization"] == "run_phase1477_c082_atlas_audit_and_synthesis",
        "no_model": metadata["model_run"] is False,
    }
    result = {"phase": 1476, "campaign": "C082", "checks": checks, "fixed_panel_checks": panel_checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
