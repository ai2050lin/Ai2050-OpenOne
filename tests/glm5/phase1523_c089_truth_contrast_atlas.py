#!/usr/bin/env python3
"""Phase1523: exact counterbalanced truth-contrast atlas for C089."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
CAPTURE = RESULT / "phase1522_c089_unified_forward_capture"
OUT = RESULT / "phase1523_c089_truth_contrast_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CELLS = ("aa", "ab", "ba", "bb")
SURFACES = ("a_question", "b_question")


def contrast(block: np.ndarray) -> np.ndarray:
    """Mean(true aa,bb) minus mean(false ab,ba)."""
    return (block[0] + block[3] - block[1] - block[2]) / 2.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1523 exists")
    parent = core.load(CAPTURE / "analysis/final.json")
    parent_audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    capture_summary = core.load(CAPTURE / "analysis/unified_behavior_and_capture_summary.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1523_c089_truth_contrast_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1522 authorization missing")
    field_path = CAPTURE / "raw/all_role_field.float16.npy"
    if core.sha(field_path) != capture_summary["files"]["field"]["sha256"]:
        raise RuntimeError("field hash mismatch")
    field = np.load(field_path, mmap_mode="r")
    index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    lookup = {(row["set_id"], row["surface"], row["cell"]): row for row in index}
    groups = core.rows(CONTRACT / "material/relation_composition_sets.jsonl")
    families, partitions = tuple(protocol["families"]), tuple(protocol["partitions"])
    family_index = {value: i for i, value in enumerate(families)}
    partition_index = {value: i for i, value in enumerate(partitions)}
    group_path = OUT / "atlas/group_truth_contrast.float16.npy"
    group_path.parent.mkdir(parents=True, exist_ok=True)
    group_effect = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.float16, shape=(45, 2, 37, 4, 2560))
    sums = np.zeros((3, 3, 2, 37, 4, 2560), dtype=np.float64)
    counts = np.zeros((3, 3), dtype=np.int32)
    group_index, logit_rows = [], []
    for gi, group in enumerate(groups):
        pi, fi = partition_index[group["partition"]], family_index[group["family"]]
        counts[pi, fi] += 1
        for ui, surface in enumerate(SURFACES):
            rows = [lookup[(group["set_id"], surface, cell)] for cell in CELLS]
            block = np.asarray(field[[row["row_index"] for row in rows]], dtype=np.float32)
            effect = contrast(block)
            group_effect[gi, ui] = effect.astype(np.float16)
            sums[pi, fi, ui] += effect
            margins = np.asarray([
                row["scores"][row["candidates"].index("yes")] - row["scores"][row["candidates"].index("no")]
                for row in rows
            ], dtype=np.float64)
            logit_rows.append({
                "group_index": gi, "set_id": group["set_id"], "family": group["family"],
                "partition": group["partition"], "surface": surface,
                "truth_contrast": float((margins[0] + margins[3] - margins[1] - margins[2]) / 2.0),
            })
        group_index.append({"group_index": gi, "set_id": group["set_id"], "family": group["family"], "partition": group["partition"]})
    group_effect.flush()
    del group_effect
    mean_path = OUT / "atlas/partition_family_truth_contrast_mean.float32.npy"
    mean = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(3, 3, 2, 37, 4, 2560))
    for pi in range(3):
        for fi in range(3):
            mean[pi, fi] = sums[pi, fi] / counts[pi, fi]
    mean.flush()
    del mean
    group_index_path = OUT / "atlas/group_truth_contrast_index.jsonl"
    logit_path = OUT / "analysis/group_logit_truth_contrasts.jsonl"
    core.write_rows(group_index_path, group_index)
    core.write_rows(logit_path, logit_rows)
    np.save(OUT / "atlas/partition_family_counts.int32.npy", counts)
    group_read = np.load(group_path, mmap_mode="r")
    mean_read = np.load(mean_path, mmap_mode="r")
    checks = {
        "group_shape": list(group_read.shape) == [45, 2, 37, 4, 2560],
        "mean_shape": list(mean_read.shape) == [3, 3, 2, 37, 4, 2560],
        "counts": bool(np.all(counts == 5)),
        "finite": bool(np.isfinite(np.asarray(mean_read)).all()) and all(bool(np.isfinite(np.asarray(group_read[start:start + 8])).all()) for start in range(0, 45, 8)),
        "state0_counterbalance": float(np.max(np.abs(np.asarray(group_read[:, :, 0], dtype=np.float32)))) == 0.0,
        "logit_rows": len(logit_rows) == 90,
        "qualification_preserved": parent["behavior_qualified_families"] == [],
        "semantic_interpretation_blocked": len(parent["behavior_qualified_families"]) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1523, "campaign": "C089",
        "formula": "C_truth=(H_aa+H_bb-H_ab-H_ba)/2 = mean(true)-mean(false)",
        "identity_counterbalance": "each source and target identity appears once with each truth sign inside every set and surface",
        "axis_orders": {
            "group": ["group", "surface", "state", "role", "coordinate"],
            "mean": ["partition", "family", "surface", "state", "role", "coordinate"],
        },
        "partitions": partitions, "families": families, "surfaces": SURFACES, "roles": tuple(protocol["roles"]),
        "behavior_qualified_families": parent["behavior_qualified_families"],
        "evidence_scope": "full-dimensional descriptive response atlas only; no family passed the behavior qualification required for semantic Hidden-State interpretation",
        "checks": checks,
        "files": {
            "group": {"bytes": group_path.stat().st_size, "sha256": core.sha(group_path)},
            "group_index": {"sha256": core.sha(group_index_path)},
            "mean": {"bytes": mean_path.stat().st_size, "sha256": core.sha(mean_path)},
            "logit": {"sha256": core.sha(logit_path)},
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/truth_contrast_atlas_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1523, "campaign": "C089", "status": "counterbalanced_truth_contrast_atlas_complete_unqualified", "authorization": "run_phase1524_c089_discovery_observation_freeze"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
