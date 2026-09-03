#!/usr/bin/env python3
"""Phase1531: canonical counterbalanced truth-contrast atlas after right-padding calibration."""
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
CAPTURE = RESULT / "phase1530_c090_canonical_full_recapture"
OUT = RESULT / "phase1531_c090_canonical_truth_contrast_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CELLS = ("aa", "ab", "ba", "bb")
SURFACES = ("a_question", "b_question")


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1531 exists")
    parent = core.load(CAPTURE / "analysis/final.json")
    parent_audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    capture = core.load(CAPTURE / "analysis/canonical_behavior_and_capture_summary.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1531_c090_canonical_truth_contrast_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1530 authorization missing")
    field_path = CAPTURE / "raw/canonical_all_role_field.float16.npy"
    if core.sha(field_path) != capture["files"]["field"]["sha256"]:
        raise RuntimeError("capture hash mismatch")
    field = np.load(field_path, mmap_mode="r")
    index = core.rows(CAPTURE / "raw/canonical_all_role_field_index.jsonl")
    lookup = {(row["set_id"], row["surface"], row["cell"]): row for row in index}
    groups = core.rows(CONTRACT / "material/relation_composition_sets.jsonl")
    families, partitions = protocol["families"], protocol["partitions"]
    fi = {value: i for i, value in enumerate(families)}
    pi = {value: i for i, value in enumerate(partitions)}
    group_path = OUT / "atlas/canonical_group_truth_contrast.float16.npy"
    group_path.parent.mkdir(parents=True, exist_ok=True)
    effects = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.float16, shape=(45, 2, 37, 4, 2560))
    sums = np.zeros((3, 3, 2, 37, 4, 2560), dtype=np.float64)
    counts = np.zeros((3, 3), dtype=np.int32)
    group_index, logit_rows = [], []
    for gi, group in enumerate(groups):
        p, f = pi[group["partition"]], fi[group["family"]]
        counts[p, f] += 1
        for ui, surface in enumerate(SURFACES):
            rows = [lookup[(group["set_id"], surface, cell)] for cell in CELLS]
            block = np.asarray(field[[row["row_index"] for row in rows]], dtype=np.float32)
            effect = (block[0] + block[3] - block[1] - block[2]) / 2.0
            effects[gi, ui] = effect.astype(np.float16)
            sums[p, f, ui] += effect
            margins = np.asarray([row["scores"][row["candidates"].index("yes")] - row["scores"][row["candidates"].index("no")] for row in rows])
            logit_rows.append({"group_index": gi, "set_id": group["set_id"], "family": group["family"], "partition": group["partition"], "surface": surface, "truth_contrast": float((margins[0] + margins[3] - margins[1] - margins[2]) / 2)})
        group_index.append({"group_index": gi, "set_id": group["set_id"], "family": group["family"], "partition": group["partition"]})
    effects.flush()
    del effects
    mean_path = OUT / "atlas/canonical_partition_family_truth_mean.float32.npy"
    mean = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=(3, 3, 2, 37, 4, 2560))
    for p in range(3):
        for f in range(3):
            mean[p, f] = sums[p, f] / counts[p, f]
    mean.flush()
    del mean
    index_path = OUT / "atlas/canonical_group_truth_contrast_index.jsonl"
    logit_path = OUT / "analysis/canonical_group_logit_truth_contrasts.jsonl"
    core.write_rows(index_path, group_index)
    core.write_rows(logit_path, logit_rows)
    np.save(OUT / "atlas/partition_family_counts.int32.npy", counts)
    group_read = np.load(group_path, mmap_mode="r")
    mean_read = np.load(mean_path, mmap_mode="r")
    checks = {
        "shapes": list(group_read.shape) == [45, 2, 37, 4, 2560] and list(mean_read.shape) == [3, 3, 2, 37, 4, 2560],
        "counts": bool(np.all(counts == 5)), "finite": bool(np.isfinite(np.asarray(mean_read)).all()),
        "state0": float(np.max(np.abs(np.asarray(group_read[:, :, 0], dtype=np.float32)))) == 0.0,
        "source_all_states": float(np.max(np.abs(np.asarray(group_read[:, :, :, 0], dtype=np.float32)))) == 0.0,
        "logits": len(logit_rows) == 90, "unqualified": capture["behavior_qualified_families"] == [],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1531, "campaign": "C090", "formula": "C_truth=(H_aa+H_bb-H_ab-H_ba)/2",
        "engine": capture["engine"], "behavior_qualified_families": capture["behavior_qualified_families"],
        "evidence_scope": "canonical descriptive atlas; semantic interpretation remains blocked",
        "checks": checks,
        "files": {"group": {"sha256": core.sha(group_path), "bytes": group_path.stat().st_size}, "mean": {"sha256": core.sha(mean_path), "bytes": mean_path.stat().st_size}, "index": {"sha256": core.sha(index_path)}, "logit": {"sha256": core.sha(logit_path)}},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/canonical_truth_contrast_atlas.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1531, "campaign": "C090", "status": "canonical_truth_contrast_atlas_complete", "authorization": "run_phase1532_c090_discovery_observation_freeze"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
