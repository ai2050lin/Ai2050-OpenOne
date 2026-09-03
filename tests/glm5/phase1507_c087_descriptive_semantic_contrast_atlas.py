#!/usr/bin/env python3
"""Phase1507: full-dimensional descriptive C087 semantic contrast atlas."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1504_c087_cross_root_semeval_contract"
BEHAVIOR = RESULT / "phase1505_c087_behavior_stratification"
CAPTURE = RESULT / "phase1506_c087_all_case_field_capture"
OUT = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

STRATA = ("all", "success", "mixed")


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1507 exists")
    parent = core.load(CAPTURE / "analysis/final.json")
    audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    meta = core.load(CAPTURE / "analysis/capture_metadata.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        parent["authorization"] != "run_phase1507_c087_descriptive_semantic_contrast_atlas"
        or not audit["all_checks_passed"]
        or meta["execution_identity_gate_passed"]
        or core.sha(CAPTURE / "raw/all_role_field.float16.npy") != meta["raw_sha256"]
    ):
        raise RuntimeError("Phase1506 descriptive authorization/integrity missing")

    field = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    lookup = {row["case_id"]: row for row in index}
    groups = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    partitions = tuple(protocol["partitions"])
    surfaces = tuple(protocol["surfaces"])
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)

    group_path = OUT / "atlas/group_semantic_contrast.float32.npy"
    group_atlas = np.lib.format.open_memmap(
        group_path, mode="w+", dtype=np.float32, shape=(216, 2, 37, 3, 2560)
    )
    group_index = []
    for gi, group in enumerate(groups):
        for ui, surface in enumerate(surfaces):
            same = lookup[group[f"{surface}_same"]]["row_index"]
            different = lookup[group[f"{surface}_different"]]["row_index"]
            group_atlas[gi, ui] = (
                np.asarray(field[same], dtype=np.float32)
                - np.asarray(field[different], dtype=np.float32)
            )
        group_index.append({
            "group_index": gi,
            "set_id": group["set_id"],
            "partition": group["partition"],
            "item": group["item"],
            "lemma": group["lemma"],
            "source_instance_id": group["source_instance_id"],
            "stratum": group["stratum"],
        })
    group_atlas.flush()
    del group_atlas
    group_index_path = OUT / "atlas/group_semantic_contrast_index.jsonl"
    core.write_rows(group_index_path, group_index)

    group_read = np.load(group_path, mmap_mode="r")
    aggregate_path = OUT / "atlas/partition_stratum_semantic_mean.float32.npy"
    aggregate = np.lib.format.open_memmap(
        aggregate_path, mode="w+", dtype=np.float32, shape=(3, 3, 2, 37, 3, 2560)
    )
    aggregate[:] = 0
    counts = np.zeros((3, 3), dtype=np.int32)
    for si, stratum in enumerate(STRATA):
        for pi, partition in enumerate(partitions):
            selected = [
                row["group_index"] for row in group_index
                if row["partition"] == partition
                and (stratum == "all" or row["stratum"] == stratum)
            ]
            counts[si, pi] = len(selected)
            if selected:
                aggregate[si, pi] = np.asarray(group_read[selected], dtype=np.float64).mean(axis=0)
    aggregate.flush()
    del aggregate
    np.save(OUT / "atlas/partition_stratum_counts.int32.npy", counts)
    aggregate_read = np.load(aggregate_path, mmap_mode="r")

    selected = core.rows(CONTRACT / "material/selected_instances.jsonl")
    candidate_balance = {
        partition: Counter(
            row["positive_candidate"] for row in selected if row["partition"] == partition
        ) == Counter(
            row["negative_candidate"] for row in selected if row["partition"] == partition
        )
        for partition in partitions
    }
    state0_all_max_abs = float(np.max(np.abs(np.asarray(aggregate_read[0, :, :, 0]))))
    checks = {
        "group_shape": list(group_read.shape) == [216, 2, 37, 3, 2560],
        "aggregate_shape": list(aggregate_read.shape) == [3, 3, 2, 37, 3, 2560],
        "finite": bool(np.isfinite(np.asarray(group_read)).all()) and bool(np.isfinite(np.asarray(aggregate_read)).all()),
        "all_counts": counts[0].tolist() == [72, 72, 72],
        "typed_counts": counts[1].tolist() == [39, 42, 47] and counts[2].tolist() == [33, 30, 25],
        "typed_partition": bool(np.all(counts[1] + counts[2] == counts[0])),
        "partition_candidate_balance": all(candidate_balance.values()),
        "state0_counterbalance": state0_all_max_abs <= 1e-6,
        "descriptive_scope": not meta["execution_identity_gate_passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1507,
        "campaign": "C087",
        "formula": "Delta_H(g,u)=H_same(g,u)-H_different(g,u)",
        "strata": STRATA,
        "partitions": partitions,
        "surfaces": surfaces,
        "roles": protocol["roles"],
        "axis_orders": {
            "group": ["group", "surface", "state", "role", "coordinate"],
            "aggregate": ["stratum", "partition", "surface", "state", "role", "coordinate"],
        },
        "counts": counts.tolist(),
        "candidate_balance_by_partition": candidate_balance,
        "state0_all_partition_mean_max_abs": state0_all_max_abs,
        "checks": checks,
        "evidence_scope": "descriptive atlas using Phase1505 frozen strata; not confirmatory because Phase1506 execution identity gate failed",
        "files": {
            "group": {"bytes": group_path.stat().st_size, "sha256": core.sha(group_path)},
            "group_index": {"sha256": core.sha(group_index_path)},
            "aggregate": {"bytes": aggregate_path.stat().st_size, "sha256": core.sha(aggregate_path)},
            "counts": {"sha256": core.sha(OUT / "atlas/partition_stratum_counts.int32.npy")},
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/semantic_contrast_atlas_summary.json", summary)
    core.save(OUT / "analysis/final.json", {
        "phase": 1507,
        "campaign": "C087",
        "status": "descriptive_semantic_contrast_atlas_complete",
        "authorization": "run_phase1508_c087_discovery_observation_freeze",
    })
    print(json.dumps({key: value for key, value in summary.items() if key != "counts"}, indent=2))


if __name__ == "__main__":
    main()
