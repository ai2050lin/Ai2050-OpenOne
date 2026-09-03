#!/usr/bin/env python3
"""Phase1514: exact two-factor full-dimensional atlas for C088."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
CAPTURE = RESULT / "phase1513_c088_unified_forward_capture"
OUT = RESULT / "phase1514_c088_factorial_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

EFFECTS = ("semantic", "code", "semantic_code")


def weights(rows):
    return np.asarray([
        [row["semantic_sign"] / 2.0 for row in rows],
        [row["code_sign"] / 2.0 for row in rows],
        [row["semantic_sign"] * row["code_sign"] / 2.0 for row in rows],
    ], dtype=np.float32)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1514 exists")
    parent = core.load(CAPTURE / "analysis/final.json")
    parent_audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    capture_summary = core.load(CAPTURE / "analysis/unified_behavior_and_capture_summary.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1514_c088_factorial_field_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1513 authorization missing")
    if core.sha(CAPTURE / "raw/all_role_field.float16.npy") != capture_summary["files"]["field"]["sha256"]:
        raise RuntimeError("capture hash mismatch")
    field = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    lookup = {row["case_id"]: row for row in index}
    groups = core.rows(CAPTURE / "material/stratified_composition_sets.jsonl")
    surfaces = tuple(protocol["surfaces"])
    codebooks = tuple(protocol["codebooks"])
    partitions = tuple(protocol["partitions"])
    partition_index = {partition: i for i, partition in enumerate(partitions)}
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    group_path = OUT / "atlas/group_factorial_effect.float16.npy"
    group_effect = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.float16, shape=(248, 2, 3, 37, 4, 2560))
    partition_sums = np.zeros((4, 2, 3, 37, 4, 2560), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int32)
    group_index, logit_effects = [], []
    for gi, group in enumerate(groups):
        pi = partition_index[group["partition"]]
        counts[pi] += 1
        for ui, surface in enumerate(surfaces):
            rows = [lookup[group[f"{surface}_{codebook}_{semantic}"]] for codebook in codebooks for semantic in ("same", "different")]
            block = np.asarray(field[[row["row_index"] for row in rows]], dtype=np.float32)
            effect_block = np.tensordot(weights(rows), block, axes=(1, 0))
            group_effect[gi, ui] = effect_block.astype(np.float16)
            partition_sums[pi, ui] += effect_block
            margins = np.asarray([
                row["scores"][row["candidates"].index("yes")] - row["scores"][row["candidates"].index("no")]
                for row in rows
            ], dtype=np.float64)
            effect_values = weights(rows).astype(np.float64) @ margins
            logit_effects.append({
                "group_index": gi,
                "set_id": group["set_id"],
                "partition": group["partition"],
                "surface": surface,
                **{effect: float(effect_values[i]) for i, effect in enumerate(EFFECTS)},
            })
        group_index.append({
            "group_index": gi,
            "set_id": group["set_id"],
            "partition": group["partition"],
            "material_source": group["material_source"],
            "item": group["item"],
            "source_instance_id": group["source_instance_id"],
            "stratum": group["stratum"],
        })
    group_effect.flush()
    del group_effect
    group_index_path = OUT / "atlas/group_factorial_effect_index.jsonl"
    logit_path = OUT / "analysis/group_logit_factorial_effects.jsonl"
    core.write_rows(group_index_path, group_index)
    core.write_rows(logit_path, logit_effects)
    group_read = np.load(group_path, mmap_mode="r")
    aggregate_path = OUT / "atlas/partition_factorial_effect_mean.float32.npy"
    aggregate = np.lib.format.open_memmap(aggregate_path, mode="w+", dtype=np.float32, shape=(4, 2, 3, 37, 4, 2560))
    for pi in range(len(partitions)):
        aggregate[pi] = partition_sums[pi] / counts[pi]
    aggregate.flush()
    del aggregate
    np.save(OUT / "atlas/partition_counts.int32.npy", counts)
    aggregate_read = np.load(aggregate_path, mmap_mode="r")
    roles = tuple(protocol["roles"])
    source_role = roles.index("source_relation")
    candidate_role = roles.index("candidate_relation")
    semantic_index, code_index, interaction_index = range(3)
    checks = {
        "group_shape": list(group_read.shape) == [248, 2, 3, 37, 4, 2560],
        "aggregate_shape": list(aggregate_read.shape) == [4, 2, 3, 37, 4, 2560],
        "finite": all(bool(np.isfinite(np.asarray(group_read[start:start + 16])).all()) for start in range(0, 248, 16)) and bool(np.isfinite(np.asarray(aggregate_read)).all()),
        "counts": counts.tolist() == [72, 72, 72, 32],
        "all_mixed": all(row["stratum"] == "mixed" for row in group_index),
        "state0_semantic_counterbalance": float(np.max(np.abs(np.asarray(aggregate_read[:, :, semantic_index, 0])))) == 0.0,
        "state0_interaction_zero": float(np.max(np.abs(np.asarray(aggregate_read[:, :, interaction_index, 0])))) == 0.0,
        "source_causal_zero": float(np.max(np.abs(np.asarray(aggregate_read[:, :, :, :, source_role])))) == 0.0,
        "candidate_code_causal_zero": float(np.max(np.abs(np.asarray(aggregate_read[:, :, code_index:, :, candidate_role])))) == 0.0,
        "logit_rows": len(logit_effects) == 496,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1514,
        "campaign": "C088",
        "effects": EFFECTS,
        "formula": "C_S=(1/2)*sum_{R,P} product(sign_j) H(R,P)=2*beta_S",
        "aggregate_source": "float64 accumulation of pre-storage float32 effects; group atlas is float16 archival data",
        "conditional_semantic": "D_standard=C_semantic+C_semantic_code; D_reversed=C_semantic-C_semantic_code",
        "axis_orders": {
            "group": ["group", "surface", "effect", "state", "role", "coordinate"],
            "aggregate": ["partition", "surface", "effect", "state", "role", "coordinate"],
        },
        "partitions": partitions,
        "surfaces": surfaces,
        "roles": roles,
        "counts": counts.tolist(),
        "checks": checks,
        "files": {
            "group": {"bytes": group_path.stat().st_size, "sha256": core.sha(group_path)},
            "group_index": {"sha256": core.sha(group_index_path)},
            "aggregate": {"bytes": aggregate_path.stat().st_size, "sha256": core.sha(aggregate_path)},
            "logit": {"sha256": core.sha(logit_path)},
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/factorial_field_atlas_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1514, "campaign": "C088", "status": "factorial_field_atlas_complete", "authorization": "run_phase1515_c088_discovery_observation_freeze"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
