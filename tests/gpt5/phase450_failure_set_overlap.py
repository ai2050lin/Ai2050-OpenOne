#!/usr/bin/env python3
"""Phase450 failure-set overlap audit for Phase446 GLM4 knowledge orbit.

No model run, no CUDA, no physical trace. This stage fixes the Phase447
localization metric by using failure unions, intersections, and core/stress
orbit splits.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_behavior" / "phase446_glm4_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase450_failure_set_overlap"
OUT_PATH = OUT_DIR / "phase450_failure_set_overlap.json"

TARGET_ABILITY = "knowledge_network"
TARGET_TASK = "relation_truth_judgment"
CORE_TRANSFORMS = ("lexical_rewrite", "distance_rewrite", "syntax_rewrite")
STRESS_TRANSFORMS = ("boundary_rewrite", "order_rewrite", "query_rewrite")
ALL_TRANSFORMS = CORE_TRANSFORMS + STRESS_TRANSFORMS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    out = {transform: set() for transform in ALL_TRANSFORMS}
    for row in rows:
        if row["classification"] != "semantic":
            out[row["transform"]].add(row["sample_id"])
    return out


def overlap_matrix(sets: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    return {
        left: {right: len(sets[left] & sets[right]) for right in ALL_TRANSFORMS}
        for left in ALL_TRANSFORMS
    }


def jaccard_matrix(sets: dict[str, set[str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for left in ALL_TRANSFORMS:
        out[left] = {}
        for right in ALL_TRANSFORMS:
            union = sets[left] | sets[right]
            out[left][right] = len(sets[left] & sets[right]) / len(union) if union else 1.0
    return out


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    ps = [count / total for count in counts if count]
    return -sum(p * math.log(p) for p in ps) / math.log(len(counts))


def orbit_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["transform"]] = row

    def count_consistent(transforms: tuple[str, ...]) -> int:
        return sum(
            int(
                all(transform in items for transform in transforms)
                and all(items[transform]["classification"] == "semantic" for transform in transforms)
            )
            for items in by_sample.values()
        )

    n = len(by_sample)
    core_ok = count_consistent(CORE_TRANSFORMS)
    stress_ok = count_consistent(STRESS_TRANSFORMS)
    all_ok = count_consistent(ALL_TRANSFORMS)
    return {
        "n_samples": n,
        "core_transforms": list(CORE_TRANSFORMS),
        "stress_transforms": list(STRESS_TRANSFORMS),
        "core_consistent": core_ok,
        "core_consistency_rate": core_ok / n if n else 0.0,
        "stress_consistent": stress_ok,
        "stress_consistency_rate": stress_ok / n if n else 0.0,
        "all_consistent": all_ok,
        "all_consistency_rate": all_ok / n if n else 0.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in load_jsonl(GEN_PATH)
        if row["stage"] == "counterfactual_orbit_holdout"
        and row["ability"] == TARGET_ABILITY
        and row["task"] == TARGET_TASK
        and row["transform"] in ALL_TRANSFORMS
    ]
    sets = failure_sets(rows)
    union = set().union(*sets.values())
    counts = {transform: len(sets[transform]) for transform in ALL_TRANSFORMS}
    local_cover = {
        transform: (len(sets[transform]) / len(union) if union else 0.0)
        for transform in ALL_TRANSFORMS
    }
    old_top_share = max(counts.values()) / sum(counts.values()) if sum(counts.values()) else 0.0
    union_top_cover = max(local_cover.values()) if local_cover else 0.0
    out = {
        "schema_version": "phase450_failure_set_overlap.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "target": "glm4/knowledge_network/relation_truth_judgment",
        "failure_counts": counts,
        "failure_union_size": len(union),
        "old_repeated_count_top_failure_share": old_top_share,
        "union_based_localization_cover": union_top_cover,
        "localization_cover_by_transform": local_cover,
        "failure_entropy": normalized_entropy(list(counts.values())),
        "failure_intersection_matrix": overlap_matrix(sets),
        "failure_jaccard_matrix": jaccard_matrix(sets),
        "orbit_split": orbit_split(rows),
        "interpretation": {
            "phase447_conclusion": "conservative_but_incomplete",
            "core_orbit_status": "core_track_reanalysis_required_before_v2_model_retest",
            "physical_trace_authorized": False,
            "model_rerun_authorized": True,
            "next_step": "glm4_phase449_v2_pilot_behavior_retest",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
