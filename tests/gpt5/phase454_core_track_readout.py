#!/usr/bin/env python3
"""Phase454 core-track readout for Phase453 GLM4 v2 large holdout.

No model run, no CUDA, no physical trace. Reclassifies v2_strong_statement_control
as a template-stress control and reads out the core interface track separately.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase451_glm4_v2_pilot_behavior import wilson_bounds


ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase453_glm4_v2_large_holdout_behavior" / "phase453_glm4_v2_large_holdout_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase454_core_track_readout"
OUT_PATH = OUT_DIR / "phase454_core_track_readout.json"

CORE_TRANSFORMS = (
    "v2_boundary_bullets",
    "v2_lexical_frame",
    "v2_local_order_control",
    "v2_query_claim_sync",
)
TEMPLATE_STRESS_TRANSFORMS = ("v2_strong_statement_control",)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in records)
    n = len(records)
    semantic = counts["semantic"]
    lcb, ucb = wilson_bounds(semantic, n)
    return {
        "n": n,
        "semantic": semantic,
        "wrong": counts["wrong"],
        "other": counts["other"],
        "semantic_rate": semantic / n if n else 0.0,
        "semantic_lcb_95": lcb,
        "semantic_ucb_95": ucb,
        "output_distribution": dict(Counter(row["normalized_generated"] or "<empty>" for row in records)),
    }


def pair_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[f"{row['transform']}::{row['source_pair_id']}"].append(row)
    ok = 0
    for rows in grouped.values():
        roles = {row["pair_role"]: row for row in rows}
        ok += int("base" in roles and "counterfactual" in roles and all(row["classification"] == "semantic" for row in roles.values()))
    n = len(grouped)
    lcb, ucb = wilson_bounds(ok, n)
    return {
        "n_transform_pairs": n,
        "consistent_transform_pairs": ok,
        "consistent_rate": ok / n if n else 0.0,
        "consistent_lcb_95": lcb,
        "consistent_ucb_95": ucb,
    }


def orbit_consistency(records: list[dict[str, Any]], transforms: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["sample_id"]].append(row)
    ok = 0
    transform_set = set(transforms)
    for rows in grouped.values():
        seen = {row["transform"] for row in rows}
        ok += int(seen == transform_set and all(row["classification"] == "semantic" for row in rows))
    n = len(grouped)
    lcb, ucb = wilson_bounds(ok, n)
    return {
        "n_samples": n,
        "transforms": list(transforms),
        "orbit_consistent_samples": ok,
        "orbit_consistency_rate": ok / n if n else 0.0,
        "orbit_lcb_95": lcb,
        "orbit_ucb_95": ucb,
    }


def by_transform(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["transform"]].append(row)
    return [{"transform": transform, **summarize(rows)} for transform, rows in sorted(buckets.items())]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(GEN_PATH)
    core_rows = [row for row in rows if row["transform"] in CORE_TRANSFORMS]
    stress_rows = [row for row in rows if row["transform"] in TEMPLATE_STRESS_TRANSFORMS]
    core_summary = summarize(core_rows)
    core_pairs = pair_consistency(core_rows)
    core_orbit = orbit_consistency(core_rows, CORE_TRANSFORMS)
    stress_summary = summarize(stress_rows)
    stress_pairs = pair_consistency(stress_rows)
    stress_orbit = orbit_consistency(stress_rows, TEMPLATE_STRESS_TRANSFORMS)
    out = {
        "schema_version": "phase454_core_track_readout.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "target": "glm4/knowledge_network/relation_truth_judgment",
        "core_track": {
            "summary": core_summary,
            "by_transform": by_transform(core_rows),
            "counterfactual": core_pairs,
            "orbit": core_orbit,
            "behavior_stable_window_candidate": (
                core_summary["semantic_lcb_95"] >= 0.85
                and core_summary["other"] == 0
                and core_pairs["consistent_lcb_95"] >= 0.85
                and core_orbit["orbit_lcb_95"] >= 0.80
            ),
        },
        "template_stress_track": {
            "summary": stress_summary,
            "by_transform": by_transform(stress_rows),
            "counterfactual": stress_pairs,
            "orbit": stress_orbit,
            "stress_failure": stress_pairs["consistent_lcb_95"] < 0.85,
        },
        "authorization": {
            "physical_trace_authorized": False,
            "reason": "Core behavior candidate requires independent replication and baseline audits before physical tracing.",
            "next_step": "phase455_independent_core_replicate_or_baseline_audit",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
