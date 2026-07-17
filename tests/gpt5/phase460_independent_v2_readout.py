#!/usr/bin/env python3
"""Phase460 readout for Phase459 with Phase458 transform names."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase451_glm4_v2_pilot_behavior import load_jsonl, summarize_transform, wilson_bounds  # noqa: E402


GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase459_glm4_independent_v2_behavior" / "phase459_glm4_independent_v2_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase460_independent_v2_readout"
OUT_PATH = OUT_DIR / "phase460_independent_v2_readout.json"

CORE_TRANSFORMS = (
    "core_table_frame",
    "core_record_lines",
    "core_evidence_then_claim",
    "core_claim_reference_sync",
)
STRESS_TRANSFORMS = (
    "stress_claim_first",
    "stress_dense_records",
)


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
    transform_set = set(transforms)
    for row in records:
        grouped[row["sample_id"]].append(row)
    ok = 0
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


def track_summary(records: list[dict[str, Any]], transforms: tuple[str, ...]) -> dict[str, Any]:
    rows = [row for row in records if row["transform"] in transforms]
    summary = summarize(rows)
    cf = pair_consistency(rows)
    orbit = orbit_consistency(rows, transforms)
    return {
        "summary": summary,
        "by_transform": summarize_transform(rows),
        "counterfactual": cf,
        "orbit": orbit,
        "confirmed_s3_core_behavior_window": (
            summary["semantic_lcb_95"] >= 0.85
            and summary["other"] == 0
            and cf["consistent_lcb_95"] >= 0.85
            and orbit["orbit_lcb_95"] >= 0.80
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(GEN_PATH)
    out = {
        "schema_version": "phase460_independent_v2_readout.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "overall": summarize(rows),
        "core_track": track_summary(rows, CORE_TRANSFORMS),
        "stress_track": track_summary(rows, STRESS_TRANSFORMS),
        "authorization": {
            "physical_trace_authorized": False,
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
