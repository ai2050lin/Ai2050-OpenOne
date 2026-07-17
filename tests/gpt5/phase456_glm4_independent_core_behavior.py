#!/usr/bin/env python3
"""Phase456 GLM4 independent core-track behavior replicate.

Runs GLM4 on Phase455 independent samples. No physical trace collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import (  # noqa: E402
    load_jsonl,
    run_generation,
    summarize_pairs,
    summarize_transform,
    wilson_bounds,
    write_jsonl,
)


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase455_independent_core_protocol" / "phase455_independent_core_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase456_glm4_independent_core_behavior"
GENERATIONS_PATH = OUT_DIR / "phase456_glm4_independent_core_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase456_glm4_independent_core_summary.json"

CORE_TRANSFORMS = (
    "core_catalog_frame",
    "core_numbered_records",
    "core_evidence_claim",
    "core_question_sync",
)
STRESS_TRANSFORMS = (
    "stress_claim_first",
    "stress_compact_semicolon",
)


def eval_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for variant in sample["surface_variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_sample_id": sample.get("source_sample_id", sample["sample_id"]),
                "source_pair_id": sample["source_pair_id"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "ability": sample["ability"],
                "task": sample["task"],
                "canonical_answer": sample["canonical_answer"],
                "truth_value": sample["truth_value"],
                "transform": variant["transform"],
                "semantic_hash": variant["semantic_hash"],
                "eval_text": variant["text"],
            })
    return rows


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


def track_summary(records: list[dict[str, Any]], transforms: tuple[str, ...]) -> dict[str, Any]:
    track_rows = [row for row in records if row["transform"] in transforms]
    return {
        "summary": summarize(track_rows),
        "by_transform": summarize_transform(track_rows),
        "counterfactual": pair_consistency(track_rows),
        "orbit": orbit_consistency(track_rows, transforms),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    core = track_summary(records, CORE_TRANSFORMS)
    stress = track_summary(records, STRESS_TRANSFORMS)
    core_pass = (
        core["summary"]["semantic_lcb_95"] >= 0.85
        and core["summary"]["other"] == 0
        and core["counterfactual"]["consistent_lcb_95"] >= 0.85
        and core["orbit"]["orbit_lcb_95"] >= 0.80
    )
    return {
        "schema_version": "phase456_glm4_independent_core_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "independent_behavior_complete_no_physical_trace",
        "model": "glm4",
        "target": "knowledge_network/independent_marker_truth",
        "pre_registered_tracks": True,
        "uses_phase446_generator": False,
        "strict_qualification_claimed": False,
        "physical_collection_performed": False,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "overall": summarize(records),
        "core_track": {
            **core,
            "confirmed_s3_core_behavior_window": core_pass,
        },
        "stress_track": stress,
        "authorization": {
            "physical_trace_authorized": False,
            "next_step": "phase457_pre_physical_split_freeze_if_core_confirmed",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = eval_rows(load_jsonl(SAMPLES_PATH))
    model = None
    try:
        model, tokenizer, device = load_model("glm4", use_8bit=True if args.use_8bit else None)
        records = run_generation(model, tokenizer, device, rows, args.batch_size, args.max_new_tokens)
        write_jsonl(GENERATIONS_PATH, records)
        SUMMARY_PATH.write_text(json.dumps(build_summary(records), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(SUMMARY_PATH)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
