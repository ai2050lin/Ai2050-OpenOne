#!/usr/bin/env python3
"""Phase453 GLM4 v2 large independent holdout behavior retest.

Runs GLM4 only on the Phase452 v2 knowledge holdout. This is still behavior
qualification, not physical trace collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import (  # noqa: E402
    eval_rows,
    load_jsonl,
    run_generation,
    summarize_orbit,
    summarize_pairs,
    summarize_transform,
    wilson_bounds,
    write_jsonl,
)


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase452_v2_large_holdout_protocol" / "phase452_v2_glm4_knowledge_large_holdout_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase453_glm4_v2_large_holdout_behavior"
GENERATIONS_PATH = OUT_DIR / "phase453_glm4_v2_large_holdout_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase453_glm4_v2_large_holdout_summary.json"


def build_summary(records: list[dict]) -> dict:
    counts = Counter(row["classification"] for row in records)
    n = len(records)
    semantic = counts["semantic"]
    lcb, ucb = wilson_bounds(semantic, n)
    orbit = summarize_orbit(records)
    transform_summary = summarize_transform(records)
    pair_summary = summarize_pairs(records)
    all_transform_lcb_min = min(row["semantic_lcb_95"] for row in transform_summary)
    all_pair_lcb_min = min(row["consistent_lcb_95"] for row in pair_summary)
    strict_behavior_like_pass = (
        lcb >= 0.85
        and counts["other"] == 0
        and all_transform_lcb_min >= 0.85
        and all_pair_lcb_min >= 0.85
        and orbit["orbit_lcb_95"] >= 0.80
    )
    return {
        "schema_version": "phase453_glm4_v2_large_holdout_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "large_holdout_behavior_complete_no_physical_trace",
        "model": "glm4",
        "target": "knowledge_network/relation_truth_judgment",
        "source_split": "physical_window_freeze",
        "strict_qualification_claimed": False,
        "physical_collection_performed": False,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "overall": {
            "n": n,
            "semantic": semantic,
            "wrong": counts["wrong"],
            "other": counts["other"],
            "semantic_rate": semantic / n if n else 0.0,
            "semantic_lcb_95": lcb,
            "semantic_ucb_95": ucb,
        },
        "by_transform": transform_summary,
        "counterfactual_by_transform": pair_summary,
        "orbit": orbit,
        "gate_readout": {
            "all_transform_semantic_lcb_min": all_transform_lcb_min,
            "all_counterfactual_lcb_min": all_pair_lcb_min,
            "strict_behavior_like_pass": strict_behavior_like_pass,
            "note": "This is a behavior gate readout only; it does not authorize physical tracing by itself.",
        },
        "authorization": {
            "physical_trace_authorized": False,
            "next_step": "independent_replicate_or_static_baseline_before_any_physical_trace",
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
