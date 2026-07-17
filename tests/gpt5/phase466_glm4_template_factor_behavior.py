#!/usr/bin/env python3
"""Phase466 GLM4 template-factor behavior replicate."""

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
from phase451_glm4_v2_pilot_behavior import load_jsonl, run_generation, summarize_transform, wilson_bounds, write_jsonl  # noqa: E402
from phase456_glm4_independent_core_behavior import eval_rows  # noqa: E402


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase465_template_factor_protocol" / "phase465_template_factor_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase466_glm4_template_factor_behavior"
GENERATIONS_PATH = OUT_DIR / "phase466_glm4_template_factor_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase466_glm4_template_factor_summary.json"

FACTOR_TRANSFORMS = (
    "factor_plain_anchor",
    "factor_numbered_only",
    "factor_evidence_label_only",
    "factor_claim_sync_only",
    "factor_semicolon_only",
    "factor_claim_first_only",
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


def orbit_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transform_set = set(FACTOR_TRANSFORMS)
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
        "transforms": list(FACTOR_TRANSFORMS),
        "orbit_consistent_samples": ok,
        "orbit_consistency_rate": ok / n if n else 0.0,
        "orbit_lcb_95": lcb,
        "orbit_ucb_95": ucb,
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summarize(records)
    cf = pair_consistency(records)
    orbit = orbit_consistency(records)
    return {
        "schema_version": "phase466_glm4_template_factor_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "template_factor_behavior_complete_no_physical_trace",
        "model": "glm4",
        "target": "knowledge_network/template_factor_bridge_marker_truth",
        "pre_registered_factor_transforms": list(FACTOR_TRANSFORMS),
        "strict_qualification_claimed": False,
        "physical_collection_performed": False,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "overall": overall,
        "by_transform": summarize_transform(records),
        "counterfactual": cf,
        "orbit": orbit,
        "confirmed_generator_independent_s3": False,
        "template_factor_behavior_window": (
            overall["semantic_lcb_95"] >= 0.85
            and overall["other"] == 0
            and cf["consistent_lcb_95"] >= 0.80
        ),
        "authorization": {
            "physical_trace_authorized": False,
            "next_step": "phase467_template_factor_failure_audit",
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
