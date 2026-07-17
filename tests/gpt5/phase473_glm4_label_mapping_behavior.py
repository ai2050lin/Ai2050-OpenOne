#!/usr/bin/env python3
"""Phase473 GLM4 label-mapping reversal behavior test."""

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


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase472_label_mapping_reversal_protocol" / "phase472_label_mapping_reversal_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase473_glm4_label_mapping_behavior"
GENERATIONS_PATH = OUT_DIR / "phase473_glm4_label_mapping_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase473_glm4_label_mapping_summary.json"
TRANSFORMS = ("order_evidence_first", "order_claim_first")


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


def enrich(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = {sample["sample_id"]: sample for sample in samples}
    out = []
    for row in records:
        sample = meta[row["sample_id"]]
        item = dict(row)
        item["label_mapping"] = sample["label_mapping"]
        item["expected_label"] = sample["canonical_answer"]
        item["truth_value"] = sample["truth_value"]
        item["target_position"] = sample["role_nodes"]["target_position"]
        item["query_position"] = sample["role_nodes"]["query_position"]
        out.append(item)
    return out


def grouped_summary(records: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, rows in sorted(buckets.items()):
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update(summarize(rows))
        out.append(item)
    return out


def mapping_flip_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["source_sample_id"], row["transform"])].append(row)
    ok = 0
    complete = 0
    for rows in grouped.values():
        by_mu = {row["label_mapping"]: row for row in rows}
        if set(by_mu) != {"mu_ab", "mu_ba"}:
            continue
        complete += 1
        ok += int(
            by_mu["mu_ab"]["classification"] == "semantic"
            and by_mu["mu_ba"]["classification"] == "semantic"
            and by_mu["mu_ab"]["normalized_generated"] != by_mu["mu_ba"]["normalized_generated"]
        )
    lcb, ucb = wilson_bounds(ok, complete)
    return {
        "n_mapping_pairs": complete,
        "both_correct_and_flipped": ok,
        "rate": ok / complete if complete else 0.0,
        "lcb_95": lcb,
        "ucb_95": ucb,
    }


def build_summary(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = enrich(records, samples)
    return {
        "schema_version": "phase473_glm4_label_mapping_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "label_mapping_behavior_complete_no_physical_trace",
        "model": "glm4",
        "target": "knowledge_network/label_mapping_reversal_truth",
        "physical_collection_performed": False,
        "cuda_used": torch.cuda.is_available(),
        "overall": summarize(enriched),
        "by_transform": summarize_transform(enriched),
        "by_mapping": grouped_summary(enriched, ["label_mapping"]),
        "by_transform_mapping": grouped_summary(enriched, ["transform", "label_mapping"]),
        "by_transform_mapping_truth": grouped_summary(enriched, ["transform", "label_mapping", "truth_value"]),
        "mapping_flip_consistency": mapping_flip_consistency(enriched),
        "authorization": {
            "physical_trace_authorized": False,
            "next_step": "phase474_label_mapping_failure_audit",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(SAMPLES_PATH)
    rows = eval_rows(samples)
    model = None
    try:
        model, tokenizer, device = load_model("glm4", use_8bit=True if args.use_8bit else None)
        records = run_generation(model, tokenizer, device, rows, args.batch_size, args.max_new_tokens)
        write_jsonl(GENERATIONS_PATH, records)
        SUMMARY_PATH.write_text(json.dumps(build_summary(records, samples), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(SUMMARY_PATH)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
