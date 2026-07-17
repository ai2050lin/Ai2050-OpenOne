#!/usr/bin/env python3
"""Phase451 GLM4 v2 interface pilot behavior retest.

Runs GLM4 only on the Phase449 v2 knowledge samples. This is a pilot behavior
retest, not strict qualification and not physical trace collection.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase449_v2_interface_protocol" / "phase449_v2_glm4_knowledge_retest_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase451_glm4_v2_pilot_behavior"
GENERATIONS_PATH = OUT_DIR / "phase451_glm4_v2_pilot_generations.jsonl"
SUMMARY_PATH = OUT_DIR / "phase451_glm4_v2_pilot_summary.json"
Z_TWO_SIDED_95 = 1.96


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_generated(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s:：,，.;。!！?？\"'`]+", "", text)
    return text[:1].upper() if text else ""


def classify(expected: str, generated: str) -> str:
    token = normalize_generated(generated)
    if token == expected:
        return "semantic"
    if token in {"A", "B"}:
        return "wrong"
    return "other"


def prompt_for(text: str) -> str:
    return f"{text}\nAnswer:"


def wilson_bounds(k: int, n: int, z: float = Z_TWO_SIDED_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def generate_batch(model: Any, tokenizer: Any, device: torch.device, prompts: list[str], max_new_tokens: int) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    prompt_width = encoded["input_ids"].shape[1]
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    out = []
    for ids in output:
        new_ids = ids[prompt_width:]
        out.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return out


def eval_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for variant in sample["surface_variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_sample_id": sample["source_sample_id"],
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


def run_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        generated = generate_batch(model, tokenizer, device, [prompt_for(row["eval_text"]) for row in batch], max_new_tokens)
        for row, gen in zip(batch, generated, strict=True):
            out.append({
                "model": "glm4",
                "stage": "phase451_v2_pilot_behavior",
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "source_pair_id": row["source_pair_id"],
                "pair_index": row["pair_index"],
                "pair_role": row["pair_role"],
                "ability": row["ability"],
                "task": row["task"],
                "transform": row["transform"],
                "canonical_answer": row["canonical_answer"],
                "truth_value": row["truth_value"],
                "generated": gen,
                "normalized_generated": normalize_generated(gen),
                "classification": classify(row["canonical_answer"], gen),
            })
        log(f"glm4 phase451: {min(start + len(batch), len(rows))}/{len(rows)}")
    return out


def summarize_transform(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["transform"]].append(row)
    out = []
    for transform, rows in sorted(buckets.items()):
        counts = Counter(row["classification"] for row in rows)
        n = len(rows)
        semantic = counts["semantic"]
        lcb, ucb = wilson_bounds(semantic, n)
        other_lcb, other_ucb = wilson_bounds(counts["other"], n)
        out.append({
            "transform": transform,
            "n": n,
            "semantic": semantic,
            "wrong": counts["wrong"],
            "other": counts["other"],
            "semantic_rate": semantic / n if n else 0.0,
            "semantic_lcb_95": lcb,
            "semantic_ucb_95": ucb,
            "other_ucb_95": other_ucb,
            "output_distribution": dict(Counter(row["normalized_generated"] or "<empty>" for row in rows)),
        })
    return out


def summarize_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["transform"]][row["source_pair_id"]].append(row)
    out = []
    for transform, pairs in sorted(grouped.items()):
        ok = 0
        for rows in pairs.values():
            roles = {row["pair_role"]: row for row in rows}
            ok += int("base" in roles and "counterfactual" in roles and all(row["classification"] == "semantic" for row in roles.values()))
        n = len(pairs)
        lcb, ucb = wilson_bounds(ok, n)
        out.append({
            "transform": transform,
            "n_pairs": n,
            "consistent_pairs": ok,
            "consistent_rate": ok / n if n else 0.0,
            "consistent_lcb_95": lcb,
            "consistent_ucb_95": ucb,
        })
    return out


def summarize_orbit(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transforms = {row["transform"] for row in records}
    for row in records:
        grouped[row["sample_id"]].append(row)
    ok = 0
    for rows in grouped.values():
        seen = {row["transform"] for row in rows}
        ok += int(seen == transforms and all(row["classification"] == "semantic" for row in rows))
    n = len(grouped)
    lcb, ucb = wilson_bounds(ok, n)
    return {
        "n_samples": n,
        "transforms": sorted(transforms),
        "orbit_consistent_samples": ok,
        "orbit_consistency_rate": ok / n if n else 0.0,
        "orbit_lcb_95": lcb,
        "orbit_ucb_95": ucb,
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in records)
    n = len(records)
    semantic = counts["semantic"]
    lcb, ucb = wilson_bounds(semantic, n)
    return {
        "schema_version": "phase451_glm4_v2_pilot_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pilot_behavior_complete_no_physical_trace",
        "model": "glm4",
        "target": "knowledge_network/relation_truth_judgment",
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
        "by_transform": summarize_transform(records),
        "counterfactual_by_transform": summarize_pairs(records),
        "orbit": summarize_orbit(records),
        "authorization": {
            "physical_trace_authorized": False,
            "large_independent_retest_recommended": True,
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
