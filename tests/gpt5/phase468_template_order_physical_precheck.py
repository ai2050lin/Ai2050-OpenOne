#!/usr/bin/env python3
"""Phase468 small diagnostic physical precheck for template order.

This is not a semantic mechanism atlas. It records scalar hidden-state summaries
only, for a small frozen contrast between successful templates and the
claim-first failure template.
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
from phase451_glm4_v2_pilot_behavior import load_jsonl, prompt_for, write_jsonl  # noqa: E402


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase465_template_factor_protocol" / "phase465_template_factor_samples.jsonl"
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase466_glm4_template_factor_behavior" / "phase466_glm4_template_factor_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase468_template_order_physical_precheck"
PROTOCOL_PATH = OUT_DIR / "phase468_template_order_physical_precheck_protocol.json"
ROWS_PATH = OUT_DIR / "phase468_template_order_physical_scalar_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase468_template_order_physical_precheck_summary.json"

SELECTED_TRANSFORMS = (
    "factor_plain_anchor",
    "factor_semicolon_only",
    "factor_claim_first_only",
)
MAX_PAIR_INDEX = 15


def build_eval_rows(samples: list[dict[str, Any]], generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gen_meta = {(row["sample_id"], row["transform"]): row for row in generations}
    rows = []
    for sample in samples:
        if sample["pair_index"] > MAX_PAIR_INDEX:
            continue
        for variant in sample["surface_variants"]:
            if variant["transform"] not in SELECTED_TRANSFORMS:
                continue
            gen = gen_meta[(sample["sample_id"], variant["transform"])]
            rows.append({
                "sample_id": sample["sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "pair_index": sample["pair_index"],
                "pair_role": sample["pair_role"],
                "transform": variant["transform"],
                "expected_label": sample["canonical_answer"],
                "classification": gen["classification"],
                "normalized_generated": gen["normalized_generated"],
                "target_position": sample["role_nodes"]["target_position"],
                "query_position": sample["role_nodes"]["query_position"],
                "eval_text": variant["text"],
            })
    return rows


def trace_scalar_rows(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    out = []
    for idx, row in enumerate(rows, start=1):
        encoded = tokenizer(prompt_for(row["eval_text"]), return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        last_index = int(encoded["attention_mask"][0].sum().item() - 1)
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for layer_index, hidden in enumerate(outputs.hidden_states):
            vec = hidden[0, last_index].detach().float()
            out.append({
                "model": "glm4",
                "phase": "phase468",
                "sample_id": row["sample_id"],
                "source_pair_id": row["source_pair_id"],
                "pair_index": row["pair_index"],
                "pair_role": row["pair_role"],
                "transform": row["transform"],
                "expected_label": row["expected_label"],
                "classification": row["classification"],
                "normalized_generated": row["normalized_generated"],
                "target_position": row["target_position"],
                "query_position": row["query_position"],
                "layer_index": layer_index,
                "last_token_l2": float(torch.linalg.vector_norm(vec).item()),
                "last_token_abs_mean": float(vec.abs().mean().item()),
                "last_token_mean": float(vec.mean().item()),
            })
        if idx % 12 == 0:
            print(f"[phase468] traced {idx}/{len(rows)} prompts", flush=True)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["transform"], row["expected_label"], row["classification"], row["layer_index"])].append(row)
    scalar_summary = []
    for key, items in sorted(buckets.items()):
        transform, expected_label, classification, layer_index = key
        n = len(items)
        scalar_summary.append({
            "transform": transform,
            "expected_label": expected_label,
            "classification": classification,
            "layer_index": layer_index,
            "n": n,
            "mean_l2": sum(item["last_token_l2"] for item in items) / n,
            "mean_abs_mean": sum(item["last_token_abs_mean"] for item in items) / n,
            "mean_signed_mean": sum(item["last_token_mean"] for item in items) / n,
        })
    behavior_counts = Counter((row["transform"], row["expected_label"], row["classification"]) for row in rows if row["layer_index"] == 0)
    return {
        "schema_version": "phase468_template_order_physical_precheck.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_physical_precheck_scalar_only",
        "model": "glm4",
        "physical_scope": "scalar hidden-state summaries only; no vectors, no head/neuron claims, no causal edges",
        "selected_transforms": list(SELECTED_TRANSFORMS),
        "max_pair_index": MAX_PAIR_INDEX,
        "prompt_count": sum(1 for row in rows if row["layer_index"] == 0),
        "trace_row_count": len(rows),
        "behavior_counts": {str(key): value for key, value in behavior_counts.items()},
        "scalar_summary": scalar_summary,
        "authorization": {
            "semantic_mechanism_atlas_authorized": False,
            "causal_or_neuron_claim_authorized": False,
            "next_step": "phase469_analyze_template_order_scalar_precheck",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(SAMPLES_PATH)
    generations = load_jsonl(GEN_PATH)
    rows = build_eval_rows(samples, generations)
    protocol = {
        "schema_version": "phase468_template_order_physical_precheck_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_small_diagnostic_physical_precheck",
        "model": "glm4",
        "selected_transforms": list(SELECTED_TRANSFORMS),
        "pair_index_range": [0, MAX_PAIR_INDEX],
        "prompt_count": len(rows),
        "stores_raw_vectors": False,
        "stores_attention_heads": False,
        "stores_neuron_channels": False,
        "semantic_mechanism_claim_authorized": False,
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = None
    try:
        model, tokenizer, device = load_model("glm4", use_8bit=True if args.use_8bit else None)
        trace_rows = trace_scalar_rows(model, tokenizer, device, rows)
        write_jsonl(ROWS_PATH, trace_rows)
        SUMMARY_PATH.write_text(json.dumps(summarize(trace_rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(SUMMARY_PATH)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
