#!/usr/bin/env python3
"""Phase470 position-resolved scalar precheck.

Vector-free diagnostic over frozen Phase465 prompts. Records scalar summaries
for evidence, claim and terminal roles. No heads, channels, neurons, causality
or sealed data.
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
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase470_position_resolved_scalar_precheck"
PROTOCOL_PATH = OUT_DIR / "phase470_position_resolved_scalar_protocol.json"
ROWS_PATH = OUT_DIR / "phase470_position_resolved_scalar_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase470_position_resolved_scalar_summary.json"

SELECTED_TRANSFORMS = (
    "factor_plain_anchor",
    "factor_semicolon_only",
    "factor_claim_first_only",
)
MAX_PAIR_INDEX = 15
ROLE_NAMES = ("evidence_span", "claim_span", "terminal_token")


def parse_role_sections(text: str) -> dict[str, str]:
    if text.startswith("Claim:"):
        claim = "Claim: " + text.split("Claim: ", 1)[1].split(". Records:", 1)[0] + "."
        evidence = "Records: " + text.split(". Records: ", 1)[1].split(" Reply ", 1)[0]
    else:
        evidence = "Records: " + text.split("Records: ", 1)[1].split(" Claim:", 1)[0]
        claim = "Claim: " + text.split(" Claim: ", 1)[1].split(" Reply ", 1)[0]
    return {"evidence_span": evidence, "claim_span": claim}


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle:
        return None
    limit = len(haystack) - len(needle) + 1
    for start in range(max(0, limit)):
        if haystack[start:start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def locate_role_positions(tokenizer: Any, prompt: str, text: str) -> dict[str, list[int]]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    roles: dict[str, list[int]] = {}
    for role, section in parse_role_sections(text).items():
        found = None
        for candidate in (section, " " + section, section + " "):
            ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            found = find_subsequence(full_ids, ids)
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"Could not locate {role}: {section[:80]}")
        roles[role] = list(range(found[0], found[1]))
    roles["terminal_token"] = [len(full_ids) - 1]
    return roles


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


def role_metrics(vecs: torch.Tensor) -> dict[str, float]:
    vecs = vecs.detach().float()
    mean_vec = vecs.mean(dim=0)
    token_l2 = torch.linalg.vector_norm(vecs, dim=1)
    return {
        "role_token_count": int(vecs.shape[0]),
        "mean_token_l2": float(token_l2.mean().item()),
        "mean_vector_l2": float(torch.linalg.vector_norm(mean_vec).item()),
        "mean_abs_mean": float(vecs.abs().mean().item()),
        "mean_signed_mean": float(vecs.mean().item()),
    }


def trace_rows(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    out = []
    locate_failures = []
    for idx, row in enumerate(rows, start=1):
        prompt = prompt_for(row["eval_text"])
        role_positions = locate_role_positions(tokenizer, prompt, row["eval_text"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False)
        if encoded["input_ids"].shape[1] > 1024:
            raise RuntimeError("Phase470 prompt unexpectedly truncated")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        seq_len = int(encoded["attention_mask"][0].sum().item())
        for role, positions in role_positions.items():
            if any(pos < 0 or pos >= seq_len for pos in positions):
                locate_failures.append({"sample_id": row["sample_id"], "transform": row["transform"], "role": role})
                continue
            for layer_index, hidden in enumerate(outputs.hidden_states):
                metrics = role_metrics(hidden[0, positions])
                out.append({
                    "model": "glm4",
                    "phase": "phase470",
                    "sample_id": row["sample_id"],
                    "source_pair_id": row["source_pair_id"],
                    "pair_index": row["pair_index"],
                    "pair_role": row["pair_role"],
                    "transform": row["transform"],
                    "role": role,
                    "expected_label": row["expected_label"],
                    "classification": row["classification"],
                    "normalized_generated": row["normalized_generated"],
                    "target_position": row["target_position"],
                    "query_position": row["query_position"],
                    "layer_index": layer_index,
                    **metrics,
                })
        if idx % 12 == 0:
            print(f"[phase470] traced {idx}/{len(rows)} prompts", flush=True)
    if locate_failures:
        raise RuntimeError(f"Phase470 role location failures: {locate_failures[:5]}")
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {(row["sample_id"], row["transform"]) for row in rows if row["layer_index"] == 0 and row["role"] == "terminal_token"}
    role_counts = Counter(row["role"] for row in rows if row["layer_index"] == 0)
    behavior_counts = Counter(
        (row["transform"], row["expected_label"], row["classification"])
        for row in rows
        if row["layer_index"] == 0 and row["role"] == "terminal_token"
    )
    role_token_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if row["layer_index"] == 0:
            role_token_counts[row["role"]][int(row["role_token_count"])] += 1
    return {
        "schema_version": "phase470_position_resolved_scalar_precheck.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "position_resolved_scalar_precheck_complete",
        "model": "glm4",
        "physical_scope": "role-position scalar summaries only; no vectors, no head/neuron claims, no causal edges",
        "selected_transforms": list(SELECTED_TRANSFORMS),
        "roles": list(ROLE_NAMES),
        "max_pair_index": MAX_PAIR_INDEX,
        "prompt_count": len(prompts),
        "trace_row_count": len(rows),
        "role_counts_at_layer0": dict(role_counts),
        "role_token_count_distribution": {role: dict(counts) for role, counts in role_token_counts.items()},
        "behavior_counts": {str(key): value for key, value in behavior_counts.items()},
        "authorization": {
            "semantic_mechanism_atlas_authorized": False,
            "causal_or_neuron_claim_authorized": False,
            "next_step": "phase471_position_resolved_paired_analysis",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_eval_rows(load_jsonl(SAMPLES_PATH), load_jsonl(GEN_PATH))
    protocol = {
        "schema_version": "phase470_position_resolved_scalar_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_position_resolved_scalar_precheck",
        "model": "glm4",
        "selected_transforms": list(SELECTED_TRANSFORMS),
        "roles": list(ROLE_NAMES),
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
        scalar_rows = trace_rows(model, tokenizer, device, rows)
        write_jsonl(ROWS_PATH, scalar_rows)
        SUMMARY_PATH.write_text(json.dumps(summarize(scalar_rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(SUMMARY_PATH)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
