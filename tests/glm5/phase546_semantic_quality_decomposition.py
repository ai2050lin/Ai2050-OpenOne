#!/usr/bin/env python3
"""
Phase 546: semantic quality and label-vs-paraphrase decomposition.

Phase545 showed multi-seed natural generation closure, especially in GLM4.
This phase decomposes generated suffixes into exact label, non-exact family
semantic hits, wrong-family hits, generic-only outputs, and degenerate outputs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows, readout_direction  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402


OUT_ROOT = Path("results/glm5_phase546_semantic_quality_decomposition")
DEFAULT_PAIRS = ["vehicle_tool", "fruit_vegetable", "animal_tool", "fruit_tool"]
DEFAULT_SCAFFOLDS = ["natural_qa", "definition", "sentence_completion"]
DEFAULT_MODES = ["top_p", "temperature"]
DEFAULT_CONDITIONS = ["baseline", "residual_parallel", "residual_full"]
GENERIC_TERMS = [
    "object", "objects", "thing", "things", "item", "items", "category",
    "type", "kind", "entity", "stuff", "product", "products",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def contains_term(text: str, term: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    if re.search(r"[a-z0-9]", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(contains_term(low, term) for term in terms)


def exact_terms(label: str) -> list[str]:
    return [label, f"{label}s"]


def clean_family_terms(label: str) -> list[str]:
    exact = set(exact_terms(label))
    return [x for x in p544.FAMILY_TERMS.get(label, [label]) if x not in exact]


def classify_suffix(suffix: str, target_label: str, competitor_label: str) -> dict[str, Any]:
    low = suffix.lower()
    alpha_chars = sum(ch.isalpha() for ch in low)
    target_exact = has_any(low, exact_terms(target_label))
    competitor_exact = has_any(low, exact_terms(competitor_label))
    target_family = has_any(low, clean_family_terms(target_label))
    competitor_family = has_any(low, clean_family_terms(competitor_label))
    generic = has_any(low, GENERIC_TERMS)
    degenerate = alpha_chars < 2

    if target_exact:
        quality = "exact_label"
    elif target_family and (competitor_exact or competitor_family):
        quality = "mixed_family"
    elif target_family:
        quality = "family_non_exact"
    elif competitor_exact or competitor_family:
        quality = "wrong_family"
    elif generic:
        quality = "generic_only"
    elif degenerate:
        quality = "degenerate"
    else:
        quality = "other"

    return {
        "quality": quality,
        "target_exact": target_exact,
        "target_family_non_exact": target_family,
        "competitor_exact": competitor_exact,
        "competitor_family_non_exact": competitor_family,
        "generic": generic,
        "degenerate": degenerate,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(records))
    counts = Counter(r["quality"] for r in records)
    rates = {k: float(v / n) for k, v in sorted(counts.items())}
    exact = counts["exact_label"] / n
    non_exact = counts["family_non_exact"] / n
    mixed = counts["mixed_family"] / n
    semantic_ok = exact + non_exact
    family_related = semantic_ok + mixed
    return {
        "n": len(records),
        "counts": dict(sorted(counts.items())),
        "rates": rates,
        "exact_label_rate": float(exact),
        "family_non_exact_rate": float(non_exact),
        "mixed_family_rate": float(mixed),
        "semantic_ok_rate": float(semantic_ok),
        "family_related_rate": float(family_related),
        "wrong_family_rate": float(counts["wrong_family"] / n),
        "generic_only_rate": float(counts["generic_only"] / n),
        "degenerate_rate": float(counts["degenerate"] / n),
        "label_share_of_semantic_ok": float(exact / semantic_ok) if semantic_ok > 0 else 0.0,
        "paraphrase_share_of_semantic_ok": float(non_exact / semantic_ok) if semantic_ok > 0 else 0.0,
        "sample_records": records[:8],
    }


def build_candidates_for_pairs(pairs: list[str], train_n: int) -> dict[str, dict[str, Any]]:
    out = {}
    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        for template in TEMPLATES:
            name = f"{pair}_{template}"
            out[name] = {
                "pair": pair,
                "template": template,
                "pos": [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][:train_n]],
                "neg": [cat_prompt(template, x) for x in CATEGORY_BANK[neg_label][:train_n]],
            }
    return out


def build_components_for_pairs(
    pairs: list[str],
    dirs: dict[str, np.ndarray],
    W_U: np.ndarray,
    tokenizer: Any,
) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for pair in pairs:
        by_template = {template: dirs[f"{pair}_{template}"] for template in TEMPLATES}
        common_unit = p545.normalize(np.mean([p545.normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
        common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
        common_full = (common_unit * common_norm).astype(np.float32)
        readout = readout_direction(W_U, tokenizer, pair)
        dec = p545.decompose(common_full, readout)
        out[pair] = {
            "residual_full": common_full,
            "residual_perp": dec["perp"],
            "residual_parallel": dec["parallel"],
            "_readout": readout,
            "_common_full": common_full,
        }
    return out


def build_prompts_for_pairs(pairs: list[str], test_n: int, scaffolds: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        objects = CATEGORY_BANK[pos_label][-test_n:]
        out[pair] = {}
        for scaffold in scaffolds:
            out[pair][scaffold] = [p544.scaffold_prompt(scaffold, x, pos_label, neg_label) for x in objects]
    return out


def interventions_for(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]] | None:
    if condition == "baseline":
        return None
    return {layer_id: (components_by_layer[str(layer_id)][source_pair][condition], alpha) for layer_id in window}


def decode_and_classify(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = p544.prepare_interventions(interventions)
    rng = np.random.default_rng(seed)
    generated, suffixes, first_types, target_ranks, competitor_ranks = p544.run_linear_decode(
        model, tokenizer, device, layers, prompts, prepared, groups, mode,
        max_new_tokens, batch_size, max_length, rng, temperature, top_p
    )
    pos_label, neg_label = PAIR_SPECS[pair]
    records = []
    for i, (prompt, suffix, ids, first_type, target_rank, competitor_rank) in enumerate(
        zip(prompts, suffixes, generated, first_types, target_ranks, competitor_ranks)
    ):
        cls = classify_suffix(suffix, pos_label, neg_label)
        records.append({
            "prompt_index": i,
            "prompt": prompt,
            "generated_suffix": suffix,
            "generated_ids": ids,
            "first_type": first_type,
            "first_target_rank": float(target_rank),
            "first_competitor_rank": float(competitor_rank),
            **cls,
        })
    return aggregate(records), records


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pairs = parse_csv(args.pairs)
    scaffolds = parse_csv(args.scaffolds)
    modes = parse_csv(args.decode_modes)
    conditions = parse_csv(args.conditions)
    sample_seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = max(float(x) for x in args.alphas.split(",") if x.strip())
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: phase546 pairs={pairs}, windows={windows}, seeds={sample_seeds}")

        candidates = build_candidates_for_pairs(pairs, args.train_n)
        source_prompts = build_prompts_for_pairs(pairs, args.test_n, scaffolds)

        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            components_by_layer[str(layer_id)] = build_components_for_pairs(pairs, dirs, W_U, tokenizer)

        audit = {}
        samples_path_records = []
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for pair, by_scaffold in source_prompts.items():
                groups = p544.token_groups(tokenizer, pair)
                audit[win_name]["sources"][pair] = {}
                for scaffold, prompts in by_scaffold.items():
                    audit[win_name]["sources"][pair][scaffold] = {}
                    for mode in modes:
                        audit[win_name]["sources"][pair][scaffold][mode] = {}
                        for condition in conditions:
                            seed_aggs = []
                            all_records = []
                            for seed in sample_seeds:
                                agg, records = decode_and_classify(
                                    model, tokenizer, device, layers, prompts,
                                    interventions_for(components_by_layer, pair, window, condition, alpha),
                                    groups, pair, mode, args.max_new_tokens, args.batch_size,
                                    args.max_length, seed, args.temperature, args.top_p,
                                )
                                seed_aggs.append({"seed": seed, **agg})
                                for r in records:
                                    r2 = {
                                        "window": win_name,
                                        "pair": pair,
                                        "scaffold": scaffold,
                                        "mode": mode,
                                        "condition": condition,
                                        "seed": seed,
                                        **r,
                                    }
                                    all_records.append(r2)
                            row = aggregate(all_records)
                            row["seed_aggregates"] = seed_aggs
                            audit[win_name]["sources"][pair][scaffold][mode][condition] = row
                            samples_path_records.extend(all_records[: args.samples_per_row])
                        base = audit[win_name]["sources"][pair][scaffold][mode]["baseline"]
                        rp = audit[win_name]["sources"][pair][scaffold][mode].get("residual_parallel", base)
                        log(
                            f"    {win_name} {pair} {scaffold} {mode}: "
                            f"base sem={base['semantic_ok_rate']:.2f} rp sem={rp['semantic_ok_rate']:.2f} "
                            f"rp exact={rp['exact_label_rate']:.2f} rp para={rp['family_non_exact_rate']:.2f}"
                        )

        return {
            "phase": 546,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pairs": pairs,
            "conditions": conditions,
            "scaffolds": scaffolds,
            "decode_modes": modes,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "max_new_tokens": args.max_new_tokens,
            "alpha": alpha,
            "sample_seeds": sample_seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "family_terms": p544.FAMILY_TERMS,
            "generic_terms": GENERIC_TERMS,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "sample_records": samples_path_records[: args.max_saved_samples],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--alphas", default="6")
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--scaffolds", default=",".join(DEFAULT_SCAFFOLDS))
    parser.add_argument("--decode-modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=600)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase546_{args.model}_semantic_quality_decomposition.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
