#!/usr/bin/env python3
"""
Phase 547: label-forbidden paraphrase gate split.

This phase tests whether category directions still improve non-exact semantic
paraphrases when exact category labels are forbidden in the prompt.
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
from phase546_semantic_quality_decomposition import GENERIC_TERMS, contains_term, exact_terms  # noqa: E402


OUT_ROOT = Path("results/glm5_phase547_label_forbidden_paraphrase_gate")
DEFAULT_PAIRS = ["vehicle_tool", "fruit_vegetable", "animal_tool", "fruit_tool"]
DEFAULT_SCAFFOLDS = ["forbidden_natural_qa", "forbidden_definition", "forbidden_sentence_completion"]
DEFAULT_MODES = ["top_p", "temperature"]
DEFAULT_CONDITIONS = ["baseline", "residual_parallel", "residual_full", "residual_perp", "readout"]
SYNONYM_TERMS = {
    "vehicle": [
        "transport", "transportation", "automobile", "car", "truck", "bus", "train", "boat",
        "airplane", "aircraft", "spacecraft", "watercraft", "vessel", "craft", "conveyance",
        "machine", "motorized", "propelled", "travel", "carry", "move people", "move through",
    ],
    "tool": [
        "instrument", "instruments", "device", "devices", "equipment", "implement", "implements",
        "utensil", "apparatus", "machine", "hammer", "wrench", "drill", "used to", "designed to",
        "helps", "repair", "build", "cut", "measure",
    ],
    "fruit": [
        "apple", "banana", "orange", "grape", "mango", "pear", "peach", "berry", "berries",
        "edible plant", "produce", "sweet", "juicy", "seed", "seeds", "fleshy", "food",
    ],
    "vegetable": [
        "carrot", "potato", "onion", "lettuce", "broccoli", "spinach", "pepper", "produce",
        "edible plant", "leafy", "root", "food", "grown", "cooked",
    ],
    "animal": [
        "creature", "creatures", "mammal", "mammals", "organism", "organisms", "living being",
        "living creature", "dog", "cat", "horse", "cow", "lion", "tiger", "bird", "fish",
        "moves", "breathes", "alive",
    ],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(contains_term(low, term) for term in terms)


def all_family_terms(label: str) -> list[str]:
    base = list(p544.FAMILY_TERMS.get(label, [label]))
    extra = SYNONYM_TERMS.get(label, [])
    seen, out = set(), []
    for term in base + extra:
        term = term.strip().lower()
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def synonym_terms(label: str) -> list[str]:
    exact = set(exact_terms(label))
    return [x for x in all_family_terms(label) if x not in exact]


def forbidden_prompt(scaffold: str, obj: str, pos_label: str, neg_label: str) -> str:
    cap = obj.capitalize()
    banned = f"{pos_label}, {pos_label}s, {neg_label}, {neg_label}s"
    if scaffold == "forbidden_natural_qa":
        return (
            f"What kind of thing is a {obj}? Answer naturally, but do not use these words: "
            f"{banned}. A {obj} is"
        )
    if scaffold == "forbidden_definition":
        return (
            f"Define {obj} without using these words: {banned}. "
            f"{cap} is best described as"
        )
    if scaffold == "forbidden_sentence_completion":
        return (
            f"Complete the sentence without using these words: {banned}. "
            f"A {obj} is commonly used as"
        )
    raise ValueError(f"unknown scaffold: {scaffold}")


def classify_suffix(suffix: str, target_label: str, competitor_label: str) -> dict[str, Any]:
    low = suffix.lower()
    alpha_chars = sum(ch.isalpha() for ch in low)
    target_label_hit = has_any(low, exact_terms(target_label))
    competitor_label_hit = has_any(low, exact_terms(competitor_label))
    target_synonym = has_any(low, synonym_terms(target_label))
    competitor_synonym = has_any(low, synonym_terms(competitor_label))
    generic = has_any(low, GENERIC_TERMS)
    degenerate = alpha_chars < 2
    any_forbidden = target_label_hit or competitor_label_hit

    if any_forbidden and target_synonym:
        quality = "synonym_with_label_violation"
    elif target_synonym and competitor_synonym:
        quality = "mixed_synonym"
    elif target_synonym:
        quality = "clean_synonym"
    elif any_forbidden:
        quality = "label_violation"
    elif competitor_synonym:
        quality = "wrong_synonym"
    elif generic:
        quality = "generic_only"
    elif degenerate:
        quality = "degenerate"
    else:
        quality = "other"

    return {
        "quality": quality,
        "target_label_violation": target_label_hit,
        "competitor_label_violation": competitor_label_hit,
        "any_label_violation": any_forbidden,
        "target_synonym": target_synonym,
        "competitor_synonym": competitor_synonym,
        "generic": generic,
        "degenerate": degenerate,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(records))
    counts = Counter(r["quality"] for r in records)
    clean_syn = counts["clean_synonym"] / n
    syn_with_label = counts["synonym_with_label_violation"] / n
    mixed_syn = counts["mixed_synonym"] / n
    label_violation = sum(1 for r in records if r["any_label_violation"]) / n
    target_label_violation = sum(1 for r in records if r["target_label_violation"]) / n
    competitor_label_violation = sum(1 for r in records if r["competitor_label_violation"]) / n
    synonym_any = sum(1 for r in records if r["target_synonym"]) / n
    wrong_synonym = sum(1 for r in records if r["competitor_synonym"] and not r["target_synonym"]) / n
    return {
        "n": len(records),
        "counts": dict(sorted(counts.items())),
        "rates": {k: float(v / n) for k, v in sorted(counts.items())},
        "clean_synonym_rate": float(clean_syn),
        "synonym_with_label_violation_rate": float(syn_with_label),
        "mixed_synonym_rate": float(mixed_syn),
        "target_synonym_any_rate": float(synonym_any),
        "any_label_violation_rate": float(label_violation),
        "target_label_violation_rate": float(target_label_violation),
        "competitor_label_violation_rate": float(competitor_label_violation),
        "wrong_synonym_rate": float(wrong_synonym),
        "generic_only_rate": float(counts["generic_only"] / n),
        "degenerate_rate": float(counts["degenerate"] / n),
        "clean_paraphrase_score": float(clean_syn - wrong_synonym - label_violation),
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
            "readout": readout,
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
            out[pair][scaffold] = [forbidden_prompt(scaffold, x, pos_label, neg_label) for x in objects]
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
        log(f"{args.model}: phase547 pairs={pairs}, windows={windows}, seeds={sample_seeds}")

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
        saved_samples = []
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
                                for rec in records:
                                    rec2 = {
                                        "window": win_name,
                                        "pair": pair,
                                        "scaffold": scaffold,
                                        "mode": mode,
                                        "condition": condition,
                                        "seed": seed,
                                        **rec,
                                    }
                                    all_records.append(rec2)
                            row = aggregate(all_records)
                            row["seed_aggregates"] = seed_aggs
                            audit[win_name]["sources"][pair][scaffold][mode][condition] = row
                            saved_samples.extend(all_records[: args.samples_per_row])
                        base = audit[win_name]["sources"][pair][scaffold][mode]["baseline"]
                        rp = audit[win_name]["sources"][pair][scaffold][mode].get("residual_parallel", base)
                        rf = audit[win_name]["sources"][pair][scaffold][mode].get("residual_full", base)
                        log(
                            f"    {win_name} {pair} {scaffold} {mode}: "
                            f"base clean={base['clean_synonym_rate']:.2f} label={base['any_label_violation_rate']:.2f}; "
                            f"rp clean={rp['clean_synonym_rate']:.2f} label={rp['any_label_violation_rate']:.2f}; "
                            f"full clean={rf['clean_synonym_rate']:.2f} label={rf['any_label_violation_rate']:.2f}"
                        )

        return {
            "phase": 547,
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
            "synonym_terms": SYNONYM_TERMS,
            "generic_terms": GENERIC_TERMS,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "sample_records": saved_samples[: args.max_saved_samples],
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
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=800)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase547_{args.model}_label_forbidden_paraphrase_gate.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
