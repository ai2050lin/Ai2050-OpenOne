#!/usr/bin/env python3
"""
Phase 545: multi-seed sampling stability and cross-category natural closure.

This phase audits whether Phase544 sampling positives are stable across random
seeds, and whether natural generation closure extends beyond the vehicle-centered
artifact cluster.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import decompose, hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase532_multi_seed_controls import normalize  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows, readout_direction  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402


OUT_ROOT = Path("results/glm5_phase545_sampling_stability_cross_category")
DEFAULT_PAIRS = ["vehicle_clothing", "vehicle_tool", "fruit_vegetable", "animal_tool", "fruit_tool"]
DEFAULT_SCAFFOLDS = ["natural_qa", "definition", "sentence_completion"]
DEFAULT_MODES = ["top_p", "temperature"]
DEFAULT_CONDITIONS = ["baseline", "residual_parallel", "residual_full"]
EXTRA_FAMILY_TERMS = {
    "fruit": ["fruit", "fruits", "apple", "banana", "orange", "grape", "mango", "pear", "peach", "berry"],
    "vegetable": ["vegetable", "vegetables", "carrot", "potato", "onion", "lettuce", "broccoli", "spinach", "pepper"],
    "animal": ["animal", "animals", "creature", "creatures", "mammal", "dog", "cat", "horse", "cow", "lion", "tiger"],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
        common_unit = normalize(np.mean([normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
        common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
        common_full = (common_unit * common_norm).astype(np.float32)
        readout = readout_direction(W_U, tokenizer, pair)
        dec = decompose(common_full, readout)
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


def aggregate_seed_rows(rows: list[dict[str, Any]], checkpoints: list[int], max_new_tokens: int) -> dict[str, Any]:
    out = {
        "n_seeds": len(rows),
        "checkpoints": checkpoints,
        "hit_at_k_mean": {},
        "hit_at_k_std": {},
        "mean_first_target_rank": float(np.mean([r["mean_first_target_rank"] for r in rows])) if rows else 0.0,
        "std_first_target_rank": float(np.std([r["mean_first_target_rank"] for r in rows])) if rows else 0.0,
        "seed_rows": rows,
    }
    metrics = ["family_target", "family_competitor", "exact_target", "exact_competitor", "other_only"]
    for cp in checkpoints:
        key = str(min(cp, max_new_tokens))
        out["hit_at_k_mean"][key] = {}
        out["hit_at_k_std"][key] = {}
        for metric in metrics:
            vals = [float(r["hit_at_k"][key][metric]) for r in rows]
            out["hit_at_k_mean"][key][metric] = float(np.mean(vals)) if vals else 0.0
            out["hit_at_k_std"][key][metric] = float(np.std(vals)) if vals else 0.0
    final_key = str(max_new_tokens)
    out["hit_rates"] = out["hit_at_k_mean"][final_key]
    out["hit_stds"] = out["hit_at_k_std"][final_key]
    return out


def stability_score(base: dict[str, Any], row: dict[str, Any], k: int) -> float:
    key = str(k)
    gain = float(row["hit_at_k_mean"][key]["family_target"] - base["hit_at_k_mean"][key]["family_target"])
    std = float(row["hit_at_k_std"][key]["family_target"])
    return gain / (std + 1e-6)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(EXTRA_FAMILY_TERMS)
    pairs = parse_csv(args.pairs)
    scaffolds = parse_csv(args.scaffolds)
    modes = parse_csv(args.decode_modes)
    conditions = parse_csv(args.conditions)
    sample_seeds = parse_int_csv(args.sample_seeds)
    checkpoints = parse_int_csv(args.checkpoints)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = max(float(x) for x in args.alphas.split(",") if x.strip())
        component_seeds = parse_int_csv(args.component_seeds)
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: phase545 pairs={pairs}, windows={windows}, sample_seeds={sample_seeds}")

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
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for pair, by_scaffold in source_prompts.items():
                groups = p544.token_groups(tokenizer, pair)
                audit[win_name]["sources"][pair] = {}
                for scaffold, prompts in by_scaffold.items():
                    audit[win_name]["sources"][pair][scaffold] = {}
                    for mode in modes:
                        row = {}
                        for condition in conditions:
                            seed_rows = []
                            for seed in sample_seeds:
                                seed_rows.append(p544.decode_probe(
                                    model, tokenizer, device, layers, prompts,
                                    interventions_for(components_by_layer, pair, window, condition, alpha),
                                    groups, pair, mode, args.max_new_tokens, checkpoints,
                                    args.batch_size, args.max_length, seed,
                                    args.temperature, args.top_p, args.beam_width,
                                ))
                            row[condition] = aggregate_seed_rows(seed_rows, checkpoints, args.max_new_tokens)
                        audit[win_name]["sources"][pair][scaffold][mode] = row
                        base = row["baseline"]["hit_rates"]["family_target"]
                        best_cond = max([c for c in conditions if c != "baseline"], key=lambda c: row[c]["hit_rates"]["family_target"])
                        best = row[best_cond]["hit_rates"]["family_target"]
                        log(f"    {win_name} {pair} {scaffold} {mode}: base={base:.2f} best={best:.2f} {best_cond}")

        return {
            "phase": 545,
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
            "checkpoints": checkpoints,
            "alpha": alpha,
            "sample_seeds": sample_seeds,
            "component_seeds": component_seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "family_terms": p544.FAMILY_TERMS,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
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
    parser.add_argument("--component-seeds", default="11,23")
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--scaffolds", default=",".join(DEFAULT_SCAFFOLDS))
    parser.add_argument("--decode-modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--checkpoints", default="1,3,5,10,12")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase545_{args.model}_sampling_stability_cross_category.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
