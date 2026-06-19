#!/usr/bin/env python3
"""
Phase 539: interface cluster mechanism decomposition.

Phase538 found a vehicle-centered cluster. This phase tests whether the cluster
is better explained by hidden direction overlap, readout overlap, or by where
the perturbation is injected: residual stream, attention output, or MLP output.
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
from phase530_state_pair_decomposition import (  # noqa: E402
    PEAK_LAYERS,
    decompose,
    hidden_at_layer,
    load_model_bf16_flash,
    mean_dir,
    score_logits,
    token_ids,
)
from phase532_multi_seed_controls import normalize, random_orthogonal, cos  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt, encode_batch  # noqa: E402


OUT_ROOT = Path("results/glm5_phase539_interface_cluster_mechanism")
CORE_SOURCES = ["vehicle_furniture", "vehicle_tool", "vehicle_clothing"]
PAIR_SPECS = {
    "vehicle_furniture": ("vehicle", "furniture"),
    "vehicle_tool": ("vehicle", "tool"),
    "vehicle_clothing": ("vehicle", "clothing"),
    "clothing_tool": ("clothing", "tool"),
    "furniture_clothing": ("furniture", "clothing"),
    "furniture_tool": ("furniture", "tool"),
    "fruit_tool": ("fruit", "tool"),
    "animal_tool": ("animal", "tool"),
    "fruit_vegetable": ("fruit", "vegetable"),
    "animal_furniture": ("animal", "furniture"),
}
CONDITIONS = [
    "residual_full",
    "residual_perp",
    "residual_parallel",
    "attention_perp",
    "mlp_perp",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_name(pair: str, template: str) -> str:
    return f"{pair}_{template}"


def pair_from_task(task: str) -> str:
    return task.rsplit("_", 1)[0]


def pair_targets(pair: str) -> list[str]:
    pos, _neg = PAIR_SPECS[pair]
    return [pos, f" {pos}", pos.capitalize(), f" {pos}s", f"{pos}s"]


def pair_competitors(pair: str) -> list[str]:
    _pos, neg = PAIR_SPECS[pair]
    extras = ["object", " thing", " item"]
    return [neg, f" {neg}", neg.capitalize(), f" {neg}s", f"{neg}s"] + extras


def readout_direction(W_U: np.ndarray, tokenizer: Any, pair: str) -> np.ndarray:
    t_ids = token_ids(tokenizer, pair_targets(pair))
    c_ids = token_ids(tokenizer, pair_competitors(pair))
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def build_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    out = {}
    for pair in CORE_SOURCES:
        pos_label, neg_label = PAIR_SPECS[pair]
        for template in TEMPLATES:
            out[task_name(pair, template)] = {
                "pair": pair,
                "template": template,
                "pos": [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][:train_n]],
                "neg": [cat_prompt(template, x) for x in CATEGORY_BANK[neg_label][:train_n]],
            }
    return out


def build_tasks(test_n: int) -> dict[str, list[str]]:
    out = {}
    for pair, (pos_label, _neg_label) in PAIR_SPECS.items():
        for template in TEMPLATES:
            out[task_name(pair, template)] = [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][-test_n:]]
    return out


def layer_windows(model: str, n_layers: int, spec: str | None) -> dict[str, list[int]]:
    if spec:
        out = {}
        for chunk in spec.split(";"):
            vals = [int(x) for x in chunk.split(",") if x.strip()]
            out["-".join(map(str, vals))] = [x for x in vals if 0 <= x < n_layers]
        return out
    peak = PEAK_LAYERS[model]
    raw = {
        "center": [peak - 2, peak, peak + 2],
        "extended": [peak - 2, peak, peak + 2, peak + 4],
    }
    return {k: [x for x in v if 0 <= x < n_layers] for k, v in raw.items()}


def add_to_output(output: Any, pos: torch.Tensor, d_vec: torch.Tensor):
    if isinstance(output, tuple):
        hs = output[0].clone()
        hs[torch.arange(hs.shape[0], device=hs.device), pos.to(hs.device)] += d_vec.to(hs.device).to(hs.dtype)
        return (hs,) + output[1:]
    hs = output.clone()
    hs[torch.arange(hs.shape[0], device=hs.device), pos.to(hs.device)] += d_vec.to(hs.device).to(hs.dtype)
    return hs


def logits_with_patch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float, str]] | None,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha, site) in interventions.items():
            prepared[layer_id] = (torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16), site)
    outs = []
    for start in range(0, len(prompts), batch_size):
        batch = encode_batch(tokenizer, prompts[start:start + batch_size], device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handles = []
        for layer_id, (d_tensor, site) in prepared.items():
            layer = layers[layer_id]
            if site == "residual":
                module = layer
            elif site == "attention":
                module = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            elif site == "mlp":
                module = getattr(layer, "mlp", None)
            else:
                module = None
            if module is None:
                continue
            layer_device = next(module.parameters()).device
            d_local = d_tensor.to(layer_device)
            pos_local = pos.to(layer_device)

            def make_hook(d_vec: torch.Tensor, pos_vec: torch.Tensor):
                def hook(_module, _inp, output):
                    return add_to_output(output, pos_vec, d_vec)
                return hook

            handles.append(module.register_forward_hook(make_hook(d_local, pos_local)))
        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)
        for handle in handles:
            handle.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos.to(out.logits.device)]
        outs.append(logits.float().cpu().numpy().astype(np.float32))
        del out, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def score_rank(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, float]:
    sc = score_logits(logits, target_ids, competitor_ids)
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[1]]
    ranks = []
    if t_ids:
        for row in logits:
            target_logit = float(np.max(row[t_ids]))
            ranks.append(float(1 + np.sum(row > target_logit)))
    return {**sc, "mean_target_rank": float(np.mean(ranks)) if ranks else 0.0}


def run_all_tasks(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    tasks: dict[str, list[str]],
    token_sets: dict[str, dict[str, list[int]]],
    baseline: dict[str, Any] | None,
    interventions: dict[int, tuple[np.ndarray, float, str]] | None,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    out = {}
    for task, prompts in tasks.items():
        logits = logits_with_patch(model, tokenizer, device, layers, prompts, interventions, batch_size, max_length)
        sc = score_rank(logits, token_sets[task]["target"], token_sets[task]["competitor"])
        base_margin = float(baseline[task]["target_margin"]) if baseline is not None else float(sc["target_margin"])
        base_rank = float(baseline[task]["mean_target_rank"]) if baseline is not None else float(sc["mean_target_rank"])
        out[task] = {
            **sc,
            "delta_margin": float(sc["target_margin"] - base_margin),
            "delta_rank": float(sc["mean_target_rank"] - base_rank),
        }
    return out


def build_components(dirs: dict[str, np.ndarray], W_U: np.ndarray, tokenizer: Any, seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for pair in CORE_SOURCES:
        by_template = {template: dirs[task_name(pair, template)] for template in TEMPLATES}
        common_unit = normalize(np.mean([normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
        common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
        common_full = (common_unit * common_norm).astype(np.float32)
        readout = readout_direction(W_U, tokenizer, pair)
        dec = decompose(common_full, readout)
        comps = {
            "residual_full": common_full,
            "residual_perp": dec["perp"],
            "residual_parallel": dec["parallel"],
            "attention_perp": dec["perp"],
            "mlp_perp": dec["perp"],
            "_readout": readout,
            "_common_full": common_full,
            "_cos_templates": np.array([
                cos(by_template["direct"], by_template["belongs"]),
                cos(by_template["direct"], by_template["kind"]),
                cos(by_template["belongs"], by_template["kind"]),
            ], dtype=np.float32),
        }
        for seed in seeds:
            comps[f"random_{seed}"] = random_orthogonal(
                dec["perp"].shape[0], [readout], float(np.linalg.norm(dec["perp"])), seed=seed
            )
        out[pair] = comps
    return out


def patch_site(condition: str) -> str:
    if condition.startswith("attention"):
        return "attention"
    if condition.startswith("mlp"):
        return "mlp"
    return "residual"


def interventions_for(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float, str]]:
    site = patch_site(condition)
    return {layer_id: (components_by_layer[str(layer_id)][source_pair][condition], alpha, site) for layer_id in window}


def summarize_pair_response(rows: dict[str, Any], target_pair: str) -> dict[str, Any]:
    vals = [float(rows[task_name(target_pair, template)]["delta_margin"]) for template in TEMPLATES]
    return {
        "min_delta": float(min(vals)),
        "mean_delta": float(np.mean(vals)),
        "max_abs_delta": float(max(abs(v) for v in vals)),
        "template_deltas": {template: vals[i] for i, template in enumerate(TEMPLATES)},
    }


def condition_scan(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    tasks: dict[str, list[str]],
    token_sets: dict[str, dict[str, list[int]]],
    baseline: dict[str, Any],
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alphas: list[float],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    alpha_rows = {}
    for alpha in alphas:
        rows = run_all_tasks(
            model, tokenizer, device, layers, tasks, token_sets, baseline,
            interventions_for(components_by_layer, source_pair, window, condition, alpha),
            batch_size, max_length,
        )
        alpha_rows[str(alpha)] = {pair: summarize_pair_response(rows, pair) for pair in PAIR_SPECS}
    best_alpha = max(alpha_rows, key=lambda a: alpha_rows[a][source_pair]["min_delta"])
    best = alpha_rows[best_alpha]
    off_pairs = [p for p in PAIR_SPECS if p != source_pair]
    top_off = max(off_pairs, key=lambda p: float(best[p]["max_abs_delta"]))
    return {
        "best_alpha": float(best_alpha),
        "self_min_delta": float(best[source_pair]["min_delta"]),
        "self_mean_delta": float(best[source_pair]["mean_delta"]),
        "off_pair_max_abs": float(best[top_off]["max_abs_delta"]),
        "specificity": abs(float(best[source_pair]["min_delta"])) / (float(best[top_off]["max_abs_delta"]) + 1e-8),
        "top_off_pair": top_off,
        "matrix_at_best_alpha": best,
        "alpha_rows": alpha_rows,
    }


def cosine_matrices(components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]], window: list[int]) -> dict[str, Any]:
    pairs = CORE_SOURCES
    common = {p: [] for p in pairs}
    readout = {p: [] for p in pairs}
    for layer_id in window:
        comps = components_by_layer[str(layer_id)]
        for p in pairs:
            common[p].append(comps[p]["residual_perp"])
            readout[p].append(comps[p]["_readout"])
    common_mean = {p: np.mean(common[p], axis=0).astype(np.float32) for p in pairs}
    readout_mean = {p: np.mean(readout[p], axis=0).astype(np.float32) for p in pairs}
    return {
        "common_perp_cos": {p: {q: cos(common_mean[p], common_mean[q]) for q in pairs} for p in pairs},
        "readout_cos": {p: {q: cos(readout_mean[p], readout_mean[q]) for q in pairs} for p in pairs},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: core={CORE_SOURCES}, targets={len(PAIR_SPECS)}, windows={windows}")

        candidates = build_candidates(args.train_n)
        tasks = build_tasks(args.test_n)
        token_sets = {
            task: {
                "target": token_ids(tokenizer, pair_targets(pair_from_task(task))),
                "competitor": token_ids(tokenizer, pair_competitors(pair_from_task(task))),
            }
            for task in tasks
        }
        baseline = run_all_tasks(model, tokenizer, device, layers, tasks, token_sets, None, None, args.batch_size, args.max_length)
        for row in baseline.values():
            row["delta_margin"] = 0.0
            row["delta_rank"] = 0.0

        components_by_layer = {}
        layer_stats = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps = build_components(dirs, W_U, tokenizer, seeds)
            components_by_layer[str(layer_id)] = comps
            layer_stats[str(layer_id)] = {
                pair: {
                    "template_cos_direct_belongs": float(comps[pair]["_cos_templates"][0]),
                    "template_cos_direct_kind": float(comps[pair]["_cos_templates"][1]),
                    "template_cos_belongs_kind": float(comps[pair]["_cos_templates"][2]),
                    "full_norm": float(np.linalg.norm(comps[pair]["residual_full"])),
                    "perp_norm": float(np.linalg.norm(comps[pair]["residual_perp"])),
                    "parallel_norm": float(np.linalg.norm(comps[pair]["residual_parallel"])),
                }
                for pair in CORE_SOURCES
            }

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "cosine": cosine_matrices(components_by_layer, window), "sources": {}}
            for source_pair in CORE_SOURCES:
                source_row = {}
                for condition in CONDITIONS:
                    source_row[condition] = condition_scan(
                        model, tokenizer, device, layers, tasks, token_sets, baseline,
                        components_by_layer, source_pair, window, condition, alphas,
                        args.batch_size, args.max_length,
                    )
                random_rows = {}
                for seed in seeds:
                    random_rows[str(seed)] = condition_scan(
                        model, tokenizer, device, layers, tasks, token_sets, baseline,
                        components_by_layer, source_pair, window, f"random_{seed}", alphas,
                        args.batch_size, args.max_length,
                    )
                source_row["random"] = {
                    "by_seed": random_rows,
                    "max_self_min_delta": float(max(v["self_min_delta"] for v in random_rows.values())),
                    "max_off_pair_abs": float(max(v["off_pair_max_abs"] for v in random_rows.values())),
                    "pass_like_count": int(sum(v["self_min_delta"] > 0.25 and v["specificity"] > 1.0 for v in random_rows.values())),
                }
                audit[win_name]["sources"][source_pair] = source_row
                c = source_row["residual_perp"]
                log(
                    f"    {win_name} {source_pair}: residual_perp self={c['self_min_delta']:+.3f} "
                    f"off={c['off_pair_max_abs']:.3f} top={c['top_off_pair']} "
                    f"attn={source_row['attention_perp']['self_min_delta']:+.3f} "
                    f"mlp={source_row['mlp_perp']['self_min_delta']:+.3f}"
                )

        return {
            "phase": 539,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "pairs": list(PAIR_SPECS),
            "pair_specs": PAIR_SPECS,
            "templates": list(TEMPLATES),
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "random_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "baseline": baseline,
            "layer_stats": layer_stats,
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
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--alphas", default="2,4,6")
    parser.add_argument("--random-seeds", default="11,23")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase539_{args.model}_interface_cluster_mechanism.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
