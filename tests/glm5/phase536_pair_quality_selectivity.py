#!/usr/bin/env python3
"""
Phase 536: category pair quality and selectivity factorization.

Purpose:
  Phase535 showed that fruit/nonfruit cumulative effects are real but not cleanly
  selective, and animal/vehicle did not replicate. This phase audits several
  category state-pairs before making more mechanism claims.
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


OUT_ROOT = Path("results/glm5_phase536_pair_quality_selectivity")
TEMPLATES = {
    "direct": "The category of {x} is",
    "belongs": "{cap} belongs to the category",
    "kind": "A {x} is a kind of",
}
CATEGORY_BANK = {
    "fruit": ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum", "cherry", "lemon", "kiwi", "melon", "apricot", "fig", "papaya", "guava", "lime", "coconut", "date", "berry", "nectarine", "tangerine", "persimmon", "pomegranate"],
    "tool": ["hammer", "saw", "wrench", "drill", "pliers", "screwdriver", "chisel", "rake", "shovel", "axe", "knife", "scissors", "tongs", "clamp", "level", "file", "mallet", "spade", "hoe", "ladder", "anvil", "vise", "awl", "plane"],
    "animal": ["dog", "cat", "horse", "cow", "sheep", "goat", "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "monkey", "zebra", "giraffe", "elephant", "mouse", "squirrel", "camel", "panda", "otter", "whale", "dolphin"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane", "boat", "ship", "taxi", "van", "scooter", "tram", "subway", "helicopter", "tractor", "rocket", "canoe", "ferry", "jeep", "ambulance", "cart", "sled", "wagon"],
    "furniture": ["chair", "table", "sofa", "bed", "desk", "cabinet", "shelf", "dresser", "stool", "bench", "couch", "wardrobe", "bookcase", "nightstand", "cupboard", "armchair", "ottoman", "crib", "drawer", "counter", "recliner", "sideboard", "futon", "hutch"],
    "clothing": ["shirt", "pants", "dress", "coat", "jacket", "skirt", "sweater", "sock", "shoe", "hat", "scarf", "glove", "belt", "tie", "shorts", "jeans", "blouse", "hoodie", "boot", "sandal", "vest", "robe", "uniform", "cap"],
    "vegetable": ["carrot", "potato", "onion", "lettuce", "cabbage", "broccoli", "spinach", "pepper", "tomato", "cucumber", "celery", "radish", "turnip", "beet", "pea", "bean", "corn", "zucchini", "pumpkin", "squash", "garlic", "leek", "okra", "eggplant"],
}
PAIR_SPECS = {
    "fruit_tool": ("fruit", "tool"),
    "animal_tool": ("animal", "tool"),
    "vehicle_furniture": ("vehicle", "furniture"),
    "clothing_tool": ("clothing", "tool"),
    "fruit_vegetable": ("fruit", "vegetable"),
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device, max_length: int):
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in batch.items()}


def cat_prompt(template: str, x: str) -> str:
    return TEMPLATES[template].format(x=x, cap=x.capitalize())


def task_name(pair: str, template: str) -> str:
    return f"{pair}_{template}"


def pair_from_task(task: str) -> str:
    return task.rsplit("_", 1)[0]


def pair_labels(pair: str) -> tuple[str, str]:
    return PAIR_SPECS[pair]


def pair_targets(pair: str) -> list[str]:
    pos, _neg = pair_labels(pair)
    return [pos, f" {pos}", pos.capitalize(), f" {pos}s", f"{pos}s"]


def pair_competitors(pair: str) -> list[str]:
    _pos, neg = pair_labels(pair)
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
    for pair, (pos_label, neg_label) in PAIR_SPECS.items():
        pos_items = CATEGORY_BANK[pos_label][:train_n]
        neg_items = CATEGORY_BANK[neg_label][:train_n]
        for template in TEMPLATES:
            out[task_name(pair, template)] = {
                "pair": pair,
                "template": template,
                "pos": [cat_prompt(template, x) for x in pos_items],
                "neg": [cat_prompt(template, x) for x in neg_items],
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
    raw = {"early": [peak - 4, peak - 2, peak], "center": [peak - 2, peak, peak + 2]}
    return {k: [x for x in v if 0 <= x < n_layers] for k, v in raw.items()}


def logits_with_interventions(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha) in interventions.items():
            prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    outs = []
    for start in range(0, len(prompts), batch_size):
        batch = encode_batch(tokenizer, prompts[start:start + batch_size], device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handles = []
        for layer_id, d_tensor in prepared.items():
            layer = layers[layer_id]
            layer_device = next(layer.parameters()).device
            d_local = d_tensor.to(layer_device)
            pos_t = pos.to(layer_device)

            def make_hook(d_vec: torch.Tensor, pos_vec: torch.Tensor):
                def hook(_module, _inp, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[torch.arange(hs.shape[0], device=hs.device), pos_vec.to(hs.device)] += d_vec.to(hs.dtype)
                        return (hs,) + output[1:]
                    hs = output.clone()
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_vec.to(hs.device)] += d_vec.to(hs.dtype)
                    return hs
                return hook

            handles.append(layer.register_forward_hook(make_hook(d_local, pos_t)))
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
    margins = []
    if t_ids:
        for row in logits:
            target_logit = float(np.max(row[t_ids]))
            ranks.append(float(1 + np.sum(row > target_logit)))
            c_ids = [i for i in competitor_ids if 0 <= i < row.shape[0]]
            comp = float(np.max(row[c_ids])) if c_ids else 0.0
            margins.append(target_logit - comp)
    return {**sc, "mean_target_rank": float(np.mean(ranks)) if ranks else 0.0, "mean_token_margin": float(np.mean(margins)) if margins else 0.0}


def transfer_gate(pair: str, rows_by_alpha: dict[str, Any], alphas: list[float]) -> dict[str, Any]:
    own_tasks = [task_name(pair, template) for template in TEMPLATES]
    rows = []
    for alpha in alphas:
        key = str(alpha)
        own_vals = [float(rows_by_alpha[key][task]["delta_margin"]) for task in own_tasks]
        off_vals = [
            abs(float(v["delta_margin"]))
            for task, v in rows_by_alpha[key].items()
            if pair_from_task(task) != pair
        ]
        rows.append({
            "alpha": alpha,
            "transfer_min": min(own_vals),
            "transfer_mean": float(np.mean(own_vals)),
            "own_deltas": {task: own_vals[i] for i, task in enumerate(own_tasks)},
            "off_pair_max_abs": max(off_vals) if off_vals else 0.0,
        })
    best = max(rows, key=lambda x: x["transfer_min"])
    return {
        "best_alpha": best["alpha"],
        "best_transfer_min": best["transfer_min"],
        "best_transfer_mean": best["transfer_mean"],
        "best_off_pair_max_abs": best["off_pair_max_abs"],
        "pair_specificity": abs(best["transfer_min"]) / (float(best["off_pair_max_abs"]) + 1e-8),
        "alpha_rows": rows,
    }


def build_layer_components(
    dirs: dict[str, np.ndarray],
    W_U: np.ndarray,
    tokenizer: Any,
    seeds: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for pair in PAIR_SPECS:
        names = [task_name(pair, template) for template in TEMPLATES]
        by_template = {name.rsplit("_", 1)[1]: dirs[name] for name in names}
        common_unit = normalize(np.mean([normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
        common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
        common_full = (common_unit * common_norm).astype(np.float32)
        readout = readout_direction(W_U, tokenizer, pair)
        comps = {
            "common": decompose(common_full, readout)["perp"],
            "direct": decompose(by_template["direct"], readout)["perp"],
            "shuffled": decompose(by_template["belongs"], readout)["perp"],
        }
        for seed in seeds:
            comps[f"random_{seed}"] = random_orthogonal(
                comps["common"].shape[0], [readout], float(np.linalg.norm(comps["common"])), seed=seed
            )
        comps["_cos"] = np.array([
            cos(by_template["direct"], by_template["belongs"]),
            cos(by_template["direct"], by_template["kind"]),
            cos(by_template["belongs"], by_template["kind"]),
        ], dtype=np.float32)
        out[pair] = comps
    return out


def interventions_for(
    comps_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]]:
    out = {}
    for layer_id in window:
        out[layer_id] = (comps_by_layer[str(layer_id)][pair][condition], alpha)
    return out


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    tasks: dict[str, list[str]],
    token_sets: dict[str, dict[str, list[int]]],
    baseline: dict[str, Any],
    interventions: dict[int, tuple[np.ndarray, float]],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    out = {}
    for task, prompts in tasks.items():
        logits = logits_with_interventions(model, tokenizer, device, layers, prompts, interventions, batch_size, max_length)
        sc = score_rank(logits, token_sets[task]["target"], token_sets[task]["competitor"])
        out[task] = {
            **sc,
            "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
            "delta_rank": float(sc["mean_target_rank"] - baseline[task]["mean_target_rank"]),
        }
    return out


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
        log(f"{args.model}: windows={windows}, pairs={list(PAIR_SPECS)}, alphas={alphas}")

        candidates = build_candidates(args.train_n)
        tasks = build_tasks(args.test_n)
        token_sets = {
            task: {
                "target": token_ids(tokenizer, pair_targets(pair_from_task(task))),
                "competitor": token_ids(tokenizer, pair_competitors(pair_from_task(task))),
            }
            for task in tasks
        }

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_interventions(model, tokenizer, device, layers, prompts, None, args.batch_size, args.max_length)
            baseline[task] = score_rank(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        pair_baseline = {}
        for pair in PAIR_SPECS:
            task_rows = [baseline[task_name(pair, t)] for t in TEMPLATES]
            pair_baseline[pair] = {
                "mean_margin": float(np.mean([r["target_margin"] for r in task_rows])),
                "min_margin": float(np.min([r["target_margin"] for r in task_rows])),
                "mean_rank": float(np.mean([r["mean_target_rank"] for r in task_rows])),
                "mean_top1": float(np.mean([r["target_top1_rate"] for r in task_rows])),
                "target_token_count": len(token_sets[task_name(pair, "direct")]["target"]),
                "competitor_token_count": len(token_sets[task_name(pair, "direct")]["competitor"]),
            }

        components_by_layer = {}
        layer_stats = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps = build_layer_components(dirs, W_U, tokenizer, seeds)
            components_by_layer[str(layer_id)] = comps
            layer_stats[str(layer_id)] = {
                pair: {
                    "cos_direct_belongs": float(comps[pair]["_cos"][0]),
                    "cos_direct_kind": float(comps[pair]["_cos"][1]),
                    "cos_belongs_kind": float(comps[pair]["_cos"][2]),
                }
                for pair in PAIR_SPECS
            }

        audit = {}
        for pair in PAIR_SPECS:
            audit[pair] = {}
            for win_name, window in windows.items():
                audit[pair][win_name] = {"window": window, "conditions": {}}
                for condition in ["common", "direct", "shuffled"]:
                    by_alpha = {}
                    for alpha in alphas:
                        by_alpha[str(alpha)] = run_condition(
                            model, tokenizer, device, layers, tasks, token_sets, baseline,
                            interventions_for(components_by_layer, pair, window, condition, alpha),
                            args.batch_size, args.max_length,
                        )
                    audit[pair][win_name]["conditions"][condition] = {
                        "transfer": transfer_gate(pair, by_alpha, alphas)
                    }
                random_transfers = {}
                for seed in seeds:
                    by_alpha = {}
                    for alpha in alphas:
                        by_alpha[str(alpha)] = run_condition(
                            model, tokenizer, device, layers, tasks, token_sets, baseline,
                            interventions_for(components_by_layer, pair, window, f"random_{seed}", alpha),
                            args.batch_size, args.max_length,
                        )
                    random_transfers[str(seed)] = transfer_gate(pair, by_alpha, alphas)
                audit[pair][win_name]["conditions"]["random"] = {
                    "max_transfer_min": float(max(t["best_transfer_min"] for t in random_transfers.values())),
                    "pass_like_count": int(sum(t["best_transfer_min"] > 0.25 and t["pair_specificity"] > 1.0 for t in random_transfers.values())),
                }
            best = max(
                ((win, audit[pair][win]["conditions"]["common"]["transfer"]) for win in windows),
                key=lambda x: x[1]["best_transfer_min"],
            )
            log(
                f"    {pair}: baseline_margin={pair_baseline[pair]['mean_margin']:+.3f} "
                f"rank={pair_baseline[pair]['mean_rank']:.1f} "
                f"best_common={best[0]} {best[1]['best_transfer_min']:+.3f}/spec{best[1]['pair_specificity']:.2f}"
            )

        return {
            "phase": 536,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "windows": windows,
            "all_layers": all_layers,
            "pairs": list(PAIR_SPECS),
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "random_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "pair_baseline": pair_baseline,
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
    parser.add_argument("--alphas", default="4,8")
    parser.add_argument("--random-seeds", default="11,23,37,41")
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
    out_path = out_dir / f"phase536_{args.model}_pair_quality_selectivity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
