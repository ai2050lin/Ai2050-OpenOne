#!/usr/bin/env python3
"""
Phase 535: multi-layer common direction control audit.

This phase audits Phase534's strongest new result: qwen3 multi-layer category
common cumulative transfer. It adds multi-layer random controls, direct-only
cumulative controls, template-shuffled controls, layer-window scan, and a
second category pair.
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


OUT_ROOT = Path("results/glm5_phase535_cumulative_audit")
TEMPLATES = {
    "direct": "The category of {x} is",
    "belongs": "{cap} belongs to the category",
    "kind": "A {x} is a kind of",
}
CATEGORY_PAIRS = {
    "fruit_nonfruit": {
        "pos": [
            "apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum",
            "cherry", "lemon", "kiwi", "melon", "apricot", "fig", "papaya", "guava",
            "lime", "coconut", "date", "berry", "nectarine", "tangerine", "persimmon", "pomegranate",
        ],
        "neg": [
            "car", "truck", "bus", "shirt", "table", "chair", "hammer", "river",
            "stone", "cloud", "violin", "window", "pencil", "bottle", "bridge", "planet",
            "shoe", "camera", "forest", "castle", "knife", "blanket", "lamp", "phone",
        ],
        "targets": ["fruit", " fruits", "Fruit"],
        "competitors": ["vehicle", " animal", " object", " tool", " clothing", " color"],
    },
    "animal_vehicle": {
        "pos": [
            "dog", "cat", "horse", "cow", "sheep", "goat", "lion", "tiger",
            "bear", "wolf", "fox", "deer", "rabbit", "monkey", "zebra", "giraffe",
            "elephant", "mouse", "squirrel", "camel", "panda", "otter", "whale", "dolphin",
        ],
        "neg": [
            "car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane", "boat",
            "ship", "taxi", "van", "scooter", "tram", "subway", "helicopter", "tractor",
            "rocket", "canoe", "ferry", "jeep", "ambulance", "cart", "sled", "wagon",
        ],
        "targets": ["animal", " animals", "Animal"],
        "competitors": ["vehicle", " vehicles", "fruit", " object", " tool", " clothing"],
    },
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


def readout_direction(W_U: np.ndarray, tokenizer: Any, pair: str) -> np.ndarray:
    spec = CATEGORY_PAIRS[pair]
    t_ids = token_ids(tokenizer, spec["targets"])
    c_ids = token_ids(tokenizer, spec["competitors"])
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def build_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pair, spec in CATEGORY_PAIRS.items():
        for template in TEMPLATES:
            name = task_name(pair, template)
            out[name] = {
                "pair": pair,
                "template": template,
                "pos": [cat_prompt(template, x) for x in spec["pos"][:train_n]],
                "neg": [cat_prompt(template, x) for x in spec["neg"][:train_n]],
            }
    return out


def build_tasks(test_n: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pair, spec in CATEGORY_PAIRS.items():
        for template in TEMPLATES:
            out[task_name(pair, template)] = [cat_prompt(template, x) for x in spec["pos"][-test_n:]]
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
        "early": [peak - 4, peak - 2, peak],
        "center": [peak - 2, peak, peak + 2],
        "late": [peak, peak + 2, peak + 4],
    }
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


def transfer_gate(pair: str, rows_by_alpha: dict[str, Any], alphas: list[float], min_abs_delta: float) -> dict[str, Any]:
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
    denom = float(best["off_pair_max_abs"]) + 1e-8
    return {
        "best_alpha": best["alpha"],
        "best_transfer_min": best["transfer_min"],
        "best_transfer_mean": best["transfer_mean"],
        "best_off_pair_max_abs": best["off_pair_max_abs"],
        "best_transfer_ratio": abs(best["transfer_min"]) / denom,
        "passes_transfer_gate": bool(best["transfer_min"] >= min_abs_delta and best["transfer_min"] > 1.5 * denom),
        "alpha_rows": rows,
    }


def rank_and_margin(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, float]:
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[0]]
    c_ids = [i for i in competitor_ids if 0 <= i < logits.shape[0]]
    if not t_ids:
        return {"target_rank": float(logits.shape[0]), "target_margin": 0.0}
    target_logit = float(np.max(logits[t_ids]))
    comp_logit = float(np.max(logits[c_ids])) if c_ids else 0.0
    return {"target_rank": float(1 + np.sum(logits > target_logit)), "target_margin": target_logit - comp_logit}


def generation_probe(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    pair: str,
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    spec = CATEGORY_PAIRS[pair]
    target_ids = token_ids(tokenizer, spec["targets"])
    competitor_ids = token_ids(tokenizer, spec["competitors"])
    target_set = set(target_ids)
    ranks, margins, hits, outputs = [], [], 0, []
    for prompt in prompts:
        text = prompt
        ids = []
        step_stats = []
        for _ in range(max_new_tokens):
            logits = logits_with_interventions(model, tokenizer, device, layers, [text], interventions, 1, max_length)[0]
            stats = rank_and_margin(logits, target_ids, competitor_ids)
            step_stats.append(stats)
            tok = int(np.argmax(logits))
            ids.append(tok)
            text += tokenizer.decode([tok], skip_special_tokens=False)
        if any(tok in target_set for tok in ids):
            hits += 1
        ranks.append(min(s["target_rank"] for s in step_stats))
        margins.append(step_stats[0]["target_margin"] if step_stats else 0.0)
        outputs.append({"prompt": prompt, "ids": ids, "generated_suffix": text[len(prompt):], "step_stats": step_stats})
    n = max(1, len(prompts))
    return {
        "n": len(prompts),
        "target_hit_rate": float(hits / n),
        "mean_best_target_rank": float(np.mean(ranks)) if ranks else 0.0,
        "mean_first_step_margin": float(np.mean(margins)) if margins else 0.0,
        "sample_outputs": outputs[: min(3, len(outputs))],
    }


def build_layer_components(
    directions: dict[str, np.ndarray],
    W_U: np.ndarray,
    tokenizer: Any,
    seeds: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in CATEGORY_PAIRS:
        names = [task_name(pair, template) for template in TEMPLATES]
        dirs = {name.rsplit("_", 1)[1]: directions[name] for name in names}
        common_unit = normalize(np.mean([normalize(dirs[t]) for t in TEMPLATES], axis=0).astype(np.float32))
        common_norm = float(np.mean([np.linalg.norm(dirs[t]) for t in TEMPLATES]))
        common_full = (common_unit * common_norm).astype(np.float32)
        readout = readout_direction(W_U, tokenizer, pair)
        common_perp = decompose(common_full, readout)["perp"]
        pair_comps = {
            "common_perp": common_perp,
            "direct_perp": decompose(dirs["direct"], readout)["perp"],
            "belongs_perp": decompose(dirs["belongs"], readout)["perp"],
            "kind_perp": decompose(dirs["kind"], readout)["perp"],
            "direct_residual": (dirs["direct"] - float(np.dot(dirs["direct"], common_unit)) * common_unit).astype(np.float32),
        }
        for seed in seeds:
            pair_comps[f"random_common_{seed}"] = random_orthogonal(
                common_perp.shape[0], [readout], float(np.linalg.norm(common_perp)), seed=seed
            )
        pair_comps["cos_direct_belongs"] = np.array([cos(dirs["direct"], dirs["belongs"])], dtype=np.float32)
        pair_comps["cos_direct_kind"] = np.array([cos(dirs["direct"], dirs["kind"])], dtype=np.float32)
        pair_comps["cos_belongs_kind"] = np.array([cos(dirs["belongs"], dirs["kind"])], dtype=np.float32)
        out[pair] = pair_comps
    return out


def condition_interventions(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    pair: str,
    window: list[int],
    condition: str,
    alpha: float,
    seed: int | None = None,
) -> dict[int, tuple[np.ndarray, float]]:
    interventions = {}
    shuffle = ["direct_perp", "kind_perp", "belongs_perp", "direct_perp"]
    for i, layer_id in enumerate(window):
        comps = components_by_layer[str(layer_id)][pair]
        if condition == "common":
            vec = comps["common_perp"]
        elif condition == "direct":
            vec = comps["direct_perp"]
        elif condition == "shuffled_template":
            vec = comps[shuffle[i % len(shuffle)]]
        elif condition == "random":
            assert seed is not None
            vec = comps[f"random_common_{seed}"]
        else:
            raise ValueError(condition)
        interventions[layer_id] = (vec, alpha)
    return interventions


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
    rows = {}
    for task, prompts in tasks.items():
        logits = logits_with_interventions(model, tokenizer, device, layers, prompts, interventions, batch_size, max_length)
        sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
        rows[task] = {
            **sc,
            "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
            "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
        }
    return rows


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
        log(f"{args.model}: windows={windows}, alphas={alphas}, seeds={seeds}")

        candidates = build_candidates(args.train_n)
        tasks = build_tasks(args.test_n)
        token_sets = {}
        for task in tasks:
            pair = pair_from_task(task)
            token_sets[task] = {
                "target": token_ids(tokenizer, CATEGORY_PAIRS[pair]["targets"]),
                "competitor": token_ids(tokenizer, CATEGORY_PAIRS[pair]["competitors"]),
            }

        directions_by_layer: dict[str, dict[str, np.ndarray]] = {}
        components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        layer_stats: dict[str, Any] = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            directions_by_layer[str(layer_id)] = dirs
            components_by_layer[str(layer_id)] = build_layer_components(dirs, W_U, tokenizer, seeds)
            layer_stats[str(layer_id)] = {
                pair: {
                    "cos_direct_belongs": float(components_by_layer[str(layer_id)][pair]["cos_direct_belongs"][0]),
                    "cos_direct_kind": float(components_by_layer[str(layer_id)][pair]["cos_direct_kind"][0]),
                    "cos_belongs_kind": float(components_by_layer[str(layer_id)][pair]["cos_belongs_kind"][0]),
                }
                for pair in CATEGORY_PAIRS
            }

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_interventions(model, tokenizer, device, layers, prompts, None, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        audit: dict[str, Any] = {}
        for pair in CATEGORY_PAIRS:
            audit[pair] = {}
            for win_name, window in windows.items():
                audit[pair][win_name] = {"window": window, "conditions": {}}
                for condition in ["common", "direct", "shuffled_template"]:
                    by_alpha = {}
                    for alpha in alphas:
                        by_alpha[str(alpha)] = run_condition(
                            model, tokenizer, device, layers, tasks, token_sets, baseline,
                            condition_interventions(components_by_layer, pair, window, condition, alpha),
                            args.batch_size, args.max_length,
                        )
                    audit[pair][win_name]["conditions"][condition] = {
                        "rows": by_alpha,
                        "transfer": transfer_gate(pair, by_alpha, alphas, args.min_abs_delta),
                    }
                random_rows = {}
                random_transfers = {}
                for seed in seeds:
                    seed_rows = {}
                    for alpha in alphas:
                        seed_rows[str(alpha)] = run_condition(
                            model, tokenizer, device, layers, tasks, token_sets, baseline,
                            condition_interventions(components_by_layer, pair, window, "random", alpha, seed=seed),
                            args.batch_size, args.max_length,
                        )
                    random_rows[str(seed)] = seed_rows
                    random_transfers[str(seed)] = transfer_gate(pair, seed_rows, alphas, args.min_abs_delta)
                audit[pair][win_name]["conditions"]["random"] = {
                    "transfers": random_transfers,
                    "max_transfer_min": float(max(t["best_transfer_min"] for t in random_transfers.values())),
                    "pass_count": int(sum(t["passes_transfer_gate"] for t in random_transfers.values())),
                }
                common_tr = audit[pair][win_name]["conditions"]["common"]["transfer"]
                rand = audit[pair][win_name]["conditions"]["random"]
                log(
                    f"    {pair} {win_name}: common_min={common_tr['best_transfer_min']:+.3f}/"
                    f"{'Y' if common_tr['passes_transfer_gate'] else 'n'} "
                    f"rand_max={rand['max_transfer_min']:+.3f} rand_pass={rand['pass_count']}"
                )

        generation: dict[str, Any] = {}
        bridge_n = args.bridge_n
        for pair, spec in CATEGORY_PAIRS.items():
            prompts = [cat_prompt("direct", x) for x in spec["pos"][-bridge_n:]]
            # Probe best common window by transfer_min for this pair.
            best_win = max(windows, key=lambda w: audit[pair][w]["conditions"]["common"]["transfer"]["best_transfer_min"])
            common_alpha = float(audit[pair][best_win]["conditions"]["common"]["transfer"]["best_alpha"])
            direct_alpha = float(audit[pair][best_win]["conditions"]["direct"]["transfer"]["best_alpha"])
            random_seed = seeds[0]
            random_alpha = float(audit[pair][best_win]["conditions"]["random"]["transfers"][str(random_seed)]["best_alpha"])
            generation[pair] = {
                "best_window": best_win,
                "baseline": generation_probe(
                    model, tokenizer, device, layers, prompts, None, pair, args.max_new_tokens, args.max_length
                ),
                "common": generation_probe(
                    model, tokenizer, device, layers, prompts,
                    condition_interventions(components_by_layer, pair, windows[best_win], "common", common_alpha),
                    pair, args.max_new_tokens, args.max_length,
                ),
                "direct": generation_probe(
                    model, tokenizer, device, layers, prompts,
                    condition_interventions(components_by_layer, pair, windows[best_win], "direct", direct_alpha),
                    pair, args.max_new_tokens, args.max_length,
                ),
                "random": generation_probe(
                    model, tokenizer, device, layers, prompts,
                    condition_interventions(components_by_layer, pair, windows[best_win], "random", random_alpha, seed=random_seed),
                    pair, args.max_new_tokens, args.max_length,
                ),
            }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase535_{args.model}_components.npz",
            **{
                f"L{layer_id}_{pair}_{name}": vec
                for layer_id, by_pair in components_by_layer.items()
                for pair, comps in by_pair.items()
                for name, vec in comps.items()
                if vec.ndim > 0 and vec.shape[0] > 1
            },
        )
        return {
            "phase": 535,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "bridge_n": args.bridge_n,
            "max_new_tokens": args.max_new_tokens,
            "alphas": alphas,
            "random_seeds": seeds,
            "min_abs_delta": args.min_abs_delta,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "baseline": baseline,
            "layer_stats": layer_stats,
            "audit": audit,
            "generation": generation,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None, help="semicolon-separated windows, e.g. 8,10,12;10,12,14")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--alphas", default="2,4,6,8")
    parser.add_argument("--random-seeds", default="11,23,37,41,53,67,79,83")
    parser.add_argument("--min-abs-delta", type=float, default=0.25)
    parser.add_argument("--bridge-n", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=4)
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
    out_path = out_dir / f"phase535_{args.model}_cumulative_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
