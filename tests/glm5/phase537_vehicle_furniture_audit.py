#!/usr/bin/env python3
"""
Phase 537: vehicle/furniture clean common candidate audit.

Focus:
  Phase536 selected qwen3 vehicle_furniture as the only current clean-common
  candidate. This phase performs a stricter audit with more seeds, more layer
  windows, wider alpha sweep, off-pair map, and generation bridge.
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
from phase536_pair_quality_selectivity import (  # noqa: E402
    CATEGORY_BANK,
    PAIR_SPECS,
    TEMPLATES,
    cat_prompt,
    encode_batch,
    pair_competitors,
    pair_from_task,
    pair_targets,
    task_name,
)


OUT_ROOT = Path("results/glm5_phase537_vehicle_furniture_audit")
SOURCE_PAIR = "vehicle_furniture"
OFFPAIR_LIST = ["fruit_tool", "animal_tool", "clothing_tool", "fruit_vegetable"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def readout_direction(W_U: np.ndarray, tokenizer: Any, pair: str) -> np.ndarray:
    t_ids = token_ids(tokenizer, pair_targets(pair))
    c_ids = token_ids(tokenizer, pair_competitors(pair))
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def selected_pairs() -> list[str]:
    return [SOURCE_PAIR] + OFFPAIR_LIST


def build_source_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    pos_label, neg_label = PAIR_SPECS[SOURCE_PAIR]
    out = {}
    for template in TEMPLATES:
        out[task_name(SOURCE_PAIR, template)] = {
            "template": template,
            "pos": [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][:train_n]],
            "neg": [cat_prompt(template, x) for x in CATEGORY_BANK[neg_label][:train_n]],
        }
    return out


def build_tasks(test_n: int) -> dict[str, list[str]]:
    out = {}
    for pair in selected_pairs():
        pos_label, _neg_label = PAIR_SPECS[pair]
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
        "early": [peak - 4, peak - 2, peak],
        "center": [peak - 2, peak, peak + 2],
        "late": [peak, peak + 2, peak + 4],
        "extended": [peak - 4, peak - 2, peak, peak + 2, peak + 4],
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


def score_rank(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, float]:
    sc = score_logits(logits, target_ids, competitor_ids)
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[1]]
    ranks = []
    if t_ids:
        for row in logits:
            target_logit = float(np.max(row[t_ids]))
            ranks.append(float(1 + np.sum(row > target_logit)))
    return {**sc, "mean_target_rank": float(np.mean(ranks)) if ranks else 0.0}


def transfer_gate(rows_by_alpha: dict[str, Any], alphas: list[float]) -> dict[str, Any]:
    own_tasks = [task_name(SOURCE_PAIR, template) for template in TEMPLATES]
    rows = []
    for alpha in alphas:
        key = str(alpha)
        own_vals = [float(rows_by_alpha[key][task]["delta_margin"]) for task in own_tasks]
        off_vals = [
            abs(float(v["delta_margin"]))
            for task, v in rows_by_alpha[key].items()
            if pair_from_task(task) != SOURCE_PAIR
        ]
        off_by_pair = {}
        for pair in OFFPAIR_LIST:
            vals = [
                abs(float(rows_by_alpha[key][task_name(pair, template)]["delta_margin"]))
                for template in TEMPLATES
            ]
            off_by_pair[pair] = max(vals)
        rows.append({
            "alpha": alpha,
            "transfer_min": min(own_vals),
            "transfer_mean": float(np.mean(own_vals)),
            "own_deltas": {own_tasks[i]: own_vals[i] for i in range(len(own_tasks))},
            "off_pair_max_abs": max(off_vals) if off_vals else 0.0,
            "off_by_pair": off_by_pair,
        })
    best = max(rows, key=lambda x: x["transfer_min"])
    return {
        "best_alpha": best["alpha"],
        "best_transfer_min": best["transfer_min"],
        "best_transfer_mean": best["transfer_mean"],
        "best_off_pair_max_abs": best["off_pair_max_abs"],
        "pair_specificity": abs(best["transfer_min"]) / (float(best["off_pair_max_abs"]) + 1e-8),
        "best_off_by_pair": best["off_by_pair"],
        "alpha_rows": rows,
    }


def build_components(dirs: dict[str, np.ndarray], W_U: np.ndarray, tokenizer: Any, seeds: list[int]) -> dict[str, np.ndarray]:
    names = [task_name(SOURCE_PAIR, template) for template in TEMPLATES]
    by_template = {name.rsplit("_", 1)[1]: dirs[name] for name in names}
    common_unit = normalize(np.mean([normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
    common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
    common_full = (common_unit * common_norm).astype(np.float32)
    readout = readout_direction(W_U, tokenizer, SOURCE_PAIR)
    comps = {
        "common": decompose(common_full, readout)["perp"],
        "direct": decompose(by_template["direct"], readout)["perp"],
        "shuffled": decompose(by_template["belongs"], readout)["perp"],
        "_cos": np.array([
            cos(by_template["direct"], by_template["belongs"]),
            cos(by_template["direct"], by_template["kind"]),
            cos(by_template["belongs"], by_template["kind"]),
        ], dtype=np.float32),
    }
    for seed in seeds:
        comps[f"random_{seed}"] = random_orthogonal(
            comps["common"].shape[0], [readout], float(np.linalg.norm(comps["common"])), seed=seed
        )
    return comps


def interventions_for(
    components_by_layer: dict[str, dict[str, np.ndarray]],
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]]:
    return {layer_id: (components_by_layer[str(layer_id)][condition], alpha) for layer_id in window}


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


def generation_probe(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    target_ids = token_ids(tokenizer, pair_targets(SOURCE_PAIR))
    competitor_ids = token_ids(tokenizer, pair_competitors(SOURCE_PAIR))
    target_set = set(target_ids)
    ranks, margins, hits, outputs = [], [], 0, []
    for prompt in prompts:
        text = prompt
        ids = []
        step_stats = []
        for _ in range(max_new_tokens):
            logits = logits_with_interventions(model, tokenizer, device, layers, [text], interventions, 1, max_length)[0]
            t_ids = [i for i in target_ids if 0 <= i < logits.shape[0]]
            c_ids = [i for i in competitor_ids if 0 <= i < logits.shape[0]]
            target_logit = float(np.max(logits[t_ids])) if t_ids else 0.0
            comp_logit = float(np.max(logits[c_ids])) if c_ids else 0.0
            rank = float(1 + np.sum(logits > target_logit)) if t_ids else float(logits.shape[0])
            step_stats.append({"target_rank": rank, "target_margin": target_logit - comp_logit})
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
        log(f"{args.model}: source={SOURCE_PAIR}, windows={windows}, alphas={alphas}, seeds={len(seeds)}")

        source_candidates = build_source_candidates(args.train_n)
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

        components_by_layer = {}
        layer_stats = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in source_candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps = build_components(dirs, W_U, tokenizer, seeds)
            components_by_layer[str(layer_id)] = comps
            layer_stats[str(layer_id)] = {
                "cos_direct_belongs": float(comps["_cos"][0]),
                "cos_direct_kind": float(comps["_cos"][1]),
                "cos_belongs_kind": float(comps["_cos"][2]),
            }

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "conditions": {}}
            for condition in ["common", "direct", "shuffled"]:
                by_alpha = {}
                for alpha in alphas:
                    by_alpha[str(alpha)] = run_condition(
                        model, tokenizer, device, layers, tasks, token_sets, baseline,
                        interventions_for(components_by_layer, window, condition, alpha),
                        args.batch_size, args.max_length,
                    )
                audit[win_name]["conditions"][condition] = {"transfer": transfer_gate(by_alpha, alphas)}
            random_transfers = {}
            for seed in seeds:
                by_alpha = {}
                for alpha in alphas:
                    by_alpha[str(alpha)] = run_condition(
                        model, tokenizer, device, layers, tasks, token_sets, baseline,
                        interventions_for(components_by_layer, window, f"random_{seed}", alpha),
                        args.batch_size, args.max_length,
                    )
                random_transfers[str(seed)] = transfer_gate(by_alpha, alphas)
            audit[win_name]["conditions"]["random"] = {
                "transfers": random_transfers,
                "max_transfer_min": float(max(t["best_transfer_min"] for t in random_transfers.values())),
                "pass_like_count": int(sum(t["best_transfer_min"] > 0.25 and t["pair_specificity"] > 1.0 for t in random_transfers.values())),
            }
            c = audit[win_name]["conditions"]["common"]["transfer"]
            r = audit[win_name]["conditions"]["random"]
            log(f"    {win_name}: common={c['best_transfer_min']:+.3f}/spec{c['pair_specificity']:.2f} rand={r['max_transfer_min']:+.3f}")

        best_window = max(audit, key=lambda w: audit[w]["conditions"]["common"]["transfer"]["best_transfer_min"])
        generation_prompts = [cat_prompt("direct", x) for x in CATEGORY_BANK["vehicle"][-args.bridge_n:]]
        gen = {"best_window": best_window}
        for condition in ["common", "direct", "shuffled"]:
            alpha = float(audit[best_window]["conditions"][condition]["transfer"]["best_alpha"])
            gen[condition] = generation_probe(
                model, tokenizer, device, layers, generation_prompts,
                interventions_for(components_by_layer, windows[best_window], condition, alpha),
                args.max_new_tokens, args.max_length,
            )
        rand_seed = seeds[0]
        rand_alpha = float(audit[best_window]["conditions"]["random"]["transfers"][str(rand_seed)]["best_alpha"])
        gen["baseline"] = generation_probe(
            model, tokenizer, device, layers, generation_prompts, None, args.max_new_tokens, args.max_length
        )
        gen["random"] = generation_probe(
            model, tokenizer, device, layers, generation_prompts,
            interventions_for(components_by_layer, windows[best_window], f"random_{rand_seed}", rand_alpha),
            args.max_new_tokens, args.max_length,
        )

        return {
            "phase": 537,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "source_pair": SOURCE_PAIR,
            "offpairs": OFFPAIR_LIST,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "bridge_n": args.bridge_n,
            "max_new_tokens": args.max_new_tokens,
            "alphas": alphas,
            "random_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "baseline": baseline,
            "layer_stats": layer_stats,
            "audit": audit,
            "generation": gen,
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
    parser.add_argument("--alphas", default="2,4,6,8,10,12")
    parser.add_argument("--random-seeds", default="11,23,37,41,53,67,79,83")
    parser.add_argument("--bridge-n", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=5)
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
    out_path = out_dir / f"phase537_{args.model}_vehicle_furniture_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
