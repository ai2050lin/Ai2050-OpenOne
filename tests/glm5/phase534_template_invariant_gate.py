#!/usr/bin/env python3
"""
Phase 534: template-invariant direction extraction and generation policy gate.

This phase tests whether qwen3 category has a cross-template common component,
or whether Phase533's effect is mainly a direct-template path. It also widens
the generation bridge from exact target-token hit to target rank, margin, and
category-explanation path traces.
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
    color_prompt,
    decompose,
    hidden_at_layer,
    load_model_bf16_flash,
    mean_dir,
    object_desc_prompt,
    score_logits,
    token_ids,
)
from phase532_multi_seed_controls import cos, normalize, random_orthogonal  # noqa: E402
from phase533_category_template_generation_bridge import (  # noqa: E402
    CATEGORY_TEMPLATES,
    COLOR_OBJECTS,
    FRUIT,
    NON_FRUIT,
    OBJECT_DESC,
    TASK_SPECS,
    cat_prompt,
    readout_direction,
    task_family,
)


OUT_ROOT = Path("results/glm5_phase534_template_invariant_gate")
CATEGORY_TASKS = ["category_direct", "category_belongs", "category_kind"]
CONTROL_TASKS = ["color_red_blue", "object_car_truck"]
PATH_MARKERS = [
    "type of", "kind of", "category of", "belongs to", "fruit", "fruits", "family",
    "class of", "sort of",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device, max_length: int):
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in batch.items()}


def build_category_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tmpl in CATEGORY_TEMPLATES:
        out[f"category_{tmpl}"] = {
            "family": "category",
            "own_task": f"category_{tmpl}",
            "pos": [cat_prompt(tmpl, x) for x in FRUIT[:train_n]],
            "neg": [cat_prompt(tmpl, x) for x in NON_FRUIT[:train_n]],
        }
    out["color_red_blue_direct"] = {
        "family": "color",
        "own_task": "color_red_blue",
        "pos": [color_prompt("direct", "red", obj) for obj in COLOR_OBJECTS[:train_n]],
        "neg": [color_prompt("direct", "blue", obj) for obj in COLOR_OBJECTS[:train_n]],
    }
    out["object_desc_car_truck"] = {
        "family": "object",
        "own_task": "object_car_truck",
        "pos": [object_desc_prompt(d) for d in OBJECT_DESC["car"][:train_n]],
        "neg": [object_desc_prompt(d) for d in OBJECT_DESC["truck"][:train_n]],
    }
    return out


def build_tasks(test_n: int) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    for tmpl in CATEGORY_TEMPLATES:
        tasks[f"category_{tmpl}"] = [cat_prompt(tmpl, x) for x in FRUIT[-test_n:]]
    tasks["color_red_blue"] = [color_prompt("direct", "red", obj) for obj in COLOR_OBJECTS[-test_n:]]
    tasks["object_car_truck"] = [object_desc_prompt(d) for d in OBJECT_DESC["car"][-test_n:]]
    return tasks


def project_on(direction: np.ndarray, unit: np.ndarray) -> np.ndarray:
    return (float(np.dot(direction, unit)) * unit).astype(np.float32)


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
    outs = []
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha) in interventions.items():
            prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = encode_batch(tokenizer, texts, device, max_length)
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


def transfer_gate(rows_by_alpha: dict[str, Any], alphas: list[float], min_abs_delta: float) -> dict[str, Any]:
    rows = []
    for alpha in alphas:
        key = str(alpha)
        cat_vals = [float(rows_by_alpha[key][task]["delta_margin"]) for task in CATEGORY_TASKS]
        off_vals = [
            abs(float(v["delta_margin"]))
            for task, v in rows_by_alpha[key].items()
            if task not in CATEGORY_TASKS
        ]
        rows.append({
            "alpha": alpha,
            "category_min_delta": min(cat_vals),
            "category_mean_delta": float(np.mean(cat_vals)),
            "category_deltas": {task: cat_vals[i] for i, task in enumerate(CATEGORY_TASKS)},
            "off_family_max_abs": max(off_vals) if off_vals else 0.0,
        })
    best = max(rows, key=lambda x: x["category_min_delta"])
    denom = float(best["off_family_max_abs"]) + 1e-8
    return {
        "best_alpha": best["alpha"],
        "best_category_min_delta": best["category_min_delta"],
        "best_category_mean_delta": best["category_mean_delta"],
        "best_off_family_max_abs": best["off_family_max_abs"],
        "best_transfer_ratio": abs(best["category_min_delta"]) / denom,
        "passes_transfer_gate": bool(
            best["category_min_delta"] >= min_abs_delta
            and best["category_min_delta"] > 1.5 * denom
        ),
        "alpha_rows": rows,
    }


def own_gate(
    own_task: str,
    own_family: str,
    rows_by_alpha: dict[str, Any],
    alphas: list[float],
    min_abs_delta: float,
) -> dict[str, Any]:
    rows = []
    for alpha in alphas:
        key = str(alpha)
        own = float(rows_by_alpha[key][own_task]["delta_margin"])
        same = [
            abs(float(v["delta_margin"]))
            for task, v in rows_by_alpha[key].items()
            if task != own_task and task_family(task) == own_family
        ]
        off = [
            abs(float(v["delta_margin"]))
            for task, v in rows_by_alpha[key].items()
            if task_family(task) != own_family
        ]
        rows.append({
            "alpha": alpha,
            "own_delta": own,
            "same_family_max_abs": max(same) if same else 0.0,
            "off_family_max_abs": max(off) if off else 0.0,
        })
    best = max(rows, key=lambda x: x["own_delta"])
    denom = max(best["same_family_max_abs"], best["off_family_max_abs"]) + 1e-8
    return {
        "best_alpha": best["alpha"],
        "best_own_delta": best["own_delta"],
        "best_same_family_max_abs": best["same_family_max_abs"],
        "best_off_family_max_abs": best["off_family_max_abs"],
        "best_selectivity_ratio": abs(best["own_delta"]) / denom,
        "passes_strict_gate": bool(best["own_delta"] >= min_abs_delta and best["own_delta"] > 2 * denom),
        "alpha_rows": rows,
    }


def rank_and_margin(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, float]:
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[0]]
    c_ids = [i for i in competitor_ids if 0 <= i < logits.shape[0]]
    if not t_ids:
        return {"target_rank": float(logits.shape[0]), "target_logit": 0.0, "target_margin": 0.0}
    target_vals = logits[t_ids]
    target_logit = float(np.max(target_vals))
    comp_logit = float(np.max(logits[c_ids])) if c_ids else 0.0
    rank = int(1 + np.sum(logits > target_logit))
    return {"target_rank": float(rank), "target_logit": target_logit, "target_margin": target_logit - comp_logit}


def generation_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    target_ids: list[int],
    competitor_ids: list[int],
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    rows = []
    any_target_hits = 0
    path_hits = 0
    best_ranks = []
    first_margins = []
    for prompt in prompts:
        text = prompt
        ids = []
        step_stats = []
        for _step in range(max_new_tokens):
            logits = logits_with_interventions(
                model, tokenizer, device, layers, [text], interventions, 1, max_length
            )[0]
            stats = rank_and_margin(logits, target_ids, competitor_ids)
            step_stats.append(stats)
            tok = int(np.argmax(logits))
            ids.append(tok)
            text += tokenizer.decode([tok], skip_special_tokens=False)
        if any(tok in set(target_ids) for tok in ids):
            any_target_hits += 1
        generated_suffix = text[len(prompt):]
        low = generated_suffix.lower()
        if any(marker in low for marker in PATH_MARKERS):
            path_hits += 1
        best_ranks.append(min(s["target_rank"] for s in step_stats))
        first_margins.append(step_stats[0]["target_margin"] if step_stats else 0.0)
        rows.append({
            "prompt": prompt,
            "ids": ids,
            "generated_suffix": generated_suffix,
            "text": text,
            "step_stats": step_stats,
        })
    n = max(1, len(prompts))
    return {
        "n": len(prompts),
        "target_hit_rate": float(any_target_hits / n),
        "semantic_path_hit_rate": float(path_hits / n),
        "mean_best_target_rank": float(np.mean(best_ranks)) if best_ranks else 0.0,
        "mean_first_step_margin": float(np.mean(first_margins)) if first_margins else 0.0,
        "sample_outputs": rows[: min(3, len(rows))],
    }


def layer_window(model: str, n_layers: int, layer_arg: str | None) -> list[int]:
    if layer_arg:
        layers = [int(x) for x in layer_arg.split(",") if x.strip()]
    else:
        peak = PEAK_LAYERS[model]
        layers = [peak - 2, peak, peak + 2]
    return [x for x in layers if 0 <= x < n_layers]


def build_layer_components(
    directions: dict[str, np.ndarray],
    W_U: np.ndarray,
    tokenizer: Any,
    seeds: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cat_names = ["category_direct", "category_belongs", "category_kind"]
    cat_dirs = [directions[name] for name in cat_names]
    common_unit = normalize(np.mean([normalize(x) for x in cat_dirs], axis=0).astype(np.float32))
    common_norm = float(np.mean([np.linalg.norm(x) for x in cat_dirs]))
    common_full = (common_unit * common_norm).astype(np.float32)
    category_readout = readout_direction(W_U, tokenizer, "category_direct")
    common_dec = decompose(common_full, category_readout)

    comps: dict[str, np.ndarray] = {
        "category_common_full": common_full,
        "category_common_perp": common_dec["perp"],
    }
    stats: dict[str, Any] = {
        "category_template_cosines": {
            a: {b: cos(directions[a], directions[b]) for b in cat_names}
            for a in cat_names
        },
        "common_norm": common_norm,
        "common_parallel_norm_pct": float(
            100.0 * np.linalg.norm(common_dec["parallel"]) / (np.linalg.norm(common_full) + 1e-8)
        ),
    }
    for name in cat_names:
        readout = readout_direction(W_U, tokenizer, name)
        dec = decompose(directions[name], readout)
        residual = (directions[name] - project_on(directions[name], common_unit)).astype(np.float32)
        comps[f"{name}_perp"] = dec["perp"]
        comps[f"{name}_residual"] = residual
        stats[f"{name}_cos_to_common"] = cos(directions[name], common_full)
        stats[f"{name}_residual_norm_pct"] = float(
            100.0 * np.linalg.norm(residual) / (np.linalg.norm(directions[name]) + 1e-8)
        )
    color_dec = decompose(directions["color_red_blue_direct"], readout_direction(W_U, tokenizer, "color_red_blue"))
    obj_dec = decompose(directions["object_desc_car_truck"], readout_direction(W_U, tokenizer, "object_car_truck"))
    comps["color_red_blue_perp"] = color_dec["perp"]
    comps["color_red_blue_parallel"] = color_dec["parallel"]
    comps["object_car_truck_perp"] = obj_dec["perp"]
    comps["object_car_truck_parallel"] = obj_dec["parallel"]
    for seed in seeds:
        comps[f"random_common_perp_{seed}"] = random_orthogonal(
            common_full.shape[0], [category_readout], float(np.linalg.norm(common_dec["perp"])), seed=seed
        )
    return comps, stats


def component_meta(name: str) -> tuple[str, str]:
    if name.startswith("color_"):
        return "color_red_blue", "color"
    if name.startswith("object_"):
        return "object_car_truck", "object"
    if name.startswith("category_belongs"):
        return "category_belongs", "category"
    if name.startswith("category_kind"):
        return "category_kind", "category"
    return "category_direct", "category"


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        layer_ids = layer_window(args.model, info.n_layers, args.layers)
        primary_layer = layer_ids[len(layer_ids) // 2]
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        cumulative_alphas = [float(x) for x in args.cumulative_alphas.split(",") if x.strip()]
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: layers={layer_ids}, primary={primary_layer}, alphas={alphas}, seeds={seeds}")

        candidates = build_category_candidates(args.train_n)
        tasks = build_tasks(args.test_n)
        token_sets = {
            task: {
                "target": token_ids(tokenizer, TASK_SPECS[task]["targets"]),
                "competitor": token_ids(tokenizer, TASK_SPECS[task]["competitors"]),
            }
            for task in tasks
        }

        directions_by_layer: dict[str, dict[str, np.ndarray]] = {}
        components_by_layer: dict[str, dict[str, np.ndarray]] = {}
        layer_stats: dict[str, Any] = {}
        for layer_id in layer_ids:
            log(f"  collect layer L{layer_id}")
            dirs: dict[str, np.ndarray] = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps, stats = build_layer_components(dirs, W_U, tokenizer, seeds)
            directions_by_layer[str(layer_id)] = dirs
            components_by_layer[str(layer_id)] = comps
            layer_stats[str(layer_id)] = stats

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_interventions(model, tokenizer, device, layers, prompts, None, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        primary_components = components_by_layer[str(primary_layer)]
        sweep: dict[str, Any] = {}
        admission: dict[str, Any] = {}
        for comp_name, comp_vec in primary_components.items():
            sweep[comp_name] = {}
            for alpha in alphas:
                key = str(alpha)
                sweep[comp_name][key] = {}
                for task, prompts in tasks.items():
                    logits = logits_with_interventions(
                        model, tokenizer, device, layers, prompts,
                        {primary_layer: (comp_vec, alpha)}, args.batch_size, args.max_length
                    )
                    sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                    sweep[comp_name][key][task] = {
                        **sc,
                        "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                        "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                    }
            own_task, family = component_meta(comp_name)
            admission[comp_name] = {
                "own": own_gate(own_task, family, sweep[comp_name], alphas, args.min_abs_delta),
                "transfer": transfer_gate(sweep[comp_name], alphas, args.min_abs_delta)
                if family == "category" else None,
            }
            if comp_name in ("category_common_perp", "category_direct_perp", "color_red_blue_perp"):
                own = admission[comp_name]["own"]
                tr = admission[comp_name]["transfer"]
                tr_msg = "" if tr is None else f" transfer_min={tr['best_category_min_delta']:+.3f}/{'Y' if tr['passes_transfer_gate'] else 'n'}"
                log(f"    {comp_name}: own={own['best_own_delta']:+.3f}/{'Y' if own['passes_strict_gate'] else 'n'}{tr_msg}")

        cumulative: dict[str, Any] = {}
        for alpha in cumulative_alphas:
            key = str(alpha)
            interventions = {
                layer_id: (components_by_layer[str(layer_id)]["category_common_perp"], alpha)
                for layer_id in layer_ids
            }
            cumulative[key] = {}
            for task, prompts in tasks.items():
                logits = logits_with_interventions(
                    model, tokenizer, device, layers, prompts, interventions, args.batch_size, args.max_length
                )
                sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                cumulative[key][task] = {
                    **sc,
                    "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                    "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                }
        cumulative_admission = transfer_gate(cumulative, cumulative_alphas, args.min_abs_delta)

        bridge_prompts = [cat_prompt("direct", x) for x in FRUIT[-args.bridge_n:]]
        target_ids = token_sets["category_direct"]["target"]
        competitor_ids = token_sets["category_direct"]["competitor"]
        best_single_alpha = float(admission["category_common_perp"]["own"]["best_alpha"])
        best_cum_alpha = float(cumulative_admission["best_alpha"])
        generation_bridge = {
            "baseline": generation_trace(
                model, tokenizer, device, layers, bridge_prompts, None,
                target_ids, competitor_ids, args.max_new_tokens, args.max_length
            ),
            "single_common_perp": generation_trace(
                model, tokenizer, device, layers, bridge_prompts,
                {primary_layer: (primary_components["category_common_perp"], best_single_alpha)},
                target_ids, competitor_ids, args.max_new_tokens, args.max_length
            ),
            "single_direct_perp": generation_trace(
                model, tokenizer, device, layers, bridge_prompts,
                {primary_layer: (primary_components["category_direct_perp"], best_single_alpha)},
                target_ids, competitor_ids, args.max_new_tokens, args.max_length
            ),
            "cumulative_common_perp": generation_trace(
                model, tokenizer, device, layers, bridge_prompts,
                {
                    layer_id: (components_by_layer[str(layer_id)]["category_common_perp"], best_cum_alpha)
                    for layer_id in layer_ids
                },
                target_ids, competitor_ids, args.max_new_tokens, args.max_length
            ),
        }

        random_common_summary = {
            "max_transfer_min_delta": float(max(
                admission[f"random_common_perp_{seed}"]["transfer"]["best_category_min_delta"] for seed in seeds
            )),
            "strict_transfer_pass_count": int(sum(
                admission[f"random_common_perp_{seed}"]["transfer"]["passes_transfer_gate"] for seed in seeds
            )),
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase534_{args.model}_components.npz",
            **{
                f"L{layer_id}_{name}": vec
                for layer_id, comps in components_by_layer.items()
                for name, vec in comps.items()
            },
        )
        return {
            "phase": 534,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layers": layer_ids,
            "primary_layer": primary_layer,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "bridge_n": args.bridge_n,
            "max_new_tokens": args.max_new_tokens,
            "alphas": alphas,
            "cumulative_alphas": cumulative_alphas,
            "random_seeds": seeds,
            "min_abs_delta": args.min_abs_delta,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "baseline": baseline,
            "layer_stats": layer_stats,
            "admission": admission,
            "random_common_summary": random_common_summary,
            "cumulative": cumulative,
            "cumulative_admission": cumulative_admission,
            "generation_bridge": generation_bridge,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layers", default=None, help="comma separated layer ids; default peak-2,peak,peak+2")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--alphas", default="8,12")
    parser.add_argument("--cumulative-alphas", default="2,4,6")
    parser.add_argument("--random-seeds", default="11,23,37,41,53,67,79,83")
    parser.add_argument("--min-abs-delta", type=float, default=0.25)
    parser.add_argument("--bridge-n", type=int, default=6)
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
    out_path = out_dir / f"phase534_{args.model}_template_invariant_gate.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
