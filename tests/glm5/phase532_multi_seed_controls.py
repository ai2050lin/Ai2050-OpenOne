#!/usr/bin/env python3
"""
Phase 532: multi-seed orthogonal controls and category re-anchoring.

Purpose:
  Phase531 established a strict gate and showed:
    - qwen3 color/object fail under strict gate;
    - GLM4 color/object perp is non-random under one random seed;
    - DS7B movement is mostly readout-interface movement.

  This phase re-anchors qwen3 category under the same strict/random-control
  protocol, and audits GLM4/DS7B with multi-seed random_perp controls.

Loading:
  BF16 + device_map="auto"; try flash_attention_2 and fall back to SDPA.
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
    logits_with_direction,
    mean_dir,
    object_desc_prompt,
    readout_direction as phase530_readout_direction,
    score_logits,
    token_ids,
)


OUT_ROOT = Path("results/glm5_phase532_multi_seed_controls")
FRUIT = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum", "cherry", "lemon", "kiwi", "melon"]
NON_FRUIT = ["car", "truck", "bus", "dog", "cat", "shirt", "table", "chair", "hammer", "river", "stone", "cloud"]
COLOR_OBJECTS = ["apple", "car", "shirt", "ball", "flower", "box", "cup", "door", "flag", "book", "bag", "bird"]
OBJECT_DESC = {
    "car": [
        "road vehicle with four wheels for passengers",
        "motor vehicle driven on streets",
        "personal vehicle with doors and seats",
        "machine used to drive people on roads",
    ],
    "truck": [
        "large road vehicle for carrying cargo",
        "heavy vehicle with a cargo bed",
        "vehicle used to transport goods",
        "large motor vehicle for freight",
    ],
}

TASK_SPECS = {
    "category_fruit": {
        "targets": ["fruit", " fruits", "Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " object", " tool", " color"],
    },
    "color_red_blue": {"targets": ["red", " red", "Red"], "competitors": ["blue", " blue", "Blue"]},
    "color_black_white": {"targets": ["black", " black", "Black"], "competitors": ["white", " white", "White"]},
    "object_car_truck": {"targets": ["car", " car", "Car"], "competitors": ["truck", " truck", "Truck"]},
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def random_orthogonal(dim: int, basis: list[np.ndarray], norm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    for b in basis:
        v = v - float(np.dot(v, b) / (np.dot(b, b) + 1e-8)) * b
    return (normalize(v) * norm).astype(np.float32)


def task_family(task: str) -> str:
    return task.split("_", 1)[0]


def readout_direction(W_U: np.ndarray, tokenizer: Any, task: str) -> np.ndarray:
    if task in ("color_red_blue", "color_black_white", "object_car_truck"):
        return phase530_readout_direction(W_U, tokenizer, task)
    spec = TASK_SPECS[task]
    t_ids = token_ids(tokenizer, spec["targets"])
    c_ids = token_ids(tokenizer, spec["competitors"])
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def build_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    objs = COLOR_OBJECTS[:train_n]
    return {
        "category_fruit": {
            "family": "category",
            "own_task": "category_fruit",
            "pos": [f"The category of {x} is" for x in FRUIT[:train_n]],
            "neg": [f"The category of {x} is" for x in NON_FRUIT[:train_n]],
        },
        "color_red_blue_direct": {
            "family": "color",
            "own_task": "color_red_blue",
            "pos": [color_prompt("direct", "red", obj) for obj in objs],
            "neg": [color_prompt("direct", "blue", obj) for obj in objs],
        },
        "color_black_white_direct": {
            "family": "color",
            "own_task": "color_black_white",
            "pos": [color_prompt("direct", "black", obj) for obj in objs],
            "neg": [color_prompt("direct", "white", obj) for obj in objs],
        },
        "object_desc_car_truck": {
            "family": "object",
            "own_task": "object_car_truck",
            "pos": [object_desc_prompt(d) for d in OBJECT_DESC["car"]],
            "neg": [object_desc_prompt(d) for d in OBJECT_DESC["truck"]],
        },
    }


def build_tasks(test_n: int) -> dict[str, list[str]]:
    objs = COLOR_OBJECTS[-test_n:]
    return {
        "category_fruit": [f"The category of {x} is" for x in FRUIT[-test_n:]],
        "color_red_blue": [color_prompt("direct", "red", obj) for obj in objs],
        "color_black_white": [color_prompt("direct", "black", obj) for obj in objs],
        "object_car_truck": [object_desc_prompt(d) for d in OBJECT_DESC["car"]],
    }


def gate(meta: dict[str, Any], rows_by_alpha: dict[str, Any], alphas: list[float], min_abs_delta: float) -> dict[str, Any]:
    own_task = meta["own_task"]
    own_family = meta["family"]
    rows = []
    for alpha in alphas:
        key = str(alpha)
        own = float(rows_by_alpha[key][own_task]["delta_margin"])
        same = [
            abs(float(v["delta_margin"]))
            for t, v in rows_by_alpha[key].items()
            if t != own_task and task_family(t) == own_family
        ]
        off = [
            abs(float(v["delta_margin"]))
            for t, v in rows_by_alpha[key].items()
            if task_family(t) != own_family
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
        "passes_ratio_gate": bool(best["own_delta"] > 0 and best["own_delta"] > 2 * denom),
        "passes_absolute_gate": bool(best["own_delta"] >= min_abs_delta),
        "passes_strict_gate": bool(best["own_delta"] >= min_abs_delta and best["own_delta"] > 2 * denom),
        "alpha_rows": rows,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        layer_id = args.layer if args.layer is not None else PEAK_LAYERS[args.model]
        W_U = get_W_U(model, args.model).astype(np.float32)
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        log(f"{args.model}: L={info.n_layers}, d={info.d_model}, layer={layer_id}, alphas={alphas}, seeds={seeds}")

        candidates = build_candidates(args.train_n)
        tasks = build_tasks(args.test_n)
        token_sets = {
            task: {
                "target": token_ids(tokenizer, TASK_SPECS[task]["targets"]),
                "competitor": token_ids(tokenizer, TASK_SPECS[task]["competitors"]),
            }
            for task in tasks
        }

        directions: dict[str, np.ndarray] = {}
        for cand, meta in candidates.items():
            log(f"  collect {cand}: pos={len(meta['pos'])}, neg={len(meta['neg'])}")
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
            directions[cand] = mean_dir(pos_h, neg_h)

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, None, 0.0, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        component_vectors: dict[str, dict[str, np.ndarray]] = {}
        component_stats: dict[str, Any] = {}
        for cand, direction in directions.items():
            own_task = candidates[cand]["own_task"]
            readout = readout_direction(W_U, tokenizer, own_task)
            base = decompose(direction, readout)
            comps = {
                "full": base["full"],
                "parallel": base["parallel"],
                "perp": base["perp"],
                "random_readout": normalize(readout) * float(np.linalg.norm(base["parallel"])),
            }
            for seed in seeds:
                comps[f"random_perp_{seed}"] = random_orthogonal(
                    direction.shape[0], [readout], float(np.linalg.norm(base["perp"])), seed=seed + len(cand)
                )
            component_vectors[cand] = comps
            component_stats[cand] = {
                "family": candidates[cand]["family"],
                "own_task": own_task,
                "norm": float(np.linalg.norm(direction)),
                "parallel_norm_pct": float(100.0 * np.linalg.norm(base["parallel"]) / (np.linalg.norm(direction) + 1e-8)),
                "perp_norm_pct": float(100.0 * np.linalg.norm(base["perp"]) / (np.linalg.norm(direction) + 1e-8)),
                "cos_to_readout": cos(direction, readout),
            }

        sweep: dict[str, Any] = {}
        admission: dict[str, Any] = {}
        for cand, comps in component_vectors.items():
            sweep[cand] = {}
            admission[cand] = {}
            for comp_name, comp_vec in comps.items():
                sweep[cand][comp_name] = {}
                for alpha in alphas:
                    key = str(alpha)
                    sweep[cand][comp_name][key] = {}
                    for task, prompts in tasks.items():
                        logits = logits_with_direction(
                            model, tokenizer, device, layers, prompts, layer_id,
                            comp_vec, alpha, args.batch_size, args.max_length
                        )
                        sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                        sweep[cand][comp_name][key][task] = {
                            **sc,
                            "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                            "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                        }
                admission[cand][comp_name] = gate(candidates[cand], sweep[cand][comp_name], alphas, args.min_abs_delta)

            rand = [admission[cand][f"random_perp_{s}"]["best_own_delta"] for s in seeds]
            log(
                f"    {cand:24s} full={admission[cand]['full']['best_own_delta']:+.3f}/"
                f"{'Y' if admission[cand]['full']['passes_strict_gate'] else 'n'} "
                f"perp={admission[cand]['perp']['best_own_delta']:+.3f}/"
                f"{'Y' if admission[cand]['perp']['passes_strict_gate'] else 'n'} "
                f"rand_perp_max={max(rand):+.3f}"
            )

        random_perp_summary = {}
        for cand in candidates:
            vals = [admission[cand][f"random_perp_{s}"]["best_own_delta"] for s in seeds]
            pass_count = sum(1 for s in seeds if admission[cand][f"random_perp_{s}"]["passes_strict_gate"])
            random_perp_summary[cand] = {
                "seeds": seeds,
                "own_delta_values": vals,
                "mean_own_delta": float(np.mean(vals)),
                "max_own_delta": float(np.max(vals)),
                "strict_pass_count": int(pass_count),
            }

        result = {
            "phase": 532,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "random_seeds": seeds,
            "min_abs_delta": args.min_abs_delta,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "candidate_meta": {
                k: {"family": v["family"], "own_task": v["own_task"], "pos_n": len(v["pos"]), "neg_n": len(v["neg"])}
                for k, v in candidates.items()
            },
            "baseline": baseline,
            "component_stats": component_stats,
            "admission": admission,
            "random_perp_summary": random_perp_summary,
            "sweep": sweep,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase532_{args.model}_directions.npz",
            **{f"{cand}_{comp}": vec for cand, comps in component_vectors.items() for comp, vec in comps.items()},
        )
        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--test-n", type=int, default=6)
    parser.add_argument("--alphas", default="8,12")
    parser.add_argument("--random-seeds", default="11,23,37,41")
    parser.add_argument("--min-abs-delta", type=float, default=0.25)
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
    out_path = out_dir / f"phase532_{args.model}_multi_seed_controls.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
