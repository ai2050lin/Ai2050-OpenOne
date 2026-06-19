#!/usr/bin/env python3
"""
Phase 533: category template robustness and generation bridge.

Purpose:
  Phase532 re-anchored qwen3 category as the strongest non-random orthogonal
  semantic candidate. This phase tests whether that anchor is robust across
  category templates, and whether margin movement begins to transfer into a
  short greedy generation trajectory.

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
    score_logits,
    token_ids,
)
from phase532_multi_seed_controls import random_orthogonal, normalize, cos  # noqa: E402


OUT_ROOT = Path("results/glm5_phase533_category_template_generation_bridge")
FRUIT = [
    "apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum",
    "cherry", "lemon", "kiwi", "melon", "apricot", "fig", "papaya", "guava",
    "lime", "coconut", "date", "berry", "nectarine", "tangerine", "persimmon", "pomegranate",
]
NON_FRUIT = [
    "car", "truck", "bus", "dog", "cat", "shirt", "table", "chair",
    "hammer", "river", "stone", "cloud", "violin", "window", "pencil", "bottle",
    "bridge", "planet", "shoe", "camera", "forest", "castle", "knife", "blanket",
]
COLOR_OBJECTS = [
    "apple", "car", "shirt", "ball", "flower", "box", "cup", "door",
    "flag", "book", "bag", "bird", "lamp", "chair", "phone", "plate",
    "toy", "hat", "wall", "pen", "kite", "vase", "boat", "shoe",
]
OBJECT_DESC = {
    "car": [
        "road vehicle with four wheels for passengers",
        "motor vehicle driven on streets",
        "personal vehicle with doors and seats",
        "machine used to drive people on roads",
        "automobile used for personal transportation",
        "small motor vehicle for city travel",
        "passenger vehicle with an engine",
        "vehicle people park in a garage",
        "road machine used for commuting",
        "four wheeled vehicle for families",
        "private transport vehicle with tires",
        "street vehicle controlled by a driver",
        "compact vehicle used on highways",
        "sedan used for daily transportation",
        "vehicle that carries passengers to work",
        "motorized passenger vehicle with headlights",
        "small road vehicle with seats",
        "personal transport machine with wheels",
        "automobile driven by one person",
        "road vehicle kept in parking lots",
    ],
    "truck": [
        "large road vehicle for carrying cargo",
        "heavy vehicle with a cargo bed",
        "vehicle used to transport goods",
        "large motor vehicle for freight",
        "cargo vehicle used for hauling loads",
        "heavy road machine for deliveries",
        "freight vehicle with a trailer",
        "work vehicle used to move equipment",
        "large transport vehicle for packages",
        "commercial vehicle for carrying materials",
        "road vehicle built for cargo",
        "vehicle with space for heavy loads",
        "large vehicle used by delivery companies",
        "hauling vehicle with a reinforced frame",
        "freight machine used on highways",
        "cargo carrier with a powerful engine",
        "vehicle designed to move heavy goods",
        "industrial road vehicle for transport",
        "large motor vehicle for construction loads",
        "delivery vehicle built for freight",
    ],
}

CATEGORY_TEMPLATES = {
    "direct": "The category of {x} is",
    "belongs": "{cap} belongs to the category",
    "kind": "A {x} is a kind of",
}

TASK_SPECS = {
    "category_direct": {
        "targets": ["fruit", " fruits", "Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " object", " tool", " color"],
    },
    "category_belongs": {
        "targets": ["fruit", " fruits", "Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " object", " tool", " color"],
    },
    "category_kind": {
        "targets": ["fruit", " fruits", "Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " object", " tool", " color"],
    },
    "color_red_blue": {"targets": ["red", " red", "Red"], "competitors": ["blue", " blue", "Blue"]},
    "object_car_truck": {"targets": ["car", " car", "Car"], "competitors": ["truck", " truck", "Truck"]},
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cat_prompt(template: str, x: str) -> str:
    return CATEGORY_TEMPLATES[template].format(x=x, cap=x.capitalize())


def task_family(task: str) -> str:
    return task.split("_", 1)[0]


def readout_direction(W_U: np.ndarray, tokenizer: Any, task: str) -> np.ndarray:
    spec = TASK_SPECS[task]
    t_ids = token_ids(tokenizer, spec["targets"])
    c_ids = token_ids(tokenizer, spec["competitors"])
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def build_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tmpl in CATEGORY_TEMPLATES:
        out[f"category_{tmpl}"] = {
            "family": "category",
            "own_task": f"category_{tmpl}",
            "pos": [cat_prompt(tmpl, x) for x in FRUIT[:train_n]],
            "neg": [cat_prompt(tmpl, x) for x in NON_FRUIT[:train_n]],
        }
    objs = COLOR_OBJECTS[:train_n]
    out["color_red_blue_direct"] = {
        "family": "color",
        "own_task": "color_red_blue",
        "pos": [color_prompt("direct", "red", obj) for obj in objs],
        "neg": [color_prompt("direct", "blue", obj) for obj in objs],
    }
    out["object_desc_car_truck"] = {
        "family": "object",
        "own_task": "object_car_truck",
        "pos": [object_desc_prompt(d) for d in OBJECT_DESC["car"][:train_n]],
        "neg": [object_desc_prompt(d) for d in OBJECT_DESC["truck"][:train_n]],
    }
    return out


def build_tasks(test_n: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tmpl in CATEGORY_TEMPLATES:
        out[f"category_{tmpl}"] = [cat_prompt(tmpl, x) for x in FRUIT[-test_n:]]
    out["color_red_blue"] = [color_prompt("direct", "red", obj) for obj in COLOR_OBJECTS[-test_n:]]
    out["object_car_truck"] = [object_desc_prompt(d) for d in OBJECT_DESC["car"][-test_n:]]
    return out


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
        "passes_strict_gate": bool(best["own_delta"] >= min_abs_delta and best["own_delta"] > 2 * denom),
        "alpha_rows": rows,
    }


def hook_step_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    layer_id: int,
    direction: np.ndarray | None,
    alpha: float,
    max_length: int,
) -> np.ndarray:
    logits = logits_with_direction(
        model, tokenizer, device, layers, [prompt], layer_id, direction, alpha, 1, max_length
    )
    return logits[0]


def greedy_bridge(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    direction: np.ndarray | None,
    alpha: float,
    target_ids: set[int],
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    hits = 0
    step_hits = [0 for _ in range(max_new_tokens)]
    outputs = []
    for prompt in prompts:
        text = prompt
        ids = []
        for step in range(max_new_tokens):
            logits = hook_step_logits(model, tokenizer, device, layers, text, layer_id, direction, alpha, max_length)
            tok = int(np.argmax(logits))
            ids.append(tok)
            if tok in target_ids:
                step_hits[step] += 1
            piece = tokenizer.decode([tok], skip_special_tokens=False)
            text += piece
        if any(t in target_ids for t in ids):
            hits += 1
        outputs.append({"prompt": prompt, "ids": ids, "text": text})
    n = max(1, len(prompts))
    return {
        "hit_rate": float(hits / n),
        "step_hit_rates": [float(x / n) for x in step_hits],
        "n": len(prompts),
        "sample_outputs": outputs[: min(3, len(outputs))],
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
        log(f"{args.model}: layer={layer_id}, alphas={alphas}, seeds={seeds}")

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
                "parallel_norm_pct": float(100.0 * np.linalg.norm(base["parallel"]) / (np.linalg.norm(direction) + 1e-8)),
                "perp_norm_pct": float(100.0 * np.linalg.norm(base["perp"]) / (np.linalg.norm(direction) + 1e-8)),
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
            log(
                f"    {cand:22s} full={admission[cand]['full']['best_own_delta']:+.3f}/"
                f"{'Y' if admission[cand]['full']['passes_strict_gate'] else 'n'} "
                f"perp={admission[cand]['perp']['best_own_delta']:+.3f}/"
                f"{'Y' if admission[cand]['perp']['passes_strict_gate'] else 'n'}"
            )

        random_perp_summary = {}
        for cand in candidates:
            vals = [admission[cand][f"random_perp_{s}"]["best_own_delta"] for s in seeds]
            random_perp_summary[cand] = {
                "max_own_delta": float(np.max(vals)),
                "mean_own_delta": float(np.mean(vals)),
                "strict_pass_count": int(sum(admission[cand][f"random_perp_{s}"]["passes_strict_gate"] for s in seeds)),
            }

        template_cosines = {
            a: {b: cos(directions[a], directions[b]) for b in directions if b.startswith("category_")}
            for a in directions if a.startswith("category_")
        }

        generation_bridge = {}
        target_ids = set(token_sets["category_direct"]["target"])
        bridge_prompts = [cat_prompt("direct", x) for x in FRUIT[-args.bridge_n:]]
        for cand in [c for c in candidates if c.startswith("category_")]:
            best_alpha = float(admission[cand]["perp"]["best_alpha"])
            generation_bridge[cand] = {
                "baseline": greedy_bridge(
                    model, tokenizer, device, layers, bridge_prompts, layer_id,
                    None, 0.0, target_ids, args.max_new_tokens, args.max_length
                ),
                "perp": greedy_bridge(
                    model, tokenizer, device, layers, bridge_prompts, layer_id,
                    component_vectors[cand]["perp"], best_alpha, target_ids,
                    args.max_new_tokens, args.max_length
                ),
                "random_readout": greedy_bridge(
                    model, tokenizer, device, layers, bridge_prompts, layer_id,
                    component_vectors[cand]["random_readout"], best_alpha, target_ids,
                    args.max_new_tokens, args.max_length
                ),
            }

        result = {
            "phase": 533,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "bridge_n": args.bridge_n,
            "max_new_tokens": args.max_new_tokens,
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
            "template_cosines": template_cosines,
            "generation_bridge": generation_bridge,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase533_{args.model}_directions.npz",
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
    parser.add_argument("--bridge-n", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3)
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
    out_path = out_dir / f"phase533_{args.model}_category_template_generation_bridge.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
