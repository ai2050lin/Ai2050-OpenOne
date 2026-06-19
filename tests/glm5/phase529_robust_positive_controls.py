#!/usr/bin/env python3
"""
Phase 529: robust positive-control direction construction.

Purpose:
  Phase528 found one robust qwen3 category direction, but color was weak and
  object failed. This phase audits direction construction itself before adding
  more variables to a semantic atlas.

Design:
  - Build multiple candidate directions for color and object.
  - Test each candidate across alpha sweep and off-target tasks.
  - Keep category as a stable reference direction.
  - Report only objective margin/top1 deltas and admission-rule metrics.

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

from model_utils import MODEL_CONFIGS, get_W_U, get_layers, get_model_info, release_model  # noqa: E402


OUT_ROOT = Path("results/glm5_phase529_robust_positive_controls")
PEAK_LAYERS = {"qwen3": 12, "glm4": 26, "deepseek7b": 18}

FRUIT = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum", "cherry", "lemon", "kiwi", "melon"]
NON_FRUIT = ["car", "truck", "bus", "dog", "cat", "shirt", "table", "chair", "hammer", "river", "stone", "cloud"]
COLOR_OBJECTS = ["apple", "car", "shirt", "ball", "flower", "box", "cup", "door", "flag", "book", "bag", "bird"]

COLOR_PAIRS = [("red", "blue"), ("green", "yellow"), ("black", "white")]
OBJECT_PAIRS = [("apple", "banana"), ("car", "truck"), ("shirt", "jacket")]

TASK_SPECS = {
    "category_fruit": {
        "targets": ["fruit", " fruits", " Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " color", " tool", " object"],
    },
    "color_red_blue": {"targets": ["red", " red", "Red"], "competitors": ["blue", " blue", "Blue"]},
    "color_green_yellow": {"targets": ["green", " green", "Green"], "competitors": ["yellow", " yellow", "Yellow"]},
    "color_black_white": {"targets": ["black", " black", "Black"], "competitors": ["white", " white", "White"]},
    "object_apple_banana": {"targets": ["apple", " apple", "Apple"], "competitors": ["banana", " banana", "Banana"]},
    "object_car_truck": {"targets": ["car", " car", "Car"], "competitors": ["truck", " truck", "Truck"]},
    "object_shirt_jacket": {"targets": ["shirt", " shirt", "Shirt"], "competitors": ["jacket", " jacket", "Jacket"]},
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_bf16_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    last_err = None
    for attn_impl in ["flash_attention_2", "sdpa"]:
        try:
            log(f"Loading {model_name}: bf16 + device_map=auto + {attn_impl}")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            model.eval()
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"Loaded {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
            return model, tokenizer, next(model.parameters()).device, attn_impl
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"  load failed with {attn_impl}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"failed to load {model_name}") from last_err


def token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def encode_batch(tokenizer: Any, prompts: list[str], device: torch.device, max_length: int):
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in batch.items()}


def hidden_at_layer(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    layer_id: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    rows = []
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = encode_batch(tokenizer, texts, device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        hs = out.hidden_states[layer_id + 1]
        take = hs[torch.arange(hs.shape[0], device=hs.device), pos.to(hs.device)]
        rows.append(take.float().cpu().numpy().astype(np.float32))
        del out, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(rows, axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def mean_dir(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return (pos.mean(axis=0) - neg.mean(axis=0)).astype(np.float32)


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def color_prompt(color: str, obj: str) -> str:
    return f"The color of the {color} {obj} is"


def object_prompt(obj: str, color: str) -> str:
    return f"The item is {article(obj)} {color} {obj}. The object word is"


def build_direction_prompts(train_n: int) -> dict[str, dict[str, list[str]]]:
    n = train_n
    prompts: dict[str, dict[str, list[str]]] = {
        "category_fruit": {
            "pos": [f"The category of {x} is" for x in FRUIT[:n]],
            "neg": [f"The category of {x} is" for x in NON_FRUIT[:n]],
            "family": "category",
            "own_task": "category_fruit",
        }
    }

    color_objs = COLOR_OBJECTS[:n]
    all_color_pos: list[str] = []
    all_color_neg: list[str] = []
    for a, b in COLOR_PAIRS:
        name = f"color_{a}_{b}"
        pos = [color_prompt(a, x) for x in color_objs]
        neg = [color_prompt(b, x) for x in color_objs]
        prompts[name] = {"pos": pos, "neg": neg, "family": "color", "own_task": name}
        all_color_pos.extend(pos)
        all_color_neg.extend(neg)
    prompts["color_all_pairs"] = {
        "pos": all_color_pos,
        "neg": all_color_neg,
        "family": "color",
        "own_task": "color_red_blue",
    }

    colors = ["red", "green", "blue", "yellow"][: max(2, min(4, train_n))]
    all_obj_pos: list[str] = []
    all_obj_neg: list[str] = []
    for a, b in OBJECT_PAIRS:
        name = f"object_{a}_{b}"
        pos = [object_prompt(a, c) for c in colors]
        neg = [object_prompt(b, c) for c in colors]
        prompts[name] = {"pos": pos, "neg": neg, "family": "object", "own_task": name}
        all_obj_pos.extend(pos)
        all_obj_neg.extend(neg)
    prompts["object_all_pairs"] = {
        "pos": all_obj_pos,
        "neg": all_obj_neg,
        "family": "object",
        "own_task": "object_apple_banana",
    }
    return prompts


def build_task_prompts(test_n: int) -> dict[str, list[str]]:
    n = test_n
    tasks: dict[str, list[str]] = {
        "category_fruit": [f"The category of {x} is" for x in FRUIT[-n:]],
    }
    for a, b in COLOR_PAIRS:
        tasks[f"color_{a}_{b}"] = [color_prompt(a, obj) for obj in COLOR_OBJECTS[-n:]]
    colors = ["red", "green", "blue", "yellow", "black", "white"][: max(2, min(6, n))]
    for a, b in OBJECT_PAIRS:
        tasks[f"object_{a}_{b}"] = [object_prompt(a, color) for color in colors[:n]]
    return tasks


def logits_with_direction(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    direction: np.ndarray | None,
    alpha: float,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outs = []
    d_tensor = None
    if direction is not None:
        d_tensor = torch.tensor(normalize(direction) * alpha, dtype=torch.bfloat16)
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = encode_batch(tokenizer, texts, device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handle = None
        if d_tensor is not None:
            layer = layers[layer_id]
            layer_device = next(layer.parameters()).device
            d_local = d_tensor.to(layer_device)
            pos_t = pos.to(layer_device)

            def hook(_module, _inp, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += d_local.to(hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += d_local.to(hs.dtype)
                return hs

            handle = layer.register_forward_hook(hook)
        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)
        if handle is not None:
            handle.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos.to(out.logits.device)]
        outs.append(logits.float().cpu().numpy().astype(np.float32))
        del out, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def score_logits(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, float]:
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[1]]
    c_ids = [i for i in competitor_ids if 0 <= i < logits.shape[1]]
    target = logits[:, t_ids].max(axis=1) if t_ids else np.zeros(logits.shape[0], dtype=np.float32)
    comp = logits[:, c_ids].max(axis=1) if c_ids else np.zeros(logits.shape[0], dtype=np.float32)
    top = logits.argmax(axis=1)
    return {
        "target_margin": float(np.mean(target - comp)),
        "target_top1_rate": float(np.mean([1.0 if int(x) in t_ids else 0.0 for x in top])),
        "n": int(logits.shape[0]),
    }


def readout_direction(W_U: np.ndarray, tokenizer: Any, task: str) -> np.ndarray:
    spec = TASK_SPECS[task]
    t_ids = token_ids(tokenizer, spec["targets"])
    c_ids = token_ids(tokenizer, spec["competitors"])
    t = W_U[t_ids].mean(axis=0) if t_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    c = W_U[c_ids].mean(axis=0) if c_ids else np.zeros(W_U.shape[1], dtype=np.float32)
    return (t - c).astype(np.float32)


def family_tasks(family: str) -> list[str]:
    if family == "category":
        return ["category_fruit"]
    if family == "color":
        return [f"color_{a}_{b}" for a, b in COLOR_PAIRS]
    if family == "object":
        return [f"object_{a}_{b}" for a, b in OBJECT_PAIRS]
    return []


def admission_stats(candidate: str, meta: dict[str, Any], results: dict[str, Any], alpha_values: list[float]) -> dict[str, Any]:
    own_task = meta["own_task"]
    fam = meta["family"]
    fam_tasks = set(family_tasks(fam))
    rows = []
    for alpha in alpha_values:
        key = str(alpha)
        own = float(results[candidate][key][own_task]["delta_margin"])
        same_family_other = [
            abs(float(results[candidate][key][t]["delta_margin"]))
            for t in fam_tasks
            if t != own_task
        ]
        off_family = [
            abs(float(v["delta_margin"]))
            for t, v in results[candidate][key].items()
            if t not in fam_tasks
        ]
        rows.append({
            "alpha": alpha,
            "own_delta": own,
            "max_same_family_other_abs": max(same_family_other) if same_family_other else 0.0,
            "max_off_family_abs": max(off_family) if off_family else 0.0,
        })
    best = max(rows, key=lambda r: r["own_delta"])
    denominator = max(best["max_same_family_other_abs"], best["max_off_family_abs"]) + 1e-8
    return {
        "own_task": own_task,
        "family": fam,
        "best_alpha": best["alpha"],
        "best_own_delta": best["own_delta"],
        "best_max_same_family_other_abs": best["max_same_family_other_abs"],
        "best_max_off_family_abs": best["max_off_family_abs"],
        "best_selectivity_ratio": abs(best["own_delta"]) / denominator,
        "passes_basic_gate": bool(best["own_delta"] > 0 and best["own_delta"] > 2 * denominator),
        "alpha_rows": rows,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        layer_id = args.layer if args.layer is not None else PEAK_LAYERS[args.model]
        W_U = get_W_U(model, args.model).astype(np.float32)
        alpha_values = [float(x) for x in args.alphas.split(",") if x.strip()]
        log(
            f"{args.model}: L={info.n_layers}, d={info.d_model}, layer={layer_id}, "
            f"train={args.train_n}, test={args.test_n}, alphas={alpha_values}"
        )

        direction_prompts = build_direction_prompts(args.train_n)
        directions: dict[str, np.ndarray] = {}
        for name, meta in direction_prompts.items():
            log(f"  collect direction {name}: pos={len(meta['pos'])}, neg={len(meta['neg'])}")
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
            directions[name] = mean_dir(pos_h, neg_h)

        tasks = build_task_prompts(args.test_n)
        token_sets = {
            task: {
                "target": token_ids(tokenizer, TASK_SPECS[task]["targets"]),
                "competitor": token_ids(tokenizer, TASK_SPECS[task]["competitors"]),
            }
            for task in tasks
        }
        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, None, 0.0, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        sweep: dict[str, Any] = {}
        for cand, direction in directions.items():
            sweep[cand] = {}
            for alpha in alpha_values:
                key = str(alpha)
                sweep[cand][key] = {}
                for task, prompts in tasks.items():
                    logits = logits_with_direction(
                        model, tokenizer, device, layers, prompts, layer_id, direction,
                        alpha, args.batch_size, args.max_length
                    )
                    sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                    sweep[cand][key][task] = {
                        **sc,
                        "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                        "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                    }
                own = direction_prompts[cand]["own_task"]
                log(f"    cand={cand:22s} alpha={alpha:>4g} own({own}) Δ={sweep[cand][key][own]['delta_margin']:+.3f}")

        readout_alignment = {}
        for cand, direction in directions.items():
            own_task = direction_prompts[cand]["own_task"]
            dc = readout_direction(W_U, tokenizer, own_task)
            coeff = float(np.dot(direction, dc) / (np.dot(dc, dc) + 1e-8))
            proj = coeff * dc
            readout_alignment[cand] = {
                "norm": float(np.linalg.norm(direction)),
                "readout_norm_pct": float(100.0 * np.linalg.norm(proj) / (np.linalg.norm(direction) + 1e-8)),
                "semantic_norm_pct": float(100.0 * np.linalg.norm(direction - proj) / (np.linalg.norm(direction) + 1e-8)),
                "cos_to_own_readout": cos(direction, dc),
            }

        admission = {
            cand: admission_stats(cand, meta, sweep, alpha_values)
            for cand, meta in direction_prompts.items()
        }

        cos_to_category = {
            cand: cos(directions[cand], directions["category_fruit"])
            for cand in directions
        }

        result = {
            "phase": 529,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alpha_values,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "class": info.model_class,
            },
            "candidate_meta": {
                k: {"family": v["family"], "own_task": v["own_task"], "pos_n": len(v["pos"]), "neg_n": len(v["neg"])}
                for k, v in direction_prompts.items()
            },
            "baseline": baseline,
            "sweep": sweep,
            "readout_alignment": readout_alignment,
            "admission": admission,
            "cos_to_category": cos_to_category,
        }
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"phase529_{args.model}_directions.npz", **{f"d_{k}": v for k, v in directions.items()})
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
    parser.add_argument("--alphas", default="2,4,8,12")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase529_{args.model}_robust_positive_controls.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
