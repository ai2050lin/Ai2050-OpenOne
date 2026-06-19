#!/usr/bin/env python3
"""
Phase 528: robust semantic subspace atlas and selectivity closure.

Purpose:
  Phase527 found a strong split:
    qwen3: orthogonal semantic component
    GLM4: readout-driven component
    DS7B: weak/non-specific activation

  But Phase527 also had a hard weakness: d_color was not a reliable positive
  control. This phase builds robust category/color/object directions and tests
  a cross-variable selectivity matrix.

Metrics:
  - direction cosine matrix
  - readout-aligned norm ratio
  - per-task target-vs-competitor margin delta
  - per-task top1 target rate delta

Loading:
  BF16 + device_map="auto" + flash_attention_2 preferred, with SDPA fallback.
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


OUT_ROOT = Path("results/glm5_phase528_semantic_subspace_atlas")
PEAK_LAYERS = {"qwen3": 12, "glm4": 26, "deepseek7b": 18}

FRUIT = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum", "cherry", "lemon", "kiwi", "melon"]
NON_FRUIT = ["car", "truck", "bus", "dog", "cat", "shirt", "table", "chair", "hammer", "river", "stone", "cloud"]
COLOR_OBJECTS = ["apple", "car", "shirt", "ball", "flower", "box", "cup", "door", "flag", "book", "bag", "bird"]
OBJECT_POS = ["apple", "banana", "orange", "grape", "mango", "pear"]
OBJECT_NEG = ["car", "truck", "bus", "train", "bicycle", "boat"]

TASK_SPECS = {
    "category": {
        "targets": ["fruit", " fruits", " Fruit"],
        "competitors": ["vehicle", " animal", " clothing", " color", " tool", " object"],
    },
    "color": {
        "targets": ["red", " red", "Red"],
        "competitors": ["blue", " green", " yellow", " black", " white", " brown"],
    },
    "object": {
        "targets": ["apple", " apple", "Apple"],
        "competitors": ["car", " banana", " orange", " truck", " dog", " shirt"],
    },
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


def mean_dir(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    d = pos.mean(axis=0) - neg.mean(axis=0)
    return d.astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def build_direction_prompts(train_n: int) -> dict[str, dict[str, list[str]]]:
    fruit = FRUIT[:train_n]
    non = NON_FRUIT[:train_n]
    color_objs = COLOR_OBJECTS[:train_n]
    obj_pos = OBJECT_POS[: max(3, min(train_n, len(OBJECT_POS)))]
    obj_neg = OBJECT_NEG[: max(3, min(train_n, len(OBJECT_NEG)))]
    return {
        "category": {
            "pos": [f"The category of {x} is" for x in fruit],
            "neg": [f"The category of {x} is" for x in non],
        },
        "color": {
            "pos": [f"The color of the red {x} is" for x in color_objs],
            "neg": [f"The color of the blue {x} is" for x in color_objs],
        },
        "object": {
            "pos": [f"The object is an {x}. The object word is" if x[0] in "aeiou" else f"The object is a {x}. The object word is" for x in obj_pos],
            "neg": [f"The object is a {x}. The object word is" for x in obj_neg],
        },
    }


def build_task_prompts(test_n: int) -> dict[str, list[str]]:
    n = test_n
    return {
        "category": [f"The category of {x} is" for x in FRUIT[-n:]],
        "color": [f"The color of the red {x} is" for x in COLOR_OBJECTS[-n:]],
        "object": [
            f"The object is an apple. The object word is",
            f"This item is an apple. Its name is",
            f"Look at the apple. The object name is",
            f"The fruit shown is apple. The word is",
            f"An apple is shown. The object is",
            f"The object to name is apple. Answer:",
        ][:n],
    }


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
    module_index = layer_id
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
            layer = layers[module_index]
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


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        layer_id = args.layer if args.layer is not None else PEAK_LAYERS[args.model]
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: L={info.n_layers}, d={info.d_model}, layer={layer_id}, train={args.train_n}, test={args.test_n}")

        direction_prompts = build_direction_prompts(args.train_n)
        directions: dict[str, np.ndarray] = {}
        for var, p in direction_prompts.items():
            log(f"  collect direction {var}: pos={len(p['pos'])}, neg={len(p['neg'])}")
            pos_h = hidden_at_layer(model, tokenizer, device, p["pos"], layer_id, args.batch_size, args.max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, p["neg"], layer_id, args.batch_size, args.max_length)
            directions[var] = mean_dir(pos_h, neg_h)

        tasks = build_task_prompts(args.test_n)
        readouts = {task: readout_direction(W_U, tokenizer, task) for task in TASK_SPECS}
        token_sets = {
            task: {
                "target": token_ids(tokenizer, TASK_SPECS[task]["targets"]),
                "competitor": token_ids(tokenizer, TASK_SPECS[task]["competitors"]),
            }
            for task in TASK_SPECS
        }

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, None, args.alpha, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        selectivity = {}
        for dir_name, direction in directions.items():
            selectivity[dir_name] = {}
            for task, prompts in tasks.items():
                logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, direction, args.alpha, args.batch_size, args.max_length)
                sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                selectivity[dir_name][task] = {
                    **sc,
                    "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                    "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                }
                log(
                    f"    dir={dir_name:8s} task={task:8s} "
                    f"Δmargin={selectivity[dir_name][task]['delta_margin']:+.3f} "
                    f"Δtop1={selectivity[dir_name][task]['delta_top1']:+.3f}"
                )

        cos_matrix = {
            a: {b: cos(directions[a], directions[b]) for b in directions}
            for a in directions
        }
        readout_alignment = {}
        for var, d in directions.items():
            dc = readouts[var]
            coeff = float(np.dot(d, dc) / (np.dot(dc, dc) + 1e-8))
            proj = coeff * dc
            readout_alignment[var] = {
                "norm": float(np.linalg.norm(d)),
                "readout_norm_pct": float(100.0 * np.linalg.norm(proj) / (np.linalg.norm(d) + 1e-8)),
                "semantic_norm_pct": float(100.0 * np.linalg.norm(d - proj) / (np.linalg.norm(d) + 1e-8)),
                "cos_to_readout": cos(d, dc),
            }

        positive_control = {
            var: {
                "own_task_delta_margin": selectivity[var][var]["delta_margin"],
                "own_task_delta_top1": selectivity[var][var]["delta_top1"],
                "max_other_abs_delta_margin": max(
                    abs(selectivity[var][t]["delta_margin"]) for t in selectivity[var] if t != var
                ),
            }
            for var in directions
        }

        result = {
            "phase": 528,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "alpha": args.alpha,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "class": info.model_class,
            },
            "baseline": baseline,
            "selectivity": selectivity,
            "cos_matrix": cos_matrix,
            "readout_alignment": readout_alignment,
            "positive_control": positive_control,
        }
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase528_{args.model}_directions.npz",
            **{f"d_{k}": v for k, v in directions.items()},
            **{f"readout_{k}": v for k, v in readouts.items()},
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
    parser.add_argument("--train-n", type=int, default=10)
    parser.add_argument("--test-n", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase528_{args.model}_semantic_subspace_atlas.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
