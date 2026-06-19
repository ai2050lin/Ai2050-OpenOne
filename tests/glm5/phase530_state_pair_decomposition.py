#!/usr/bin/env python3
"""
Phase 530: state-pair direction decomposition and template robustness.

Purpose:
  Phase529 showed qwen3 black/white is a possible color state-pair direction,
  qwen3 red/blue still fails, qwen3 object fails, and GLM4 color/object is very
  strong but likely readout/control-heavy. This phase tests:

  1. color state-pair template robustness;
  2. readout-parallel vs readout-orthogonal decomposition;
  3. object identity directions without directly copying target object words.

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


OUT_ROOT = Path("results/glm5_phase530_state_pair_decomposition")
PEAK_LAYERS = {"qwen3": 12, "glm4": 26, "deepseek7b": 18}

COLOR_OBJECTS = ["apple", "car", "shirt", "ball", "flower", "box", "cup", "door", "flag", "book", "bag", "bird"]
COLOR_PAIRS = [("red", "blue"), ("black", "white")]
COLOR_TEMPLATES = {
    "direct": "The color of the {color} {obj} is",
    "painted": "A {obj} painted {color} has color",
    "property": "This {obj} is {color}. Its color is",
}

OBJECT_DESC = {
    "apple": [
        "round fruit, often red or green, grows on trees",
        "small edible fruit with seeds inside",
        "fruit often used in pies and juice",
        "tree fruit that can be crisp and sweet",
    ],
    "banana": [
        "long curved yellow fruit with a peel",
        "soft sweet fruit that grows in bunches",
        "fruit peeled before eating, often yellow",
        "curved tropical fruit with soft flesh",
    ],
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
OBJECT_PAIRS = [("apple", "banana"), ("car", "truck")]

TASK_SPECS = {
    "color_red_blue": {"targets": ["red", " red", "Red"], "competitors": ["blue", " blue", "Blue"]},
    "color_black_white": {"targets": ["black", " black", "Black"], "competitors": ["white", " white", "White"]},
    "object_apple_banana": {"targets": ["apple", " apple", "Apple"], "competitors": ["banana", " banana", "Banana"]},
    "object_car_truck": {"targets": ["car", " car", "Car"], "competitors": ["truck", " truck", "Truck"]},
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


def color_prompt(template: str, color: str, obj: str) -> str:
    return COLOR_TEMPLATES[template].format(color=color, obj=obj)


def object_desc_prompt(desc: str) -> str:
    return f"The described item is a {desc}. The object name is"


def build_candidates(train_n: int) -> dict[str, dict[str, Any]]:
    objs = COLOR_OBJECTS[:train_n]
    candidates: dict[str, dict[str, Any]] = {}
    for a, b in COLOR_PAIRS:
        for tmpl in COLOR_TEMPLATES:
            name = f"color_{a}_{b}_{tmpl}"
            candidates[name] = {
                "family": "color",
                "own_task": f"color_{a}_{b}",
                "pos": [color_prompt(tmpl, a, obj) for obj in objs],
                "neg": [color_prompt(tmpl, b, obj) for obj in objs],
            }
    for a, b in OBJECT_PAIRS:
        name = f"object_desc_{a}_{b}"
        candidates[name] = {
            "family": "object",
            "own_task": f"object_{a}_{b}",
            "pos": [object_desc_prompt(d) for d in OBJECT_DESC[a]],
            "neg": [object_desc_prompt(d) for d in OBJECT_DESC[b]],
        }
    return candidates


def build_tasks(test_n: int) -> dict[str, list[str]]:
    objs = COLOR_OBJECTS[-test_n:]
    tasks: dict[str, list[str]] = {}
    for a, b in COLOR_PAIRS:
        prompts = []
        for tmpl in COLOR_TEMPLATES:
            prompts.extend(color_prompt(tmpl, a, obj) for obj in objs)
        tasks[f"color_{a}_{b}"] = prompts
    for a, b in OBJECT_PAIRS:
        tasks[f"object_{a}_{b}"] = [object_desc_prompt(d) for d in OBJECT_DESC[a]]
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


def decompose(direction: np.ndarray, readout: np.ndarray) -> dict[str, np.ndarray]:
    coeff = float(np.dot(direction, readout) / (np.dot(readout, readout) + 1e-8))
    parallel = (coeff * readout).astype(np.float32)
    perp = (direction - parallel).astype(np.float32)
    return {"full": direction, "parallel": parallel, "perp": perp}


def task_family(task: str) -> str:
    return task.split("_", 1)[0]


def admission_for(candidate: str, meta: dict[str, Any], component_rows: dict[str, Any], alphas: list[float]) -> dict[str, Any]:
    own_task = meta["own_task"]
    own_family = meta["family"]
    rows = []
    for alpha in alphas:
        key = str(alpha)
        own = float(component_rows[key][own_task]["delta_margin"])
        same = [
            abs(float(v["delta_margin"]))
            for t, v in component_rows[key].items()
            if t != own_task and task_family(t) == own_family
        ]
        off = [
            abs(float(v["delta_margin"]))
            for t, v in component_rows[key].items()
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
        "passes_basic_gate": bool(best["own_delta"] > 0 and best["own_delta"] > 2 * denom),
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
        log(f"{args.model}: L={info.n_layers}, d={info.d_model}, layer={layer_id}, alphas={alphas}")

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
        for name, meta in candidates.items():
            log(f"  collect {name}: pos={len(meta['pos'])}, neg={len(meta['neg'])}")
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
            directions[name] = mean_dir(pos_h, neg_h)

        baseline = {}
        for task, prompts in tasks.items():
            logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, None, 0.0, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])

        components: dict[str, dict[str, np.ndarray]] = {}
        component_stats: dict[str, Any] = {}
        for name, direction in directions.items():
            own_task = candidates[name]["own_task"]
            readout = readout_direction(W_U, tokenizer, own_task)
            comp = decompose(direction, readout)
            components[name] = comp
            parallel = comp["parallel"]
            perp = comp["perp"]
            component_stats[name] = {
                "family": candidates[name]["family"],
                "own_task": own_task,
                "norm": float(np.linalg.norm(direction)),
                "parallel_norm_pct": float(100.0 * np.linalg.norm(parallel) / (np.linalg.norm(direction) + 1e-8)),
                "perp_norm_pct": float(100.0 * np.linalg.norm(perp) / (np.linalg.norm(direction) + 1e-8)),
                "cos_to_readout": cos(direction, readout),
            }

        sweep: dict[str, Any] = {}
        admission: dict[str, Any] = {}
        for cand, comps in components.items():
            sweep[cand] = {}
            admission[cand] = {}
            for comp_name, comp_dir in comps.items():
                sweep[cand][comp_name] = {}
                for alpha in alphas:
                    key = str(alpha)
                    sweep[cand][comp_name][key] = {}
                    for task, prompts in tasks.items():
                        logits = logits_with_direction(
                            model, tokenizer, device, layers, prompts, layer_id, comp_dir,
                            alpha, args.batch_size, args.max_length
                        )
                        sc = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
                        sweep[cand][comp_name][key][task] = {
                            **sc,
                            "delta_margin": float(sc["target_margin"] - baseline[task]["target_margin"]),
                            "delta_top1": float(sc["target_top1_rate"] - baseline[task]["target_top1_rate"]),
                        }
                admission[cand][comp_name] = admission_for(cand, candidates[cand], sweep[cand][comp_name], alphas)
            own = candidates[cand]["own_task"]
            full = admission[cand]["full"]
            par = admission[cand]["parallel"]
            perp = admission[cand]["perp"]
            log(
                f"    cand={cand:28s} own={own:20s} "
                f"full={full['best_own_delta']:+.3f} par={par['best_own_delta']:+.3f} perp={perp['best_own_delta']:+.3f}"
            )

        result = {
            "phase": 530,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "class": info.model_class,
            },
            "candidate_meta": {
                k: {
                    "family": v["family"],
                    "own_task": v["own_task"],
                    "pos_n": len(v["pos"]),
                    "neg_n": len(v["neg"]),
                }
                for k, v in candidates.items()
            },
            "baseline": baseline,
            "component_stats": component_stats,
            "admission": admission,
            "sweep": sweep,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase530_{args.model}_directions.npz",
            **{f"{cand}_{comp}": vec for cand, comps in components.items() for comp, vec in comps.items()},
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
    parser.add_argument("--alphas", default="4,8,12")
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
    out_path = out_dir / f"phase530_{args.model}_state_pair_decomposition.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
