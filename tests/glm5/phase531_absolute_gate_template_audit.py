#!/usr/bin/env python3
"""
Phase 531: absolute-threshold gate and template path audit.

Purpose:
  Phase530 showed several "passes" based on selectivity ratio alone. Some had
  small absolute deltas, so this phase adds a minimum absolute delta gate and
  random same-norm controls.

Tests:
  - full / parallel / perp / random_perp_same_norm / random_readout_same_norm
  - minimum absolute own delta
  - color template direction cosine and norm audit
  - cross-model readout-interface audit

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
    TASK_SPECS,
    build_candidates,
    build_tasks,
    cos,
    decompose,
    hidden_at_layer,
    load_model_bf16_flash,
    logits_with_direction,
    mean_dir,
    readout_direction,
    score_logits,
    token_ids,
)


OUT_ROOT = Path("results/glm5_phase531_absolute_gate_template_audit")
COMPONENTS = ["full", "parallel", "perp", "random_perp", "random_readout"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def random_orthogonal(dim: int, basis: list[np.ndarray], norm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    for b in basis:
        denom = float(np.dot(b, b) + 1e-8)
        v = v - float(np.dot(v, b) / denom) * b
    return (normalize(v) * norm).astype(np.float32)


def task_family(task: str) -> str:
    return task.split("_", 1)[0]


def margin_values(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> np.ndarray:
    t_ids = [i for i in target_ids if 0 <= i < logits.shape[1]]
    c_ids = [i for i in competitor_ids if 0 <= i < logits.shape[1]]
    target = logits[:, t_ids].max(axis=1) if t_ids else np.zeros(logits.shape[0], dtype=np.float32)
    comp = logits[:, c_ids].max(axis=1) if c_ids else np.zeros(logits.shape[0], dtype=np.float32)
    return (target - comp).astype(np.float32)


def strict_gate(meta: dict[str, Any], component_rows: dict[str, Any], alphas: list[float], min_abs_delta: float) -> dict[str, Any]:
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
    best = max(rows, key=lambda r: r["own_delta"])
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


def template_audit(directions: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in ["red_blue", "black_white"]:
        names = [f"color_{pair}_direct", f"color_{pair}_painted", f"color_{pair}_property"]
        names = [n for n in names if n in directions]
        out[pair] = {
            "norms": {n: float(np.linalg.norm(directions[n])) for n in names},
            "cosines": {a: {b: cos(directions[a], directions[b]) for b in names} for a in names},
        }
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        layer_id = args.layer if args.layer is not None else PEAK_LAYERS[args.model]
        W_U = get_W_U(model, args.model).astype(np.float32)
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        log(
            f"{args.model}: L={info.n_layers}, d={info.d_model}, layer={layer_id}, "
            f"alphas={alphas}, min_abs={args.min_abs_delta}"
        )

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
        baseline_margin_stats = {}
        for task, prompts in tasks.items():
            logits = logits_with_direction(model, tokenizer, device, layers, prompts, layer_id, None, 0.0, args.batch_size, args.max_length)
            baseline[task] = score_logits(logits, token_sets[task]["target"], token_sets[task]["competitor"])
            margins = margin_values(logits, token_sets[task]["target"], token_sets[task]["competitor"])
            baseline_margin_stats[task] = {
                "mean": float(np.mean(margins)),
                "std": float(np.std(margins)),
                "n": int(margins.shape[0]),
            }

        component_vectors: dict[str, dict[str, np.ndarray]] = {}
        component_stats: dict[str, Any] = {}
        for idx, (cand, direction) in enumerate(directions.items()):
            own_task = candidates[cand]["own_task"]
            readout = readout_direction(W_U, tokenizer, own_task)
            base = decompose(direction, readout)
            random_perp = random_orthogonal(
                direction.shape[0], [readout], float(np.linalg.norm(base["perp"])), seed=1000 + idx
            )
            random_readout = normalize(readout) * float(np.linalg.norm(base["parallel"]))
            component_vectors[cand] = {
                "full": base["full"],
                "parallel": base["parallel"],
                "perp": base["perp"],
                "random_perp": random_perp,
                "random_readout": random_readout.astype(np.float32),
            }
            component_stats[cand] = {
                "family": candidates[cand]["family"],
                "own_task": own_task,
                "norm": float(np.linalg.norm(direction)),
                "parallel_norm_pct": float(100.0 * np.linalg.norm(base["parallel"]) / (np.linalg.norm(direction) + 1e-8)),
                "perp_norm_pct": float(100.0 * np.linalg.norm(base["perp"]) / (np.linalg.norm(direction) + 1e-8)),
                "cos_to_readout": cos(direction, readout),
                "random_perp_norm": float(np.linalg.norm(random_perp)),
                "random_readout_norm": float(np.linalg.norm(random_readout)),
            }

        sweep: dict[str, Any] = {}
        admission: dict[str, Any] = {}
        for cand, comps in component_vectors.items():
            sweep[cand] = {}
            admission[cand] = {}
            for comp_name in COMPONENTS:
                comp_dir = comps[comp_name]
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
                admission[cand][comp_name] = strict_gate(candidates[cand], sweep[cand][comp_name], alphas, args.min_abs_delta)
            own = candidates[cand]["own_task"]
            parts = " ".join(
                f"{c}={admission[cand][c]['best_own_delta']:+.3f}/"
                f"{'Y' if admission[cand][c]['passes_strict_gate'] else 'n'}"
                for c in COMPONENTS
            )
            log(f"    cand={cand:28s} own={own:20s} {parts}")

        result = {
            "phase": 531,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "layer": layer_id,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "min_abs_delta": args.min_abs_delta,
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
            "baseline_margin_stats": baseline_margin_stats,
            "template_audit": template_audit(directions),
            "component_stats": component_stats,
            "admission": admission,
            "sweep": sweep,
        }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"phase531_{args.model}_directions.npz",
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
    parser.add_argument("--alphas", default="4,8,12")
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
    out_path = out_dir / f"phase531_{args.model}_absolute_gate_template_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
