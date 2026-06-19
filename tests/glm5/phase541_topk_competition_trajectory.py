#!/usr/bin/env python3
"""
Phase 541: top-k competition trajectory audit.

Phase540 used hand-defined token groups. This phase records the real baseline
top-k tokens and measures how residual_perp / residual_parallel / residual_full
move those tokens, target labels, and competitor labels.
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
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir, token_ids  # noqa: E402
from phase532_multi_seed_controls import normalize  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt, encode_batch  # noqa: E402
from phase539_interface_cluster_mechanism import (  # noqa: E402
    CORE_SOURCES,
    PAIR_SPECS,
    build_candidates,
    build_components,
    layer_windows,
    pair_targets,
    pair_competitors,
    task_name,
)


OUT_ROOT = Path("results/glm5_phase541_topk_competition_trajectory")
CONDITIONS = ["residual_perp", "residual_parallel", "residual_full"]
LABELS = ["vehicle", "furniture", "tool", "clothing", "fruit", "animal", "vegetable", "object", "thing", "item"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def label_token_map(tokenizer: Any) -> dict[str, list[int]]:
    out = {}
    for label in LABELS:
        words = [label, f" {label}", label.capitalize(), f"{label}s", f" {label}s"]
        out[label] = token_ids(tokenizer, words)
    return out


def token_type(tok: int, label_map: dict[str, list[int]], source_pair: str) -> str:
    pos, neg = PAIR_SPECS[source_pair]
    for label, ids in label_map.items():
        if tok in ids:
            if label == pos:
                return "target_label"
            if label == neg:
                return "competitor_label"
            if label in ("vehicle", "furniture", "tool", "clothing"):
                return "cluster_label"
            return "off_cluster_label"
    return "other"


def build_source_prompts(test_n: int) -> dict[str, list[str]]:
    out = {}
    for pair in CORE_SOURCES:
        pos_label, _neg_label = PAIR_SPECS[pair]
        prompts = []
        for template in TEMPLATES:
            prompts.extend(cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][-test_n:])
        out[pair] = prompts
    return out


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
            pos_local = pos.to(layer_device)

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

            handles.append(layer.register_forward_hook(make_hook(d_local, pos_local)))
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


def interventions_for(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]]:
    return {layer_id: (components_by_layer[str(layer_id)][source_pair][condition], alpha) for layer_id in window}


def best_rank(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    if not valid:
        return float(row.shape[0])
    val = float(np.max(row[valid]))
    return float(1 + np.sum(row > val))


def best_logit(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    return float(np.max(row[valid])) if valid else 0.0


def trajectory_stats(
    tokenizer: Any,
    source_pair: str,
    baseline: np.ndarray,
    intervention: np.ndarray,
    top_k: int,
    label_map: dict[str, list[int]],
) -> dict[str, Any]:
    target_ids = token_ids(tokenizer, pair_targets(source_pair))
    competitor_ids = token_ids(tokenizer, pair_competitors(source_pair))
    top_delta_by_type: dict[str, list[float]] = {}
    rank_delta_by_type: dict[str, list[float]] = {}
    churns, target_rank_delta, competitor_rank_delta = [], [], []
    target_logit_delta, competitor_logit_delta = [], []
    examples = []
    for i, base_row in enumerate(baseline):
        int_row = intervention[i]
        base_top = np.argsort(base_row)[-top_k:][::-1]
        int_top = np.argsort(int_row)[-top_k:][::-1]
        churns.append(1.0 - len(set(base_top).intersection(set(int_top))) / float(top_k))
        target_rank_delta.append(best_rank(int_row, target_ids) - best_rank(base_row, target_ids))
        competitor_rank_delta.append(best_rank(int_row, competitor_ids) - best_rank(base_row, competitor_ids))
        target_logit_delta.append(best_logit(int_row, target_ids) - best_logit(base_row, target_ids))
        competitor_logit_delta.append(best_logit(int_row, competitor_ids) - best_logit(base_row, competitor_ids))
        rows = []
        for rank, tok in enumerate(base_top, start=1):
            typ = token_type(int(tok), label_map, source_pair)
            delta = float(int_row[tok] - base_row[tok])
            new_rank = float(1 + np.sum(int_row > int_row[tok]))
            rank_delta = new_rank - rank
            top_delta_by_type.setdefault(typ, []).append(delta)
            rank_delta_by_type.setdefault(typ, []).append(rank_delta)
            if i < 3:
                rows.append({
                    "token_id": int(tok),
                    "token": tokenizer.decode([int(tok)], skip_special_tokens=False),
                    "type": typ,
                    "base_rank": rank,
                    "new_rank": new_rank,
                    "logit_delta": delta,
                })
        if i < 3:
            examples.append({"prompt_index": i, "baseline_top": rows})
    all_types = ["target_label", "competitor_label", "cluster_label", "off_cluster_label", "other"]
    return {
        "mean_topk_churn": float(np.mean(churns)),
        "target_rank_delta": float(np.mean(target_rank_delta)),
        "competitor_rank_delta": float(np.mean(competitor_rank_delta)),
        "target_logit_delta": float(np.mean(target_logit_delta)),
        "competitor_logit_delta": float(np.mean(competitor_logit_delta)),
        "baseline_topk_delta_by_type": {
            typ: float(np.mean(top_delta_by_type.get(typ, [0.0]))) for typ in all_types
        },
        "baseline_topk_rank_delta_by_type": {
            typ: float(np.mean(rank_delta_by_type.get(typ, [0.0]))) for typ in all_types
        },
        "sample_examples": examples,
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
        log(f"{args.model}: top-k trajectory audit, windows={windows}, top_k={args.top_k}")

        candidates = build_candidates(args.train_n)
        source_prompts = build_source_prompts(args.test_n)
        label_map = label_token_map(tokenizer)

        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            components_by_layer[str(layer_id)] = build_components(dirs, W_U, tokenizer, seeds)

        baselines = {
            source: logits_with_interventions(model, tokenizer, device, layers, prompts, None, args.batch_size, args.max_length)
            for source, prompts in source_prompts.items()
        }

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for source_pair, prompts in source_prompts.items():
                source_row = {}
                for condition in CONDITIONS:
                    alpha_rows = {}
                    for alpha in alphas:
                        logits = logits_with_interventions(
                            model, tokenizer, device, layers, prompts,
                            interventions_for(components_by_layer, source_pair, window, condition, alpha),
                            args.batch_size, args.max_length,
                        )
                        alpha_rows[str(alpha)] = trajectory_stats(
                            tokenizer, source_pair, baselines[source_pair], logits, args.top_k, label_map
                        )
                    best_alpha = max(alpha_rows, key=lambda a: alpha_rows[a]["target_logit_delta"] - alpha_rows[a]["competitor_logit_delta"])
                    source_row[condition] = {
                        "best_alpha": float(best_alpha),
                        **alpha_rows[best_alpha],
                        "alpha_rows": alpha_rows,
                    }
                audit[win_name]["sources"][source_pair] = source_row
                rp = source_row["residual_parallel"]
                log(
                    f"    {win_name} {source_pair}: parallel targetΔ={rp['target_logit_delta']:+.3f} "
                    f"competitorΔ={rp['competitor_logit_delta']:+.3f} "
                    f"targetRankΔ={rp['target_rank_delta']:+.1f} churn={rp['mean_topk_churn']:.2f}"
                )

        return {
            "phase": 541,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "templates": list(TEMPLATES),
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "top_k": args.top_k,
            "alphas": alphas,
            "random_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
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
    parser.add_argument("--alphas", default="2,4,6")
    parser.add_argument("--random-seeds", default="11,23")
    parser.add_argument("--top-k", type=int, default=20)
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
    out_path = out_dir / f"phase541_{args.model}_topk_competition_trajectory.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
