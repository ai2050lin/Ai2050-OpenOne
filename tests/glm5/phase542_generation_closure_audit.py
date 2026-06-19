#!/usr/bin/env python3
"""
Phase 542: generation closure audit.

Phase541 showed top-k competition movement. This phase tests whether those
single-step changes close into actual greedy generation hits over 5 tokens.
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


OUT_ROOT = Path("results/glm5_phase542_generation_closure_audit")
CONDITIONS = ["baseline", "residual_perp", "residual_parallel", "residual_full"]
CLUSTER_LABELS = ["vehicle", "furniture", "tool", "clothing"]
OFF_LABELS = ["fruit", "animal", "vegetable", "object", "thing", "item"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def label_ids(tokenizer: Any, labels: list[str]) -> list[int]:
    words = []
    for label in labels:
        words.extend([label, f" {label}", label.capitalize(), f"{label}s", f" {label}s"])
    return token_ids(tokenizer, words)


def token_groups(tokenizer: Any, source_pair: str) -> dict[str, list[int]]:
    pos, neg = PAIR_SPECS[source_pair]
    cluster_other = [x for x in CLUSTER_LABELS if x not in (pos, neg)]
    return {
        "target": label_ids(tokenizer, [pos]),
        "competitor": label_ids(tokenizer, [neg]),
        "cluster_other": label_ids(tokenizer, cluster_other),
        "off_cluster": label_ids(tokenizer, OFF_LABELS),
    }


def token_type(tok: int, groups: dict[str, list[int]]) -> str:
    for name, ids in groups.items():
        if tok in ids:
            return name
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


def interventions_for(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]] | None:
    if condition == "baseline":
        return None
    return {layer_id: (components_by_layer[str(layer_id)][source_pair][condition], alpha) for layer_id in window}


def next_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    text: str,
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    max_length: int,
) -> np.ndarray:
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha) in interventions.items():
            prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    batch = encode_batch(tokenizer, [text], device, max_length)
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
    logits = out.logits[0, int(pos.item())].float().cpu().numpy().astype(np.float32)
    del out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def best_rank(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    if not valid:
        return float(row.shape[0])
    val = float(np.max(row[valid]))
    return float(1 + np.sum(row > val))


def greedy_probe(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    groups: dict[str, list[int]],
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    hits = {"target": 0, "competitor": 0, "cluster_other": 0, "off_cluster": 0, "other_only": 0}
    first_types, target_ranks, competitor_ranks = [], [], []
    outputs = []
    group_sets = {k: set(v) for k, v in groups.items()}
    for idx, prompt in enumerate(prompts):
        text = prompt
        ids = []
        first_top = None
        for step in range(max_new_tokens):
            logits = next_logits(model, tokenizer, device, layers, text, interventions, max_length)
            if step == 0:
                tok = int(np.argmax(logits))
                first_top = {
                    "token_id": tok,
                    "token": tokenizer.decode([tok], skip_special_tokens=False),
                    "type": token_type(tok, groups),
                }
                first_types.append(first_top["type"])
                target_ranks.append(best_rank(logits, groups["target"]))
                competitor_ranks.append(best_rank(logits, groups["competitor"]))
            tok = int(np.argmax(logits))
            ids.append(tok)
            text += tokenizer.decode([tok], skip_special_tokens=False)
        hit_any = False
        for name in ["target", "competitor", "cluster_other", "off_cluster"]:
            if any(tok in group_sets[name] for tok in ids):
                hits[name] += 1
                hit_any = True
        if not hit_any:
            hits["other_only"] += 1
        if idx < 5:
            outputs.append({
                "prompt": prompt,
                "generated_suffix": text[len(prompt):],
                "generated_ids": ids,
                "first_top": first_top,
            })
    n = max(1, len(prompts))
    first_counts = {k: first_types.count(k) for k in ["target", "competitor", "cluster_other", "off_cluster", "other"]}
    return {
        "n": len(prompts),
        "hit_rates": {k: float(v / n) for k, v in hits.items()},
        "first_type_rates": {k: float(v / n) for k, v in first_counts.items()},
        "mean_first_target_rank": float(np.mean(target_ranks)) if target_ranks else 0.0,
        "mean_first_competitor_rank": float(np.mean(competitor_ranks)) if competitor_ranks else 0.0,
        "sample_outputs": outputs,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        alpha = max(alphas)
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: generation closure audit, windows={windows}, alpha={alpha}")

        candidates = build_candidates(args.train_n)
        source_prompts = build_source_prompts(args.test_n)

        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            components_by_layer[str(layer_id)] = build_components(dirs, W_U, tokenizer, seeds)

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for source_pair, prompts in source_prompts.items():
                groups = token_groups(tokenizer, source_pair)
                row = {}
                for condition in CONDITIONS:
                    row[condition] = greedy_probe(
                        model, tokenizer, device, layers, prompts,
                        interventions_for(components_by_layer, source_pair, window, condition, alpha),
                        groups, args.max_new_tokens, args.max_length,
                    )
                audit[win_name]["sources"][source_pair] = row
                rp = row["residual_parallel"]
                base = row["baseline"]
                log(
                    f"    {win_name} {source_pair}: base targetHit={base['hit_rates']['target']:.2f} "
                    f"parallel targetHit={rp['hit_rates']['target']:.2f} "
                    f"firstTarget={rp['first_type_rates']['target']:.2f}"
                )

        return {
            "phase": 542,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "templates": list(TEMPLATES),
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "max_new_tokens": args.max_new_tokens,
            "alpha": alpha,
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
    parser.add_argument("--alphas", default="6")
    parser.add_argument("--random-seeds", default="11,23")
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
    out_path = out_dir / f"phase542_{args.model}_generation_closure_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
