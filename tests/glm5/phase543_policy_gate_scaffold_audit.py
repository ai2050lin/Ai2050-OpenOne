#!/usr/bin/env python3
"""
Phase 543: policy gate and scaffold sensitivity audit.

Phase542 showed that top-k movement does not necessarily close into generation.
This phase keeps the same vehicle-centered interface sources, but varies prompt
scaffolds and measures target-family hits at several generation lengths from one
single greedy rollout.
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
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase532_multi_seed_controls import normalize  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, encode_batch  # noqa: E402
from phase539_interface_cluster_mechanism import (  # noqa: E402
    CORE_SOURCES,
    PAIR_SPECS,
    build_candidates,
    build_components,
    layer_windows,
)
from phase542_generation_closure_audit import CONDITIONS, label_ids, token_groups  # noqa: E402


OUT_ROOT = Path("results/glm5_phase543_policy_gate_scaffold_audit")
SCAFFOLDS = ["direct", "one_word", "choose_pair", "label_only"]
CHECKPOINTS = [1, 3, 5, 10]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scaffold_prompt(scaffold: str, obj: str, pos_label: str, neg_label: str) -> str:
    cap = obj.capitalize()
    if scaffold == "direct":
        return f"The category of {obj} is"
    if scaffold == "one_word":
        return f"Answer with one category word. {cap} is a"
    if scaffold == "choose_pair":
        return f"Choose one category: {pos_label} or {neg_label}. {cap} is"
    if scaffold == "label_only":
        return f"Category label for {obj}:"
    raise ValueError(f"unknown scaffold: {scaffold}")


def build_scaffold_prompts(test_n: int, scaffolds: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for pair in CORE_SOURCES:
        pos_label, neg_label = PAIR_SPECS[pair]
        objects = CATEGORY_BANK[pos_label][-test_n:]
        out[pair] = {}
        for scaffold in scaffolds:
            out[pair][scaffold] = [scaffold_prompt(scaffold, x, pos_label, neg_label) for x in objects]
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


def prepare_interventions(
    interventions: dict[int, tuple[np.ndarray, float]] | None,
) -> dict[int, torch.Tensor]:
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha) in interventions.items():
            prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    return prepared


def batched_next_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    texts: list[str],
    prepared: dict[int, torch.Tensor],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = encode_batch(tokenizer, texts[start:start + batch_size], device, max_length)
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
            idx = pos.to(out.logits.device)
            rows = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), idx]
            outs.append(rows.float().cpu().numpy().astype(np.float32))
        for handle in handles:
            handle.remove()
        del out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def token_type(tok: int, groups: dict[str, list[int]]) -> str:
    for name, ids in groups.items():
        if tok in ids:
            return name
    return "other"


def best_rank(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    if not valid:
        return float(row.shape[0])
    val = float(np.max(row[valid]))
    return float(1 + np.sum(row > val))


def greedy_probe_scaffold(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    groups: dict[str, list[int]],
    max_new_tokens: int,
    checkpoints: list[int],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    prepared = prepare_interventions(interventions)
    texts = list(prompts)
    generated: list[list[int]] = [[] for _ in prompts]
    first_types, target_ranks, competitor_ranks = [], [], []

    for step in range(max_new_tokens):
        logits = batched_next_logits(model, tokenizer, device, layers, texts, prepared, batch_size, max_length)
        toks = np.argmax(logits, axis=1).astype(np.int64).tolist()
        if step == 0:
            for row, tok in zip(logits, toks):
                first_types.append(token_type(int(tok), groups))
                target_ranks.append(best_rank(row, groups["target"]))
                competitor_ranks.append(best_rank(row, groups["competitor"]))
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            texts[i] += tokenizer.decode([int(tok)], skip_special_tokens=False)

    group_sets = {k: set(v) for k, v in groups.items()}
    checkpoint_metrics = {}
    for cp in checkpoints:
        cp = min(cp, max_new_tokens)
        hits = {"target": 0, "competitor": 0, "cluster_other": 0, "off_cluster": 0, "other_only": 0}
        for ids in generated:
            prefix = ids[:cp]
            hit_any = False
            for name in ["target", "competitor", "cluster_other", "off_cluster"]:
                if any(tok in group_sets[name] for tok in prefix):
                    hits[name] += 1
                    hit_any = True
            if not hit_any:
                hits["other_only"] += 1
        n = max(1, len(generated))
        checkpoint_metrics[str(cp)] = {k: float(v / n) for k, v in hits.items()}

    n = max(1, len(prompts))
    first_counts = {k: first_types.count(k) for k in ["target", "competitor", "cluster_other", "off_cluster", "other"]}
    return {
        "n": len(prompts),
        "checkpoints": sorted(set(min(x, max_new_tokens) for x in checkpoints)),
        "hit_at_k": checkpoint_metrics,
        "hit_rates": checkpoint_metrics[str(max_new_tokens)],
        "first_type_rates": {k: float(v / n) for k, v in first_counts.items()},
        "mean_first_target_rank": float(np.mean(target_ranks)) if target_ranks else 0.0,
        "mean_first_competitor_rank": float(np.mean(competitor_ranks)) if competitor_ranks else 0.0,
        "sample_outputs": [
            {
                "prompt": prompts[i],
                "generated_suffix": texts[i][len(prompts[i]):],
                "generated_ids": generated[i],
            }
            for i in range(min(4, len(prompts)))
        ],
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = max(float(x) for x in args.alphas.split(",") if x.strip())
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        scaffolds = [x.strip() for x in args.scaffolds.split(",") if x.strip()]
        checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
        checkpoints = sorted(set(x for x in checkpoints if 1 <= x <= args.max_new_tokens))
        if args.max_new_tokens not in checkpoints:
            checkpoints.append(args.max_new_tokens)
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: policy gate scaffold audit, windows={windows}, alpha={alpha}, scaffolds={scaffolds}")

        candidates = build_candidates(args.train_n)
        source_prompts = build_scaffold_prompts(args.test_n, scaffolds)

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
            for source_pair, by_scaffold in source_prompts.items():
                groups = token_groups(tokenizer, source_pair)
                audit[win_name]["sources"][source_pair] = {}
                for scaffold, prompts in by_scaffold.items():
                    row = {}
                    for condition in CONDITIONS:
                        row[condition] = greedy_probe_scaffold(
                            model, tokenizer, device, layers, prompts,
                            interventions_for(components_by_layer, source_pair, window, condition, alpha),
                            groups, args.max_new_tokens, checkpoints, args.batch_size, args.max_length,
                        )
                    audit[win_name]["sources"][source_pair][scaffold] = row
                    base = row["baseline"]["hit_rates"]["target"]
                    rp = row["residual_parallel"]["hit_rates"]["target"]
                    log(f"    {win_name} {source_pair} {scaffold}: base={base:.2f} parallel={rp:.2f}")

        return {
            "phase": 543,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "conditions": CONDITIONS,
            "scaffolds": scaffolds,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "max_new_tokens": args.max_new_tokens,
            "checkpoints": checkpoints,
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
    parser.add_argument("--scaffolds", default="direct,one_word,choose_pair,label_only")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--checkpoints", default="1,3,5,10")
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
    out_path = out_dir / f"phase543_{args.model}_policy_gate_scaffold_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
