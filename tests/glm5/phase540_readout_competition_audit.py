#!/usr/bin/env python3
"""
Phase 540: readout-competition control audit.

Purpose:
  Phase539 showed residual_parallel is very strong in GLM4/DS7B. This phase
  audits whether that effect is a real interface/competition pattern or mostly
  a direct readout token shortcut.
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
from phase536_pair_quality_selectivity import TEMPLATES, CATEGORY_BANK, cat_prompt, encode_batch  # noqa: E402
from phase539_interface_cluster_mechanism import (  # noqa: E402
    CORE_SOURCES,
    PAIR_SPECS,
    build_candidates,
    build_components,
    layer_windows,
    pair_competitors,
    pair_from_task,
    pair_targets,
    task_name,
)
from phase532_multi_seed_controls import normalize  # noqa: E402


OUT_ROOT = Path("results/glm5_phase540_readout_competition_audit")
CONDITIONS = ["residual_perp", "residual_parallel", "residual_full"]
CLUSTER_LABELS = ["vehicle", "furniture", "tool", "clothing"]
OFF_CLUSTER_LABELS = ["fruit", "animal", "vegetable", "object", "thing", "item"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tasks(test_n: int) -> dict[str, list[str]]:
    out = {}
    for pair, (pos_label, _neg_label) in PAIR_SPECS.items():
        for template in TEMPLATES:
            out[task_name(pair, template)] = [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][-test_n:]]
    return out


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
        "off_cluster": label_ids(tokenizer, OFF_CLUSTER_LABELS),
        "object_control": label_ids(tokenizer, ["object", "thing", "item"]),
    }


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


def group_logit_stats(delta_logits: np.ndarray, groups: dict[str, list[int]]) -> dict[str, float]:
    out = {}
    for name, ids in groups.items():
        valid = [i for i in ids if 0 <= i < delta_logits.shape[1]]
        if not valid:
            out[f"{name}_max_delta"] = 0.0
            out[f"{name}_mean_delta"] = 0.0
            continue
        vals = delta_logits[:, valid]
        out[f"{name}_max_delta"] = float(np.mean(np.max(vals, axis=1)))
        out[f"{name}_mean_delta"] = float(np.mean(vals))
    target = out["target_max_delta"]
    competitor = out["competitor_max_delta"]
    cluster_other = out["cluster_other_max_delta"]
    off_cluster = out["off_cluster_max_delta"]
    out["margin_delta"] = float(target - competitor)
    out["competitor_suppression"] = float(-competitor)
    out["suppression_ratio"] = float((-competitor) / (abs(target) + 1e-8))
    out["cluster_selectivity"] = float((target + cluster_other) / 2.0 - off_cluster)
    out["shortcut_index"] = float(target - max(-competitor, cluster_other, off_cluster))
    return out


def scan_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    baseline_logits: np.ndarray,
    groups: dict[str, list[int]],
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alphas: list[float],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    rows = {}
    for alpha in alphas:
        logits = logits_with_interventions(
            model, tokenizer, device, layers, prompts,
            interventions_for(components_by_layer, source_pair, window, condition, alpha),
            batch_size, max_length,
        )
        rows[str(alpha)] = group_logit_stats(logits - baseline_logits, groups)
    best_alpha = max(rows, key=lambda a: rows[a]["margin_delta"])
    best = rows[best_alpha]
    return {"best_alpha": float(best_alpha), **best, "alpha_rows": rows}


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
        log(f"{args.model}: readout competition audit, windows={windows}, alphas={alphas}")

        candidates = build_candidates(args.train_n)
        tasks = build_tasks(args.test_n)

        components_by_layer = {}
        layer_stats = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps = build_components(dirs, W_U, tokenizer, seeds)
            components_by_layer[str(layer_id)] = comps
            layer_stats[str(layer_id)] = {
                pair: {
                    "full_norm": float(np.linalg.norm(comps[pair]["residual_full"])),
                    "perp_norm": float(np.linalg.norm(comps[pair]["residual_perp"])),
                    "parallel_norm": float(np.linalg.norm(comps[pair]["residual_parallel"])),
                }
                for pair in CORE_SOURCES
            }

        baselines = {}
        groups_by_source = {}
        source_prompts = {}
        for source_pair in CORE_SOURCES:
            prompts = []
            for template in TEMPLATES:
                prompts.extend(tasks[task_name(source_pair, template)])
            source_prompts[source_pair] = prompts
            groups_by_source[source_pair] = token_groups(tokenizer, source_pair)
            baselines[source_pair] = logits_with_interventions(
                model, tokenizer, device, layers, prompts, None, args.batch_size, args.max_length
            )

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for source_pair in CORE_SOURCES:
                row = {}
                for condition in CONDITIONS:
                    row[condition] = scan_condition(
                        model, tokenizer, device, layers,
                        source_prompts[source_pair], baselines[source_pair], groups_by_source[source_pair],
                        components_by_layer, source_pair, window, condition, alphas,
                        args.batch_size, args.max_length,
                    )
                audit[win_name]["sources"][source_pair] = row
                rp = row["residual_parallel"]
                pp = row["residual_perp"]
                log(
                    f"    {win_name} {source_pair}: parallel margin={rp['margin_delta']:+.3f} "
                    f"target={rp['target_max_delta']:+.3f} comp={rp['competitor_max_delta']:+.3f} "
                    f"perp margin={pp['margin_delta']:+.3f}"
                )

        return {
            "phase": 540,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "pairs": list(PAIR_SPECS),
            "templates": list(TEMPLATES),
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alphas": alphas,
            "random_seeds": seeds,
            "token_groups": groups_by_source,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "layer_stats": layer_stats,
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
    out_path = out_dir / f"phase540_{args.model}_readout_competition_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
