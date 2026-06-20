#!/usr/bin/env python3
"""
Phase 558 fast audit: prototype versus object binding by next-token logits.

The generation version is too slow for broad cross-model sweeps because each
donor restore condition doubles autoregressive forwards. This fast audit keeps
the same cache surgery but evaluates the next-token rank/margin in one forward.
It is intended to decide which donor states deserve later generation closure.
"""
from __future__ import annotations

import argparse
import gc
import itertools
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

import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402


OUT_ROOT = Path("results/glm5_phase558_prototype_object_binding_fast")
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DEFAULT_CONDITIONS = [
    "baseline",
    "add_perp",
    "resid_remove_perp",
    "resid_remove_random_perp",
    "resid_donor_vehicle_same_add",
    "resid_donor_vehicle_shuffle_add",
    "resid_donor_vehicle_repeat0_add",
    "resid_donor_vehicle_repeat1_add",
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat3_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_repeat5_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_pca1_cache_add",
    "resid_donor_vehicle_pca3_cache_add",
    "resid_donor_vehicle_random_cache_add",
    "resid_donor_tool_same_add",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_best(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    if not ids:
        return np.full((logits.shape[0],), -1e9, dtype=np.float32)
    return logits[:, ids].max(axis=1)


def group_rank(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    if not ids:
        return np.full((logits.shape[0],), float(logits.shape[1]), dtype=np.float32)
    best = group_best(logits, ids)
    return (logits > best[:, None]).sum(axis=1).astype(np.float32) + 1.0


def summarize_logits(logits: np.ndarray, groups: dict[str, list[int]]) -> dict[str, Any]:
    target_best = group_best(logits, groups["target"])
    competitor_best = group_best(logits, groups["competitor"])
    target_rank = group_rank(logits, groups["target"])
    competitor_rank = group_rank(logits, groups["competitor"])
    greedy = logits.argmax(axis=1).tolist()
    greedy_types = [p544.token_type(int(tok), groups) for tok in greedy]
    n = max(1, len(greedy_types))
    return {
        "n": int(logits.shape[0]),
        "target_margin_mean": float(np.mean(target_best - competitor_best)),
        "target_margin_median": float(np.median(target_best - competitor_best)),
        "target_rank_mean": float(np.mean(target_rank)),
        "competitor_rank_mean": float(np.mean(competitor_rank)),
        "target_top1_rate": float(sum(1 for x in greedy_types if x == "target") / n),
        "competitor_top1_rate": float(sum(1 for x in greedy_types if x == "competitor") / n),
        "other_top1_rate": float(sum(1 for x in greedy_types if x == "other") / n),
    }


def compact_metrics(row: dict[str, Any], base: dict[str, Any], remove: dict[str, Any]) -> dict[str, Any]:
    margin_delta = row["target_margin_mean"] - base["target_margin_mean"]
    rank_delta = row["target_rank_mean"] - base["target_rank_mean"]
    remove_delta = remove["target_margin_mean"] - base["target_margin_mean"]
    restore_gain = row["target_margin_mean"] - remove["target_margin_mean"]
    rank_restore = remove["target_rank_mean"] - row["target_rank_mean"]
    if "_donor_" in row["condition"]:
        if remove_delta <= -0.05 and restore_gain >= 0.10:
            cls = "rank_restore_success"
        elif restore_gain >= 0.10:
            cls = "restore_without_drop"
        else:
            cls = "restore_fail"
    elif margin_delta <= -0.05:
        cls = "rank_necessity_drop"
    elif margin_delta >= 0.10:
        cls = "positive_add_or_release"
    else:
        cls = "flat"
    return {
        "margin_delta": float(margin_delta),
        "rank_delta": float(rank_delta),
        "remove_delta": float(remove_delta),
        "restore_gain": float(restore_gain),
        "rank_restore": float(rank_restore),
        "class": cls,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = p558.parse_csv(args.conditions)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase558 fast expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = p558.combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase558 fast pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                prompts = [r["prompt"] for r in prompt_rows]
                for condition in conditions:
                    plan = p558.condition_plan(condition)
                    donor_rows = p558.donor_rows_for(
                        pair,
                        route["donor_scaffold"],
                        plan.get("donor_category"),
                        plan.get("donor_variant"),
                        args.test_n,
                    )
                    donor_prompts = [r["prompt"] for r in donor_rows] if donor_rows is not None else None
                    logits, donor_logits = p558.batched_next_logits_surgery(
                        model,
                        tokenizer,
                        device,
                        layers,
                        prompts,
                        donor_prompts,
                        components_by_layer,
                        layer_ids,
                        condition,
                        args.batch_size,
                        args.max_length,
                        args.remove_scale,
                        args.add_alpha,
                    )
                    row = summarize_logits(logits, groups)
                    row.update({
                        "condition": condition,
                        "donor_category": plan.get("donor_category") or "",
                        "donor_variant": plan.get("donor_variant") or "",
                    })
                    if donor_logits is not None:
                        donor_row = summarize_logits(donor_logits, groups)
                        row["donor_target_margin_mean"] = donor_row["target_margin_mean"]
                    audit[combo_name]["rows"][route_name][condition] = row
                    log(
                        f"    {combo_name} {route_name} {condition}: "
                        f"margin={row['target_margin_mean']:+.3f}, rank={row['target_rank_mean']:.1f}, "
                        f"top1={row['target_top1_rate']:.2f}"
                    )
                rows = audit[combo_name]["rows"][route_name]
                base = rows["baseline"]
                remove = rows.get("resid_remove_perp", base)
                for condition, row in rows.items():
                    if condition == "baseline":
                        continue
                    compact.append({
                        "combo": combo_name,
                        "layers": layer_ids,
                        "route": route_name,
                        "recipient_scaffold": route["recipient_scaffold"],
                        "donor_scaffold": route["donor_scaffold"],
                        "mode": route["mode"],
                        "condition": condition,
                        "donor_category": row.get("donor_category", ""),
                        "donor_variant": row.get("donor_variant", ""),
                        "base_target_margin_mean": base["target_margin_mean"],
                        "target_margin_mean": row["target_margin_mean"],
                        "target_rank_mean": row["target_rank_mean"],
                        "target_top1_rate": row["target_top1_rate"],
                        **compact_metrics(row, base, remove),
                    })
        return {
            "phase": 558,
            "audit_type": "next_token_rank_margin_fast",
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "compact_rows": compact,
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
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase558_{args.model}_prototype_object_binding_fast.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
