#!/usr/bin/env python3
"""
Phase 572: Gate-Dim Identity and Path Bottleneck Decomposition
门控维度身份与路径瓶颈分解

Exp1: top5 gate dims identity — what tokens do these dims correspond to in W_U?
Exp2: gate dims stability — overlap across conditions and categories
Exp3: signed ablation — separate negative and positive gate dims
Exp4: C_path bottleneck decomposition — test multiple scaffolds on C_path categories
Exp5: path bottleneck repair — scaffold switch effect

Run:
  python tests/glm5/phase572_gate_dim_path_bottleneck.py qwen3 --smoke
  python tests/glm5/phase572_gate_dim_path_bottleneck.py qwen3
  python tests/glm5/phase572_gate_dim_path_bottleneck.py glm4
  python tests/glm5/phase572_gate_dim_path_bottleneck.py deepseek7b
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
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase568_locked_prefix_swap as p568  # noqa: E402
import phase569_pre_layer_source_tracing as p569  # noqa: E402
import phase571_norm_weight_dimension_gate as p571  # noqa: E402


OUT_ROOT = Path("results/glm5_phase572_gate_dim_path")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

# Scaffolds for path bottleneck decomposition
PATH_SCAFFOLDS = [
    "forbidden_sentence_completion",
    "forbidden_definition",
    "sentence_completion",
    "definition",
]

# Pairs for path bottleneck (C_path categories from Phase 571)
PATH_PAIRS = ["clothing_tool", "furniture_tool", "animal_tool", "fruit_vegetable"]
# vehicle_tool is the Norm gate pair — included for comparison
NORM_GATE_PAIR = "vehicle_tool"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Exp1: Gate-dim identity — top tokens from W_U for each gate dim
# ============================================================================

def gate_dim_identity(
    top_dims: list[int],
    w_norm: np.ndarray,
    d_TC: np.ndarray,
    W_U: np.ndarray,
    tokenizer: Any,
    h_raw_baseline: np.ndarray,
    h_raw_repeat4: np.ndarray,
    eps: float,
    top_tokens_k: int = 10,
) -> list[dict[str, Any]]:
    """For each gate dim, compute its identity:
    - dim_id, w_norm value, d_TC value
    - baseline activation, repeat4 activation
    - top tokens from W_U column (what this dim promotes)
    - delta_contrib (norm - rms contribution)
    """
    h_raw_b = h_raw_baseline.astype(np.float64)
    h_raw_r = h_raw_repeat4.astype(np.float64)

    # RMS normalize both
    var_b = np.mean(h_raw_b ** 2, axis=-1, keepdims=True)
    h_rms_b = h_raw_b / np.sqrt(var_b + eps)
    var_r = np.mean(h_raw_r ** 2, axis=-1, keepdims=True)
    h_rms_r = h_raw_r / np.sqrt(var_r + eps)

    # h_norm = w_norm * h_rms (averaged over batch)
    h_norm_b = (w_norm * h_rms_b).mean(0)
    h_rms_b_mean = h_rms_b.mean(0)

    results = []
    for dim in top_dims:
        # Top tokens from W_U column
        col = W_U[:, dim]
        top_idx = np.argsort(col)[::-1][:top_tokens_k]
        top_tok = [tokenizer.decode([int(i)]) for i in top_idx]
        top_val = col[top_idx].tolist()

        # Bottom tokens (negative end)
        bot_idx = np.argsort(col)[:top_tokens_k]
        bot_tok = [tokenizer.decode([int(i)]) for i in bot_idx]
        bot_val = col[bot_idx].tolist()

        # Contributions
        rms_contrib = h_rms_b_mean[dim] * d_TC[dim]
        norm_contrib = h_norm_b[dim] * d_TC[dim]
        delta = norm_contrib - rms_contrib

        results.append({
            "dim_id": int(dim),
            "w_norm": float(w_norm[dim]),
            "d_TC": float(d_TC[dim]),
            "h_baseline_rms": float(h_rms_b_mean[dim]),
            "h_baseline_norm": float(h_norm_b[dim]),
            "rms_contrib": float(rms_contrib),
            "norm_contrib": float(norm_contrib),
            "delta_contrib": float(delta),
            "gate_type": "negative" if delta < 0 else "positive",
            "top_tokens": top_tok,
            "top_token_values": top_val,
            "bottom_tokens": bot_tok,
            "bottom_token_values": bot_val,
        })

    return results


# ============================================================================
# Exp3: Signed ablation — separate negative and positive gate dims
# ============================================================================

def signed_ablation(
    h_raw: np.ndarray,
    w_norm: np.ndarray,
    d_TC: np.ndarray,
    W_U: np.ndarray,
    groups: dict[str, list[int]],
    eps: float,
    top_k: int = 50,
) -> dict[str, Any]:
    """Ablate negative and positive gate dims separately."""
    h_raw_f64 = h_raw.astype(np.float64)
    batch_size = h_raw_f64.shape[0]
    variance = np.mean(h_raw_f64 ** 2, axis=-1, keepdims=True)
    h_rms = h_raw_f64 / np.sqrt(variance + eps)

    # Compute delta_contrib for all dims
    h_rms_mean = h_rms.mean(0)
    h_norm_mean = (w_norm * h_rms).mean(0)
    rms_contrib = h_rms_mean * d_TC
    norm_contrib = h_norm_mean * d_TC
    delta = norm_contrib - rms_contrib  # gate effect per dim

    # Separate by sign
    neg_dims = np.argsort(delta)[:top_k]  # most negative first
    pos_dims = np.argsort(delta)[::-1][:top_k]  # most positive first

    def compute_margin(w_modified):
        h_mod = w_modified * h_rms
        z = h_mod @ W_U.T.astype(np.float64)
        return float(np.mean([
            p569.detailed_logit_stats(z[i], groups)["target_minus_competitor"]
            for i in range(batch_size)
        ]))

    results = {}
    # Original
    results["original"] = compute_margin(w_norm)
    # RMS only
    results["rms_only"] = compute_margin(np.ones_like(w_norm))

    # Ablate negative top-k (set their w=1)
    for k in [5, 10, 20]:
        w_abl = w_norm.copy()
        for d in neg_dims[:k]:
            w_abl[d] = 1.0
        results[f"ablate_neg_top{k}"] = compute_margin(w_abl)

    # Ablate positive top-k
    for k in [5, 10, 20]:
        w_abl = w_norm.copy()
        for d in pos_dims[:k]:
            w_abl[d] = 1.0
        results[f"ablate_pos_top{k}"] = compute_margin(w_abl)

    # Keep only negative top-k (set rest to 1)
    for k in [5, 10, 20]:
        w_keep = np.ones_like(w_norm)
        for d in neg_dims[:k]:
            w_keep[d] = w_norm[d]
        results[f"keep_neg_top{k}"] = compute_margin(w_keep)

    # Keep only positive top-k
    for k in [5, 10, 20]:
        w_keep = np.ones_like(w_norm)
        for d in pos_dims[:k]:
            w_keep[d] = w_norm[d]
        results[f"keep_pos_top{k}"] = compute_margin(w_keep)

    # Ablate both neg+pos top5
    w_both = w_norm.copy()
    for d in list(neg_dims[:5]) + list(pos_dims[:5]):
        w_both[d] = 1.0
    results["ablate_neg5_pos5"] = compute_margin(w_both)

    results["neg_top5_dims"] = neg_dims[:5].tolist()
    results["pos_top5_dims"] = pos_dims[:5].tolist()
    results["neg_top5_delta"] = delta[neg_dims[:5]].tolist()
    results["pos_top5_delta"] = delta[pos_dims[:5]].tolist()

    return results


# ============================================================================
# Exp2: Gate dims overlap across conditions
# ============================================================================

def gate_dims_overlap(
    dim_sets: dict[str, list[int]],
) -> dict[str, Any]:
    """Compute pairwise overlap (Jaccard) between dim sets."""
    import itertools
    keys = list(dim_sets.keys())
    overlap = {}
    for a, b in itertools.combinations(keys, 2):
        sa, sb = set(dim_sets[a]), set(dim_sets[b])
        union = sa | sb
        inter = sa & sb
        jac = len(inter) / len(union) if union else 0.0
        overlap[f"{a}_vs_{b}"] = {
            "jaccard": round(jac, 4),
            "intersection_size": len(inter),
            "union_size": len(union),
            "intersection": sorted(inter),
        }
    return overlap


# ============================================================================
# Exp4+5: Path bottleneck decomposition — test multiple scaffolds
# ============================================================================

def scaffold_prompt_nonforbidden(scaffold: str, obj: str, pos_label: str, neg_label: str) -> str:
    """Non-forbidden scaffold prompts."""
    cap = obj.capitalize()
    if scaffold == "sentence_completion":
        return f"A {obj} is commonly used as"
    if scaffold == "definition":
        return f"A {obj} is best defined as"
    if scaffold == "natural_qa":
        return f"What kind of thing is a {obj}? A {obj} is"
    if scaffold == "direct":
        return f"The category of {obj} is"
    raise ValueError(f"unknown scaffold: {scaffold}")


def run_path_bottleneck(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    W_U: np.ndarray,
    pair: str,
    scaffolds: list[str],
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logit_steps: int,
) -> dict[str, Any]:
    """Test multiple scaffolds for path bottleneck decomposition."""
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    objects = CATEGORY_BANK[pos_label][-test_n:]

    results: dict[str, Any] = {}

    for scaffold in scaffolds:
        # Build prompts for this scaffold
        if scaffold.startswith("forbidden_"):
            prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
        else:
            prompts = [scaffold_prompt_nonforbidden(scaffold, obj, pos_label, neg_label) for obj in objects]

        # Baseline generation (no swap)
        all_records = []
        for seed in seeds:
            res = p569.generate_with_swap_and_stats(
                model, tokenizer, device, layers, window, prompts,
                "baseline", None, groups, "temperature", seed,
                max_new_tokens, temperature, top_p, max_length, logit_steps,
            )
            for i, obj in enumerate(objects):
                all_records.append({
                    "prompt_index": i, "object": obj, "seed": seed,
                    "condition": f"{scaffold}_baseline",
                    **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]},
                })

        # Classify and aggregate
        classified = [
            {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
            for r in all_records
        ]
        agg = p548.aggregate(classified)

        # Step0 metrics
        step0_vals = [r["step_stats"][0] for r in all_records if len(r.get("step_stats", [])) > 0]
        if step0_vals:
            agg["step0_margin"] = float(np.mean([v["target_minus_competitor"] for v in step0_vals]))
            agg["step0_rank"] = float(np.mean([v["target_best_rank"] for v in step0_vals]))
            agg["step0_mass"] = float(np.mean([v["target_group_mass"] for v in step0_vals]))
            agg["step0_entropy"] = float(np.mean([v["entropy"] for v in step0_vals]))

        results[scaffold] = {
            "clean_rate": agg["clean_non_object_rate"],
            "step0_margin": agg.get("step0_margin", 0),
            "step0_rank": agg.get("step0_rank", 0),
            "step0_mass": agg.get("step0_mass", 0),
            "step0_entropy": agg.get("step0_entropy", 0),
            "object_echo_rate": agg.get("object_echo_rate", 0),
            "label_violation_rate": agg.get("any_label_violation_rate", 0),
            "n": len(classified),
        }
        log(f"    {scaffold}: clean={agg['clean_non_object_rate']:.2f}, "
            f"margin={agg.get('step0_margin',0):.2f}, "
            f"rank={agg.get('step0_rank',0):.0f}, "
            f"echo={agg.get('object_echo_rate',0):.2f}")

    return results


# ============================================================================
# Main
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    routes = p558.parse_routes(args.routes)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        _, window = next(iter(windows.items()))
        W_U = get_W_U(model, args.model).astype(np.float32)

        final_norm = model.model.norm
        eps = getattr(final_norm, 'variance_epsilon', None)
        if eps is None:
            eps = getattr(final_norm, 'eps', None)
        if eps is None:
            eps = getattr(model.config, 'rms_norm_eps', 1e-6)

        log(f"{args.model}: phase572 window={window}, eps={eps}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase572_{args.model}_checkpoint.json"

        result: dict[str, Any] = {
            "phase": 572, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "window": window, "eps": eps,
            "routes": routes, "sample_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "gate_dim_identity": {},
            "signed_ablation": {},
            "gate_dims_overlap": {},
            "path_bottleneck": {},
        }

        # === Exp1+3: Gate dim identity and signed ablation for vehicle_tool ===
        log(f"=== Exp1+3: Gate dim identity and signed ablation ({NORM_GATE_PAIR}) ===")
        pair = NORM_GATE_PAIR
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)

        # Collect donor caches
        route = routes[0]
        prompt_rows = prompt_sets[route["recipient_scaffold"]]
        prompts = [r["prompt"] for r in prompt_rows]
        donor_scaffold = route["donor_scaffold"]
        donor_objs = CATEGORY_BANK[pos_label][-args.test_n:]
        repeat_idx = min(4, len(donor_objs) - 1)
        donor_prompts = [
            p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)
        ] * args.test_n
        donor_enc = tokenizer(donor_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
        donor_caches = p568.collect_three_state_cache(model, layers, donor_batch, donor_pos, window)

        # Capture h_raw for baseline and repeat4
        baseline_cap = p571.capture_hidden_states(
            model, tokenizer, device, layers, prompts, window, "baseline", None, [], args.max_length
        )
        repeat4_cap = p571.capture_hidden_states(
            model, tokenizer, device, layers, prompts, window, "h_out", donor_caches, [], args.max_length
        )

        h_raw_base = baseline_cap["captured"]["h_raw"].numpy()
        h_norm_base = baseline_cap["captured"]["h_norm"].numpy()
        h_raw_rep4 = repeat4_cap["captured"]["h_raw"].numpy()

        # Compute w_norm and d_TC
        h_raw_f64 = h_raw_base.astype(np.float64)
        variance = np.mean(h_raw_f64 ** 2, axis=-1, keepdims=True)
        h_rms = h_raw_f64 / np.sqrt(variance + eps)
        h_rms_safe = np.where(np.abs(h_rms) < 1e-10, 1.0, h_rms)
        w_norm = (h_norm_base.astype(np.float64) / h_rms_safe).mean(0)
        w_norm = np.nan_to_num(w_norm, nan=1.0, posinf=1.0, neginf=1.0)

        tgt_ids = [i for i in groups["target"] if 0 <= i < W_U.shape[0]]
        cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U.shape[0]]
        d_TC = W_U[tgt_ids].mean(0).astype(np.float64) - W_U[cmp_ids].mean(0).astype(np.float64)

        # Exp1: Gate dim identity (top 50 by |delta|)
        h_rms_mean = h_rms.mean(0)
        h_norm_mean = (w_norm * h_rms).mean(0)
        rms_contrib = h_rms_mean * d_TC
        norm_contrib = h_norm_mean * d_TC
        delta = norm_contrib - rms_contrib
        top50_dims = np.argsort(np.abs(delta))[::-1][:50].tolist()

        identity = gate_dim_identity(top50_dims, w_norm, d_TC, W_U, tokenizer, h_raw_base, h_raw_rep4, eps)
        result["gate_dim_identity"][pair] = identity
        log(f"  Gate dim identity computed for {len(identity)} dims")
        log(f"  Top 5 negative gate dims: {[r['dim_id'] for r in identity if r['gate_type']=='negative'][:5]}")
        log(f"  Top 5 positive gate dims: {[r['dim_id'] for r in identity if r['gate_type']=='positive'][:5]}")
        for r in identity[:5]:
            log(f"    dim={r['dim_id']} w={r['w_norm']:.3f} d_TC={r['d_TC']:.4f} "
                f"delta={r['delta_contrib']:.4f} type={r['gate_type']} "
                f"top_tokens={r['top_tokens'][:3]}")

        # Exp3: Signed ablation
        abl = signed_ablation(h_raw_base, w_norm, d_TC, W_U, groups, eps)
        result["signed_ablation"][pair] = abl
        log(f"  Signed ablation: orig={abl['original']:+.2f} rms={abl['rms_only']:+.2f} "
            f"abl_neg5={abl['ablate_neg_top5']:+.2f} abl_pos5={abl['ablate_pos_top5']:+.2f} "
            f"keep_neg5={abl['keep_neg_top5']:+.2f} keep_pos5={abl['keep_pos_top5']:+.2f} "
            f"both5={abl['ablate_neg5_pos5']:+.2f}")

        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp2: Gate dims overlap across categories ===
        log(f"=== Exp2: Gate dims overlap across categories ===")
        dim_sets: dict[str, list[int]] = {pair: top50_dims[:20]}

        for cat_pair in [p for p in ["clothing_tool", "furniture_tool", "animal_tool", "fruit_vegetable"] if p != pair]:
            try:
                cp_pos, cp_neg = PAIR_SPECS[cat_pair]
                cp_groups = p544.token_groups(tokenizer, cat_pair)
                cp_prompts_rows = p548.build_prompts(cat_pair, args.test_n, ["forbidden_sentence_completion"])
                cp_prompts = [r["prompt"] for r in cp_prompts_rows["forbidden_sentence_completion"]]

                cp_cap = p571.capture_hidden_states(
                    model, tokenizer, device, layers, cp_prompts, window, "baseline", None, [], args.max_length
                )
                cp_h_raw = cp_cap["captured"]["h_raw"].numpy()
                cp_h_norm = cp_cap["captured"]["h_norm"].numpy()

                cp_h_raw_f64 = cp_h_raw.astype(np.float64)
                cp_var = np.mean(cp_h_raw_f64 ** 2, axis=-1, keepdims=True)
                cp_h_rms = cp_h_raw_f64 / np.sqrt(cp_var + eps)
                cp_h_rms_safe = np.where(np.abs(cp_h_rms) < 1e-10, 1.0, cp_h_rms)
                cp_w_norm = (cp_h_norm.astype(np.float64) / cp_h_rms_safe).mean(0)
                cp_w_norm = np.nan_to_num(cp_w_norm, nan=1.0, posinf=1.0, neginf=1.0)

                cp_tgt = [i for i in cp_groups["target"] if 0 <= i < W_U.shape[0]]
                cp_cmp = [i for i in cp_groups["competitor"] if 0 <= i < W_U.shape[0]]
                cp_d_TC = W_U[cp_tgt].mean(0).astype(np.float64) - W_U[cp_cmp].mean(0).astype(np.float64)

                cp_rms_c = cp_h_rms.mean(0) * cp_d_TC
                cp_norm_c = (cp_w_norm * cp_h_rms).mean(0) * cp_d_TC
                cp_delta = cp_norm_c - cp_rms_c
                cp_top20 = np.argsort(np.abs(cp_delta))[::-1][:20].tolist()
                dim_sets[cat_pair] = cp_top20
                log(f"  {cat_pair} top20 gate dims computed")
            except Exception as e:
                log(f"  {cat_pair} failed: {e}")

        overlap = gate_dims_overlap(dim_sets)
        result["gate_dims_overlap"] = overlap
        for key, val in overlap.items():
            log(f"  {key}: jaccard={val['jaccard']:.4f} inter={val['intersection_size']}/{val['union_size']}")

        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp4+5: Path bottleneck decomposition ===
        log(f"=== Exp4+5: Path bottleneck decomposition ===")
        path_pairs = [NORM_GATE_PAIR] + [p for p in PATH_PAIRS if p != NORM_GATE_PAIR]
        for p in path_pairs:
            log(f"  Path bottleneck for {p}:")
            pb = run_path_bottleneck(
                model, tokenizer, device, layers, window, W_U, p,
                PATH_SCAFFOLDS, args.test_n, seeds,
                args.max_length, args.max_new_tokens,
                args.temperature, args.top_p, args.logit_steps,
            )
            result["path_bottleneck"][p] = pb
            checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")

        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.test_n = 4
        args.sample_seeds = "101,103"
        args.max_new_tokens = 8
        args.logit_steps = 3
        log("SMOKE TEST MODE: test_n=4, seeds=2, max_tokens=8")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase572_{args.model}_gate_dim_path{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
