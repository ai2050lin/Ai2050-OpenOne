#!/usr/bin/env python3
"""
Phase 571: Norm-Weight Dimension Gate and Early Writer Tracing
归一化权重维度门与早期写入者追踪

Exp1: Norm weight dimension contribution — which dims cause the sign flip?
Exp2: Dimension ablation — set w_norm=1 for top-k dims, does margin flip back?
Exp3: Early writer tracing — track gate dims across L12-L22
Exp4: Bottleneck taxonomy — classify 5 category pairs by bottleneck type
Exp5: Cross-model compression — qwen3/ds7b only vehicle_tool + top gate dims

Run:
  python tests/glm5/phase571_norm_weight_dimension_gate.py qwen3 --smoke
  python tests/glm5/phase571_norm_weight_dimension_gate.py qwen3
  python tests/glm5/phase571_norm_weight_dimension_gate.py glm4
  python tests/glm5/phase571_norm_weight_dimension_gate.py deepseek7b
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


OUT_ROOT = Path("results/glm5_phase571_norm_weight_dim_gate")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

# Early writer trace layers (model-specific)
TRACE_LAYERS = {
    "qwen3": [6, 8, 10, 12, 14],
    "glm4": [12, 14, 16, 18, 20, 22, 24],
    "deepseek7b": [10, 12, 14, 16, 18, 20],
}

# Best h_in swap layer from Phase 569
BEST_H_IN_LAYER = {"qwen3": 8, "glm4": 22, "deepseek7b": 20}

# Bottleneck taxonomy pairs (GLM4 only, full set)
BOTTLENECK_PAIRS = ["vehicle_tool", "clothing_tool", "furniture_tool", "animal_tool", "fruit_vegetable"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Capture h_raw, h_norm, and intermediate layer hidden states in one forward pass
# ============================================================================

def capture_hidden_states(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    swap_layer_ids: list[int],
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    trace_layer_ids: list[int],
    max_length: int,
) -> dict[str, Any]:
    """Run one forward pass capturing:
    - h_raw (final layer output before norm)
    - h_norm (final norm output)
    - h_out at each trace layer (for writer tracing)
    Also returns post-norm logits.
    """
    batch_size = len(prompts)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    captured: dict[str, Any] = {}
    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)

    # Hook for final layer output (h_raw)
    final_layer = layers[-1]
    def make_final_hook():
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            p_dev = torch.full((hs.shape[0],), answer_pos, dtype=torch.long, device=hs.device)
            captured["h_raw"] = hs[bidx, p_dev, :].detach().float().cpu()
        return hook

    # Hook for final norm output (h_norm)
    final_norm = model.model.norm
    def make_norm_hook():
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            p_dev = torch.full((hs.shape[0],), answer_pos, dtype=torch.long, device=hs.device)
            captured["h_norm"] = hs[bidx, p_dev, :].detach().float().cpu()
        return hook

    # Install swap hooks FIRST
    swap_handles = p568.install_state_swap_hooks(
        layers, swap_layer_ids, batch_size, answer_pos, donor_caches, swap_spec
    )

    # Then install capture hooks (final layer, norm, and trace layers)
    all_handles = list(swap_handles)
    all_handles.append(final_layer.register_forward_hook(make_final_hook()))
    all_handles.append(final_norm.register_forward_hook(make_norm_hook()))

    for lid in trace_layer_ids:
        def make_trace_hook(layer_id: int):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                p_dev = pos_cpu.to(hs.device)
                captured[f"h_out_L{layer_id}"] = hs[bidx, p_dev, :].detach().float().cpu()
            return hook
        all_handles.append(layers[lid].register_forward_hook(make_trace_hook(lid)))

    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
        post_logits = out.logits[:, answer_pos, :].float().cpu().numpy()
    finally:
        for h in all_handles:
            h.remove()

    return {"captured": captured, "post_logits": post_logits}


# ============================================================================
# Exp1: Norm weight dimension contribution analysis
# ============================================================================

def dimension_contribution_analysis(
    h_raw: np.ndarray,  # [batch, d_model]
    h_norm: np.ndarray,  # [batch, d_model]
    W_U: np.ndarray,    # [vocab, d_model]
    groups: dict[str, list[int]],
    eps: float,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compute per-dimension contribution to margin at raw, rms, norm stages.

    Returns:
        - margins at each stage
        - per-dim contributions
        - top gate dims (largest |delta_contrib|)
        - w_norm derived from h_norm / h_rms
    """
    h_raw_f64 = h_raw.astype(np.float64)
    h_norm_f64 = h_norm.astype(np.float64)
    batch_size = h_raw_f64.shape[0]

    # RMS normalization
    variance = np.mean(h_raw_f64 ** 2, axis=-1, keepdims=True)
    h_rms = h_raw_f64 / np.sqrt(variance + eps)

    # Derive w_norm = h_norm / h_rms (should be same per batch element)
    # Handle division by near-zero h_rms values
    h_rms_safe = np.where(np.abs(h_rms) < 1e-10, 1.0, h_rms)
    w_norm = (h_norm_f64 / h_rms_safe).mean(0)  # [d_model]
    w_norm = np.nan_to_num(w_norm, nan=1.0, posinf=1.0, neginf=1.0)

    # target-competitor direction
    tgt_ids = [i for i in groups["target"] if 0 <= i < W_U.shape[0]]
    cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U.shape[0]]
    d_TC = W_U[tgt_ids].mean(0).astype(np.float64) - W_U[cmp_ids].mean(0).astype(np.float64)

    # Per-dim contributions (averaged over batch)
    raw_contrib = (h_raw_f64 * d_TC).mean(0)   # [d_model]
    rms_contrib = (h_rms * d_TC).mean(0)
    norm_contrib = (h_norm_f64 * d_TC).mean(0)

    # Delta contributions
    delta_rms_raw = rms_contrib - raw_contrib    # effect of RMS normalization
    delta_norm_rms = norm_contrib - rms_contrib  # effect of Norm weight

    # Total margin at each stage
    # Note: sum of contributions ≈ margin only if d_TC is the exact difference direction
    # Actual margin uses max over group, so we compute directly too
    z_raw = h_raw_f64 @ W_U.T.astype(np.float64)
    z_rms = h_rms @ W_U.T.astype(np.float64)
    z_norm = h_norm_f64 @ W_U.T.astype(np.float64)

    def margin(z):
        stats = [p569.detailed_logit_stats(z[i], groups) for i in range(batch_size)]
        return float(np.mean([s["target_minus_competitor"] for s in stats]))

    m_raw = margin(z_raw)
    m_rms = margin(z_rms)
    m_norm = margin(z_norm)

    # Top gate dims by |delta_norm_rms|
    abs_delta = np.abs(delta_norm_rms)
    top_dims = np.argsort(abs_delta)[::-1][:top_k]

    return {
        "m_raw": m_raw, "m_rms": m_rms, "m_norm": m_norm,
        "w_norm": w_norm,
        "d_TC": d_TC,
        "raw_contrib": raw_contrib,
        "rms_contrib": rms_contrib,
        "norm_contrib": norm_contrib,
        "delta_rms_raw": delta_rms_raw,
        "delta_norm_rms": delta_norm_rms,
        "top_gate_dims": top_dims.tolist(),
        "top_delta_norm_rms": delta_norm_rms[top_dims].tolist(),
        "top_w_norm": w_norm[top_dims].tolist(),
        "top_d_TC": d_TC[top_dims].tolist(),
        "top_raw_contrib": raw_contrib[top_dims].tolist(),
        "top_norm_contrib": norm_contrib[top_dims].tolist(),
    }


# ============================================================================
# Exp2: Dimension ablation — set w_norm=1 for top-k dims
# ============================================================================

def dimension_ablation(
    h_raw: np.ndarray,
    w_norm: np.ndarray,
    W_U: np.ndarray,
    groups: dict[str, list[int]],
    eps: float,
    top_dims: list[int],
    top_k_list: list[int],
) -> dict[str, Any]:
    """Ablate top-k gate dims by setting their w_norm to 1.0, then compute margin.

    Tests if removing the weight effect on these dims reverses the sign flip.
    """
    h_raw_f64 = h_raw.astype(np.float64)
    batch_size = h_raw_f64.shape[0]
    variance = np.mean(h_raw_f64 ** 2, axis=-1, keepdims=True)
    h_rms = h_raw_f64 / np.sqrt(variance + eps)

    results = {}

    # Original (full w_norm)
    h_norm_orig = w_norm * h_rms
    z_orig = h_norm_orig @ W_U.T.astype(np.float64)
    m_orig = float(np.mean([
        p569.detailed_logit_stats(z_orig[i], groups)["target_minus_competitor"]
        for i in range(batch_size)
    ]))
    results["original"] = m_orig

    # RMS only (w_norm=1 for all)
    z_rms = h_rms @ W_U.T.astype(np.float64)
    m_rms = float(np.mean([
        p569.detailed_logit_stats(z_rms[i], groups)["target_minus_competitor"]
        for i in range(batch_size)
    ]))
    results["rms_only"] = m_rms

    # Ablate top-k dims (set w_norm=1 for those dims)
    for k in top_k_list:
        k = min(k, len(top_dims))
        w_ablated = w_norm.copy()
        for d in top_dims[:k]:
            w_ablated[d] = 1.0
        h_norm_abl = w_ablated * h_rms
        z_abl = h_norm_abl @ W_U.T.astype(np.float64)
        m_abl = float(np.mean([
            p569.detailed_logit_stats(z_abl[i], groups)["target_minus_competitor"]
            for i in range(batch_size)
        ]))
        results[f"ablate_top{k}"] = m_abl

    # Invert: keep ONLY top-k dims with original weight, set rest to 1
    for k in top_k_list:
        k = min(k, len(top_dims))
        w_keep = np.ones_like(w_norm)
        for d in top_dims[:k]:
            w_keep[d] = w_norm[d]
        h_norm_keep = w_keep * h_rms
        z_keep = h_norm_keep @ W_U.T.astype(np.float64)
        m_keep = float(np.mean([
            p569.detailed_logit_stats(z_keep[i], groups)["target_minus_competitor"]
            for i in range(batch_size)
        ]))
        results[f"keep_only_top{k}"] = m_keep

    return results


# ============================================================================
# Exp3: Early writer tracing — gate dim activations across layers
# ============================================================================

def early_writer_trace(
    captured: dict[str, torch.Tensor],
    top_dims: list[int],
    W_U: np.ndarray,
    groups: dict[str, list[int]],
    eps: float,
    trace_layer_ids: list[int],
) -> dict[str, Any]:
    """Trace gate dim contributions across layers L12-L22.

    For each trace layer, compute:
    - h_out at that layer
    - h_out projected through W_U → margin at that layer (if it were the final layer)
    - gate dim activations h_out_L[d] for top_dims
    """
    tgt_ids = [i for i in groups["target"] if 0 <= i < W_U.shape[0]]
    cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U.shape[0]]
    d_TC = W_U[tgt_ids].mean(0).astype(np.float64) - W_U[cmp_ids].mean(0).astype(np.float64)

    results: dict[str, Any] = {}

    for lid in trace_layer_ids:
        key = f"h_out_L{lid}"
        if key not in captured:
            continue
        h = captured[key].numpy().astype(np.float64)  # [batch, d_model]
        batch_size = h.shape[0]

        # Margin if this layer's output were directly read out (with RMS norm)
        variance = np.mean(h ** 2, axis=-1, keepdims=True)
        h_rms = h / np.sqrt(variance + eps)
        z = h_rms @ W_U.T.astype(np.float64)
        m_rms = float(np.mean([
            p569.detailed_logit_stats(z[i], groups)["target_minus_competitor"]
            for i in range(batch_size)
        ]))

        # Gate dim activations
        gate_acts = h.mean(0)[top_dims]  # [top_k]

        # Gate dim contributions to d_TC
        gate_contrib = (h.mean(0) * d_TC)[top_dims]

        results[f"L{lid}"] = {
            "margin_rms": m_rms,
            "gate_dim_activations": gate_acts.tolist(),
            "gate_dim_contributions": gate_contrib.tolist(),
            "h_norm_mean": float(np.mean(np.linalg.norm(h, axis=-1))),
        }

    return results


# ============================================================================
# Exp4: Bottleneck taxonomy
# ============================================================================

def bottleneck_taxonomy_entry(
    m_raw: float, m_rms: float, m_norm: float,
    target_rank: float, target_mass: float, entropy: float,
    clean_rate: float, random_margin: float = None,
) -> dict[str, Any]:
    """Classify the bottleneck type for a category pair."""
    entry = {
        "m_raw": m_raw, "m_rms": m_rms, "m_norm": m_norm,
        "target_rank": target_rank, "target_mass": target_mass,
        "entropy": entropy, "clean_rate": clean_rate,
    }

    # A. Norm gate bottleneck: rms positive but norm negative
    if m_rms > 0 and m_norm < 0:
        entry["bottleneck"] = "A_norm_gate"
    # B. Rank/mass bottleneck: margin positive but target mass very low
    elif m_norm > 0 and target_mass < 1e-4:
        entry["bottleneck"] = "B_rank_mass"
    # C. Path bottleneck: next-token strong but clean low
    elif m_norm > 0 and clean_rate < 0.20:
        entry["bottleneck"] = "C_path"
    # D. Semantic alignment: random equals or exceeds real
    elif random_margin is not None and random_margin >= m_norm:
        entry["bottleneck"] = "D_alignment"
    else:
        entry["bottleneck"] = "none"

    return entry


# ============================================================================
# Main per-pair experiment
# ============================================================================

def run_pair_experiment(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    info: Any,
    window: list[int],
    W_U: np.ndarray,
    eps: float,
    pair: str,
    routes: list[dict],
    test_n: int,
    seeds: list[int],
    trace_layer_ids: list[int],
    best_h_in_L: int,
    do_dim_analysis: bool = True,
    do_ablation: bool = True,
    do_writer_trace: bool = True,
    do_generation: bool = True,
    max_length: int = 192,
    max_new_tokens: int = 12,
    temperature: float = 0.8,
    top_p: float = 0.9,
    logit_steps: int = 4,
) -> dict[str, Any]:
    """Run all experiments for one pair."""
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    prompt_sets = p548.build_prompts(pair, test_n, scaffolds)

    all_cache_layers = sorted(set(trace_layer_ids + window + [best_h_in_L]))
    pair_result: dict[str, Any] = {"pair": pair, "routes": {}}

    for route in routes:
        route_name = route["name"]
        prompt_rows = prompt_sets[route["recipient_scaffold"]]
        prompts = [r["prompt"] for r in prompt_rows]

        # Collect donor caches
        donor_scaffold = route["donor_scaffold"]
        donor_objs = CATEGORY_BANK[pos_label][-test_n:]
        repeat_idx = min(4, len(donor_objs) - 1)
        donor_repeat4_prompts = [
            p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)
        ] * test_n
        donor_enc = tokenizer(donor_repeat4_prompts, return_tensors="pt",
                              padding=True, truncation=True, max_length=max_length)
        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
        donor_caches = p568.collect_three_state_cache(
            model, layers, donor_batch, donor_pos, all_cache_layers
        )
        log(f"  [{pair}/{route_name}] Collected donor caches")

        route_result: dict[str, Any] = {}

        # === Exp1+3: Capture hidden states for 3 conditions ===
        conditions_to_capture = [
            ("baseline", "baseline", window, None),
            ("repeat4_h_out", "h_out", window, donor_caches),
            ("best_h_in_swap", "h_in", [best_h_in_L], donor_caches),
        ]

        dim_analyses: dict[str, Any] = {}
        all_top_dims: list[int] = []
        writer_traces: dict[str, Any] = {}

        for cond_name, sw_spec, sw_layers, d_caches in conditions_to_capture:
            log(f"  [{pair}/{route_name}] Capturing {cond_name}...")
            result = capture_hidden_states(
                model, tokenizer, device, layers, prompts,
                sw_layers, sw_spec, d_caches, trace_layer_ids, max_length,
            )
            captured = result["captured"]
            post_logits = result["post_logits"]

            # Compute post-norm margin/rank/mass/entropy from logits
            batch_size = len(prompts)
            stats = [p569.detailed_logit_stats(post_logits[i], groups) for i in range(batch_size)]
            m_post = float(np.mean([s["target_minus_competitor"] for s in stats]))
            rank_post = float(np.mean([s["target_best_rank"] for s in stats]))
            mass_post = float(np.mean([s["target_group_mass"] for s in stats]))
            entropy_post = float(np.mean([s["entropy"] for s in stats]))

            # Exp1: Dimension contribution analysis
            if do_dim_analysis and "h_raw" in captured and "h_norm" in captured:
                h_raw = captured["h_raw"].numpy()
                h_norm = captured["h_norm"].numpy()
                da = dimension_contribution_analysis(h_raw, h_norm, W_U, groups, eps)
                da["m_post"] = m_post
                da["rank_post"] = rank_post
                da["mass_post"] = mass_post
                da["entropy_post"] = entropy_post
                dim_analyses[cond_name] = da
                all_top_dims.extend(da["top_gate_dims"])
                log(f"  [{pair}/{route_name}] {cond_name} dim analysis: "
                    f"raw={da['m_raw']:+.2f} rms={da['m_rms']:+.2f} norm={da['m_norm']:+.2f} "
                    f"post={m_post:+.2f} rank={rank_post:.0f} mass={mass_post:.2e}")

            # Exp3: Early writer trace
            if do_writer_trace:
                # Use top_dims from baseline dim analysis
                top_dims_for_trace = dim_analyses.get("baseline", {}).get("top_gate_dims", list(range(50)))
                wt = early_writer_trace(captured, top_dims_for_trace, W_U, groups, eps, trace_layer_ids)
                writer_traces[cond_name] = wt

        # Deduplicate and sort top dims by frequency
        unique_top_dims = sorted(set(all_top_dims))

        # === Exp2: Dimension ablation on baseline ===
        if do_ablation and "baseline" in dim_analyses:
            baseline_da = dim_analyses["baseline"]
            # Re-capture baseline h_raw for ablation
            result = capture_hidden_states(
                model, tokenizer, device, layers, prompts,
                window, "baseline", None, trace_layer_ids, max_length,
            )
            h_raw_base = result["captured"]["h_raw"].numpy()
            w_norm_base = baseline_da["w_norm"]
            top_dims_base = baseline_da["top_gate_dims"]

            abl = dimension_ablation(
                h_raw_base, w_norm_base, W_U, groups, eps,
                top_dims_base, [5, 10, 20, 50, 100],
            )
            route_result["ablation"] = abl
            log(f"  [{pair}/{route_name}] Ablation: orig={abl['original']:+.2f} "
                f"rms={abl['rms_only']:+.2f} "
                f"abl5={abl.get('ablate_top5',0):+.2f} "
                f"abl20={abl.get('ablate_top20',0):+.2f} "
                f"keep20={abl.get('keep_only_top20',0):+.2f}")

        # === Exp4: Generation for clean rate ===
        if do_generation:
            all_records: dict[str, list[dict[str, Any]]] = {}
            for seed in seeds:
                # baseline generation
                res = p569.generate_with_swap_and_stats(
                    model, tokenizer, device, layers, window, prompts,
                    "baseline", None, groups, route["mode"], seed,
                    max_new_tokens, temperature, top_p, max_length, logit_steps,
                )
                all_records.setdefault("baseline_free", []).extend([
                    {"prompt_index": i, "object": r["object"], "seed": seed,
                     "condition": "baseline_free",
                     **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, r in enumerate(prompt_rows)
                ])

                # repeat4 generation
                res = p569.generate_with_swap_and_stats(
                    model, tokenizer, device, layers, window, prompts,
                    "h_out", donor_caches, groups, route["mode"], seed,
                    max_new_tokens, temperature, top_p, max_length, logit_steps,
                )
                all_records.setdefault("repeat4_free", []).extend([
                    {"prompt_index": i, "object": r["object"], "seed": seed,
                     "condition": "repeat4_free",
                     **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, r in enumerate(prompt_rows)
                ])

            # Aggregate
            gen_stats: dict[str, Any] = {}
            for cond in ["baseline_free", "repeat4_free"]:
                recs = all_records.get(cond, [])
                if not recs:
                    continue
                classified = [
                    {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                    for r in recs
                ]
                agg = p548.aggregate(classified)
                gen_stats[cond] = {
                    "clean_rate": agg["clean_non_object_rate"],
                    "n": len(recs),
                }
            route_result["generation"] = gen_stats
            log(f"  [{pair}/{route_name}] Gen: baseline_clean={gen_stats.get('baseline_free',{}).get('clean_rate',0):.2f} "
                f"repeat4_clean={gen_stats.get('repeat4_free',{}).get('clean_rate',0):.2f}")

        # === Exp4: Bottleneck taxonomy ===
        if "baseline" in dim_analyses:
            da_base = dim_analyses["baseline"]
            gen_clean = route_result.get("generation", {}).get("baseline_free", {}).get("clean_rate", 0)
            # For random margin, use Phase 569 data if available (skip for now)
            taxonomy = bottleneck_taxonomy_entry(
                da_base["m_raw"], da_base["m_rms"], da_base["m_norm"],
                da_base["rank_post"], da_base["mass_post"], da_base["entropy_post"],
                gen_clean,
            )
            route_result["taxonomy"] = taxonomy
            log(f"  [{pair}/{route_name}] Bottleneck: {taxonomy['bottleneck']} "
                f"(rms={taxonomy['m_rms']:+.2f} norm={taxonomy['m_norm']:+.2f} "
                f"clean={taxonomy['clean_rate']:.2f})")

        route_result["dim_analyses"] = dim_analyses
        route_result["writer_traces"] = writer_traces
        pair_result["routes"][route_name] = route_result

    return pair_result


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

        # Get final norm eps
        final_norm = model.model.norm
        eps = getattr(final_norm, 'variance_epsilon', None)
        if eps is None:
            eps = getattr(final_norm, 'eps', None)
        if eps is None:
            eps = getattr(model.config, 'rms_norm_eps', 1e-6)

        trace_layers = [L for L in TRACE_LAYERS[args.model] if 0 <= L < info.n_layers]
        best_h_in_L = BEST_H_IN_LAYER[args.model]

        log(f"{args.model}: phase571 window={window}, trace_layers={trace_layers}, "
            f"eps={eps}, best_h_in_L={best_h_in_L}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase571_{args.model}_checkpoint.json"

        result: dict[str, Any] = {
            "phase": 571, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "window": window, "trace_layers": trace_layers,
            "best_h_in_layer": best_h_in_L, "eps": eps,
            "routes": routes, "sample_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "pairs": {},
        }

        # === Main pair: vehicle_tool (all models) ===
        log(f"=== Main pair: {args.pair} ===")
        main_result = run_pair_experiment(
            model, tokenizer, device, layers, info, window,
            W_U, eps, args.pair, routes, args.test_n, seeds,
            trace_layers, best_h_in_L,
            do_dim_analysis=True, do_ablation=True,
            do_writer_trace=True, do_generation=True,
            max_length=args.max_length, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            logit_steps=args.logit_steps,
        )
        result["pairs"][args.pair] = main_result
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Bottleneck taxonomy (GLM4 only, more pairs) ===
        if args.model == "glm4" and not args.skip_cross_category:
            cross_seeds = parse_int_csv(args.cross_seeds)
            for pair in BOTTLENECK_PAIRS:
                if pair == args.pair:
                    continue
                log(f"=== Bottleneck pair: {pair} ===")
                cross_result = run_pair_experiment(
                    model, tokenizer, device, layers, info, window,
                    W_U, eps, pair, routes, args.test_n, cross_seeds,
                    trace_layers, best_h_in_L,
                    do_dim_analysis=True, do_ablation=False,
                    do_writer_trace=False, do_generation=True,
                    max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    logit_steps=args.logit_steps,
                )
                result["pairs"][pair] = cross_result
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
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--cross-seeds", default="101,103,107,109")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--skip-cross-category", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.test_n = 4
        args.sample_seeds = "101,103"
        args.cross_seeds = "101"
        args.max_new_tokens = 8
        args.logit_steps = 3
        log("SMOKE TEST MODE: test_n=4, seeds=2, max_tokens=8")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase571_{args.model}_norm_weight_dim_gate{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
