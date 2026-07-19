#!/usr/bin/env python3
"""
Phase 971: 协议场定位与语义路径-停止联合图谱
=============================================
Phase 970证明: attention head只覆盖7% gap, 外部b是唯一停止驱动.
Phase 971 转向: MLP channel搜索 + gap分解 + 协议场定位.

Task 1: Logit lens gap轨迹 (每层gap, 找gap在哪里建立)
Task 2: MLP层级DLA (每层MLP对gap的直接贡献)
Task 3: MLP channel搜索 (解析预筛+前向差分验证, 找Δgap<-1的channel)
Task 4: 组件对比 (attention 7% vs MLP ?%)
Task 5: 语义失败分类 (Phase 970的19个失败)
Task 6: 最佳MLP channel + adaptive b联合测试
"""

from __future__ import annotations
import gc, json, sys, time, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from transformers import LogitsProcessor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log, register_multi_head_ablation, get_boundary_ids
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50
from phase970_completion_audit import (
    SAFE_HEADS, SEMANTIC_EQUIV_V5, SPECIAL_TOKENS, GARBAGE_PATTERNS,
    BoundaryDynamicProcessor, evaluate_clean_v5, measure_gap,
    generate_with_processor, EN_PROMPTS_65,
)

PHASE = 971
RESULT_DIR = Path("results/phase971_protocol_field")


# ============================================================
# UTILITIES
# ============================================================
def get_final_layer_norm(model):
    """Get the final layer norm module (compatible across architectures)."""
    if hasattr(model, "model"):
        m = model.model
        for name in ["final_layer_norm", "norm", "ln_f"]:
            if hasattr(m, name):
                return getattr(m, name)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise ValueError(f"Cannot find final layer norm in {type(model).__name__}")


def get_mlp_down_proj(layer):
    """Get the down_proj weight [d_model, intermediate_size] of a layer's MLP."""
    mlp = layer.mlp
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj.weight  # [d_model, intermediate_size]
    if hasattr(mlp, "dense_4h_to_h"):
        return mlp.dense_4h_to_h.weight
    raise ValueError("Cannot find down_proj in MLP")


def get_intermediate_size(model, layers):
    """Get the MLP intermediate size."""
    w = get_mlp_down_proj(layers[0])
    return w.shape[1]


def make_mlp_capture_hook(captured, key):
    """Capture MLP output (post down_proj)."""
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured[key] = out.detach()
    return hook


def make_intermediate_capture_hook(captured, key):
    """Capture the intermediate activation (input to down_proj, post-activation)."""
    def hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        captured[key] = inp.detach()
        return None  # don't modify
    return hook


def make_channel_ablate_hook(channel, scale=0.0):
    """Ablate a single MLP channel at the last position (for forward difference)."""
    def hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            patched[:, -1, channel] = 0.0 if scale == 0.0 else patched[:, -1, channel] * scale
        return (patched,)
    return hook


# ============================================================
# TASK 1: Logit Lens Gap Trajectory
# ============================================================
def task1_logit_lens_gap(model, tokenizer, device, layers, eos_id, prompts, n_prompts=5):
    """Measure gap at each layer using logit lens (project hidden states through ln_f + W_U)."""
    log(f"  Task 1: Logit lens gap trajectory ({n_prompts} prompts)...")
    ln_f = get_final_layer_norm(model)
    W_U = model.lm_head.weight.detach().float()  # [vocab, d_model]
    n_layers = len(layers)

    all_gaps = []  # [n_prompts, n_layers+1]
    t0 = time.time()
    for pi, prompt in enumerate(prompts[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, seq, d_model], len=n_layers+1

        gaps_per_layer = []
        for L, hs in enumerate(hidden_states):
            hs_last = hs[0, -1:, :].float()  # [1, d_model]
            with torch.no_grad():
                hs_normed = ln_f(hs_last)
                logits = hs_normed @ W_U.T  # [1, vocab]
            logit_np = logits[0].cpu().numpy()
            top1 = float(np.sort(logit_np)[-1])
            eos = float(logit_np[eos_id]) if eos_id else 0
            gap = top1 - eos
            gaps_per_layer.append(gap)
        all_gaps.append(gaps_per_layer)
        log(f"    p{pi}: gap L0={gaps_per_layer[0]:.2f} L10={gaps_per_layer[min(10,len(gaps_per_layer)-1)]:.2f} "
            f"L{n_layers}={gaps_per_layer[-1]:.2f}")

    mean_gaps = [float(np.mean([g[L] for g in all_gaps])) for L in range(n_layers+1)]
    # Find where gap builds up: compute Δgap between consecutive layers
    delta_gaps = [mean_gaps[i+1] - mean_gaps[i] for i in range(n_layers)]
    # Top 5 gap-building layers
    top_build = sorted(range(n_layers), key=lambda i: -delta_gaps[i])[:5]

    log(f"    Gap trajectory (mean of {n_prompts} prompts):")
    for L in range(0, n_layers+1, 5):
        log(f"      L{L:2d}: gap={mean_gaps[L]:.3f}")
    log(f"    Top 5 gap-building layers: {[(f'L{L}', f'+{delta_gaps[L]:.3f}') for L in top_build]}")
    log(f"    Task 1 done ({time.time()-t0:.0f}s)")

    return {"mean_gaps_per_layer": mean_gaps, "delta_gaps_per_layer": delta_gaps,
            "top_build_layers": top_build, "all_gaps": all_gaps,
            "prompts": prompts[:n_prompts]}


# ============================================================
# TASK 2: MLP Layer-Level DLA (Direct Logit Attribution)
# ============================================================
def task2_mlp_layer_dla(model, tokenizer, device, layers, eos_id, prompts, n_prompts=5):
    """Measure each MLP layer's direct contribution to gap.
    
    For each layer L:
      - Capture MLP output (post down_proj)
      - Direct contribution to top1 logit: mlp_out @ W_U[top1]
      - Direct contribution to EOS logit: mlp_out @ W_U[EOS]
      - gap_contribution = top1_contrib - eos_contrib
      - Positive = MLP pushes top1 up relative to EOS (increases gap)
      - Negative = MLP pushes EOS up relative to top1 (reduces gap)
    """
    log(f"  Task 2: MLP layer-level DLA ({n_prompts} prompts)...")
    n_layers = len(layers)
    W_U = model.lm_head.weight.detach().float()  # [vocab, d_model]

    all_contribs = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Get top1 token id first
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].float()
        top1_id = int(base_logits.argmax().item())
        eos_logit = float(base_logits[eos_id]) if eos_id else 0
        top1_logit = float(base_logits[top1_id])
        base_gap = top1_logit - eos_logit

        # Capture MLP outputs
        mlp_captured = {}
        handles = []
        for L in range(n_layers):
            h = layers[L].mlp.register_forward_hook(make_mlp_capture_hook(mlp_captured, L))
            handles.append(h)

        with torch.no_grad():
            _ = model(input_ids, use_cache=False)

        for h in handles:
            h.remove()

        # Compute DLA per layer
        w_top1 = W_U[top1_id].cpu()  # [d_model]
        w_eos = W_U[eos_id].cpu() if eos_id else torch.zeros_like(w_top1)
        gap_dir = (w_top1 - w_eos)  # [d_model]

        contribs = []
        for L in range(n_layers):
            mlp_out = mlp_captured.get(L)
            if mlp_out is None:
                contribs.append(0.0)
                continue
            mlp_last = mlp_out[0, -1, :].float().cpu()  # [d_model]
            gap_contrib = float(torch.dot(mlp_last, gap_dir))
            top1_contrib = float(torch.dot(mlp_last, w_top1))
            eos_contrib = float(torch.dot(mlp_last, w_eos))
            contribs.append({"gap_contrib": gap_contrib,
                            "top1_contrib": top1_contrib, "eos_contrib": eos_contrib})

        all_contribs.append({"prompt": prompt, "base_gap": base_gap,
                              "top1_id": top1_id, "contribs": contribs})
        log(f"    p{pi}: base_gap={base_gap:.2f} top1_id={top1_id}")

    # Aggregate
    mean_gap_contrib = [float(np.mean([a["contribs"][L]["gap_contrib"] for a in all_contribs]))
                         for L in range(n_layers)]
    mean_top1_contrib = [float(np.mean([a["contribs"][L]["top1_contrib"] for a in all_contribs]))
                          for L in range(n_layers)]
    mean_eos_contrib = [float(np.mean([a["contribs"][L]["eos_contrib"] for a in all_contribs]))
                         for L in range(n_layers)]

    # Find layers with most positive (gap-building) and most negative (gap-reducing) contributions
    sorted_by_gap = sorted(range(n_layers), key=lambda L: -mean_gap_contrib[L])
    log(f"    Top 5 gap-building MLP layers (positive = increases gap):")
    for L in sorted_by_gap[:5]:
        log(f"      L{L}: gap_contrib={mean_gap_contrib[L]:+.3f}  "
            f"top1={mean_top1_contrib[L]:+.3f}  eos={mean_eos_contrib[L]:+.3f}")
    log(f"    Top 5 gap-reducing MLP layers (negative = reduces gap):")
    for L in sorted_by_gap[-5:][::-1]:
        log(f"      L{L}: gap_contrib={mean_gap_contrib[L]:+.3f}  "
            f"top1={mean_top1_contrib[L]:+.3f}  eos={mean_eos_contrib[L]:+.3f}")

    # Total MLP contribution vs base gap
    total_mlp_gap = sum(mean_gap_contrib)
    log(f"    Total MLP gap contribution: {total_mlp_gap:+.3f} (base gap ~{all_contribs[0]['base_gap']:.2f})")
    log(f"    Task 2 done ({time.time()-t0:.0f}s)")

    return {"mean_gap_contrib": mean_gap_contrib, "mean_top1_contrib": mean_top1_contrib,
            "mean_eos_contrib": mean_eos_contrib, "all_contribs": all_contribs,
            "total_mlp_gap": total_mlp_gap}


# ============================================================
# TASK 3: MLP Channel Search (Analytic + Forward Verification)
# ============================================================
def task3_mlp_channel_search(model, tokenizer, device, layers, eos_id, info,
                              prompts, n_prompts=3, n_top_channels=50, search_layers=None):
    """Search MLP channels that reduce gap.
    
    Stage 1 (Analytic): For each channel, compute gap contribution using captured intermediates.
      gap_contrib_c = intermediate[c] * (W_down[:,c] @ (W_U[top1] - W_U[eos]))
      Fast: O(intermediate_size * d_model) per layer per prompt.
    
    Stage 2 (Forward verification): Ablate top candidates, measure actual Δgap.
    """
    n_layers = len(layers)
    if search_layers is None:
        search_layers = list(range(max(0, n_layers - 8), n_layers))  # last 8 layers
    intermediate_size = get_intermediate_size(model, layers)
    W_U = model.lm_head.weight.detach().float().cpu()  # [vocab, d_model]

    log(f"  Task 3: MLP channel search (layers {search_layers[0]}-{search_layers[-1]}, "
        f"intermediate_size={intermediate_size}, {n_prompts} prompts)...")
    log(f"    Stage 1: Analytic pre-screening...")

    t0 = time.time()
    # Stage 1: Analytic pre-screening
    channel_scores = defaultdict(lambda: {"gap_contrib": [], "layer": 0})
    for pi, prompt in enumerate(prompts[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Get top1 token
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].float()
        top1_id = int(base_logits.argmax().item())
        base_gap = float(base_logits[top1_id]) - (float(base_logits[eos_id]) if eos_id else 0)

        # Capture intermediate activations (input to down_proj)
        inter_captured = {}
        handles = []
        for L in search_layers:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(
                make_intermediate_capture_hook(inter_captured, L))
            handles.append(h)

        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        for h in handles:
            h.remove()

        # Compute gap contribution per channel
        w_top1 = W_U[top1_id]  # [d_model]
        w_eos = W_U[eos_id] if eos_id else torch.zeros_like(w_top1)
        gap_dir = (w_top1 - w_eos)  # [d_model]

        for L in search_layers:
            inter = inter_captured.get(L)
            if inter is None:
                continue
            inter_last = inter[0, -1, :].float().cpu()  # [intermediate_size]
            W_down = get_mlp_down_proj(layers[L]).detach().float().cpu()  # [d_model, intermediate_size]
            # gap_contrib_c = inter[c] * (W_down[:,c] @ gap_dir)
            # = inter * (W_down.T @ gap_dir)
            direction = W_down.T @ gap_dir  # [intermediate_size]
            gap_contribs = inter_last * direction  # [intermediate_size]
            for c in range(intermediate_size):
                key = f"L{L}_C{c}"
                channel_scores[key]["gap_contrib"].append(float(gap_contribs[c]))
                channel_scores[key]["layer"] = L

        log(f"    Stage 1: p{pi} done ({time.time()-t0:.0f}s)")

    # Aggregate and sort
    channel_agg = {}
    for key, v in channel_scores.items():
        channel_agg[key] = {
            "mean_gap_contrib": float(np.mean(v["gap_contrib"])),
            "layer": v["layer"],
        }

    # Sort: most negative gap_contrib = best candidates for ablation (reduces gap)
    sorted_channels = sorted(channel_agg.items(), key=lambda x: x[1]["mean_gap_contrib"])
    top_negative = [(k, v) for k, v in sorted_channels if v["mean_gap_contrib"] < -0.05][:n_top_channels]
    top_positive = [(k, v) for k, v in sorted_channels if v["mean_gap_contrib"] > 0.05][:10]

    log(f"    Stage 1 done. Top {len(top_negative)} gap-reducing channels found.")
    for k, v in top_negative[:10]:
        log(f"      {k}: gap_contrib={v['mean_gap_contrib']:+.4f}")

    # Stage 2: Forward difference verification
    log(f"    Stage 2: Forward difference verification (top {min(30, len(top_negative))} channels)...")
    verify_channels = top_negative[:30]
    forward_results = []
    for ci, (key, v) in enumerate(verify_channels):
        parts = key.split("_")
        L = int(parts[0][1:]); C = int(parts[1][1:])
        # Measure actual Δgap from ablation
        delta_gaps = []
        for prompt in prompts[:n_prompts]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                bl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            base_g = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)

            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
            try:
                with torch.no_grad():
                    pl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            except:
                pl = bl.copy()
            h.remove()
            abl_g = float(np.sort(pl)[-1]) - (float(pl[eos_id]) if eos_id else 0)
            # Check if top1 changed
            top1_changed = int(bl.argmax()) != int(pl.argmax())
            delta_gaps.append({"base_gap": base_g, "abl_gap": abl_g,
                               "delta_gap": abl_g - base_g, "top1_changed": top1_changed})

        mean_dg = float(np.mean([d["delta_gap"] for d in delta_gaps]))
        n_changed = sum(d["top1_changed"] for d in delta_gaps)
        forward_results.append({"key": key, "layer": L, "channel": C,
                                "analytic_gap_contrib": v["mean_gap_contrib"],
                                "forward_mean_delta_gap": mean_dg,
                                "top1_changed_count": n_changed,
                                "per_prompt": delta_gaps})
        if (ci+1) % 10 == 0:
            log(f"      {ci+1}/{len(verify_channels)} verified ({time.time()-t0:.0f}s)")

    # Sort by forward delta_gap (most negative = best)
    forward_sorted = sorted(forward_results, key=lambda x: x["forward_mean_delta_gap"])
    log(f"    Top 10 verified gap-reducing channels (forward Δgap):")
    for r in forward_sorted[:10]:
        log(f"      {r['key']}: forward_Δgap={r['forward_mean_delta_gap']:+.4f}  "
            f"analytic={r['analytic_gap_contrib']:+.4f}  top1_changed={r['top1_changed_count']}/{n_prompts}")

    log(f"    Task 3 done ({time.time()-t0:.0f}s)")
    return {"top_negative_analytic": top_negative[:20],
            "top_positive_analytic": top_positive,
            "forward_verified": forward_sorted[:20],
            "all_forward": forward_results,
            "search_layers": search_layers,
            "intermediate_size": intermediate_size}


# ============================================================
# TASK 4: Component Comparison
# ============================================================
def task4_component_comparison(model, tokenizer, device, layers, eos_id,
                                safe_heads, d_head, task3_results, prompts, n_prompts=5):
    """Compare attention head ablation vs MLP channel ablation for gap reduction."""
    log(f"  Task 4: Component comparison (attention vs MLP, {n_prompts} prompts)...")
    t0 = time.time()

    # Get top MLP channels from Task 3
    top_mlp = []
    for r in task3_results.get("forward_verified", [])[:8]:
        if r["top1_changed_count"] == 0:  # only safe channels (no content change)
            top_mlp.append((r["layer"], r["channel"]))

    # Test sets
    test_sets = {
        "attention_4heads": ("attn", safe_heads),
        "mlp_top8_safe": ("mlp", top_mlp[:8]) if len(top_mlp) >= 4 else ("mlp", []),
        "mlp_top4_safe": ("mlp", top_mlp[:4]) if len(top_mlp) >= 2 else ("mlp", []),
    }

    results = {}
    for name, (comp_type, items) in test_sets.items():
        if not items:
            log(f"    {name}: no items to test")
            continue
        gap_reductions = []
        for prompt in prompts[:n_prompts]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                bl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            base_g = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)

            handles = []
            if comp_type == "attn":
                handles = register_multi_head_ablation(layers, items, d_head)
            elif comp_type == "mlp":
                for L, C in items:
                    h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
                    handles.append(h)

            with torch.no_grad():
                pl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            for h in handles:
                h.remove()
            abl_g = float(np.sort(pl)[-1]) - (float(pl[eos_id]) if eos_id else 0)
            gap_reductions.append(base_g - abl_g)

        mean_red = float(np.mean(gap_reductions)) if gap_reductions else 0
        results[name] = {"n_items": len(items), "mean_gap_reduction": mean_red,
                         "gap_reductions": gap_reductions}
        log(f"    {name} ({len(items)} items): mean_gap_reduction={mean_red:.3f}")

    # Combined: attention + MLP
    combined_attn = safe_heads
    combined_mlp = top_mlp[:4]
    if combined_mlp:
        gap_reductions = []
        for prompt in prompts[:n_prompts]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                bl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            base_g = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)

            handles = register_multi_head_ablation(layers, combined_attn, d_head)
            for L, C in combined_mlp:
                h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
                handles.append(h)

            with torch.no_grad():
                pl = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
            for h in handles:
                h.remove()
            abl_g = float(np.sort(pl)[-1]) - (float(pl[eos_id]) if eos_id else 0)
            gap_reductions.append(base_g - abl_g)

        mean_red = float(np.mean(gap_reductions))
        results["combined_attn+mlp"] = {"n_items": len(combined_attn)+len(combined_mlp),
                                         "mean_gap_reduction": mean_red}
        log(f"    combined_attn+mlp ({len(combined_attn)}+{len(combined_mlp)} items): "
            f"mean_gap_reduction={mean_red:.3f}")

    log(f"    Task 4 done ({time.time()-t0:.0f}s)")
    return results


# ============================================================
# TASK 5: Semantic Failure Classification
# ============================================================
def task5_semantic_classification(phase970_path="results/phase970_completion_audit/glm4/task1_large_scale_65p.json"):
    """Classify the 19 GLM4 failures from Phase 970 Task 1."""
    log(f"  Task 5: Semantic failure classification...")
    t0 = time.time()
    try:
        data = json.load(open(phase970_path, encoding="utf-8"))
    except FileNotFoundError:
        log(f"    Phase 970 results not found at {phase970_path}")
        return {"error": "file not found"}

    raw = data["raw_results"]
    failures = [r for r in raw if r["has_eos"] and not r["strict_clean"]]
    no_eos = [r for r in raw if not r["has_eos"]]

    classified = []
    for r in failures:
        prompt = r["prompt"]
        gen = r["generated"]
        has_expected = r.get("has_expected", False)
        has_garbage = r.get("has_garbage", False)
        n_tok = r.get("n_tokens", 0)

        # Classify
        if has_garbage:
            category = "garbage_template"
        elif n_tok >= 25:
            category = "too_long"
        elif not has_expected:
            # Check if the answer is actually reasonable but doesn't match keywords
            equiv = SEMANTIC_EQUIV_V5.get(prompt, [])
            gen_lower = gen.lower()
            # Heuristic: if generated text contains explanation patterns
            explanation_markers = [" is a", " are ", " that ", " which ", " because", " including"]
            has_explanation = any(m in gen_lower for m in explanation_markers)
            if has_explanation and len(gen) > 30:
                category = "eval_too_narrow"  # answer might be reasonable but doesn't match keyword
            else:
                category = "semantic_path_error"
        else:
            category = "other"

        classified.append({"prompt": prompt, "generated": gen[:80],
                           "category": category, "has_expected": has_expected,
                           "n_tokens": n_tok, "has_garbage": has_garbage})

    # Count categories
    cat_counts = defaultdict(int)
    for c in classified:
        cat_counts[c["category"]] += 1

    log(f"    Total failures (EOS but not clean): {len(failures)}")
    log(f"    No EOS: {len(no_eos)}")
    log(f"    Category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        log(f"      {cat}: {count}")
        for c in classified:
            if c["category"] == cat:
                log(f"        '{c['prompt'][:30]}': '{c['generated'][:50]}'")

    log(f"    Task 5 done ({time.time()-t0:.0f}s)")
    return {"classified_failures": classified, "category_counts": dict(cat_counts),
            "n_failures": len(failures), "n_no_eos": len(no_eos)}


# ============================================================
# TASK 6: Joint Test (Best MLP + Adaptive b)
# ============================================================
def task6_joint_test(model, tokenizer, device, layers, eos_id,
                      safe_heads, d_head, boundary_ids, task3_results, prompts, n_prompts=10):
    """Test: best MLP channels + attention safe heads + adaptive b + boundary dynamic."""
    log(f"  Task 6: Joint test (MLP+attn+adaptive b, {n_prompts} prompts)...")
    t0 = time.time()

    # Get top safe MLP channels (top1 unchanged)
    top_mlp = []
    for r in task3_results.get("forward_verified", [])[:6]:
        if r["top1_changed_count"] == 0:
            top_mlp.append((r["layer"], r["channel"]))

    log(f"    Using {len(safe_heads)} attn heads + {len(top_mlp)} MLP channels")

    results = []
    for prompt in prompts[:n_prompts]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Measure gap with combined ablation
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        for L, C in top_mlp:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
            handles.append(h)
        with torch.no_grad():
            al = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
        for h in handles:
            h.remove()
        ablate_gap = float(np.sort(al)[-1]) - (float(al[eos_id]) if eos_id else 0)
        b_adaptive = int(ablate_gap + 2)

        # Generate with combined ablation + adaptive b + boundary
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        for L, C in top_mlp:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
            handles.append(h)
        proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
        gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
        for h in handles:
            h.remove()
        ce = evaluate_clean_v5(prompt, gen, he, ng)
        results.append({"prompt": prompt, "b": b_adaptive, "ablate_gap": ablate_gap,
                        "generated": gen[:60], "has_eos": he, "n_tokens": ng,
                        "strict_clean": ce["strict_clean"]})
        log(f"    '{prompt[:25]}': b={b_adaptive} gap={ablate_gap:.1f} eos={he} clean={ce['strict_clean']} '{gen[:35]}'")

    clean = sum(r["strict_clean"] for r in results)
    eos = sum(r["has_eos"] for r in results)
    log(f"    Joint test: clean={clean}/{len(results)}  eos={eos}/{len(results)}")
    log(f"    Task 6 done ({time.time()-t0:.0f}s)")
    return {"clean_rate": clean/len(results), "eos_rate": eos/len(results),
            "raw_results": results, "top_mlp_used": top_mlp}


# ============================================================
# MAIN RUNNER
# ============================================================
def run_model(model_name: str):
    log(f"\n{'='*60}\nPhase 971: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    eos_id = tokenizer.eos_token_id
    safe_heads = SAFE_HEADS.get(model_name, [])
    boundary_ids = get_boundary_ids(tokenizer)
    intermediate_size = get_intermediate_size(model, layers)
    log(f"  {info.model_class}, {info.n_layers}L, {n_heads}H, d_head={d_head}, "
        f"intermediate={intermediate_size}")

    results = {"model": model_name, "phase": PHASE}
    t_start = time.time()

    # Task 1: Logit lens gap trajectory
    r1 = task1_logit_lens_gap(model, tokenizer, device, layers, eos_id, EN_PROMPTS_65, n_prompts=5)
    results["task1"] = r1
    (model_dir / "task1_logit_lens.json").write_text(
        json.dumps(r1, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 2: MLP layer-level DLA
    r2 = task2_mlp_layer_dla(model, tokenizer, device, layers, eos_id, EN_PROMPTS_65, n_prompts=5)
    results["task2"] = r2
    (model_dir / "task2_mlp_dla.json").write_text(
        json.dumps(r2, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 3: MLP channel search (the key new experiment)
    r3 = task3_mlp_channel_search(model, tokenizer, device, layers, eos_id, info,
                                   EN_PROMPTS_65, n_prompts=3, n_top_channels=50)
    results["task3"] = r3
    (model_dir / "task3_mlp_channel_search.json").write_text(
        json.dumps(r3, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 4: Component comparison
    r4 = task4_component_comparison(model, tokenizer, device, layers, eos_id,
                                    safe_heads, d_head, r3, EN_PROMPTS_65, n_prompts=5)
    results["task4"] = r4
    (model_dir / "task4_component_comparison.json").write_text(
        json.dumps(r4, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 5: Semantic classification (no model needed)
    r5 = task5_semantic_classification()
    results["task5"] = r5
    (model_dir / "task5_semantic_classification.json").write_text(
        json.dumps(r5, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Task 6: Joint test
    r6 = task6_joint_test(model, tokenizer, device, layers, eos_id,
                          safe_heads, d_head, boundary_ids, r3, EN_PROMPTS_65, n_prompts=10)
    results["task6"] = r6
    (model_dir / "task6_joint_test.json").write_text(
        json.dumps(r6, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=logit_lens, 2=mlp_dla, 3=mlp_channel_search, "
        f"4=component_compare, 5=semantic_classify, 6=joint_test")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["glm4", "deepseek7b"]:
            try:
                run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
