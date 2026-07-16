#!/usr/bin/env python3
"""
Phase 952: Protocol Token-Specific Trajectory and Causal Channel Audit
=====================================================================
协议词元特异轨迹与因果通道审计

Task1+2: 个体protocol token响应 × beta扫描 (合并)
Task3:   Protocol场逐层轨迹图谱 (per-token)
Task4:   200+ prompts交叉验证通道归因
Task5:   协议通道因果干预 (条件执行)

设计原则:
  - 个体token: 不取平均, 每个protocol token单独记录
  - beta扫描: 0.5, 1, 2, 4, 8 五个强度
  - 增量保存: 每个Task完成后立即写JSON
  - 跨模型兼容: split_gate_up / merged_gate_up
"""

from __future__ import annotations
import argparse, gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS
from phase951_protocol_atlas import (
    ALL_PROMPT_GROUPS, get_protocol_token_ids, _get_mlp_activation_hook,
    safe_decode, COLOR_WORDS, ensure_dir, PROTOCOL_NEUTRAL_PROMPTS
)

PHASE = 952
RESULT_DIR = Path("results/phase952_token_trajectory")

# Individual protocol tokens to track (NOT averaged)
INDIVIDUAL_PROTOCOL_TOKENS = [
    ".", " .", " ", "\n", "the", " The", "is", " a",
    "Answer", "Solution", "Step", "1",
]

# Beta values for scanning
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]

# Color direction pairs
COLOR_PAIRS = [("red", "blue"), ("yellow", "black"), ("white", "black")]

# Prompts for injection (color-relevant)
INJECTION_PROMPTS = [
    "The apple is", "The sky is", "The grass is",
    "The sun is", "The night is", "The snow is",
    "The fire is", "The flower is", "The ocean is",
    "The lemon is", "The rose is", "The coal is",
    "The cloud is", "The wood is", "The wine is",
]

# Prompts for trajectory (mix of color and non-color)
TRAJECTORY_PROMPTS = [
    "The apple is red.",
    "The sky is blue.",
    "The grass is green.",
    "The sun is yellow.",
    "The night is black.",
    "The snow is white.",
    "The fire is orange.",
    "The flower is purple.",
    "The apple is sweet.",
    "The sky is clear.",
    "The grass is tall.",
    "The sun is bright.",
    "The night is quiet.",
    "The snow is cold.",
    "The fire is hot.",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# TASK 1+2: Token-Specific Response × Beta Scan
# ============================================================

def task1_2_token_beta_scan(model, tokenizer, device, info, model_name: str,
                             save_dir: Path) -> dict:
    """For each protocol token, test each beta × color direction."""
    log("  Task 1+2: Token-specific response x beta scan...")

    W_U = get_W_U(model, model_name)

    # Get color directions
    color_dirs = {}
    for c1, c2 in COLOR_PAIRS:
        ids1 = tokenizer.encode(c1, add_special_tokens=False)
        ids2 = tokenizer.encode(c2, add_special_tokens=False)
        if ids1 and ids2 and ids1[0] < W_U.shape[0] and ids2[0] < W_U.shape[0]:
            d = W_U[ids1[0]] - W_U[ids2[0]]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                color_dirs[f"{c1}-{c2}"] = d / norm

    # Get protocol token IDs
    proto_ids = {}
    for pt in INDIVIDUAL_PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]

    # Also get EOS
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    log(f"    Color directions: {list(color_dirs.keys())}")
    log(f"    Protocol tokens: {len(proto_ids)}")
    log(f"    Betas: {BETAS}")

    embed = model.get_input_embeddings()
    results = []

    for pi, prompt in enumerate(INJECTION_PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline logits
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for dname, direction in color_dirs.items():
            dir_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

            for beta in BETAS:
                # Inject at embedding level
                inputs_embeds = embed(input_ids).detach().clone()
                inputs_embeds[0, -1, :] += (beta * dir_tensor).to(inputs_embeds.dtype)

                with torch.no_grad():
                    inj_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                    inj_logits = inj_out.logits[0, -1].detach().float().cpu().numpy()

                # Record per-token delta
                token_deltas = {}
                for pt, tid in proto_ids.items():
                    if tid < len(inj_logits):
                        token_deltas[pt] = float(inj_logits[tid] - base_logits[tid])

                # Also record color token changes
                color_deltas = {}
                for cw in COLOR_WORDS:
                    ids = tokenizer.encode(cw, add_special_tokens=False)
                    if ids and ids[0] < len(inj_logits):
                        color_deltas[cw] = float(inj_logits[ids[0]] - base_logits[ids[0]])

                results.append({
                    "prompt": prompt,
                    "direction": dname,
                    "beta": beta,
                    "token_deltas": token_deltas,
                    "color_deltas": color_deltas,
                })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(INJECTION_PROMPTS)} prompts")

    # Aggregate: per-token × per-beta × per-direction
    agg = {}
    for pt in list(proto_ids.keys()) + COLOR_WORDS:
        agg[pt] = {}
        for dname in color_dirs:
            agg[pt][dname] = {}
            for beta in BETAS:
                vals = []
                for r in results:
                    if r["direction"] == dname and r["beta"] == beta:
                        if pt in r["token_deltas"]:
                            vals.append(r["token_deltas"][pt])
                        elif pt in r["color_deltas"]:
                            vals.append(r["color_deltas"][pt])
                if vals:
                    agg[pt][dname][beta] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "n": len(vals),
                    }

    output = {
        "task": "task1_2_token_beta_scan",
        "model": model_name,
        "n_prompts": len(INJECTION_PROMPTS),
        "n_tokens": len(proto_ids),
        "n_directions": len(color_dirs),
        "betas": BETAS,
        "protocol_tokens": list(proto_ids.keys()),
        "color_pairs": [f"{c1}-{c2}" for c1, c2 in COLOR_PAIRS],
        "aggregated": agg,
        "raw_results": results[:30],  # First 30 for inspection
    }

    save_path = save_dir / "task1_2_token_beta.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key per-token results at beta=4.0
    log(f"    Per-token delta at beta=4.0 (red-blue direction):")
    for pt in list(proto_ids.keys()):
        d = agg.get(pt, {}).get("red-blue", {}).get(4.0, {})
        if d:
            log(f"      {pt:12s}: mean={d['mean']:+.4f} std={d['std']:.4f} range=[{d['min']:+.4f}, {d['max']:+.4f}]")

    return output


# ============================================================
# TASK 3: Per-Token Protocol Trajectory Atlas
# ============================================================

def task3_trajectory(model, tokenizer, device, info, model_name: str,
                     save_dir: Path) -> dict:
    """Record per-layer projection for each protocol token direction."""
    log("  Task 3: Per-token protocol trajectory atlas...")

    n_layers = info.n_layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    W_U = get_W_U(model, model_name)

    # Get protocol token directions (normalized W_U rows)
    proto_dirs = {}
    for pt in INDIVIDUAL_PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids and ids[0] < W_U.shape[0]:
            d = W_U[ids[0]].copy()
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                proto_dirs[pt] = d / norm

    # Color direction
    red_ids = tokenizer.encode("red", add_special_tokens=False)
    blue_ids = tokenizer.encode("blue", add_special_tokens=False)
    color_dir = None
    if red_ids and blue_ids:
        d = W_U[red_ids[0]] - W_U[blue_ids[0]]
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            color_dir = d / norm

    # EOS direction
    eos_dir = None
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < W_U.shape[0]:
        d = W_U[tokenizer.eos_token_id].copy()
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            eos_dir = d / norm
            proto_dirs["<EOS>"] = eos_dir

    layers_list = get_layers(model)
    results = []

    for pi, prompt in enumerate(TRAJECTORY_PROMPTS):
        has_color = any(cw in prompt for cw in COLOR_WORDS)

        captured = {}
        hooks = []

        def make_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
                else:
                    captured[li] = output[0, -1, :].detach().float().cpu().numpy()
            return hook

        for li in sample_layers:
            h = layers_list[li].register_forward_hook(make_hook(li))
            hooks.append(h)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)

        for h in hooks:
            h.remove()

        # Compute per-layer projections
        layer_proj = {}
        for li in sample_layers:
            h_state = captured.get(li)
            if h_state is None:
                continue
            h_norm = max(np.linalg.norm(h_state), 1e-10)

            token_projs = {}
            for pt, d in proto_dirs.items():
                token_projs[pt] = float(np.dot(h_state, d) / h_norm)

            color_proj = 0.0
            if color_dir is not None:
                color_proj = float(np.dot(h_state, color_dir) / h_norm)

            # Also compute logit (W_U[p] @ h without normalization)
            token_logits = {}
            for pt, d in proto_dirs.items():
                # Find the token ID
                ids = tokenizer.encode(pt, add_special_tokens=False)
                if ids and ids[0] < W_U.shape[0]:
                    token_logits[pt] = float(np.dot(W_U[ids[0]], h_state))
                elif pt == "<EOS>" and tokenizer.eos_token_id is not None:
                    token_logits[pt] = float(np.dot(W_U[tokenizer.eos_token_id], h_state))

            layer_proj[f"L{li}"] = {
                "cos_projections": token_projs,
                "logit_projections": token_logits,
                "color_cos": color_proj,
            }

        results.append({
            "prompt": prompt,
            "has_color": has_color,
            "layer_projections": layer_proj,
        })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(TRAJECTORY_PROMPTS)} prompts")

    # Aggregate per-layer
    layer_agg = {}
    for li in sample_layers:
        key = f"L{li}"
        color_vals = []
        nocolor_vals = []
        token_cos = defaultdict(lambda: {"color": [], "nocolor": []})
        token_logit = defaultdict(lambda: {"color": [], "nocolor": []})

        for r in results:
            lp = r["layer_projections"].get(key)
            if not lp:
                continue
            if r["has_color"]:
                color_vals.append(lp["color_cos"])
                for pt, v in lp["cos_projections"].items():
                    token_cos[pt]["color"].append(v)
                for pt, v in lp["logit_projections"].items():
                    token_logit[pt]["color"].append(v)
            else:
                nocolor_vals.append(lp["color_cos"])
                for pt, v in lp["cos_projections"].items():
                    token_cos[pt]["nocolor"].append(v)
                for pt, v in lp["logit_projections"].items():
                    token_logit[pt]["nocolor"].append(v)

        layer_agg[key] = {
            "color_cos_color": float(np.mean(color_vals)) if color_vals else 0,
            "color_cos_nocolor": float(np.mean(nocolor_vals)) if nocolor_vals else 0,
            "color_diff": (float(np.mean(color_vals)) - float(np.mean(nocolor_vals)))
                          if color_vals and nocolor_vals else 0,
            "token_cos": {
                pt: {
                    "color_mean": float(np.mean(v["color"])) if v["color"] else 0,
                    "nocolor_mean": float(np.mean(v["nocolor"])) if v["nocolor"] else 0,
                    "diff": (float(np.mean(v["color"])) - float(np.mean(v["nocolor"])))
                            if v["color"] and v["nocolor"] else 0,
                }
                for pt, v in token_cos.items()
            },
            "token_logit": {
                pt: {
                    "color_mean": float(np.mean(v["color"])) if v["color"] else 0,
                    "nocolor_mean": float(np.mean(v["nocolor"])) if v["nocolor"] else 0,
                    "diff": (float(np.mean(v["color"])) - float(np.mean(v["nocolor"])))
                            if v["color"] and v["nocolor"] else 0,
                }
                for pt, v in token_logit.items()
            },
        }

    output = {
        "task": "task3_trajectory",
        "model": model_name,
        "n_prompts": len(TRAJECTORY_PROMPTS),
        "n_color": sum(1 for r in results if r["has_color"]),
        "n_nocolor": sum(1 for r in results if not r["has_color"]),
        "sample_layers": sample_layers,
        "protocol_tokens": list(proto_dirs.keys()),
        "layer_aggregation": layer_agg,
        "raw_results": results[:5],
    }

    save_path = save_dir / "task3_trajectory.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key trajectory for period and space
    log(f"    Trajectory (cos projection, avg of color+nocolor):")
    for key in sorted(layer_agg.keys(), key=lambda x: int(x[1:])):
        la = layer_agg[key]
        period_cos = la["token_cos"].get(".", {})
        space_cos = la["token_cos"].get(" ", {})
        eos_cos = la["token_cos"].get("<EOS>", {})
        p_val = period_cos.get("color_mean", 0)  # Just use color for display
        s_val = space_cos.get("color_mean", 0)
        e_val = eos_cos.get("color_mean", 0)
        log(f"      {key:5s}: period={p_val:+.4f}  space={s_val:+.4f}  EOS={e_val:+.4f}")

    return output


# ============================================================
# TASK 4: CV-Based Channel Attribution (200 prompts)
# ============================================================

def task4_cv_attribution(model, tokenizer, device, info, model_name: str,
                         protocol_tokens: dict, save_dir: Path,
                         n_prompts: int = 200) -> dict:
    """CV-based channel attribution with more data."""
    log(f"  Task 4: CV channel attribution ({n_prompts} prompts)...")

    all_prompts = []
    for gname, prompts in ALL_PROMPT_GROUPS.items():
        all_prompts.extend(prompts)
    # Use only unique prompts to avoid CV data leakage
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    selected = unique_prompts[:n_prompts]
    actual_n = len(selected)
    if actual_n < n_prompts:
        log(f"    Warning: only {actual_n} unique prompts available (requested {n_prompts})")

    n_layers = info.n_layers
    # Test last 3 layers + one mid layer
    target_layers = [n_layers - 1, n_layers - 2, n_layers - 3]
    # Add a mid layer for comparison
    mid_layer = n_layers // 2
    if mid_layer not in target_layers:
        target_layers.append(mid_layer)
    target_layers = sorted(set(target_layers))

    layer_activations = {li: [] for li in target_layers}
    all_logits = []

    for pi, prompt in enumerate(selected):
        hooks_data = {}
        hooks_list = []
        for li in target_layers:
            hs, cap = _get_mlp_activation_hook(model, info, li)
            hooks_data[li] = cap
            hooks_list.extend(hs)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu()

        for h in hooks_list:
            h.remove()

        for li in target_layers:
            act = hooks_data[li].get("act")
            if act is not None:
                layer_activations[li].append(act[0, -1, :].numpy())

        all_logits.append(logits.numpy())

        if (pi + 1) % 50 == 0:
            log(f"    {pi+1}/{n_prompts} prompts")

    all_logits = np.array(all_logits)

    layer_results = {}
    for li in target_layers:
        X = np.array(layer_activations[li])
        if X.shape[0] < 20:
            continue

        token_cv = {}
        cv_r2_list = []

        for ptid_str, ptinfo in list(protocol_tokens.items())[:40]:
            tid = ptinfo["id"]
            if tid >= all_logits.shape[1]:
                continue
            y = all_logits[:, tid]

            ridge = Ridge(alpha=10.0)
            cv_scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')

            ridge.fit(X, y)
            coefs = ridge.coef_
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]

            token_cv[ptid_str] = {
                "str": ptinfo["str"],
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "cv_r2_max": float(np.max(cv_scores)),
                "top_channels": [{"channel": int(i), "coef": float(coefs[i])} for i in top_indices],
            }
            cv_r2_list.append(np.mean(cv_scores))

        layer_results[f"L{li}"] = {
            "n_prompts": X.shape[0],
            "n_intermediate": X.shape[1],
            "mean_cv_r2": float(np.mean(cv_r2_list)),
            "median_cv_r2": float(np.median(cv_r2_list)),
            "max_cv_r2": float(np.max(cv_r2_list)),
            "n_positive_r2": sum(1 for r in cv_r2_list if r > 0),
            "n_tokens_tested": len(cv_r2_list),
            "token_results": token_cv,
        }

        log(f"    L{li}: CV-R2 mean={np.mean(cv_r2_list):.3f}, "
            f"median={np.median(cv_r2_list):.3f}, "
            f"positive={sum(1 for r in cv_r2_list if r > 0)}/{len(cv_r2_list)}")

    output = {
        "task": "task4_cv_attribution",
        "model": model_name,
        "method": "ridge_5fold_cv_alpha10",
        "n_prompts": n_prompts,
        "target_layers": target_layers,
        "layer_results": layer_results,
    }

    save_path = save_dir / "task4_cv.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    return output


# ============================================================
# TASK 5: Causal Channel Intervention (conditional)
# ============================================================

def task5_causal_intervention(model, tokenizer, device, info, model_name: str,
                               task4_result: dict, save_dir: Path,
                               n_prompts: int = 15) -> dict:
    """If Task 4 finds stable channels, do causal intervention."""
    log("  Task 5: Causal channel intervention...")

    # Find best layer and its top channels
    best_layer_key = None
    best_r2 = -999
    for lkey, lr in task4_result.get("layer_results", {}).items():
        if lr["mean_cv_r2"] > best_r2:
            best_r2 = lr["mean_cv_r2"]
            best_layer_key = lkey

    if best_layer_key is None or best_r2 < 0:
        log(f"    No layer with positive CV-R2 (best={best_r2:.3f}), skipping intervention")
        return {"task": "task5", "model": model_name, "skipped": True, "reason": "no_positive_r2"}

    best_layer_idx = int(best_layer_key[1:])
    log(f"    Best layer: {best_layer_key} (CV-R2={best_r2:.3f})")

    # Get top channels for key protocol tokens
    layer_data = task4_result["layer_results"][best_layer_key]
    token_results = layer_data.get("token_results", {})

    # Find channels that appear in multiple tokens' top-10
    channel_token_map = defaultdict(list)
    for ptid_str, tr in token_results.items():
        if tr["cv_r2_mean"] > 0:  # Only consider tokens with positive R2
            for ch in tr["top_channels"][:5]:
                channel_token_map[ch["channel"]].append({
                    "token": tr["str"],
                    "coef": ch["coef"],
                })

    # Channels supporting 2+ tokens
    shared_channels = {ch: toks for ch, toks in channel_token_map.items() if len(toks) >= 2}
    shared_channels = dict(sorted(shared_channels.items(), key=lambda x: -len(x[1])))

    if not shared_channels:
        log(f"    No shared channels found, skipping intervention")
        return {"task": "task5", "model": model_name, "skipped": True, "reason": "no_shared_channels"}

    top_channels = list(shared_channels.keys())[:5]
    log(f"    Top {len(top_channels)} shared channels: {top_channels}")

    # Get protocol token IDs for measurement
    proto_ids = {}
    for pt in INDIVIDUAL_PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Intervention: zero out specific channels in MLP intermediate
    layers_list = get_layers(model)
    target_layer = layers_list[best_layer_idx]

    results = []

    for pi, prompt in enumerate(INJECTION_PROMPTS[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        # For each channel, zero it out
        for ch_id in top_channels:
            # Hook to zero specific channel in down_proj input
            captured = {}

            def make_zero_hook(channel_id):
                def hook(module, inputs, output):
                    # inputs[0] is the activated intermediate [batch, seq, inter]
                    patched = inputs[0].clone()
                    patched[:, -1, channel_id] = 0  # Zero last token's channel
                    return (patched,) + inputs[1:] if len(inputs) > 1 else (patched,)
                return hook

            # We need to modify the down_proj's INPUT, not output
            # Use a forward_pre_hook
            def make_pre_hook(channel_id):
                def pre_hook(module, args):
                    inp = args[0]
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    patched = inp.clone()
                    if patched.ndim >= 3:
                        patched[:, -1, channel_id] = 0
                    elif patched.ndim >= 2:
                        patched[:, channel_id] = 0
                    return (patched,)
                return pre_hook

            handle = target_layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(ch_id))

            with torch.no_grad():
                try:
                    patched_out = model(input_ids=input_ids, use_cache=False)
                    patched_logits = patched_out.logits[0, -1].detach().float().cpu().numpy()
                except Exception:
                    patched_logits = base_logits.copy()

            handle.remove()

            # Record per-token delta
            token_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(patched_logits):
                    token_deltas[pt] = float(patched_logits[tid] - base_logits[tid])

            results.append({
                "prompt": prompt,
                "channel": ch_id,
                "token_deltas": token_deltas,
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{n_prompts} prompts")

    # Aggregate per-channel
    channel_agg = {}
    for ch_id in top_channels:
        ch_results = [r for r in results if r["channel"] == ch_id]
        token_means = {}
        for pt in proto_ids:
            vals = [r["token_deltas"].get(pt, 0) for r in ch_results]
            token_means[pt] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
        channel_agg[str(ch_id)] = {
            "n_prompts": len(ch_results),
            "supported_tokens": [t["token"] for t in shared_channels[ch_id]],
            "token_deltas": token_means,
        }

    output = {
        "task": "task5_causal",
        "model": model_name,
        "best_layer": best_layer_key,
        "best_cv_r2": best_r2,
        "n_channels_tested": len(top_channels),
        "channels": top_channels,
        "channel_results": channel_agg,
    }

    save_path = save_dir / "task5_causal.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print results
    for ch_str, ch_data in channel_agg.items():
        log(f"    Channel {ch_str} (supports {ch_data['supported_tokens'][:3]}):")
        for pt, d in sorted(ch_data["token_deltas"].items(), key=lambda x: abs(x[1]["mean"]), reverse=True)[:5]:
            log(f"      {pt:12s}: delta={d['mean']:+.4f} +/- {d['std']:.4f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str, args: argparse.Namespace) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 952: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    protocol_tokens = get_protocol_token_ids(tokenizer)
    t_start = time.time()

    # Task 1+2: Token-specific × beta scan
    try:
        task1_2_token_beta_scan(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1+2 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 3: Trajectory
    try:
        task3_trajectory(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 4: CV attribution
    task4_result = None
    try:
        task4_result = task4_cv_attribution(model, tokenizer, device, info, model_name,
                                             protocol_tokens, model_dir, n_prompts=args.task4_prompts)
    except Exception as e:
        log(f"  Task 4 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 5: Causal intervention (conditional on Task 4)
    if task4_result:
        try:
            task5_causal_intervention(model, tokenizer, device, info, model_name,
                                       task4_result, model_dir)
        except Exception as e:
            log(f"  Task 5 FAILED: {e}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    parser = argparse.ArgumentParser(description="Phase 952")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b", "all"])
    parser.add_argument("--task4_prompts", type=int, default=200)
    args = parser.parse_args()

    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started, model={args.model}")

    models = [args.model] if args.model != "all" else ["qwen3", "glm4", "deepseek7b"]
    for m in models:
        run_model(m, args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
