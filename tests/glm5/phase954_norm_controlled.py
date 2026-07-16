#!/usr/bin/env python3
"""
Phase 954: Norm-Controlled Protocol Field and Semantic Residual Audit
=====================================================================
范数控制协议场与语义残差审计

Task1-4合并: 范数测量 + 随机基线 + 语义残差矩阵 + 跨语义扩展
Task5:       协议通道范数敏感性
Task6:       多通道boost/ablate干预

核心公式:
  semantic_residual(p,s) = delta_z_p(semantic) - mean(delta_z_p(random))
  norm_delta = ||h + beta*d|| - ||h||
"""

from __future__ import annotations
import argparse, gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import _get_mlp_activation_hook, ensure_dir

PHASE = 954
RESULT_DIR = Path("results/phase954_norm_controlled")
BETA = 4.0
N_RANDOM = 8  # random directions per semantic direction

# Semantic word pairs (expanded)
SEMANTIC_DIRECTIONS = {
    "color_red-blue": ("red", "blue"),
    "color_yellow-black": ("yellow", "black"),
    "size_big-small": ("big", "small"),
    "size_large-tiny": ("large", "tiny"),
    "shape_round-square": ("round", "square"),
    "emotion_happy-sad": ("happy", "sad"),
    "emotion_good-bad": ("good", "bad"),
    "speed_fast-slow": ("fast", "slow"),
    "function_tool-food": ("tool", "food"),
    "category_animal-plant": ("animal", "plant"),
}

# Protocol tokens to track
PROTO_TOKENS = [".", " ", "\n", "the", " The", "is", " a", "Solution", "Step", "1"]

# Prompts
PROMPTS = [
    "The apple is", "The sky is", "The grass is", "The sun is",
    "The night is", "The snow is", "The fire is", "The ocean is",
]

# Super channels from Phase 953
SUPER_CHANNELS = {
    "qwen3": [935, 36, 284, 153, 188],
    "glm4": [12274, 7968, 5155, 5902, 1106],
    "deepseek7b": [15791, 15305, 1106, 4985, 14464],
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# TASK 1-4: Norm-Controlled Semantic Residual
# ============================================================

def task1_4_norm_residual(model, tokenizer, device, info, model_name: str,
                           save_dir: Path) -> dict:
    """Norm-controlled semantic residual with expanded categories."""
    log("  Task 1-4: Norm-controlled semantic residual...")

    W_U = get_W_U(model, model_name)
    embed = model.get_input_embeddings()
    last_layer = info.n_layers - 1

    # Build semantic directions
    sem_dirs = {}
    for dname, (w1, w2) in SEMANTIC_DIRECTIONS.items():
        ids1 = tokenizer.encode(w1, add_special_tokens=False)
        ids2 = tokenizer.encode(w2, add_special_tokens=False)
        if ids1 and ids2 and ids1[0] < W_U.shape[0] and ids2[0] < W_U.shape[0]:
            d = W_U[ids1[0]] - W_U[ids2[0]]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                sem_dirs[dname] = d / norm

    log(f"    Semantic directions: {len(sem_dirs)}")

    # Protocol token IDs
    proto_ids = {}
    for pt in PROTO_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Generate random directions
    rng = np.random.RandomState(42)
    d_model = W_U.shape[1]

    # Results storage
    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        hooks, cap = _get_mlp_activation_hook(model, info, last_layer)
        with torch.no_grad():
            # Get hidden state norm at last layer
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()
        base_act = cap.get("act")
        for h in hooks:
            h.remove()
        base_last_act = base_act[0, -1, :].numpy() if base_act is not None else np.zeros(1)
        base_norm = float(np.linalg.norm(base_last_act))

        for dname, sdir in sem_dirs.items():
            # Semantic injection
            dir_t = torch.tensor(sdir, dtype=torch.float32, device=device)
            inputs_embeds = embed(input_ids).detach().clone()
            inputs_embeds[0, -1, :] += (BETA * dir_t).to(inputs_embeds.dtype)

            hooks2, cap2 = _get_mlp_activation_hook(model, info, last_layer)
            with torch.no_grad():
                sem_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                sem_logits = sem_out.logits[0, -1].detach().float().cpu().numpy()
            sem_act = cap2.get("act")
            for h in hooks2:
                h.remove()
            sem_last_act = sem_act[0, -1, :].numpy() if sem_act is not None else np.zeros(1)
            sem_norm = float(np.linalg.norm(sem_last_act))

            # Semantic deltas
            sem_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(sem_logits):
                    sem_deltas[pt] = float(sem_logits[tid] - base_logits[tid])

            sem_norm_delta = sem_norm - base_norm

            # Random direction injections
            rand_deltas_list = defaultdict(list)
            rand_norm_deltas = []

            for ri in range(N_RANDOM):
                rdir = rng.randn(d_model).astype(np.float32)
                rdir = rdir / np.linalg.norm(rdir)

                dir_t = torch.tensor(rdir, dtype=torch.float32, device=device)
                inputs_embeds = embed(input_ids).detach().clone()
                inputs_embeds[0, -1, :] += (BETA * dir_t).to(inputs_embeds.dtype)

                with torch.no_grad():
                    rand_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                    rand_logits = rand_out.logits[0, -1].detach().float().cpu().numpy()

                for pt, tid in proto_ids.items():
                    if tid < len(rand_logits):
                        rand_deltas_list[pt].append(float(rand_logits[tid] - base_logits[tid]))

            # Compute random baseline and semantic residual
            rand_baseline = {}
            sem_residual = {}
            for pt in proto_ids:
                rvals = rand_deltas_list.get(pt, [])
                rmean = float(np.mean(rvals)) if rvals else 0
                rstd = float(np.std(rvals)) if rvals else 0
                rand_baseline[pt] = {"mean": rmean, "std": rstd}
                sem_residual[pt] = sem_deltas.get(pt, 0) - rmean

            all_results.append({
                "prompt": prompt,
                "direction": dname,
                "sem_norm_delta": sem_norm_delta,
                "sem_deltas": sem_deltas,
                "rand_baseline": rand_baseline,
                "sem_residual": sem_residual,
            })

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Aggregate: per-direction × per-token
    dir_token_agg = defaultdict(lambda: defaultdict(lambda: {
        "sem_deltas": [], "rand_means": [], "residuals": [], "norm_deltas": []
    }))

    for r in all_results:
        d = r["direction"]
        nd = r["sem_norm_delta"]
        for pt in proto_ids:
            sd = r["sem_deltas"].get(pt, 0)
            rm = r["rand_baseline"].get(pt, {}).get("mean", 0)
            res = r["sem_residual"].get(pt, 0)
            dir_token_agg[d][pt]["sem_deltas"].append(sd)
            dir_token_agg[d][pt]["rand_means"].append(rm)
            dir_token_agg[d][pt]["residuals"].append(res)
            dir_token_agg[d][pt]["norm_deltas"].append(nd)

    # Build summary matrix
    summary = {}
    for d, tokens in dir_token_agg.items():
        summary[d] = {}
        for pt, vals in tokens.items():
            summary[d][pt] = {
                "sem_mean": float(np.mean(vals["sem_deltas"])),
                "rand_mean": float(np.mean(vals["rand_means"])),
                "residual_mean": float(np.mean(vals["residuals"])),
                "residual_std": float(np.std(vals["residuals"])),
                "residual_t": float(np.mean(vals["residuals"]) / max(np.std(vals["residuals"]) / np.sqrt(len(vals["residuals"])), 0.001)),
                "norm_delta_mean": float(np.mean(vals["norm_deltas"])),
                "n": len(vals["sem_deltas"]),
            }

    output = {
        "task": "task1_4_norm_residual",
        "model": model_name,
        "beta": BETA,
        "n_prompts": len(PROMPTS),
        "n_directions": len(sem_dirs),
        "n_random_per_dir": N_RANDOM,
        "summary": summary,
        "raw_results": all_results[:10],
    }

    save_path = save_dir / "task1_4_residual.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key results: semantic residual for period
    log(f"    Semantic residual (sem - random) for key tokens:")
    log(f"    {'Direction':25s}  {'.':>8s}  {' ':>8s}  {'a':>8s}  {'EOS':>8s}  {'is':>8s}")
    for d in sorted(summary.keys()):
        s = summary[d]
        period_res = s.get(".", {}).get("residual_mean", 0)
        space_res = s.get(" ", {}).get("residual_mean", 0)
        a_res = s.get(" a", {}).get("residual_mean", 0)
        eos_res = s.get("<EOS>", {}).get("residual_mean", 0)
        is_res = s.get("is", {}).get("residual_mean", 0)
        log(f"    {d:25s}  {period_res:+8.3f}  {space_res:+8.3f}  {a_res:+8.3f}  {eos_res:+8.3f}  {is_res:+8.3f}")

    return output


# ============================================================
# TASK 5: Protocol Channel Norm Sensitivity
# ============================================================

def task5_channel_norm(model, tokenizer, device, info, model_name: str,
                        save_dir: Path) -> dict:
    """Test if super channel activation correlates with hidden state norm."""
    log("  Task 5: Protocol channel norm sensitivity...")

    W_U = get_W_U(model, model_name)
    embed = model.get_input_embeddings()
    last_layer = info.n_layers - 1
    super_channels = SUPER_CHANNELS.get(model_name, [])

    if not super_channels:
        log("    No super channels for this model")
        return {"task": "task5", "model": model_name, "skipped": True}

    # Use a single semantic direction for norm variation
    red_ids = tokenizer.encode("red", add_special_tokens=False)
    blue_ids = tokenizer.encode("blue", add_special_tokens=False)
    sem_dir = W_U[red_ids[0]] - W_U[blue_ids[0]]
    sem_dir = sem_dir / np.linalg.norm(sem_dir)

    rng = np.random.RandomState(42)
    d_model = W_U.shape[1]

    # Test multiple betas to vary norm
    betas = [0, 0.5, 1.0, 2.0, 4.0, 8.0]

    results = []

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for beta in betas:
            if beta == 0:
                # Baseline
                hooks, cap = _get_mlp_activation_hook(model, info, last_layer)
                with torch.no_grad():
                    out = model(input_ids=input_ids, use_cache=False)
                    logits = out.logits[0, -1].detach().float().cpu().numpy()
                act = cap.get("act")
                for h in hooks:
                    h.remove()
            else:
                # Semantic injection
                dir_t = torch.tensor(sem_dir, dtype=torch.float32, device=device)
                inputs_embeds = embed(input_ids).detach().clone()
                inputs_embeds[0, -1, :] += (beta * dir_t).to(inputs_embeds.dtype)

                hooks, cap = _get_mlp_activation_hook(model, info, last_layer)
                with torch.no_grad():
                    out = model(inputs_embeds=inputs_embeds, use_cache=False)
                    logits = out.logits[0, -1].detach().float().cpu().numpy()
                act = cap.get("act")
                for h in hooks:
                    h.remove()

            if act is not None:
                last_act = act[0, -1, :].numpy()
                norm = float(np.linalg.norm(last_act))

                # Channel activations
                ch_acts = {str(ch): float(last_act[ch]) for ch in super_channels}

                # Period logit
                period_ids = tokenizer.encode(".", add_special_tokens=False)
                period_logit = float(logits[period_ids[0]]) if period_ids else 0

                results.append({
                    "prompt": prompt,
                    "beta": beta,
                    "norm": norm,
                    "channel_acts": ch_acts,
                    "period_logit": period_logit,
                })

        # Also test random directions at beta=4
        for ri in range(5):
            rdir = rng.randn(d_model).astype(np.float32)
            rdir = rdir / np.linalg.norm(rdir)
            dir_t = torch.tensor(rdir, dtype=torch.float32, device=device)
            inputs_embeds = embed(input_ids).detach().clone()
            inputs_embeds[0, -1, :] += (BETA * dir_t).to(inputs_embeds.dtype)

            hooks, cap = _get_mlp_activation_hook(model, info, last_layer)
            with torch.no_grad():
                out = model(inputs_embeds=inputs_embeds, use_cache=False)
                logits = out.logits[0, -1].detach().float().cpu().numpy()
            act = cap.get("act")
            for h in hooks:
                h.remove()

            if act is not None:
                last_act = act[0, -1, :].numpy()
                norm = float(np.linalg.norm(last_act))
                ch_acts = {str(ch): float(last_act[ch]) for ch in super_channels}
                period_ids = tokenizer.encode(".", add_special_tokens=False)
                period_logit = float(logits[period_ids[0]]) if period_ids else 0

                results.append({
                    "prompt": prompt,
                    "beta": -1,  # mark as random
                    "norm": norm,
                    "channel_acts": ch_acts,
                    "period_logit": period_logit,
                })

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Compute correlations
    norms = [r["norm"] for r in results]
    period_logits = [r["period_logit"] for r in results]

    # Norm vs period logit correlation
    if len(norms) > 3:
        norm_period_corr = float(np.corrcoef(norms, period_logits)[0, 1])
    else:
        norm_period_corr = 0

    # Channel activation vs norm correlation
    ch_norm_corr = {}
    ch_period_corr = {}
    for ch in super_channels:
        ch_vals = [r["channel_acts"][str(ch)] for r in results]
        if len(ch_vals) > 3:
            ch_norm_corr[str(ch)] = float(np.corrcoef(ch_vals, norms)[0, 1])
            ch_period_corr[str(ch)] = float(np.corrcoef(ch_vals, period_logits)[0, 1])
        else:
            ch_norm_corr[str(ch)] = 0
            ch_period_corr[str(ch)] = 0

    output = {
        "task": "task5_channel_norm",
        "model": model_name,
        "n_results": len(results),
        "super_channels": super_channels,
        "norm_period_correlation": norm_period_corr,
        "channel_norm_correlation": ch_norm_corr,
        "channel_period_correlation": ch_period_corr,
        "raw_results": results[:15],
    }

    save_path = save_dir / "task5_channel_norm.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    log(f"    Norm-Period correlation: {norm_period_corr:.3f}")
    for ch, corr in sorted(ch_norm_corr.items(), key=lambda x: -abs(x[1])):
        log(f"    Channel {ch}: norm_corr={corr:.3f}, period_corr={ch_period_corr[ch]:.3f}")

    return output


# ============================================================
# TASK 6: Multi-Channel Boost/Ablate
# ============================================================

def task6_boost_ablate(model, tokenizer, device, info, model_name: str,
                        save_dir: Path) -> dict:
    """Boost (1.5x) or ablate (0x) top channels, measure protocol change."""
    log("  Task 6: Multi-channel boost/ablate...")

    last_layer = info.n_layers - 1
    layers_list = get_layers(model)
    target_layer = layers_list[last_layer]
    super_channels = SUPER_CHANNELS.get(model_name, [])[:5]

    if not super_channels:
        log("    No super channels")
        return {"task": "task6", "model": model_name, "skipped": True}

    proto_ids = {}
    for pt in PROTO_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    results = []
    conditions = [
        ("ablate_K5", 0.0, super_channels[:5]),
        ("boost_K5", 1.5, super_channels[:5]),
        ("ablate_K3", 0.0, super_channels[:3]),
        ("boost_K3", 1.5, super_channels[:3]),
    ]

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for cond_name, scale, channels in conditions:
            def make_pre_hook(chs, scl):
                ch_set = set(int(c) for c in chs)
                def pre_hook(module, args):
                    inp = args[0]
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    patched = inp.clone()
                    if patched.ndim >= 3:
                        for c in ch_set:
                            if scl == 0:
                                patched[:, -1, c] = 0
                            else:
                                patched[:, -1, c] = patched[:, -1, c] * scl
                    return (patched,)
                return pre_hook

            handle = target_layer.mlp.down_proj.register_forward_pre_hook(
                make_pre_hook(channels, scale)
            )

            with torch.no_grad():
                try:
                    patched_out = model(input_ids=input_ids, use_cache=False)
                    patched_logits = patched_out.logits[0, -1].detach().float().cpu().numpy()
                except:
                    patched_logits = base_logits.copy()

            handle.remove()

            token_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(patched_logits):
                    token_deltas[pt] = float(patched_logits[tid] - base_logits[tid])

            results.append({
                "prompt": prompt,
                "condition": cond_name,
                "scale": scale,
                "channels": channels,
                "token_deltas": token_deltas,
            })

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Aggregate
    cond_agg = {}
    for cond_name, _, _ in conditions:
        cond_results = [r for r in results if r["condition"] == cond_name]
        token_means = {}
        for pt in proto_ids:
            vals = [r["token_deltas"].get(pt, 0) for r in cond_results]
            token_means[pt] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        cond_agg[cond_name] = token_means

    output = {
        "task": "task6_boost_ablate",
        "model": model_name,
        "n_prompts": len(PROMPTS),
        "conditions": [c[0] for c in conditions],
        "results": cond_agg,
    }

    save_path = save_dir / "task6_boost.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for cond_name, _, _ in conditions:
        ca = cond_agg.get(cond_name, {})
        log(f"    {cond_name}:")
        for pt in [".", " ", "<EOS>", "Solution", "is"]:
            if pt in ca:
                log(f"      {pt:12s}: {ca[pt]['mean']:+.4f} +/- {ca[pt]['std']:.4f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 954: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        task1_4_norm_residual(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1-4 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task5_channel_norm(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 5 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task6_boost_ablate(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 6 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")

    for m in ["qwen3", "glm4", "deepseek7b"]:
        run_model(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
