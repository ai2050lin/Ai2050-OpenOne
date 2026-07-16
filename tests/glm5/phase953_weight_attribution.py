#!/usr/bin/env python3
"""
Phase 953: 权重级Protocol通道归因 + 多通道组合干预
===================================================
关键创新: 用解析公式计算通道贡献, 不用回归

  contribution_j(v, x) = a_j(x) * (W_U[v,:] · W_down[:,j])

其中:
  a_j(x)     = MLP中间激活 (hook获取)
  W_down[:,j] = down_proj权重的第j列 [d_model]
  W_U[v,:]   = lm_head权重的第v行 [d_model]

Task1: 权重级通道归因 — 找到每个protocol token的top贡献通道
Task2: 多通道组合干预 — 同时零化K=1,5,10,20个通道
Task3: 非color语义注入 — size/shape/emotion方向对protocol的影响
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
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS
from phase951_protocol_atlas import (
    ALL_PROMPT_GROUPS, get_protocol_token_ids, _get_mlp_activation_hook,
    safe_decode, COLOR_WORDS, ensure_dir
)

PHASE = 953
RESULT_DIR = Path("results/phase953_weight_attribution")

# Protocol tokens to track individually
PROTOCOL_TOKENS = [".", " .", " ", "\n", "the", " The", "is", " a",
                   "Answer", "Solution", "Step", "1"]

# Non-color semantic word pairs
SEMANTIC_PAIRS = {
    "color": [("red", "blue"), ("yellow", "black"), ("white", "black")],
    "size": [("big", "small"), ("large", "tiny"), ("huge", "microscopic")],
    "shape": [("round", "square"), ("circle", "triangle")],
    "emotion": [("happy", "sad"), ("good", "bad"), ("beautiful", "ugly")],
    "speed": [("fast", "slow"), ("quick", "gradual")],
}

# Prompts for injection
INJECTION_PROMPTS = [
    "The apple is", "The sky is", "The grass is",
    "The sun is", "The night is", "The snow is",
    "The fire is", "The flower is", "The ocean is",
    "The lemon is", "The rose is", "The coal is",
    "The cloud is", "The wood is", "The wine is",
]

# Prompts for channel attribution
ATTRIBUTION_PROMPTS = []
for gname, prompts in ALL_PROMPT_GROUPS.items():
    ATTRIBUTION_PROMPTS.extend(prompts)
ATTRIBUTION_PROMPTS = ATTRIBUTION_PROMPTS[:80]  # 80 prompts for attribution


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_down_proj_weight(model, info, layer_idx: int) -> np.ndarray:
    """Get down_proj weight matrix [d_model, intermediate] as numpy."""
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    W_down = mlp.down_proj.weight.detach().float().cpu().numpy()
    # W_down shape: [d_model, intermediate]
    return W_down


# ============================================================
# TASK 1: Weight-Based Channel Attribution
# ============================================================

def task1_weight_attribution(model, tokenizer, device, info, model_name: str,
                              save_dir: Path, n_prompts: int = 60) -> dict:
    """Analytical channel attribution using actual weights."""
    log("  Task 1: Weight-based channel attribution...")

    W_U = get_W_U(model, model_name)  # [vocab, d_model]

    # Target: last layer
    last_layer = info.n_layers - 1
    W_down = get_down_proj_weight(model, info, last_layer)  # [d_model, intermediate]
    inter_size = W_down.shape[1]
    d_model = W_down.shape[0]

    log(f"    Layer L{last_layer}: W_down={W_down.shape}, W_U={W_U.shape}, inter={inter_size}")

    # Precompute write_weights: W_write[v, j] = W_U[v, :] · W_down[:, j]
    # This is W_U @ W_down → [vocab, intermediate]
    # But vocab is huge (~150K), so only compute for protocol tokens
    proto_ids = {}
    for pt in PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Compute write weights for each protocol token
    write_weights = {}  # {token_str: [intermediate]} = W_U[v,:] @ W_down
    for pt, tid in proto_ids.items():
        if tid < W_U.shape[0]:
            w = W_U[tid] @ W_down  # [intermediate]
            write_weights[pt] = w

    # For each prompt, get MLP activation and compute contributions
    selected = ATTRIBUTION_PROMPTS[:n_prompts]
    all_contributions = []  # [{token: {channel: contribution}}]
    all_activations = []

    for pi, prompt in enumerate(selected):
        # Get MLP activation
        hooks, cap = _get_mlp_activation_hook(model, info, last_layer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            base_logits = out.logits[0, -1].detach().float().cpu().numpy()
        for h in hooks:
            h.remove()

        act = cap.get("act")
        if act is None:
            continue
        last_act = act[0, -1, :].numpy()  # [intermediate]

        # Compute per-token per-channel contribution
        contributions = {}
        for pt, w_write in write_weights.items():
            # contribution_j = a_j * w_write[j]
            contrib = last_act * w_write  # [intermediate]
            contributions[pt] = contrib

        all_contributions.append({
            "prompt": prompt,
            "contributions": contributions,
            "base_logits": base_logits,
        })
        all_activations.append(last_act)

        if (pi + 1) % 20 == 0:
            log(f"    {pi+1}/{len(selected)} prompts")

    # Aggregate: mean |contribution| per channel per token
    token_channel_importance = {}
    for pt in write_weights:
        all_contribs = np.array([ac["contributions"][pt] for ac in all_contributions])
        mean_abs_contrib = np.mean(np.abs(all_contribs), axis=0)  # [intermediate]
        mean_contrib = np.mean(all_contribs, axis=0)  # [intermediate]

        # Top channels by |contribution|
        top_indices = np.argsort(mean_abs_contrib)[-30:][::-1]
        token_channel_importance[pt] = {
            "token": pt,
            "top_channels": [
                {
                    "channel": int(idx),
                    "mean_abs_contrib": float(mean_abs_contrib[idx]),
                    "mean_contrib": float(mean_contrib[idx]),
                    "write_weight": float(write_weights[pt][idx]),
                }
                for idx in top_indices
            ],
            "total_abs_contrib": float(np.sum(mean_abs_contrib)),
        }

    # Find shared channels (contribute to 3+ protocol tokens)
    channel_tokens = defaultdict(list)
    for pt, data in token_channel_importance.items():
        for ch in data["top_channels"][:15]:
            channel_tokens[ch["channel"]].append({
                "token": pt,
                "contrib": ch["mean_contrib"],
                "abs_contrib": ch["mean_abs_contrib"],
            })

    shared_channels = {
        str(ch): toks for ch, toks in channel_tokens.items() if len(toks) >= 3
    }
    shared_channels = dict(sorted(shared_channels.items(),
                                   key=lambda x: -len(x[1])))

    output = {
        "task": "task1_weight_attribution",
        "model": model_name,
        "method": "analytical_weight_decomposition",
        "layer": last_layer,
        "n_prompts": len(all_contributions),
        "n_intermediate": inter_size,
        "formula": "contribution_j(v) = a_j * (W_U[v,:] . W_down[:,j])",
        "token_channel_importance": token_channel_importance,
        "shared_channels": {k: v for k, v in list(shared_channels.items())[:30]},
        "n_shared_channels": len(shared_channels),
    }

    save_path = save_dir / "task1_weight.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")
    log(f"    Shared channels (3+ tokens): {len(shared_channels)}")

    # Print top channels for key tokens
    for pt in [".", " ", "<EOS>", "Solution"]:
        if pt in token_channel_importance:
            data = token_channel_importance[pt]
            top3 = data["top_channels"][:3]
            ch_str = ", ".join(f"ch{c['channel']}={c['mean_abs_contrib']:.3f}" for c in top3)
            log(f"    {pt:12s}: top3=[{ch_str}]  total={data['total_abs_contrib']:.2f}")

    return output


# ============================================================
# TASK 2: Multi-Channel Combination Intervention
# ============================================================

def task2_multi_channel_intervention(model, tokenizer, device, info, model_name: str,
                                      task1_result: dict, save_dir: Path,
                                      n_prompts: int = 15) -> dict:
    """Zero top-K channels simultaneously, measure protocol logit change."""
    log("  Task 2: Multi-channel combination intervention...")

    last_layer = info.n_layers - 1
    layers_list = get_layers(model)
    target_layer = layers_list[last_layer]

    # Get shared channels from Task 1
    shared_channels = task1_result.get("shared_channels", {})
    if not shared_channels:
        log("    No shared channels from Task 1, using top channels per token")
        # Fall back to top channels across all tokens
        all_channels = set()
        for pt, data in task1_result.get("token_channel_importance", {}).items():
            for ch in data.get("top_channels", [])[:5]:
                all_channels.add(ch["channel"])
        channel_list = sorted(all_channels)[:20]
    else:
        channel_list = [int(ch) for ch in list(shared_channels.keys())[:20]]

    log(f"    Using {len(channel_list)} channels for intervention")

    # Protocol tokens to measure
    proto_ids = {}
    for pt in PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Test K = 1, 5, 10, 20
    K_VALUES = [1, 5, 10, 20]
    results = []

    for pi, prompt in enumerate(INJECTION_PROMPTS[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for K in K_VALUES:
            if K > len(channel_list):
                continue
            channels_to_zero = channel_list[:K]

            # Register pre-hook to zero multiple channels
            def make_pre_hook(channels):
                ch_set = set(int(c) for c in channels)
                def pre_hook(module, args):
                    inp = args[0]
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    patched = inp.clone()
                    if patched.ndim >= 3:
                        for c in ch_set:
                            patched[:, -1, c] = 0
                    elif patched.ndim >= 2:
                        for c in ch_set:
                            patched[:, c] = 0
                    return (patched,)
                return pre_hook

            handle = target_layer.mlp.down_proj.register_forward_pre_hook(
                make_pre_hook(channels_to_zero)
            )

            with torch.no_grad():
                try:
                    patched_out = model(input_ids=input_ids, use_cache=False)
                    patched_logits = patched_out.logits[0, -1].detach().float().cpu().numpy()
                except Exception:
                    patched_logits = base_logits.copy()

            handle.remove()

            # Per-token delta
            token_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(patched_logits):
                    token_deltas[pt] = float(patched_logits[tid] - base_logits[tid])

            results.append({
                "prompt": prompt,
                "K": K,
                "channels_zeroed": channels_to_zero,
                "token_deltas": token_deltas,
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{n_prompts} prompts")

    # Aggregate per-K
    k_agg = {}
    for K in K_VALUES:
        k_results = [r for r in results if r["K"] == K]
        token_means = {}
        for pt in proto_ids:
            vals = [r["token_deltas"].get(pt, 0) for r in k_results]
            token_means[pt] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "abs_mean": float(np.mean(np.abs(vals))),
                "n": len(vals),
            }
        k_agg[str(K)] = {
            "n_prompts": len(k_results),
            "channels": channel_list[:K],
            "token_deltas": token_means,
        }

    output = {
        "task": "task2_multi_channel",
        "model": model_name,
        "layer": last_layer,
        "n_prompts": n_prompts,
        "K_values": K_VALUES,
        "channel_list": channel_list,
        "results": k_agg,
    }

    save_path = save_dir / "task2_multi_channel.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print results
    for K in K_VALUES:
        if str(K) not in k_agg:
            continue
        ka = k_agg[str(K)]
        log(f"    K={K:2d} channels zeroed:")
        for pt in [".", " ", "<EOS>", "Solution", "is"]:
            if pt in ka["token_deltas"]:
                d = ka["token_deltas"][pt]
                log(f"      {pt:12s}: {d['mean']:+.4f} +/- {d['std']:.4f} (abs_mean={d['abs_mean']:.4f})")

    return output


# ============================================================
# TASK 3: Non-Color Semantic Injection
# ============================================================

def task3_non_color_semantic(model, tokenizer, device, info, model_name: str,
                              save_dir: Path, n_prompts: int = 12) -> dict:
    """Test non-color semantic directions (size, shape, emotion, speed)."""
    log("  Task 3: Non-color semantic injection...")

    W_U = get_W_U(model, model_name)
    embed = model.get_input_embeddings()

    # Build semantic directions
    all_dirs = {}
    for category, pairs in SEMANTIC_PAIRS.items():
        for w1, w2 in pairs:
            ids1 = tokenizer.encode(w1, add_special_tokens=False)
            ids2 = tokenizer.encode(w2, add_special_tokens=False)
            if ids1 and ids2 and ids1[0] < W_U.shape[0] and ids2[0] < W_U.shape[0]:
                d = W_U[ids1[0]] - W_U[ids2[0]]
                norm = np.linalg.norm(d)
                if norm > 1e-10:
                    all_dirs[f"{category}:{w1}-{w2}"] = d / norm

    log(f"    Semantic directions: {len(all_dirs)}")
    log(f"    Categories: {set(k.split(':')[0] for k in all_dirs)}")

    # Protocol tokens
    proto_ids = {}
    for pt in PROTOCOL_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    beta = 4.0
    results = []

    for pi, prompt in enumerate(INJECTION_PROMPTS[:n_prompts]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for dname, direction in all_dirs.items():
            dir_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
            inputs_embeds = embed(input_ids).detach().clone()
            inputs_embeds[0, -1, :] += (beta * dir_tensor).to(inputs_embeds.dtype)

            with torch.no_grad():
                inj_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                inj_logits = inj_out.logits[0, -1].detach().float().cpu().numpy()

            token_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(inj_logits):
                    token_deltas[pt] = float(inj_logits[tid] - base_logits[tid])

            results.append({
                "prompt": prompt,
                "direction": dname,
                "beta": beta,
                "token_deltas": token_deltas,
            })

        if (pi + 1) % 4 == 0:
            log(f"    {pi+1}/{n_prompts} prompts")

    # Aggregate per-direction-category
    cat_agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r["direction"].split(":")[0]
        for pt, delta in r["token_deltas"].items():
            cat_agg[cat][pt].append(delta)

    summary = {}
    for cat, token_vals in cat_agg.items():
        summary[cat] = {
            pt: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            for pt, vals in token_vals.items()
        }

    output = {
        "task": "task3_non_color",
        "model": model_name,
        "beta": beta,
        "n_prompts": n_prompts,
        "n_directions": len(all_dirs),
        "directions": list(all_dirs.keys()),
        "summary": summary,
        "raw_results": results[:20],
    }

    save_path = save_dir / "task3_non_color.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print comparison across categories
    log(f"    Per-token delta by semantic category (beta={beta}):")
    for pt in [".", " ", "<EOS>", "Solution", "is", " a"]:
        vals = []
        for cat in ["color", "size", "shape", "emotion", "speed"]:
            if cat in summary and pt in summary[cat]:
                vals.append(f"{cat}={summary[cat][pt]['mean']:+.3f}")
        if vals:
            log(f"      {pt:12s}: {', '.join(vals)}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str, args: argparse.Namespace) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 953: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}, inter={info.intermediate_size}")

    t_start = time.time()

    # Task 1: Weight-based attribution
    task1_result = None
    try:
        task1_result = task1_weight_attribution(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 2: Multi-channel intervention
    if task1_result:
        try:
            task2_multi_channel_intervention(model, tokenizer, device, info, model_name,
                                              task1_result, model_dir)
        except Exception as e:
            log(f"  Task 2 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Task 3: Non-color semantic
    try:
        task3_non_color_semantic(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    parser = argparse.ArgumentParser(description="Phase 953")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b", "all"])
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
