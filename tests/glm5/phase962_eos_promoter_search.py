#!/usr/bin/env python3
"""
Phase 962: EOS Promoter Search and Reverse Lock Intervention
=============================================================
Task 1: Search for EOS-promoting heads (cos(O_h, W_U[EOS]) > 0)
Task 2: Search for EOS-promoting MLP channels (cos(W_down[:,c], W_U[EOS]) > 0)
Task 3: Reverse lock intervention (inject -λ*O_lock, not just ablate)
Task 4: Combined: reverse lock + boost EOS-promoting channel
Task 5: De-headed mode direction (remove circularity from Phase 961)
Task 6: Large-scale rollout validation

Models: qwen3 -> GLM4 -> DS7B (sequential)
"""

from __future__ import annotations
import gc, json, sys, time, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir

PHASE = 962
RESULT_DIR = Path("results/phase962_eos_promoter")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_HEADS = {
    "qwen3": [
        {"layer": 35, "head": 0, "role": "period/lock"},
        {"layer": 33, "head": 8, "role": "period"},
    ],
    "glm4": [
        {"layer": 39, "head": 21, "role": "mode-lock"},   # Key head
        {"layer": 38, "head": 0, "role": "logit-only"},
        {"layer": 38, "head": 7, "role": "logit-only"},
    ],
    "deepseek7b": [
        {"layer": 26, "head": 19, "role": "space"},
        {"layer": 26, "head": 25, "role": "space/format"},
    ],
}

EN_PROMPTS_50 = [
    "The capital of France is", "The largest planet is", "Water boils at",
    "The speed of light is", "The sun is a", "Dogs are", "The sky is",
    "Grass is", "Fire needs", "Ice is", "The Earth is", "A triangle has",
    "Shakespeare was", "Tokyo is the capital of", "The Pacific Ocean is",
    "Gold is a", "Plants need", "Humans breathe", "The moon is", "Birds can",
    "The largest country is", "A square has", "Mathematics is", "Music is",
    "The brain is", "Iron is a", "Trees produce", "Rivers flow",
    "Volcanoes erupt", "Stars are", "The heart pumps", "DNA contains",
    "Gravity pulls", "Light travels", "Sound is", "Heat is",
    "A compass points", "The equator is", "Antarctica is", "Diamonds are",
    "Oxygen is", "The kidney filters", "Whales are", "The alphabet has",
    "A century is", "The constitution is", "Bridges connect",
    "Computers process", "Languages evolve", "The internet is",
]

CN_PROMPTS_10 = [
    "法国的首都是", "最大的行星是", "水的沸点是", "光速是", "太阳是一颗",
    "狗是", "天空是", "草是", "火需要", "冰是",
]

EN_PROMPTS_10 = EN_PROMPTS_50[:10]

MAX_TOKENS = 40

EXPECTED_ANSWERS = {
    "The capital of France is": "Paris", "The largest planet is": "Jupiter",
    "Water boils at": "100", "The speed of light is": "299",
    "The sun is a": "star", "Dogs are": "animal", "The sky is": "blue",
    "Grass is": "green", "Fire needs": "oxygen", "Ice is": "frozen",
    "The Earth is": "round", "A triangle has": "three",
}


# ============================================================
# UTILITIES
# ============================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_head_dims(model, info):
    n_heads = getattr(model.config, "num_attention_heads", 32)
    layers = get_layers(model)
    o_proj_in = layers[0].self_attn.o_proj.weight.shape[1]
    d_head = o_proj_in // n_heads
    return n_heads, d_head


def cosine_similarity(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_strict_clean(prompt, generated, has_eos, n_tokens):
    expected = EXPECTED_ANSWERS.get(prompt, "")
    is_ascii = all(ord(c) < 256 for c in generated)
    is_short = 0 < n_tokens < 15
    has_expected = (expected == "") or (expected.lower() in generated.lower())
    return {
        "strict_clean": has_eos and is_short and has_expected and is_ascii,
        "has_eos": has_eos, "is_short": is_short,
        "has_expected": has_expected, "is_ascii": is_ascii,
    }


def make_head_hook(sc, ec, scale):
    def hook(module, args):
        if scale == 1.0: return None
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            if scale == 0.0: patched[:, :, sc:ec] = 0
            else: patched[:, :, sc:ec] = patched[:, :, sc:ec] * scale
        return (patched,)
    return hook


def make_channel_hook(channels, scale):
    ch_list = [int(c) for c in channels]
    def hook(module, args):
        if scale == 1.0: return None
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            for c in ch_list:
                if c < patched.shape[-1]:
                    if scale == 0.0: patched[:, -1, c] = 0
                    else: patched[:, -1, c] = patched[:, -1, c] * scale
        return (patched,)
    return hook


def make_capture_hook(captured, key):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured[key] = out[0, -1, :].detach().float().cpu().numpy()
    return hook


# ============================================================
# TASK 1: EOS-Promoting Head Search
# ============================================================

def task1_eos_head_search(model, tokenizer, device, info, model_name, save_dir):
    """Search all layers × heads for cos(O_h, W_U[EOS]) > 0."""
    log("  Task 1: EOS-promoting head search (10 prompts, all layers)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    W_U = get_W_U(model, model_name)  # [vocab, d_model]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        log("    No EOS token, skipping"); return None
    W_U_eos = W_U[eos_id].astype(np.float64)  # [d_model]
    W_U_eos_norm = np.linalg.norm(W_U_eos)

    # Also compute for period and space for comparison
    proto_dirs = {"EOS": W_U_eos}
    for tok_str, tok_key in [(".", "period"), (" ", "space")]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks:
            proto_dirs[tok_key] = W_U[toks[0]].astype(np.float64)

    # Accumulate cos scores: {layer_head: {direction: [scores]}}
    cos_scores = defaultdict(lambda: defaultdict(list))

    prompts = EN_PROMPTS_10
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Capture o_proj inputs for ALL layers in one forward pass
        captured = {}
        hooks = []
        for L in range(n_layers):
            def make_pre_hook(layer_idx):
                def hook(module, args):
                    inp = args[0] if isinstance(args, tuple) else args
                    captured[layer_idx] = inp[0, -1, :].detach().float().cpu().numpy()
                return hook
            hooks.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(L)))

        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in hooks: h.remove()

        # Compute O_h for each head in each layer
        for L in range(n_layers):
            if L not in captured: continue
            o_proj_input = captured[L]  # [n_heads * d_head]
            W_O = layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()  # [d_model, n_heads*d_head]

            for H in range(n_heads):
                sc = H * d_head; ec = sc + d_head
                head_input = o_proj_input[sc:ec]  # [d_head]
                W_O_slice = W_O[:, sc:ec]  # [d_model, d_head]
                O_h = W_O_slice @ head_input  # [d_model]
                oh_norm = np.linalg.norm(O_h)

                key = f"L{L}_H{H}"
                for dir_name, dir_vec in proto_dirs.items():
                    cos = cosine_similarity(O_h, dir_vec)
                    cos_scores[key][dir_name].append(cos)

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    head_results = {}
    for key, dirs in cos_scores.items():
        means = {d: float(np.mean(v)) for d, v in dirs.items()}
        stds = {d: float(np.std(v)) for d, v in dirs.items()}
        head_results[key] = {"mean": means, "std": stds, "n_prompts": len(prompts)}

    # Find top EOS-promoting heads (cos > 0)
    eos_sorted = sorted(head_results.items(), key=lambda x: -x[1]["mean"]["EOS"])
    top_eos_promoters = [(k, v["mean"]["EOS"]) for k, v in eos_sorted if v["mean"]["EOS"] > 0.02][:20]

    # Find top EOS-suppressing heads (cos < 0)
    top_eos_suppressors = [(k, v["mean"]["EOS"]) for k, v in eos_sorted if v["mean"]["EOS"] < -0.02][-10:]

    output = {
        "task": "task1_eos_head_search", "model": model_name,
        "n_layers": n_layers, "n_heads": n_heads, "d_head": d_head,
        "n_prompts": len(prompts),
        "top_eos_promoters": top_eos_promoters,
        "top_eos_suppressors": top_eos_suppressors,
        "all_heads": head_results,
    }
    save_path = save_dir / "task1_eos_head_search.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    log(f"    Top 10 EOS-promoting heads:")
    for key, cos in top_eos_promoters[:10]:
        v = head_results[key]
        log(f"      {key}: cos_EOS={cos:.4f}  cos_period={v['mean'].get('period', 0):.4f}  cos_space={v['mean'].get('space', 0):.4f}")

    log(f"    Top 5 EOS-suppressing heads:")
    for key, cos in top_eos_suppressors[:5]:
        v = head_results[key]
        log(f"      {key}: cos_EOS={cos:.4f}")

    return output


# ============================================================
# TASK 2: EOS-Promoting Channel Search
# ============================================================

def task2_eos_channel_search(model, tokenizer, device, info, model_name, save_dir):
    """Search all MLP channels for cos(W_down[:,c], W_U[EOS]) > 0."""
    log("  Task 2: EOS-promoting channel search (weight-only + 1 forward)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    W_U = get_W_U(model, model_name)
    eos_id = tokenizer.eos_token_id
    W_U_eos = W_U[eos_id].astype(np.float64)
    W_U_eos_norm = np.linalg.norm(W_U_eos)

    # Weight-only cosine for all layers × channels
    channel_results = {}
    for L in range(n_layers):
        W_down = layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()  # [d_model, intermediate]
        # cos = (W_U_eos @ W_down) / (||W_U_eos|| * ||W_down||_colwise)
        dot = W_U_eos @ W_down  # [intermediate]
        col_norms = np.linalg.norm(W_down, axis=0)  # [intermediate]
        cos_vals = dot / (W_U_eos_norm * col_norms + 1e-10)

        # Find top channels
        top_idx = np.argsort(cos_vals)[-10:][::-1]
        for idx in top_idx:
            key = f"L{L}_C{idx}"
            channel_results[key] = {
                "layer": L, "channel": int(idx),
                "cos_eos": float(cos_vals[idx]),
            }

    # Also check natural activation: forward pass to capture down_proj inputs
    prompt = EN_PROMPTS_10[0]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured_down = {}
    hooks = []
    for L in range(n_layers):
        def make_hook(layer_idx):
            def hook(module, args):
                inp = args[0] if isinstance(args, tuple) else args
                captured_down[layer_idx] = inp[0, -1, :].detach().float().cpu().numpy()
            return hook
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L)))

    with torch.no_grad():
        model(input_ids, use_cache=False)
    for h in hooks: h.remove()

    # Compute contribution = activation * cos_eos * W_U_eos_norm
    for key, cr in channel_results.items():
        L = cr["layer"]; c = cr["channel"]
        if L in captured_down and c < len(captured_down[L]):
            activation = float(captured_down[L][c])
            contribution = activation * cr["cos_eos"] * W_U_eos_norm
            cr["activation"] = activation
            cr["contribution_eos"] = float(contribution)

    # Sort by contribution
    sorted_channels = sorted(channel_results.items(), key=lambda x: -abs(x[1].get("contribution_eos", 0)))
    top_promoting = [(k, v) for k, v in sorted_channels if v.get("contribution_eos", 0) > 0][:20]

    output = {
        "task": "task2_eos_channel_search", "model": model_name,
        "n_layers": n_layers,
        "top_eos_promoting_channels": [(k, v) for k, v in top_promoting],
        "all_channel_results": channel_results,
    }
    save_path = save_dir / "task2_eos_channel_search.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"    Saved: {save_path}")

    log(f"    Top 10 EOS-promoting channels (by contribution):")
    for key, v in top_promoting[:10]:
        log(f"      {key}: cos_eos={v['cos_eos']:.4f}  activation={v.get('activation', 0):.3f}  contribution={v.get('contribution_eos', 0):.3f}")

    return output, [v["layer"] for _, v in top_promoting[:5]], [v["channel"] for _, v in top_promoting[:5]]


# ============================================================
# TASK 3: Reverse Lock Intervention
# ============================================================

def task3_reverse_lock(model, tokenizer, device, info, model_name, heads, save_dir):
    """Inject -λ*O_lock via o_proj scaling: scale = 1-λ.
    λ=0: normal, λ=0.5: partial ablate, λ=1.0: full ablate, λ=2.0: reverse."""
    log("  Task 3: Reverse lock intervention...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    primary = heads[0]
    L0, H0 = primary["layer"], primary["head"]
    sc = H0 * d_head; ec = sc + d_head

    # Prompts
    if model_name == "glm4":
        prompts = EN_PROMPTS_10[:3]
    else:
        prompts = EN_PROMPTS_10[:5]

    lambdas = [0.0, 0.5, 1.0, 2.0]  # 0=normal, 0.5=partial, 1.0=full ablate, 2.0=reverse

    all_results = []

    for pi, prompt in enumerate(prompts):
        for lam in lambdas:
            handles = []

            if lam != 0.0:
                # Use o_proj scaling: scale = 1-λ directly modifies head's input
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))

            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"; has_eos = False; n_gen = 0

            for h in handles: h.remove()

            is_ascii = all(ord(c) < 256 for c in generated)
            all_results.append({
                "prompt": prompt, "lambda": lam,
                "generated": generated[:200], "has_eos": has_eos,
                "n_tokens": n_gen, "lang_switched": not is_ascii,
            })

        log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    summary = {}
    for lam in lambdas:
        lr = [r for r in all_results if r["lambda"] == lam]
        summary[f"lam_{lam}"] = {
            "eos_rate": np.mean([r["has_eos"] for r in lr]) if lr else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in lr]) if lr else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in lr]) if lr else 0,
            "n": len(lr),
        }

    output = {
        "task": "task3_reverse_lock", "model": model_name,
        "primary_head": f"L{L0}_H{H0}", "lambdas": lambdas,
        "summary": summary, "raw_results": all_results,
    }
    save_path = save_dir / "task3_reverse_lock.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for lam in lambdas:
        s = summary.get(f"lam_{lam}", {})
        log(f"    λ={lam:.1f}: eos={s.get('eos_rate', 0):.2f}  switch={s.get('lang_switch_rate', 0):.2f}  tokens={s.get('mean_tokens', 0):.1f}")

    # Print samples for first prompt
    log(f"    Sample (prompt 0):")
    for r in all_results:
        if r["prompt"] == prompts[0]:
            log(f"      λ={r['lambda']:.1f}: eos={r['has_eos']}  text={r['generated'][:80]}")

    return output


# ============================================================
# TASK 4: Combined Intervention (reverse lock + boost EOS channel)
# ============================================================

def task4_combined_intervention(model, tokenizer, device, info, model_name, heads,
                                eos_channel_layer, eos_channel_idx, save_dir):
    """Combine reverse lock with EOS channel boost."""
    log("  Task 4: Combined intervention (reverse lock + EOS channel boost)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    primary = heads[0]
    L0, H0 = primary["layer"], primary["head"]
    sc = H0 * d_head; ec = sc + d_head

    if model_name == "glm4":
        prompts = EN_PROMPTS_10[:3]
    else:
        prompts = EN_PROMPTS_10[:5]

    # Conditions: (name, lam, channel_scale, channel_layer, channel_idx)
    conditions = [
        ("normal",             0.0, 1.0, None, None),
        ("ablate_lock",        1.0, 1.0, None, None),
        ("reverse_lock_2.0",   2.0, 1.0, None, None),
        ("boost_eos_ch",       0.0, 3.0, eos_channel_layer, eos_channel_idx),
        ("rev2.0+boost_eos",   2.0, 3.0, eos_channel_layer, eos_channel_idx),
        ("ablate+boost_eos",   1.0, 3.0, eos_channel_layer, eos_channel_idx),
    ]

    all_results = []

    for pi, prompt in enumerate(prompts):
        for cond_name, lam, ch_scale, ch_layer, ch_idx in conditions:
            handles = []

            # Reverse lock via o_proj scaling (scale = 1-λ)
            if lam != 0.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))

            # Channel boost hook
            if ch_scale != 1.0 and ch_layer is not None:
                target_mlp = layers[ch_layer].mlp.down_proj
                handles.append(target_mlp.register_forward_pre_hook(
                    make_channel_hook([ch_idx], ch_scale)))

            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"; has_eos = False; n_gen = 0

            for h in handles: h.remove()

            ce = evaluate_strict_clean(prompt, generated, has_eos, n_gen)
            all_results.append({
                "prompt": prompt, "condition": cond_name,
                "generated": generated[:200], "has_eos": has_eos,
                "n_tokens": n_gen, "strict_clean": ce["strict_clean"],
                "is_ascii": ce["is_ascii"], "lang_switched": not ce["is_ascii"],
            })

        log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_results:
        c = r["condition"]; cond_agg[c]["eos"] += int(r["has_eos"])
        cond_agg[c]["clean"] += int(r["strict_clean"])
        cond_agg[c]["switch"] += int(r["lang_switched"])
        cond_agg[c]["n"] += 1; cond_agg[c]["toks"].append(r["n_tokens"])

    summary = {c: {"eos_rate": d["eos"]/max(d["n"],1),
                   "strict_clean_rate": d["clean"]/max(d["n"],1),
                   "lang_switch_rate": d["switch"]/max(d["n"],1),
                   "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
               for c, d in cond_agg.items()}

    output = {
        "task": "task4_combined", "model": model_name,
        "primary_head": f"L{L0}_H{H0}",
        "eos_channel": f"L{eos_channel_layer}_C{eos_channel_idx}",
        "summary": summary, "raw_results": all_results,
    }
    save_path = save_dir / "task4_combined.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for c in [c[0] for c in conditions]:
        s = summary.get(c, {})
        log(f"    {c:25s}: eos={s.get('eos_rate',0):.2f}  clean={s.get('strict_clean_rate',0):.2f}  "
            f"switch={s.get('lang_switch_rate',0):.2f}  toks={s.get('mean_tokens',0):.1f}")

    log(f"    Sample (prompt 0):")
    for r in all_results:
        if r["prompt"] == prompts[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")

    return output


# ============================================================
# TASK 5: De-headed Mode Direction
# ============================================================

def task5_deheaded_mode(model, tokenizer, device, info, model_name, heads, save_dir):
    """Reconstruct mode direction without L39_H21 contribution."""
    log("  Task 5: De-headed mode direction (5 EN + 5 CN)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    primary = heads[0]
    L0, H0 = primary["layer"], primary["head"]
    sc = H0 * d_head; ec = sc + d_head

    en_normal = defaultdict(list)   # layer -> [residual]
    en_ablated = defaultdict(list)  # layer -> [residual without head]
    cn_normal = defaultdict(list)
    cn_ablated = defaultdict(list)
    head_outputs_en = []  # O_h for each EN prompt

    all_prompts = [(p, "en") for p in EN_PROMPTS_10[:5]] + [(p, "cn") for p in CN_PROMPTS_10[:5]]

    for pi, (prompt, lang) in enumerate(all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Normal forward
        cap_n = {}
        hk_n = layers[L0].register_forward_hook(make_capture_hook(cap_n, "normal"))
        with torch.no_grad(): model(input_ids, use_cache=False)
        hk_n.remove()

        # Ablated forward (zero head)
        cap_a = {}
        hk_ab = layers[L0].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
        hk_ca = layers[L0].register_forward_hook(make_capture_hook(cap_a, "ablated"))
        with torch.no_grad(): model(input_ids, use_cache=False)
        hk_ab.remove(); hk_ca.remove()

        nv = cap_n.get("normal"); av = cap_a.get("ablated")
        if nv is not None and av is not None:
            if lang == "en":
                en_normal[L0].append(nv); en_ablated[L0].append(av)
                head_outputs_en.append(nv - av)  # O_h
            else:
                cn_normal[L0].append(nv); cn_ablated[L0].append(av)

        log(f"    {pi+1}/{len(all_prompts)} prompts")

    # Compute mode directions
    if en_normal[L0] and cn_normal[L0]:
        en_mean_n = np.mean(en_normal[L0], axis=0)
        cn_mean_n = np.mean(cn_normal[L0], axis=0)
        d_mode_normal = en_mean_n - cn_mean_n

        en_mean_a = np.mean(en_ablated[L0], axis=0)
        cn_mean_a = np.mean(cn_ablated[L0], axis=0)
        d_mode_deheaded = en_mean_a - cn_mean_a

        # Compute cosines
        mean_Oh = np.mean(head_outputs_en, axis=0)

        cos_normal = cosine_similarity(mean_Oh, d_mode_normal)
        cos_deheaded = cosine_similarity(mean_Oh, d_mode_deheaded)

        # Also compute per-prompt
        per_prompt_normal = [cosine_similarity(Oh, d_mode_normal) for Oh in head_outputs_en]
        per_prompt_deheaded = [cosine_similarity(Oh, d_mode_deheaded) for Oh in head_outputs_en]

        output = {
            "task": "task5_deheaded_mode", "model": model_name,
            "head": f"L{L0}_H{H0}",
            "d_mode_normal_norm": float(np.linalg.norm(d_mode_normal)),
            "d_mode_deheaded_norm": float(np.linalg.norm(d_mode_deheaded)),
            "cos_Oh_vs_d_mode_normal": cos_normal,
            "cos_Oh_vs_d_mode_deheaded": cos_deheaded,
            "mean_per_prompt_cos_normal": float(np.mean(per_prompt_normal)),
            "mean_per_prompt_cos_deheaded": float(np.mean(per_prompt_deheaded)),
            "norm_Oh": float(np.linalg.norm(mean_Oh)),
        }
    else:
        output = {"task": "task5_deheaded_mode", "model": model_name, "error": "insufficient data"}

    save_path = save_dir / "task5_deheaded_mode.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    if "cos_Oh_vs_d_mode_normal" in output:
        log(f"    d_mode_normal norm: {output['d_mode_normal_norm']:.3f}")
        log(f"    d_mode_deheaded norm: {output['d_mode_deheaded_norm']:.3f}")
        log(f"    cos(O_h, d_mode_normal)   = {output['cos_Oh_vs_d_mode_normal']:.4f}")
        log(f"    cos(O_h, d_mode_deheaded) = {output['cos_Oh_vs_d_mode_deheaded']:.4f}")
        log(f"    Per-prompt: normal={output['mean_per_prompt_cos_normal']:.4f}  deheaded={output['mean_per_prompt_cos_deheaded']:.4f}")

    return output


# ============================================================
# TASK 6: Large-scale Rollout
# ============================================================

def task6_large_rollout(model, tokenizer, device, info, model_name, heads,
                        eos_channel_layer, eos_channel_idx, save_dir):
    """Large-scale rollout with best conditions."""
    log("  Task 6: Large-scale rollout...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    primary = heads[0]
    L0, H0 = primary["layer"], primary["head"]
    sc = H0 * d_head; ec = sc + d_head

    if model_name == "glm4":
        prompts = EN_PROMPTS_10[:5]
    else:
        prompts = EN_PROMPTS_50[:15]

    conditions = [
        ("normal",             0.0, 1.0, None, None),
        ("ablate_lock",        1.0, 1.0, None, None),
        ("reverse_lock_2.0",   2.0, 1.0, None, None),
        ("rev2.0+boost_eos",   2.0, 3.0, eos_channel_layer, eos_channel_idx),
    ]

    all_results = []

    for pi, prompt in enumerate(prompts):
        for cond_name, lam, ch_scale, ch_layer, ch_idx in conditions:
            handles = []

            if lam != 0.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))

            if ch_scale != 1.0 and ch_layer is not None:
                handles.append(layers[ch_layer].mlp.down_proj.register_forward_pre_hook(
                    make_channel_hook([ch_idx], ch_scale)))

            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"; has_eos = False; n_gen = 0

            for h in handles: h.remove()

            ce = evaluate_strict_clean(prompt, generated, has_eos, n_gen)
            all_results.append({
                "prompt": prompt, "condition": cond_name,
                "generated": generated[:200], "has_eos": has_eos,
                "n_tokens": n_gen, "strict_clean": ce["strict_clean"],
                "lang_switched": not ce["is_ascii"],
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_results:
        c = r["condition"]; cond_agg[c]["eos"] += int(r["has_eos"])
        cond_agg[c]["clean"] += int(r["strict_clean"])
        cond_agg[c]["switch"] += int(r["lang_switched"])
        cond_agg[c]["n"] += 1; cond_agg[c]["toks"].append(r["n_tokens"])

    summary = {c: {"eos_rate": d["eos"]/max(d["n"],1),
                   "strict_clean_rate": d["clean"]/max(d["n"],1),
                   "lang_switch_rate": d["switch"]/max(d["n"],1),
                   "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
               for c, d in cond_agg.items()}

    output = {
        "task": "task6_large_rollout", "model": model_name,
        "primary_head": f"L{L0}_H{H0}",
        "n_prompts": len(prompts), "max_tokens": MAX_TOKENS,
        "summary": summary, "raw_results": all_results,
    }
    save_path = save_dir / "task6_large_rollout.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for c in [c[0] for c in conditions]:
        s = summary.get(c, {})
        log(f"    {c:25s}: eos={s.get('eos_rate',0):.2f}  clean={s.get('strict_clean_rate',0):.2f}  "
            f"switch={s.get('lang_switch_rate',0):.2f}  toks={s.get('mean_tokens',0):.1f}")

    return output


# ============================================================
# MODEL RUNNER
# ============================================================

def run_model(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 962: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    heads = MODEL_HEADS.get(model_name, [])
    results = {"model": model_name}
    t_start = time.time()

    # Task 1: EOS head search
    try:
        results["task1"] = task1_eos_head_search(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}"); import traceback; traceback.print_exc()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 2: EOS channel search
    eos_ch_layer = n_layers_backup = info.n_layers - 1
    eos_ch_idx = 0
    try:
        t2_result, ch_layers, ch_idxs = task2_eos_channel_search(model, tokenizer, device, info, model_name, model_dir)
        results["task2"] = t2_result
        if ch_layers: eos_ch_layer = ch_layers[0]
        if ch_idxs: eos_ch_idx = ch_idxs[0]
    except Exception as e:
        log(f"  Task 2 FAILED: {e}"); import traceback; traceback.print_exc()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 3: Reverse lock
    try:
        results["task3"] = task3_reverse_lock(model, tokenizer, device, info, model_name, heads, model_dir)
    except Exception as e:
        log(f"  Task 3 FAILED: {e}"); import traceback; traceback.print_exc()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 4: Combined
    try:
        results["task4"] = task4_combined_intervention(
            model, tokenizer, device, info, model_name, heads,
            eos_ch_layer, eos_ch_idx, model_dir)
    except Exception as e:
        log(f"  Task 4 FAILED: {e}"); import traceback; traceback.print_exc()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 5: De-headed mode direction
    try:
        results["task5"] = task5_deheaded_mode(model, tokenizer, device, info, model_name, heads, model_dir)
    except Exception as e:
        log(f"  Task 5 FAILED: {e}"); import traceback; traceback.print_exc()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 6: Large rollout
    try:
        results["task6"] = task6_large_rollout(
            model, tokenizer, device, info, model_name, heads,
            eos_ch_layer, eos_ch_idx, model_dir)
    except Exception as e:
        log(f"  Task 6 FAILED: {e}"); import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    release_model(model)

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=EOS_head_search, 2=EOS_channel_search, 3=reverse_lock, 4=combined, 5=deheaded_mode, 6=large_rollout")

    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
