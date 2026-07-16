#!/usr/bin/env python3
"""
Phase 961: L39_H21 Mode-Head Mechanism and Cross-Model Functional Equivalence Audit
===================================================================================
Task 1: Attention pattern analysis — what tokens does the head attend to?
Task 2: Value output direction — what does the head write to residual stream?
Task 3: Mode direction construction and cosine similarity
Task 4: Boost failure analysis — multiple alpha values
Task 5: Head + EOS channel joint intervention
Task 6: Cross-model functional equivalence audit

Models: qwen3 -> GLM4 -> DS7B (sequential, one at a time to avoid GPU OOM)
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

PHASE = 961
RESULT_DIR = Path("results/phase961_mode_head_mechanism")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_HEADS = {
    "qwen3": [
        {"layer": 35, "head": 0, "role": "period"},
        {"layer": 33, "head": 8, "role": "period"},
    ],
    "glm4": [
        {"layer": 39, "head": 21, "role": "EOS/mode"},   # Key head from Phase 960
        {"layer": 38, "head": 0, "role": "EOS-logit-only"},
        {"layer": 38, "head": 7, "role": "EOS-logit-only"},
    ],
    "deepseek7b": [
        {"layer": 26, "head": 19, "role": "space"},
        {"layer": 26, "head": 25, "role": "space"},
    ],
}

SUPER_CHANNELS = {
    "qwen3": [935, 36, 284, 153, 188],
    "glm4": [12274, 7968, 5155, 5902, 1106],
    "deepseek7b": [15791, 15305, 1106, 4985, 14464],
}

# 50 English factual prompts (for attention pattern analysis)
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

# 20 Chinese prompts (semantically matched to EN_PROMPTS_50[:20])
CN_PROMPTS_20 = [
    "法国的首都是", "最大的行星是", "水的沸点是", "光速是", "太阳是一颗",
    "狗是", "天空是", "草是", "火需要", "冰是",
    "地球是", "三角形有", "莎士比亚是", "东京是", "太平洋是",
    "金是一种", "植物需要", "人类呼吸", "月亮是", "鸟能",
]

ROLLOUT_PROMPTS_20 = EN_PROMPTS_50[:20]
MAX_TOKENS_ROLLOUT = 30

EXPECTED_ANSWERS = {
    "The capital of France is": "Paris", "The largest planet is": "Jupiter",
    "Water boils at": "100", "The speed of light is": "299",
    "The sun is a": "star", "Dogs are": "animal", "The sky is": "blue",
    "Grass is": "green", "Fire needs": "oxygen", "Ice is": "frozen",
    "The Earth is": "round", "A triangle has": "three",
    "Shakespeare was": "English", "Tokyo is the capital of": "Japan",
    "The Pacific Ocean is": "large", "Gold is a": "metal",
    "Plants need": "water", "Humans breathe": "oxygen",
    "The moon is": "natural", "Birds can": "fly",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_input_device(model):
    """Get the device where input embeddings live."""
    return model.get_input_embeddings().weight.device


def get_head_dims(model, info):
    """Get n_heads and d_head from the model, using o_proj weight shape for safety."""
    n_heads = getattr(model.config, "num_attention_heads", 32)
    layers = get_layers(model)
    o_proj_in = layers[0].self_attn.o_proj.weight.shape[1]
    d_head = o_proj_in // n_heads
    return n_heads, d_head


def classify_token(token_str: str) -> str:
    """Classify a decoded token string into a category."""
    s = token_str.strip()
    if not s or s == " ":
        return "space"
    if s in [".", ",", ";", ":", "!", "?", "。", "，", "；", "：", "！", "？", "\n", "\r"]:
        return "punct"
    if s.lower() in ["is", "a", "an", "the", "of", "in", "on", "at", "to", "for",
                     "and", "or", "but", "are", "was", "were", "be", "been", "being",
                     "has", "have", "had", "do", "does", "did", "will", "would",
                     "can", "could", "should", "not", "no", "yes"]:
        return "function"
    if s.startswith("<") and s.endswith(">"):
        return "special"
    # CJK characters
    if any('\u4e00' <= c <= '\u9fff' for c in s):
        return "cjk"
    # Check if mostly digits
    if s.replace(".", "").replace(",", "").isdigit():
        return "number"
    return "content"


def get_proto_token_ids(tokenizer):
    """Get token IDs for protocol tokens."""
    ids = {}
    if tokenizer.eos_token_id is not None:
        ids["EOS"] = tokenizer.eos_token_id
    for tok_str in [".", " ", "\n", " is", " a", " the", "是", "的", "\n根据"]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks:
            ids[tok_str] = toks[0]
    return ids


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_head_hook(sc: int, ec: int, scale: float):
    """Hook to zero (scale=0) or boost (scale>1) a specific head's o_proj input."""
    def hook(module, args):
        if scale == 1.0:
            return None
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            if scale == 0.0:
                patched[:, :, sc:ec] = 0
            else:
                patched[:, :, sc:ec] = patched[:, :, sc:ec] * scale
        return (patched,)
    return hook


def make_channel_hook(channels, scale: float):
    """Hook to zero/boost specific MLP channels in down_proj input (last position)."""
    ch_list = [int(c) for c in channels]
    def hook(module, args):
        if scale == 1.0:
            return None
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            for c in ch_list:
                if c < patched.shape[-1]:
                    if scale == 0.0:
                        patched[:, -1, c] = 0
                    else:
                        patched[:, -1, c] = patched[:, -1, c] * scale
        return (patched,)
    return hook


def make_capture_hook(captured: dict, key: str):
    """Hook to capture layer output at last position as numpy."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        captured[key] = out[0, -1, :].detach().float().cpu().numpy()
    return hook


def evaluate_strict_clean(prompt, generated, has_eos, n_tokens):
    """Evaluate strict-clean criteria."""
    expected = EXPECTED_ANSWERS.get(prompt, "")
    is_ascii = all(ord(c) < 256 for c in generated)
    is_short = 0 < n_tokens < 15
    has_expected = (expected == "") or (expected.lower() in generated.lower())
    strict_clean = has_eos and is_short and has_expected and is_ascii
    return {
        "strict_clean": strict_clean,
        "has_eos": has_eos,
        "is_short": is_short,
        "has_expected": has_expected,
        "is_ascii": is_ascii,
    }


# ============================================================
# TASK 1: Attention Pattern Analysis
# ============================================================

def task1_attention_pattern(model, tokenizer, device, info, model_name, heads, save_dir):
    """Analyze what tokens each head attends to across 50 prompts."""
    log("  Task 1: Attention pattern analysis (50 prompts)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    # Attention mass accumulation: {head_key: {category: total_mass}}
    attn_mass = defaultdict(lambda: defaultdict(float))
    attn_entropy = defaultdict(list)
    # Per-position attention for first prompt (for detailed view)
    detail_attn = {}

    prompt_count = 0
    for pi, prompt in enumerate(EN_PROMPTS_50):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 3:
            continue

        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True, use_cache=False)
        except Exception as e:
            log(f"    output_attentions failed for prompt {pi}: {e}")
            continue

        attentions = outputs.attentions
        if attentions is None:
            log("    attentions is None, skipping Task 1")
            return None

        # Decode input tokens for classification
        token_strs = []
        for i in range(seq_len):
            tid = input_ids[0, i].item()
            token_strs.append(tokenizer.decode([tid]))

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            if L >= len(attentions) or L >= n_layers:
                continue
            attn_tensor = attentions[L]  # [1, n_heads, seq, seq]
            if H >= attn_tensor.shape[1]:
                continue

            # Last token's attention to all previous tokens
            head_attn = attn_tensor[0, H, -1, :].float().cpu().numpy()  # [seq_len]

            key = f"L{L}_H{H}"
            for i in range(seq_len):
                cat = classify_token(token_strs[i])
                attn_mass[key][cat] += float(head_attn[i])

            # Entropy
            p = head_attn[head_attn > 1e-10]
            if len(p) > 0:
                ent = -np.sum(p * np.log(p))
                norm_ent = ent / max(math.log(seq_len), 1e-10)
                attn_entropy[key].append(float(norm_ent))

            # Save detail for first prompt
            if pi == 0:
                detail_attn[key] = {
                    "prompt": prompt,
                    "tokens": token_strs,
                    "attention": head_attn.tolist(),
                    "categories": [classify_token(t) for t in token_strs],
                }

        prompt_count += 1
        del outputs, attentions
        if (pi + 1) % 10 == 0:
            log(f"    {pi+1}/{len(EN_PROMPTS_50)} prompts")

    # Normalize attention mass
    summary = {}
    for key, cats in attn_mass.items():
        total = sum(cats.values())
        summary[key] = {
            cat: (mass / total if total > 0 else 0.0)
            for cat, mass in cats.items()
        }
        summary[key]["n_prompts"] = prompt_count
        summary[key]["mean_entropy"] = float(np.mean(attn_entropy[key])) if attn_entropy[key] else 0.0
        summary[key]["total_mass"] = total / max(prompt_count, 1)

    output = {
        "task": "task1_attention_pattern",
        "model": model_name,
        "n_prompts": prompt_count,
        "heads": [(h["layer"], h["head"], h["role"]) for h in heads],
        "attention_mass_by_category": summary,
        "detail_first_prompt": detail_attn,
    }

    save_path = save_dir / "task1_attention_pattern.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print summary
    for key in summary:
        s = summary[key]
        cats_str = "  ".join(f"{c}={v:.3f}" for c, v in sorted(s.items())
                             if c not in ["n_prompts", "mean_entropy", "total_mass"])
        log(f"    {key}: entropy={s.get('mean_entropy', 0):.3f}  {cats_str}")

    return output


# ============================================================
# TASK 2+3: Head Output Direction + Mode Direction
# ============================================================

def task2_3_head_output_and_mode(model, tokenizer, device, info, model_name, heads, save_dir):
    """Extract head output direction O_h and mode direction d_mode."""
    log("  Task 2+3: Head output direction + mode direction (20 EN + 20 CN)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    # Get W_U for direction projection
    try:
        W_U = get_W_U(model, model_name)  # [vocab, d_model]
        log(f"    W_U loaded: {W_U.shape}")
    except Exception as e:
        log(f"    W_U load failed: {e}, skipping projection")
        W_U = None

    proto_ids = get_proto_token_ids(tokenizer)

    # Collect layer outputs for English and Chinese prompts
    en_layer_outputs = defaultdict(list)  # {layer_idx: [residual vectors]}
    cn_layer_outputs = defaultdict(list)
    head_outputs = defaultdict(lambda: {"en": [], "cn": []})  # {head_key: {"en": [O_h], "cn": [O_h]}}

    all_prompts = [(p, "en") for p in EN_PROMPTS_50[:20]] + [(p, "cn") for p in CN_PROMPTS_20]

    for pi, (prompt, lang) in enumerate(all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Forward pass 1: normal — capture layer outputs at each head's layer
        captured_normal = {}
        normal_hooks = []
        for h_info in heads:
            L = h_info["layer"]
            key = f"normal_L{L}"
            if key not in captured_normal:
                hk = layers[L].register_forward_hook(make_capture_hook(captured_normal, key))
                normal_hooks.append(hk)

        with torch.no_grad():
            model(input_ids, use_cache=False)
        for hk in normal_hooks:
            hk.remove()

        # Forward pass 2: ablated — for each head, zero it and capture layer output
        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc = H * d_head
            ec = sc + d_head

            captured_ablated = {}
            hk_ablate = layers[L].self_attn.o_proj.register_forward_pre_hook(
                make_head_hook(sc, ec, 0.0)
            )
            hk_capture = layers[L].register_forward_hook(
                make_capture_hook(captured_ablated, f"ablated_L{L}")
            )

            with torch.no_grad():
                model(input_ids, use_cache=False)

            hk_ablate.remove()
            hk_capture.remove()

            # O_h = normal - ablated
            normal_vec = captured_normal.get(f"normal_L{L}")
            ablated_vec = captured_ablated.get(f"ablated_L{L}")
            if normal_vec is not None and ablated_vec is not None:
                O_h = normal_vec - ablated_vec
                head_key = f"L{L}_H{H}"
                head_outputs[head_key][lang].append(O_h)

        # Store layer outputs for mode direction
        for h_info in heads:
            L = h_info["layer"]
            key = f"normal_L{L}"
            if key in captured_normal:
                if lang == "en":
                    en_layer_outputs[L].append(captured_normal[key])
                else:
                    cn_layer_outputs[L].append(captured_normal[key])

        if (pi + 1) % 10 == 0:
            log(f"    {pi+1}/{len(all_prompts)} prompts")

    # Compute mode direction: d_mode = mean(EN) - mean(CN) at each layer
    mode_directions = {}
    for L in en_layer_outputs:
        en_mean = np.mean(en_layer_outputs[L], axis=0)
        cn_mean = np.mean(cn_layer_outputs[L], axis=0)
        d_mode = en_mean - cn_mean
        mode_directions[L] = {
            "d_mode": d_mode.tolist(),
            "en_mean_norm": float(np.linalg.norm(en_mean)),
            "cn_mean_norm": float(np.linalg.norm(cn_mean)),
            "d_mode_norm": float(np.linalg.norm(d_mode)),
        }

    # Compute cosine similarities and W_U projections
    head_analysis = {}
    for h_info in heads:
        L, H = h_info["layer"], h_info["head"]
        head_key = f"L{L}_H{H}"

        en_Ohs = head_outputs[head_key]["en"]
        cn_Ohs = head_outputs[head_key]["cn"]

        if not en_Ohs:
            continue

        # Mean O_h for English and Chinese
        mean_Oh_en = np.mean(en_Ohs, axis=0)
        mean_Oh_cn = np.mean(cn_Ohs, axis=0) if cn_Ohs else np.zeros_like(mean_Oh_en)

        # Norms
        norm_en = float(np.linalg.norm(mean_Oh_en))
        norm_cn = float(np.linalg.norm(mean_Oh_cn))

        # Mode direction cosine
        d_mode = np.array(mode_directions.get(L, {}).get("d_mode", [0] * len(mean_Oh_en)))
        cos_mode_en = cosine_similarity(mean_Oh_en, d_mode)
        cos_mode_cn = cosine_similarity(mean_Oh_cn, d_mode)

        # W_U direction cosines (for protocol tokens)
        wu_cosines = {}
        if W_U is not None:
            for tok_name, tid in proto_ids.items():
                if tid < W_U.shape[0]:
                    wu_dir = W_U[tid]
                    wu_cosines[tok_name] = cosine_similarity(mean_Oh_en, wu_dir)

            # Top promoted tokens: delta_logits = W_U @ O_h
            delta_logits = W_U @ mean_Oh_en
            top_idx = np.argsort(delta_logits)[-15:][::-1]
            top_promoted = []
            for tid in top_idx:
                tok_str = tokenizer.decode([int(tid)])
                top_promoted.append({
                    "token": tok_str,
                    "token_id": int(tid),
                    "delta_logit": float(delta_logits[tid]),
                })
        else:
            top_promoted = []

        # Per-prompt cosine with mode direction
        per_prompt_cos = []
        for i, Oh in enumerate(en_Ohs):
            per_prompt_cos.append(cosine_similarity(Oh, d_mode))

        head_analysis[head_key] = {
            "layer": L, "head": H, "role": h_info["role"],
            "norm_Oh_en": norm_en,
            "norm_Oh_cn": norm_cn,
            "cos_Oh_en_vs_dmode": cos_mode_en,
            "cos_Oh_cn_vs_dmode": cos_mode_cn,
            "wu_cosines": wu_cosines,
            "top_promoted_tokens_en": top_promoted,
            "per_prompt_cos_mode_en": per_prompt_cos,
            "mean_per_prompt_cos_mode_en": float(np.mean(per_prompt_cos)) if per_prompt_cos else 0.0,
            "std_per_prompt_cos_mode_en": float(np.std(per_prompt_cos)) if per_prompt_cos else 0.0,
        }

    output = {
        "task": "task2_3_head_output_mode",
        "model": model_name,
        "n_heads": n_heads,
        "d_head": d_head,
        "n_en_prompts": len(EN_PROMPTS_50[:20]),
        "n_cn_prompts": len(CN_PROMPTS_20),
        "mode_directions": {str(k): v for k, v in mode_directions.items()},
        "head_analysis": head_analysis,
    }

    save_path = save_dir / "task2_3_head_output_mode.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print summary
    for key, ha in head_analysis.items():
        log(f"    {key} ({ha['role']}): ||O_h||_en={ha['norm_Oh_en']:.3f}  "
            f"cos(O_h_en, d_mode)={ha['cos_Oh_en_vs_dmode']:.4f}  "
            f"cos(O_h_cn, d_mode)={ha['cos_Oh_cn_vs_dmode']:.4f}")
        if ha["wu_cosines"]:
            top_wu = sorted(ha["wu_cosines"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            wu_str = "  ".join(f"{n}={v:.4f}" for n, v in top_wu)
            log(f"      W_U cos: {wu_str}")
        if ha["top_promoted_tokens_en"]:
            top3 = ha["top_promoted_tokens_en"][:3]
            tok_str = "  ".join(f"'{t['token']}'({t['delta_logit']:.3f})" for t in top3)
            log(f"      Top promoted: {tok_str}")

    return output


# ============================================================
# TASK 4: Boost Failure Analysis
# ============================================================

def task4_boost_analysis(model, tokenizer, device, info, model_name, heads, save_dir):
    """Analyze why boost doesn't increase EOS rate with multiple alpha values."""
    log("  Task 4: Boost failure analysis (multiple alphas)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    proto_ids = get_proto_token_ids(tokenizer)

    alphas = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    prompts_20 = EN_PROMPTS_50[:20]

    # Part A: Logit analysis (20 prompts × 6 alphas)
    logit_results = defaultdict(lambda: defaultdict(list))

    for pi, prompt in enumerate(prompts_20):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Normal logits
        with torch.no_grad():
            base_out = model(input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        base_top1 = float(np.sort(base_logits)[-1])
        base_top2 = float(np.sort(base_logits)[-2])
        base_margin = base_top1 - base_top2

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc = H * d_head
            ec = sc + d_head
            head_key = f"L{L}_H{H}"

            for alpha in alphas:
                if alpha == 1.0:
                    # Use base logits
                    patched_logits = base_logits.copy()
                else:
                    handle = layers[L].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(sc, ec, alpha)
                    )
                    try:
                        with torch.no_grad():
                            out = model(input_ids, use_cache=False)
                            patched_logits = out.logits[0, -1].detach().float().cpu().numpy()
                    except:
                        patched_logits = base_logits.copy()
                    handle.remove()

                # Measure deltas
                patched_argmax = int(np.argmax(patched_logits))
                argmax_changed = int(patched_argmax != base_argmax)
                patched_top1 = float(np.sort(patched_logits)[-1])
                patched_top2 = float(np.sort(patched_logits)[-2])

                for tok_name, tid in proto_ids.items():
                    if tid < len(patched_logits):
                        delta = float(patched_logits[tid] - base_logits[tid])
                        logit_results[head_key][f"delta_{tok_name}_a{alpha}"].append(delta)

                logit_results[head_key][f"argmax_changed_a{alpha}"].append(argmax_changed)
                logit_results[head_key][f"margin_a{alpha}"].append(patched_top1 - patched_top2)
                logit_results[head_key][f"base_margin"].append(base_margin)

        if (pi + 1) % 10 == 0:
            log(f"    Logit analysis: {pi+1}/{len(prompts_20)} prompts")

    # Aggregate logit analysis
    logit_summary = {}
    for head_key, data in logit_results.items():
        logit_summary[head_key] = {}
        for metric, values in data.items():
            logit_summary[head_key][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    # Part B: Rollout with selected alphas (8 prompts × 4 alphas)
    rollout_alphas = [1.0, 1.5, 2.0, 3.0]
    rollout_prompts = prompts_20[:8]
    rollout_results = []

    # Find the primary head (first in the list)
    primary_head = heads[0]
    L0, H0 = primary_head["layer"], primary_head["head"]
    sc0 = H0 * d_head
    ec0 = sc0 + d_head

    for pi, prompt in enumerate(rollout_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for alpha in rollout_alphas:
            handles = []
            if alpha != 1.0:
                handles.append(
                    layers[L0].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(sc0, ec0, alpha)
                    )
                )

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids, max_new_tokens=MAX_TOKENS_ROLLOUT,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"
                has_eos = False
                n_gen = 0

            for h in handles:
                h.remove()

            is_ascii = all(ord(c) < 256 for c in generated)
            rollout_results.append({
                "prompt": prompt,
                "alpha": alpha,
                "head": f"L{L0}_H{H0}",
                "generated": generated[:200],
                "has_eos": has_eos,
                "n_tokens": n_gen,
                "is_ascii": is_ascii,
                "lang_switched": not is_ascii,
            })

        if (pi + 1) % 4 == 0:
            log(f"    Rollout: {pi+1}/{len(rollout_prompts)} prompts")

    # Aggregate rollout
    rollout_summary = {}
    for alpha in rollout_alphas:
        alpha_results = [r for r in rollout_results if r["alpha"] == alpha]
        rollout_summary[f"alpha_{alpha}"] = {
            "eos_rate": np.mean([r["has_eos"] for r in alpha_results]) if alpha_results else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in alpha_results]) if alpha_results else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in alpha_results]) if alpha_results else 0,
            "n": len(alpha_results),
        }

    output = {
        "task": "task4_boost_analysis",
        "model": model_name,
        "alphas_logit": alphas,
        "alphas_rollout": rollout_alphas,
        "primary_head": f"L{L0}_H{H0}",
        "logit_summary": logit_summary,
        "rollout_summary": rollout_summary,
        "rollout_results": rollout_results,
    }

    save_path = save_dir / "task4_boost_analysis.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print summary
    for head_key in logit_summary:
        s = logit_summary[head_key]
        log(f"    {head_key}:")
        for alpha in alphas:
            eos_delta_key = f"delta_EOS_a{alpha}"
            if eos_delta_key in s:
                log(f"      alpha={alpha}: ΔEOS={s[eos_delta_key]['mean']:.4f}  "
                    f"argmax_change={s.get(f'argmax_changed_a{alpha}', {}).get('mean', 0):.2f}  "
                    f"margin={s.get(f'margin_a{alpha}', {}).get('mean', 0):.4f}")

    log(f"    Rollout summary (L{L0}_H{H0}):")
    for alpha in rollout_alphas:
        rs = rollout_summary.get(f"alpha_{alpha}", {})
        log(f"      alpha={alpha}: eos={rs.get('eos_rate', 0):.2f}  "
            f"switch={rs.get('lang_switch_rate', 0):.2f}  tokens={rs.get('mean_tokens', 0):.1f}")

    return output


# ============================================================
# TASK 5: Head + EOS Channel Joint Intervention
# ============================================================

def task5_joint_intervention(model, tokenizer, device, info, model_name, heads, channels, save_dir):
    """Test joint head + channel intervention for strict-clean improvement."""
    log("  Task 5: Head + channel joint intervention (10 prompts)...")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers

    primary_head = heads[0]
    L0, H0 = primary_head["layer"], primary_head["head"]
    sc0 = H0 * d_head
    ec0 = sc0 + d_head

    last_layer = n_layers - 1
    target_mlp = layers[last_layer].mlp.down_proj
    top_channels = channels[:5]

    # Conditions: (name, head_scale, channel_scale)
    conditions = [
        ("normal",             1.0, 1.0),
        ("ablate_head",        0.0, 1.0),
        ("boost_channel",      1.0, 2.0),
        ("ablate_head+boost_ch", 0.0, 2.0),
        ("boost_head+boost_ch",  2.0, 2.0),
    ]

    rollout_prompts = EN_PROMPTS_50[:10]
    all_results = []

    for pi, prompt in enumerate(rollout_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cond_name, head_scale, ch_scale in conditions:
            handles = []

            # Head hook
            if head_scale != 1.0:
                handles.append(
                    layers[L0].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(sc0, ec0, head_scale)
                    )
                )

            # Channel hook
            if ch_scale != 1.0:
                handles.append(
                    target_mlp.register_forward_pre_hook(
                        make_channel_hook(top_channels, ch_scale)
                    )
                )

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids, max_new_tokens=MAX_TOKENS_ROLLOUT,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"
                has_eos = False
                n_gen = 0

            for h in handles:
                h.remove()

            clean_eval = evaluate_strict_clean(prompt, generated, has_eos, n_gen)
            all_results.append({
                "prompt": prompt,
                "condition": cond_name,
                "generated": generated[:200],
                "has_eos": has_eos,
                "n_tokens": n_gen,
                "strict_clean": clean_eval["strict_clean"],
                "is_ascii": clean_eval["is_ascii"],
                "lang_switched": not clean_eval["is_ascii"],
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(rollout_prompts)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {
        "eos_count": 0, "clean_count": 0, "switch_count": 0, "n": 0, "tokens": []
    })
    for r in all_results:
        c = r["condition"]
        cond_agg[c]["eos_count"] += int(r["has_eos"])
        cond_agg[c]["clean_count"] += int(r["strict_clean"])
        cond_agg[c]["switch_count"] += int(r["lang_switched"])
        cond_agg[c]["n"] += 1
        cond_agg[c]["tokens"].append(r["n_tokens"])

    summary = {}
    for c, d in cond_agg.items():
        summary[c] = {
            "eos_rate": d["eos_count"] / max(d["n"], 1),
            "strict_clean_rate": d["clean_count"] / max(d["n"], 1),
            "lang_switch_rate": d["switch_count"] / max(d["n"], 1),
            "mean_tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
            "n": d["n"],
        }

    output = {
        "task": "task5_joint_intervention",
        "model": model_name,
        "primary_head": f"L{L0}_H{H0}",
        "channels": top_channels,
        "conditions": [c[0] for c in conditions],
        "summary": summary,
        "raw_results": all_results,
    }

    save_path = save_dir / "task5_joint_intervention.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for c in [c[0] for c in conditions]:
        s = summary.get(c, {})
        log(f"    {c:25s}: eos={s.get('eos_rate', 0):.2f}  "
            f"clean={s.get('strict_clean_rate', 0):.2f}  "
            f"switch={s.get('lang_switch_rate', 0):.2f}  "
            f"tokens={s.get('mean_tokens', 0):.1f}")

    # Print sample for first prompt
    log(f"    Sample (prompt 0: '{rollout_prompts[0][:30]}'):")
    for r in all_results:
        if r["prompt"] == rollout_prompts[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  "
                f"clean={r['strict_clean']}  text={r['generated'][:80]}")

    return output


# ============================================================
# MODEL RUNNER
# ============================================================

def run_model(model_name: str) -> dict:
    log(f"\n{'='*60}")
    log(f"Phase 961: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    n_heads, d_head = get_head_dims(model, info)
    log(f"  n_heads={n_heads}, d_head={d_head}")

    heads = MODEL_HEADS.get(model_name, [])
    channels = SUPER_CHANNELS.get(model_name, [])

    # Filter heads that exist in this model
    valid_heads = [h for h in heads if h["layer"] < info.n_layers and h["head"] < n_heads]
    if len(valid_heads) < len(heads):
        log(f"  Warning: filtered {len(heads) - len(valid_heads)} invalid heads")

    results = {"model": model_name, "heads": [(h["layer"], h["head"], h["role"]) for h in valid_heads]}
    t_start = time.time()

    # Task 1: Attention pattern
    try:
        results["task1"] = task1_attention_pattern(model, tokenizer, device, info, model_name, valid_heads, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Task 2+3: Head output + mode direction
    try:
        results["task2_3"] = task2_3_head_output_and_mode(model, tokenizer, device, info, model_name, valid_heads, model_dir)
    except Exception as e:
        log(f"  Task 2+3 FAILED: {e}")
        import traceback; traceback.print_exc()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Task 4: Boost analysis
    try:
        results["task4"] = task4_boost_analysis(model, tokenizer, device, info, model_name, valid_heads, model_dir)
    except Exception as e:
        log(f"  Task 4 FAILED: {e}")
        import traceback; traceback.print_exc()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Task 5: Joint intervention
    try:
        results["task5"] = task5_joint_intervention(model, tokenizer, device, info, model_name, valid_heads, channels, model_dir)
    except Exception as e:
        log(f"  Task 5 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    release_model(model)
    log(f"  {model_name} complete")
    return results


# ============================================================
# TASK 6: Cross-Model Comparison
# ============================================================

def task6_cross_model(all_results: list, save_dir: Path):
    """Compare head mechanisms across models."""
    log("\n" + "="*60)
    log("Task 6: Cross-model functional equivalence comparison")
    log("="*60)

    comparison = {
        "attention_patterns": [],
        "head_output_mode": [],
        "boost_effects": [],
        "joint_intervention": [],
    }

    for mr in all_results:
        model_name = mr["model"]

        # Task 1: Attention patterns
        t1 = mr.get("task1")
        if t1:
            for head_key, mass in t1.get("attention_mass_by_category", {}).items():
                comparison["attention_patterns"].append({
                    "model": model_name,
                    "head": head_key,
                    "content": mass.get("content", 0),
                    "punct": mass.get("punct", 0),
                    "function": mass.get("function", 0),
                    "space": mass.get("space", 0),
                    "special": mass.get("special", 0),
                    "cjk": mass.get("cjk", 0),
                    "number": mass.get("number", 0),
                    "entropy": mass.get("mean_entropy", 0),
                })

        # Task 2+3: Head output + mode
        t23 = mr.get("task2_3")
        if t23:
            for head_key, ha in t23.get("head_analysis", {}).items():
                top_promoted = ha.get("top_promoted_tokens_en", [])
                top3_strs = [t["token"] for t in top_promoted[:3]]
                comparison["head_output_mode"].append({
                    "model": model_name,
                    "head": head_key,
                    "role": ha["role"],
                    "norm_Oh_en": ha["norm_Oh_en"],
                    "cos_Oh_en_vs_dmode": ha["cos_Oh_en_vs_dmode"],
                    "cos_Oh_cn_vs_dmode": ha["cos_Oh_cn_vs_dmode"],
                    "mean_cos_per_prompt": ha.get("mean_per_prompt_cos_mode_en", 0),
                    "top_promoted_tokens": top3_strs,
                    "wu_cos_EOS": ha.get("wu_cosines", {}).get("EOS", 0),
                    "wu_cos_period": ha.get("wu_cosines", {}).get(".", 0),
                    "wu_cos_space": ha.get("wu_cosines", {}).get(" ", 0),
                })

        # Task 4: Boost effects
        t4 = mr.get("task4")
        if t4:
            for head_key, s in t4.get("logit_summary", {}).items():
                comparison["boost_effects"].append({
                    "model": model_name,
                    "head": head_key,
                    "delta_EOS_a1.5": s.get("delta_EOS_a1.5", {}).get("mean", 0),
                    "delta_EOS_a2.0": s.get("delta_EOS_a2.0", {}).get("mean", 0),
                    "delta_EOS_a3.0": s.get("delta_EOS_a3.0", {}).get("mean", 0),
                    "argmax_change_a2.0": s.get("argmax_changed_a2.0", {}).get("mean", 0),
                    "delta_period_a2.0": s.get("delta_.a2.0", {}).get("mean", 0),
                })

            rs = t4.get("rollout_summary", {})
            for alpha_key, alpha_data in rs.items():
                comparison["boost_effects"].append({
                    "model": model_name,
                    "head": t4.get("primary_head", ""),
                    "type": "rollout",
                    "alpha": alpha_key,
                    "eos_rate": alpha_data.get("eos_rate", 0),
                    "lang_switch_rate": alpha_data.get("lang_switch_rate", 0),
                    "mean_tokens": alpha_data.get("mean_tokens", 0),
                })

        # Task 5: Joint intervention
        t5 = mr.get("task5")
        if t5:
            for cond, s in t5.get("summary", {}).items():
                comparison["joint_intervention"].append({
                    "model": model_name,
                    "head": t5.get("primary_head", ""),
                    "condition": cond,
                    "eos_rate": s.get("eos_rate", 0),
                    "strict_clean_rate": s.get("strict_clean_rate", 0),
                    "lang_switch_rate": s.get("lang_switch_rate", 0),
                    "mean_tokens": s.get("mean_tokens", 0),
                })

    # Print comparison tables
    log("\n--- Attention Pattern Comparison ---")
    log(f"{'Model':<12} {'Head':<12} {'Content':>8} {'Punct':>8} {'Func':>8} {'Space':>8} {'Special':>8} {'Entropy':>8}")
    for ap in comparison["attention_patterns"]:
        log(f"{ap['model']:<12} {ap['head']:<12} {ap['content']:>8.3f} {ap['punct']:>8.3f} "
            f"{ap['function']:>8.3f} {ap['space']:>8.3f} {ap['special']:>8.3f} {ap['entropy']:>8.3f}")

    log("\n--- Head Output vs Mode Direction ---")
    log(f"{'Model':<12} {'Head':<12} {'Role':<16} {'||O_h||':>8} {'cos_en':>8} {'cos_cn':>8} {'cos_EOS':>8} {'cos_per':>8} {'Top3':>30}")
    for ho in comparison["head_output_mode"]:
        top3 = " ".join(ho["top_promoted_tokens"][:3])
        log(f"{ho['model']:<12} {ho['head']:<12} {ho['role']:<16} {ho['norm_Oh_en']:>8.3f} "
            f"{ho['cos_Oh_en_vs_dmode']:>8.4f} {ho['cos_Oh_cn_vs_dmode']:>8.4f} "
            f"{ho['wu_cos_EOS']:>8.4f} {ho['wu_cos_period']:>8.4f} {top3:>30}")

    log("\n--- Boost Effects ---")
    for be in comparison["boost_effects"]:
        if "type" in be and be["type"] == "rollout":
            log(f"  {be['model']:<12} {be['head']:<12} {be['alpha']:<12} "
                f"eos={be['eos_rate']:.2f} switch={be['lang_switch_rate']:.2f} tokens={be['mean_tokens']:.1f}")
        else:
            log(f"  {be['model']:<12} {be['head']:<12} "
                f"ΔEOS@1.5={be.get('delta_EOS_a1.5', 0):.4f} ΔEOS@2.0={be.get('delta_EOS_a2.0', 0):.4f} "
                f"ΔEOS@3.0={be.get('delta_EOS_a3.0', 0):.4f} argmax_chg@2.0={be.get('argmax_change_a2.0', 0):.2f}")

    log("\n--- Joint Intervention ---")
    log(f"{'Model':<12} {'Head':<12} {'Condition':<25} {'EOS':>6} {'Clean':>6} {'Switch':>7} {'Tokens':>7}")
    for ji in comparison["joint_intervention"]:
        log(f"{ji['model']:<12} {ji['head']:<12} {ji['condition']:<25} "
            f"{ji['eos_rate']:>6.2f} {ji['strict_clean_rate']:>6.2f} "
            f"{ji['lang_switch_rate']:>7.2f} {ji['mean_tokens']:>7.1f}")

    # Save
    output = {"task": "task6_cross_model", "comparison": comparison}
    save_path = save_dir / "task6_cross_model.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n  Saved: {save_path}")

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=attention, 2+3=head_output+mode, 4=boost, 5=joint, 6=cross-model")

    all_results = []

    # Run models sequentially: qwen3 -> GLM4 -> DS7B
    for m in ["qwen3", "glm4", "deepseek7b"]:
        try:
            mr = run_model(m)
            all_results.append(mr)
        except Exception as e:
            log(f"  {m} FAILED entirely: {e}")
            import traceback; traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Task 6: Cross-model comparison
    try:
        task6_cross_model(all_results, RESULT_DIR)
    except Exception as e:
        log(f"Task 6 FAILED: {e}")
        import traceback; traceback.print_exc()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
