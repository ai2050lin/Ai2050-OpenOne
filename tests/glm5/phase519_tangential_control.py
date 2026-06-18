"""
Phase 519: Tangential Orthogonality Control & Spherical Intervention
=====================================================================
Phase 518 发现读出方向 d 与 h_post 几乎正交 (cos ≈ 0.001-0.008)。
分析一指出：在几千维空间中，随机向量也天然近似正交。
必须做随机对照实验才能确认这是结构性正交还是高维随机假象。

Phase 519 实验：
Exp1: 随机对照验证切向正交性（最关键）
  - 比较 4 类方向的 cos(d, h_post): 真实读出方向、随机W_U差分、随机残差方向、d_traj
  - 如果真实读出方向的 cos 显著低于随机基线，说明是结构性正交
Exp2: 球面 vs 欧氏干预对比
  - 欧氏: h' = h + alpha * d
  - 球面: h' = R * (h + alpha * d) / ||h + alpha * d||
Exp3: 切向过滤 d_traj
  - raw d_traj vs tangential d_traj vs radial d_traj vs random tangent
Exp4: d_value (hub对比) vs d_traj (成功/失败对比)

用法:
  python tests/glm5/phase519_tangential_control.py qwen3
  python tests/glm5/phase519_tangential_control.py glm4
  python tests/glm5/phase519_tangential_control.py deepseek7b
"""
import sys, os, gc, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

FRUIT_OBJECTS = ["apple", "banana", "orange", "grape", "strawberry",
                 "mango", "pear", "cherry", "watermelon", "pineapple",
                 "peach", "lemon", "lime", "coconut", "kiwi"]

PROMPT_CUES = {
    "strong": ["belongs to the category of", "is classified as a type of", "is a kind of"],
    "weak": ["is a", "is an"],
    "none": ["is:", ":"],
}

# Hub tokens for d_value construction
HIGH_VALUE_HUBS = [" kind", " type", " a", " category"]
LOW_VALUE_HUBS = [" thing", " stuff", " item"]

_WEIGHT_CACHE = {}


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bfloat16 + device_map=auto, sdpa)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa")
    model.eval()
    input_device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"{model_name} loaded: class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, input_device


def safe_encode(tokenizer, text, device, max_length=64):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {"input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device)}


def get_W_U_cached(model, model_name):
    cache_key = f"{model_name}_WU"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    W_U = get_W_U(model, model_name)
    _WEIGHT_CACHE[cache_key] = W_U
    return W_U


def correct_readout(h_post, W_U):
    """Post-norm linear readout: logits = h_post @ W_U.T"""
    return h_post @ W_U.T


def classify_layered(full_text, prompt, cat_words):
    text_lower = full_text.lower()
    cont = full_text[len(prompt):].strip().lower()
    found_cat = None
    for cw in cat_words:
        if cw.lower() in text_lower:
            found_cat = cw.lower()
            break
    if found_cat is None:
        return "S0_miss"
    if found_cat not in cont:
        return "S1_lexical"
    cont_phrases = [f"a {found_cat}", f"an {found_cat}", f"the {found_cat}",
                    f"type of {found_cat}", f"kind of {found_cat}",
                    f"category of {found_cat}", f"is a {found_cat}", f"is an {found_cat}"]
    has_cont_phrase = any(p in cont for p in cont_phrases)
    if has_cont_phrase:
        return "S3_cont_phrase"
    scaffold_phrases = [f"category of {found_cat}", f"type of {found_cat}",
                        f"kind of {found_cat}", f"a {found_cat}", f"an {found_cat}"]
    has_scaffold = any(p in text_lower for p in scaffold_phrases)
    if has_scaffold:
        return "S2_scaffold"
    return "S4_free"


# ============== Exp1: Random Control for Tangential Orthogonality ==============

def exp1_random_control(model, tokenizer, input_device, model_name, n_objects=10):
    """
    CRITICAL: Is cos(d, h_post) ≈ 0 structural or just high-dimensional randomness?

    Compare 4 direction types:
    1. Real readout: d = W_U(target) - W_U(competitor)
    2. Random W_U diff: d = W_U(rand1) - W_U(rand2)
    3. Random residual: d = random Gaussian vector (same norm as real)
    4. d_traj: success-fail mean difference

    For each, compute cos(d, h_post) across many samples.
    If real readout cos is NOT significantly lower than random, the orthogonality is trivial.
    """
    log("="*60)
    log("Exp1: Random Control for Tangential Orthogonality")
    log("="*60)

    info = get_model_info(model, model_name)
    d_model = info.d_model
    W_U = get_W_U_cached(model, model_name)
    vocab_size = W_U.shape[0]

    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect h_post samples
    h_samples = []
    for obj in objects:
        for tmpl in PROMPT_CUES["strong"]:
            prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
            enc = safe_encode(tokenizer, prompt, input_device)
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            output_hidden_states=True)
            h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            h_samples.append(h_post)

    h_samples = np.array(h_samples)  # [n_samples, d_model]
    n_samples = len(h_samples)
    log(f"  n_samples={n_samples}, d_model={d_model}")

    # Type 1: Real readout direction
    # For each sample, competitor = top-1 (if not target)
    real_cosines = []
    for i, h in enumerate(h_samples):
        logits = correct_readout(h, W_U)
        top1 = int(np.argmax(logits))
        if top1 == target_id:
            top1 = int(np.argsort(logits)[-2])
        d_real = W_U[target_id] - W_U[top1]
        cos_real = np.dot(d_real, h) / (np.linalg.norm(d_real) * np.linalg.norm(h) + 1e-8)
        real_cosines.append(abs(cos_real))

    # Type 2: Random W_U difference (100 random pairs)
    np.random.seed(42)
    rand_wu_cosines = []
    for _ in range(100):
        r1, r2 = np.random.randint(0, vocab_size, 2)
        d_rand = W_U[r1] - W_U[r2]
        for h in h_samples[:5]:  # 5 samples per random pair
            cos_rand = np.dot(d_rand, h) / (np.linalg.norm(d_rand) * np.linalg.norm(h) + 1e-8)
            rand_wu_cosines.append(abs(cos_rand))

    # Type 3: Random Gaussian vector (same norm as real readout direction)
    real_norm = np.mean([np.linalg.norm(W_U[target_id] - W_U[int(np.argmax(correct_readout(h, W_U)))])
                         if int(np.argmax(correct_readout(h, W_U))) != target_id
                         else np.linalg.norm(W_U[target_id] - W_U[int(np.argsort(correct_readout(h, W_U))[-2])])
                         for h in h_samples])
    rand_gauss_cosines = []
    for _ in range(100):
        d_gauss = np.random.randn(d_model)
        d_gauss = d_gauss / np.linalg.norm(d_gauss) * real_norm
        for h in h_samples[:5]:
            cos_gauss = np.dot(d_gauss, h) / (np.linalg.norm(d_gauss) * np.linalg.norm(h) + 1e-8)
            rand_gauss_cosines.append(abs(cos_gauss))

    # Type 4: d_traj (computed from success/fail)
    success_h = []
    fail_h = []
    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                layer = classify_layered(gen_text, prompt, cat_words)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)

    dtraj_cosines = []
    if len(success_h) >= 2 and len(fail_h) >= 2:
        d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
        for h in h_samples:
            cos_dt = np.dot(d_traj, h) / (np.linalg.norm(d_traj) * np.linalg.norm(h) + 1e-8)
            dtraj_cosines.append(abs(cos_dt))

    # Also compute: expected |cos| for random vectors in d_model dimensions
    # For random unit vectors in d dimensions, E[|cos|] ≈ sqrt(2/(pi*d))
    expected_random_cos = np.sqrt(2.0 / (np.pi * d_model))

    log(f"\n  --- |cos(direction, h_post)| comparison ---")
    log(f"  d_model = {d_model}")
    log(f"  Theoretical E[|cos|] for random vectors = {expected_random_cos:.6f}")
    log(f"")
    log(f"  Real readout direction:  mean={np.mean(real_cosines):.6f}, median={np.median(real_cosines):.6f}, max={np.max(real_cosines):.6f}")
    log(f"  Random W_U diff:         mean={np.mean(rand_wu_cosines):.6f}, median={np.median(rand_wu_cosines):.6f}")
    log(f"  Random Gaussian:         mean={np.mean(rand_gauss_cosines):.6f}, median={np.median(rand_gauss_cosines):.6f}")
    if dtraj_cosines:
        log(f"  d_traj direction:        mean={np.mean(dtraj_cosines):.6f}, median={np.median(dtraj_cosines):.6f}")

    # Key comparison: is real readout MORE orthogonal than random?
    ratio_real_vs_rand_wu = np.mean(real_cosines) / (np.mean(rand_wu_cosines) + 1e-8)
    ratio_real_vs_gauss = np.mean(real_cosines) / (np.mean(rand_gauss_cosines) + 1e-8)
    log(f"\n  --- Structural orthogonality test ---")
    log(f"  real/random_WU ratio = {ratio_real_vs_rand_wu:.4f}")
    log(f"  real/random_Gauss ratio = {ratio_real_vs_gauss:.4f}")
    log(f"  (ratio < 1.0 means real readout is MORE orthogonal than random)")
    log(f"  (ratio ≈ 1.0 means orthogonality is just high-dimensional randomness)")

    if ratio_real_vs_rand_wu < 0.5:
        verdict = "STRUCTURAL — real readout is significantly more orthogonal than random"
    elif ratio_real_vs_rand_wu < 0.9:
        verdict = "PARTIALLY STRUCTURAL — some evidence but weak"
    else:
        verdict = "RANDOM — orthogonality is just high-dimensional sparsity"
    log(f"  Verdict: {verdict}")

    return {
        "d_model": d_model,
        "expected_random_cos": float(expected_random_cos),
        "real_readout_cos_mean": float(np.mean(real_cosines)),
        "real_readout_cos_median": float(np.median(real_cosines)),
        "random_wu_cos_mean": float(np.mean(rand_wu_cosines)),
        "random_gauss_cos_mean": float(np.mean(rand_gauss_cosines)),
        "dtraj_cos_mean": float(np.mean(dtraj_cosines)) if dtraj_cosines else None,
        "ratio_real_vs_rand_wu": float(ratio_real_vs_rand_wu),
        "ratio_real_vs_gauss": float(ratio_real_vs_gauss),
        "verdict": verdict,
    }


# ============== Exp2: Spherical vs Euclidean Intervention ==============

def exp2_spherical_vs_euclidean(model, tokenizer, input_device, model_name, n_objects=8):
    """
    Compare Euclidean (h + alpha*d) vs Spherical (R*(h+alpha*d)/||h+alpha*d||) intervention.
    Test on failure samples: does spherical preserve generation quality better?
    """
    log("="*60)
    log("Exp2: Spherical vs Euclidean Intervention")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect failure samples (S0/S1)
    fail_samples = []
    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                layer = classify_layered(gen_text, prompt, cat_words)
                if layer in ["S0_miss", "S1_lexical"]:
                    with torch.no_grad():
                        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                    output_hidden_states=True)
                    h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
                    logits = correct_readout(h_post, W_U)
                    top1 = int(np.argmax(logits))
                    if top1 == target_id:
                        top1 = int(np.argsort(logits)[-2])
                    d_readout = W_U[target_id] - W_U[top1]
                    d_norm = np.linalg.norm(d_readout)
                    if d_norm > 1e-8:
                        d_unit = d_readout / d_norm
                        fail_samples.append({
                            "h": h_post, "d": d_unit, "prompt": prompt,
                            "enc": enc, "D_c_base": logits[target_id] - logits[top1],
                        })

    log(f"  n_fail_samples={len(fail_samples)}")
    if len(fail_samples) < 3:
        log("  Insufficient failure samples")
        return {"n_samples": len(fail_samples), "error": "insufficient data"}

    alphas = [1.0, 5.0, 10.0, 20.0]
    results = {"euclidean": {}, "spherical": {}}

    for alpha in alphas:
        euclid_deltas = []
        sphere_deltas = []
        euclid_norms = []
        sphere_norms = []

        for s in fail_samples[:8]:
            h, d = s["h"], s["d"]
            R = np.linalg.norm(h)

            # Euclidean
            h_euclid = h + alpha * d
            logits_euclid = correct_readout(h_euclid, W_U)
            D_c_euclid = logits_euclid[target_id] - logits_euclid[top1] if 'top1' in s else None
            # Recompute competitor for modified logits
            top1_e = int(np.argmax(logits_euclid))
            if top1_e == target_id:
                top1_e = int(np.argsort(logits_euclid)[-2])
            D_c_euclid = logits_euclid[target_id] - logits_euclid[top1_e]
            euclid_deltas.append(float(D_c_euclid - s["D_c_base"]))
            euclid_norms.append(float(np.linalg.norm(h_euclid)))

            # Spherical
            h_raw = h + alpha * d
            h_sphere = R * h_raw / (np.linalg.norm(h_raw) + 1e-8)
            logits_sphere = correct_readout(h_sphere, W_U)
            top1_s = int(np.argmax(logits_sphere))
            if top1_s == target_id:
                top1_s = int(np.argsort(logits_sphere)[-2])
            D_c_sphere = logits_sphere[target_id] - logits_sphere[top1_s]
            sphere_deltas.append(float(D_c_sphere - s["D_c_base"]))
            sphere_norms.append(float(np.linalg.norm(h_sphere)))

        results["euclidean"][str(alpha)] = {
            "mean_delta_Dc": float(np.mean(euclid_deltas)),
            "mean_norm": float(np.mean(euclid_norms)),
        }
        results["spherical"][str(alpha)] = {
            "mean_delta_Dc": float(np.mean(sphere_deltas)),
            "mean_norm": float(np.mean(sphere_norms)),
        }
        log(f"  α={alpha}: Euclid ΔD_c={np.mean(euclid_deltas):+.4f} (norm={np.mean(euclid_norms):.1f}), "
            f"Sphere ΔD_c={np.mean(sphere_deltas):+.4f} (norm={np.mean(sphere_norms):.1f})")

    # Key: does spherical preserve norm? Does it have similar D_c effect?
    log(f"\n  Base h norm: {np.linalg.norm(fail_samples[0]['h']):.4f}")
    log(f"  Spherical preserves norm (should ≈ base)")
    log(f"  Euclidean inflates norm (radial component added)")

    return {"n_samples": len(fail_samples), "results": results}


# ============== Exp3: Tangential Filtered d_traj ==============

def exp3_tangential_dtraj(model, tokenizer, input_device, model_name, n_objects=10):
    """
    Compare raw d_traj vs tangential d_traj vs radial d_traj vs random tangent.
    """
    log("="*60)
    log("Exp3: Tangential Filtered d_traj")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect success/fail
    success_h = []
    fail_h = []
    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                layer = classify_layered(gen_text, prompt, cat_words)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)

    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success (S3+S4): {n_suc}, Fail (S0+S1): {n_fail}")

    if n_suc < 2 or n_fail < 2:
        log("  Insufficient data")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0

    # Test on failure samples
    test_h = fail_h[:5]
    alphas = [1.0, 5.0]
    directions = {
        "raw": d_scaled,
        "tangential": None,  # computed per-sample
        "radial": None,  # computed per-sample
        "random_tangent": None,  # computed per-sample
    }

    results = {}
    np.random.seed(42)

    for alpha in alphas:
        results[str(alpha)] = {}
        for dir_name in directions:
            deltas = []
            for h in test_h:
                logits_base = correct_readout(h, W_U)
                top1 = int(np.argmax(logits_base))
                if top1 == target_id:
                    top1 = int(np.argsort(logits_base)[-2])
                D_c_base = logits_base[target_id] - logits_base[top1]

                if dir_name == "raw":
                    d = d_scaled
                elif dir_name == "tangential":
                    # Project d_scaled onto tangent space of h
                    proj = np.dot(d_scaled, h) / (np.linalg.norm(h)**2 + 1e-8) * h
                    d = d_scaled - proj
                elif dir_name == "radial":
                    proj = np.dot(d_scaled, h) / (np.linalg.norm(h)**2 + 1e-8) * h
                    d = proj
                elif dir_name == "random_tangent":
                    rand_d = np.random.randn(len(h))
                    proj = np.dot(rand_d, h) / (np.linalg.norm(h)**2 + 1e-8) * h
                    d = rand_d - proj
                    d = d / (np.linalg.norm(d) + 1e-8) * 5.0  # same scale

                h_mod = h + alpha * d
                logits_mod = correct_readout(h_mod, W_U)
                top1_m = int(np.argmax(logits_mod))
                if top1_m == target_id:
                    top1_m = int(np.argsort(logits_mod)[-2])
                D_c_mod = logits_mod[target_id] - logits_mod[top1_m]
                deltas.append(float(D_c_mod - D_c_base))

            mean_delta = np.mean(deltas)
            results[str(alpha)][dir_name] = float(mean_delta)
            log(f"  α={alpha} {dir_name:15s}: ΔD_c={mean_delta:+.4f}")

    return {
        "n_success": n_suc, "n_failure": n_fail,
        "d_traj_norm": float(d_norm),
        "results": results,
    }


# ============== Exp4: d_value (hub-based) vs d_traj ==============

def exp4_dvalue_vs_dtraj(model, tokenizer, input_device, model_name, n_objects=10):
    """
    Compare d_traj (success-fail) vs d_value (high-value hub vs low-value deadend).
    d_value may be more stable and less contaminated by template/object differences.
    """
    log("="*60)
    log("Exp4: d_value (hub-based) vs d_traj (success-fail)")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # For d_value: compare prompts that start with high-value hub vs low-value hub
    # Use the FIRST generated token to classify hub type
    high_hub_h = []
    low_hub_h = []
    success_h = []
    fail_h = []

    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                continuation = gen_text[len(prompt):].strip().lower()

                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()

                # Classify hub based on first generated token
                first_word = continuation.split()[0] if continuation else ""
                is_high_hub = any(hub.strip() in first_word for hub in HIGH_VALUE_HUBS)
                is_low_hub = any(hub.strip() in first_word for hub in LOW_VALUE_HUBS)

                if is_high_hub:
                    high_hub_h.append(h_post)
                elif is_low_hub:
                    low_hub_h.append(h_post)

                # Also classify success/fail
                layer = classify_layered(gen_text, prompt, cat_words)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)

    log(f"  High-value hub: {len(high_hub_h)}, Low-value hub: {len(low_hub_h)}")
    log(f"  Success (S3+S4): {len(success_h)}, Fail (S0+S1): {len(fail_h)}")

    results = {}

    # Compute d_value if possible, otherwise use high_hub vs fail as proxy
    if len(high_hub_h) >= 2 and len(low_hub_h) >= 2:
        d_value = np.mean(high_hub_h, axis=0) - np.mean(low_hub_h, axis=0)
        d_value_source = "high_hub vs low_hub"
    elif len(high_hub_h) >= 2 and len(fail_h) >= 2:
        d_value = np.mean(high_hub_h, axis=0) - np.mean(fail_h, axis=0)
        d_value_source = "high_hub vs fail (low_hub insufficient)"
        log(f"  Using high_hub vs fail as d_value proxy (low_hub={len(low_hub_h)})")
    else:
        log("  Insufficient hub data for d_value")
        results["d_value"] = {"error": "insufficient hub data",
                              "n_high": len(high_hub_h), "n_low": len(low_hub_h)}
        d_value = None

    if d_value is not None:
        d_value_norm = np.linalg.norm(d_value)
        d_value_scaled = d_value / (d_value_norm + 1e-8) * 5.0
        log(f"  d_value norm: {d_value_norm:.4f} (source: {d_value_source})")

        test_h = fail_h[:5] if len(fail_h) >= 5 else fail_h
        if len(test_h) < 2:
            test_h = success_h[:5]  # fallback
        alphas = [1.0, 5.0]

        results["d_value"] = {"source": d_value_source, "norm": float(d_value_norm)}
        for alpha in alphas:
            deltas = []
            for h in test_h:
                logits_base = correct_readout(h, W_U)
                top1 = int(np.argmax(logits_base))
                if top1 == target_id:
                    top1 = int(np.argsort(logits_base)[-2])
                D_c_base = logits_base[target_id] - logits_base[top1]
                h_mod = h + alpha * d_value_scaled
                logits_mod = correct_readout(h_mod, W_U)
                top1_m = int(np.argmax(logits_mod))
                if top1_m == target_id:
                    top1_m = int(np.argsort(logits_mod)[-2])
                D_c_mod = logits_mod[target_id] - logits_mod[top1_m]
                deltas.append(float(D_c_mod - D_c_base))
            mean_delta = np.mean(deltas)
            results["d_value"][str(alpha)] = float(mean_delta)
            log(f"  d_value α={alpha}: ΔD_c={mean_delta:+.4f}")

    # Compute d_traj for comparison
    if len(success_h) >= 2 and len(fail_h) >= 2:
        d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
        d_traj_norm = np.linalg.norm(d_traj)
        d_traj_scaled = d_traj / (d_traj_norm + 1e-8) * 5.0
        log(f"  d_traj norm: {d_traj_norm:.4f}")

        test_h = fail_h[:5] if len(fail_h) >= 5 else fail_h
        alphas = [1.0, 5.0]
        results["d_traj"] = {}
        for alpha in alphas:
            deltas = []
            for h in test_h:
                logits_base = correct_readout(h, W_U)
                top1 = int(np.argmax(logits_base))
                if top1 == target_id:
                    top1 = int(np.argsort(logits_base)[-2])
                D_c_base = logits_base[target_id] - logits_base[top1]
                h_mod = h + alpha * d_traj_scaled
                logits_mod = correct_readout(h_mod, W_U)
                top1_m = int(np.argmax(logits_mod))
                if top1_m == target_id:
                    top1_m = int(np.argsort(logits_mod)[-2])
                D_c_mod = logits_mod[target_id] - logits_mod[top1_m]
                deltas.append(float(D_c_mod - D_c_base))
            mean_delta = np.mean(deltas)
            results["d_traj"][str(alpha)] = float(mean_delta)
            log(f"  d_traj  α={alpha}: ΔD_c={mean_delta:+.4f}")

        # Compare cos(d_value, d_traj) — are they aligned?
        if d_value is not None:
            cos_dv_dt = np.dot(d_value, d_traj) / (np.linalg.norm(d_value) * np.linalg.norm(d_traj) + 1e-8)
            log(f"  cos(d_value, d_traj) = {cos_dv_dt:.4f}")
            results["cos_dvalue_dtraj"] = float(cos_dv_dt)

    return results


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_objects = 4
        log("SMOKE TEST MODE: n_objects=4")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    results = {"model": args.model, "model_info": {"n_layers": info.n_layers, "d_model": info.d_model}}

    # Exp1: Random Control (most critical)
    try:
        results["exp1_random_control"] = exp1_random_control(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_random_control"] = {"error": str(e)}

    # Exp2: Spherical vs Euclidean
    try:
        n2 = min(args.n_objects, 8)
        results["exp2_spherical_vs_euclidean"] = exp2_spherical_vs_euclidean(
            model, tokenizer, input_device, args.model, n2)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_spherical_vs_euclidean"] = {"error": str(e)}

    # Exp3: Tangential d_traj
    try:
        results["exp3_tangential_dtraj"] = exp3_tangential_dtraj(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp3 failed: {e}")
        traceback.print_exc()
        results["exp3_tangential_dtraj"] = {"error": str(e)}

    # Exp4: d_value vs d_traj
    try:
        results["exp4_dvalue_vs_dtraj"] = exp4_dvalue_vs_dtraj(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp4 failed: {e}")
        traceback.print_exc()
        results["exp4_dvalue_vs_dtraj"] = {"error": str(e)}

    os.makedirs("results/glm5_phase519_tangential_control", exist_ok=True)
    out_path = f"results/glm5_phase519_tangential_control/phase519_{args.model}_tangential_control.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nSaved to {out_path}")

    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
