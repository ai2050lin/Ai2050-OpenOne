"""
Phase 522: Orthogonal Component Ablation & Large-Scale Validation
=================================================================
Phase 521 证明 d_traj 在 same-Dc 对照下仍显著 (GLM4 p<0.05, GCG=3.14)。
两份分析指出关键新实验：d_plan 正交分量消融。

d_traj = d_margin + d_plan
  d_margin = proj_{d_c}(d_traj)  — 沿读出方向，提升 D_c
  d_plan   = d_traj - d_margin    — 正交于 d_c，纯规划分量

核心问题：d_plan 单独（不提升 D_c）能否改善生成？

Exp1: d_plan vs d_margin vs d_traj vs random 正交分量消融
Exp2: 扩大样本 same-Dc 对照 (n=30, 10 seeds)
Exp3: d_value 从路径价值数据构造

用法:
  python tests/glm5/phase522_orthogonal_ablation.py qwen3
  python tests/glm5/phase522_orthogonal_ablation.py glm4
  python tests/glm5/phase522_orthogonal_ablation.py deepseek7b
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


def generate_greedy(model, tokenizer, input_device, prompt, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def generate_with_steering(model, tokenizer, input_device, prompt, direction, alpha, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(enc["input_ids"]).detach().clone()
    d = torch.tensor(direction, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inputs_embeds[0, -1, :] += d * alpha
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(inputs_embeds=inputs_embeds, attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def force_first_token_generate(model, tokenizer, input_device, prompt, forced_token_id, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    forced_tensor = torch.tensor([[forced_token_id]], device=input_device)
    new_input_ids = torch.cat([enc["input_ids"], forced_tensor], dim=-1)
    new_attention_mask = torch.cat([enc["attention_mask"],
                                     torch.ones(1, 1, device=input_device, dtype=torch.long)], dim=-1)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(new_input_ids, attention_mask=new_attention_mask, **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ============== Exp1: Orthogonal Component Ablation ==============

def exp1_orthogonal_ablation(model, tokenizer, input_device, model_name, n_objects=12):
    """
    CRITICAL: Decompose d_traj into d_margin (along d_c) and d_plan (orthogonal to d_c).
    Test each component separately for generation improvement.

    d_traj = d_margin + d_plan
    d_margin = (<d_traj, d_c> / |d_c|^2) * d_c
    d_plan = d_traj - d_margin

    If d_plan alone improves S3/S4, it proves orthogonal path value exists.
    """
    log("="*60)
    log("Exp1: Orthogonal Component Ablation (d_plan vs d_margin vs d_traj)")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    d_model = W_U.shape[1]

    # Collect success/failure
    success_h = []
    fail_h = []
    fail_prompts = []

    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                gen_text = generate_greedy(model, tokenizer, input_device, prompt)
                layer = classify_layered(gen_text, prompt, cat_words)
                enc = safe_encode(tokenizer, prompt, input_device)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)
                    fail_prompts.append(prompt)

    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success (S3+S4): {n_suc}, Fail (S0+S1): {n_fail}")

    if n_suc < 2 or n_fail < 5:
        log("  Insufficient data")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    # Build d_traj
    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)

    # Get d_c (readout direction)
    h_rep = fail_h[0]
    logits_rep = correct_readout(h_rep, W_U)
    top1 = int(np.argmax(logits_rep))
    if top1 == target_id:
        top1 = int(np.argsort(logits_rep)[-2])
    d_c = W_U[target_id] - W_U[top1]
    d_c_norm = np.linalg.norm(d_c)

    # Decompose d_traj
    dtraj_dc_proj = np.dot(d_traj, d_c) / (d_c_norm**2)
    d_margin = dtraj_dc_proj * d_c  # along d_c
    d_plan = d_traj - d_margin  # orthogonal to d_c

    d_margin_norm = np.linalg.norm(d_margin)
    d_plan_norm = np.linalg.norm(d_plan)

    log(f"  d_traj norm: {d_norm:.4f}")
    log(f"  d_margin norm (along d_c): {d_margin_norm:.4f} ({100*d_margin_norm/d_norm:.1f}% of d_traj)")
    log(f"  d_plan norm (orthogonal to d_c): {d_plan_norm:.4f} ({100*d_plan_norm/d_norm:.1f}% of d_traj)")
    log(f"  cos(d_traj, d_c) = {np.dot(d_traj, d_c)/(d_norm*d_c_norm):.4f}")

    # Scale all to same norm (5.0) for fair comparison
    alpha = 10.0
    scale = 5.0

    d_traj_scaled = d_traj / (d_norm + 1e-8) * scale
    d_margin_scaled = d_margin / (d_margin_norm + 1e-8) * scale if d_margin_norm > 1e-8 else d_margin
    d_plan_scaled = d_plan / (d_plan_norm + 1e-8) * scale if d_plan_norm > 1e-8 else d_plan

    # Also create random orthogonal direction (same norm, orthogonal to d_c)
    np.random.seed(42)
    rand_dir = np.random.randn(d_model)
    rand_dc_proj = np.dot(rand_dir, d_c) / (d_c_norm**2)
    rand_ortho = rand_dir - rand_dc_proj * d_c  # orthogonal to d_c
    rand_ortho_norm = np.linalg.norm(rand_ortho)
    rand_ortho_scaled = rand_ortho / (rand_ortho_norm + 1e-8) * scale if rand_ortho_norm > 1e-8 else rand_ortho

    # Test on failure samples
    test_fails = fail_prompts[:min(20, len(fail_prompts))]
    n_test = len(test_fails)
    log(f"  Testing on {n_test} failure samples, alpha={alpha}")

    # Test 4 conditions: d_traj, d_margin, d_plan, random_ortho
    conditions = {
        "d_traj": d_traj_scaled,
        "d_margin": d_margin_scaled,
        "d_plan": d_plan_scaled,
        "random_ortho": rand_ortho_scaled,
    }

    results = {}
    for name, direction in conditions.items():
        s34 = 0
        for prompt in test_fails:
            gen = generate_with_steering(model, tokenizer, input_device, prompt, direction, alpha)
            layer = classify_layered(gen, prompt, cat_words)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
        rate = s34 / n_test if n_test > 0 else 0
        results[name] = {"s34": s34, "n": n_test, "rate": float(rate)}
        log(f"  {name:15s}: S3+S4 = {s34}/{n_test} ({100*rate:.0f}%)")

    # Key comparison
    log(f"\n  --- Key Comparison ---")
    log(f"  d_traj:       {results['d_traj']['rate']:.0%}")
    log(f"  d_plan (ortho): {results['d_plan']['rate']:.0%}  ← should be >0 if orthogonal path value exists")
    log(f"  d_margin:     {results['d_margin']['rate']:.0%}  ← should be ~0 if D_c alone insufficient")
    log(f"  random_ortho: {results['random_ortho']['rate']:.0%}  ← baseline for orthogonal directions")

    if results['d_plan']['rate'] > results['random_ortho']['rate'] + 0.05:
        verdict = "d_plan SIGNIFICANTLY better than random_ortho — orthogonal path value EXISTS"
    elif results['d_plan']['rate'] > results['random_ortho']['rate']:
        verdict = "d_plan slightly better than random_ortho"
    elif results['d_plan']['rate'] > 0:
        verdict = "d_plan has some effect but not more than random"
    else:
        verdict = "d_plan has NO effect — orthogonal path value not found"
    log(f"  Verdict: {verdict}")

    return {
        "n_success": n_suc, "n_failure": n_fail, "n_test": n_test,
        "d_traj_norm": float(d_norm),
        "d_margin_norm": float(d_margin_norm),
        "d_plan_norm": float(d_plan_norm),
        "d_margin_pct": float(d_margin_norm / d_norm * 100),
        "d_plan_pct": float(d_plan_norm / d_norm * 100),
        "cos_dtraj_dc": float(np.dot(d_traj, d_c) / (d_norm * d_c_norm)),
        "results": results,
        "verdict": verdict,
    }


# ============== Exp2: Large-Scale Same-Dc Control ==============

def exp2_large_scale_same_dc(model, tokenizer, input_device, model_name, n_objects=15):
    """
    Expand same-Dc control to n=30 with 10 random seeds.
    """
    log("="*60)
    log("Exp2: Large-Scale Same-Dc Control (n=30, 10 seeds)")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    d_model = W_U.shape[1]

    success_h = []
    fail_h = []
    fail_prompts = []

    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                gen_text = generate_greedy(model, tokenizer, input_device, prompt)
                layer = classify_layered(gen_text, prompt, cat_words)
                enc = safe_encode(tokenizer, prompt, input_device)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)
                    fail_prompts.append(prompt)

    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success: {n_suc}, Fail: {n_fail}")

    if n_suc < 2 or n_fail < 10:
        log("  Insufficient data")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)
    alpha = 10.0
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0

    h_rep = fail_h[0]
    logits_rep = correct_readout(h_rep, W_U)
    top1 = int(np.argmax(logits_rep))
    if top1 == target_id:
        top1 = int(np.argsort(logits_rep)[-2])
    d_c = W_U[target_id] - W_U[top1]
    dtraj_dc_proj = np.dot(d_scaled, d_c)

    test_fails = fail_prompts[:min(30, len(fail_prompts))]
    n_test = len(test_fails)
    log(f"  Testing on {n_test} failure samples, alpha={alpha}")

    # d_traj
    dtraj_s34 = 0
    for prompt in test_fails:
        gen = generate_with_steering(model, tokenizer, input_device, prompt, d_scaled, alpha)
        layer = classify_layered(gen, prompt, cat_words)
        if layer in ["S3_cont_phrase", "S4_free"]:
            dtraj_s34 += 1
    log(f"  d_traj S3+S4: {dtraj_s34}/{n_test} ({100*dtraj_s34/n_test:.0f}%)")

    # 10 random seeds
    n_seeds = 10
    random_s34_list = []
    for seed in range(n_seeds):
        np.random.seed(seed * 17 + 42)
        rand_dir = np.random.randn(d_model)
        rand_dc_proj = np.dot(rand_dir, d_c)
        if abs(rand_dc_proj) < 1e-8:
            rand_dc_proj = 1e-8
        scale_factor = dtraj_dc_proj / rand_dc_proj
        rand_scaled = rand_dir * scale_factor
        rand_norm = np.linalg.norm(rand_scaled)
        if rand_norm > 1e-8:
            rand_scaled = rand_scaled / rand_norm * 5.0
            new_proj = np.dot(rand_scaled, d_c)
            if abs(new_proj) > 1e-8:
                rand_scaled = rand_scaled * (dtraj_dc_proj / new_proj)

        rand_s34 = 0
        for prompt in test_fails:
            gen = generate_with_steering(model, tokenizer, input_device, prompt, rand_scaled, alpha)
            layer = classify_layered(gen, prompt, cat_words)
            if layer in ["S3_cont_phrase", "S4_free"]:
                rand_s34 += 1
        random_s34_list.append(rand_s34)
        if seed < 5 or rand_s34 > 0:
            log(f"  random seed={seed} S3+S4: {rand_s34}/{n_test}")

    rand_mean = np.mean(random_s34_list)
    rand_std = np.std(random_s34_list)
    rand_nonzero = sum(1 for x in random_s34_list if x > 0)

    # Statistical test (simple z-test)
    if rand_std > 0:
        z_score = (dtraj_s34 - rand_mean) / rand_std
    else:
        z_score = float('inf') if dtraj_s34 > rand_mean else 0.0
    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.norm.cdf(z_score) if z_score != float('inf') else 0.0

    if rand_mean > 0:
        gcg = (dtraj_s34 - rand_mean) / (rand_mean + 0.01)
    else:
        gcg = float(dtraj_s34) if dtraj_s34 > 0 else 0.0

    log(f"\n  --- Summary (n_test={n_test}, n_seeds={n_seeds}) ---")
    log(f"  d_traj S3+S4:      {dtraj_s34}/{n_test} ({100*dtraj_s34/n_test:.0f}%)")
    log(f"  random mean S3+S4: {rand_mean:.1f}/{n_test} ({100*rand_mean/n_test:.0f}%) ± {rand_std:.1f}")
    log(f"  random nonzero:    {rand_nonzero}/{n_seeds} seeds")
    log(f"  z-score: {z_score:.2f}, p-value: {p_value:.4f}")
    log(f"  GCG = {gcg:.2f}")

    if p_value < 0.05:
        verdict = f"SIGNIFICANT (p={p_value:.4f} < 0.05)"
    elif p_value < 0.1:
        verdict = f"marginally significant (p={p_value:.4f})"
    else:
        verdict = f"not significant (p={p_value:.4f})"
    log(f"  Verdict: {verdict}")

    return {
        "n_success": n_suc, "n_failure": n_fail, "n_test": n_test, "n_seeds": n_seeds,
        "dtraj_s34": dtraj_s34,
        "dtraj_rate": float(dtraj_s34 / n_test),
        "random_mean": float(rand_mean),
        "random_std": float(rand_std),
        "random_nonzero": rand_nonzero,
        "z_score": float(z_score) if z_score != float('inf') else 999.0,
        "p_value": float(p_value),
        "gcg": float(gcg),
        "verdict": verdict,
    }


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_objects = 6
        log("SMOKE TEST MODE: n_objects=6")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    results = {"model": args.model, "model_info": {"n_layers": info.n_layers, "d_model": info.d_model}}

    # Exp1: Orthogonal Component Ablation (most critical)
    try:
        results["exp1_orthogonal_ablation"] = exp1_orthogonal_ablation(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_orthogonal_ablation"] = {"error": str(e)}

    # Exp2: Large-Scale Same-Dc Control
    try:
        n2 = min(args.n_objects + 3, 15)
        results["exp2_large_scale_same_dc"] = exp2_large_scale_same_dc(
            model, tokenizer, input_device, args.model, n2)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_large_scale_same_dc"] = {"error": str(e)}

    os.makedirs("results/glm5_phase522_orthogonal_ablation", exist_ok=True)
    out_path = f"results/glm5_phase522_orthogonal_ablation/phase522_{args.model}_orthogonal_ablation.json"
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
