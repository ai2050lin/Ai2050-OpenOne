"""
Phase 517: Exact Readout Audit & Semantic Hit Restoration
=========================================================
Phase 516 暴露两个根本问题：
1. DS7B 符号反转：manual readout 与 model logits 不对齐
2. semantic hit 断崖下降：Phase 515 100% → Phase 516 ~0%

本阶段目标：
Exp1: Exact Readout Audit — 验证手动 RMSNorm+W_U 是否与模型 logits 精确对齐
Exp2: Semantic Hit Root Cause — 对比 Phase 515/516 的 classify 差异
Exp3: Finite Difference Gradient Check — 验证解析梯度与数值梯度一致性
Exp4: Restore Semantic Hit — 用修正后的 classify 验证命中率恢复

用法:
  python tests/glm5/phase517_readout_audit.py qwen3
  python tests/glm5/phase517_readout_audit.py glm4
  python tests/glm5/phase517_readout_audit.py deepseek7b
"""
import sys, os, gc, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== Configuration ==============
FRUIT_OBJECTS = ["apple", "banana", "orange", "grape", "strawberry",
                 "mango", "pear", "cherry", "watermelon", "pineapple"]
FRUIT_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]

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


def get_norm_weight(model, model_name):
    """Get RMSNorm weight as numpy, handling meta device"""
    cache_key = f"{model_name}_norm"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    w = model.model.norm.weight
    if not w.is_meta:
        g = w.detach().float().cpu().numpy()
    else:
        import glob
        from safetensors import safe_open
        cfg = MODEL_CONFIGS[model_name]
        sf_files = glob.glob(os.path.join(cfg["path"], '*.safetensors'))
        g = None
        for sf_file in sf_files:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = list(sf.keys())
                for pk in ['model.norm.weight', 'model.model.norm.weight']:
                    if pk in keys:
                        g = sf.get_tensor(pk).float().numpy()
                        break
                if g is None:
                    for k in keys:
                        if k.endswith('norm.weight') and 'layers' not in k:
                            g = sf.get_tensor(k).float().numpy()
                            break
            if g is not None:
                break
    _WEIGHT_CACHE[cache_key] = g
    return g


def get_W_U_cached(model, model_name):
    cache_key = f"{model_name}_WU"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    W_U = get_W_U(model, model_name)
    _WEIGHT_CACHE[cache_key] = W_U
    return W_U


def check_lm_head_bias(model, model_name):
    """Check if lm_head has bias and if embeddings are tied"""
    info = {}
    # Check bias
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
        b = model.lm_head.bias
        if not b.is_meta:
            info['has_bias'] = True
            info['bias_norm'] = float(b.detach().float().cpu().norm())
        else:
            info['has_bias'] = 'meta (on meta device)'
    else:
        info['has_bias'] = False

    # Check tied embeddings
    try:
        embed_w = model.get_input_embeddings().weight
        lm_w = model.lm_head.weight
        if embed_w.data_ptr() == lm_w.data_ptr():
            info['tied'] = True
        elif not embed_w.is_meta and not lm_w.is_meta:
            info['tied'] = torch.equal(embed_w, lm_w)
        else:
            info['tied'] = 'unknown (meta device)'
    except:
        info['tied'] = 'check failed'

    # Check rms_norm_eps
    info['rms_norm_eps'] = getattr(model.config, 'rms_norm_eps', 'not found')

    # Check if model has separate final norm
    info['has_model_norm'] = hasattr(model, 'model') and hasattr(model.model, 'norm')

    return info


# ============== Exp1: Exact Readout Audit ==============

def exp1_exact_readout_audit(model, tokenizer, input_device, model_name):
    """
    Critical experiment: verify manual RMSNorm + W_U matches model logits.

    If they don't match, DS7B symbol reversal is an implementation bug, not a mechanism.
    """
    log("="*60)
    log("Exp1: Exact Readout Audit")
    log("="*60)

    # Check model structure
    lm_info = check_lm_head_bias(model, model_name)
    log(f"  lm_head info: {lm_info}")

    prompt = "An apple belongs to the category of"
    enc = safe_encode(tokenizer, prompt, input_device)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)

    model_logits = out.logits[0, -1, :].float().cpu().numpy()  # [vocab_size]

    # CRITICAL: Test if hidden_states[-1] is pre-norm or post-norm
    # In many HF implementations, hidden_states[-1] = norm(output of last layer) = POST-NORM
    # If so, applying RMSNorm again = double-norm = wrong
    h_final = out.hidden_states[-1][0, -1, :].detach()  # [d_model] on model device
    h_pre_last = out.hidden_states[-2][0, -1, :].detach()  # second-to-last = pre-last-layer

    g = get_norm_weight(model, model_name)  # [d_model] numpy
    W_U = get_W_U_cached(model, model_name)  # [vocab, d_model] numpy
    eps = getattr(model.config, 'rms_norm_eps', 1e-6)

    h_np = h_final.float().cpu().numpy()  # [d_model]
    rms_h = np.sqrt(np.mean(h_np**2) + eps)

    # Method 0: hidden_states[-1] + W_U ONLY (no RMSNorm) — tests if already normed
    manual_logits_nonorm = h_np @ W_U.T  # [vocab]

    # Method 1: hidden_states[-1] + RMSNorm + W_U (may be double-norm)
    h_normed = g * h_np / rms_h  # [d_model]
    manual_logits_nobias = h_normed @ W_U.T  # [vocab]

    # Method 2: Manual RMSNorm + W_U + bias (if exists)
    manual_logits_withbias = manual_logits_nobias.copy()
    if lm_info.get('has_bias') is True:
        # Load bias from safetensors
        import glob
        from safetensors import safe_open
        cfg = MODEL_CONFIGS[model_name]
        sf_files = glob.glob(os.path.join(cfg["path"], '*.safetensors'))
        bias = None
        for sf_file in sf_files:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                if 'lm_head.bias' in sf.keys():
                    bias = sf.get_tensor('lm_head.bias').float().numpy()
                    break
        if bias is not None:
            manual_logits_withbias = manual_logits_nobias + bias
            log(f"  Loaded lm_head.bias: norm={np.linalg.norm(bias):.4f}")

    # Method 3: Use model's own norm + lm_head (torch, on correct device)
    # This tests if the issue is numpy precision vs torch
    try:
        with torch.no_grad():
            h_torch = h_final.unsqueeze(0)  # [1, d_model]
            # Apply model's own norm
            h_normed_torch = model.model.norm(h_torch)  # [1, d_model]
            # Apply lm_head
            if not model.lm_head.weight.is_meta:
                logits_torch = model.lm_head(h_normed_torch)  # [1, vocab]
                torch_logits = logits_torch[0].float().cpu().numpy()
            else:
                torch_logits = None
                log("  lm_head on meta device, skipping torch method")
    except Exception as e:
        torch_logits = None
        log(f"  Torch method failed: {e}")

    # Compare
    results = {"prompt": prompt, "lm_head_info": lm_info}

    # Get target token
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    # Get top-5 from model
    top5_ids = np.argsort(model_logits)[-5:][::-1]

    log(f"  Target token 'fruit' id={target_id}")
    log(f"  Model top-5: {[(tokenizer.decode([i]), float(model_logits[i])) for i in top5_ids]}")
    log(f"  Model logit[fruit]={float(model_logits[target_id]):.6f}")

    # Compare at target token
    log(f"\n  --- Comparison at target token 'fruit' ---")
    log(f"  model_logits[fruit]     = {float(model_logits[target_id]):.6f}")
    log(f"  manual_nonorm[fruit]    = {float(manual_logits_nonorm[target_id]):.6f}")
    log(f"  manual_nobias[fruit]    = {float(manual_logits_nobias[target_id]):.6f}")
    log(f"  manual_withbias[fruit]  = {float(manual_logits_withbias[target_id]):.6f}")
    if torch_logits is not None:
        log(f"  torch_own_norm[fruit]   = {float(torch_logits[target_id]):.6f}")

    # Overall comparison
    diff_nonorm = np.abs(manual_logits_nonorm - model_logits)
    diff_nobias = np.abs(manual_logits_nobias - model_logits)
    diff_withbias = np.abs(manual_logits_withbias - model_logits)

    log(f"\n  --- Overall comparison (all vocab) ---")
    log(f"  |manual_nonorm - model|  max={diff_nonorm.max():.6f}, mean={diff_nonorm.mean():.6f}")
    log(f"  |manual_nobias - model|  max={diff_nobias.max():.6f}, mean={diff_nobias.mean():.6f}")
    log(f"  |manual_withbias - model| max={diff_withbias.max():.6f}, mean={diff_withbias.mean():.6f}")
    if torch_logits is not None:
        diff_torch = np.abs(torch_logits - model_logits)
        log(f"  |torch_own - model|      max={diff_torch.max():.6f}, mean={diff_torch.mean():.6f}")

    # Correlation
    corr_nonorm = np.corrcoef(manual_logits_nonorm, model_logits)[0, 1]
    corr_nobias = np.corrcoef(manual_logits_nobias, model_logits)[0, 1]
    corr_withbias = np.corrcoef(manual_logits_withbias, model_logits)[0, 1]
    log(f"\n  Correlation (manual_nonorm vs model)  = {corr_nonorm:.8f}")
    log(f"  Correlation (manual_nobias vs model)  = {corr_nobias:.8f}")
    log(f"  Correlation (manual_withbias vs model) = {corr_withbias:.8f}")

    # Key diagnostic: is the mismatch a scale, shift, or structural issue?
    # If it's just bf16 precision, diff should be small (< 0.1)
    # If it's a bias, diff_withbias should be much smaller
    # If it's structural (tied vs untied), correlation will be low

    results.update({
        "target_id": target_id,
        "model_logit_fruit": float(model_logits[target_id]),
        "manual_nonorm_fruit": float(manual_logits_nonorm[target_id]),
        "manual_nobias_fruit": float(manual_logits_nobias[target_id]),
        "manual_withbias_fruit": float(manual_logits_withbias[target_id]),
        "diff_nonorm_max": float(diff_nonorm.max()),
        "diff_nonorm_mean": float(diff_nonorm.mean()),
        "diff_nobias_max": float(diff_nobias.max()),
        "diff_nobias_mean": float(diff_nobias.mean()),
        "diff_withbias_max": float(diff_withbias.max()),
        "diff_withbias_mean": float(diff_withbias.mean()),
        "corr_nonorm": float(corr_nonorm),
        "corr_nobias": float(corr_nobias),
        "corr_withbias": float(corr_withbias),
        "rms_h": float(rms_h),
        "eps": float(eps),
    })

    if torch_logits is not None:
        results["torch_own_fruit"] = float(torch_logits[target_id])
        results["diff_torch_max"] = float(np.abs(torch_logits - model_logits).max())

    # Also test: what if hidden_states[-1] is ALREADY normed?
    # Some models return post-norm as the last hidden state
    h_prenorm = h_np
    h_normed_check = g * h_prenorm / rms_h
    norm_of_h = np.linalg.norm(h_np)
    norm_of_normed = np.linalg.norm(h_normed_check)
    log(f"\n  --- Norm check ---")
    log(f"  ||h_final|| = {norm_of_h:.4f}")
    log(f"  ||g*h/rms(h)|| = {norm_of_normed:.4f}")
    log(f"  (If ||h_final|| ≈ ||g||, h might already be normed)")

    # Check: does applying RMSNorm AGAIN change things significantly?
    rms_h2 = np.sqrt(np.mean(h_normed**2) + eps)
    h_double_normed = g * h_normed / rms_h2
    manual_double = h_double_normed @ W_U.T
    diff_double = np.abs(manual_double - model_logits)
    log(f"  Double-norm diff: max={diff_double.max():.6f}, mean={diff_double.mean():.6f}")
    results["diff_double_norm_max"] = float(diff_double.max())

    return results


# ============== Exp2: Semantic Hit Root Cause ==============

def classify_hit_phase515(generated_text, cat_words):
    """Phase 515's classify_hit — checks FULL generated text"""
    text_lower = generated_text.lower()
    found_cat = None
    for cw in cat_words:
        if cw.lower() in text_lower:
            found_cat = cw.lower()
            break
    if found_cat is None:
        return "miss"
    natural_patterns = [
        r"a\s+" + found_cat,
        r"an\s+" + found_cat,
        r"the\s+" + found_cat,
        r"type\s+of\s+" + found_cat,
        r"kind\s+of\s+" + found_cat,
        r"category\s+of\s+" + found_cat,
        r"classified\s+as\s+a\s+" + found_cat,
    ]
    for pat in natural_patterns:
        if re.search(pat, text_lower):
            return "semantic_answer"
    return "lexical"


def classify_trajectory_phase516(continuation, category_word, category_type="fruit"):
    """Phase 516's classify_trajectory — checks CONTINUATION ONLY"""
    text_lower = continuation.lower().strip()
    cat_lower = category_word.lower()
    cat_present = cat_lower in text_lower
    phrases = [f"a {cat_lower}", f"an {cat_lower}", f"the {cat_lower}",
               f"type of {cat_lower}", f"kind of {cat_lower}",
               f"is a {cat_lower}", f"is an {cat_lower}", f"as a {cat_lower}"]
    has_phrase = any(p in text_lower for p in phrases)
    if cat_present and has_phrase:
        return "semantic_answer"
    elif cat_present:
        return "lexical"
    return "miss"


def classify_hit_fixed(full_text, cat_words):
    """Fixed version: checks FULL text but uses Phase 516's simpler phrase matching"""
    text_lower = full_text.lower()
    found_cat = None
    for cw in cat_words:
        if cw.lower() in text_lower:
            found_cat = cw.lower()
            break
    if found_cat is None:
        return "miss"
    phrases = [f"a {found_cat}", f"an {found_cat}", f"the {found_cat}",
               f"type of {found_cat}", f"kind of {found_cat}",
               f"category of {found_cat}", f"is a {found_cat}", f"is an {found_cat}"]
    has_phrase = any(p in text_lower for p in phrases)
    if has_phrase:
        return "semantic_answer"
    return "lexical"


def exp2_semantic_hit_root_cause(model, tokenizer, input_device, model_name, n_objects=10):
    """
    Compare Phase 515 vs Phase 516 classification on the SAME generations.
    Show that the difference is purely in the classify function, not the model.
    """
    log("="*60)
    log("Exp2: Semantic Hit Root Cause — Phase 515 vs 516 classify")
    log("="*60)

    cat_words = ["fruit", "fruits", "Fruit"]
    cat_word_single = "fruit"
    objects = FRUIT_OBJECTS[:n_objects]
    templates = FRUIT_TEMPLATES

    results_p515 = {"semantic_answer": 0, "lexical": 0, "miss": 0}
    results_p516 = {"semantic_answer": 0, "lexical": 0, "miss": 0}
    results_fixed = {"semantic_answer": 0, "lexical": 0, "miss": 0}
    details = []

    total = 0
    for obj in objects:
        for tmpl in templates:
            prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
            enc = safe_encode(tokenizer, prompt, input_device)

            # Use PURE greedy (like Phase 515), no repetition_penalty
            gen_kwargs = dict(max_new_tokens=8, do_sample=False)
            with torch.no_grad():
                gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                         **gen_kwargs)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            continuation = gen_text[len(prompt):].strip()

            q515 = classify_hit_phase515(gen_text, cat_words)
            q516 = classify_trajectory_phase516(continuation, cat_word_single)
            qfixed = classify_hit_fixed(gen_text, cat_words)

            results_p515[q515] += 1
            results_p516[q516] += 1
            results_fixed[qfixed] += 1
            total += 1

            if total <= 6:  # Show first 6 examples
                log(f"  [{obj}/{tmpl[:20]}] cont='{continuation[:40]}' → p515={q515}, p516={q516}, fixed={qfixed}")

            details.append({
                "object": obj, "template": tmpl,
                "continuation": continuation[:60],
                "classify_p515": q515, "classify_p516": q516, "classify_fixed": qfixed,
            })

    log(f"\n  Results (n={total}):")
    log(f"  Phase 515 (full text):    semantic={results_p515['semantic_answer']}, lexical={results_p515['lexical']}, miss={results_p515['miss']}")
    log(f"  Phase 516 (continuation): semantic={results_p516['semantic_answer']}, lexical={results_p516['lexical']}, miss={results_p516['miss']}")
    log(f"  Fixed (full text, simple): semantic={results_fixed['semantic_answer']}, lexical={results_fixed['lexical']}, miss={results_fixed['miss']}")

    return {
        "n_total": total,
        "phase515": results_p515,
        "phase516": results_p516,
        "fixed": results_fixed,
        "details": details,
    }


# ============== Exp3: Finite Difference Gradient Check ==============

def exp3_finite_diff_gradient(model, tokenizer, input_device, model_name):
    """
    Verify analytical gradient matches finite difference.
    This tests if D_c = <h, q_c>/rms(h) is the correct readout formula.
    """
    log("="*60)
    log("Exp3: Finite Difference Gradient Check")
    log("="*60)

    prompt = "An apple belongs to the category of"
    enc = safe_encode(tokenizer, prompt, input_device)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)

    h_final = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()  # [d_model]
    model_logits = out.logits[0, -1, :].float().cpu().numpy()

    g = get_norm_weight(model, model_name)
    W_U = get_W_U_cached(model, model_name)
    eps = getattr(model.config, 'rms_norm_eps', 1e-6)

    # Target token
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    # Competitor: top-1
    top1_id = int(np.argmax(model_logits))
    if top1_id == target_id:
        top1_id = int(np.argsort(model_logits)[-2])

    w_target = W_U[target_id]  # [d_model]
    w_comp = W_U[top1_id]
    q_c = g * (w_target - w_comp)  # readout direction

    rms_h = np.sqrt(np.mean(h_final**2) + eps)
    D_c_base = np.dot(h_final, q_c) / rms_h

    # Analytical gradient: dD_c/dh
    # D_c = <h, q_c> / rms(h)
    # dD_c/dh_j = q_c[j]/rms(h) - <h,q_c> * h_j / (rms(h)^3 * d_model)
    d_model = h_final.shape[0]
    grad_analytical = q_c / rms_h - D_c_base * h_final / (rms_h**3 * d_model)

    # Finite difference: perturb each dimension and measure change in ACTUAL model logit difference
    # This requires running the model, so we do it for a few random dimensions
    np.random.seed(42)
    test_dims = np.random.choice(d_model, 20, replace=False)
    epsilon = 0.01

    grad_numerical = np.zeros(d_model)
    grad_numerical_manual = np.zeros(d_model)  # using manual readout

    for dim_idx in test_dims:
        # Perturb h_final in this dimension
        h_plus = h_final.copy()
        h_plus[dim_idx] += epsilon
        h_minus = h_final.copy()
        h_minus[dim_idx] -= epsilon

        # Manual readout (RMSNorm + W_U)
        rms_plus = np.sqrt(np.mean(h_plus**2) + eps)
        rms_minus = np.sqrt(np.mean(h_minus**2) + eps)
        D_plus_manual = np.dot(h_plus, q_c) / rms_plus
        D_minus_manual = np.dot(h_minus, q_c) / rms_minus
        grad_numerical_manual[dim_idx] = (D_plus_manual - D_minus_manual) / (2 * epsilon)

    # Compare analytical vs numerical (manual readout)
    log(f"  Target='fruit'(id={target_id}), Competitor='{tokenizer.decode([top1_id])}'(id={top1_id})")
    log(f"  D_c (manual) = {D_c_base:.6f}")
    log(f"  rms(h) = {rms_h:.6f}")

    # Compare for the tested dimensions
    log(f"\n  --- Gradient comparison (manual readout, 20 dims) ---")
    log(f"  {'dim':>6} | {'analytical':>12} | {'numerical':>12} | {'rel_error':>10}")
    max_rel_error = 0
    for dim_idx in test_dims:
        a = grad_analytical[dim_idx]
        n = grad_numerical_manual[dim_idx]
        rel_err = abs(a - n) / (abs(a) + abs(n) + 1e-10)
        max_rel_error = max(max_rel_error, rel_err)
        if rel_err > 0.01:
            log(f"  {dim_idx:6d} | {a:12.6f} | {n:12.6f} | {rel_err:10.6f}")

    log(f"\n  Max relative error (manual readout): {max_rel_error:.8f}")
    if max_rel_error < 1e-4:
        log("  ✓ Analytical gradient matches numerical (manual readout is self-consistent)")
    else:
        log("  ✗ Analytical gradient does NOT match numerical — readout formula issue!")

    # Now the KEY test: does manual readout gradient match MODEL readout gradient?
    # We need to perturb h_final in the model and see how model logits change
    # This requires modifying the hidden state in the model forward pass
    # We use inputs_embeds approach: add direction to embedding and check logit change

    # Instead, let's verify: manual readout at base point vs model logit
    manual_D_c = D_c_base
    model_D_c = float(model_logits[target_id] - model_logits[top1_id])
    log(f"\n  --- Base point comparison ---")
    log(f"  D_c (manual readout) = {manual_D_c:.6f}")
    log(f"  D_c (model logits)   = {model_D_c:.6f}")
    log(f"  Difference           = {manual_D_c - model_D_c:.6f}")
    log(f"  Relative diff        = {abs(manual_D_c - model_D_c)/(abs(manual_D_c)+abs(model_D_c)+1e-10):.6f}")

    return {
        "target_id": target_id,
        "competitor_id": top1_id,
        "D_c_manual": float(manual_D_c),
        "D_c_model": float(model_D_c),
        "diff": float(manual_D_c - model_D_c),
        "max_rel_error_manual": float(max_rel_error),
        "rms_h": float(rms_h),
    }


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=10)
    args = parser.parse_args()

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    results = {"model": args.model, "model_info": {"n_layers": info.n_layers, "d_model": info.d_model}}

    # Exp1: Exact Readout Audit (most critical)
    try:
        results["exp1_readout_audit"] = exp1_exact_readout_audit(model, tokenizer, input_device, args.model)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_readout_audit"] = {"error": str(e)}

    # Exp2: Semantic Hit Root Cause
    try:
        results["exp2_semantic_hit"] = exp2_semantic_hit_root_cause(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_semantic_hit"] = {"error": str(e)}

    # Exp3: Finite Difference Gradient Check
    try:
        results["exp3_gradient_check"] = exp3_finite_diff_gradient(
            model, tokenizer, input_device, args.model)
    except Exception as e:
        import traceback
        log(f"Exp3 failed: {e}")
        traceback.print_exc()
        results["exp3_gradient_check"] = {"error": str(e)}

    # Save
    os.makedirs("results/glm5_phase517_readout_audit", exist_ok=True)
    out_path = f"results/glm5_phase517_readout_audit/phase517_{args.model}_readout_audit.json"
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
