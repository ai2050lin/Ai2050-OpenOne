"""
Phase 521: Same-Dc Control & Top-Token Path Value
=================================================
Phase 520 证明 d_traj 能提升 S3/S4 (qwen3: 30%, GLM4: 50%)。
但两份分析指出关键缺失：
1. 样本太小 (n=10)，30%=3/10 vs 10%=1/10
2. 没有 same-Dc 随机对照 — 随机方向也可能提升 D_c
3. 需要证明 d_traj 改善路径选择，不只是提升类别边际

Phase 521 核心实验：
Exp1: Same-Dc 随机对照 — 构造产生相同 ΔD_c 的随机方向，比较 S3/S4
Exp2: 干预后 top token 路径价值变化 — 新 top token 是否有更高路径价值
Exp3: 扩大样本 + 多随机种子 (n_test=20, 5 seeds)

用法:
  python tests/glm5/phase521_same_dc_control.py qwen3
  python tests/glm5/phase521_same_dc_control.py glm4
  python tests/glm5/phase521_same_dc_control.py deepseek7b
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


def estimate_path_value(model, tokenizer, input_device, prompt, token_id, cat_words):
    """Estimate V_c(y|h) = P(S3/S4 | h, y) by forcing first token and generating."""
    gen_text = force_first_token_generate(model, tokenizer, input_device, prompt, token_id)
    layer = classify_layered(gen_text, prompt, cat_words)
    return 1.0 if layer in ["S3_cont_phrase", "S4_free"] else 0.0


# ============== Exp1: Same-Dc Random Control ==============

def exp1_same_dc_control(model, tokenizer, input_device, model_name, n_objects=12):
    """
    CRITICAL: Construct random direction that produces SAME ΔD_c as d_traj.
    Then compare S3/S4. If d_traj wins, it's not just raising D_c.

    For each failure sample:
    1. Compute d_traj's ΔD_c
    2. Find random direction r scaled to produce same ΔD_c
    3. Compare S3/S4 of d_traj vs same-Dc random
    4. Use multiple random seeds
    """
    log("="*60)
    log("Exp1: Same-Dc Random Control (关键对照实验)")
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
    alpha = 10.0  # Use alpha=10 (most effective from Phase 520)
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0

    # Get readout direction d_c for D_c computation
    # Use a representative sample to get competitor
    h_rep = fail_h[0]
    logits_rep = correct_readout(h_rep, W_U)
    top1 = int(np.argmax(logits_rep))
    if top1 == target_id:
        top1 = int(np.argsort(logits_rep)[-2])
    d_c = W_U[target_id] - W_U[top1]

    # d_traj's ΔD_c at embedding level (approximate using post-norm h)
    # Since embedding steering changes everything, we measure ΔD_c at post-norm level
    # For same-Dc control, we need random direction r such that <r, d_c> = <d_traj, d_c>
    # i.e., r has same projection onto d_c as d_traj
    dtraj_dc_proj = np.dot(d_scaled, d_c)
    log(f"  d_traj norm: {d_norm:.4f}, scaled to 5.0")
    log(f"  d_traj · d_c = {dtraj_dc_proj:.4f}")

    # Test on up to 20 failure samples
    test_fails = fail_prompts[:min(20, len(fail_prompts))]
    n_test = len(test_fails)
    log(f"  Testing on {n_test} failure samples, alpha={alpha}")

    # Generate with d_traj steering
    dtraj_results = []
    for prompt in test_fails:
        gen = generate_with_steering(model, tokenizer, input_device, prompt, d_scaled, alpha)
        layer = classify_layered(gen, prompt, cat_words)
        dtraj_results.append(layer)

    dtraj_s34 = sum(1 for l in dtraj_results if l in ["S3_cont_phrase", "S4_free"])
    log(f"  d_traj S3+S4: {dtraj_s34}/{n_test} ({100*dtraj_s34/n_test:.0f}%)")

    # Generate with 5 same-Dc random directions
    n_seeds = 5
    random_s34_list = []
    for seed in range(n_seeds):
        np.random.seed(seed * 17 + 42)
        rand_dir = np.random.randn(d_model)
        # Project out d_c component, then add back same amount
        rand_dc_proj = np.dot(rand_dir, d_c)
        if abs(rand_dc_proj) < 1e-8:
            rand_dc_proj = 1e-8
        # Scale random to have same d_c projection as d_traj
        scale = dtraj_dc_proj / rand_dc_proj
        rand_scaled = rand_dir * scale

        # Also match overall norm
        rand_norm = np.linalg.norm(rand_scaled)
        if rand_norm > 1e-8:
            rand_scaled = rand_scaled / rand_norm * 5.0
            # Re-adjust to match d_c projection exactly
            new_proj = np.dot(rand_scaled, d_c)
            if abs(new_proj) > 1e-8:
                rand_scaled = rand_scaled * (dtraj_dc_proj / new_proj)

        rand_results = []
        for prompt in test_fails:
            gen = generate_with_steering(model, tokenizer, input_device, prompt, rand_scaled, alpha)
            layer = classify_layered(gen, prompt, cat_words)
            rand_results.append(layer)

        rand_s34 = sum(1 for l in rand_results if l in ["S3_cont_phrase", "S4_free"])
        random_s34_list.append(rand_s34)
        log(f"  random seed={seed} S3+S4: {rand_s34}/{n_test} ({100*rand_s34/n_test:.0f}%)")

    rand_mean = np.mean(random_s34_list)
    rand_std = np.std(random_s34_list)
    log(f"\n  --- Summary ---")
    log(f"  d_traj S3+S4:     {dtraj_s34}/{n_test} ({100*dtraj_s34/n_test:.0f}%)")
    log(f"  random mean S3+S4: {rand_mean:.1f}/{n_test} ({100*rand_mean/n_test:.0f}%) ± {rand_std:.1f}")

    # GCG (Generative Causal Gain)
    if rand_mean > 0:
        gcg = (dtraj_s34 - rand_mean) / (rand_mean + 0.01)
    else:
        gcg = float(dtraj_s34) if dtraj_s34 > 0 else 0.0
    log(f"  GCG = {gcg:.2f}")

    if dtraj_s34 > rand_mean + 2 * rand_std:
        verdict = "d_traj SIGNIFICANTLY better than same-Dc random (p<0.05)"
    elif dtraj_s34 > rand_mean:
        verdict = "d_traj better than same-Dc random (not significant)"
    elif dtraj_s34 > 0:
        verdict = "d_traj same as same-Dc random"
    else:
        verdict = "d_traj does NOT improve generation"
    log(f"  Verdict: {verdict}")

    return {
        "n_success": n_suc, "n_failure": n_fail, "n_test": n_test,
        "alpha": alpha,
        "d_traj_norm": float(d_norm),
        "dtraj_dc_proj": float(dtraj_dc_proj),
        "dtraj_s34": dtraj_s34,
        "dtraj_s34_rate": float(dtraj_s34 / n_test),
        "random_s34_list": random_s34_list,
        "random_mean": float(rand_mean),
        "random_std": float(rand_std),
        "gcg": float(gcg),
        "verdict": verdict,
    }


# ============== Exp2: Top-Token Path Value Change ==============

def exp2_top_token_path_value(model, tokenizer, input_device, model_name, n_objects=8):
    """
    After d_traj intervention, does the new top token have higher path value?
    V_c(y_top') vs V_c(y_top)
    """
    log("="*60)
    log("Exp2: Top-Token Path Value Change")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect success/failure for d_traj
    success_h = []
    fail_h = []
    all_prompts = []

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
                logits = correct_readout(h_post, W_U)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)
                all_prompts.append((prompt, layer, h_post, logits))

    if len(success_h) < 2 or len(fail_h) < 3:
        log("  Insufficient data")
        return {"error": "insufficient data", "n_success": len(success_h), "n_failure": len(fail_h)}

    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0
    alpha = 10.0

    # For each failure prompt, compare top token path value before/after intervention
    # Note: we can't easily get post-intervention hidden state from embedding steering
    # (would need to re-run model). So we use a simpler approach:
    # 1. Get baseline top token and its path value
    # 2. After steering generation, get the actual first generated token
    # 3. Compare path values

    test_fails = [(p, l) for p, l, h, lg in all_prompts if l in ["S0_miss", "S1_lexical"]][:10]

    results = []
    v_improved = 0
    v_same = 0
    v_worse = 0

    for prompt, baseline_layer in test_fails:
        # Baseline: get top token from greedy generation
        baseline_gen = generate_greedy(model, tokenizer, input_device, prompt)
        baseline_first_token = baseline_gen[len(prompt):].strip().split()[0] if baseline_gen[len(prompt):].strip() else ""
        baseline_v = 1.0 if baseline_layer in ["S3_cont_phrase", "S4_free"] else 0.0

        # Steered: generate with d_traj
        steered_gen = generate_with_steering(model, tokenizer, input_device, prompt, d_scaled, alpha)
        steered_layer = classify_layered(steered_gen, prompt, cat_words)
        steered_first_token = steered_gen[len(prompt):].strip().split()[0] if steered_gen[len(prompt):].strip() else ""
        steered_v = 1.0 if steered_layer in ["S3_cont_phrase", "S4_free"] else 0.0

        delta_v = steered_v - baseline_v
        if delta_v > 0:
            v_improved += 1
        elif delta_v == 0:
            v_same += 1
        else:
            v_worse += 1

        results.append({
            "prompt": prompt[:40],
            "baseline_token": baseline_first_token,
            "baseline_layer": baseline_layer,
            "baseline_v": baseline_v,
            "steered_token": steered_first_token,
            "steered_layer": steered_layer,
            "steered_v": steered_v,
            "delta_v": delta_v,
        })

    n = len(results)
    log(f"  n_test={n}")
    log(f"  V improved: {v_improved}/{n} ({100*v_improved/n:.0f}%)")
    log(f"  V same:     {v_same}/{n} ({100*v_same/n:.0f}%)")
    log(f"  V worse:    {v_worse}/{n} ({100*v_worse/n:.0f}%)")

    for r in results[:6]:
        log(f"  [{r['prompt']}] {r['baseline_token']}({r['baseline_layer']}) -> "
            f"{r['steered_token']}({r['steered_layer']}) dV={r['delta_v']:+.1f}")

    return {
        "n_test": n,
        "v_improved": v_improved,
        "v_same": v_same,
        "v_worse": v_worse,
        "improve_rate": float(v_improved / n) if n > 0 else 0,
        "results": results,
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

    # Exp1: Same-Dc Control (most critical)
    try:
        results["exp1_same_dc_control"] = exp1_same_dc_control(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_same_dc_control"] = {"error": str(e)}

    # Exp2: Top-Token Path Value
    try:
        n2 = min(args.n_objects, 8)
        results["exp2_top_token_path_value"] = exp2_top_token_path_value(
            model, tokenizer, input_device, args.model, n2)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_top_token_path_value"] = {"error": str(e)}

    os.makedirs("results/glm5_phase521_same_dc_control", exist_ok=True)
    out_path = f"results/glm5_phase521_same_dc_control/phase521_{args.model}_same_dc_control.json"
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
