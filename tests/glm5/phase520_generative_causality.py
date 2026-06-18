"""
Phase 520: Generative Causality — Path Value & Steering Test
=============================================================
Phase 519 确认 d_traj 是真实结构信号 (cos=0.35 >> random 0.01)。
但两份分析都指出：只测了 ΔD_c，没测生成质量 (S3/S4)。

Phase 520 核心问题：d_traj 能否实质提升 S3/S4 语义命中？

Exp1: 路径价值估计 — 对每个 prompt 的 top-K 候选首 token，强制生成后测 S3/S4
Exp2: 生成引导测试 — 对失败样本施加 d_traj 干预（embedding层），测 S3/S4 变化
Exp3: d_traj vs 随机方向 — 在生成中对比 d_traj 和随机方向的 S3/S4 效果
Exp4: Logit vs 路径价值分离 — top-logit token 是否 ≠ top-path-value token

用法:
  python tests/glm5/phase520_generative_causality.py qwen3
  python tests/glm5/phase520_generative_causality.py glm4
  python tests/glm5/phase520_generative_causality.py deepseek7b
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
    """Post-norm linear readout"""
    return h_post @ W_U.T


def classify_layered(full_text, prompt, cat_words):
    """S0-S4 layered classification"""
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
    """Pure greedy generation, no repetition penalty"""
    enc = safe_encode(tokenizer, prompt, input_device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                 **gen_kwargs)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text


def generate_with_embedding_steering(model, tokenizer, input_device, prompt, direction, alpha,
                                      max_new_tokens=8):
    """Generate with direction added to embedding at last position"""
    enc = safe_encode(tokenizer, prompt, input_device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(enc["input_ids"]).detach().clone()
    d = torch.tensor(direction, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inputs_embeds[0, -1, :] += d * alpha

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(inputs_embeds=inputs_embeds,
                                 attention_mask=enc["attention_mask"],
                                 **gen_kwargs)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text


def force_first_token_generate(model, tokenizer, input_device, prompt, forced_token_id,
                                max_new_tokens=8):
    """Force a specific first token, then generate the rest"""
    enc = safe_encode(tokenizer, prompt, input_device)
    # Append forced token to input_ids
    forced_tensor = torch.tensor([[forced_token_id]], device=input_device)
    new_input_ids = torch.cat([enc["input_ids"], forced_tensor], dim=-1)
    new_attention_mask = torch.cat([enc["attention_mask"],
                                     torch.ones(1, 1, device=input_device, dtype=torch.long)], dim=-1)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        gen_ids = model.generate(new_input_ids, attention_mask=new_attention_mask,
                                 **gen_kwargs)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text


# ============== Exp1: Path Value Estimation ==============

def exp1_path_value(model, tokenizer, input_device, model_name, n_objects=6):
    """
    For each prompt, get top-K candidate first tokens.
    For each candidate, force it as first token, generate, classify S0-S4.
    This gives V_{S3/S4}(y|h) for each candidate.
    """
    log("="*60)
    log("Exp1: Path Value Estimation")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    K = 5  # top-K candidates
    all_results = []

    for obj in objects:
        for tmpl in PROMPT_CUES["strong"][:2]:  # 2 strong templates
            prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"

            # Get logits to find top-K candidates
            enc = safe_encode(tokenizer, prompt, input_device)
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            output_hidden_states=True)
            logits = out.logits[0, -1, :].float().cpu().numpy()
            top_k_ids = np.argsort(logits)[-K:][::-1]

            # Baseline generation (greedy, no forcing)
            baseline_gen = generate_greedy(model, tokenizer, input_device, prompt)
            baseline_layer = classify_layered(baseline_gen, prompt, cat_words)

            # For each top-K candidate, force and generate
            candidate_results = []
            for cand_id in top_k_ids:
                cand_token = tokenizer.decode([int(cand_id)])
                forced_gen = force_first_token_generate(
                    model, tokenizer, input_device, prompt, int(cand_id))
                forced_layer = classify_layered(forced_gen, prompt, cat_words)
                candidate_results.append({
                    "token": cand_token,
                    "token_id": int(cand_id),
                    "logit": float(logits[int(cand_id)]),
                    "forced_layer": forced_layer,
                    "forced_gen": forced_gen[len(prompt):].strip()[:40],
                })

            # Sort by path value (S3/S4 first, then S2, then S1, then S0)
            layer_order = {"S3_cont_phrase": 4, "S4_free": 4, "S2_scaffold": 3,
                          "S1_lexical": 2, "S0_miss": 1}
            candidate_results.sort(key=lambda x: layer_order.get(x["forced_layer"], 0), reverse=True)

            all_results.append({
                "object": obj,
                "template": tmpl,
                "baseline_layer": baseline_layer,
                "baseline_gen": baseline_gen[len(prompt):].strip()[:40],
                "candidates": candidate_results,
            })

    # Analyze: is top-logit token ≠ top-path-value token?
    logit_path_agree = 0
    logit_path_disagree = 0
    for r in all_results:
        # Top-logit = candidates[0] by logit (original order before sort)
        # Re-sort by logit
        by_logit = sorted(r["candidates"], key=lambda x: x["logit"], reverse=True)
        top_logit = by_logit[0]
        top_path = r["candidates"][0]  # already sorted by path value

        if top_logit["token_id"] == top_path["token_id"]:
            logit_path_agree += 1
        else:
            logit_path_disagree += 1

    total = len(all_results)
    log(f"\n  n_samples={total}")
    log(f"  Logit-PathValue agreement: {logit_path_agree}/{total} ({100*logit_path_agree/total:.0f}%)")
    log(f"  Logit-PathValue disagreement: {logit_path_disagree}/{total} ({100*logit_path_disagree/total:.0f}%)")

    # Show some examples
    for r in all_results[:4]:
        log(f"\n  [{r['object']}] baseline={r['baseline_layer']}, cont='{r['baseline_gen']}'")
        for c in r["candidates"][:3]:
            log(f"    '{c['token']}' (logit={c['logit']:.1f}) → {c['forced_layer']}, cont='{c['forced_gen']}'")

    return {
        "n_samples": total,
        "logit_path_agree": logit_path_agree,
        "logit_path_disagree": logit_path_disagree,
        "agreement_rate": float(logit_path_agree / total) if total > 0 else 0,
        "details": all_results,
    }


# ============== Exp2: Generation Steering Test ==============

def exp2_generation_steering(model, tokenizer, input_device, model_name, n_objects=8):
    """
    KEY EXPERIMENT: Can d_traj improve S3/S4 in actual generation?

    For failure samples:
    1. Generate baseline (no steering)
    2. Generate with d_traj steering (alpha * d_traj at embedding)
    3. Generate with random direction steering (control)
    4. Compare S3/S4 rates
    """
    log("="*60)
    log("Exp2: Generation Steering Test (d_traj vs random vs baseline)")
    log("="*60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect success/failure to build d_traj
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
                    fail_prompts.append((prompt, layer))

    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success (S3+S4): {n_suc}, Fail (S0+S1): {n_fail}")

    if n_suc < 2 or n_fail < 3:
        log("  Insufficient data for steering test")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    # Build d_traj in post-norm space
    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0  # scale to 5.0
    log(f"  d_traj norm: {d_norm:.4f}, scaled to 5.0")

    # Test on failure samples
    alphas = [0.0, 1.0, 3.0, 5.0, 10.0]
    np.random.seed(42)
    random_dir = np.random.randn(len(d_traj))
    random_dir = random_dir / np.linalg.norm(random_dir) * 5.0

    # Use first N failure prompts for testing
    test_fails = fail_prompts[:min(10, len(fail_prompts))]
    n_test = len(test_fails)
    log(f"  Testing on {n_test} failure samples")

    results = {"dtraj": {}, "random": {}}

    for alpha in alphas:
        dtraj_layers = []
        random_layers = []

        for prompt, baseline_layer in test_fails:
            # d_traj steering
            if alpha == 0.0:
                gen_text = generate_greedy(model, tokenizer, input_device, prompt)
            else:
                gen_text = generate_with_embedding_steering(
                    model, tokenizer, input_device, prompt, d_scaled, alpha)
            layer = classify_layered(gen_text, prompt, cat_words)
            dtraj_layers.append(layer)

            # Random steering
            if alpha == 0.0:
                random_layers.append(baseline_layer)
            else:
                gen_text_r = generate_with_embedding_steering(
                    model, tokenizer, input_device, prompt, random_dir, alpha)
                layer_r = classify_layered(gen_text_r, prompt, cat_words)
                random_layers.append(layer_r)

        # Count S3+S4
        dtraj_s34 = sum(1 for l in dtraj_layers if l in ["S3_cont_phrase", "S4_free"])
        dtraj_s2 = sum(1 for l in dtraj_layers if l == "S2_scaffold")
        dtraj_s01 = sum(1 for l in dtraj_layers if l in ["S0_miss", "S1_lexical"])

        random_s34 = sum(1 for l in random_layers if l in ["S3_cont_phrase", "S4_free"])
        random_s2 = sum(1 for l in random_layers if l == "S2_scaffold")
        random_s01 = sum(1 for l in random_layers if l in ["S0_miss", "S1_lexical"])

        results["dtraj"][str(alpha)] = {
            "S0_S1": dtraj_s01, "S2": dtraj_s2, "S3_S4": dtraj_s34,
            "S3_S4_rate": float(dtraj_s34 / n_test) if n_test > 0 else 0,
        }
        results["random"][str(alpha)] = {
            "S0_S1": random_s01, "S2": random_s2, "S3_S4": random_s34,
            "S3_S4_rate": float(random_s34 / n_test) if n_test > 0 else 0,
        }

        log(f"  α={alpha:5.1f}: d_traj S3+S4={dtraj_s34}/{n_test} ({100*dtraj_s34/n_test:.0f}%), "
            f"random S3+S4={random_s34}/{n_test} ({100*random_s34/n_test:.0f}%)")

    # Key comparison: does d_traj improve S3/S4 more than random?
    baseline_rate = results["dtraj"]["0.0"]["S3_S4_rate"]
    best_dtraj_rate = max(results["dtraj"][str(a)]["S3_S4_rate"] for a in alphas)
    best_random_rate = max(results["random"][str(a)]["S3_S4_rate"] for a in alphas)

    log(f"\n  Baseline S3+S4 rate: {baseline_rate:.2%}")
    log(f"  Best d_traj S3+S4 rate: {best_dtraj_rate:.2%}")
    log(f"  Best random S3+S4 rate: {best_random_rate:.2%}")
    log(f"  d_traj improvement: {best_dtraj_rate - baseline_rate:+.2%}")
    log(f"  random improvement: {best_random_rate - baseline_rate:+.2%}")

    if best_dtraj_rate > best_random_rate + 0.05:
        verdict = "d_traj SIGNIFICANTLY better than random"
    elif best_dtraj_rate > best_random_rate:
        verdict = "d_traj slightly better than random"
    elif best_dtraj_rate > baseline_rate:
        verdict = "d_traj improves but not more than random"
    else:
        verdict = "d_traj does NOT improve generation"
    log(f"  Verdict: {verdict}")

    return {
        "n_success": n_suc, "n_failure": n_fail,
        "n_test": n_test,
        "d_traj_norm": float(d_norm),
        "results": results,
        "baseline_rate": float(baseline_rate),
        "best_dtraj_rate": float(best_dtraj_rate),
        "best_random_rate": float(best_random_rate),
        "verdict": verdict,
    }


# ============== Exp3: Logit-Path Value Separation ==============

def exp3_logit_path_separation(model, tokenizer, input_device, model_name, n_objects=6):
    """
    Test if top-logit token ≠ top-path-value token.
    Uses Exp1 data but focuses on the separation metric.
    """
    log("="*60)
    log("Exp3: Logit-Path Value Separation")
    log("="*60)

    # Reuse Exp1 results if available, otherwise compute
    exp1_data = exp1_path_value(model, tokenizer, input_device, model_name, n_objects)

    # Detailed separation analysis
    separations = []
    for r in exp1_data["details"]:
        by_logit = sorted(r["candidates"], key=lambda x: x["logit"], reverse=True)
        by_path = sorted(r["candidates"],
                        key=lambda x: {"S3_cont_phrase": 4, "S4_free": 4, "S2_scaffold": 3,
                                      "S1_lexical": 2, "S0_miss": 1}.get(x["forced_layer"], 0),
                        reverse=True)

        top_logit = by_logit[0]
        top_path = by_path[0]

        separation = {
            "object": r["object"],
            "top_logit_token": top_logit["token"],
            "top_logit_layer": top_logit["forced_layer"],
            "top_path_token": top_path["token"],
            "top_path_layer": top_path["forced_layer"],
            "is_separated": top_logit["token_id"] != top_path["token_id"],
        }
        separations.append(separation)

    n_separated = sum(1 for s in separations if s["is_separated"])
    total = len(separations)
    log(f"  n_samples={total}")
    log(f"  Logit-Path separated: {n_separated}/{total} ({100*n_separated/total:.0f}%)")

    for s in separations[:5]:
        log(f"  [{s['object']}] top_logit='{s['top_logit_token']}'({s['top_logit_layer']}) "
            f"vs top_path='{s['top_path_token']}'({s['top_path_layer']}) "
            f"{'SEPARATED' if s['is_separated'] else 'same'}")

    return {
        "n_samples": total,
        "n_separated": n_separated,
        "separation_rate": float(n_separated / total) if total > 0 else 0,
        "separations": separations,
        "exp1_data": exp1_data,
    }


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=8)
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

    # Exp1: Path Value Estimation (smaller n for speed)
    try:
        n1 = min(args.n_objects, 6)
        results["exp1_path_value"] = exp1_path_value(
            model, tokenizer, input_device, args.model, n1)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_path_value"] = {"error": str(e)}

    # Exp2: Generation Steering (KEY experiment)
    try:
        results["exp2_generation_steering"] = exp2_generation_steering(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_generation_steering"] = {"error": str(e)}

    # Exp3: Logit-Path Separation (reuse Exp1 logic with smaller n)
    try:
        n3 = min(args.n_objects, 6)
        results["exp3_logit_path_separation"] = exp3_logit_path_separation(
            model, tokenizer, input_device, args.model, n3)
    except Exception as e:
        import traceback
        log(f"Exp3 failed: {e}")
        traceback.print_exc()
        results["exp3_logit_path_separation"] = {"error": str(e)}

    os.makedirs("results/glm5_phase520_generative_causality", exist_ok=True)
    out_path = f"results/glm5_phase520_generative_causality/phase520_{args.model}_generative_causality.json"
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
