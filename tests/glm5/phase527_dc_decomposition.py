"""
Phase 527: Dc Decomposition, Dose-Response & Semantic Selectivity
=================================================================
Analysis 1 的关键修正：same_dc_random ≠ 纯Dc投影成分。
真正的分解：d_category = d_margin(Proj_{d_c}) + d_semantic(正交分量)
测试哪个分量携带因果力。

Analysis 2 的关键假说：剂量响应曲线可区分门控(阈值)vs连续(平滑)。

Exp1: Dc分解 — d_category vs d_margin vs d_semantic vs random
  如果 d_semantic ≈ d_category → 因果力来自正交语义分量，不靠读出边际
  如果 d_margin ≈ d_category → 因果力来自Dc投影（读出方向）
  如果两者都有贡献 → 混合机制

Exp2: 剂量响应曲线 — alpha = [1,2,3,5,8,10,15,20]
  阈值响应 → 支持门控/齿轮假说
  平滑响应 → 支持连续累积假说

Exp3: 语义选择性 — d_category作用于颜色任务时是否改变颜色
  用颜色prompt: "A red apple is colored" → 模型应输出颜色
  施加d_category → 如果颜色不变 = 选择性高
  施加d_color → 颜色应变 = 正对照

用法:
  python tests/glm5/phase527_dc_decomposition.py qwen3
  python tests/glm5/phase527_dc_decomposition.py glm4
  python tests/glm5/phase527_dc_decomposition.py deepseek7b
  python tests/glm5/phase527_dc_decomposition.py qwen3 --smoke
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_model_info, release_model, get_W_U, MODEL_CONFIGS, get_layers

PEAK_LAYERS = {"qwen3": 12, "glm4": 26, "deepseek7b": 18}

TEMPLATE_CONTRAST = "A red {object} is a"
FRUIT_CONTRAST = ["apple", "banana", "orange"]
NON_FRUIT_CONTRAST = ["car", "truck", "bus", "rose", "lily", "tulip"]

# 颜色选择性测试
COLOR_PROMPTS = [
    "A red apple is colored",
    "A blue car is colored",
    "A green leaf is colored",
    "A yellow banana is colored",
    "A purple flower is colored",
    "A black cat is colored",
    "A white cloud is colored",
    "A brown dog is colored",
]
COLOR_WORDS = ["red", "blue", "green", "yellow", "purple", "black", "white", "brown", "orange", "pink"]

FRUIT_WORDS = ["fruit", "fruits", "Fruit"]
FRUIT_OBJECTS_LARGE = ["apple", "banana", "orange", "grape", "strawberry",
                       "mango", "pear", "cherry"]
PROMPT_CUES = {
    "strong": ["belongs to the category of", "is classified as a type of", "is a kind of"],
    "weak":   ["is a", "is an"],
    "none":   ["is:", ":"],
}

SCALE = 5.0
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
    if any(p in cont for p in cont_phrases):
        return "S3_cont_phrase"
    scaffold_phrases = [f"category of {found_cat}", f"type of {found_cat}",
                        f"kind of {found_cat}", f"a {found_cat}", f"an {found_cat}"]
    if any(p in text_lower for p in scaffold_phrases):
        return "S2_scaffold"
    return "S4_free"


def extract_color_from_output(gen_text, prompt):
    """从生成文本中提取颜色词"""
    cont = gen_text[len(prompt):].strip().lower()
    for cw in COLOR_WORDS:
        if cw in cont:
            return cw
    return None


def generate_greedy(model, tokenizer, input_device, prompt, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.inference_mode():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers):
    enc = safe_encode(tokenizer, prompt, input_device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    return np.stack([hs[0, -1, :].float().cpu().numpy() for hs in out.hidden_states])


def generate_with_layer_steering(model, tokenizer, input_device, prompt,
                                  direction, alpha, layer_idx, max_new_tokens=8):
    layers = get_layers(model)
    enc = safe_encode(tokenizer, prompt, input_device)
    layer = layers[layer_idx]
    layer_device = next(layer.parameters()).device
    layer_dtype = next(layer.parameters()).dtype

    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-8:
        d_scaled = direction / d_norm * SCALE
    else:
        d_scaled = direction
    d_tensor = torch.tensor(d_scaled, dtype=layer_dtype, device=layer_device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            new_hidden = hidden.clone()
            new_hidden[:, -1, :] += d_tensor * alpha
            return (new_hidden,) + output[1:]
        else:
            new_output = output.clone()
            new_output[:, -1, :] += d_tensor * alpha
            return new_output

    handle = layer.register_forward_hook(hook_fn)
    try:
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        with torch.inference_mode():
            gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
    finally:
        handle.remove()
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def normalize_dir(d, scale=SCALE):
    n = np.linalg.norm(d)
    if n > 1e-8:
        return d / n * scale
    return d


def collect_contrast_hidden_states(model, tokenizer, input_device, n_layers):
    all_h = {}
    objects = FRUIT_CONTRAST + NON_FRUIT_CONTRAST
    for obj in objects:
        prompt = TEMPLATE_CONTRAST.format(object=obj)
        all_h[obj] = get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_h


def collect_fruit_failures(model, tokenizer, input_device, n_layers, n_objects):
    objects = FRUIT_OBJECTS_LARGE[:n_objects]
    success_h_all = []
    fail_h_all = []
    fail_prompts = []
    total = len(objects) * sum(len(ts) for ts in PROMPT_CUES.values())
    idx = 0
    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                idx += 1
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                gen_text = generate_greedy(model, tokenizer, input_device, prompt)
                layer = classify_layered(gen_text, prompt, FRUIT_WORDS)
                h_all = get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h_all.append(h_all)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h_all.append(h_all)
                    fail_prompts.append(prompt)
                if idx % 10 == 0 or idx == total:
                    log(f"    [fruit-collect] {idx}/{total}: suc={len(success_h_all)} fail={len(fail_h_all)}")
                if idx % 6 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return success_h_all, fail_h_all, fail_prompts


def extract_d_category_at_layer(contrast_h, layer_idx):
    fruit_h = [contrast_h[obj][layer_idx + 1] for obj in FRUIT_CONTRAST]
    non_fruit_h = [contrast_h[obj][layer_idx + 1] for obj in NON_FRUIT_CONTRAST]
    return np.mean(fruit_h, axis=0) - np.mean(non_fruit_h, axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-fruit-objects", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=15)
    args = parser.parse_args()

    if args.smoke:
        args.n_fruit_objects = 4
        args.n_test = 5
        log("SMOKE TEST MODE")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    n_layers = info.n_layers
    d_model = info.d_model
    peak_L = PEAK_LAYERS[args.model]
    log(f"  n_layers={n_layers}, d_model={d_model}, peak_L={peak_L}")

    results = {
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "peak_layer": peak_L,
        "scale": SCALE,
    }

    out_dir = "results/glm5_phase527_dc_decomposition"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/phase527_{args.model}_dc_decomposition.json"

    # ===== Step 1: 数据收集 =====
    log("=" * 60)
    log("Step 1: Collecting data")
    log("=" * 60)
    contrast_h = collect_contrast_hidden_states(model, tokenizer, input_device, n_layers)
    success_h_all, fail_h_all, fail_prompts = collect_fruit_failures(
        model, tokenizer, input_device, n_layers, args.n_fruit_objects)
    n_suc, n_fail = len(success_h_all), len(fail_h_all)
    log(f"  Success: {n_suc}, Fail: {n_fail}")
    results["n_success"] = n_suc
    results["n_failure"] = n_fail

    # ===== Step 2: 提取方向 + Dc分解 =====
    log("=" * 60)
    log("Step 2: Extracting directions and Dc decomposition")
    log("=" * 60)

    d_cat = extract_d_category_at_layer(contrast_h, peak_L)
    W_U = get_W_U_cached(model, args.model)
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    if fail_h_all:
        h_rep = fail_h_all[0][peak_L + 1]
        logits_rep = h_rep @ W_U.T
        top1 = int(np.argmax(logits_rep))
        if top1 == target_id:
            top1 = int(np.argsort(logits_rep)[-2])
        competitor_id = top1
    else:
        competitor_id = target_id + 1
    d_c = W_U[target_id] - W_U[competitor_id]
    d_c_norm_sq = np.dot(d_c, d_c)

    # Dc分解
    cat_dc_coeff = np.dot(d_cat, d_c) / d_c_norm_sq
    d_margin = cat_dc_coeff * d_c  # Dc投影分量
    d_semantic = d_cat - d_margin   # 正交语义分量

    cat_norm = np.linalg.norm(d_cat)
    margin_norm = np.linalg.norm(d_margin)
    semantic_norm = np.linalg.norm(d_semantic)
    cos_cat_dc = float(np.dot(d_cat, d_c) / (cat_norm * np.linalg.norm(d_c) + 1e-12))

    log(f"  d_category: norm={cat_norm:.2f}")
    log(f"  d_margin (Dc projection): norm={margin_norm:.2f} ({100*margin_norm/cat_norm:.1f}% of d_cat)")
    log(f"  d_semantic (orthogonal): norm={semantic_norm:.2f} ({100*semantic_norm/cat_norm:.1f}% of d_cat)")
    log(f"  cos(d_category, d_c) = {cos_cat_dc:.4f}")

    # 随机对照
    np.random.seed(42)
    pure_random = normalize_dir(np.random.randn(d_model))
    same_norm_ortho = normalize_dir(d_semantic)  # 正交于d_c的同范数方向(近似)

    results["dc_decomposition"] = {
        "d_category_norm": float(cat_norm),
        "d_margin_norm": float(margin_norm),
        "d_semantic_norm": float(semantic_norm),
        "margin_pct": float(100 * margin_norm / cat_norm),
        "semantic_pct": float(100 * semantic_norm / cat_norm),
        "cos_dcat_dc": cos_cat_dc,
    }

    # ===== Exp1: Dc分解测试 =====
    log("=" * 60)
    log(f"Exp1: Dc Decomposition Test at Layer {peak_L}")
    log("=" * 60)

    test_prompts = fail_prompts[:args.n_test]
    test_n = len(test_prompts)
    directions_exp1 = {
        "d_category": d_cat,
        "d_margin": d_margin,
        "d_semantic": d_semantic,
        "pure_random": pure_random,
    }

    exp1_results = {}
    for name, d in directions_exp1.items():
        s34 = 0
        for i, prompt in enumerate(test_prompts):
            gen = generate_with_layer_steering(model, tokenizer, input_device, prompt, d, 10.0, peak_L)
            layer = classify_layered(gen, prompt, FRUIT_WORDS)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
            if (i + 1) % 5 == 0 or i + 1 == test_n:
                log(f"    [{name}] {i+1}/{test_n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rate = s34 / test_n if test_n > 0 else 0
        exp1_results[name] = {"s34": s34, "n": test_n, "rate": float(rate)}
        log(f"  {name}: {s34}/{test_n} ({100*rate:.0f}%)")

    # 判断
    cat_rate = exp1_results["d_category"]["rate"]
    margin_rate = exp1_results["d_margin"]["rate"]
    semantic_rate = exp1_results["d_semantic"]["rate"]
    random_rate = exp1_results["pure_random"]["rate"]

    if semantic_rate >= cat_rate - 0.1 and semantic_rate > random_rate + 0.1:
        exp1_verdict = "d_semantic retains effect — causal power is in ORTHOGONAL semantic component, not Dc projection"
    elif margin_rate >= cat_rate - 0.1 and margin_rate > random_rate + 0.1:
        exp1_verdict = "d_margin retains effect — causal power is in Dc PROJECTION (readout direction)"
    elif semantic_rate > random_rate + 0.05 and margin_rate > random_rate + 0.05:
        exp1_verdict = "BOTH components contribute — mixed mechanism"
    else:
        exp1_verdict = "Neither component clearly retains effect — needs investigation"
    log(f"  Verdict: {exp1_verdict}")

    results["exp1_dc_decomposition"] = {
        "test_results": exp1_results,
        "verdict": exp1_verdict,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ===== Exp2: 剂量响应曲线 =====
    log("=" * 60)
    log(f"Exp2: Dose-Response Curve at Layer {peak_L}")
    log("=" * 60)

    alphas = [1, 2, 3, 5, 8, 10, 15, 20] if not args.smoke else [2, 5, 10]
    dose_n = min(8, test_n)
    dose_prompts = test_prompts[:dose_n]

    dose_results = {}
    for alpha in alphas:
        s34 = 0
        for i, prompt in enumerate(dose_prompts):
            gen = generate_with_layer_steering(model, tokenizer, input_device, prompt, d_cat, float(alpha), peak_L)
            layer = classify_layered(gen, prompt, FRUIT_WORDS)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
            if (i + 1) % 4 == 0 or i + 1 == dose_n:
                log(f"    [alpha={alpha}] {i+1}/{dose_n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rate = s34 / dose_n if dose_n > 0 else 0
        dose_results[str(alpha)] = {"s34": s34, "n": dose_n, "rate": float(rate)}
        log(f"  alpha={alpha:2d}: {s34}/{dose_n} ({100*rate:.0f}%)")

    # 检测阈值效应
    rates_list = [dose_results[str(a)]["rate"] for a in alphas]
    if len(rates_list) >= 4:
        # 找最大跳变
        jumps = [rates_list[i+1] - rates_list[i] for i in range(len(rates_list)-1)]
        max_jump_idx = np.argmax(jumps)
        max_jump = jumps[max_jump_idx]
        if max_jump > 0.3 and rates_list[max_jump_idx] < 0.1:
            dose_verdict = f"THRESHOLD effect at alpha={alphas[max_jump_idx]}→{alphas[max_jump_idx+1]} (jump={100*max_jump:.0f}%) — supports GATING hypothesis"
        else:
            dose_verdict = "SMOOTH response — supports CONTINUOUS accumulation hypothesis"
    else:
        dose_verdict = "Insufficient data points"
    log(f"  Verdict: {dose_verdict}")

    results["exp2_dose_response"] = {
        "alphas": alphas,
        "results": dose_results,
        "verdict": dose_verdict,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ===== Exp3: 语义选择性测试 =====
    log("=" * 60)
    log(f"Exp3: Semantic Selectivity Test (d_category on color task)")
    log("=" * 60)

    # 基线：无干预时模型输出什么颜色
    n_color = len(COLOR_PROMPTS) if not args.smoke else 3
    color_test_prompts = COLOR_PROMPTS[:n_color]

    # 提取d_color（红vs蓝的对比）
    color_contrast_h = {}
    for color in ["red", "blue"]:
        for obj in ["apple", "car"]:
            prompt = f"A {color} {obj} is colored"
            color_contrast_h[(color, obj)] = get_all_layer_hidden_states(
                model, tokenizer, input_device, prompt, n_layers)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    red_h = [color_contrast_h[("red", obj)][peak_L + 1] for obj in ["apple", "car"]]
    blue_h = [color_contrast_h[("blue", obj)][peak_L + 1] for obj in ["apple", "car"]]
    d_color = np.mean(red_h, axis=0) - np.mean(blue_h, axis=0)

    selectivity_dirs = {
        "no_intervention": None,
        "d_category": d_cat,
        "d_color": d_color,
        "d_semantic": d_semantic,
        "pure_random": pure_random,
    }

    selectivity_results = {}
    for name, d in selectivity_dirs.items():
        color_changes = 0
        original_colors = []
        output_colors = []
        for prompt in color_test_prompts:
            # 提取原始颜色
            orig_color = None
            for cw in COLOR_WORDS:
                if cw in prompt.lower():
                    orig_color = cw
                    break
            original_colors.append(orig_color)

            if d is None:
                gen = generate_greedy(model, tokenizer, input_device, prompt)
            else:
                gen = generate_with_layer_steering(model, tokenizer, input_device, prompt, d, 10.0, peak_L)
            out_color = extract_color_from_output(gen, prompt)
            output_colors.append(out_color)

            if out_color and orig_color and out_color != orig_color:
                color_changes += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n = len(color_test_prompts)
        change_rate = color_changes / n if n > 0 else 0
        selectivity_results[name] = {
            "color_changes": color_changes,
            "n": n,
            "change_rate": float(change_rate),
            "original_colors": original_colors,
            "output_colors": output_colors,
        }
        log(f"  {name}: color changed {color_changes}/{n} ({100*change_rate:.0f}%)")
        log(f"    orig: {original_colors}")
        log(f"    out:  {output_colors}")

    # 判断选择性
    cat_color_change = selectivity_results["d_category"]["change_rate"]
    color_color_change = selectivity_results["d_color"]["change_rate"]
    random_color_change = selectivity_results["pure_random"]["change_rate"]

    if cat_color_change <= random_color_change + 0.1:
        sel_verdict = "d_category does NOT change color — SELECTIVE for category (good!)"
    elif cat_color_change < color_color_change:
        sel_verdict = f"d_category changes color ({100*cat_color_change:.0f}%) less than d_color ({100*color_color_change:.0f}%) — partially selective"
    else:
        sel_verdict = "d_category changes color as much as d_color — NOT selective"
    log(f"  Verdict: {sel_verdict}")

    results["exp3_selectivity"] = {
        "test_results": selectivity_results,
        "verdict": sel_verdict,
    }

    # 保存方向
    npz_path = f"{out_dir}/phase527_{args.model}_directions.npz"
    np.savez(npz_path, d_category=d_cat, d_margin=d_margin, d_semantic=d_semantic,
             d_c=d_c, d_color=d_color, pure_random=pure_random)

    results["total_time_min"] = round((time.time() - t_start) / 60, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nSaved to {out_path}")

    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"\nTotal: {results['total_time_min']} min")


if __name__ == "__main__":
    main()
