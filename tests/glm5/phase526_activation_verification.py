"""
Phase 526: Semantic Activation Verification
============================================
Phase 525 found d_category gains causal power at mid-layers (qwen3 L12=40%, GLM4 L26=70%).
Analysis 2 的关键硬伤：Hook OOD风险——中间层干预可能只是"扰动打破失败吸引子"而非真实语义因果。

核心验证：
Exp1: Same-Dc对照 — d_category vs same-Dc random vs same-norm random vs pure random
  如果 d_category >> 所有对照 → 语义效应确认
  如果 d_category ≈ 对照 → Phase 525是扰动伪影

Exp2: 精细层扫描 — 1层粒度确认峰值真实
  qwen3: L10-14, GLM4: L24-28, DS7B: L16-20

Exp3: 亚层位置测试 — Attn vs 残差流
  hook self_attn (注入注意力路径，经过MLP) vs hook full layer (直接注入残差流)
  如果两者都有效 → 残差流信号
  如果仅full layer有效 → 信号需绕过MLP

用法:
  python tests/glm5/phase526_activation_verification.py qwen3
  python tests/glm5/phase526_activation_verification.py glm4
  python tests/glm5/phase526_activation_verification.py deepseek7b
  python tests/glm5/phase526_activation_verification.py qwen3 --smoke
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_model_info, release_model, get_W_U, MODEL_CONFIGS, get_layers

# ============== 配置 ==============

PEAK_LAYERS = {"qwen3": 12, "glm4": 26, "deepseek7b": 18}
FINE_SCAN = {
    "qwen3": [10, 11, 12, 13, 14],
    "glm4": [24, 25, 26, 27, 28],
    "deepseek7b": [16, 17, 18, 19, 20],
}

TEMPLATE_CONTRAST = "A red {object} is a"
FRUIT_CONTRAST = ["apple", "banana", "orange"]
NON_FRUIT_CONTRAST = ["car", "truck", "bus", "rose", "lily", "tulip"]

FRUIT_WORDS = ["fruit", "fruits", "Fruit"]
FRUIT_OBJECTS_LARGE = ["apple", "banana", "orange", "grape", "strawberry",
                       "mango", "pear", "cherry"]
PROMPT_CUES = {
    "strong": ["belongs to the category of", "is classified as a type of", "is a kind of"],
    "weak":   ["is a", "is an"],
    "none":   ["is:", ":"],
}

ALPHA = 10.0
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


def generate_with_steering(model, tokenizer, input_device, prompt,
                           direction, alpha, layer_idx,
                           hook_target="layer", max_new_tokens=8):
    """中间层干预生成。
    hook_target: "layer" (全层输出/残差流), "attn" (注意力输出), "mlp" (MLP输出)
    """
    layers = get_layers(model)
    enc = safe_encode(tokenizer, prompt, input_device)
    layer = layers[layer_idx]

    if hook_target == "layer":
        target_module = layer
    elif hook_target == "attn":
        target_module = layer.self_attn
    elif hook_target == "mlp":
        target_module = layer.mlp
    else:
        raise ValueError(f"Unknown hook_target: {hook_target}")

    target_device = next(target_module.parameters()).device
    target_dtype = next(target_module.parameters()).dtype

    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-8:
        d_scaled = direction / d_norm * SCALE
    else:
        d_scaled = direction
    d_tensor = torch.tensor(d_scaled, dtype=target_dtype, device=target_device)

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

    handle = target_module.register_forward_hook(hook_fn)
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


def make_same_dc_random(d_category, d_c, d_model, seed):
    """构造与d_category有相同d_c投影的随机方向"""
    np.random.seed(seed)
    dc_norm_sq = np.dot(d_c, d_c)
    cat_dc_coeff = np.dot(d_category, d_c) / dc_norm_sq
    cat_dc_component = cat_dc_coeff * d_c
    cat_ortho = d_category - cat_dc_component
    cat_ortho_norm = np.linalg.norm(cat_ortho)

    rand = np.random.randn(d_model)
    rand_dc_coeff = np.dot(rand, d_c) / dc_norm_sq
    rand_ortho = rand - rand_dc_coeff * d_c
    rand_ortho_norm = np.linalg.norm(rand_ortho)
    if rand_ortho_norm > 1e-8:
        rand_ortho = rand_ortho / rand_ortho_norm * cat_ortho_norm
    return rand_ortho + cat_dc_component


def make_same_norm_random(d_category, d_model, seed):
    """构造与d_category正交、同范数的随机方向"""
    np.random.seed(seed)
    rand = np.random.randn(d_model)
    cat_norm = np.linalg.norm(d_category)
    cat_norm_sq = np.dot(d_category, d_category)
    proj = np.dot(rand, d_category) / cat_norm_sq
    rand_ortho = rand - proj * d_category
    rand_ortho_norm = np.linalg.norm(rand_ortho)
    if rand_ortho_norm > 1e-8:
        rand_ortho = rand_ortho / rand_ortho_norm * cat_norm
    return rand_ortho


def make_pure_random(d_model, seed, scale=SCALE):
    np.random.seed(seed)
    return normalize_dir(np.random.randn(d_model), scale)


# ============== 数据收集 ==============

def collect_contrast_hidden_states(model, tokenizer, input_device, n_layers):
    all_h = {}
    objects = FRUIT_CONTRAST + NON_FRUIT_CONTRAST
    for obj in objects:
        prompt = TEMPLATE_CONTRAST.format(object=obj)
        h = get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers)
        all_h[obj] = h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_h


def collect_fruit_data(model, tokenizer, input_device, n_layers, n_objects):
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


def extract_d_traj_at_layer(success_h_all, fail_h_all, layer_idx):
    suc_h = [h[layer_idx + 1] for h in success_h_all]
    fail_h = [h[layer_idx + 1] for h in fail_h_all]
    if len(suc_h) < 2 or len(fail_h) < 3:
        return None
    return np.mean(suc_h, axis=0) - np.mean(fail_h, axis=0)


def test_directions(model, tokenizer, input_device, directions_dict, fail_prompts,
                    fruit_words, layer_idx, n_test, hook_target="layer"):
    """测试多个方向在指定层的修复效果"""
    if n_test is not None and len(fail_prompts) > n_test:
        fail_prompts = fail_prompts[:n_test]
    n = len(fail_prompts)
    results = {}

    for name, d in directions_dict.items():
        if d is None:
            results[name] = {"s34": 0, "n": n, "rate": 0.0, "error": "None"}
            continue
        s34 = 0
        for i, prompt in enumerate(fail_prompts):
            gen = generate_with_steering(model, tokenizer, input_device, prompt,
                                         d, ALPHA, layer_idx, hook_target)
            layer = classify_layered(gen, prompt, fruit_words)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
            if (i + 1) % 5 == 0 or i + 1 == n:
                log(f"    [{name}] {i+1}/{n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rate = s34 / n if n > 0 else 0.0
        results[name] = {"s34": s34, "n": n, "rate": float(rate)}
        log(f"  {name}: {s34}/{n} ({100*rate:.0f}%)")
    return results


# ============== Main ==============

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
    log(f"  n_layers={n_layers}, d_model={d_model}")

    peak_L = PEAK_LAYERS[args.model]
    fine_layers = FINE_SCAN[args.model]
    log(f"  Peak layer: {peak_L}, Fine scan: {fine_layers}")

    results = {
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "alpha": ALPHA, "scale": SCALE,
        "peak_layer": peak_L,
        "fine_scan_layers": fine_layers,
    }

    out_dir = "results/glm5_phase526_activation_verification"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/phase526_{args.model}_activation_verification.json"

    # ===== Step 1: 数据收集 =====
    log("=" * 60)
    log("Step 1: Collecting data")
    log("=" * 60)
    contrast_h = collect_contrast_hidden_states(model, tokenizer, input_device, n_layers)
    success_h_all, fail_h_all, fail_prompts = collect_fruit_data(
        model, tokenizer, input_device, n_layers, args.n_fruit_objects)
    n_suc, n_fail = len(success_h_all), len(fail_h_all)
    log(f"  Success: {n_suc}, Fail: {n_fail}")
    results["n_success"] = n_suc
    results["n_failure"] = n_fail

    # ===== Step 2: 提取方向 + 计算d_c =====
    log("=" * 60)
    log("Step 2: Extracting directions and computing d_c")
    log("=" * 60)

    # d_category at peak layer
    d_cat_peak = extract_d_category_at_layer(contrast_h, peak_L)
    log(f"  d_category@L{peak_L}: norm={np.linalg.norm(d_cat_peak):.2f}")

    # d_c (类别读出方向)
    W_U = get_W_U_cached(model, args.model)
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    # 用一个失败样本找competitor
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
    d_c_norm = np.linalg.norm(d_c)
    log(f"  d_c: target={target_id}, competitor={competitor_id}, norm={d_c_norm:.2f}")

    # d_category的Dc投影
    cat_dc_cos = float(np.dot(d_cat_peak, d_c) / (np.linalg.norm(d_cat_peak) * d_c_norm + 1e-12))
    log(f"  cos(d_category, d_c) = {cat_dc_cos:.4f}")

    # 构造对照方向
    same_dc_rand = make_same_dc_random(d_cat_peak, d_c, d_model, seed=42)
    same_norm_rand = make_same_norm_random(d_cat_peak, d_model, seed=42)
    pure_rand = make_pure_random(d_model, seed=42)

    results["cos_dcat_dc"] = cat_dc_cos

    # ===== Step 3: Same-Dc对照测试 =====
    log("=" * 60)
    log(f"Step 3: Same-Dc Control Test at Layer {peak_L}")
    log("=" * 60)

    control_directions = {
        "d_category": d_cat_peak,
        "same_dc_random": same_dc_rand,
        "same_norm_random": same_norm_rand,
        "pure_random": pure_rand,
    }
    control_results = test_directions(
        model, tokenizer, input_device, control_directions,
        fail_prompts, FRUIT_WORDS, peak_L, args.n_test, hook_target="layer")
    results["exp1_same_dc_control"] = {
        "layer": peak_L,
        "test_results": control_results,
        "cos_dcat_dc": cat_dc_cos,
    }

    # 判断
    cat_rate = control_results["d_category"]["rate"]
    best_control = max(control_results["same_dc_random"]["rate"],
                       control_results["same_norm_random"]["rate"],
                       control_results["pure_random"]["rate"])
    if cat_rate > best_control + 0.15:
        control_verdict = "d_category SIGNIFICANTLY > controls — semantic effect CONFIRMED"
    elif cat_rate > best_control:
        control_verdict = "d_category > controls but margin small — weak evidence"
    elif cat_rate > 0:
        control_verdict = "d_category ≈ controls — effect may be non-specific perturbation"
    else:
        control_verdict = "d_category = 0 — no effect at this layer"
    log(f"  Verdict: {control_verdict}")
    results["exp1_same_dc_control"]["verdict"] = control_verdict

    # 中间保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ===== Step 4: 精细层扫描 =====
    log("=" * 60)
    log("Step 4: Fine Layer Scan (d_category)")
    log("=" * 60)

    scan_results = {}
    for L in fine_layers:
        d_cat_L = extract_d_category_at_layer(contrast_h, L)
        s34 = 0
        test_n = min(args.n_test, len(fail_prompts))
        for i, prompt in enumerate(fail_prompts[:test_n]):
            gen = generate_with_steering(model, tokenizer, input_device, prompt,
                                         d_cat_L, ALPHA, L, "layer")
            layer = classify_layered(gen, prompt, FRUIT_WORDS)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
            if (i + 1) % 5 == 0 or i + 1 == test_n:
                log(f"    [L{L}] {i+1}/{test_n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rate = s34 / test_n if test_n > 0 else 0
        scan_results[f"layer_{L}"] = {"s34": s34, "n": test_n, "rate": float(rate)}
        log(f"  Layer {L}: {s34}/{test_n} ({100*rate:.0f}%)")

    results["exp2_fine_scan"] = scan_results

    # 找峰值
    scan_rates = [(L, scan_results[f"layer_{L}"]["rate"]) for L in fine_layers]
    best_scan_L, best_scan_rate = max(scan_rates, key=lambda x: x[1])
    log(f"  Fine scan peak: Layer {best_scan_L} ({100*best_scan_rate:.0f}%)")
    results["exp2_fine_scan_peak"] = {"layer": best_scan_L, "rate": float(best_scan_rate)}

    # 中间保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ===== Step 5: 亚层位置测试 =====
    log("=" * 60)
    log(f"Step 5: Sub-Layer Position Test at Layer {peak_L}")
    log("=" * 60)

    sublayer_n = min(10, len(fail_prompts))
    sublayer_results = {}

    for hook_target in ["layer", "attn", "mlp"]:
        s34 = 0
        for i, prompt in enumerate(fail_prompts[:sublayer_n]):
            try:
                gen = generate_with_steering(model, tokenizer, input_device, prompt,
                                             d_cat_peak, ALPHA, peak_L, hook_target)
                layer = classify_layered(gen, prompt, FRUIT_WORDS)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    s34 += 1
            except Exception as e:
                log(f"    [{hook_target}] Error at sample {i}: {e}")
            if (i + 1) % 5 == 0 or i + 1 == sublayer_n:
                log(f"    [{hook_target}] {i+1}/{sublayer_n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rate = s34 / sublayer_n if sublayer_n > 0 else 0
        sublayer_results[hook_target] = {"s34": s34, "n": sublayer_n, "rate": float(rate)}
        log(f"  hook={hook_target}: {s34}/{sublayer_n} ({100*rate:.0f}%)")

    results["exp3_sublayer"] = sublayer_results

    # 亚层判断
    layer_rate = sublayer_results["layer"]["rate"]
    attn_rate = sublayer_results["attn"]["rate"]
    mlp_rate = sublayer_results["mlp"]["rate"]
    if layer_rate > 0 and attn_rate > 0 and mlp_rate > 0:
        sub_verdict = "All positions effective — residual stream signal (additive)"
    elif layer_rate > 0 and attn_rate == 0 and mlp_rate == 0:
        sub_verdict = "Only full-layer effective — signal must bypass both Attn and MLP processing"
    elif layer_rate > 0 and attn_rate > 0 and mlp_rate == 0:
        sub_verdict = "Layer + Attn effective, MLP not — signal via attention pathway"
    elif layer_rate > 0 and mlp_rate > 0 and attn_rate == 0:
        sub_verdict = "Layer + MLP effective, Attn not — signal via MLP pathway"
    elif layer_rate > 0:
        sub_verdict = "Only full-layer effective — direct residual injection needed"
    else:
        sub_verdict = "No position effective — inconsistent with Phase 525"
    log(f"  Verdict: {sub_verdict}")
    results["exp3_sublayer_verdict"] = sub_verdict

    # ===== 保存方向向量 =====
    npz_path = f"{out_dir}/phase526_{args.model}_directions.npz"
    np.savez(npz_path,
             d_category_peak=d_cat_peak, d_c=d_c,
             same_dc_random=same_dc_rand, same_norm_random=same_norm_rand,
             pure_random=pure_rand)

    # ===== 最终保存 =====
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
