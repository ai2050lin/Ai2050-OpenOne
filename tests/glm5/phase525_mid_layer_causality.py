"""
Phase 525: Mid-Layer Semantic Causality Scan
=============================================
Phase 524 found d_traj doesn't encode category at embedding layer (d_category=0%).
Analysis 2 的关键洞察：d_category 在 embedding 层无效 ≠ 在中间层也无效。
语义绑定很可能在中间层完成。

核心问题：d_category 是否在某个中间层获得生成因果性？

Exp1: 中间层 d_category vs d_traj vs random 因果扫描
  - 7个层 × 3个方向 × 10个测试样本
  - Layer 0 = 第一个transformer层（对比 embedding 基线）
  - 如果某层 d_category 突然有效 → 语义激活层

Exp2: 失败模式分类
  - 将失败分为 repetition/wrong_category/non_answer/scaffold_fail
  - 报告 d_traj 对每类失败的效果

技术要点：
  - 中间层干预通过 register_forward_hook 实现
  - hook 在 layer 输出上添加 direction * alpha 到最后一个 token
  - 支持 device_map="auto"（方向张量自动放到层所在设备）

用法:
  python tests/glm5/phase525_mid_layer_causality.py qwen3
  python tests/glm5/phase525_mid_layer_causality.py glm4
  python tests/glm5/phase525_mid_layer_causality.py deepseek7b
  python tests/glm5/phase525_mid_layer_causality.py qwen3 --smoke
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_model_info, release_model, get_W_U, MODEL_CONFIGS, get_layers

# ============== 配置 ==============

# 最小对比 prompt（用于提取 d_category）
TEMPLATE_CONTRAST = "A red {object} is a"
FRUIT_CONTRAST = ["apple", "banana", "orange"]
NON_FRUIT_CONTRAST = ["car", "truck", "bus", "rose", "lily", "tulip"]

# fruit 失败收集（同 Phase 524）
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
N_TEST = 10
N_FRUIT_OBJECTS = 8


# ============== 工具函数 ==============

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


def classify_failure_type(gen_text, prompt, cat_words):
    """将失败样本分类为具体失败模式"""
    cont = gen_text[len(prompt):].strip().lower()
    # 检查重复
    tokens = cont.split()
    if len(tokens) >= 3:
        unique = len(set(tokens))
        if unique / len(tokens) < 0.5:
            return "repetition"
    # 检查数字垃圾
    if len(cont) >= 4 and all(c.isdigit() or c.isspace() or c == '.' for c in cont[:20]):
        return "digit_garbage"
    # 检查类别词是否出现
    found_cat = False
    for cw in cat_words:
        if cw.lower() in cont:
            found_cat = True
            break
    if found_cat:
        return "scaffold_fail"  # 有类别词但不是 S3/S4
    # 检查是否是错误类别
    other_cats = ["vegetable", "animal", "vehicle", "flower", "plant", "thing", "object"]
    for oc in other_cats:
        if oc in cont:
            return "wrong_category"
    return "non_answer"


def generate_greedy(model, tokenizer, input_device, prompt, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.inference_mode():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers):
    """获取所有层的 last-token hidden states。
    返回: [n_layers+1, d_model] numpy array
    hidden_states[0] = embedding output
    hidden_states[L+1] = output of layer L
    """
    enc = safe_encode(tokenizer, prompt, input_device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    all_h = np.stack([hs[0, -1, :].float().cpu().numpy() for hs in out.hidden_states])
    return all_h  # [n_layers+1, d_model]


def generate_with_layer_steering(model, tokenizer, input_device, prompt,
                                  direction, alpha, layer_idx, max_new_tokens=8):
    """在指定层注入方向进行生成。
    layer_idx: 0 到 n_layers-1，指 transformer 层索引
    direction: [d_model] numpy array，会自动归一化到 SCALE
    """
    layers = get_layers(model)
    enc = safe_encode(tokenizer, prompt, input_device)

    # 获取目标层的设备和数据类型
    layer = layers[layer_idx]
    layer_device = next(layer.parameters()).device
    layer_dtype = next(layer.parameters()).dtype

    # 归一化方向
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


def make_random_dir(d_model, seed, scale=SCALE):
    np.random.seed(seed)
    return normalize_dir(np.random.randn(d_model), scale)


def get_test_layers(n_layers, n_test=7):
    """均匀采样测试层，包括第0层和最后一层"""
    if n_layers <= n_test:
        return list(range(n_layers))
    layers = [0]
    for i in range(1, n_test - 1):
        layers.append(int(i * n_layers / (n_test - 1)))
    layers.append(n_layers - 1)
    return sorted(set(layers))


# ============== 数据收集 ==============

def collect_contrast_hidden_states(model, tokenizer, input_device, n_layers):
    """收集最小对比 prompt 的所有层 hidden states"""
    all_h = {}
    objects = FRUIT_CONTRAST + NON_FRUIT_CONTRAST
    for obj in objects:
        prompt = TEMPLATE_CONTRAST.format(object=obj)
        h = get_all_layer_hidden_states(model, tokenizer, input_device, prompt, n_layers)
        all_h[obj] = h  # [n_layers+1, d_model]
        is_fruit = obj in FRUIT_CONTRAST
        log(f"    [contrast] {obj}: {'fruit' if is_fruit else 'non-fruit'}, h shape={h.shape}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_h


def collect_fruit_data(model, tokenizer, input_device, n_layers, n_objects):
    """收集 fruit 成功/失败数据，包括所有层 hidden states 和失败类型"""
    objects = FRUIT_OBJECTS_LARGE[:n_objects]
    success_h_all = []  # list of [n_layers+1, d_model]
    fail_h_all = []
    fail_prompts = []
    fail_types = []
    fail_gens = []

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
                    fail_types.append(classify_failure_type(gen_text, prompt, FRUIT_WORDS))
                    fail_gens.append(gen_text[len(prompt):].strip()[:40])

                if idx % 10 == 0 or idx == total:
                    log(f"    [fruit-collect] {idx}/{total}: suc={len(success_h_all)} fail={len(fail_h_all)}")
                if idx % 6 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return success_h_all, fail_h_all, fail_prompts, fail_types, fail_gens


# ============== 方向提取 ==============

def extract_d_category_at_layer(contrast_h, layer_idx):
    """在第 layer_idx 层提取 d_category
    layer_idx: 0 到 n_layers-1（transformer层索引）
    hidden_states[layer_idx+1] 是该层输出
    """
    fruit_h = [contrast_h[obj][layer_idx + 1] for obj in FRUIT_CONTRAST]
    non_fruit_h = [contrast_h[obj][layer_idx + 1] for obj in NON_FRUIT_CONTRAST]
    d_cat = np.mean(fruit_h, axis=0) - np.mean(non_fruit_h, axis=0)
    return d_cat


def extract_d_traj_at_layer(success_h_all, fail_h_all, layer_idx):
    """在第 layer_idx 层提取 d_traj"""
    suc_h = [h[layer_idx + 1] for h in success_h_all]
    fail_h = [h[layer_idx + 1] for h in fail_h_all]
    if len(suc_h) < 2 or len(fail_h) < 3:
        return None
    return np.mean(suc_h, axis=0) - np.mean(fail_h, axis=0)


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-fruit-objects", type=int, default=N_FRUIT_OBJECTS)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    args = parser.parse_args()

    if args.smoke:
        args.n_fruit_objects = 4
        args.n_test = 5
        log("SMOKE TEST MODE: reduced parameters")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    n_layers = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={n_layers}, d_model={d_model}")

    # 确定测试层
    n_test_layers = 3 if args.smoke else 7
    test_layers = get_test_layers(n_layers, n_test_layers)
    log(f"  Test layers: {test_layers}")

    results = {
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "alpha": ALPHA,
        "scale": SCALE,
        "test_layers": test_layers,
    }

    out_dir = "results/glm5_phase525_mid_layer_causality"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/phase525_{args.model}_mid_layer_causality.json"

    # ===== Step 1: 收集最小对比数据 =====
    log("=" * 60)
    log("Step 1: Collecting minimal contrast data (for d_category)")
    log("=" * 60)
    contrast_h = collect_contrast_hidden_states(model, tokenizer, input_device, n_layers)

    # ===== Step 2: 收集 fruit 成功/失败数据 =====
    log("=" * 60)
    log("Step 2: Collecting fruit success/failure data (for d_traj)")
    log("=" * 60)
    success_h_all, fail_h_all, fail_prompts, fail_types, fail_gens = collect_fruit_data(
        model, tokenizer, input_device, n_layers, args.n_fruit_objects)
    n_suc, n_fail = len(success_h_all), len(fail_h_all)
    log(f"  Success: {n_suc}, Fail: {n_fail}")

    # 失败模式统计
    from collections import Counter
    fail_type_counts = Counter(fail_types)
    log(f"  Failure types: {dict(fail_type_counts)}")
    results["failure_types"] = dict(fail_type_counts)
    results["n_success"] = n_suc
    results["n_failure"] = n_fail

    # ===== Step 3: 在每个测试层提取方向 =====
    log("=" * 60)
    log("Step 3: Extracting directions at each test layer")
    log("=" * 60)

    layer_directions = {}
    for L in test_layers:
        d_cat = extract_d_category_at_layer(contrast_h, L)
        d_traj = extract_d_traj_at_layer(success_h_all, fail_h_all, L)
        random = make_random_dir(d_model, seed=42)

        layer_directions[L] = {
            "d_category": d_cat,
            "d_traj": d_traj if d_traj is not None else np.zeros(d_model),
            "random": random,
        }

        # 计算互相关
        cat_traj_cos = None
        d_traj_norm = 0.0
        if d_traj is not None:
            d_traj_norm = float(np.linalg.norm(d_traj))
            cat_traj_cos = float(np.dot(d_cat, d_traj) / (np.linalg.norm(d_cat) * d_traj_norm + 1e-12))
        d_cat_norm = float(np.linalg.norm(d_cat))
        if cat_traj_cos is not None:
            log(f"  Layer {L:2d}: |d_cat|={d_cat_norm:.2f}, |d_traj|={d_traj_norm:.2f}, cos(d_cat,d_traj)={cat_traj_cos:.4f}")
        else:
            log(f"  Layer {L:2d}: |d_cat|={d_cat_norm:.2f}, d_traj=None")

    # ===== Step 4: 中间层因果扫描 =====
    log("=" * 60)
    log("Step 4: Mid-Layer Causality Scan")
    log("=" * 60)

    # 准备测试样本
    test_prompts = fail_prompts[:args.n_test]
    test_n = len(test_prompts)
    log(f"  Testing {len(test_layers)} layers × 3 directions × {test_n} samples = {len(test_layers)*3*test_n} generations")

    scan_results = {}
    for L in test_layers:
        log(f"\n  --- Layer {L}/{n_layers-1} ---")
        layer_result = {}

        for dir_name in ["d_category", "d_traj", "random"]:
            direction = layer_directions[L][dir_name]
            s34 = 0
            for i, prompt in enumerate(test_prompts):
                gen = generate_with_layer_steering(
                    model, tokenizer, input_device, prompt, direction, ALPHA, L)
                layer_cls = classify_layered(gen, prompt, FRUIT_WORDS)
                if layer_cls in ["S3_cont_phrase", "S4_free"]:
                    s34 += 1
                if (i + 1) % 5 == 0 or i + 1 == test_n:
                    log(f"    [L{L}] {dir_name}: {i+1}/{test_n}, s34={s34}")
                if (i + 1) % 3 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            rate = s34 / test_n if test_n > 0 else 0
            layer_result[dir_name] = {"s34": s34, "n": test_n, "rate": float(rate)}
            log(f"  Layer {L} {dir_name}: {s34}/{test_n} ({100*rate:.0f}%)")

        scan_results[f"layer_{L}"] = layer_result

        # 中间保存
        results["scan_results"] = scan_results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # ===== Step 5: 分析 =====
    log("=" * 60)
    log("Step 5: Analysis")
    log("=" * 60)

    # 打印汇总表
    log(f"\n  {'Layer':>6} | {'d_category':>12} | {'d_traj':>12} | {'random':>12}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for L in test_layers:
        r = scan_results[f"layer_{L}"]
        log(f"  {L:6d} | {100*r['d_category']['rate']:10.0f}%  | {100*r['d_traj']['rate']:10.0f}%  | {100*r['random']['rate']:10.0f}%")

    # 找到 d_category 最有效的层
    cat_rates = [(L, scan_results[f"layer_{L}"]["d_category"]["rate"]) for L in test_layers]
    best_cat_layer, best_cat_rate = max(cat_rates, key=lambda x: x[1])
    log(f"\n  Best d_category layer: {best_cat_layer} ({100*best_cat_rate:.0f}%)")

    # 找到 d_traj 最有效的层
    traj_rates = [(L, scan_results[f"layer_{L}"]["d_traj"]["rate"]) for L in test_layers]
    best_traj_layer, best_traj_rate = max(traj_rates, key=lambda x: x[1])
    log(f"  Best d_traj layer: {best_traj_layer} ({100*best_traj_rate:.0f}%)")

    # 判断
    if best_cat_rate > 0.15:
        verdict = f"d_category GAINS causal power at layer {best_cat_layer} ({100*best_cat_rate:.0f}%) — semantic activation found!"
    elif best_cat_rate > 0:
        verdict = f"d_category has weak effect at layer {best_cat_layer} ({100*best_cat_rate:.0f}%) — possible semantic activation"
    else:
        verdict = "d_category has NO causal power at ANY layer — failure is purely generation-state, not semantic"
    log(f"  Verdict: {verdict}")

    results["best_cat_layer"] = best_cat_layer
    results["best_cat_rate"] = float(best_cat_rate)
    results["best_traj_layer"] = best_traj_layer
    results["best_traj_rate"] = float(best_traj_rate)
    results["verdict"] = verdict

    # 保存方向向量
    npz_path = f"{out_dir}/phase525_{args.model}_directions.npz"
    np_arrays = {}
    for L in test_layers:
        for dir_name in ["d_category", "d_traj", "random"]:
            np_arrays[f"L{L}_{dir_name}"] = layer_directions[L][dir_name]
    np.savez(npz_path, **np_arrays)
    log(f"  Directions saved to {npz_path}")

    # 最终保存
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
