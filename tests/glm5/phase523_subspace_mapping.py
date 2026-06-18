"""
Phase 523: Value Subspace Mapping & Cross-Category Generalization
=================================================================
Phase 522 发现 random_ortho (seed=42) 在 GLM4 上达 35%，超过 d_plan (25%)。
但那只用了 1 个种子，无法判断 random_ortho 有效性是否稳定。

核心问题：
1. random_ortho 的有效性是否跨多个种子稳定？(分布式编码 vs 偶然命中)
2. d_traj 是否类别特异？(fruit 的 d_traj 对 animal/flower 是否有效)
3. 价值子空间的维度估计？(窄锥还是宽扇面)

Exp1: 多种子正交方向测试 (10 seeds, fruit 类别)
  - 生成 10 个随机正交方向 (orthogonal to d_c, same norm)
  - 每个方向测试 10 个失败样本
  - 比较 d_plan vs 10 个 random_ortho 的分布
  - PCA 分析有效方向，估计子空间维度

Exp2: 跨类别泛化测试 (fruit/vehicle/flower)
  - 为每个类别构建 d_traj
  - 3×3 转移矩阵：d_traj(X) on category Y failures
  - 测量 cos(d_traj(X), d_c(Y)) for all pairs

用法:
  python tests/glm5/phase523_subspace_mapping.py qwen3
  python tests/glm5/phase523_subspace_mapping.py glm4
  python tests/glm5/phase523_subspace_mapping.py deepseek7b
  python tests/glm5/phase523_subspace_mapping.py qwen3 --smoke
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== 类别配置 ==============

CATEGORIES = {
    "fruit": {
        "words": ["fruit", "fruits", "Fruit"],
        "objects": ["apple", "banana", "orange", "grape", "strawberry",
                     "mango", "pear", "cherry", "watermelon", "pineapple",
                     "peach", "lemon"],
    },
    "vehicle": {
        "words": ["vehicle", "vehicles", "Vehicle"],
        "objects": ["car", "truck", "bicycle", "motorcycle", "bus",
                     "train", "airplane", "boat", "ship", "scooter",
                     "tractor", "van"],
    },
    "flower": {
        "words": ["flower", "flowers", "Flower"],
        "objects": ["rose", "lily", "tulip", "daisy", "sunflower",
                     "orchid", "jasmine", "lotus", "peony", "carnation",
                     "daffodil", "lavender"],
    },
}

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


def get_h_post(model, tokenizer, input_device, prompt):
    """获取 post-norm hidden state (hidden_states[-1][0, -1, :])"""
    enc = safe_encode(tokenizer, prompt, input_device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()


# ============== 数据收集 ==============

def collect_success_failure(model, tokenizer, input_device, cat_key, n_objects):
    """
    为指定类别收集 success/failure hidden states。
    返回: success_h, fail_h, fail_prompts
    """
    cat_cfg = CATEGORIES[cat_key]
    cat_words = cat_cfg["words"]
    objects = cat_cfg["objects"][:n_objects]

    success_h = []
    fail_h = []
    fail_prompts = []
    total = len(objects) * sum(len(ts) for ts in PROMPT_CUES.values())
    idx = 0

    for obj in objects:
        for cue_type, templates in PROMPT_CUES.items():
            for tmpl in templates:
                idx += 1
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                gen_text = generate_greedy(model, tokenizer, input_device, prompt)
                layer = classify_layered(gen_text, prompt, cat_words)
                h_post = get_h_post(model, tokenizer, input_device, prompt)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)
                    fail_prompts.append(prompt)
                if idx % 10 == 0 or idx == total:
                    log(f"    [{cat_key}] {idx}/{total}: suc={len(success_h)} fail={len(fail_h)}")

    return success_h, fail_h, fail_prompts


def build_directions(success_h, fail_h, W_U, target_id):
    """
    构建 d_traj, d_c, d_plan。
    d_traj = mean(success) - mean(fail)
    d_c = W_U[target] - W_U[competitor]
    d_plan = d_traj - proj_{d_c}(d_traj)
    """
    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    d_norm = np.linalg.norm(d_traj)

    h_rep = fail_h[0]
    logits_rep = correct_readout(h_rep, W_U)
    top1 = int(np.argmax(logits_rep))
    if top1 == target_id:
        top1 = int(np.argsort(logits_rep)[-2])
    d_c = W_U[target_id] - W_U[top1]
    d_c_norm = np.linalg.norm(d_c)

    # 分解
    dtraj_dc_proj = np.dot(d_traj, d_c) / (d_c_norm ** 2)
    d_margin = dtraj_dc_proj * d_c
    d_plan = d_traj - d_margin
    d_plan_norm = np.linalg.norm(d_plan)

    cos_dtraj_dc = float(np.dot(d_traj, d_c) / (d_norm * d_c_norm + 1e-12))

    return {
        "d_traj": d_traj, "d_traj_norm": float(d_norm),
        "d_c": d_c, "d_c_norm": float(d_c_norm),
        "d_margin": d_margin, "d_plan": d_plan,
        "d_plan_norm": float(d_plan_norm),
        "d_margin_norm": float(np.linalg.norm(d_margin)),
        "cos_dtraj_dc": cos_dtraj_dc,
        "target_id": target_id, "competitor_id": top1,
    }


def make_random_ortho(d_c, d_model, seed, scale=5.0):
    """生成正交于 d_c 的随机方向，缩放到指定范数。"""
    np.random.seed(seed)
    rand_dir = np.random.randn(d_model)
    d_c_norm = np.linalg.norm(d_c)
    rand_dc_proj = np.dot(rand_dir, d_c) / (d_c_norm ** 2)
    rand_ortho = rand_dir - rand_dc_proj * d_c
    rand_ortho_norm = np.linalg.norm(rand_ortho)
    if rand_ortho_norm > 1e-8:
        rand_ortho = rand_ortho / rand_ortho_norm * scale
    return rand_ortho


def test_direction_on_failures(model, tokenizer, input_device, direction, alpha,
                                fail_prompts, cat_words, n_test=None):
    """在失败样本上测试方向的 S3+S4 改善率。"""
    if n_test is not None:
        fail_prompts = fail_prompts[:n_test]
    s34 = 0
    for prompt in fail_prompts:
        gen = generate_with_steering(model, tokenizer, input_device, prompt, direction, alpha)
        layer = classify_layered(gen, prompt, cat_words)
        if layer in ["S3_cont_phrase", "S4_free"]:
            s34 += 1
    return s34, len(fail_prompts)


# ============== Exp1: 多种子正交方向测试 ==============

def exp1_multi_seed_ortho(model, tokenizer, input_device, model_name, n_objects=12,
                           n_seeds=10, n_test=10):
    """
    核心问题：random_ortho 有效性是否跨种子稳定？

    Phase 522 只用了 1 个 seed (42)，random_ortho=35% > d_plan=25%。
    如果换 10 个种子，random_ortho 的分布如何？
    """
    log("=" * 60)
    log("Exp1: Multi-Seed Orthogonal Direction Test")
    log("=" * 60)

    W_U = get_W_U_cached(model, model_name)
    cat_words = CATEGORIES["fruit"]["words"]
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]
    d_model = W_U.shape[1]

    # 收集 success/failure
    log("  Collecting success/failure for fruit...")
    success_h, fail_h, fail_prompts = collect_success_failure(
        model, tokenizer, input_device, "fruit", n_objects)
    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success: {n_suc}, Fail: {n_fail}")

    if n_suc < 2 or n_fail < 5:
        log("  Insufficient data")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    # 构建方向
    dirs = build_directions(success_h, fail_h, W_U, target_id)
    log(f"  d_traj norm: {dirs['d_traj_norm']:.4f}")
    log(f"  d_plan norm: {dirs['d_plan_norm']:.4f} ({100*dirs['d_plan_norm']/dirs['d_traj_norm']:.1f}% of d_traj)")
    log(f"  cos(d_traj, d_c) = {dirs['cos_dtraj_dc']:.4f}")

    alpha = 10.0
    scale = 5.0

    # 缩放 d_plan
    d_plan = dirs["d_plan"]
    d_plan_norm = dirs["d_plan_norm"]
    if d_plan_norm > 1e-8:
        d_plan_scaled = d_plan / d_plan_norm * scale
    else:
        d_plan_scaled = d_plan

    # 测试 d_plan
    log(f"  Testing d_plan on {n_test} failures, alpha={alpha}...")
    dplan_s34, dplan_n = test_direction_on_failures(
        model, tokenizer, input_device, d_plan_scaled, alpha,
        fail_prompts, cat_words, n_test)
    log(f"  d_plan S3+S4: {dplan_s34}/{dplan_n} ({100*dplan_s34/dplan_n:.0f}%)")

    # 测试 10 个 random_ortho 种子
    random_results = []
    effective_directions = [d_plan_scaled]  # 收集有效方向用于 PCA

    for seed in range(n_seeds):
        rand_ortho = make_random_ortho(dirs["d_c"], d_model, seed, scale)
        rand_s34, rand_n = test_direction_on_failures(
            model, tokenizer, input_device, rand_ortho, alpha,
            fail_prompts, cat_words, n_test)
        rate = rand_s34 / rand_n if rand_n > 0 else 0
        random_results.append({
            "seed": seed, "s34": rand_s34, "n": rand_n, "rate": float(rate)
        })
        log(f"  random_ortho seed={seed}: S3+S4 = {rand_s34}/{rand_n} ({100*rate:.0f}%)")
        if rand_s34 > 0:
            effective_directions.append(rand_ortho)

    rand_rates = [r["rate"] for r in random_results]
    rand_mean = float(np.mean(rand_rates))
    rand_std = float(np.std(rand_rates))
    rand_max = float(np.max(rand_rates))
    rand_min = float(np.min(rand_rates))
    n_effective = sum(1 for r in rand_rates if r > 0)

    log(f"\n  --- Exp1 Summary ---")
    log(f"  d_plan:          {dplan_s34}/{dplan_n} ({100*dplan_s34/dplan_n:.0f}%)")
    log(f"  random_ortho:    mean={100*rand_mean:.0f}% ± {100*rand_std:.0f}%")
    log(f"                   min={100*rand_min:.0f}%, max={100*rand_max:.0f}%")
    log(f"  effective seeds: {n_effective}/{n_seeds}")

    # 统计检验：d_plan vs random_ortho 分布
    from scipy import stats as sp_stats
    if rand_std > 0:
        z_score = (dplan_s34 / dplan_n - rand_mean) / rand_std
        p_value = float(1 - sp_stats.norm.cdf(z_score))
    else:
        z_score = float('inf') if dplan_s34 / dplan_n > rand_mean else 0.0
        p_value = 0.0 if z_score == float('inf') else 1.0

    log(f"  z-score (d_plan vs random): {z_score:.2f}, p-value: {p_value:.4f}")

    # PCA 分析有效方向
    pca_result = None
    if len(effective_directions) >= 2:
        eff_matrix = np.array(effective_directions)  # [n_eff, d_model]
        # 中心化
        eff_centered = eff_matrix - eff_matrix.mean(axis=0, keepdims=True)
        # SVD
        U_svd, S_svd, Vt_svd = np.linalg.svd(eff_centered, full_matrices=False)
        total_var = float(np.sum(S_svd ** 2))
        cumvar = np.cumsum(S_svd ** 2) / (total_var + 1e-12)
        n_90 = int(np.searchsorted(cumvar, 0.9) + 1)
        n_50 = int(np.searchsorted(cumvar, 0.5) + 1)

        pca_result = {
            "n_effective_dirs": len(effective_directions),
            "singular_values": [float(s) for s in S_svd[:min(10, len(S_svd))]],
            "cumulative_variance": [float(c) for c in cumvar[:min(10, len(cumvar))]],
            "dims_for_50pct": n_50,
            "dims_for_90pct": n_90,
        }
        log(f"  PCA: {len(effective_directions)} effective dirs, "
            f"{n_50} dims for 50% var, {n_90} dims for 90% var")
        log(f"  Top-5 singular values: {[f'{s:.2f}' for s in S_svd[:5]]}")

    # 判断
    if rand_mean > dplan_s34 / dplan_n + 0.05:
        verdict = "random_ortho STABLE and > d_plan — distributed encoding confirmed"
    elif rand_mean > 0.05:
        verdict = "random_ortho has baseline effect but varies by seed"
    elif dplan_s34 / dplan_n > rand_mean + 0.05:
        verdict = "d_plan significantly better than random_ortho — d_plan is special"
    else:
        verdict = "d_plan ≈ random_ortho — no clear specificity"
    log(f"  Verdict: {verdict}")

    return {
        "n_success": n_suc, "n_failure": n_fail, "n_test": dplan_n,
        "n_seeds": n_seeds,
        "cos_dtraj_dc": dirs["cos_dtraj_dc"],
        "d_plan_pct": float(dirs["d_plan_norm"] / dirs["d_traj_norm"] * 100),
        "dplan_s34": dplan_s34, "dplan_rate": float(dplan_s34 / dplan_n),
        "random_mean": rand_mean, "random_std": rand_std,
        "random_max": rand_max, "random_min": rand_min,
        "n_effective": n_effective,
        "random_results": random_results,
        "z_score": float(z_score) if z_score != float('inf') else 999.0,
        "p_value": float(p_value),
        "pca": pca_result,
        "verdict": verdict,
    }


# ============== Exp2: 跨类别泛化测试 ==============

def exp2_cross_category(model, tokenizer, input_device, model_name, n_objects=8, n_test=8):
    """
    核心问题：d_traj 是否类别特异？

    为 fruit/animal/flower 各构建 d_traj，测试 3×3 转移矩阵。
    如果 d_traj(fruit) 对 animal 失败也有效 → 通用规划方向
    如果仅对 fruit 有效 → 类别特异方向
    """
    log("=" * 60)
    log("Exp2: Cross-Category Generalization (fruit/vehicle/flower)")
    log("=" * 60)

    W_U = get_W_U_cached(model, model_name)
    d_model = W_U.shape[1]
    alpha = 10.0
    scale = 5.0

    # 为每个类别收集数据并构建方向
    cat_data = {}
    for cat_key in ["fruit", "vehicle", "flower"]:
        cat_cfg = CATEGORIES[cat_key]
        cat_words = cat_cfg["words"]
        cat_token = cat_words[0]
        cat_ids = tokenizer.encode(cat_token, add_special_tokens=False)
        target_id = cat_ids[0]

        log(f"  Collecting {cat_key} (target_id={target_id})...")
        success_h, fail_h, fail_prompts = collect_success_failure(
            model, tokenizer, input_device, cat_key, n_objects)
        n_suc, n_fail = len(success_h), len(fail_h)
        log(f"    {cat_key}: success={n_suc}, fail={n_fail}")

        if n_suc < 2 or n_fail < 3:
            log(f"    {cat_key} insufficient data, skipping")
            cat_data[cat_key] = None
            continue

        dirs = build_directions(success_h, fail_h, W_U, target_id)
        d_traj = dirs["d_traj"]
        d_traj_norm = dirs["d_traj_norm"]
        if d_traj_norm > 1e-8:
            d_traj_scaled = d_traj / d_traj_norm * scale
        else:
            d_traj_scaled = d_traj

        cat_data[cat_key] = {
            "cat_words": cat_words,
            "target_id": target_id,
            "success_h": success_h,
            "fail_h": fail_h,
            "fail_prompts": fail_prompts,
            "dirs": dirs,
            "d_traj_scaled": d_traj_scaled,
            "n_success": n_suc,
            "n_failure": n_fail,
        }
        log(f"    cos(d_traj_{cat_key}, d_c_{cat_key}) = {dirs['cos_dtraj_dc']:.4f}")

    # 3×3 转移矩阵
    transfer_matrix = {}
    cos_matrix = {}

    available_cats = [k for k in ["fruit", "vehicle", "flower"] if cat_data[k] is not None]

    for src_cat in available_cats:
        for tgt_cat in available_cats:
            src_data = cat_data[src_cat]
            tgt_data = cat_data[tgt_cat]
            d_traj_src = src_data["d_traj_scaled"]
            tgt_fail_prompts = tgt_data["fail_prompts"]
            tgt_cat_words = tgt_data["cat_words"]

            s34, n = test_direction_on_failures(
                model, tokenizer, input_device, d_traj_src, alpha,
                tgt_fail_prompts, tgt_cat_words, n_test)
            rate = s34 / n if n > 0 else 0
            within = (src_cat == tgt_cat)

            key = f"{src_cat}->{tgt_cat}"
            transfer_matrix[key] = {
                "s34": s34, "n": n, "rate": float(rate), "within": within
            }
            log(f"  d_traj({src_cat}) on {tgt_cat} failures: {s34}/{n} ({100*rate:.0f}%)"
                f"{' [WITHIN]' if within else ' [CROSS]'}")

            # cos(d_traj_src, d_c_tgt)
            d_c_tgt = tgt_data["dirs"]["d_c"]
            d_c_tgt_norm = tgt_data["dirs"]["d_c_norm"]
            d_traj_src_orig = src_data["dirs"]["d_traj"]
            cos_val = float(np.dot(d_traj_src_orig, d_c_tgt) /
                           (np.linalg.norm(d_traj_src_orig) * d_c_tgt_norm + 1e-12))
            cos_matrix[key] = cos_val

    # 汇总
    within_rates = [transfer_matrix[f"{c}->{c}"]["rate"]
                    for c in available_cats if f"{c}->{c}" in transfer_matrix]
    cross_rates = [transfer_matrix[f"{s}->{t}"]["rate"]
                   for s in available_cats for t in available_cats
                   if s != t and f"{s}->{t}" in transfer_matrix]

    log(f"\n  --- Exp2 Summary ---")
    log(f"  Within-category mean: {100*np.mean(within_rates):.0f}% (n={len(within_rates)})")
    if cross_rates:
        log(f"  Cross-category mean:  {100*np.mean(cross_rates):.0f}% (n={len(cross_rates)})")
    log(f"  Transfer matrix:")
    for src_cat in available_cats:
        row = []
        for tgt_cat in available_cats:
            key = f"{src_cat}->{tgt_cat}"
            if key in transfer_matrix:
                row.append(f"{tgt_cat}:{100*transfer_matrix[key]['rate']:.0f}%")
        log(f"    {src_cat:8s} -> {', '.join(row)}")

    log(f"  Cosine matrix (d_traj_src vs d_c_tgt):")
    for src_cat in available_cats:
        row = []
        for tgt_cat in available_cats:
            key = f"{src_cat}->{tgt_cat}"
            if key in cos_matrix:
                row.append(f"{tgt_cat}:{cos_matrix[key]:.3f}")
        log(f"    {src_cat:8s} -> {', '.join(row)}")

    # 判断
    if within_rates and cross_rates:
        if np.mean(within_rates) > np.mean(cross_rates) + 0.1:
            verdict = "d_traj is CATEGORY-SPECIFIC (within >> cross)"
        elif np.mean(cross_rates) > 0.05:
            verdict = "d_traj TRANSFERS across categories (general planning direction)"
        else:
            verdict = "d_traj weak in both within and cross — inconclusive"
    else:
        verdict = "insufficient data for cross-category comparison"
    log(f"  Verdict: {verdict}")

    return {
        "available_cats": available_cats,
        "transfer_matrix": transfer_matrix,
        "cos_matrix": cos_matrix,
        "within_mean": float(np.mean(within_rates)) if within_rates else 0,
        "cross_mean": float(np.mean(cross_rates)) if cross_rates else 0,
        "verdict": verdict,
    }


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=12)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--n-objects-cat", type=int, default=8)
    parser.add_argument("--n-test-cat", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Exp1 (for resuming interrupted runs)")
    args = parser.parse_args()

    if args.smoke:
        args.n_objects = 6
        args.n_seeds = 3
        args.n_test = 5
        args.n_objects_cat = 4
        args.n_test_cat = 4
        log("SMOKE TEST MODE: reduced parameters")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    results = {"model": args.model, "model_info": {"n_layers": info.n_layers, "d_model": info.d_model}}

    os.makedirs("results/glm5_phase523_subspace_mapping", exist_ok=True)
    out_path = f"results/glm5_phase523_subspace_mapping/phase523_{args.model}_subspace_mapping.json"

    # Exp1: 多种子正交方向测试
    if not args.skip_exp1:
        try:
            results["exp1_multi_seed_ortho"] = exp1_multi_seed_ortho(
                model, tokenizer, input_device, args.model,
                args.n_objects, args.n_seeds, args.n_test)
        except Exception as e:
            import traceback
            log(f"Exp1 failed: {e}")
            traceback.print_exc()
            results["exp1_multi_seed_ortho"] = {"error": str(e)}

        # 中间保存 (防止中断丢失结果)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        log(f"  [Intermediate save] Exp1 saved to {out_path}")
    else:
        log("Skipping Exp1 (--skip-exp1)")

    # Exp2: 跨类别泛化测试
    try:
        results["exp2_cross_category"] = exp2_cross_category(
            model, tokenizer, input_device, args.model,
            args.n_objects_cat, args.n_test_cat)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_cross_category"] = {"error": str(e)}

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
