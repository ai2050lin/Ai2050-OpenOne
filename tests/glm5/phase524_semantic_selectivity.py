"""
Phase 524: Semantic Selectivity — What Does d_traj Encode?
==========================================================
Analysis 2 的核心批评：当前 d_traj 是统计方向，不知道编码了什么语义变量。
Phase 524 用最小对比法构造特定语义变量的方向，测试：
1. 语义变量方向是否可分离（正交）？
2. d_traj 与哪个语义变量方向最相关？
3. 每个方向是否选择性控制其目标变量？

Exp1: 语义方向提取（最小对比）
  - d_color:   颜色变化，对象/类别固定 → "A red apple is a" vs "A blue apple is a"
  - d_object:  对象变化（同类内），颜色固定 → "A red apple is a" vs "A red banana is a"
  - d_category: 类别变化，颜色固定 → mean(h("A red apple/banana/orange is a"))
                                       - mean(h("A red car/truck/bus is a"))
  - d_traj:    成功 vs 失败 (同 Phase 523, 从无颜色 prompt 收集)

Exp2: 互相关矩阵
  - cos(d_traj, d_color), cos(d_traj, d_object), cos(d_traj, d_category)
  - cos(d_color, d_object), cos(d_color, d_category), cos(d_object, d_category)
  → 揭示 d_traj 编码了什么语义变量

Exp3: 选择性测试（核心实验）
  - 在 fruit 失败样本上测试每个方向 + random_ortho 对照
  - d_category (+alpha): 应修复失败（如果 d_traj 编码类别信息）
  - d_color (+/-alpha):  不应修复（颜色与类别任务无关）
  - d_object (+/-alpha): 不应修复（同类内对象变化与类别任务无关）
  - random_ortho:        随机对照基线
  - Selectivity = S34(d_category) / (S34(d_color) + S34(d_object) + ε)

Exp4: 跨模板不变性
  - 从 2 个模板提取 d_category，检查 cos(d_category_T1, d_category_T2)
  - 高余弦 → 类别是稳定的语义变量

用法:
  python tests/glm5/phase524_semantic_selectivity.py qwen3
  python tests/glm5/phase524_semantic_selectivity.py glm4
  python tests/glm5/phase524_semantic_selectivity.py deepseek7b
  python tests/glm5/phase524_semantic_selectivity.py qwen3 --smoke
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== 配置 ==============

COLORS = ["red", "blue", "green"]

OBJECTS_BY_CATEGORY = {
    "fruit":   ["apple", "banana", "orange"],
    "vehicle": ["car", "truck", "bus"],
    "flower":  ["rose", "lily", "tulip"],
}
ALL_OBJECTS = [obj for objs in OBJECTS_BY_CATEGORY.values() for obj in objs]

# 用于收集失败样本的更大 fruit 对象集
FRUIT_OBJECTS_LARGE = ["apple", "banana", "orange", "grape", "strawberry",
                       "mango", "pear", "cherry"]

# 两套模板用于跨模板不变性测试
TEMPLATE_T1 = "A {color} {object} is a"
TEMPLATE_T2 = "A {color} {object} belongs to the category of"

# 失败收集用模板（同 Phase 523）
PROMPT_CUES = {
    "strong": ["belongs to the category of", "is classified as a type of", "is a kind of"],
    "weak":   ["is a", "is an"],
    "none":   ["is:", ":"],
}

# fruit 类别词（用于分类）
FRUIT_WORDS = ["fruit", "fruits", "Fruit"]

_WEIGHT_CACHE = {}
ALPHA = 10.0
SCALE = 5.0  # 方向归一化范数


# ============== 工具函数 ==============

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def load_model_bf16(model_name):
    """BF16 + device_map=auto + sdpa (含FlashAttention)"""
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
    """同 Phase 523 的分层分类"""
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
    with torch.inference_mode():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def generate_with_steering(model, tokenizer, input_device, prompt, direction, alpha, max_new_tokens=8):
    enc = safe_encode(tokenizer, prompt, input_device)
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(enc["input_ids"]).detach().clone()
    d = torch.tensor(direction, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inputs_embeds[0, -1, :] += d * alpha
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.inference_mode():
        gen_ids = model.generate(inputs_embeds=inputs_embeds, attention_mask=enc["attention_mask"], **gen_kwargs)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def get_h_post(model, tokenizer, input_device, prompt):
    """获取 post-norm hidden state (hidden_states[-1][0, -1, :])"""
    enc = safe_encode(tokenizer, prompt, input_device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()


def normalize_dir(d, scale=SCALE):
    n = np.linalg.norm(d)
    if n > 1e-8:
        return d / n * scale
    return d


def make_random_ortho(d_ref, d_model, seed, scale=SCALE):
    """生成正交于 d_ref 的随机方向"""
    np.random.seed(seed)
    rand_dir = np.random.randn(d_model)
    ref_norm = np.linalg.norm(d_ref)
    proj = np.dot(rand_dir, d_ref) / (ref_norm ** 2)
    rand_ortho = rand_dir - proj * d_ref
    return normalize_dir(rand_ortho, scale)


# ============== Exp1: 语义方向提取 ==============

def collect_color_object_data(model, tokenizer, input_device, colors, objects, template):
    """
    收集 "A {color} {object} is a" 的生成结果和 h_post。
    返回: h_post_dict[(color, object)] = np.array, gen_dict[(color, object)] = str
    """
    h_post_dict = {}
    gen_dict = {}
    total = len(colors) * len(objects)
    idx = 0
    for color in colors:
        for obj in objects:
            idx += 1
            prompt = template.format(color=color, object=obj)
            gen_text = generate_greedy(model, tokenizer, input_device, prompt)
            h_post = get_h_post(model, tokenizer, input_device, prompt)
            h_post_dict[(color, obj)] = h_post
            gen_dict[(color, obj)] = gen_text
            if idx % 9 == 0 or idx == total:
                log(f"    [color-object] {idx}/{total}: {color} {obj} -> '{gen_text[len(prompt):].strip()[:30]}'")
            # 定期清理缓存
            if idx % 6 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return h_post_dict, gen_dict


def extract_semantic_directions(h_post_dict, colors, objects_by_category, ref_color="red"):
    """
    从 h_post 数据中提取语义方向：
    - d_color: 颜色变化（固定对象，变化颜色）
    - d_object: 对象变化（同类内，固定颜色）
    - d_category: 类别变化（固定颜色，变化类别）
    """
    # d_color: 平均 over objects of (h(ref_color, obj) - h(other_color, obj))
    d_color_list = []
    other_colors = [c for c in colors if c != ref_color]
    for obj in ALL_OBJECTS:
        for oc in other_colors:
            if (ref_color, obj) in h_post_dict and (oc, obj) in h_post_dict:
                d = h_post_dict[(ref_color, obj)] - h_post_dict[(oc, obj)]
                d_color_list.append(d)
    d_color = np.mean(d_color_list, axis=0) if d_color_list else None

    # d_object: 平均 over within-category pairs of (h(ref_color, obj1) - h(ref_color, obj2))
    d_object_list = []
    for cat, objs in objects_by_category.items():
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if (ref_color, objs[i]) in h_post_dict and (ref_color, objs[j]) in h_post_dict:
                    d = h_post_dict[(ref_color, objs[i])] - h_post_dict[(ref_color, objs[j])]
                    d_object_list.append(d)
    d_object = np.mean(d_object_list, axis=0) if d_object_list else None

    # d_category: mean(h(ref_color, fruit_objs)) - mean(h(ref_color, non_fruit_objs))
    fruit_h = [h_post_dict[(ref_color, obj)] for obj in objects_by_category["fruit"]
               if (ref_color, obj) in h_post_dict]
    non_fruit_h = [h_post_dict[(ref_color, obj)] for obj in ALL_OBJECTS
                   if obj not in objects_by_category["fruit"] and (ref_color, obj) in h_post_dict]
    if fruit_h and non_fruit_h:
        d_category = np.mean(fruit_h, axis=0) - np.mean(non_fruit_h, axis=0)
    else:
        d_category = None

    return {
        "d_color": d_color,
        "d_object": d_object,
        "d_category": d_category,
        "n_color_pairs": len(d_color_list),
        "n_object_pairs": len(d_object_list),
        "n_fruit": len(fruit_h),
        "n_non_fruit": len(non_fruit_h),
    }


# ============== Exp2: d_traj 提取 ==============

def collect_fruit_failures(model, tokenizer, input_device, n_objects=8):
    """
    收集 fruit 类别的成功/失败 hidden states（同 Phase 523）。
    返回: success_h, fail_h, fail_prompts
    """
    objects = FRUIT_OBJECTS_LARGE[:n_objects]
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
                layer = classify_layered(gen_text, prompt, FRUIT_WORDS)
                h_post = get_h_post(model, tokenizer, input_device, prompt)
                if layer in ["S3_cont_phrase", "S4_free"]:
                    success_h.append(h_post)
                elif layer in ["S0_miss", "S1_lexical"]:
                    fail_h.append(h_post)
                    fail_prompts.append(prompt)
                if idx % 10 == 0 or idx == total:
                    log(f"    [fruit-collect] {idx}/{total}: suc={len(success_h)} fail={len(fail_h)}")
                if idx % 6 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return success_h, fail_h, fail_prompts


def extract_d_traj(success_h, fail_h):
    """d_traj = mean(success) - mean(fail)"""
    if len(success_h) < 2 or len(fail_h) < 3:
        return None
    d_traj = np.mean(success_h, axis=0) - np.mean(fail_h, axis=0)
    return d_traj


# ============== Exp3: 互相关矩阵 ==============

def compute_correlation_matrix(directions_dict):
    """计算所有方向之间的余弦相似度"""
    names = list(directions_dict.keys())
    cos_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                d1 = directions_dict[n1]
                d2 = directions_dict[n2]
                if d1 is not None and d2 is not None:
                    n1_norm = np.linalg.norm(d1)
                    n2_norm = np.linalg.norm(d2)
                    cos_val = float(np.dot(d1, d2) / (n1_norm * n2_norm + 1e-12))
                else:
                    cos_val = None
                cos_matrix[f"{n1}_vs_{n2}"] = cos_val
    return cos_matrix


# ============== Exp4: 选择性测试 ==============

def test_selectivity(model, tokenizer, input_device, directions, fail_prompts,
                     fruit_words, n_test=12):
    """
    在 fruit 失败样本上测试每个方向的选择性。
    directions: dict of {name: direction_array} (已归一化到 SCALE)
    """
    if n_test is not None and len(fail_prompts) > n_test:
        fail_prompts = fail_prompts[:n_test]
    n = len(fail_prompts)

    results = {}
    for name, d in directions.items():
        if d is None:
            results[name] = {"s34": 0, "n": n, "rate": 0.0, "error": "direction is None"}
            continue

        s34 = 0
        for i, prompt in enumerate(fail_prompts):
            gen = generate_with_steering(model, tokenizer, input_device, prompt, d, ALPHA)
            layer = classify_layered(gen, prompt, fruit_words)
            if layer in ["S3_cont_phrase", "S4_free"]:
                s34 += 1
            if (i + 1) % 5 == 0 or i + 1 == n:
                log(f"    [selectivity] {name}: {i+1}/{n}, s34={s34}")
            if (i + 1) % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        rate = s34 / n if n > 0 else 0.0
        results[name] = {"s34": s34, "n": n, "rate": float(rate)}
        log(f"  {name}: S3+S4 = {s34}/{n} ({100*rate:.0f}%)")

    return results


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-fruit-objects", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=12)
    args = parser.parse_args()

    if args.smoke:
        args.n_fruit_objects = 4
        args.n_test = 5
        COLORS_SMOKE = ["red", "blue"]
        OBJECTS_SMOKE = {k: v[:2] for k, v in OBJECTS_BY_CATEGORY.items()}
        log("SMOKE TEST MODE: reduced parameters")
    else:
        COLORS_SMOKE = COLORS
        OBJECTS_SMOKE = OBJECTS_BY_CATEGORY

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U_cached(model, args.model)
    d_model = info.d_model

    results = {
        "model": args.model,
        "model_info": {"n_layers": info.n_layers, "d_model": info.d_model},
        "alpha": ALPHA,
        "scale": SCALE,
    }

    out_dir = "results/glm5_phase524_semantic_selectivity"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/phase524_{args.model}_semantic_selectivity.json"

    # ===== Exp1: 语义方向提取 =====
    log("=" * 60)
    log("Exp1: Semantic Direction Extraction (Minimal Contrast)")
    log("=" * 60)

    # Template T1: "A {color} {object} is a"
    log("  Collecting T1 data...")
    all_objects_smoke = [obj for objs in OBJECTS_SMOKE.values() for obj in objs]
    h_post_T1, gen_T1 = collect_color_object_data(
        model, tokenizer, input_device, COLORS_SMOKE, all_objects_smoke, TEMPLATE_T1)

    # 检查 T1 生成是否合理
    t1_success = 0
    t1_total = 0
    for (color, obj), gen in gen_T1.items():
        cat = None
        for k, objs in OBJECTS_SMOKE.items():
            if obj in objs:
                cat = k
                break
        if cat:
            cat_words = [cat, cat + "s", cat.capitalize()]
            prompt = TEMPLATE_T1.format(color=color, object=obj)
            layer = classify_layered(gen, prompt, cat_words)
            if layer in ["S3_cont_phrase", "S4_free"]:
                t1_success += 1
            t1_total += 1
    log(f"  T1 category success: {t1_success}/{t1_total} ({100*t1_success/max(t1_total,1):.0f}%)")

    # Template T2: "A {color} {object} belongs to the category of"
    log("  Collecting T2 data...")
    h_post_T2, gen_T2 = collect_color_object_data(
        model, tokenizer, input_device, COLORS_SMOKE, all_objects_smoke, TEMPLATE_T2)

    # 提取语义方向 (from T1)
    log("  Extracting semantic directions from T1...")
    sem_dirs_T1 = extract_semantic_directions(h_post_T1, COLORS_SMOKE, OBJECTS_SMOKE, ref_color="red")
    log(f"  d_color:   {sem_dirs_T1['n_color_pairs']} pairs, norm={np.linalg.norm(sem_dirs_T1['d_color']):.4f}" if sem_dirs_T1['d_color'] is not None else "  d_color: None")
    log(f"  d_object:  {sem_dirs_T1['n_object_pairs']} pairs, norm={np.linalg.norm(sem_dirs_T1['d_object']):.4f}" if sem_dirs_T1['d_object'] is not None else "  d_object: None")
    log(f"  d_category: fruit={sem_dirs_T1['n_fruit']}, non_fruit={sem_dirs_T1['n_non_fruit']}, norm={np.linalg.norm(sem_dirs_T1['d_category']):.4f}" if sem_dirs_T1['d_category'] is not None else "  d_category: None")

    # 提取语义方向 (from T2, 用于跨模板不变性)
    log("  Extracting semantic directions from T2...")
    sem_dirs_T2 = extract_semantic_directions(h_post_T2, COLORS_SMOKE, OBJECTS_SMOKE, ref_color="red")

    # 中间保存
    results["exp1_semantic_directions"] = {
        "n_colors": len(COLORS_SMOKE),
        "n_objects": len(all_objects_smoke),
        "t1_category_success": t1_success,
        "t1_category_total": t1_total,
        "d_color_norm": float(np.linalg.norm(sem_dirs_T1['d_color'])) if sem_dirs_T1['d_color'] is not None else None,
        "d_object_norm": float(np.linalg.norm(sem_dirs_T1['d_object'])) if sem_dirs_T1['d_object'] is not None else None,
        "d_category_norm": float(np.linalg.norm(sem_dirs_T1['d_category'])) if sem_dirs_T1['d_category'] is not None else None,
        "n_color_pairs": sem_dirs_T1['n_color_pairs'],
        "n_object_pairs": sem_dirs_T1['n_object_pairs'],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"  [Intermediate save] Exp1 saved")

    # ===== Exp2: d_traj 提取 =====
    log("=" * 60)
    log("Exp2: d_traj Extraction (from fruit success/failure)")
    log("=" * 60)

    log("  Collecting fruit success/failure...")
    success_h, fail_h, fail_prompts = collect_fruit_failures(
        model, tokenizer, input_device, args.n_fruit_objects)
    n_suc, n_fail = len(success_h), len(fail_h)
    log(f"  Success: {n_suc}, Fail: {n_fail}")

    d_traj = extract_d_traj(success_h, fail_h)
    if d_traj is not None:
        log(f"  d_traj norm: {np.linalg.norm(d_traj):.4f}")
    else:
        log("  WARNING: Insufficient data for d_traj")

    results["exp2_d_traj"] = {
        "n_success": n_suc,
        "n_failure": n_fail,
        "d_traj_norm": float(np.linalg.norm(d_traj)) if d_traj is not None else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"  [Intermediate save] Exp2 saved")

    # ===== Exp3: 互相关矩阵 =====
    log("=" * 60)
    log("Exp3: Cross-Correlation Matrix")
    log("=" * 60)

    # 归一化所有方向到相同范数
    dirs_raw = {
        "d_color_T1": sem_dirs_T1["d_color"],
        "d_object_T1": sem_dirs_T1["d_object"],
        "d_category_T1": sem_dirs_T1["d_category"],
        "d_category_T2": sem_dirs_T2["d_category"],
        "d_traj": d_traj,
    }

    # 计算原始方向的余弦（不归一化，看自然对齐度）
    cos_matrix = compute_correlation_matrix(dirs_raw)
    log("  Cosine similarity matrix:")
    for pair, val in cos_matrix.items():
        if val is not None:
            log(f"    cos({pair}) = {val:.4f}")
        else:
            log(f"    cos({pair}) = N/A")

    results["exp3_correlation"] = {k: v for k, v in cos_matrix.items()}

    # ===== Exp4: 选择性测试 =====
    log("=" * 60)
    log("Exp4: Selectivity Test (on fruit failures)")
    log("=" * 60)

    if n_fail < 3:
        log(f"  WARNING: Only {n_fail} failures, selectivity test may be unreliable")

    # 准备测试方向（归一化到 SCALE）
    test_directions = {}

    # d_traj (toward success)
    if d_traj is not None:
        test_directions["d_traj(+)"] = normalize_dir(d_traj)

    # d_category (toward fruit: d_category = fruit - non_fruit, so + = toward fruit)
    if sem_dirs_T1["d_category"] is not None:
        d_cat = sem_dirs_T1["d_category"]
        test_directions["d_category(+)"] = normalize_dir(d_cat)
        test_directions["d_category(-)"] = normalize_dir(-d_cat)

    # d_color (irrelevant control: test both signs)
    if sem_dirs_T1["d_color"] is not None:
        d_col = sem_dirs_T1["d_color"]
        test_directions["d_color(+)"] = normalize_dir(d_col)
        test_directions["d_color(-)"] = normalize_dir(-d_col)

    # d_object (irrelevant control: test both signs)
    if sem_dirs_T1["d_object"] is not None:
        d_obj = sem_dirs_T1["d_object"]
        test_directions["d_object(+)"] = normalize_dir(d_obj)
        test_directions["d_object(-)"] = normalize_dir(-d_obj)

    # random_ortho control (orthogonal to d_traj)
    if d_traj is not None:
        rand_ortho = make_random_ortho(d_traj, d_model, seed=42)
        test_directions["random_ortho"] = rand_ortho

    log(f"  Testing {len(test_directions)} directions on {min(n_fail, args.n_test)} failures...")
    selectivity_results = test_selectivity(
        model, tokenizer, input_device, test_directions, fail_prompts,
        FRUIT_WORDS, args.n_test)

    # 计算选择性指标
    s34_cat_pos = selectivity_results.get("d_category(+)", {}).get("rate", 0)
    s34_cat_neg = selectivity_results.get("d_category(-)", {}).get("rate", 0)
    s34_col_pos = selectivity_results.get("d_color(+)", {}).get("rate", 0)
    s34_col_neg = selectivity_results.get("d_color(-)", {}).get("rate", 0)
    s34_obj_pos = selectivity_results.get("d_object(+)", {}).get("rate", 0)
    s34_obj_neg = selectivity_results.get("d_object(-)", {}).get("rate", 0)
    s34_traj = selectivity_results.get("d_traj(+)", {}).get("rate", 0)
    s34_rand = selectivity_results.get("random_ortho", {}).get("rate", 0)

    # Selectivity = S34(d_category_best) / (S34(d_color_best) + S34(d_object_best) + ε)
    cat_best = max(s34_cat_pos, s34_cat_neg)
    col_best = max(s34_col_pos, s34_col_neg)
    obj_best = max(s34_obj_pos, s34_obj_neg)
    selectivity_ratio = cat_best / (col_best + obj_best + 1e-6)

    log(f"\n  --- Selectivity Summary ---")
    log(f"  d_traj(+):        {100*s34_traj:.0f}%")
    log(f"  d_category(+):    {100*s34_cat_pos:.0f}%  (toward fruit)")
    log(f"  d_category(-):    {100*s34_cat_neg:.0f}%  (away from fruit)")
    log(f"  d_color(+):       {100*s34_col_pos:.0f}%  (irrelevant)")
    log(f"  d_color(-):       {100*s34_col_neg:.0f}%  (irrelevant)")
    log(f"  d_object(+):      {100*s34_obj_pos:.0f}%  (irrelevant)")
    log(f"  d_object(-):      {100*s34_obj_neg:.0f}%  (irrelevant)")
    log(f"  random_ortho:     {100*s34_rand:.0f}%  (control)")
    log(f"  Selectivity ratio (cat_best / (col_best + obj_best)): {selectivity_ratio:.2f}")

    # 判断
    if s34_traj > 0 and s34_cat_pos > s34_col_pos + 0.1 and s34_cat_pos > s34_obj_pos + 0.1:
        if s34_cat_pos > s34_traj - 0.1:
            verdict = "d_traj ENCODES CATEGORY: d_category matches d_traj effectiveness and beats irrelevant controls"
        else:
            verdict = "d_traj PARTIALLY encodes category: d_category is selective but weaker than d_traj"
    elif s34_traj > 0 and s34_col_pos > 0.1:
        verdict = "d_traj is NON-SPECIFIC: color direction also fixes failures (general fluency boost)"
    elif s34_traj > 0:
        verdict = "d_traj encodes something OTHER than color/object/category (possibly format/fluency)"
    else:
        verdict = "d_traj INEFFECTIVE on this model (may need intermediate layer)"

    log(f"  Verdict: {verdict}")

    results["exp4_selectivity"] = {
        "test_results": selectivity_results,
        "selectivity_ratio": float(selectivity_ratio),
        "s34_traj": float(s34_traj),
        "s34_cat_pos": float(s34_cat_pos),
        "s34_cat_neg": float(s34_cat_neg),
        "s34_col_pos": float(s34_col_pos),
        "s34_col_neg": float(s34_col_neg),
        "s34_obj_pos": float(s34_obj_pos),
        "s34_obj_neg": float(s34_obj_neg),
        "s34_random": float(s34_rand),
        "verdict": verdict,
    }

    # ===== Exp5: 跨模板不变性 =====
    log("=" * 60)
    log("Exp5: Cross-Template Invariance")
    log("=" * 60)

    if sem_dirs_T1["d_category"] is not None and sem_dirs_T2["d_category"] is not None:
        d_cat_T1 = sem_dirs_T1["d_category"]
        d_cat_T2 = sem_dirs_T2["d_category"]
        cos_cross = float(np.dot(d_cat_T1, d_cat_T2) /
                          (np.linalg.norm(d_cat_T1) * np.linalg.norm(d_cat_T2) + 1e-12))
        log(f"  cos(d_category_T1, d_category_T2) = {cos_cross:.4f}")
        if cos_cross > 0.7:
            template_verdict = "Category direction is STABLE across templates (semantic variable confirmed)"
        elif cos_cross > 0.3:
            template_verdict = "Category direction is PARTIALLY stable across templates"
        else:
            template_verdict = "Category direction is NOT stable across templates (template-dependent)"
        log(f"  Verdict: {template_verdict}")
    else:
        cos_cross = None
        template_verdict = "Insufficient data for cross-template comparison"
        log(f"  {template_verdict}")

    # d_color 和 d_object 的跨模板不变性
    cos_color_cross = None
    cos_object_cross = None
    if sem_dirs_T1["d_color"] is not None and sem_dirs_T2["d_color"] is not None:
        d_col_T1 = sem_dirs_T1["d_color"]
        d_col_T2 = sem_dirs_T2["d_color"]
        cos_color_cross = float(np.dot(d_col_T1, d_col_T2) /
                                (np.linalg.norm(d_col_T1) * np.linalg.norm(d_col_T2) + 1e-12))
        log(f"  cos(d_color_T1, d_color_T2) = {cos_color_cross:.4f}")
    if sem_dirs_T1["d_object"] is not None and sem_dirs_T2["d_object"] is not None:
        d_obj_T1 = sem_dirs_T1["d_object"]
        d_obj_T2 = sem_dirs_T2["d_object"]
        cos_object_cross = float(np.dot(d_obj_T1, d_obj_T2) /
                                 (np.linalg.norm(d_obj_T1) * np.linalg.norm(d_obj_T2) + 1e-12))
        log(f"  cos(d_object_T1, d_object_T2) = {cos_object_cross:.4f}")

    results["exp5_cross_template"] = {
        "cos_d_category": cos_cross,
        "cos_d_color": cos_color_cross,
        "cos_d_object": cos_object_cross,
        "verdict": template_verdict,
    }

    # ===== 保存方向向量 (NPZ) =====
    npz_path = f"{out_dir}/phase524_{args.model}_directions.npz"
    np_arrays = {}
    for name, d in dirs_raw.items():
        if d is not None:
            np_arrays[name] = d
    if np_arrays:
        np.savez(npz_path, **np_arrays)
        log(f"  Directions saved to {npz_path}")

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
