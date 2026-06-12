"""
Phase 469: PC1因果稳健性验证、多变量分解与受控评分范式
=========================================================
核心改进(解决Phase 468硬伤):
1. Exp1: PC1因果强度扫描 — 5个强度等级(0.1x~2.0x), 8对象/类, 50随机方向
2. Exp2: PC1多变量分解 — 用多元回归替代Gram-Schmidt, 分离entropy/position/template
3. Exp3: DS7B受控评分范式 — forced-choice logprob + 多选评分, 避免数学模式触发
4. Exp4: 生成质量基线校正 — baseline-adjusted评估

用法:
  python tests/glm5/phase469_pc1_causal_robustness_controlled.py qwen3 1
  python tests/glm5/phase469_pc1_causal_robustness_controlled.py glm4 1
  python tests/glm5/phase469_pc1_causal_robustness_controlled.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":   ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":     ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle":  ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove", "scarf", "boot"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf", "lamp", "cabinet"],
}

FAMILIES_EN = {
    "fruit":    ["fruit", "produce", "crop"],
    "animal":   ["animal", "creature", "beast"],
    "tool":     ["tool", "implement", "device"],
    "vehicle":  ["vehicle", "transport", "automobile"],
    "clothing": ["clothing", "attire", "wear"],
    "furniture":["furniture", "furnishing", "fixture"],
}

TEMPLATES = {
    "is_a":          "The {obj} is a kind of",
    "category_of":   "The category of {obj} is",
    "classified_as": "A {obj} is commonly classified as",
}

# DS7B受控评分模板 — 避免自由生成
MC_TEMPLATES = {
    "mc_4way": "What category does {obj} belong to? A. fruit B. animal C. vehicle D. tool",
    "yes_no":  "Is {obj} a kind of {cat}? Answer yes or no:",
    "translate":"The Chinese translation of '{obj}' is",
    "fill_blank": "A {obj} is a type of _____.",
}

ROUNDS = {
    1: {k: v[:6] for k, v in CATEGORIES.items()},   # R1: 6对象/类(基础)
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # R2: 8对象/类(确认)
}

# 注入强度等级
INJECTION_RATIOS = [0.1, 0.25, 0.5, 1.0, 2.0]

# 随机方向数量
N_RANDOM_DIRS = 50


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash_attn)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        plog(f"  flash_attention_2 loaded OK")
    except Exception as e:
        plog(f"  flash_attention_2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        plog(f"  Layer distribution: GPU={gpu_count} components, CPU={cpu_count} components")

    return model, tokenizer, device


# ==================== 基础工具函数 ====================
def get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device, pos=-1):
    """提取指定层指定位置的残差流向量"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    captured = {}
    layers = get_layers(model)

    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            captured['resid'] = input[0].detach().float().cpu()

    h = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()

    if 'resid' in captured:
        seq_len = attention_mask.sum().item()
        if pos == -1:
            pos = seq_len - 1
        return captured['resid'][0, pos].numpy(), seq_len
    return None, 0


def get_residual_full_seq(model, tokenizer, prompt, layer_idx, device):
    """提取指定层所有位置的残差流"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    captured = {}
    layers = get_layers(model)

    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            captured['resid'] = input[0].detach().float().cpu()

    h = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()

    if 'resid' in captured:
        seq_len = attention_mask.sum().item()
        return captured['resid'][0, :seq_len].numpy(), seq_len
    return None, 0


def get_final_logits(model, tokenizer, prompt, device):
    """获取最后一层的logits"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[0, -1].float().cpu().numpy()


def run_with_additive_patch(model, tokenizer, prompt, device, patch_layer, delta_vec):
    """加法patch: 在patch_layer的输入中加上delta_vec(最后token位置)"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()

    layers = get_layers(model)
    delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)

    patched = [False]
    def make_hook():
        def hook(module, input, output):
            if not patched[0]:
                patched[0] = True
                if isinstance(output, tuple):
                    out_tensor = output[0].clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return (out_tensor,) + output[1:]
                else:
                    out_tensor = output.clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return out_tensor
            return None
        return hook

    h = layers[patch_layer].register_forward_hook(make_hook())
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()
    return out.logits[0, -1].float().cpu().numpy()


def compute_en_family_margin(logits, tokenizer, target_cat, compete_cats):
    """计算英文候选族边际"""
    target_words = FAMILIES_EN.get(target_cat, [])
    compete_words = []
    for cc in compete_cats:
        compete_words.extend(FAMILIES_EN.get(cc, []))

    vocab = tokenizer.get_vocab()
    target_logits, compete_logits = [], []
    for w in target_words:
        w_clean = w.strip()
        if w_clean in vocab:
            target_logits.append(float(logits[vocab[w_clean]]))
        elif f" {w_clean}" in vocab:
            target_logits.append(float(logits[vocab[f" {w_clean}"]]))
    for w in compete_words:
        w_clean = w.strip()
        if w_clean in vocab:
            compete_logits.append(float(logits[vocab[w_clean]]))
        elif f" {w_clean}" in vocab:
            compete_logits.append(float(logits[vocab[f" {w_clean}"]]))

    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


def logit_entropy(logits_vec):
    """计算logit分布的熵"""
    log_probs = logits_vec - np.max(logits_vec)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
    return -float(np.sum(np.exp(log_probs) * log_probs))


def top1_probability(logits_vec):
    """最高候选的概率"""
    log_probs = logits_vec - np.max(logits_vec)
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    return float(np.max(probs))


def estimate_pcs_at_layer(model, tokenizer, prompts, layer_idx, device, n_pcs=10):
    """估计指定层的主成分"""
    all_vecs = []
    for prompt in prompts:
        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
        if resid is not None:
            all_vecs.append(resid)

    if len(all_vecs) < 6:
        return None, None, None, None

    vecs_matrix = np.array(all_vecs)
    vec_mean = np.mean(vecs_matrix, axis=0)
    vecs_centered = vecs_matrix - vec_mean

    cov = np.cov(vecs_centered.T)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return None, None, None, None

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return cov, eigvals, eigvecs, vec_mean
    except:
        return None, None, None, None


def compute_natural_delta_norm(model, tokenizer, obj_dict, layer_idx, device, cat, n=6):
    """计算某层某类别的自然delta范数"""
    objs = obj_dict.get(cat, [])[:n]
    vecs = []
    for obj in objs:
        prompt = TEMPLATES["is_a"].format(obj=obj)
        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
        if resid is not None:
            vecs.append(resid)

    if len(vecs) < 2:
        return 1.0

    center = np.mean(vecs, axis=0)
    delta_norms = [np.linalg.norm(v - center) for v in vecs]
    return float(np.mean(delta_norms))


# ==================== Exp1: PC1因果强度扫描(改进版) ====================
def exp1_pc1_causal_strength_scan(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    改进:
    1. 8个测试对象(跨4个类别), 不只3个
    2. 5个注入强度(0.1x, 0.25x, 0.5x, 1.0x, 2.0x)
    3. 50个随机方向对照(带z-score和p-value)
    4. 检验entropy单调性
    """
    plog("=== Exp1: PC1因果强度扫描(改进版) ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    key_layers = sorted(set([
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))

    n_obj = 6 if round_num == 1 else 8
    results = {}

    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}

        # ---- 1. 估计PC1 ----
        all_prompts = []
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                all_prompts.append(TEMPLATES["is_a"].format(obj=obj))

        cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
            model, tokenizer, all_prompts, layer_idx, device)

        if eigvecs is None:
            plog(f"    L{layer_idx}: PCA failed, skip")
            continue

        pc1 = eigvecs[:, 0]
        pc1_var_ratio = float(eigvals[0] / max(sum(eigvals[:50]), 1e-10))
        plog(f"    PC1 variance ratio: {pc1_var_ratio:.4f}")
        layer_result["pc1_var_ratio"] = round(pc1_var_ratio, 4)

        # ---- 2. 计算自然PC1投影范数 ----
        natural_proj_norms = []
        for p in all_prompts[:15]:
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                centered = resid - vec_mean
                natural_proj_norms.append(abs(float(np.dot(centered, pc1))))

        if not natural_proj_norms:
            continue

        natural_std = float(np.std(natural_proj_norms))
        if natural_std < 1e-10:
            natural_std = float(np.mean(natural_proj_norms))
        plog(f"    Natural PC1 projection: mean={np.mean(natural_proj_norms):.4f}, std={natural_std:.4f}")
        layer_result["natural_std"] = round(natural_std, 4)

        # ---- 3. 多对象因果注入测试 ----
        test_objects = [
            ("car", "vehicle"), ("dog", "animal"), ("apple", "fruit"),
            ("hammer", "tool"), ("chair", "furniture"), ("shirt", "clothing"),
            ("banana", "fruit"), ("horse", "animal"),
        ][:n_obj]

        causal_results = {}

        for obj_idx, (obj_name, obj_cat) in enumerate(test_objects):
            plog(f"    [{obj_idx+1}/{len(test_objects)}] {obj_name} ({obj_cat})...")
            prompt = TEMPLATES["is_a"].format(obj=obj_name)
            other_cats = [c for c in ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"] if c != obj_cat]

            # 基线logits
            base_logits = get_final_logits(model, tokenizer, prompt, device)
            base_entropy = logit_entropy(base_logits)
            base_top1 = top1_probability(base_logits)
            base_margin, _, _ = compute_en_family_margin(base_logits, tokenizer, obj_cat, other_cats)

            obj_results = {
                "baseline": {
                    "entropy": round(base_entropy, 4),
                    "top1_prob": round(base_top1, 4),
                    "margin": round(base_margin, 4),
                }
            }

            # ---- PC1正负注入(5个强度) ----
            for ratio in INJECTION_RATIOS:
                delta_pos = ratio * natural_std * pc1
                delta_neg = -ratio * natural_std * pc1

                # +PC1
                logits_pos = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_pos)
                ent_pos = logit_entropy(logits_pos)
                top1_pos = top1_probability(logits_pos)
                margin_pos, _, _ = compute_en_family_margin(logits_pos, tokenizer, obj_cat, other_cats)

                # -PC1
                logits_neg = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_neg)
                ent_neg = logit_entropy(logits_neg)
                top1_neg = top1_probability(logits_neg)
                margin_neg, _, _ = compute_en_family_margin(logits_neg, tokenizer, obj_cat, other_cats)

                obj_results[f"+pc1_{ratio}x"] = {
                    "entropy": round(ent_pos, 4),
                    "top1_prob": round(top1_pos, 4),
                    "margin": round(margin_pos, 4),
                    "delta_entropy": round(ent_pos - base_entropy, 4),
                    "delta_top1": round(top1_pos - base_top1, 4),
                    "delta_margin": round(margin_pos - base_margin, 4),
                }
                obj_results[f"-pc1_{ratio}x"] = {
                    "entropy": round(ent_neg, 4),
                    "top1_prob": round(top1_neg, 4),
                    "margin": round(margin_neg, 4),
                    "delta_entropy": round(ent_neg - base_entropy, 4),
                    "delta_top1": round(top1_neg - base_top1, 4),
                    "delta_margin": round(margin_neg - base_margin, 4),
                }

            # ---- PC1消融 ----
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
            if resid is not None:
                centered = resid - vec_mean
                pc1_component = np.dot(centered, pc1) * pc1
                delta_ablate = -pc1_component

                logits_ablate = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_ablate)
                ent_ablate = logit_entropy(logits_ablate)
                top1_ablate = top1_probability(logits_ablate)
                margin_ablate, _, _ = compute_en_family_margin(logits_ablate, tokenizer, obj_cat, other_cats)

                obj_results["ablate_pc1"] = {
                    "entropy": round(ent_ablate, 4),
                    "top1_prob": round(top1_ablate, 4),
                    "margin": round(margin_ablate, 4),
                    "delta_entropy": round(ent_ablate - base_entropy, 4),
                    "delta_top1": round(top1_ablate - base_top1, 4),
                    "delta_margin": round(margin_ablate - base_margin, 4),
                }

            # ---- 50个随机方向对照 ----
            random_entropy_deltas = []
            random_margin_deltas = []
            for ri in range(N_RANDOM_DIRS):
                rand_dir = np.random.randn(len(pc1))
                rand_dir = rand_dir / np.linalg.norm(rand_dir) * natural_std

                logits_rand = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, rand_dir)
                ent_rand = logit_entropy(logits_rand)
                margin_rand, _, _ = compute_en_family_margin(logits_rand, tokenizer, obj_cat, other_cats)

                random_entropy_deltas.append(ent_rand - base_entropy)
                random_margin_deltas.append(margin_rand - base_margin)

                if (ri + 1) % 25 == 0:
                    plog(f"      Random dirs: {ri+1}/{N_RANDOM_DIRS} done")

            rand_ent_mean = float(np.mean(random_entropy_deltas))
            rand_ent_std = float(np.std(random_entropy_deltas))
            rand_margin_mean = float(np.mean(random_margin_deltas))
            rand_margin_std = float(np.std(random_margin_deltas))

            # z-score: PC1效应相对于随机分布
            pc1_delta_ent_1x = obj_results["+pc1_1.0x"]["delta_entropy"]
            z_score = (pc1_delta_ent_1x - rand_ent_mean) / max(rand_ent_std, 1e-6)

            obj_results["random_control"] = {
                "mean_delta_entropy": round(rand_ent_mean, 4),
                "std_delta_entropy": round(rand_ent_std, 4),
                "mean_delta_margin": round(rand_margin_mean, 4),
                "std_delta_margin": round(rand_margin_std, 4),
                "z_score_pc1_vs_random": round(z_score, 4),
            }

            # ---- 单调性检验 ----
            # 如果+PC1的entropy在5个强度下单调递减(或递增), 则PC1是线性因果轴
            ent_values_pos = [obj_results[f"+pc1_{r}x"]["delta_entropy"] for r in INJECTION_RATIOS]
            ent_values_neg = [obj_results[f"-pc1_{r}x"]["delta_entropy"] for r in INJECTION_RATIOS]

            # 检查+PC1的delta_entropy是否单调
            monotonic_pos = all(ent_values_pos[i] <= ent_values_pos[i+1] + 0.01 for i in range(len(ent_values_pos)-1)) or \
                           all(ent_values_pos[i] >= ent_values_pos[i+1] - 0.01 for i in range(len(ent_values_pos)-1))
            monotonic_neg = all(ent_values_neg[i] <= ent_values_neg[i+1] + 0.01 for i in range(len(ent_values_neg)-1)) or \
                           all(ent_values_neg[i] >= ent_values_neg[i+1] - 0.01 for i in range(len(ent_values_neg)-1))

            obj_results["monotonicity"] = {
                "pos_monotonic": monotonic_pos,
                "neg_monotonic": monotonic_neg,
                "both_monotonic": monotonic_pos and monotonic_neg,
            }

            causal_results[obj_name] = obj_results

            plog(f"      base_ent={base_entropy:.3f}, "
                 f"+pc1_1x_Δent={obj_results['+pc1_1.0x']['delta_entropy']:.4f}, "
                 f"-pc1_1x_Δent={obj_results['-pc1_1.0x']['delta_entropy']:.4f}, "
                 f"z={z_score:.2f}, mono={monotonic_pos and monotonic_neg}")

        layer_result["causal_results"] = causal_results

        # ---- 4. 汇总统计 ----
        pc1_entropy_effects = []
        random_entropy_means = []
        random_entropy_stds = []
        z_scores = []
        n_monotonic = 0

        for obj_name in causal_results:
            r = causal_results[obj_name]
            pc1_entropy_effects.append(r["+pc1_1.0x"]["delta_entropy"])
            random_entropy_means.append(r["random_control"]["mean_delta_entropy"])
            random_entropy_stds.append(r["random_control"]["std_delta_entropy"])
            z_scores.append(r["random_control"]["z_score_pc1_vs_random"])
            if r["monotonicity"]["both_monotonic"]:
                n_monotonic += 1

        mean_z = float(np.mean(z_scores))
        # p-value近似: P(|Z|>|mean_z|*sqrt(n))
        from scipy import stats
        t_stat = mean_z * np.sqrt(len(z_scores)) if len(z_scores) > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if abs(t_stat) < 100 else 0.0

        layer_result["summary"] = {
            "pc1_mean_delta_entropy": round(float(np.mean(pc1_entropy_effects)), 4),
            "random_mean_delta_entropy": round(float(np.mean(random_entropy_means)), 4),
            "random_mean_std_entropy": round(float(np.mean(random_entropy_stds)), 4),
            "mean_z_score": round(mean_z, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "n_monotonic_out_of": f"{n_monotonic}/{len(causal_results)}",
            "pc1_vs_random_entropy_ratio": round(
                float(np.mean(pc1_entropy_effects)) / max(abs(float(np.mean(random_entropy_means))), 1e-6), 2),
            "is_significant_causal_axis": abs(mean_z) > 2.0 and p_value < 0.05,
        }

        plog(f"    Summary: PC1_Δent={layer_result['summary']['pc1_mean_delta_entropy']:.4f}, "
             f"random_Δent={layer_result['summary']['random_mean_delta_entropy']:.4f}±{layer_result['summary']['random_mean_std_entropy']:.4f}, "
             f"mean_z={mean_z:.2f}, p={p_value:.4f}, "
             f"monotonic={n_monotonic}/{len(causal_results)}, "
             f"significant={layer_result['summary']['is_significant_causal_axis']}")

        results[f"L{layer_idx}"] = layer_result

    return results


# ==================== Exp2: PC1多变量分解(多元回归) ====================
def exp2_pc1_multivariate_decomposition(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    用多元回归替代Gram-Schmidt:
    - 收集每个样本的: PC1投影, entropy, 序列位置, 模板类型, 对象类别
    - 用sklearn Ridge回归: PC1投影 = a*entropy + b*position + c*template + d*category + residual
    - 计算每个预测变量的偏相关和标准化系数
    """
    plog("=== Exp2: PC1多变量分解(多元回归) ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    key_layers = sorted(set([
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))

    n_obj = 6 if round_num == 1 else 8
    results = {}

    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}

        # ---- 1. 估计PC1 ----
        all_prompts = []
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                all_prompts.append(TEMPLATES["is_a"].format(obj=obj))

        cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
            model, tokenizer, all_prompts, layer_idx, device)

        if eigvecs is None:
            continue

        pc1 = eigvecs[:, 0]

        # ---- 2. 收集样本数据 ----
        # 每个样本: (pc1_projection, entropy, position, template_id, category_id)
        sample_data = []

        # (a) 不同对象 + is_a模板
        cat_list = ["fruit", "animal", "vehicle", "tool", "furniture"]
        for cat_idx, cat in enumerate(cat_list):
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj_idx, obj in enumerate(objs):
                prompt = TEMPLATES["is_a"].format(obj=obj)
                resid, seq_len = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
                if resid is None:
                    continue
                logits = get_final_logits(model, tokenizer, prompt, device)
                ent = logit_entropy(logits)
                centered = resid - vec_mean
                pc1_proj = float(np.dot(centered, pc1))
                sample_data.append({
                    "pc1_proj": pc1_proj,
                    "entropy": ent,
                    "position": seq_len,
                    "template_id": 0,  # is_a
                    "category_id": cat_idx,
                    "category": cat,
                })

        # (b) 不同模板 (用3个对象)
        template_keys = list(TEMPLATES.keys())
        for tmpl_idx, tmpl_key in enumerate(template_keys):
            for obj in ["apple", "car", "dog"]:
                prompt = TEMPLATES[tmpl_key].format(obj=obj)
                resid, seq_len = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
                if resid is None:
                    continue
                logits = get_final_logits(model, tokenizer, prompt, device)
                ent = logit_entropy(logits)
                centered = resid - vec_mean
                pc1_proj = float(np.dot(centered, pc1))
                sample_data.append({
                    "pc1_proj": pc1_proj,
                    "entropy": ent,
                    "position": seq_len,
                    "template_id": tmpl_idx,
                    "category_id": 0,
                    "category": "fruit",
                })

        # (c) 不同位置 (用1个长句的各位置)
        long_prompt = "The apple is a kind of fruit and the banana is also a type of fruit"
        resid_full, seq_len = get_residual_full_seq(model, tokenizer, long_prompt, layer_idx, device)
        if resid_full is not None and seq_len > 4:
            for pos in range(seq_len):
                centered = resid_full[pos] - vec_mean
                pc1_proj = float(np.dot(centered, pc1))
                sample_data.append({
                    "pc1_proj": pc1_proj,
                    "entropy": 0.0,  # 位置样本没有独立entropy
                    "position": pos,
                    "template_id": 0,
                    "category_id": 0,
                    "category": "position_sample",
                })

        if len(sample_data) < 10:
            plog(f"    L{layer_idx}: too few samples ({len(sample_data)}), skip")
            continue

        plog(f"    Collected {len(sample_data)} samples")

        # ---- 3. 多元回归 ----
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import pearsonr

        # 构造特征矩阵(排除位置样本做独立分析)
        obj_samples = [s for s in sample_data if s["category"] != "position_sample"]
        pos_samples = [s for s in sample_data if s["category"] == "position_sample"]

        # (a) 对象样本: PC1_proj ~ entropy + template_id + category_id
        if len(obj_samples) >= 8:
            y = np.array([s["pc1_proj"] for s in obj_samples])
            X = np.array([[s["entropy"], s["template_id"], s["category_id"]] for s in obj_samples])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            reg = Ridge(alpha=1.0)
            reg.fit(X_scaled, y)

            r2 = reg.score(X_scaled, y)
            coef_names = ["entropy", "template", "category"]
            coefs = dict(zip(coef_names, reg.coef_.tolist()))

            # 偏相关: 控制其他变量后每个变量与PC1投影的相关
            partial_corrs = {}
            for i, name in enumerate(coef_names):
                # 残差化: y对其他X回归的残差 vs xi对其他X回归的残差
                other_cols = [j for j in range(X_scaled.shape[1]) if j != i]
                if len(other_cols) > 0:
                    reg_y = Ridge(alpha=1.0).fit(X_scaled[:, other_cols], y)
                    resid_y = y - reg_y.predict(X_scaled[:, other_cols])
                    reg_xi = Ridge(alpha=1.0).fit(X_scaled[:, other_cols], X_scaled[:, i])
                    resid_xi = X_scaled[:, i] - reg_xi.predict(X_scaled[:, other_cols])
                    if np.std(resid_y) > 1e-10 and np.std(resid_xi) > 1e-10:
                        pr, _ = pearsonr(resid_y, resid_xi)
                        partial_corrs[name] = round(float(pr), 4)
                    else:
                        partial_corrs[name] = 0.0
                else:
                    pr, _ = pearsonr(y, X_scaled[:, i])
                    partial_corrs[name] = round(float(pr), 4)

            layer_result["regression"] = {
                "r2": round(r2, 4),
                "coefficients": {k: round(v, 4) for k, v in coefs.items()},
                "partial_correlations": partial_corrs,
                "n_samples": len(obj_samples),
            }

            plog(f"    Regression R²={r2:.4f}, partial_corr: "
                 f"entropy={partial_corrs.get('entropy', 0):.3f}, "
                 f"template={partial_corrs.get('template', 0):.3f}, "
                 f"category={partial_corrs.get('category', 0):.3f}")

        # (b) 位置样本: PC1_proj ~ position
        if len(pos_samples) >= 5:
            pos_positions = np.array([s["position"] for s in pos_samples], dtype=float)
            pos_projs = np.array([s["pc1_proj"] for s in pos_samples])

            if np.std(pos_projs) > 1e-10 and np.std(pos_positions) > 1e-10:
                corr_pos, p_pos = pearsonr(pos_positions, pos_projs)
            else:
                corr_pos, p_pos = 0.0, 1.0

            layer_result["position_correlation"] = {
                "corr": round(corr_pos, 4),
                "p_value": round(p_pos, 6),
                "n_positions": len(pos_samples),
            }
            plog(f"    Position correlation: r={corr_pos:.4f}, p={p_pos:.4f}")

        # (c) W_U对齐检查 — 使用右奇异向量(在residual space)
        W_U = get_W_U(model, model_name)
        if W_U is not None:
            try:
                # W_U shape: [vocab_size, d_model]
                # 右奇异向量在d_model空间, 左奇异向量在vocab空间
                U, S, Vt = np.linalg.svd(W_U, full_matrices=False)
                # Vt shape: [min(vocab, d_model), d_model] — 行是d_model空间中的向量
                # 右奇异向量 = Vt的行
                wu_right_sv1 = Vt[0]  # W_U的第一右奇异向量(在residual space)
                readout_alignment_right = float(abs(np.dot(pc1, wu_right_sv1)))

                # 也检查左奇异向量(错误用法, 用于对比)
                wu_left_sv1 = U[:, 0]  # 在vocab space
                # 不能直接与pc1(在residual space)做点积, 但可以报告形状

                # 检查前10个右奇异向量
                readout_alignments = []
                for k in range(min(10, Vt.shape[0])):
                    ra = float(abs(np.dot(pc1, Vt[k])))
                    readout_alignments.append(round(ra, 4))

                layer_result["readout_alignment"] = {
                    "pc1_vs_right_sv1": round(readout_alignment_right, 4),
                    "top10_right_sv_alignments": readout_alignments,
                    "W_U_shape": list(W_U.shape),
                    "note": "right singular vectors are in residual space",
                }
                plog(f"    Readout alignment (right SV1): {readout_alignment_right:.4f}")
            except Exception as e:
                plog(f"    Readout alignment failed: {e}")

        # (d) 简单相关矩阵
        if len(obj_samples) >= 8:
            ent_vals = [s["entropy"] for s in obj_samples]
            proj_vals = [s["pc1_proj"] for s in obj_samples]

            if np.std(ent_vals) > 1e-10 and np.std(proj_vals) > 1e-10:
                corr_ent, p_ent = pearsonr(proj_vals, ent_vals)
            else:
                corr_ent, p_ent = 0.0, 1.0

            layer_result["simple_correlations"] = {
                "pc1_entropy_r": round(corr_ent, 4),
                "pc1_entropy_p": round(p_ent, 6),
            }
            plog(f"    Simple PC1-entropy corr: r={corr_ent:.4f}, p={p_ent:.4f}")

        results[f"L{layer_idx}"] = layer_result

    return results


# ==================== Exp3: DS7B受控评分范式 ====================
def exp3_ds7b_controlled_scoring(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    为DS7B(和所有模型)建立不触发数学模式的受控评分范式:
    1. forced-choice logprob: 多选题格式, 只比较选项的logprob
    2. yes/no verification: 判断对错
    3. 翻译评分: 翻译任务避免分类模板
    4. 自由生成基线记录(对照用)

    对所有3个模型运行, 不仅DS7B
    """
    plog("=== Exp3: 受控评分范式 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    n_obj = 4 if round_num == 1 else 6
    results = {}

    # ---- 1. 多选题logprob评分 ----
    plog("  Sub-exp 3a: Multiple-choice logprob scoring...")
    mc_results = {}

    for cat in ["fruit", "animal", "vehicle", "tool"]:
        objs = obj_dict.get(cat, [])[:n_obj]
        cat_results = []

        for obj in objs:
            # 4选1格式
            prompt = MC_TEMPLATES["mc_4way"].format(obj=obj)

            logits = get_final_logits(model, tokenizer, prompt, device)

            # 获取A/B/C/D选项的logprob
            vocab = tokenizer.get_vocab()
            option_tokens = {}
            for opt in ["A", "B", "C", "D"]:
                if opt in vocab:
                    option_tokens[opt] = float(logits[vocab[opt]])
                elif f" {opt}" in vocab:
                    option_tokens[opt] = float(logits[vocab[f" {opt}"]])

            # 确定正确答案
            correct_map = {"fruit": "A", "animal": "B", "vehicle": "C", "tool": "D"}
            correct_opt = correct_map.get(cat, "A")

            # softmax得到选项概率
            if option_tokens:
                opt_vals = list(option_tokens.values())
                opt_max = max(opt_vals)
                opt_probs = np.exp(np.array(opt_vals) - opt_max)
                opt_probs = opt_probs / opt_probs.sum()
                opt_keys = list(option_tokens.keys())
                opt_prob_dict = {k: round(float(p), 4) for k, p in zip(opt_keys, opt_probs)}

                correct_prob = opt_prob_dict.get(correct_opt, 0.0)
                top_choice = opt_keys[np.argmax(opt_vals)]

                cat_results.append({
                    "object": obj,
                    "correct_option": correct_opt,
                    "correct_prob": correct_prob,
                    "top_choice": top_choice,
                    "is_correct": top_choice == correct_opt,
                    "option_probs": opt_prob_dict,
                })

        if cat_results:
            accuracy = sum(1 for r in cat_results if r["is_correct"]) / len(cat_results)
            mc_results[cat] = {
                "accuracy": round(accuracy, 4),
                "items": cat_results,
            }
            plog(f"    {cat}: accuracy={accuracy:.2%} ({len(cat_results)} items)")

    results["mc_scoring"] = mc_results

    # ---- 2. Yes/No verification评分 ----
    plog("  Sub-exp 3b: Yes/No verification scoring...")
    yn_results = {}

    for cat in ["fruit", "animal", "vehicle", "tool"]:
        objs = obj_dict.get(cat, [])[:n_obj]
        cat_results = []

        for obj in objs:
            # 正确类别
            prompt_correct = MC_TEMPLATES["yes_no"].format(obj=obj, cat=cat)
            logits_correct = get_final_logits(model, tokenizer, prompt_correct, device)

            # 错误类别
            wrong_cats = [c for c in ["fruit", "animal", "vehicle", "tool", "furniture"] if c != cat]
            wrong_cat = wrong_cats[0]  # 取第一个错误类别
            prompt_wrong = MC_TEMPLATES["yes_no"].format(obj=obj, cat=wrong_cat)
            logits_wrong = get_final_logits(model, tokenizer, prompt_wrong, device)

            # 比较"yes"和"no"的logprob
            vocab = tokenizer.get_vocab()
            yes_scores, no_scores = {}, {}

            for yes_tok in ["yes", "Yes", " yes", " Yes"]:
                if yes_tok in vocab:
                    yes_scores[yes_tok] = float(logits_correct[vocab[yes_tok]])
            for no_tok in ["no", "No", " no", " No"]:
                if no_tok in vocab:
                    no_scores[no_tok] = float(logits_wrong[vocab[no_tok]])

            if yes_scores and no_scores:
                best_yes = max(yes_scores.values())
                best_no_correct = max(no_scores.values())  # 对错误问题回答no的logprob

                # 对错误问题也应该得到"no"
                logits_wrong_yn = get_final_logits(model, tokenizer, prompt_wrong, device)
                yes_scores_wrong, no_scores_wrong = {}, {}
                for yes_tok in ["yes", "Yes", " yes", " Yes"]:
                    if yes_tok in vocab:
                        yes_scores_wrong[yes_tok] = float(logits_wrong_yn[vocab[yes_tok]])
                for no_tok in ["no", "No", " no", " No"]:
                    if no_tok in vocab:
                        no_scores_wrong[no_tok] = float(logits_wrong_yn[vocab[no_tok]])

                best_yes_wrong = max(yes_scores_wrong.values()) if yes_scores_wrong else 0
                best_no_wrong = max(no_scores_wrong.values()) if no_scores_wrong else 0

                # 判断: 正确问题→yes, 错误问题→no
                correct_judgment = best_yes > best_no_correct
                wrong_judgment = best_no_wrong > best_yes_wrong

                cat_results.append({
                    "object": obj,
                    "correct_cat": cat,
                    "wrong_cat": wrong_cat,
                    "correct_judgment_yes": correct_judgment,
                    "wrong_judgment_no": wrong_judgment,
                    "both_correct": correct_judgment and wrong_judgment,
                })

        if cat_results:
            acc = sum(1 for r in cat_results if r["both_correct"]) / len(cat_results)
            yn_results[cat] = {
                "accuracy": round(acc, 4),
                "items": cat_results,
            }
            plog(f"    {cat}: Y/N accuracy={acc:.2%}")

    results["yn_scoring"] = yn_results

    # ---- 3. 自由生成基线(检测数学模式触发) ----
    plog("  Sub-exp 3c: Free generation baseline (math detection)...")
    gen_results = {}

    test_objects = ["car", "dog", "apple", "hammer", "chair", "shirt"]

    for tmpl_key in ["is_a", "category_of"]:
        tmpl_str = TEMPLATES[tmpl_key]
        math_triggers = 0
        gen_texts = []

        for obj in test_objects[:4]:
            prompt = tmpl_str.format(obj=obj)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            gen_kwargs = dict(max_new_tokens=20, do_sample=False, repetition_penalty=1.2)
            with torch.no_grad():
                gen_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # 检测数学模式
            math_indicators = ["equation", "formula", "matrix", "vector", "function",
                             "graph", "theorem", "integral", "polynomial", "1.", "2.",
                             "0.", "0x", "∞", "Σ", "π", "∫", "x=", "n=", "f(x)"]
            math_trigger = any(ind in gen_text[len(prompt):] for ind in math_indicators)
            if math_trigger:
                math_triggers += 1

            gen_texts.append({
                "object": obj,
                "gen_part": gen_text[len(prompt):].strip()[:100],
                "math_triggered": math_trigger,
            })

        gen_results[tmpl_key] = {
            "math_trigger_rate": round(math_triggers / len(gen_texts), 4) if gen_texts else 0,
            "items": gen_texts,
        }
        plog(f"    {tmpl_key}: math_trigger_rate={gen_results[tmpl_key]['math_trigger_rate']:.2%}")

    results["free_generation"] = gen_results

    # ---- 4. 汇总 ----
    mc_accs = [v["accuracy"] for v in mc_results.values() if "accuracy" in v]
    yn_accs = [v["accuracy"] for v in yn_results.values() if "accuracy" in v]
    math_rates = [v["math_trigger_rate"] for v in gen_results.values()]

    results["summary"] = {
        "mc_mean_accuracy": round(float(np.mean(mc_accs)), 4) if mc_accs else 0,
        "yn_mean_accuracy": round(float(np.mean(yn_accs)), 4) if yn_accs else 0,
        "mean_math_trigger_rate": round(float(np.mean(math_rates)), 4) if math_rates else 0,
        "recommendation": "Use forced-choice/MC scoring" if float(np.mean(math_rates)) > 0.3 else "Free generation OK",
    }
    plog(f"  Summary: MC_acc={results['summary']['mc_mean_accuracy']:.2%}, "
         f"YN_acc={results['summary']['yn_mean_accuracy']:.2%}, "
         f"math_rate={results['summary']['mean_math_trigger_rate']:.2%}")

    return results


# ==================== Exp4: 生成质量基线校正 ====================
def exp4_baseline_corrected_quality(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    所有注入实验记录baseline质量, 计算ΔQuality
    分类: baseline_good→injected_good, baseline_good→injected_bad, etc.
    """
    plog("=== Exp4: 生成质量基线校正 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    key_layer = n_layers // 2
    n_obj = 4 if round_num == 1 else 6

    results = {}

    # 估计PC1
    all_prompts = []
    for cat in ["fruit", "animal", "vehicle"]:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            all_prompts.append(TEMPLATES["is_a"].format(obj=obj))

    cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
        model, tokenizer, all_prompts, key_layer, device)

    if eigvecs is None:
        return {"error": "PCA failed"}

    pc1 = eigvecs[:, 0]

    # 自然标准差
    natural_proj_norms = []
    for p in all_prompts[:10]:
        resid, _ = get_residual_at_layer_pos(model, tokenizer, p, key_layer, device)
        if resid is not None:
            centered = resid - vec_mean
            natural_proj_norms.append(abs(float(np.dot(centered, pc1))))
    natural_std = float(np.std(natural_proj_norms)) if natural_proj_norms else 1.0

    test_objects = [
        ("car", "vehicle"), ("dog", "animal"), ("apple", "fruit"),
        ("hammer", "tool"), ("chair", "furniture"),
    ][:n_obj]

    quality_results = []

    for obj_name, obj_cat in test_objects:
        prompt = TEMPLATES["is_a"].format(obj=obj_name)
        other_cats = [c for c in ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"] if c != obj_cat]

        # ---- baseline logits和生成 ----
        base_logits = get_final_logits(model, tokenizer, prompt, device)
        base_entropy = logit_entropy(base_logits)
        base_margin, _, _ = compute_en_family_margin(base_logits, tokenizer, obj_cat, other_cats)
        base_top1 = top1_probability(base_logits)

        # baseline生成
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                    max_new_tokens=20, do_sample=False, repetition_penalty=1.2)
        base_gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # 检测数学触发
        math_indicators = ["equation", "formula", "matrix", "vector", "function",
                         "1.", "2.", "0.", "∫", "Σ"]
        base_math = any(ind in base_gen_text[len(prompt):] for ind in math_indicators)

        # ---- +PC1注入(1x) ----
        delta_pos = natural_std * pc1
        logits_pos = run_with_additive_patch(model, tokenizer, prompt, device, key_layer, delta_pos)
        pos_entropy = logit_entropy(logits_pos)
        pos_margin, _, _ = compute_en_family_margin(logits_pos, tokenizer, obj_cat, other_cats)
        pos_top1 = top1_probability(logits_pos)

        # ---- -PC1注入(1x) ----
        delta_neg = -natural_std * pc1
        logits_neg = run_with_additive_patch(model, tokenizer, prompt, device, key_layer, delta_neg)
        neg_entropy = logit_entropy(logits_neg)
        neg_margin, _, _ = compute_en_family_margin(logits_neg, tokenizer, obj_cat, other_cats)
        neg_top1 = top1_probability(logits_neg)

        # ---- 评估质量变化 ----
        # baseline质量: margin > 0 且 entropy < 6(合理范围)
        base_quality_good = base_margin > 0 and not base_math

        pos_quality_good = pos_margin > 0
        neg_quality_good = neg_margin > 0

        def classify_quality(base_good, injected_good):
            if base_good and injected_good:
                return "baseline_good→injected_good"
            elif base_good and not injected_good:
                return "baseline_good→injected_bad"
            elif not base_good and injected_good:
                return "baseline_bad→injected_better"
            else:
                return "baseline_bad→injected_worse"

        quality_results.append({
            "object": obj_name,
            "category": obj_cat,
            "baseline": {
                "entropy": round(base_entropy, 4),
                "top1_prob": round(base_top1, 4),
                "margin": round(base_margin, 4),
                "math_triggered": base_math,
                "quality_good": base_quality_good,
                "gen_text": base_gen_text[len(prompt):].strip()[:80],
            },
            "+pc1_1x": {
                "entropy": round(pos_entropy, 4),
                "top1_prob": round(pos_top1, 4),
                "margin": round(pos_margin, 4),
                "delta_entropy": round(pos_entropy - base_entropy, 4),
                "delta_margin": round(pos_margin - base_margin, 4),
                "quality_good": pos_quality_good,
                "quality_class": classify_quality(base_quality_good, pos_quality_good),
            },
            "-pc1_1x": {
                "entropy": round(neg_entropy, 4),
                "top1_prob": round(neg_top1, 4),
                "margin": round(neg_margin, 4),
                "delta_entropy": round(neg_entropy - base_entropy, 4),
                "delta_margin": round(neg_margin - base_margin, 4),
                "quality_good": neg_quality_good,
                "quality_class": classify_quality(base_quality_good, neg_quality_good),
            },
        })

    # ---- 汇总 ----
    quality_classes_pos = [r["+pc1_1x"]["quality_class"] for r in quality_results]
    quality_classes_neg = [r["-pc1_1x"]["quality_class"] for r in quality_results]

    pos_summary = {c: quality_classes_pos.count(c) for c in set(quality_classes_pos)}
    neg_summary = {c: quality_classes_neg.count(c) for c in set(quality_classes_neg)}

    base_good_count = sum(1 for r in quality_results if r["baseline"]["quality_good"])
    base_math_count = sum(1 for r in quality_results if r["baseline"]["math_triggered"])

    results["items"] = quality_results
    results["summary"] = {
        "test_layer": key_layer,
        "n_objects": len(quality_results),
        "baseline_good_count": base_good_count,
        "baseline_math_triggered": base_math_count,
        "+pc1_quality_distribution": pos_summary,
        "-pc1_quality_distribution": neg_summary,
    }

    plog(f"  Summary: baseline_good={base_good_count}/{len(quality_results)}, "
         f"math_triggered={base_math_count}, "
         f"+pc1 dist={pos_summary}, -pc1 dist={neg_summary}")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS[round_num]

    plog(f"Phase 469: {model_name}, Round {round_num}")
    plog(f"Objects per category: {len(list(obj_dict.values())[0])}")
    plog(f"Random directions: {N_RANDOM_DIRS}")
    plog(f"Injection ratios: {INJECTION_RATIOS}")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "config": {
            "n_random_dirs": N_RANDOM_DIRS,
            "injection_ratios": INJECTION_RATIOS,
            "n_obj_per_cat": len(list(obj_dict.values())[0]),
        },
    }

    # Exp1: PC1因果强度扫描(核心改进)
    t1 = time.time()
    all_results["exp1_pc1_causal_scan"] = exp1_pc1_causal_strength_scan(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: PC1多变量分解(回归替代Gram-Schmidt)
    t2 = time.time()
    all_results["exp2_pc1_decomposition"] = exp2_pc1_multivariate_decomposition(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: 受控评分范式
    t3 = time.time()
    all_results["exp3_controlled_scoring"] = exp3_ds7b_controlled_scoring(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # Exp4: 生成质量基线校正
    t4 = time.time()
    all_results["exp4_baseline_quality"] = exp4_baseline_corrected_quality(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp4 done in {time.time()-t4:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase469_{model_name}_r{round_num}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        return str(obj)

    all_results = convert(all_results)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # ---- 4. 释放模型 ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    plog(f"Phase 469 {model_name} Round {round_num} complete in {total_time:.0f}s")


if __name__ == "__main__":
    main()
