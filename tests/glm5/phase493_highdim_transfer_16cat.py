"""
Phase 493: 高维支撑子空间、跨层刹车释放传递与16类全局验证
===========================================================

Phase 492遗留问题:
1. Qwen3末层support效应高维分散, 前8个SVD方向预测仅12.5%
2. L(n-2)刹车如何传递到L(n-1)释放层 — 跨层因果未验证
3. 末端机制是否在16类中普遍成立

Phase 493核心实验:
- Exp1: 高维SVD分解(k=8,16,32,64), 解决Qwen3 support/inhibit预测
- Exp2: 跨层因果追踪 — 在L(n-2)操作后追踪L(n-1)残差变化
- Exp3: 16类全局验证 — 验证末端刹车-释放机制普遍性

用法:
  python tests/glm5/phase493_highdim_transfer_16cat.py qwen3 1
  python tests/glm5/phase493_highdim_transfer_16cat.py glm4 1
  python tests/glm5/phase493_highdim_transfer_16cat.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES_8 = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum",
                  "cherry", "lemon", "lime", "kiwi", "melon", "fig", "date", "guava"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish",
                  "tiger", "deer", "wolf", "fox", "owl", "hawk", "crab", "frog"],
    "clothing":  ["shirt", "dress", "hat", "coat", "jacket", "skirt", "scarf", "boot",
                  "pants", "vest", "glove", "sock", "belt", "tie", "cap", "robe"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "cake", "salad", "meat",
                  "pizza", "taco", "stew", "pie", "roll", "ham", "corn", "bean"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "plane", "boat", "motorcycle",
                  "van", "scooter", "taxi", "tram", "ship", "jet", "subway", "sled"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "moss", "vine", "shrub",
                  "palm", "oak", "pine", "rose", "lily", "weed", "reed", "cactus"],
    "tool":      ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "ruler", "knife",
                  "screwdriver", "axe", "shovel", "mallet", "clamp", "vise", "file", "level"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "cabinet", "bench",
                  "stool", "dresser", "wardrobe", "couch", "hutch", "ottoman", "cot", "armoire"],
}

# 扩展到16类
CATEGORIES_16 = {**CATEGORIES_8,
    "body_part":   ["hand", "foot", "head", "arm", "leg", "eye", "ear", "nose",
                    "mouth", "finger", "toe", "knee", "elbow", "wrist", "ankle", "hip"],
    "building":    ["house", "church", "school", "store", "factory", "tower", "bridge", "castle",
                    "barn", "temple", "museum", "hotel", "cabin", "palace", "fort", "shrine"],
    "container":   ["box", "bottle", "cup", "jar", "bowl", "basket", "bucket", "pot",
                    "can", "tub", "mug", "flask", "vase", "crate", "barrel", "tub"],
    "device":      ["phone", "radio", "clock", "lamp", "camera", "speaker", "monitor", "printer",
                    "fan", "heater", "scale", "timer", "scanner", "router", "switch", "pager"],
    "place":       ["park", "beach", "forest", "lake", "mountain", "river", "desert", "island",
                    "valley", "cliff", "cave", "swamp", "meadow", "canyon", "bay", "glacier"],
    "material":    ["wood", "stone", "metal", "glass", "paper", "cloth", "rubber", "leather",
                    "plastic", "cotton", "silk", "wool", "clay", "sand", "gold", "iron"],
    "emotion":     ["joy", "fear", "anger", "sadness", "hope", "love", "pride", "shame",
                    "guilt", "envy", "pity", "awe", "grief", "bliss", "rage", "dread"],
    "action":      ["run", "walk", "jump", "swim", "climb", "throw", "catch", "push",
                    "pull", "lift", "carry", "drag", "drop", "kick", "hit", "hold"],
}

CAT_NAMES_8 = list(CATEGORIES_8.keys())
CAT_NAMES_16 = list(CATEGORIES_16.keys())


def get_model_and_tokenizer(model_name):
    """BF16加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        plog(f"{model_name}: GPU={gpu_count}, CPU={cpu_count}")

    device = next(model.parameters()).device
    return model, tokenizer, device


def encode_prompts(tokenizer, device, objects, template="The {obj} is a kind of"):
    """编码提示"""
    texts = [template.format(obj=obj) for obj in objects]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def get_category_hidden(model, tokenizer, device, cat_name, objects, layer_idx,
                        template="The {obj} is a kind of"):
    """获取某类在某层的所有hidden states"""
    layers = get_layers(model)
    inputs = encode_prompts(tokenizer, device, objects, template)

    captured = {}
    def make_hook(key):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hook = layers[layer_idx].register_forward_hook(make_hook("h"))
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    h = captured["h"]
    mask = inputs["attention_mask"].bool()
    last_idx = mask.sum(dim=1) - 1
    h_last = h[torch.arange(h.size(0)), last_idx.cpu()]
    return h_last.numpy()


def get_category_hidden_two_layers(model, tokenizer, device, cat_name, objects,
                                   layer1, layer2, template="The {obj} is a kind of"):
    """一次前向传播同时获取两层的hidden states"""
    layers = get_layers(model)
    inputs = encode_prompts(tokenizer, device, objects, template)

    captured = {}
    def make_hook(key):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hooks = []
    hooks.append(layers[layer1].register_forward_hook(make_hook(f"L{layer1}")))
    hooks.append(layers[layer2].register_forward_hook(make_hook(f"L{layer2}")))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    mask = inputs["attention_mask"].bool()
    last_idx = mask.sum(dim=1) - 1

    results = {}
    for key in captured:
        h = captured[key]
        h_last = h[torch.arange(h.size(0)), last_idx.cpu()]
        results[key] = h_last.numpy()

    return results


def compute_target_logit(h, W_U, cat_name, cat_names):
    """计算目标类别的logit值"""
    logit = W_U @ h
    return float(logit[cat_names.index(cat_name)])


# ==================== 方向操作 ====================
def ablate_direction(h, direction):
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - proj


def double_direction(h, direction):
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h + proj


def reverse_direction(h, direction):
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - 2 * proj


# ==================== 子空间分解 ====================
def get_bc_and_orth_dirs(h_target, h_others, n_dirs=8):
    """获取B_c方向和orth子空间分解"""
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)

    contrast_matrix = []
    for h_c in np.vstack([h_target[np.newaxis], h_others]):
        diff = h_c - others_mean
        diff_orth = diff - np.dot(diff, Bc_norm) * Bc_norm
        if np.linalg.norm(diff_orth) > 1e-6:
            contrast_matrix.append(diff_orth)

    if len(contrast_matrix) > 0:
        M = np.stack(contrast_matrix)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        n_return = min(n_dirs, len(Vt))
        orth_subspace = Vt[:n_return]
        singular_values = S[:n_return]
    else:
        orth_subspace = np.eye(len(B_c))[:n_dirs]
        singular_values = np.ones(n_dirs)

    return Bc_norm, orth_subspace, singular_values


def classify_orth_dirs(h_target, orth_subspace, W_U, cat_name, cat_names, threshold=0.05):
    """将orth子空间方向分类为support/inhibit/neutral"""
    baseline_target_D = compute_target_logit(h_target, W_U, cat_name, cat_names)

    support_dirs = []
    inhibit_dirs = []
    neutral_dirs = []

    for i, d in enumerate(orth_subspace):
        h_mod = ablate_direction(h_target, d)
        target_D_after = compute_target_logit(h_mod, W_U, cat_name, cat_names)
        delta = target_D_after - baseline_target_D

        info = {
            "dir_idx": i,
            "ablate_delta": float(delta),
            "direction_norm": float(np.linalg.norm(d)),
        }

        if delta < -threshold:
            support_dirs.append(info)
        elif delta > threshold:
            inhibit_dirs.append(info)
        else:
            neutral_dirs.append(info)

    return support_dirs, inhibit_dirs, neutral_dirs


def get_shared_direction(h_target, h_others):
    """计算shared_semantic方向"""
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    Bc_norm_dir = B_c / (np.linalg.norm(B_c) + 1e-10)

    all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
    shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
    shared_dir = shared_dir - np.dot(shared_dir, Bc_norm_dir) * Bc_norm_dir
    if np.linalg.norm(shared_dir) > 1e-6:
        shared_dir = shared_dir / np.linalg.norm(shared_dir)

    return shared_dir, Bc_norm_dir


# ==================== Exp1: 高维SVD分解 ====================
def exp1_highdim_svd(model, tokenizer, device, model_name, W_U):
    """
    对Qwen3末层做高维SVD分解(k=8,16,32,64)
    验证support/inhibit预测是否随维度增加而改善
    同时对GLM4和DS7B做对比
    """
    plog(f"=== Exp1: High-dimensional SVD Decomposition ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1

    test_cats = CAT_NAMES_8  # 8类
    other_cats_pool = test_cats.copy()
    svd_dims = [8, 16, 32, 64]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        li = L_n1
        t0 = time.time()

        cat_objs = CATEGORIES_16[cat_name][:12]
        h_target_all = get_category_hidden(model, tokenizer, device, cat_name, cat_objs, li)
        h_target = np.mean(h_target_all, axis=0)

        other_cats = [c for c in other_cats_pool if c != cat_name][:7]
        other_hiddens = {}
        for oc in other_cats:
            other_hiddens[oc] = get_category_hidden(
                model, tokenizer, device, oc, CATEGORIES_16[oc][:8], li
            )
        h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

        # ablate_shared (ground truth)
        shared_dir, Bc_norm = get_shared_direction(h_target, h_others)
        baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES_8)
        h_ablate = ablate_direction(h_target, shared_dir)
        ablate_shared_delta = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES_8) - baseline_D

        # 不同维度SVD分解
        dim_results = {}
        for k in svd_dims:
            Bc_norm_k, orth_subspace, singular_values = get_bc_and_orth_dirs(
                h_target, h_others, n_dirs=k
            )

            support_dirs, inhibit_dirs, neutral_dirs = classify_orth_dirs(
                h_target, orth_subspace, W_U, cat_name, CAT_NAMES_8, threshold=0.05
            )

            support_sum = sum(d["ablate_delta"] for d in support_dirs)
            support_count = len(support_dirs)
            inhibit_sum = sum(d["ablate_delta"] for d in inhibit_dirs)
            inhibit_count = len(inhibit_dirs)
            neutral_count = len(neutral_dirs)

            net_release = abs(support_sum) - abs(inhibit_sum)

            predicted_reversal = net_release > 0
            actual_reversal = ablate_shared_delta < 0
            correct = predicted_reversal == actual_reversal

            # 累计效应: 按ablate_delta绝对值排序, 逐步累加
            all_dirs_sorted = sorted(
                support_dirs + inhibit_dirs + neutral_dirs,
                key=lambda d: abs(d["ablate_delta"]), reverse=True
            )
            cumulative_effects = []
            cumsum = 0.0
            for d in all_dirs_sorted[:min(16, len(all_dirs_sorted))]:
                cumsum += d["ablate_delta"]
                cumulative_effects.append({
                    "dir_idx": d["dir_idx"],
                    "ablate_delta": d["ablate_delta"],
                    "cumulative": float(cumsum),
                })

            dim_results[f"k{k}"] = {
                "n_support": support_count,
                "n_inhibit": inhibit_count,
                "n_neutral": neutral_count,
                "support_sum": float(support_sum),
                "inhibit_sum": float(inhibit_sum),
                "net_release": float(net_release),
                "predicted_reversal": predicted_reversal,
                "actual_reversal": actual_reversal,
                "correct": correct,
                "top5_cumulative": cumulative_effects[:5],
            }

            plog(f"    k={k}: n_s={support_count}, n_i={inhibit_count}, "
                 f"net_release={net_release:+.3f}, "
                 f"pred={predicted_reversal}, actual={actual_reversal}, "
                 f"correct={correct}")

        cat_results[f"L{li}"] = dim_results
        cat_results["ablate_shared_delta"] = float(ablate_shared_delta)
        cat_results["baseline_D"] = float(baseline_D)
        cat_results["elapsed"] = time.time() - t0

        results[cat_name] = cat_results

    # 统计预测准确率
    for k in svd_dims:
        correct_count = sum(
            1 for c in results.values()
            if c.get(f"L{n_layers-1}", {}).get(f"k{k}", {}).get("correct", False)
        )
        plog(f"  k={k} prediction accuracy: {correct_count}/{len(results)}")

    return results


# ==================== Exp2: 跨层刹车-释放传递 ====================
def exp2_cross_layer_transfer(model, tokenizer, device, model_name, W_U):
    """
    在L(n-2)操作shared_semantic后, 追踪L(n-1)残差的变化:
    - L(n-2) ablate_shared → L(n-1) support/inhibit composition如何变化
    - L(n-2) double_shared → L(n-1) support/inhibit composition如何变化
    - L(n-2) matched_norm inject → L(n-1)如何响应

    这是真正的跨层因果追踪。
    """
    plog(f"=== Exp2: Cross-layer Brake-Release Transfer ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    # 只测4个关键类别, 控制时间
    test_cats = ["fruit", "clothing", "animal", "food"]
    other_cats_pool = CAT_NAMES_8.copy()

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        # 步骤1: 获取自然状态下L(n-2)和L(n-1)的hidden states
        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in other_cats_pool if c != cat_name][:7]

        # 自然状态: 同时获取两层
        h_two = get_category_hidden_two_layers(
            model, tokenizer, device, cat_name, cat_objs, L_n2, L_n1
        )
        h_target_n2 = np.mean(h_two[f"L{L_n2}"], axis=0)
        h_target_n1 = np.mean(h_two[f"L{L_n1}"], axis=0)

        # 获取其他类别在两层的hidden
        other_h_n2 = {}
        other_h_n1 = {}
        for oc in other_cats:
            h_two_oc = get_category_hidden_two_layers(
                model, tokenizer, device, oc, CATEGORIES_16[oc][:8], L_n2, L_n1
            )
            other_h_n2[oc] = h_two_oc[f"L{L_n2}"]
            other_h_n1[oc] = h_two_oc[f"L{L_n1}"]

        h_others_n2 = np.array([np.mean(other_h_n2[oc], axis=0) for oc in other_cats])
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])

        # 步骤2: 计算L(n-2)的shared方向
        shared_dir_n2, Bc_norm_n2 = get_shared_direction(h_target_n2, h_others_n2)

        # 步骤3: 计算L(n-1)的shared方向和support/inhibit分解
        shared_dir_n1, Bc_norm_n1 = get_shared_direction(h_target_n1, h_others_n1)

        # L(n-1)自然状态
        Bc_n1, orth_n1, sv_n1 = get_bc_and_orth_dirs(h_target_n1, h_others_n1, n_dirs=16)
        support_n1, inhibit_n1, neutral_n1 = classify_orth_dirs(
            h_target_n1, orth_n1, W_U, cat_name, CAT_NAMES_8, threshold=0.05
        )

        baseline_D_n1 = compute_target_logit(h_target_n1, W_U, cat_name, CAT_NAMES_8)
        h_ablate_n1 = ablate_direction(h_target_n1, shared_dir_n1)
        ablate_shared_n1_delta = compute_target_logit(h_ablate_n1, W_U, cat_name, CAT_NAMES_8) - baseline_D_n1

        # 步骤4: 模拟L(n-2)操作对L(n-1)的影响
        # 方法: 修改L(n-2)的hidden → 估算L(n-1)的变化
        # 因为transformer是逐层计算的, 我们用线性近似:
        # Δh_{n-1} ≈ W_attn * Δh_{n-2} + W_mlp * Δh_{n-2}
        # 但直接计算这个太复杂, 我们改用间接方法:
        # 对L(n-2)做操作后, 观察L(n-1)中shared方向投影的变化

        # L(n-2)操作: ablate, double, reverse, matched_norm_inject
        interventions = {
            "ablate": ablate_direction(h_target_n2, shared_dir_n2),
            "double": double_direction(h_target_n2, shared_dir_n2),
            "reverse": reverse_direction(h_target_n2, shared_dir_n2),
        }

        # matched_norm inject: 注入shared方向, 匹配自然投影范数
        proj_shared_n2 = np.dot(h_target_n2, shared_dir_n2)
        matched_scale = abs(proj_shared_n2)
        h_inject = h_target_n2 + matched_scale * shared_dir_n2
        interventions["matched_inject"] = h_inject

        # 步骤5: 对每个干预, 计算L(n-2)修改后的shared方向投影变化
        # 以及估算L(n-1)中shared方向和support/inhibit成分的变化
        proj_shared_n2_baseline = np.dot(h_target_n2, shared_dir_n2)
        proj_shared_n1_baseline = np.dot(h_target_n1, shared_dir_n1)

        intervention_results = {}
        for intv_name, h_modified_n2 in interventions.items():
            # L(n-2)的shared投影变化
            proj_shared_n2_after = np.dot(h_modified_n2, shared_dir_n2)
            delta_proj_n2 = proj_shared_n2_after - proj_shared_n2_baseline

            # L(n-2)的DCF变化
            baseline_D_n2 = compute_target_logit(h_target_n2, W_U, cat_name, CAT_NAMES_8)
            delta_D_n2 = compute_target_logit(h_modified_n2, W_U, cat_name, CAT_NAMES_8) - baseline_D_n2

            # 估算: L(n-2)的shared分量变化如何传递到L(n-1)
            # 使用线性近似: Δh_{n-1} ≈ transfer_matrix * Δh_{n-2}
            # 但我们没有transfer_matrix, 所以用相关性推断:
            # 观察L(n-2) shared投影变化和L(n-1) shared投影的关系

            # 更直接的方法: 计算L(n-2)操作后, L(n-1)中的orth子空间成分如何变化
            # 我们不能直接修改L(n-1), 但可以分析:
            # L(n-2)的shared投影变化量 → 对L(n-1)的各成分的间接影响

            # 关键度量: L(n-2) shared消融对L(n-1) shared消融的比例
            # 即: 如果L(n-2)完全消融shared, L(n-1)的shared会变多少?

            # 我们用样本级分析:
            # 对每个样本, L(n-2)和L(n-1)的shared投影的相关性
            proj_shared_n2_all = np.array([np.dot(h, shared_dir_n2) for h in h_two[f"L{L_n2}"]])
            proj_shared_n1_all = np.array([np.dot(h, shared_dir_n1) for h in h_two[f"L{L_n1}"]])

            # Pearson相关系数
            if np.std(proj_shared_n2_all) > 1e-6 and np.std(proj_shared_n1_all) > 1e-6:
                corr_n2_n1 = float(np.corrcoef(proj_shared_n2_all, proj_shared_n1_all)[0, 1])
            else:
                corr_n2_n1 = 0.0

            # 回归系数: Δproj_n1 ≈ slope * Δproj_n2
            if np.std(proj_shared_n2_all) > 1e-6:
                slope = float(np.cov(proj_shared_n1_all, proj_shared_n2_all)[0, 1] /
                             np.var(proj_shared_n2_all))
            else:
                slope = 0.0

            # 估算L(n-1)的shared投影变化
            estimated_delta_proj_n1 = slope * delta_proj_n2

            intervention_results[intv_name] = {
                "delta_proj_shared_n2": float(delta_proj_n2),
                "delta_D_n2": float(delta_D_n2),
                "corr_n2_to_n1_shared": float(corr_n2_n1),
                "regression_slope_n2_to_n1": float(slope),
                "estimated_delta_proj_n1": float(estimated_delta_proj_n1),
            }

            plog(f"    {intv_name}: Δproj_n2={delta_proj_n2:+.2f}, "
                 f"ΔD_n2={delta_D_n2:+.2f}, "
                 f"corr={corr_n2_n1:.3f}, slope={slope:.3f}")

        # 汇总: L(n-2)和L(n-1)的shared投影关系
        cat_results = {
            f"L{L_n2}_shared_proj": float(proj_shared_n2_baseline),
            f"L{L_n1}_shared_proj": float(proj_shared_n1_baseline),
            "ablate_shared_n1_delta": float(ablate_shared_n1_delta),
            "n1_support_count": len(support_n1),
            "n1_inhibit_count": len(inhibit_n1),
            "n1_support_sum": float(sum(d["ablate_delta"] for d in support_n1)),
            "n1_inhibit_sum": float(sum(d["ablate_delta"] for d in inhibit_n1)),
            "interventions": intervention_results,
            "baseline_D_n1": float(baseline_D_n1),
        }

        results[cat_name] = cat_results

    # 汇总跨层传递效率
    plog(f"\n  === Cross-layer Transfer Summary ===")
    for cat_name, data in results.items():
        intv = data["interventions"]
        ablate = intv["ablate"]
        plog(f"  {cat_name}: ablate Δproj_n2={ablate['delta_proj_shared_n2']:+.1f}, "
             f"corr={ablate['corr_n2_to_n1_shared']:.3f}, "
             f"slope={ablate['regression_slope_n2_to_n1']:.3f}")

    return results


# ==================== Exp3: 16类全局验证 ====================
def exp3_16cat_global_verification(model, tokenizer, device, model_name, W_U):
    """
    16类全局验证末端刹车-释放机制:
    - 每类的 L(n-2) ablate_shared
    - 每类的 L(n-1) ablate_shared
    - norm_delta, z_delta
    - 是否末层反转
    """
    plog(f"=== Exp3: 16-category Global Verification ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    results = {}
    reversal_count = 0

    for cat_name in CAT_NAMES_16:
        plog(f"  Cat: {cat_name} ({CAT_NAMES_16.index(cat_name)+1}/16)")
        t0 = time.time()

        cat_objs = CATEGORIES_16[cat_name][:12]
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # 只测末两层, 节省时间
        for li in [L_n2, L_n1]:
            h_target_all = get_category_hidden(
                model, tokenizer, device, cat_name, cat_objs, li
            )
            h_target = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, oc, CATEGORIES_16[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            # shared方向
            shared_dir, Bc_norm = get_shared_direction(h_target, h_others)

            # 尺度
            residual_norm = float(np.linalg.norm(h_target))
            proj_shared = float(abs(np.dot(h_target, shared_dir)))

            # ablate_shared
            baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES_16)
            h_ablate = ablate_direction(h_target, shared_dir)
            ablate_shared_delta = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES_16) - baseline_D

            # 归一化
            all_logits = W_U @ h_target
            dcf_std = float(np.std(all_logits))
            norm_delta = ablate_shared_delta / (residual_norm + 1e-10)
            z_delta = ablate_shared_delta / (dcf_std + 1e-10)

            if f"L{li}" not in results.get(cat_name, {}):
                if cat_name not in results:
                    results[cat_name] = {}

            results[cat_name][f"L{li}"] = {
                "ablate_shared_delta": float(ablate_shared_delta),
                "norm_delta": float(norm_delta),
                "z_delta": float(z_delta),
                "residual_norm": float(residual_norm),
                "proj_shared": float(proj_shared),
                "dcf_std": float(dcf_std),
                "baseline_D": float(baseline_D),
            }

        # 判断末层是否反转
        ln2_delta = results[cat_name][f"L{L_n2}"]["ablate_shared_delta"]
        ln1_delta = results[cat_name][f"L{L_n1}"]["ablate_shared_delta"]
        is_reversal = (ln2_delta > 0 and ln1_delta < 0)  # L(n-2)正(刹车), L(n-1)负(释放)
        results[cat_name]["is_reversal"] = is_reversal
        if is_reversal:
            reversal_count += 1

        elapsed = time.time() - t0
        results[cat_name]["elapsed"] = elapsed

        plog(f"    L{n_layers-2}={ln2_delta:+.2f}, L{n_layers-1}={ln1_delta:+.2f}, "
             f"reversal={is_reversal}, z_Ln1={results[cat_name][f'L{L_n1}']['z_delta']:+.2f}")

    # 汇总
    plog(f"\n  === 16-category Summary ===")
    plog(f"  Reversal: {reversal_count}/16")

    # 统计: 前8类 vs 后8类的反转率
    first8_reversal = sum(1 for c in CAT_NAMES_8 if results[c]["is_reversal"])
    last8_cats = [c for c in CAT_NAMES_16 if c not in CAT_NAMES_8]
    last8_reversal = sum(1 for c in last8_cats if results[c]["is_reversal"])
    plog(f"  First 8: {first8_reversal}/8 reversal")
    plog(f"  Last 8: {last8_reversal}/8 reversal")

    # 末层z_delta统计
    z_deltas = [results[c][f"L{L_n1}"]["z_delta"] for c in CAT_NAMES_16]
    plog(f"  L(n-1) z_delta: mean={np.mean(z_deltas):.2f}, "
         f"std={np.std(z_deltas):.2f}, "
         f"min={np.min(z_deltas):.2f}, max={np.max(z_deltas):.2f}")

    results["_summary"] = {
        "reversal_count": reversal_count,
        "total_categories": 16,
        "first8_reversal": first8_reversal,
        "last8_reversal": last8_reversal,
        "z_delta_mean": float(np.mean(z_deltas)),
        "z_delta_std": float(np.std(z_deltas)),
    }

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 493: {model_name}, round={round_num}")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # 获取W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 493,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        }
    }

    # Exp1: 高维SVD分解
    plog(f"\n{'='*60}")
    all_results["exp1_highdim_svd"] = exp1_highdim_svd(
        model, tokenizer, device, model_name, W_U
    )

    # Exp2: 跨层因果追踪
    plog(f"\n{'='*60}")
    all_results["exp2_cross_layer_transfer"] = exp2_cross_layer_transfer(
        model, tokenizer, device, model_name, W_U
    )

    # Exp3: 16类全局验证
    plog(f"\n{'='*60}")
    all_results["exp3_16cat_verification"] = exp3_16cat_global_verification(
        model, tokenizer, device, model_name, W_U
    )

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase493_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    plog(f"Phase 493 {model_name} complete!")


if __name__ == "__main__":
    main()
