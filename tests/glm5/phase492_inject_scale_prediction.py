"""
Phase 492: 末端刹车-释放机制的注入修复、尺度校准与预测验证
===========================================================

Phase 491核心发现:
1. ablate/double/reverse完美闭环, 但inject失效(比ablate小2-3量级)
2. support/inhibit ratio决定末层是否反转(DS7B强反转,Qwen3温和反转,GLM4无反转)
3. 关系槽位只调幅度不调方向
4. ablate_shared ≈ ablate_orth_bc (shared方向主导orth子空间)

Phase 492核心目标:
- Exp1: 解决inject失效 — 4种注入方法对比
- Exp2: 尺度校准 — residual norm, normalized delta, lm_head gain
- Exp3: support/inhibit ratio预测8类末层反转
- Exp4: 竞争类别末层控制闭环

用法:
  python tests/glm5/phase492_inject_scale_prediction.py qwen3 1
  python tests/glm5/phase492_inject_scale_prediction.py glm4 1
  python tests/glm5/phase492_inject_scale_prediction.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS, safe_decode)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
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

CAT_NAMES = list(CATEGORIES.keys())


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


def get_category_hidden(model, tokenizer, device, cat_name, objects, layer_idx, template="The {obj} is a kind of"):
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


def compute_dcf(h_center, W_U, cat_name, cat_names):
    """计算DCF: 目标类别logit - 其他类别平均logit"""
    logit = W_U @ h_center
    target_D = logit[cat_names.index(cat_name)]
    return float(target_D)


def compute_target_logit(h, W_U, cat_name, cat_names):
    """计算目标类别的logit值"""
    logit = W_U @ h
    return float(logit[cat_names.index(cat_name)])


# ==================== 方向操作 ====================
def ablate_direction(h, direction):
    """消融h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - proj


def inject_direction(h, direction, scale=1.0):
    """注入direction到h"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    return h + scale * d_norm


def double_direction(h, direction):
    """加倍h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h + proj


def reverse_direction(h, direction):
    """反转h在direction方向的投影"""
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
        orth_subspace = Vt[:n_dirs]
        singular_values = S[:n_dirs]
    else:
        orth_subspace = np.eye(len(B_c))[:n_dirs]
        singular_values = np.ones(n_dirs)

    return Bc_norm, orth_subspace, singular_values


def classify_orth_dirs(h_target, orth_subspace, W_U, cat_name, threshold=0.05):
    """将orth子空间方向分类为support/inhibit/neutral"""
    baseline_target_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES)

    support_dirs = []
    inhibit_dirs = []
    neutral_dirs = []

    for i, d in enumerate(orth_subspace):
        h_mod = ablate_direction(h_target, d)
        target_D_after = compute_target_logit(h_mod, W_U, cat_name, CAT_NAMES)
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


# ==================== Exp1: 解决inject失效 ====================
def exp1_inject_methods(model, tokenizer, device, model_name, W_U):
    """
    比较4种注入方法:
    1. direction_inject: 单位向量注入 (原始方法)
    2. matched_norm_inject: 匹配原始投影范数的注入
    3. sample_wise_inject: 逐样本注入各自的shared分量
    4. component_replacement: 用另一类的shared分量替换
    """
    plog(f"=== Exp1: Inject Method Comparison ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    # 只测fruit类别, 减少时间
    test_cats = ["fruit", "clothing"]
    other_cats = ["food", "vehicle", "tool", "plant", "furniture", "animal"]

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        for li in [L_n2, L_n1]:
            plog(f"    Layer {li}...")
            t0 = time.time()

            # 获取所有类别的hidden states
            cat_objs = CATEGORIES[cat_name][:12]
            h_target_all = get_category_hidden(model, tokenizer, device, cat_name, cat_objs, li)
            h_target = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, oc, CATEGORIES[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            # 计算shared_semantic方向
            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)

            all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
            shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
            shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
            if np.linalg.norm(shared_dir) > 1e-6:
                shared_dir = shared_dir / np.linalg.norm(shared_dir)

            # 关键度量: shared方向的投影范数
            proj_shared_on_target = np.dot(h_target, shared_dir)
            proj_shared_norm = abs(proj_shared_on_target)

            baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES)

            # 方法1: 原始direction inject (单位向量, scale=1.0)
            h_mod1 = inject_direction(h_target, shared_dir, scale=1.0)
            delta1 = compute_target_logit(h_mod1, W_U, cat_name, CAT_NAMES) - baseline_D

            # 方法2: matched-norm inject (匹配原始投影范数)
            # 注入量 = 原始投影量, 效果应该 ≈ double_shared - baseline
            matched_scale = proj_shared_norm
            h_mod2 = inject_direction(h_target, shared_dir, scale=matched_scale)
            delta2 = compute_target_logit(h_mod2, W_U, cat_name, CAT_NAMES) - baseline_D

            # 方法3: sample-wise inject (每个样本注入自己的shared分量)
            # 用h_target_all中每个样本的shared投影
            sample_deltas = []
            for h_s in h_target_all:
                proj_s = np.dot(h_s, shared_dir)
                h_mod_s = h_s + proj_s * shared_dir  # 加倍shared分量
                d = compute_target_logit(h_mod_s, W_U, cat_name, CAT_NAMES) - compute_target_logit(h_s, W_U, cat_name, CAT_NAMES)
                sample_deltas.append(d)
            delta3 = float(np.mean(sample_deltas))

            # 方法4: component replacement — 用另一类的shared分量替换
            # 取food的shared分量替换fruit的shared分量
            replace_cat = "food" if cat_name != "food" else "vehicle"
            h_replace_all = get_category_hidden(
                model, tokenizer, device, replace_cat, CATEGORIES[replace_cat][:8], li
            )
            h_replace = np.mean(h_replace_all, axis=0)
            # 替换: 保留B_c方向, 把shared分量换成另一类的
            proj_bc = np.dot(h_target, Bc_norm) * Bc_norm
            # 另一类的shared分量
            all_mean_replace = np.mean(np.vstack([h_replace[np.newaxis], h_others]), axis=0)
            shared_dir_replace = all_mean_replace / (np.linalg.norm(all_mean_replace) + 1e-10)
            shared_dir_replace = shared_dir_replace - np.dot(shared_dir_replace, Bc_norm) * Bc_norm
            if np.linalg.norm(shared_dir_replace) > 1e-6:
                shared_dir_replace = shared_dir_replace / np.linalg.norm(shared_dir_replace)
            proj_shared_replace = np.dot(h_replace, shared_dir_replace)
            h_mod4 = proj_bc + proj_shared_replace * shared_dir_replace
            delta4 = compute_target_logit(h_mod4, W_U, cat_name, CAT_NAMES) - baseline_D

            # 对照: ablate_shared 和 double_shared
            h_ablate = ablate_direction(h_target, shared_dir)
            delta_ablate = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES) - baseline_D

            h_double = double_direction(h_target, shared_dir)
            delta_double = compute_target_logit(h_double, W_U, cat_name, CAT_NAMES) - baseline_D

            # 额外: 大尺度inject (scale=5, 10)
            h_mod_s5 = inject_direction(h_target, shared_dir, scale=5.0)
            delta_s5 = compute_target_logit(h_mod_s5, W_U, cat_name, CAT_NAMES) - baseline_D

            h_mod_s10 = inject_direction(h_target, shared_dir, scale=10.0)
            delta_s10 = compute_target_logit(h_mod_s10, W_U, cat_name, CAT_NAMES) - baseline_D

            elapsed = time.time() - t0

            cat_results[f"L{li}"] = {
                "direction_inject_s1": float(delta1),
                "matched_norm_inject": float(delta2),
                "sample_wise_inject": float(delta3),
                "component_replacement": float(delta4),
                "scaled_inject_s5": float(delta_s5),
                "scaled_inject_s10": float(delta_s10),
                "ablate_shared": float(delta_ablate),
                "double_shared": float(delta_double),
                "proj_shared_norm": float(proj_shared_norm),
                "baseline_D": float(baseline_D),
                "elapsed": elapsed,
            }

            plog(f"      ablate={delta_ablate:+.3f}, double={delta_double:+.3f}, "
                 f"dir_inject={delta1:+.4f}, matched={delta2:+.3f}, "
                 f"sample_wise={delta3:+.3f}, replace={delta4:+.3f}, "
                 f"s5={delta_s5:+.3f}, s10={delta_s10:+.3f}")

        results[cat_name] = cat_results

    return results


# ==================== Exp2: 尺度校准 ====================
def exp2_scale_calibration(model, tokenizer, device, model_name, W_U):
    """
    尺度校准: 计算每层的residual norm, shared_component_norm, normalized delta等
    判断DS7B的极端数值是真实强控制还是读出放大
    """
    plog(f"=== Exp2: Scale Calibration ===")
    n_layers = get_model_info(model, model_name).n_layers
    info = get_model_info(model, model_name)

    test_cats = ["fruit", "clothing", "food"]
    other_cats = ["food", "vehicle", "tool", "plant", "furniture", "animal"]

    # 关键层位
    test_layers = list(range(0, n_layers, max(1, n_layers // 6))) + list(range(max(0, n_layers - 3), n_layers))
    test_layers = sorted(set(test_layers))

    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        for li in test_layers:
            t0 = time.time()

            cat_objs = CATEGORIES[cat_name][:12]
            h_target_all = get_category_hidden(model, tokenizer, device, cat_name, cat_objs, li)
            h_target = np.mean(h_target_all, axis=0)

            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, oc, CATEGORIES[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            # 尺度信息
            residual_norm = float(np.linalg.norm(h_target))
            residual_std = float(np.std(h_target_all))

            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            bc_norm = float(np.linalg.norm(B_c))

            Bc_norm_dir = B_c / (np.linalg.norm(B_c) + 1e-10)
            all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
            shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
            shared_dir = shared_dir - np.dot(shared_dir, Bc_norm_dir) * Bc_norm_dir
            shared_dir_norm = float(np.linalg.norm(shared_dir))

            # shared分量在目标中的投影范数
            proj_shared = float(abs(np.dot(h_target, shared_dir) / (np.linalg.norm(shared_dir) + 1e-10)))

            # ablate_shared的delta
            baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES)
            h_ablate = ablate_direction(h_target, shared_dir)
            ablate_delta = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES) - baseline_D

            # 归一化delta: delta / residual_norm
            normalized_delta = ablate_delta / (residual_norm + 1e-10)

            # lm_head增益: W_U的norm (粗略估计读出放大)
            w_u_norm = float(np.linalg.norm(W_U))

            # 标准分delta: delta / (DCF_std across categories)
            all_logits = W_U @ h_target
            dcf_std = float(np.std(all_logits))
            z_score_delta = ablate_delta / (dcf_std + 1e-10)

            elapsed = time.time() - t0

            cat_results[f"L{li}"] = {
                "residual_norm": residual_norm,
                "residual_std": residual_std,
                "bc_norm": bc_norm,
                "shared_dir_norm": shared_dir_norm,
                "proj_shared_abs": proj_shared,
                "ablate_shared_delta": float(ablate_delta),
                "normalized_delta": normalized_delta,
                "z_score_delta": z_score_delta,
                "dcf_std": dcf_std,
                "w_u_norm": w_u_norm,
                "baseline_D": float(baseline_D),
                "elapsed": elapsed,
            }

            if li >= n_layers - 3:
                plog(f"    L{li}: norm={residual_norm:.1f}, proj_shared={proj_shared:.3f}, "
                     f"ablate={ablate_delta:+.2f}, norm_delta={normalized_delta:+.4f}, "
                     f"z_delta={z_score_delta:+.4f}")

        results[cat_name] = cat_results

    return results


# ==================== Exp3: support/inhibit ratio预测 ====================
def exp3_ratio_prediction(model, tokenizer, device, model_name, W_U):
    """
    对8个类别计算末层support/inhibit分解, 验证是否能预测ablate_shared的符号
    """
    plog(f"=== Exp3: Support/Inhibit Ratio Prediction ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n2 = n_layers - 2
    L_n1 = n_layers - 1

    other_cats_pool = CAT_NAMES.copy()

    results = {}
    for cat_name in CAT_NAMES:  # 全部8类
        plog(f"  Cat: {cat_name}")
        cat_results = {}

        for li in [L_n2, L_n1]:
            t0 = time.time()

            # 获取目标类别和对比类别的hidden
            cat_objs = CATEGORIES[cat_name][:12]
            h_target_all = get_category_hidden(model, tokenizer, device, cat_name, cat_objs, li)
            h_target = np.mean(h_target_all, axis=0)

            other_cats = [c for c in other_cats_pool if c != cat_name][:7]
            other_hiddens = {}
            for oc in other_cats:
                other_hiddens[oc] = get_category_hidden(
                    model, tokenizer, device, oc, CATEGORIES[oc][:8], li
                )
            h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

            # 子空间分解
            Bc_norm, orth_subspace, singular_values = get_bc_and_orth_dirs(h_target, h_others, n_dirs=8)

            # 分类support/inhibit
            support_dirs, inhibit_dirs, neutral_dirs = classify_orth_dirs(
                h_target, orth_subspace, W_U, cat_name, threshold=0.05
            )

            # 计算support/inhibit的总效应
            support_sum = sum(d["ablate_delta"] for d in support_dirs)
            support_count = len(support_dirs)
            support_mean = support_sum / support_count if support_count > 0 else 0
            support_max = max((abs(d["ablate_delta"]) for d in support_dirs), default=0)

            inhibit_sum = sum(d["ablate_delta"] for d in inhibit_dirs)
            inhibit_count = len(inhibit_dirs)
            inhibit_mean = inhibit_sum / inhibit_count if inhibit_count > 0 else 0
            inhibit_max = max((abs(d["ablate_delta"]) for d in inhibit_dirs), default=0)

            # net_release = support总效应 - inhibit总效应(注意符号: support消融为负, inhibit消融为正)
            net_release = abs(support_sum) - abs(inhibit_sum)

            # ablate_shared
            others_mean = np.mean(h_others, axis=0)
            all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
            shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
            shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
            if np.linalg.norm(shared_dir) > 1e-6:
                shared_dir = shared_dir / np.linalg.norm(shared_dir)

            baseline_D = compute_target_logit(h_target, W_U, cat_name, CAT_NAMES)
            h_ablate = ablate_direction(h_target, shared_dir)
            ablate_shared_delta = compute_target_logit(h_ablate, W_U, cat_name, CAT_NAMES) - baseline_D

            elapsed = time.time() - t0

            cat_results[f"L{li}"] = {
                "n_support": support_count,
                "n_inhibit": inhibit_count,
                "n_neutral": len(neutral_dirs),
                "support_sum": float(support_sum),
                "support_mean": float(support_mean),
                "support_max": float(support_max),
                "inhibit_sum": float(inhibit_sum),
                "inhibit_mean": float(inhibit_mean),
                "inhibit_max": float(inhibit_max),
                "net_release": float(net_release),
                "ablate_shared_delta": float(ablate_shared_delta),
                "baseline_D": float(baseline_D),
                "elapsed": elapsed,
            }

            plog(f"    L{li}: n_s={support_count}, n_i={inhibit_count}, "
                 f"s_sum={support_sum:+.3f}, i_sum={inhibit_sum:+.3f}, "
                 f"net_release={net_release:+.3f}, "
                 f"ablate_shared={ablate_shared_delta:+.3f}")

        # 预测正确性
        L_n1_data = cat_results[f"L{n_layers-1}"]
        predicted_reversal = L_n1_data["net_release"] > 0
        actual_reversal = L_n1_data["ablate_shared_delta"] < 0
        cat_results["prediction"] = {
            "predicted_reversal": predicted_reversal,
            "actual_reversal": actual_reversal,
            "correct": predicted_reversal == actual_reversal,
        }
        plog(f"  Prediction: predicted={predicted_reversal}, actual={actual_reversal}, "
             f"correct={predicted_reversal == actual_reversal}")

        results[cat_name] = cat_results

    # 统计预测准确率
    correct_count = sum(1 for c in results.values() if c["prediction"]["correct"])
    plog(f"  Prediction accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.1%}")

    return results


# ==================== Exp4: 竞争类别末层控制 ====================
def exp4_competition_control(model, tokenizer, device, model_name, W_U):
    """
    验证竞争类别方向在末层的因果效应
    对竞争类别方向做 ablate/double/reverse
    """
    plog(f"=== Exp4: Competition Control at Last Layer ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1

    # 测试fruit的竞争类别
    test_configs = [
        ("fruit", "food"),   # fruit和food竞争
        ("clothing", "tool"), # clothing和tool竞争
    ]

    results = {}
    for target_cat, comp_cat in test_configs:
        plog(f"  Target: {target_cat}, Competitor: {comp_cat}")
        config_results = {}

        li = L_n1
        t0 = time.time()

        # 获取各类hidden
        h_target_all = get_category_hidden(
            model, tokenizer, device, target_cat, CATEGORIES[target_cat][:12], li
        )
        h_target = np.mean(h_target_all, axis=0)

        h_comp_all = get_category_hidden(
            model, tokenizer, device, comp_cat, CATEGORIES[comp_cat][:8], li
        )
        h_comp = np.mean(h_comp_all, axis=0)

        # 其他类别
        other_cats = [c for c in CAT_NAMES if c not in [target_cat, comp_cat]][:6]
        other_hiddens = {}
        for oc in other_cats:
            other_hiddens[oc] = get_category_hidden(
                model, tokenizer, device, oc, CATEGORIES[oc][:8], li
            )
        h_others = np.array([np.mean(other_hiddens[oc], axis=0) for oc in other_cats])

        # 子空间分解
        all_other_cats = [comp_cat] + other_cats
        h_all_others = np.vstack([h_comp[np.newaxis], h_others])
        Bc_norm, orth_subspace, singular_values = get_bc_and_orth_dirs(h_target, h_all_others, n_dirs=8)

        # 竞争方向: comp_center - other_centers (不含target)
        others_mean_excl_comp = np.mean(h_others, axis=0)
        comp_direction = h_comp - others_mean_excl_comp
        comp_dir_norm = float(np.linalg.norm(comp_direction))

        baseline_D = compute_target_logit(h_target, W_U, target_cat, CAT_NAMES)

        # 操作1: ablate竞争方向
        h_mod = ablate_direction(h_target, comp_direction)
        delta_ablate_comp = compute_target_logit(h_mod, W_U, target_cat, CAT_NAMES) - baseline_D

        # 操作2: double竞争方向
        h_mod = double_direction(h_target, comp_direction)
        delta_double_comp = compute_target_logit(h_mod, W_U, target_cat, CAT_NAMES) - baseline_D

        # 操作3: reverse竞争方向
        h_mod = reverse_direction(h_target, comp_direction)
        delta_reverse_comp = compute_target_logit(h_mod, W_U, target_cat, CAT_NAMES) - baseline_D

        # 对照: ablate shared
        all_mean = np.mean(np.vstack([h_target[np.newaxis], h_all_others]), axis=0)
        shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
        shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
        if np.linalg.norm(shared_dir) > 1e-6:
            shared_dir = shared_dir / np.linalg.norm(shared_dir)

        h_mod = ablate_direction(h_target, shared_dir)
        delta_ablate_shared = compute_target_logit(h_mod, W_U, target_cat, CAT_NAMES) - baseline_D

        # 对comp类别的target logit影响
        comp_baseline = compute_target_logit(h_comp, W_U, comp_cat, CAT_NAMES)

        elapsed = time.time() - t0

        config_results[f"L{li}"] = {
            "ablate_comp_direction": float(delta_ablate_comp),
            "double_comp_direction": float(delta_double_comp),
            "reverse_comp_direction": float(delta_reverse_comp),
            "ablate_shared": float(delta_ablate_shared),
            "comp_direction_norm": comp_dir_norm,
            "baseline_D": float(baseline_D),
            "comp_baseline_D": float(comp_baseline),
            "elapsed": elapsed,
        }

        plog(f"    L{li}: ablate_comp={delta_ablate_comp:+.3f}, double_comp={delta_double_comp:+.3f}, "
             f"reverse_comp={delta_reverse_comp:+.3f}, ablate_shared={delta_ablate_shared:+.3f}")

        results[f"{target_cat}_vs_{comp_cat}"] = config_results

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 492: {model_name}, round={round_num}")

    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # 获取W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 492,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        }
    }

    # Exp1: inject方法对比
    plog(f"\n{'='*60}")
    all_results["exp1_inject_methods"] = exp1_inject_methods(
        model, tokenizer, device, model_name, W_U
    )

    # Exp2: 尺度校准
    plog(f"\n{'='*60}")
    all_results["exp2_scale_calibration"] = exp2_scale_calibration(
        model, tokenizer, device, model_name, W_U
    )

    # Exp3: support/inhibit ratio预测
    plog(f"\n{'='*60}")
    all_results["exp3_ratio_prediction"] = exp3_ratio_prediction(
        model, tokenizer, device, model_name, W_U
    )

    # Exp4: 竞争类别末层控制
    plog(f"\n{'='*60}")
    all_results["exp4_competition_control"] = exp4_competition_control(
        model, tokenizer, device, model_name, W_U
    )

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase492_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    plog(f"Phase 492 {model_name} complete!")


if __name__ == "__main__":
    main()
