"""
Phase 100: 层间动力学分析 — 从token概率到状态空间变换
=====================================================

批判核心升级:
  之前的分析对象: token概率、logit分布 → 只是decoder投影的"影子"
  Phase 100分析对象: hidden state层间变换 → 真正的计算过程

核心理论框架:
  1. 语义对象(苹果/水果) → 稳定吸引子 → 研究hidden state几何
  2. 计算功能(翻译/CoT) → 子空间映射 → 研究层间变换h_{l+1}-h_l
  3. 两者可能共享底层机制 → 高维概率流形中的不同结构

实验设计:
  Exp1: 层间变换分析 — 每层到底做了什么变换?
    - 计算 ||h_{l+1} - h_l|| (残差变换幅度)
    - 分析变换方向: 主成分方向是否一致?
    - 对比翻译 vs 补全 vs 纯中文 的变换轨迹

  Exp2: 语义对象几何学 — 概念流形分析
    - 收集大量名词/动词/形容词的hidden state
    - 分析不同类别是否形成子空间
    - 验证是否存在"动物流形""食物流形"等

  Exp3: 翻译轨迹分析 — 从中文子空间到英文子空间的变换
    - 追踪翻译过程中hidden state在语义子空间中的轨迹
    - 对比翻译成功 vs 翻译失败的轨迹差异
    - 验证"翻译=子空间映射"假说

  Exp4: L6机制深度分析 — L6到底在计算什么?
    - L6的MLP/Attn对hidden state做了什么变换?
    - L6的变换方向是否指向英文子空间?
    - L6 ablate后，后续层的变换轨迹如何偏移?

Run:
  python tests/glm5/ccml_phase100_layer_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase100_layer_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase100_layer_dynamics.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase100_layer_dynamics.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model


# ============================================================
# 任务定义
# ============================================================
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"),
    ("水", "water"), ("火", "fire"), ("花", "flower"),
    ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"),
    ("茶", "tea"), ("米", "rice"), ("血", "blood"),
    ("眼", "eye"), ("手", "hand"), ("风", "wind"),
    ("雪", "snow"), ("星", "star"),
]

# 语义类别
SEMANTIC_CATEGORIES = {
    "动物": ["猫", "狗", "鱼", "鸟", "马", "牛", "羊", "猪", "鸡", "虎"],
    "食物": ["米", "茶", "肉", "面", "饼", "菜", "果", "糖", "酒", "奶"],
    "自然": ["水", "火", "风", "雪", "星", "月", "日", "山", "河", "海"],
    "身体": ["眼", "手", "头", "足", "心", "耳", "口", "鼻", "指", "骨"],
    "颜色": ["红", "白", "黑", "绿", "蓝", "黄", "紫", "灰", "橙", "粉"],
}

# 英文对照
EN_WORDS = {
    "动物": ["cat", "dog", "fish", "bird", "horse", "cow", "sheep", "pig", "chicken", "tiger"],
    "食物": ["rice", "tea", "meat", "noodle", "cake", "vegetable", "fruit", "sugar", "wine", "milk"],
    "自然": ["water", "fire", "wind", "snow", "star", "moon", "sun", "mountain", "river", "sea"],
    "身体": ["eye", "hand", "head", "foot", "heart", "ear", "mouth", "nose", "finger", "bone"],
    "颜色": ["red", "white", "black", "green", "blue", "yellow", "purple", "gray", "orange", "pink"],
}


def get_last_token_hidden(model, input_ids, device):
    """获取所有层的last token hidden state"""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hiddens = []
    for l in range(len(outputs.hidden_states)):
        h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
        hiddens.append(h)
    return hiddens


def get_logits_at_layer(model, input_ids, device):
    """获取最终层的logits"""
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    return logits


# ============================================================
# Exp 1: 层间变换分析
# ============================================================
def exp1_layer_transformations(model_name):
    """分析每层的hidden state变换: ||h_{l+1} - h_l|| 和变换方向"""
    print(f"\n{'='*60}")
    print(f"Exp 1: 层间变换分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    test_pairs = TRANSLATION_PAIRS[:15]

    # 三种条件的hidden state收集
    conditions = {
        "translation": [],  # "猫的英文是"
        "continuation": [],  # "猫"
        "zh_context": [],    # "猫是一种动物" (纯中文上下文)
    }

    for zh, en in test_pairs:
        # 翻译条件
        trans_prompt = f"{zh}的英文是"
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        conditions["translation"].append(hiddens)

        # 补全条件
        cont_prompt = zh
        inputs = tokenizer(cont_prompt, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        conditions["continuation"].append(hiddens)

        # 纯中文上下文
        zh_prompt = f"{zh}是一种"
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        conditions["zh_context"].append(hiddens)

    # 计算层间变换
    results = {}
    for cond_name, all_hiddens in conditions.items():
        layer_deltas = []  # ||h_{l+1} - h_l||
        layer_delta_dirs = []  # 变换方向(归一化)

        for sample_idx in range(len(all_hiddens)):
            deltas = []
            for l in range(n_layers):
                delta = all_hiddens[sample_idx][l+1] - all_hiddens[sample_idx][l]
                delta_norm = np.linalg.norm(delta)
                deltas.append(delta_norm)
            layer_deltas.append(deltas)

        # 平均变换幅度
        mean_deltas = np.mean(layer_deltas, axis=0)

        # 计算层间余弦相似度: 同一层不同样本的变换方向是否一致?
        delta_cosines = []
        for l in range(n_layers):
            cosines = []
            for i in range(len(all_hiddens)):
                for j in range(i+1, len(all_hiddens)):
                    d_i = all_hiddens[i][l+1] - all_hiddens[i][l]
                    d_j = all_hiddens[j][l+1] - all_hiddens[j][l]
                    n_i, n_j = np.linalg.norm(d_i), np.linalg.norm(d_j)
                    if n_i > 1e-6 and n_j > 1e-6:
                        cos = np.dot(d_i, d_j) / (n_i * n_j)
                        cosines.append(cos)
            delta_cosines.append(np.mean(cosines) if cosines else 0.0)

        results[cond_name] = {
            "mean_deltas": {str(l): float(mean_deltas[l]) for l in range(n_layers)},
            "delta_cosine_consistency": {str(l): float(delta_cosines[l]) for l in range(n_layers)},
        }

    # 跨条件对比: 翻译 vs 补全 vs 中文上下文 的变换差异
    cross_comparison = {}
    for l in range(n_layers):
        trans_deltas = [conditions["translation"][i][l+1] - conditions["translation"][i][l]
                       for i in range(len(conditions["translation"]))]
        cont_deltas = [conditions["continuation"][i][l+1] - conditions["continuation"][i][l]
                      for i in range(len(conditions["continuation"]))]
        zh_deltas = [conditions["zh_context"][i][l+1] - conditions["zh_context"][i][l]
                    for i in range(len(conditions["zh_context"]))]

        # 翻译变换 vs 补全变换的余弦相似度
        trans_cont_cos = []
        for i in range(len(trans_deltas)):
            n_t, n_c = np.linalg.norm(trans_deltas[i]), np.linalg.norm(cont_deltas[i])
            if n_t > 1e-6 and n_c > 1e-6:
                trans_cont_cos.append(np.dot(trans_deltas[i], cont_deltas[i]) / (n_t * n_c))

        # 翻译变换 vs 中文上下文变换
        trans_zh_cos = []
        for i in range(len(trans_deltas)):
            n_t, n_z = np.linalg.norm(trans_deltas[i]), np.linalg.norm(zh_deltas[i])
            if n_t > 1e-6 and n_z > 1e-6:
                trans_zh_cos.append(np.dot(trans_deltas[i], zh_deltas[i]) / (n_t * n_z))

        cross_comparison[str(l)] = {
            "trans_vs_cont_cosine": float(np.mean(trans_cont_cos)) if trans_cont_cos else 0.0,
            "trans_vs_zh_cosine": float(np.mean(trans_zh_cos)) if trans_zh_cos else 0.0,
        }

    results["cross_comparison"] = cross_comparison

    # 关键发现
    print("\n  层间变换幅度 (||h_{l+1} - h_l||):")
    for cond_name in ["translation", "continuation", "zh_context"]:
        deltas = results[cond_name]["mean_deltas"]
        # 找变换最大的层
        sorted_layers = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_layers[:5]
        print(f"    {cond_name}: Top5层 = {[(f'L{l}', f'{v:.3f}') for l, v in top5]}")

    print("\n  变换方向一致性 (余弦):")
    for cond_name in ["translation", "continuation", "zh_context"]:
        cosines = results[cond_name]["delta_cosine_consistency"]
        sorted_layers = sorted(cosines.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_layers[:5]
        print(f"    {cond_name}: Top5层 = {[(f'L{l}', f'{v:.4f}') for l, v in top5]}")

    print("\n  翻译 vs 补全 变换方向相似度:")
    trans_cont = results["cross_comparison"]
    sorted_layers = sorted(trans_cont.items(), key=lambda x: x[1]["trans_vs_cont_cosine"])
    bottom5 = sorted_layers[:5]  # 最不相似 = 最翻译特异
    bottom5_str = ", ".join([f"L{l}: {v['trans_vs_cont_cosine']:.4f}" for l, v in bottom5])
    print(f"    最不相似层(翻译特异): {bottom5_str}")

    # 保存
    save_path = f"tests/glm5_temp/phase100_exp1_{model_name}_layer_transformations.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: 语义对象几何学 — 概念流形分析
# ============================================================
def exp2_semantic_geometry(model_name):
    """分析不同语义类别(动物/食物/自然/身体/颜色)的hidden state几何结构"""
    print(f"\n{'='*60}")
    print(f"Exp 2: 语义对象几何学 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    # 收集各类别的hidden state
    category_hiddens = {}  # {category: {layer: [h1, h2, ...]}}

    for cat_name, zh_words in SEMANTIC_CATEGORIES.items():
        category_hiddens[cat_name] = defaultdict(list)
        for word in zh_words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
            for l in range(n_layers + 1):  # n_layers+1个hidden state (embedding + n_layers)
                category_hiddens[cat_name][l].append(hiddens[l])

    # 同样收集英文
    en_category_hiddens = {}
    for cat_name, en_words in EN_WORDS.items():
        en_category_hiddens[cat_name] = defaultdict(list)
        for word in en_words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
            for l in range(n_layers + 1):
                en_category_hiddens[cat_name][l].append(hiddens[l])

    # 分析1: 类内聚合度 vs 类间分离度
    results = {"intra_class_cohesion": {}, "inter_class_separation": {}, "subspace_overlap": {}}

    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        # 类内聚合度: 同类词的余弦相似度均值
        intra_cosines = {}
        for cat_name in SEMANTIC_CATEGORIES:
            hiddens_list = category_hiddens[cat_name][l]
            if len(hiddens_list) < 2:
                continue
            cosines = []
            for i in range(len(hiddens_list)):
                for j in range(i+1, len(hiddens_list)):
                    h_i, h_j = hiddens_list[i], hiddens_list[j]
                    n_i, n_j = np.linalg.norm(h_i), np.linalg.norm(h_j)
                    if n_i > 1e-6 and n_j > 1e-6:
                        cosines.append(np.dot(h_i, h_j) / (n_i * n_j))
            intra_cosines[cat_name] = float(np.mean(cosines)) if cosines else 0.0

        # 类间分离度: 不同类别的中心余弦相似度
        centers = {}
        for cat_name in SEMANTIC_CATEGORIES:
            hiddens_list = category_hiddens[cat_name][l]
            if hiddens_list:
                centers[cat_name] = np.mean(hiddens_list, axis=0)

        inter_cosines = {}
        cat_names = list(centers.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                n_i = np.linalg.norm(centers[cat_names[i]])
                n_j = np.linalg.norm(centers[cat_names[j]])
                if n_i > 1e-6 and n_j > 1e-6:
                    cos = np.dot(centers[cat_names[i]], centers[cat_names[j]]) / (n_i * n_j)
                    inter_cosines[f"{cat_names[i]}_vs_{cat_names[j]}"] = float(cos)

        results["intra_class_cohesion"][str(l)] = intra_cosines
        results["inter_class_separation"][str(l)] = inter_cosines

        # 子空间重叠: 用PCA分析主成分方向
        all_hiddens = []
        labels = []
        for cat_name in SEMANTIC_CATEGORIES:
            all_hiddens.extend(category_hiddens[cat_name][l])
            labels.extend([cat_name] * len(category_hiddens[cat_name][l]))

        if len(all_hiddens) >= 5:
            all_hiddens = np.array(all_hiddens)
            # 中心化
            mean_h = np.mean(all_hiddens, axis=0)
            centered = all_hiddens - mean_h
            # PCA取前10个主成分
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            top_components = Vt[:10]  # [10, d_model]

            # 每个类别在前10个主成分上的投影方差
            cat_variances = {}
            for cat_name in SEMANTIC_CATEGORIES:
                cat_hiddens = np.array(category_hiddens[cat_name][l])
                projections = cat_hiddens @ top_components.T  # [n_cat, 10]
                variances = np.var(projections, axis=0)
                cat_variances[cat_name] = [float(v) for v in variances]

            results["subspace_overlap"][str(l)] = cat_variances

    # 中英文子空间对齐分析
    results["zh_en_alignment"] = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue
        # 每个类别的中文中心 vs 英文中心
        alignment = {}
        for cat_name in SEMANTIC_CATEGORIES:
            zh_hiddens = category_hiddens[cat_name][l]
            en_hiddens = en_category_hiddens[cat_name][l]
            if zh_hiddens and en_hiddens:
                zh_center = np.mean(zh_hiddens, axis=0)
                en_center = np.mean(en_hiddens, axis=0)
                n_zh, n_en = np.linalg.norm(zh_center), np.linalg.norm(en_center)
                if n_zh > 1e-6 and n_en > 1e-6:
                    cos = np.dot(zh_center, en_center) / (n_zh * n_en)
                    alignment[cat_name] = float(cos)
        results["zh_en_alignment"][str(l)] = alignment

    # 输出关键发现
    print("\n  类内聚合度 (同类别余弦均值):")
    for l_str, cohesion in results["intra_class_cohesion"].items():
        avg_cohesion = np.mean(list(cohesion.values()))
        print(f"    L{l_str}: avg={avg_cohesion:.4f}, {cohesion}")

    print("\n  中英文子空间对齐 (同类中英文中心余弦):")
    for l_str, alignment in results["zh_en_alignment"].items():
        avg_align = np.mean(list(alignment.values()))
        print(f"    L{l_str}: avg={avg_align:.4f}")

    # 保存
    save_path = f"tests/glm5_temp/phase100_exp2_{model_name}_semantic_geometry.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: 翻译轨迹分析 — 子空间映射
# ============================================================
def exp3_translation_trajectory(model_name):
    """追踪翻译过程中hidden state在语义子空间中的轨迹"""
    print(f"\n{'='*60}")
    print(f"Exp 3: 翻译轨迹分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    test_pairs = TRANSLATION_PAIRS[:15]

    # 收集三种条件的hidden state轨迹
    trajectories = {
        "translation": [],  # "猫的英文是" → 逐层轨迹
        "zh_only": [],      # "猫" → 逐层轨迹
        "en_only": [],      # "cat" → 逐层轨迹
    }

    for zh, en in test_pairs:
        # 翻译条件
        trans_prompt = f"{zh}的英文是"
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        trajectories["translation"].append(hiddens)

        # 纯中文
        inputs = tokenizer(zh, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        trajectories["zh_only"].append(hiddens)

        # 纯英文
        inputs = tokenizer(en, return_tensors="pt").to(device)
        hiddens = get_last_token_hidden(model, inputs["input_ids"], device)
        trajectories["en_only"].append(hiddens)

    # 构建参考子空间: 用zh_only和en_only的最终层hidden state作为子空间锚点
    zh_final = np.array([traj[n_layers] for traj in trajectories["zh_only"]])  # [15, d_model]
    en_final = np.array([traj[n_layers] for traj in trajectories["en_only"]])  # [15, d_model]

    # 用PCA构建zh和en子空间
    zh_center = np.mean(zh_final, axis=0)
    en_center = np.mean(en_final, axis=0)

    # 计算翻译轨迹在每个层与zh/en中心的距离
    results = {"trajectory_analysis": {}}

    for sample_idx in range(len(test_pairs)):
        zh, en = test_pairs[sample_idx]
        traj = trajectories["translation"][sample_idx]

        sample_result = {}
        for l in range(n_layers + 1):
            h = traj[l]
            # 到zh中心的距离
            dist_zh = np.linalg.norm(h - zh_center)
            # 到en中心的距离
            dist_en = np.linalg.norm(h - en_center)
            # zh/en距离比 (越小=越接近英文)
            dist_ratio = dist_en / (dist_zh + 1e-8)

            # 与zh_final/en_final的余弦相似度
            n_h = np.linalg.norm(h)
            cos_zh, cos_en = 0.0, 0.0
            if n_h > 1e-6:
                n_zh_c = np.linalg.norm(zh_center)
                n_en_c = np.linalg.norm(en_center)
                if n_zh_c > 1e-6:
                    cos_zh = float(np.dot(h, zh_center) / (n_h * n_zh_c))
                if n_en_c > 1e-6:
                    cos_en = float(np.dot(h, en_center) / (n_h * n_en_c))

            sample_result[str(l)] = {
                "dist_zh": float(dist_zh),
                "dist_en": float(dist_en),
                "dist_ratio": float(dist_ratio),
                "cos_zh": cos_zh,
                "cos_en": cos_en,
            }

        results["trajectory_analysis"][f"{zh}->{en}"] = sample_result

    # 平均轨迹
    avg_trajectory = {}
    for l in range(n_layers + 1):
        dists_zh = [results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["dist_zh"]
                    for zh, en in test_pairs]
        dists_en = [results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["dist_en"]
                    for zh, en in test_pairs]
        cos_zhs = [results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["cos_zh"]
                   for zh, en in test_pairs]
        cos_ens = [results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["cos_en"]
                   for zh, en in test_pairs]

        avg_trajectory[str(l)] = {
            "avg_dist_zh": float(np.mean(dists_zh)),
            "avg_dist_en": float(np.mean(dists_en)),
            "avg_cos_zh": float(np.mean(cos_zhs)),
            "avg_cos_en": float(np.mean(cos_ens)),
            "cos_diff": float(np.mean(cos_ens) - np.mean(cos_zhs)),  # 正=更接近英文
        }

    results["avg_trajectory"] = avg_trajectory

    # 关键: 找到轨迹"穿越点" — cos_en > cos_zh的第一层
    switch_points = []
    for zh, en in test_pairs:
        for l in range(n_layers + 1):
            cos_zh = results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["cos_zh"]
            cos_en = results["trajectory_analysis"][f"{zh}->{en}"][str(l)]["cos_en"]
            if cos_en > cos_zh:
                switch_points.append(l)
                break

    results["switch_point_stats"] = {
        "mean": float(np.mean(switch_points)) if switch_points else -1,
        "median": float(np.median(switch_points)) if switch_points else -1,
        "min": int(np.min(switch_points)) if switch_points else -1,
        "max": int(np.max(switch_points)) if switch_points else -1,
        "distribution": {str(l): int(sum(1 for s in switch_points if s == l))
                        for l in range(n_layers + 1)},
    }

    # 输出关键发现
    print("\n  翻译轨迹 — 平均与zh/en中心余弦相似度:")
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if str(l) in avg_trajectory:
            t = avg_trajectory[str(l)]
            print(f"    L{l}: cos_zh={t['avg_cos_zh']:.4f}, cos_en={t['avg_cos_en']:.4f}, "
                  f"diff={t['cos_diff']:+.4f}")

    print(f"\n  子空间穿越点: mean={results['switch_point_stats']['mean']:.1f}, "
          f"median={results['switch_point_stats']['median']:.1f}")
    print(f"    分布: {results['switch_point_stats']['distribution']}")

    # 保存
    save_path = f"tests/glm5_temp/phase100_exp3_{model_name}_translation_trajectory.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: L6机制深度分析 — L6到底做了什么变换?
# ============================================================
def exp4_l6_mechanism(model_name):
    """分析L6的Attn/MLP对hidden state做了什么变换"""
    print(f"\n{'='*60}")
    print(f"Exp 4: L6机制深度分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    # 获取num_attention_heads
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        n_heads = d_model // 128
    head_dim = d_model // n_heads
    print(f"  n_heads: {n_heads}, head_dim: {head_dim}")

    test_pairs = TRANSLATION_PAIRS[:10]

    # 收集L6的attn output和MLP output
    l6_attn_outputs = {"translation": [], "continuation": []}
    l6_mlp_outputs = {"translation": [], "continuation": []}
    l6_hiddens = {"translation": {"before": [], "after": []},
                  "continuation": {"before": [], "after": []}}

    layers = get_layers(model)
    l6 = layers[6]

    for zh, en in test_pairs:
        for cond_name, prompt_fn in [
            ("translation", lambda z: f"{z}的英文是"),
            ("continuation", lambda z: z),
        ]:
            prompt = prompt_fn(zh)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Hook L6的attn和MLP output
            attn_out = {}
            mlp_out = {}

            def attn_hook(module, input, output):
                if isinstance(output, tuple):
                    val = output[0]
                else:
                    val = output
                if val.dim() == 3:
                    attn_out["value"] = val[0, -1, :].float().cpu().numpy()
                elif val.dim() == 2:
                    attn_out["value"] = val[-1, :].float().cpu().numpy()
                elif val.dim() == 1:
                    attn_out["value"] = val.float().cpu().numpy()
                else:
                    attn_out["value"] = val.flatten()[-2560:].float().cpu().numpy()

            def mlp_hook(module, input, output):
                if isinstance(output, tuple):
                    val = output[0]
                else:
                    val = output
                if val.dim() == 3:
                    mlp_out["value"] = val[0, -1, :].float().cpu().numpy()
                elif val.dim() == 2:
                    mlp_out["value"] = val[-1, :].float().cpu().numpy()
                elif val.dim() == 1:
                    mlp_out["value"] = val.float().cpu().numpy()
                else:
                    mlp_out["value"] = val.flatten()[-2560:].float().cpu().numpy()

            h_attn = l6.self_attn.register_forward_hook(attn_hook)
            h_mlp = l6.mlp.register_forward_hook(mlp_hook)

            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)

            h_attn.remove()
            h_mlp.remove()

            # L6前后的hidden state
            h_before = outputs.hidden_states[6][0, -1, :].float().cpu().numpy()
            h_after = outputs.hidden_states[7][0, -1, :].float().cpu().numpy()

            if "value" in attn_out:
                l6_attn_outputs[cond_name].append(attn_out["value"])
            if "value" in mlp_out:
                l6_mlp_outputs[cond_name].append(mlp_out["value"])
            l6_hiddens[cond_name]["before"].append(h_before)
            l6_hiddens[cond_name]["after"].append(h_after)

    # 分析1: L6 attn/MLP输出方向
    results = {}

    # 翻译 vs 补全 的L6 attn输出方向
    if l6_attn_outputs["translation"] and l6_attn_outputs["continuation"]:
        trans_attn = np.array(l6_attn_outputs["translation"])
        cont_attn = np.array(l6_attn_outputs["continuation"])

        # 平均方向
        trans_attn_mean = np.mean(trans_attn, axis=0)
        cont_attn_mean = np.mean(cont_attn, axis=0)

        # 翻译attn方向 vs 补全attn方向的余弦
        n_t, n_c = np.linalg.norm(trans_attn_mean), np.linalg.norm(cont_attn_mean)
        if n_t > 1e-6 and n_c > 1e-6:
            attn_dir_cos = float(np.dot(trans_attn_mean, cont_attn_mean) / (n_t * n_c))
        else:
            attn_dir_cos = 0.0

        # 翻译attn输出的范数
        trans_attn_norms = [float(np.linalg.norm(v)) for v in l6_attn_outputs["translation"]]
        cont_attn_norms = [float(np.linalg.norm(v)) for v in l6_attn_outputs["continuation"]]

        results["l6_attn"] = {
            "direction_cosine_trans_vs_cont": attn_dir_cos,
            "translation_norm_mean": float(np.mean(trans_attn_norms)),
            "continuation_norm_mean": float(np.mean(cont_attn_norms)),
        }

    # 翻译 vs 补全 的L6 MLP输出方向
    if l6_mlp_outputs["translation"] and l6_mlp_outputs["continuation"]:
        trans_mlp = np.array(l6_mlp_outputs["translation"])
        cont_mlp = np.array(l6_mlp_outputs["continuation"])

        trans_mlp_mean = np.mean(trans_mlp, axis=0)
        cont_mlp_mean = np.mean(cont_mlp, axis=0)

        n_t, n_c = np.linalg.norm(trans_mlp_mean), np.linalg.norm(cont_mlp_mean)
        if n_t > 1e-6 and n_c > 1e-6:
            mlp_dir_cos = float(np.dot(trans_mlp_mean, cont_mlp_mean) / (n_t * n_c))
        else:
            mlp_dir_cos = 0.0

        trans_mlp_norms = [float(np.linalg.norm(v)) for v in l6_mlp_outputs["translation"]]
        cont_mlp_norms = [float(np.linalg.norm(v)) for v in l6_mlp_outputs["continuation"]]

        results["l6_mlp"] = {
            "direction_cosine_trans_vs_cont": mlp_dir_cos,
            "translation_norm_mean": float(np.mean(trans_mlp_norms)),
            "continuation_norm_mean": float(np.mean(cont_mlp_norms)),
        }

    # 分析2: L6变换后hidden state向哪个方向移动?
    # 计算 L6变换向量 = h_after - h_before
    for cond_name in ["translation", "continuation"]:
        deltas = []
        for i in range(len(l6_hiddens[cond_name]["before"])):
            delta = l6_hiddens[cond_name]["after"][i] - l6_hiddens[cond_name]["before"][i]
            deltas.append(delta)

        results[f"l6_delta_{cond_name}"] = {
            "mean_norm": float(np.mean([np.linalg.norm(d) for d in deltas])),
            "mean_direction": [float(x) for x in np.mean(deltas, axis=0)[:50]],  # 只存前50维
        }

    # 分析3: L6变换是否指向英文子空间?
    # 收集英文词的最终hidden state作为英文子空间锚点
    en_hiddens_final = []
    for _, en in test_pairs:
        inputs = tokenizer(en, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        en_hiddens_final.append(outputs.hidden_states[n_layers][0, -1, :].float().cpu().numpy())

    en_center = np.mean(en_hiddens_final, axis=0)

    # L6变换向量与"指向英文中心"方向的余弦相似度
    trans_delta_to_en_cos = []
    for i in range(len(l6_hiddens["translation"]["before"])):
        delta = l6_hiddens["translation"]["after"][i] - l6_hiddens["translation"]["before"][i]
        # 指向英文中心的方向
        to_en = en_center - l6_hiddens["translation"]["before"][i]
        n_d, n_e = np.linalg.norm(delta), np.linalg.norm(to_en)
        if n_d > 1e-6 and n_e > 1e-6:
            cos = np.dot(delta, to_en) / (n_d * n_e)
            trans_delta_to_en_cos.append(float(cos))

    cont_delta_to_en_cos = []
    for i in range(len(l6_hiddens["continuation"]["before"])):
        delta = l6_hiddens["continuation"]["after"][i] - l6_hiddens["continuation"]["before"][i]
        to_en = en_center - l6_hiddens["continuation"]["before"][i]
        n_d, n_e = np.linalg.norm(delta), np.linalg.norm(to_en)
        if n_d > 1e-6 and n_e > 1e-6:
            cos = np.dot(delta, to_en) / (n_d * n_e)
            cont_delta_to_en_cos.append(float(cos))

    results["l6_delta_to_en_alignment"] = {
        "translation": {
            "mean_cosine": float(np.mean(trans_delta_to_en_cos)) if trans_delta_to_en_cos else 0.0,
            "positive_fraction": float(np.mean([1 if c > 0 else 0 for c in trans_delta_to_en_cos])) if trans_delta_to_en_cos else 0.0,
        },
        "continuation": {
            "mean_cosine": float(np.mean(cont_delta_to_en_cos)) if cont_delta_to_en_cos else 0.0,
            "positive_fraction": float(np.mean([1 if c > 0 else 0 for c in cont_delta_to_en_cos])) if cont_delta_to_en_cos else 0.0,
        },
    }

    # 输出关键发现
    print("\n  L6 Attn输出:")
    if "l6_attn" in results:
        r = results["l6_attn"]
        print(f"    翻译 vs 补全 方向余弦: {r['direction_cosine_trans_vs_cont']:.4f}")
        print(f"    翻译范数: {r['translation_norm_mean']:.4f}, 补全范数: {r['continuation_norm_mean']:.4f}")

    print("\n  L6 MLP输出:")
    if "l6_mlp" in results:
        r = results["l6_mlp"]
        print(f"    翻译 vs 补全 方向余弦: {r['direction_cosine_trans_vs_cont']:.4f}")
        print(f"    翻译范数: {r['translation_norm_mean']:.4f}, 补全范数: {r['continuation_norm_mean']:.4f}")

    print("\n  L6变换是否指向英文子空间?")
    r = results["l6_delta_to_en_alignment"]
    print(f"    翻译条件: cosine={r['translation']['mean_cosine']:.4f}, "
          f"正比例={r['translation']['positive_fraction']:.2f}")
    print(f"    补全条件: cosine={r['continuation']['mean_cosine']:.4f}, "
          f"正比例={r['continuation']['positive_fraction']:.2f}")

    # 保存
    save_path = f"tests/glm5_temp/phase100_exp4_{model_name}_l6_mechanism.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_layer_transformations(args.model)
    elif args.exp == 2:
        exp2_semantic_geometry(args.model)
    elif args.exp == 3:
        exp3_translation_trajectory(args.model)
    elif args.exp == 4:
        exp4_l6_mechanism(args.model)
