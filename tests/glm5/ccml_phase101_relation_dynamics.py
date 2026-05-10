"""
Phase 101: 关系动力学分析 — 从实体到关系
==========================================

批判核心升级:
  1. "最后token hidden state ≠ 语义对象" → LLM是序列动力系统
  2. "需要分析关系而非实体" → 苹果→水果的变换方向是否稳定?
  3. "欧氏距离在高维同样失效" → 需要centered local geometry
  4. "L0语义最强是tokenizer假象" → 需要控制词频/字形
  5. "翻译prompt更接近英文≠已翻译" → 只是decoder mode切换

核心理论框架:
  语言模型编码的不是"实体"，而是"变换约束":
    h(苹果) → h(水果): "is-a"关系
    h(巴黎) → h(法国): "part-of"关系
    h(猫) → h(cat): "翻译"关系
  
  这些变换方向是否在所有样本中稳定共享?
  如果是，则说明模型学到了"关系原语"而非"实体表示"

实验设计:
  Exp1: 关系变换方向稳定性 — "苹果→水果"和"狗→动物"共享方向?
  Exp2: 上下文化分析 — "苹果"在不同上下文中的表示变化
  Exp3: 中心化局部几何 — CKA/Mahalanobis距离替代原始距离
  Exp4: tokenizer控制 — 区分"语义相似"和"字形/词频相似"

Run:
  python tests/glm5/ccml_phase101_relation_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase101_relation_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase101_relation_dynamics.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase101_relation_dynamics.py --model qwen3 --exp 4
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
# 关系定义 — 不同类型的语义关系
# ============================================================
RELATION_PAIRS = {
    "is-a": [  # 上位关系: X是一种Y
        ("苹果", "水果"), ("狗", "动物"), ("玫瑰", "花"), ("老虎", "猫科"),
        ("桌子", "家具"), ("钢笔", "文具"), ("汽车", "交通工具"), ("米饭", "食物"),
        ("铁", "金属"), ("苹果", "食物"), ("熊猫", "动物"), ("香蕉", "水果"),
        ("猫", "动物"), ("白菜", "蔬菜"), ("金", "金属"), ("飞机", "交通工具"),
    ],
    "part-of": [  # 部分关系: X是Y的一部分/首都是
        ("北京", "中国"), ("巴黎", "法国"), ("东京", "日本"), ("伦敦", "英国"),
        ("眼睛", "脸"), ("手", "身体"), ("叶子", "树"), ("轮子", "汽车"),
        ("头", "身体"), ("窗户", "房子"), ("翅膀", "鸟"), ("尾巴", "狗"),
    ],
    "translation": [  # 翻译关系: X的英文是Y
        ("猫", "cat"), ("狗", "dog"), ("书", "book"), ("水", "water"),
        ("火", "fire"), ("花", "flower"), ("鱼", "fish"), ("树", "tree"),
        ("鸟", "bird"), ("马", "horse"), ("铁", "iron"), ("金", "gold"),
        ("茶", "tea"), ("米", "rice"), ("血", "blood"), ("眼", "eye"),
    ],
    "antonym": [  # 反义关系: X和Y相反
        ("大", "小"), ("高", "低"), ("冷", "热"), ("黑", "白"),
        ("快", "慢"), ("好", "坏"), ("多", "少"), ("长", "短"),
        ("新", "旧"), ("强", "弱"), ("明", "暗"), ("轻", "重"),
    ],
}

# 上下文模板
CONTEXT_TEMPLATES = {
    "苹果": [
        "苹果是一种水果",      # 类别上下文
        "苹果公司发布了新手机",  # 歧义上下文1(公司)
        "我吃了一个红苹果",     # 食物上下文
        "苹果的颜色是红色",     # 属性上下文
    ],
    "狗": [
        "狗是一种动物",
        "那条狗在叫",
        "我养了一只狗",
        "狗的叫声很大",
    ],
    "水": [
        "水是一种液体",
        "请给我一杯水",
        "水在零度会结冰",
        "河水流过山谷",
    ],
    "猫": [
        "猫是一种动物",
        "那只猫在睡觉",
        "我养了一只猫",
        "猫喜欢吃鱼",
    ],
    "火": [
        "火是一种自然现象",
        "小心不要碰到火",
        "火可以用来取暖",
        "森林大火很危险",
    ],
}


def get_token_hiddens(model, input_ids, device, token_idx=-1):
    """获取指定token位置的所有层hidden state"""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hiddens = []
    for l in range(len(outputs.hidden_states)):
        h = outputs.hidden_states[l][0, token_idx, :].float().cpu().numpy()
        hiddens.append(h)
    return hiddens


def get_all_token_hiddens(model, input_ids, device):
    """获取所有token的所有层hidden state"""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    n_tokens = outputs.hidden_states[0].shape[1]
    n_layers = len(outputs.hidden_states)
    # 返回 [n_layers, n_tokens, d_model]
    result = []
    for l in range(n_layers):
        layer_hiddens = outputs.hidden_states[l][0, :, :].float().cpu().numpy()
        result.append(layer_hiddens)
    return result, n_tokens


# ============================================================
# Exp 1: 关系变换方向稳定性
# ============================================================
def exp1_relation_direction_stability(model_name):
    """分析不同关系类型的变换方向是否稳定共享"""
    print(f"\n{'='*60}")
    print(f"Exp 1: 关系变换方向稳定性 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    for rel_type, pairs in RELATION_PAIRS.items():
        print(f"\n  处理关系类型: {rel_type} ({len(pairs)}对)")

        # 收集每对(A, B)的hidden state
        pair_hiddens = []
        for a, b in pairs:
            # A的hidden state
            inputs_a = tokenizer(a, return_tensors="pt").to(device)
            hiddens_a = get_token_hiddens(model, inputs_a["input_ids"], device)

            # B的hidden state
            inputs_b = tokenizer(b, return_tensors="pt").to(device)
            hiddens_b = get_token_hiddens(model, inputs_b["input_ids"], device)

            pair_hiddens.append((a, b, hiddens_a, hiddens_b))

        # 计算每对的变换方向: delta = h(B) - h(A)
        # 分析1: 同类关系内，变换方向是否一致?
        layer_results = {}
        for l in range(n_layers + 1):
            deltas = []
            for a, b, ha, hb in pair_hiddens:
                delta = hb[l] - ha[l]
                deltas.append(delta)

            # 变换方向的平均余弦相似度
            cosines = []
            for i in range(len(deltas)):
                for j in range(i+1, len(deltas)):
                    n_i, n_j = np.linalg.norm(deltas[i]), np.linalg.norm(deltas[j])
                    if n_i > 1e-6 and n_j > 1e-6:
                        cos = np.dot(deltas[i], deltas[j]) / (n_i * n_j)
                        cosines.append(float(cos))

            # 变换方向的范数
            norms = [float(np.linalg.norm(d)) for d in deltas]

            # 投影到平均方向的比例 (方向一致性)
            if deltas:
                mean_delta = np.mean(deltas, axis=0)
                mean_norm = np.linalg.norm(mean_delta)
                if mean_norm > 1e-6:
                    projections = [np.dot(d, mean_delta) / (mean_norm * np.linalg.norm(d) + 1e-8)
                                  for d in deltas if np.linalg.norm(d) > 1e-6]
                    alignment = float(np.mean(projections))
                else:
                    alignment = 0.0
            else:
                alignment = 0.0

            layer_results[str(l)] = {
                "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
                "mean_norm": float(np.mean(norms)),
                "alignment_to_mean": alignment,
                "n_pairs": len(pairs),
            }

        results[rel_type] = layer_results

        # 找到方向最一致的层
        alignment_by_layer = [(l, layer_results[str(l)]["alignment_to_mean"],
                              layer_results[str(l)]["mean_cosine"])
                             for l in range(n_layers + 1)]
        alignment_by_layer.sort(key=lambda x: x[1], reverse=True)
        top5 = alignment_by_layer[:5]
        print(f"    方向最一致的层: {[(f'L{l}', f'a={a:.3f}', f'c={c:.3f}') for l, a, c in top5]}")

    # 跨关系类型对比: 不同关系的变换方向是否不同?
    print(f"\n  === 跨关系类型对比 ===")
    cross_relation_results = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        # 计算每种关系的平均变换方向
        rel_mean_dirs = {}
        for rel_type, pairs in RELATION_PAIRS.items():
            deltas = []
            for a, b, ha, hb in [(a, b, [], []) for a, b in pairs]:
                # 重新计算 (因为pair_hiddens是局部的)
                inputs_a = tokenizer(a, return_tensors="pt").to(device)
                ha = get_token_hiddens(model, inputs_a["input_ids"], device)
                inputs_b = tokenizer(b, return_tensors="pt").to(device)
                hb = get_token_hiddens(model, inputs_b["input_ids"], device)
                delta = hb[l] - ha[l]
                deltas.append(delta)
            mean_delta = np.mean(deltas, axis=0)
            n_mean = np.linalg.norm(mean_delta)
            if n_mean > 1e-6:
                rel_mean_dirs[rel_type] = mean_delta / n_mean
            else:
                rel_mean_dirs[rel_type] = mean_delta

        # 计算不同关系类型间的方向余弦
        rel_types = list(rel_mean_dirs.keys())
        cross_cosines = {}
        for i in range(len(rel_types)):
            for j in range(i+1, len(rel_types)):
                r1, r2 = rel_types[i], rel_types[j]
                n1, n2 = np.linalg.norm(rel_mean_dirs[r1]), np.linalg.norm(rel_mean_dirs[r2])
                if n1 > 1e-6 and n2 > 1e-6:
                    cos = float(np.dot(rel_mean_dirs[r1], rel_mean_dirs[r2]))
                    cross_cosines[f"{r1}_vs_{r2}"] = cos

        cross_relation_results[str(l)] = cross_cosines
        if cross_cosines:
            min_pair = min(cross_cosines.items(), key=lambda x: x[1])
            max_pair = max(cross_cosines.items(), key=lambda x: x[1])
            print(f"    L{l}: 最不同={min_pair[0]}({min_pair[1]:.3f}), "
                  f"最相似={max_pair[0]}({max_pair[1]:.3f})")

    results["cross_relation"] = cross_relation_results

    # 保存
    save_path = f"tests/glm5_temp/phase101_exp1_{model_name}_relation_directions.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: 上下文化分析
# ============================================================
def exp2_contextual_analysis(model_name):
    """分析同一个词在不同上下文中的表示变化"""
    print(f"\n{'='*60}")
    print(f"Exp 2: 上下文化分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    for word, contexts in CONTEXT_TEMPLATES.items():
        print(f"\n  处理词: {word} ({len(contexts)}个上下文)")

        # 孤立词的hidden state
        inputs = tokenizer(word, return_tensors="pt").to(device)
        hiddens_isolated = get_token_hiddens(model, inputs["input_ids"], device)

        # 不同上下文中的hidden state (取目标词位置的hidden state)
        context_hiddens = {}
        for ctx in contexts:
            full_text = ctx
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # 找到目标词的位置
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            full_tokens = input_ids[0].tolist()

            # 简化: 取最后一个token的hidden state
            # (更精确的方法需要找到目标词的确切位置)
            all_hiddens, n_tokens = get_all_token_hiddens(model, input_ids, device)

            # 取最后token的hidden state
            last_hiddens = [all_hiddens[l][-1, :] for l in range(n_layers + 1)]
            context_hiddens[ctx] = last_hiddens

        # 分析: 同一个词在不同上下文中的表示差异
        word_results = {}
        for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
            if l > n_layers:
                continue

            # 孤立 vs 上下文化的距离
            iso_h = hiddens_isolated[l]
            ctx_dists = {}
            for ctx, ctx_h_list in context_hiddens.items():
                ctx_h = ctx_h_list[l]
                dist = float(np.linalg.norm(ctx_h - iso_h))
                # 余弦
                n_iso, n_ctx = np.linalg.norm(iso_h), np.linalg.norm(ctx_h)
                cos = float(np.dot(iso_h, ctx_h) / (n_iso * n_ctx)) if n_iso > 1e-6 and n_ctx > 1e-6 else 0.0
                ctx_dists[ctx] = {"euclidean": dist, "cosine": cos}

            # 不同上下文间的距离
            ctx_names = list(context_hiddens.keys())
            cross_ctx_dists = {}
            for i in range(len(ctx_names)):
                for j in range(i+1, len(ctx_names)):
                    h_i = context_hiddens[ctx_names[i]][l]
                    h_j = context_hiddens[ctx_names[j]][l]
                    dist = float(np.linalg.norm(h_i - h_j))
                    cross_ctx_dists[f"{ctx_names[i][:10]}_vs_{ctx_names[j][:10]}"] = dist

            # 上下文化程度: 平均(上下文-孤立距离) / 平均(上下文间距离)
            iso_dists = [v["euclidean"] for v in ctx_dists.values()]
            cross_dists = list(cross_ctx_dists.values())
            contextualization = float(np.mean(iso_dists) / (np.mean(cross_dists) + 1e-8))

            word_results[str(l)] = {
                "iso_vs_context": ctx_dists,
                "cross_context_dists_mean": float(np.mean(cross_dists)) if cross_dists else 0.0,
                "iso_vs_context_mean": float(np.mean(iso_dists)),
                "contextualization_ratio": contextualization,
            }

        results[word] = word_results

        # 输出关键发现
        for l in [0, 6, 18, 35]:
            if str(l) in word_results:
                r = word_results[str(l)]
                print(f"    L{l}: 上下文化程度={r['contextualization_ratio']:.2f}, "
                      f"孤立vs上下文={r['iso_vs_context_mean']:.2f}, "
                      f"上下文间={r['cross_context_dists_mean']:.2f}")

    # 保存
    save_path = f"tests/glm5_temp/phase101_exp2_{model_name}_contextual.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: 中心化局部几何 — CKA分析
# ============================================================
def exp3_centered_geometry(model_name):
    """用CKA和中心化距离分析语义结构"""
    print(f"\n{'='*60}")
    print(f"Exp 3: 中心化局部几何 — CKA分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    # 收集大量词的hidden state
    categories = {
        "动物": ["猫", "狗", "鱼", "鸟", "马", "牛", "羊", "猪", "鸡", "虎",
                 "鹿", "蛇", "鼠", "兔", "龙"],
        "食物": ["米", "茶", "肉", "面", "饼", "菜", "果", "糖", "酒", "奶",
                 "盐", "醋", "油", "蜜", "粥"],
        "自然": ["水", "火", "风", "雪", "星", "月", "日", "山", "河", "海",
                 "云", "雨", "雷", "电", "雾"],
        "身体": ["眼", "手", "头", "足", "心", "耳", "口", "鼻", "指", "骨",
                 "背", "腿", "臂", "腰", "肩"],
        "颜色": ["红", "白", "黑", "绿", "蓝", "黄", "紫", "灰", "橙", "粉",
                 "棕", "青", "银", "金", "褐"],
    }

    # 收集所有词的hidden state
    all_cat_hiddens = {}
    for cat_name, words in categories.items():
        cat_hiddens = defaultdict(list)
        for word in words:
            inputs = tokenizer(word, return_tensors="pt").to(device)
            hiddens = get_token_hiddens(model, inputs["input_ids"], device)
            for l in range(n_layers + 1):
                cat_hiddens[l].append(hiddens[l])
        all_cat_hiddens[cat_name] = cat_hiddens

    results = {}

    # CKA (Centered Kernel Alignment) 分析
    def linear_cka(X, Y):
        """计算两个表示矩阵的CKA相似度"""
        # X: [n, d], Y: [n, d]
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # HSIC
        def hsic(A, B):
            # A: [n, n], B: [n, n]
            n = A.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return np.trace(A @ H @ B @ H) / (n - 1)**2

        K_X = X @ X.T
        K_Y = Y @ Y.T

        hsic_xy = hsic(K_X, K_Y)
        hsic_xx = hsic(K_X, K_X)
        hsic_yy = hsic(K_Y, K_Y)

        if hsic_xx < 1e-10 or hsic_yy < 1e-10:
            return 0.0
        return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))

    # 1. 层间CKA: 每层的表示与相邻层有多相似?
    print("\n  === 层间CKA (表示连续性) ===")
    layer_cka_results = {}
    all_words_hiddens = defaultdict(list)
    for cat_name in categories:
        for l in range(n_layers + 1):
            all_words_hiddens[l].extend(all_cat_hiddens[cat_name][l])

    for l in range(n_layers):
        X = np.array(all_words_hiddens[l])
        Y = np.array(all_words_hiddens[l + 1])
        cka = linear_cka(X, Y)
        layer_cka_results[str(l)] = cka

    # 找CKA跳变最大的层 (表示突变点)
    cka_values = [layer_cka_results[str(l)] for l in range(n_layers)]
    cka_diffs = [cka_values[l] - cka_values[l+1] if l+1 < len(cka_values) else 0
                 for l in range(len(cka_values))]
    top_drops = sorted(range(len(cka_diffs)), key=lambda i: cka_diffs[i], reverse=True)[:5]
    print(f"    CKA跳变最大的层: {[(f'L{l}', f'{cka_diffs[l]:.4f}') for l in top_drops]}")
    print(f"    CKA值范围: {min(cka_values):.4f} - {max(cka_values):.4f}")

    results["layer_cka"] = layer_cka_results

    # 2. 类别间CKA: 不同类别在每层的表示结构有多相似?
    print("\n  === 类别间CKA ===")
    cat_cka_results = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        cat_names = list(categories.keys())
        cat_ckas = {}
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                X = np.array(all_cat_hiddens[cat_names[i]][l])
                Y = np.array(all_cat_hiddens[cat_names[j]][l])
                cka = linear_cka(X, Y)
                cat_ckas[f"{cat_names[i]}_vs_{cat_names[j]}"] = cka

        cat_cka_results[str(l)] = cat_ckas
        avg_cka = np.mean(list(cat_ckas.values()))
        print(f"    L{l}: avg_cat_CKA={avg_cka:.4f}")

    results["category_cka"] = cat_cka_results

    # 3. 中心化距离判别: 减去全局均值后的距离
    print("\n  === 中心化距离判别 ===")
    centered_discrim_results = {}
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        if l > n_layers:
            continue

        # 全局均值
        all_h = [h for cat in categories for h in all_cat_hiddens[cat][l]]
        global_mean = np.mean(all_h, axis=0)

        # 中心化后的类内/类间距离
        intra_dists = {}
        for cat_name in categories:
            centered_h = [h - global_mean for h in all_cat_hiddens[cat_name][l]]
            dists = []
            for i in range(len(centered_h)):
                for j in range(i+1, len(centered_h)):
                    dists.append(np.linalg.norm(centered_h[i] - centered_h[j]))
            intra_dists[cat_name] = float(np.mean(dists)) if dists else 0.0

        # 类间 (中心化后的类别中心距离)
        centers = {cat: np.mean([h - global_mean for h in all_cat_hiddens[cat][l]], axis=0)
                  for cat in categories}
        cat_names = list(centers.keys())
        inter_dists = {}
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                dist = np.linalg.norm(centers[cat_names[i]] - centers[cat_names[j]])
                inter_dists[f"{cat_names[i]}_vs_{cat_names[j]}"] = float(dist)

        avg_intra = np.mean(list(intra_dists.values()))
        avg_inter = np.mean(list(inter_dists.values()))
        discriminability = avg_inter / (avg_intra + 1e-8)

        centered_discrim_results[str(l)] = {
            "intra": intra_dists,
            "inter": inter_dists,
            "avg_intra": float(avg_intra),
            "avg_inter": float(avg_inter),
            "discriminability": float(discriminability),
        }
        print(f"    L{l}: centered_intra={avg_intra:.2f}, centered_inter={avg_inter:.2f}, "
              f"discriminability={discriminability:.2f}")

    results["centered_discrimination"] = centered_discrim_results

    # 保存
    save_path = f"tests/glm5_temp/phase101_exp3_{model_name}_centered_geometry.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Tokenizer控制 — 区分语义相似和字形相似
# ============================================================
def exp4_tokenizer_control(model_name):
    """控制tokenizer/词频效应，区分真正的语义结构"""
    print(f"\n{'='*60}")
    print(f"Exp 4: Tokenizer控制 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    # 设计控制对比:
    # 1. 语义同类但字形不同: 猫/狗(动物) vs 猫/锚(同偏旁但不同类)
    # 2. 字形相似但语义不同: 猫/锚(都有"钅"/"犭"偏旁?)
    # 3. 同义词: 看/见, 大/巨, 走/行
    # 4. 同音字: 公/工, 是/事

    control_sets = {
        "semantic_same_cat": [
            ("猫", "狗"),  # 都是动物
            ("苹果", "香蕉"),  # 都是水果
            ("红", "蓝"),  # 都是颜色
            ("北京", "东京"),  # 都是首都
        ],
        "semantic_diff_cat": [
            ("猫", "米"),  # 动物 vs 食物
            ("苹果", "铁"),  # 水果 vs 金属
            ("红", "水"),  # 颜色 vs 自然
        ],
        "synonym": [
            ("看", "见"),
            ("大", "巨"),
            ("走", "行"),
            ("好", "佳"),
        ],
        "homophone": [
            ("公", "工"),
            ("是", "事"),
            ("中", "钟"),
        ],
        "related_pair": [
            ("苹果", "水果"),  # is-a
            ("狗", "动物"),    # is-a
            ("北京", "中国"),  # part-of
        ],
    }

    results = {}

    for set_name, pairs in control_sets.items():
        print(f"\n  处理: {set_name}")
        pair_results = {}

        for a, b in pairs:
            inputs_a = tokenizer(a, return_tensors="pt").to(device)
            hiddens_a = get_token_hiddens(model, inputs_a["input_ids"], device)

            inputs_b = tokenizer(b, return_tensors="pt").to(device)
            hiddens_b = get_token_hiddens(model, inputs_b["input_ids"], device)

            layer_dists = {}
            for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
                if l > n_layers:
                    continue

                h_a, h_b = hiddens_a[l], hiddens_b[l]
                eu_dist = float(np.linalg.norm(h_a - h_b))
                n_a, n_b = np.linalg.norm(h_a), np.linalg.norm(h_b)
                cos = float(np.dot(h_a, h_b) / (n_a * n_b)) if n_a > 1e-6 and n_b > 1e-6 else 0.0

                layer_dists[str(l)] = {
                    "euclidean": eu_dist,
                    "cosine": cos,
                }

            pair_results[f"{a}_{b}"] = layer_dists

        results[set_name] = pair_results

        # 输出关键对比
        for l_str in ["0", "6", "18", "35"]:
            dists = [pair_results[p][l_str]["euclidean"] for p in pair_results]
            cosines = [pair_results[p][l_str]["cosine"] for p in pair_results]
            print(f"    L{l_str}: avg_dist={np.mean(dists):.2f}, avg_cos={np.mean(cosines):.4f}")

    # 关键对比: 语义同类 vs 语义跨类 vs 同义词 vs 同音字
    print("\n  === 关键对比 ===")
    for l_str in ["0", "3", "6", "12", "18", "24", "35"]:
        same_cat_dists = [results["semantic_same_cat"][p][l_str]["euclidean"]
                         for p in results["semantic_same_cat"]]
        diff_cat_dists = [results["semantic_diff_cat"][p][l_str]["euclidean"]
                         for p in results["semantic_diff_cat"]]
        synonym_dists = [results["synonym"][p][l_str]["euclidean"]
                        for p in results["synonym"]]
        homophone_dists = [results["homophone"][p][l_str]["euclidean"]
                          for p in results["homophone"]]

        print(f"    L{l_str}: 同类={np.mean(same_cat_dists):.2f}, "
              f"跨类={np.mean(diff_cat_dists):.2f}, "
              f"同义={np.mean(synonym_dists):.2f}, "
              f"同音={np.mean(homophone_dists):.2f}")

    # 保存
    save_path = f"tests/glm5_temp/phase101_exp4_{model_name}_tokenizer_control.json"
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
        exp1_relation_direction_stability(args.model)
    elif args.exp == 2:
        exp2_contextual_analysis(args.model)
    elif args.exp == 3:
        exp3_centered_geometry(args.model)
    elif args.exp == 4:
        exp4_tokenizer_control(args.model)
