"""
Phase 98: 语言切换机制与词表归一化分析
========================================
Phase 97批判的核心修正:
  1. "语言切换振荡"可能是tokenizer artifact → 必须做词表语义簇归一化
  2. Cosine=-0.997+Pearson≈0说明贡献向量极度稀疏 → 必须分析分布特性
  3. 只有Qwen3能翻译 → 必须为每个模型设计有效prompt
  4. "L27→L30"不一定是误差修正 → 必须测试多种解释
  5. Residual patching太粗 → 必须做path patching

本Phase目标:
  Exp1: 词表语义簇分析 — 区分"分词动力学"和"语义动力学"
  Exp2: 贡献向量分布分析 — 解决Cosine≈-1+Pearson≈0的统计谜题
  Exp3: Path Patching — 从residual级升级到attention path级
  Exp4: 模型专属prompt验证 — 确认GLM4/DS7B的翻译能力

Run:
  python tests/glm5/ccml_phase98_language_switch.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase98_language_switch.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase98_language_switch.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase98_language_switch.py --model glm4 --exp 4
  python tests/glm5/ccml_phase98_language_switch.py --model deepseek7b --exp 4
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
# 翻译测试对 — 包含语义簇信息
# ============================================================
TRANSLATION_PAIRS = [
    # (中文prompt, en_target, zh_source, en_cluster, zh_cluster)
    # en_cluster: 所有英文翻译token的集合
    # zh_cluster: 所有中文源token的集合
    ("猫的英文是", "cat", "猫", ["cat", "cats", "Cat", "CAT", "_cat"], ["猫", "貓", "猫咪"]),
    ("狗的英文是", "dog", "狗", ["dog", "dogs", "Dog", "DOG"], ["狗", "犬", "小狗"]),
    ("书的英文是", "book", "书", ["book", "books", "Book", "BOOK"], ["书", "書", "本"]),
    ("水的英文是", "water", "水", ["water", "Water", "WATER"], ["水", "液体"]),
    ("火的英文是", "fire", "火", ["fire", "Fire", "FIRE", "fires"], ["火", "火焰"]),
    ("花的英文是", "flower", "花", ["flower", "flowers", "Flower", "blossom"], ["花", "花朵"]),
    ("鱼的英文是", "fish", "鱼", ["fish", "Fish", "FISH", "fishes"], ["鱼", "魚"]),
    ("树的英文是", "tree", "树", ["tree", "trees", "Tree", "TREE"], ["树", "樹"]),
    ("鸟的英文是", "bird", "鸟", ["bird", "birds", "Bird"], ["鸟", "鳥"]),
    ("马的英文是", "horse", "马", ["horse", "horses", "Horse"], ["马", "馬"]),
    ("山的英文是", "mountain", "山", ["mountain", "Mountain", "mount", "hill"], ["山", "山峰"]),
    ("河的英文是", "river", "河", ["river", "River", "stream"], ["河", "河流"]),
    ("铁的英文是", "iron", "铁", ["iron", "Iron", "IRON"], ["铁", "鐵"]),
    ("金的英文是", "gold", "金", ["gold", "Gold", "GOLD", "golden"], ["金", "黄金"]),
    ("茶的英文是", "tea", "茶", ["tea", "Tea", "TEA"], ["茶", "茶叶"]),
    ("米的英文是", "rice", "米", ["rice", "Rice", "RICE"], ["米", "大米"]),
    ("血的英文是", "blood", "血", ["blood", "Blood", "BLOOD"], ["血", "血液"]),
    ("眼的英文是", "eye", "眼", ["eye", "eyes", "Eye", "EYE"], ["眼", "眼睛"]),
    ("手的英文是", "hand", "手", ["hand", "hands", "Hand", "HAND"], ["手", "双手"]),
    ("风的英文是", "wind", "风", ["wind", "Wind", "WIND", "breeze"], ["风", "風"]),
    ("雪的英文是", "snow", "雪", ["snow", "Snow", "SNOW"], ["雪", "雪花"]),
    ("星的英文是", "star", "星", ["star", "stars", "Star", "STAR"], ["星", "星星"]),
    ("海的英文是", "sea", "海", ["sea", "Sea", "SEA", "ocean"], ["海", "大海"]),
    ("石的英文是", "stone", "石", ["stone", "Stone", "rock"], ["石", "石头"]),
    ("草的英文是", "grass", "草", ["grass", "Grass", "lawn"], ["草", "草坪"]),
]

# 模型专属prompt — 每个模型用不同格式测试翻译能力
MODEL_PROMPTS = {
    "qwen3": [
        ("{zh}的英文是", "zh_suffix"),
        ("Translate {zh} to English:", "en_prefix"),
        ("The English word for {zh} is", "en_sentence"),
        ("{zh} in English is", "en_infix"),
    ],
    "glm4": [
        ("{zh}的英文是", "zh_suffix"),
        ("Translate {zh} to English:", "en_prefix"),
        ("The English word for {zh} is", "en_sentence"),
        ("{zh} in English is", "en_infix"),
        ("请将{zh}翻译成英文:", "zh_instruction"),
        ("English translation of {zh}:", "en_translation"),
    ],
    "deepseek7b": [
        ("{zh}的英文是", "zh_suffix"),
        ("Translate {zh} to English:", "en_prefix"),
        ("The English word for {zh} is", "en_sentence"),
        ("{zh} in English is", "en_infix"),
        ("请将{zh}翻译成英文:", "zh_instruction"),
        ("English translation of {zh}:", "en_translation"),
    ],
}


def json_serialize(obj):
    """递归转换numpy类型为python原生类型"""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_token_id(tokenizer, token_str):
    """获取token的id，优先精确匹配"""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if ids:
        return ids[0]
    return None


def get_cluster_ids(tokenizer, cluster_tokens):
    """获取语义簇中所有token的id集合"""
    ids = []
    for t in cluster_tokens:
        tid = get_token_id(tokenizer, t)
        if tid is not None:
            ids.append((t, tid))
    return ids


# ============================================================
# Exp 1: 词表语义簇分析 — 区分"分词动力学"和"语义动力学"
# ============================================================
def exp1_vocabulary_cluster_analysis(model_name):
    """
    核心问题: Phase 97的"语言切换振荡"是语义切换还是tokenizer artifact?
    
    方法:
    1. 对每个翻译对，追踪3类语义簇的概率随层变化:
       - en_cluster: {cat, cats, Cat, _cat, ...} — 英文语义簇
       - zh_cluster: {猫, 貓, 猫咪, ...} — 中文语义簇
       - category: {动物, animal, pet, ...} — 概念簇
    2. 比较"单token轨迹" vs "簇聚合轨迹"
    3. 如果簇聚合后振荡消失 → tokenizer artifact
       如果簇聚合后振荡仍然存在 → 语义切换
    
    这直接回答批判1: "语言切换振荡可能只是tokenizer artifact"
    """
    print(f"\n{'='*60}")
    print(f"Exp 1: 词表语义簇分析 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        sample_layers = sorted(set(
            [0, 1, 2] +
            list(range(0, n_layers, 2)) +
            [n_layers-3, n_layers-2, n_layers-1]
        ))
    print(f"  采样层: {len(sample_layers)} 层")

    pairs = TRANSLATION_PAIRS[:20]  # 20对，数据量足够

    all_results = []

    for pair_idx, (prompt, en_target, zh_source, en_cluster, zh_cluster) in enumerate(pairs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 获取target和source的token ids
        en_target_id = get_token_id(tokenizer, en_target)
        zh_source_id = get_token_id(tokenizer, zh_source)

        if en_target_id is None or zh_source_id is None:
            continue

        # 获取语义簇ids
        en_cluster_ids = get_cluster_ids(tokenizer, en_cluster)
        zh_cluster_ids = get_cluster_ids(tokenizer, zh_cluster)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        W_U = model.lm_head.weight  # [vocab_size, d_model]

        # 逐层分析
        layer_data = {}
        for l in sample_layers:
            h = outputs.hidden_states[l][0, -1, :]
            logits = F.linear(h.to(W_U.device).to(W_U.dtype), W_U)
            probs = F.softmax(logits, dim=-1)

            # 单token概率
            en_single_prob = probs[en_target_id].item()
            zh_single_prob = probs[zh_source_id].item()

            # 语义簇聚合概率
            en_cluster_prob = sum(probs[tid].item() for _, tid in en_cluster_ids)
            zh_cluster_prob = sum(probs[tid].item() for _, tid in zh_cluster_ids)

            # Top-10 tokens
            top10_vals, top10_ids = torch.topk(probs, 10)
            top10_tokens = [(tokenizer.decode([tid]), probs[tid].item()) for tid in top10_ids.tolist()]

            # 分类: 这些top-10 token属于哪个语义空间?
            en_top_prob = sum(p for t, p in top10_tokens if any(et.lower() in t.lower() for et in en_cluster))
            zh_top_prob = sum(p for t, p in top10_tokens if any(zt in t for zt in zh_cluster))
            other_top_prob = 1.0 - en_top_prob - zh_top_prob  # 其他token的概率

            layer_data[str(l)] = {
                "en_single_prob": en_single_prob,
                "zh_single_prob": zh_single_prob,
                "en_cluster_prob": en_cluster_prob,
                "zh_cluster_prob": zh_cluster_prob,
                "en_cluster_coverage": en_cluster_prob / max(en_single_prob, 1e-10),
                "zh_cluster_coverage": zh_cluster_prob / max(zh_single_prob, 1e-10),
                "top10": top10_tokens,
                "semantic_split": {
                    "en": en_top_prob,
                    "zh": zh_top_prob,
                    "other": other_top_prob,
                },
            }

        del outputs

        # 分析: 单token vs 簇聚合 的振荡差异
        # 定义振荡: 簇概率从zh>dominant切换到en>dominant的次数
        zh_single_probs = [layer_data[str(l)]["zh_single_prob"] for l in sample_layers]
        en_single_probs = [layer_data[str(l)]["en_single_prob"] for l in sample_layers]
        zh_cluster_probs = [layer_data[str(l)]["zh_cluster_prob"] for l in sample_layers]
        en_cluster_probs = [layer_data[str(l)]["en_cluster_prob"] for l in sample_layers]

        # 计算振荡次数: dominant language切换次数
        def count_switches(zh_ps, en_ps):
            switches = 0
            for i in range(1, len(zh_ps)):
                prev_zh_dominant = zh_ps[i-1] > en_ps[i-1]
                curr_zh_dominant = zh_ps[i] > en_ps[i]
                if prev_zh_dominant != curr_zh_dominant:
                    switches += 1
            return switches

        single_switches = count_switches(zh_single_probs, en_single_probs)
        cluster_switches = count_switches(zh_cluster_probs, en_cluster_probs)

        # 切换层: 簇聚合后最后一次从zh→en的切换
        last_switch_layer = None
        for i in range(1, len(sample_layers)):
            if zh_cluster_probs[i-1] > en_cluster_probs[i-1] and en_cluster_probs[i] > zh_cluster_probs[i]:
                last_switch_layer = sample_layers[i]

        result = {
            "prompt": prompt,
            "en_target": en_target,
            "zh_source": zh_source,
            "en_cluster_tokens": en_cluster,
            "zh_cluster_tokens": zh_cluster,
            "en_cluster_ids_found": [t for t, _ in en_cluster_ids],
            "zh_cluster_ids_found": [t for t, _ in zh_cluster_ids],
            "layer_data": layer_data,
            "single_token_switches": single_switches,
            "cluster_switches": cluster_switches,
            "last_switch_layer": last_switch_layer,
        }
        all_results.append(result)

        if (pair_idx + 1) % 5 == 0:
            print(f"  已完成 {pair_idx+1}/{len(pairs)} prompts")

    # ---- 聚合分析 ----
    print(f"\n{'='*40}")
    print(f"聚合分析: 单token vs 语义簇")

    # 1. 振荡次数比较
    single_sw = [r["single_token_switches"] for r in all_results]
    cluster_sw = [r["cluster_switches"] for r in all_results]
    print(f"  单token振荡次数: mean={np.mean(single_sw):.2f}, std={np.std(single_sw):.2f}")
    print(f"  语义簇振荡次数: mean={np.mean(cluster_sw):.2f}, std={np.std(cluster_sw):.2f}")
    print(f"  振荡减少比例: {(np.mean(single_sw) - np.mean(cluster_sw)) / max(np.mean(single_sw), 0.01):.2%}")

    if np.mean(single_sw) > np.mean(cluster_sw) + 0.5:
        print(f"  → 部分振荡是tokenizer artifact (簇聚合后减少)")
    elif np.mean(single_sw) <= np.mean(cluster_sw) + 0.5:
        print(f"  → 振荡是语义级别的 (簇聚合后仍存在)")

    # 2. 切换层分布
    switch_layers = [r["last_switch_layer"] for r in all_results if r["last_switch_layer"] is not None]
    if switch_layers:
        print(f"\n  语义切换层: mean={np.mean(switch_layers):.1f}, std={np.std(switch_layers):.1f}")
        print(f"  切换层相对深度: {np.mean(switch_layers)/n_layers:.1%}")

    # 3. 簇覆盖率: en_cluster_prob / en_single_prob
    # 如果覆盖率 >> 1，说明有大量变体token分走了概率
    all_en_coverages = []
    for r in all_results:
        for l in sample_layers:
            cov = r["layer_data"][str(l)]["en_cluster_coverage"]
            if cov < 100:  # 过滤异常值
                all_en_coverages.append(cov)
    if all_en_coverages:
        print(f"\n  英文簇覆盖率: mean={np.mean(all_en_coverages):.2f}x")
        print(f"  (1.0x=只有单token, >1.0x=有变体token分走概率)")

    # 4. 语义空间分裂: 逐层看en/zh/other的比例变化
    print(f"\n  逐层语义分裂:")
    for l in sample_layers[-8:]:
        en_p = np.mean([r["layer_data"][str(l)]["semantic_split"]["en"] for r in all_results])
        zh_p = np.mean([r["layer_data"][str(l)]["semantic_split"]["zh"] for r in all_results])
        other_p = np.mean([r["layer_data"][str(l)]["semantic_split"]["other"] for r in all_results])
        print(f"    L{l}: en={en_p:.3f}, zh={zh_p:.3f}, other={other_p:.3f}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "n_pairs": len(all_results),
        "summary": {
            "single_token_switches_mean": float(np.mean(single_sw)),
            "single_token_switches_std": float(np.std(single_sw)),
            "cluster_switches_mean": float(np.mean(cluster_sw)),
            "cluster_switches_std": float(np.std(cluster_sw)),
            "switch_layers_mean": float(np.mean(switch_layers)) if switch_layers else None,
            "switch_layers_std": float(np.std(switch_layers)) if switch_layers else None,
        },
        "results": all_results,
    }

    outpath = f"tests/glm5_temp/phase98_exp1_{model_name}_vocabulary_cluster.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 2: 贡献向量分布分析 — 解决Cosine≈-1+Pearson≈0谜题
# ============================================================
def exp2_contribution_vector_analysis(model_name):
    """
    Phase 97发现: Cosine=-0.997 + Pearson=-0.07
    批判指出: 这说明贡献向量极度稀疏+重尾，不是"反平行但无关"
    
    本实验:
    1. 分析贡献向量的值分布 (稀疏性、峰度、重尾)
    2. 分析是否是"少数极端值主导cosine"
    3. 去除极端值后重新计算cosine和pearson
    4. 分层看不同层的贡献分布特性
    
    这直接回答批判3: "Cosine≈-1+Pearson≈0有数学问题"
    """
    print(f"\n{'='*60}")
    print(f"Exp 2: 贡献向量分布分析 — {model_name}")
    print(f"{'='*60}")

    # 加载Phase 97 Exp1的结果
    exp1_path = f"tests/glm5_temp/phase97_exp1_{model_name}_matched_head_contribution.json"
    if not os.path.exists(exp1_path):
        print(f"  Phase 97 Exp1结果不存在: {exp1_path}")
        print(f"  请先运行: python tests/glm5/ccml_phase97_computational_primitives.py --model {model_name} --exp 1")
        return None

    with open(exp1_path, "r", encoding="utf-8") as f:
        exp1_data = json.load(f)

    head_contribs = exp1_data["head_contributions"]

    # 构建贡献向量
    trans_vec = []
    comp_vec = []
    labels = []
    for key in sorted(head_contribs.keys()):
        trans_vec.append(head_contribs[key]["translation_drop"])
        comp_vec.append(head_contribs[key]["completion_drop"])
        labels.append(key)

    trans_vec = np.array(trans_vec)
    comp_vec = np.array(comp_vec)

    print(f"  Head数量: {len(trans_vec)}")
    print(f"  翻译贡献: mean={np.mean(trans_vec):.6f}, std={np.std(trans_vec):.6f}")
    print(f"  补全贡献: mean={np.mean(comp_vec):.6f}, std={np.std(comp_vec):.6f}")

    # ---- 1. 基础统计 ----
    print(f"\n[1] 基础统计")

    # 稀疏性: |值| < 阈值的比例
    for threshold in [0.001, 0.01, 0.05]:
        trans_sparse = np.mean(np.abs(trans_vec) < threshold)
        comp_sparse = np.mean(np.abs(comp_vec) < threshold)
        print(f"  |x| < {threshold}: 翻译={trans_sparse:.2%}, 补全={comp_sparse:.2%}")

    # 峰度 (kurtosis > 3 = 重尾)
    from scipy.stats import kurtosis, skew
    trans_kurt = kurtosis(trans_vec)
    comp_kurt = kurtosis(comp_vec)
    trans_skew = skew(trans_vec)
    comp_skew = skew(comp_vec)
    print(f"  翻译峰度: {trans_kurt:.2f} (正态=3, >3=重尾)")
    print(f"  补全峰度: {comp_kurt:.2f}")
    print(f"  翻译偏度: {trans_skew:.2f}")
    print(f"  补全偏度: {comp_skew:.2f}")

    # ---- 2. 极端值分析 ----
    print(f"\n[2] 极端值分析")

    # Top-k极端值
    for k in [1, 5, 10]:
        if k > len(trans_vec):
            continue
        # 翻译贡献最大k个
        top_trans_idx = np.argsort(np.abs(trans_vec))[-k:]
        top_trans_contribution = np.sum(np.abs(trans_vec[top_trans_idx])) / np.sum(np.abs(trans_vec))
        top_comp_idx = np.argsort(np.abs(comp_vec))[-k:]
        top_comp_contribution = np.sum(np.abs(comp_vec[top_comp_idx])) / np.sum(np.abs(comp_vec))
        print(f"  Top-{k} 翻译贡献占总绝对值: {top_trans_contribution:.2%}")
        print(f"  Top-{k} 补全贡献占总绝对值: {top_comp_contribution:.2%}")

    # ---- 3. 去除极端值后重新计算相似度 ----
    print(f"\n[3] 去除极端值后相似度")

    # 全量
    cos_full = np.dot(trans_vec, comp_vec) / (np.linalg.norm(trans_vec) * np.linalg.norm(comp_vec))
    pear_full = np.corrcoef(trans_vec, comp_vec)[0, 1]
    print(f"  全量: Cosine={cos_full:.4f}, Pearson={pear_full:.4f}")

    # 去除|translation_drop|最大的1%
    for pct in [1, 5, 10, 25]:
        threshold_t = np.percentile(np.abs(trans_vec), 100 - pct)
        threshold_c = np.percentile(np.abs(comp_vec), 100 - pct)
        mask = (np.abs(trans_vec) < threshold_t) & (np.abs(comp_vec) < threshold_c)
        if mask.sum() < 5:
            continue
        tv = trans_vec[mask]
        cv = comp_vec[mask]
        nt = np.linalg.norm(tv)
        nc = np.linalg.norm(cv)
        if nt > 0 and nc > 0:
            cos_trim = np.dot(tv, cv) / (nt * nc)
        else:
            cos_trim = 0.0
        pear_trim = np.corrcoef(tv, cv)[0, 1] if np.std(tv) > 0 and np.std(cv) > 0 else 0.0
        print(f"  去除Top{pct}%极端值(n={mask.sum()}): Cosine={cos_trim:.4f}, Pearson={pear_trim:.4f}")

    # ---- 4. 分层分析 ----
    print(f"\n[4] 分层贡献分布")

    layer_stats = {}
    for key in sorted(head_contribs.keys()):
        layer = int(key.split("H")[0][1:])
        if layer not in layer_stats:
            layer_stats[layer] = {"trans": [], "comp": []}
        layer_stats[layer]["trans"].append(head_contribs[key]["translation_drop"])
        layer_stats[layer]["comp"].append(head_contribs[key]["completion_drop"])

    for layer in sorted(layer_stats.keys()):
        lt = np.array(layer_stats[layer]["trans"])
        lc = np.array(layer_stats[layer]["comp"])
        kurt_t = kurtosis(lt) if len(lt) > 3 else 0
        kurt_c = kurtosis(lc) if len(lc) > 3 else 0
        cos_l = np.dot(lt, lc) / (np.linalg.norm(lt) * np.linalg.norm(lc)) if np.linalg.norm(lt) > 0 and np.linalg.norm(lc) > 0 else 0
        pear_l = np.corrcoef(lt, lc)[0, 1] if np.std(lt) > 0 and np.std(lc) > 0 else 0
        print(f"  L{layer}: kurt_t={kurt_t:.1f}, kurt_c={kurt_c:.1f}, cos={cos_l:.3f}, pear={pear_l:.3f}, "
              f"trans_mean={np.mean(lt):.5f}, comp_mean={np.mean(lc):.5f}")

    # ---- 5. 核心结论 ----
    print(f"\n[5] 核心结论")

    # 如果去除极端值后cosine变化大 → cosine被少数值主导
    # 如果去除极端值后pearson变化大 → pearson被极端值扭曲
    mask_10 = (np.abs(trans_vec) < np.percentile(np.abs(trans_vec), 90)) & \
              (np.abs(comp_vec) < np.percentile(np.abs(comp_vec), 90))
    if mask_10.sum() > 5:
        tv10 = trans_vec[mask_10]
        cv10 = comp_vec[mask_10]
        nt10 = np.linalg.norm(tv10)
        nc10 = np.linalg.norm(cv10)
        cos_10 = np.dot(tv10, cv10) / (nt10 * nc10) if nt10 > 0 and nc10 > 0 else 0
        pear_10 = np.corrcoef(tv10, cv10)[0, 1] if np.std(tv10) > 0 and np.std(cv10) > 0 else 0

        if abs(cos_full) > 0.5 and abs(cos_10) < 0.3:
            print(f"  Cosine被少数极端值主导! 去除后从{cos_full:.3f}→{cos_10:.3f}")
            print(f"  → '反平行'是假象，实际是少数head有极端贡献")
        else:
            print(f"  Cosine相对稳定 ({cos_full:.3f}→{cos_10:.3f})")

        if abs(pear_full) < 0.1 and abs(pear_10) > 0.3:
            print(f"  Pearson被极端值扭曲! 去除后从{pear_full:.3f}→{pear_10:.3f}")
            print(f"  → '无关'是假象，去除极端值后head级别有显著相关")
        elif abs(pear_full) < 0.1 and abs(pear_10) < 0.1:
            print(f"  Pearson确实接近0 → head级别贡献确实不同")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_heads": len(trans_vec),
        "full_cosine": float(cos_full),
        "full_pearson": float(pear_full),
        "trans_kurtosis": float(trans_kurt),
        "comp_kurtosis": float(comp_kurt),
        "trans_skew": float(trans_skew),
        "comp_skew": float(comp_skew),
        "trimmed_cosine_10pct": float(cos_10) if mask_10.sum() > 5 else None,
        "trimmed_pearson_10pct": float(pear_10) if mask_10.sum() > 5 else None,
        "conclusion": {
            "cosine_dominated_by_outliers": abs(cos_full) > 0.5 and abs(cos_10) < 0.3 if mask_10.sum() > 5 else None,
            "pearson_distorted_by_outliers": abs(pear_full) < 0.1 and abs(pear_10) > 0.3 if mask_10.sum() > 5 else None,
        },
        "layer_stats": {str(k): {
            "trans_mean": float(np.mean(v["trans"])),
            "comp_mean": float(np.mean(v["comp"])),
            "trans_kurtosis": float(kurtosis(v["trans"])) if len(v["trans"]) > 3 else None,
            "comp_kurtosis": float(kurtosis(v["comp"])) if len(v["comp"]) > 3 else None,
        } for k, v in layer_stats.items()},
    }

    outpath = f"tests/glm5_temp/phase98_exp2_{model_name}_contribution_distribution.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    return output


# ============================================================
# Exp 3: Path Patching — 从residual级升级到attention path级
# ============================================================
def exp3_path_patching(model_name):
    """
    核心升级: 不再patch整个residual stream，而是patch特定的attention path
    
    方法:
    对于翻译任务 "猫的英文是" → cat:
    1. 正常forward: 收集每层的attn output和MLP output
    2. Patch实验: 将source prompt某层的attn output替换到target prompt
    3. 比较: 哪个layer的哪个component (attn vs MLP) 传递了翻译信息?
    
    这回答: 翻译信息是通过attention path还是MLP path传递的?
    
    比residual patching更精细，因为它区分了:
    - attention传递的信息
    - MLP传递的信息
    - 两者的交互
    """
    print(f"\n{'='*60}")
    print(f"Exp 3: Path Patching — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 翻译对
    pairs = TRANSLATION_PAIRS[:10]

    # 采样层 — 重点关注深层
    if n_layers <= 12:
        sample_layers = list(range(n_layers))
    else:
        # 每层都采样，这是path patching的关键实验
        q1 = n_layers // 4
        sample_layers = sorted(set(
            [0, 1, 2] +
            list(range(max(0, q1-3), q1+4)) +
            list(range(max(0, n_layers//2-3), n_layers//2+4)) +
            list(range(max(0, 3*n_layers//4-3), 3*n_layers//4+4)) +
            list(range(n_layers-5, n_layers))
        ))
    print(f"  采样层: {len(sample_layers)} 层")

    # ---- Step 1: 收集所有层的attn和MLP output ----
    print(f"\n[Step 1] 收集source的attn/MLP输出...")

    all_source_outputs = {}  # {pair_idx: {layer: {"attn": h, "mlp": h}}}

    for idx, (prompt, en_target, zh_source, en_cluster, zh_cluster) in enumerate(pairs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 注册hooks收集attn和MLP输出
        layer_outputs = {}
        hooks = []

        layers = get_layers(model)

        for li, layer in enumerate(layers):
            # Attention output hook
            def make_attn_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        layer_outputs[layer_idx] = layer_outputs.get(layer_idx, {})
                        layer_outputs[layer_idx]["attn"] = output[0][0, -1, :].detach().clone()
                    return output
                return hook_fn

            # MLP output hook
            def make_mlp_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        layer_outputs[layer_idx] = layer_outputs.get(layer_idx, {})
                        layer_outputs[layer_idx]["mlp"] = output[0][0, -1, :].detach().clone()
                    else:
                        layer_outputs[layer_idx] = layer_outputs.get(layer_idx, {})
                        layer_outputs[layer_idx]["mlp"] = output[0, -1, :].detach().clone()
                    return output
                return hook_fn

            h1 = layer.self_attn.register_forward_hook(make_attn_hook(li))
            h2 = layer.mlp.register_forward_hook(make_mlp_hook(li))
            hooks.extend([h1, h2])

        with torch.no_grad():
            outputs = model(**inputs)

        # 清理hooks
        for h in hooks:
            h.remove()
        del outputs

        all_source_outputs[idx] = layer_outputs

        if (idx + 1) % 5 == 0:
            print(f"  已完成 {idx+1}/{len(pairs)} source prompts")

    # ---- Step 2: Path Patching ----
    print(f"\n[Step 2] Path Patching: attn vs MLP...")

    # 对每对(i,j): i=source, j=target
    # Patch source的L_s层attn/MLP输出到target的L_s层
    # 测量target的概率变化

    n_test = min(5, len(pairs))

    path_patching_results = {
        "attn": {},   # {L_s: [source_leak_values]}
        "mlp": {},    # {L_s: [source_leak_values]}
        "residual": {}, # baseline: patch整个residual
    }

    for i in range(n_test):
        source_en_id = get_token_id(tokenizer, pairs[i][1])
        if source_en_id is None:
            continue

        for j in range(n_test):
            if i == j:
                continue

            target_prompt = pairs[j][0]
            target_en_id = get_token_id(tokenizer, pairs[j][1])

            # Target baseline
            inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                base_outputs = model(**inputs)
            base_prob = F.softmax(base_outputs.logits[0, -1, :], dim=-1)[source_en_id].item()
            del base_outputs

            for l_s in sample_layers:
                if l_s not in all_source_outputs[i]:
                    continue
                if "attn" not in all_source_outputs[i][l_s] or "mlp" not in all_source_outputs[i][l_s]:
                    continue

                source_attn = all_source_outputs[i][l_s]["attn"]
                source_mlp = all_source_outputs[i][l_s]["mlp"]

                layers = get_layers(model)

                # --- Patch Attention ---
                def make_attn_patch_hook(src_attn_h):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            patched = output[0].clone()
                            # 只替换最后一个token位置
                            patched[0, -1, :] = patched[0, -1, :] + src_attn_h.to(patched.device).to(patched.dtype)
                            return (patched,) + output[1:]
                        return output
                    return hook_fn

                h = layers[l_s].self_attn.register_forward_hook(make_attn_patch_hook(source_attn))
                inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                attn_prob = F.softmax(outputs.logits[0, -1, :], dim=-1)[source_en_id].item()
                h.remove()
                del outputs

                # --- Patch MLP ---
                def make_mlp_patch_hook(src_mlp_h):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            patched = output[0].clone()
                            patched[0, -1, :] = patched[0, -1, :] + src_mlp_h.to(patched.device).to(patched.dtype)
                            return (patched,) + output[1:]
                        elif isinstance(output, torch.Tensor):
                            patched = output.clone()
                            patched[0, -1, :] = patched[0, -1, :] + src_mlp_h.to(patched.device).to(patched.dtype)
                            return patched
                        return output
                    return hook_fn

                h = layers[l_s].mlp.register_forward_hook(make_mlp_patch_hook(source_mlp))
                inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                mlp_prob = F.softmax(outputs.logits[0, -1, :], dim=-1)[source_en_id].item()
                h.remove()
                del outputs

                # 记录
                attn_leak = attn_prob - base_prob
                mlp_leak = mlp_prob - base_prob

                if l_s not in path_patching_results["attn"]:
                    path_patching_results["attn"][l_s] = []
                    path_patching_results["mlp"][l_s] = []
                    path_patching_results["residual"][l_s] = []
                path_patching_results["attn"][l_s].append(attn_leak)
                path_patching_results["mlp"][l_s].append(mlp_leak)
                # 残差 = attn + mlp 近似
                path_patching_results["residual"][l_s].append(attn_leak + mlp_leak)

        print(f"  完成source {i+1}/{n_test}")

    # ---- Step 3: 聚合分析 ----
    print(f"\n[Step 3] Path Patching聚合分析")

    print(f"\n  {'Layer':>6} {'Attn Leak':>12} {'MLP Leak':>12} {'Residual':>12} {'Dominant':>10}")
    print(f"  {'-'*55}")

    attn_dominant_layers = []
    mlp_dominant_layers = []

    for l in sorted(path_patching_results["attn"].keys()):
        attn_mean = np.mean(path_patching_results["attn"][l])
        mlp_mean = np.mean(path_patching_results["mlp"][l])
        res_mean = np.mean(path_patching_results["residual"][l])

        if abs(attn_mean) > abs(mlp_mean):
            dominant = "Attn"
            attn_dominant_layers.append(l)
        else:
            dominant = "MLP"
            mlp_dominant_layers.append(l)

        if abs(attn_mean) > 0.001 or abs(mlp_mean) > 0.001:
            print(f"  L{l:>4} {attn_mean:>12.5f} {mlp_mean:>12.5f} {res_mean:>12.5f} {dominant:>10}")

    print(f"\n  Attn主导层数: {len(attn_dominant_layers)}")
    print(f"  MLP主导层数: {len(mlp_dominant_layers)}")

    # 找关键层: leak最大的层
    all_layers_leak = {}
    for l in path_patching_results["attn"]:
        all_layers_leak[l] = max(abs(np.mean(path_patching_results["attn"][l])),
                                abs(np.mean(path_patching_results["mlp"][l])))

    top_layers = sorted(all_layers_leak.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top-10 关键层:")
    for l, leak in top_layers:
        attn_l = np.mean(path_patching_results["attn"][l])
        mlp_l = np.mean(path_patching_results["mlp"][l])
        path = "Attn" if abs(attn_l) > abs(mlp_l) else "MLP"
        print(f"    L{l}: leak={leak:.5f}, path={path}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "sample_layers": sorted(path_patching_results["attn"].keys()),
        "n_test_pairs": n_test,
        "attn_dominant_count": len(attn_dominant_layers),
        "mlp_dominant_count": len(mlp_dominant_layers),
        "layer_results": {},
    }

    for l in sorted(path_patching_results["attn"].keys()):
        output["layer_results"][str(l)] = {
            "attn_mean_leak": float(np.mean(path_patching_results["attn"][l])),
            "mlp_mean_leak": float(np.mean(path_patching_results["mlp"][l])),
            "attn_std_leak": float(np.std(path_patching_results["attn"][l])),
            "mlp_std_leak": float(np.std(path_patching_results["mlp"][l])),
            "n_samples": len(path_patching_results["attn"][l]),
        }

    outpath = f"tests/glm5_temp/phase98_exp3_{model_name}_path_patching.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Exp 4: 模型专属prompt验证
# ============================================================
def exp4_model_specific_prompts(model_name):
    """
    Phase 97发现: 只有Qwen3能翻译"X的英文是"
    批判指出: 这是prompt mismatch，不是翻译能力缺失
    
    本实验:
    对GLM4和DS7B测试多种翻译prompt格式，找到能触发翻译的prompt
    
    这直接回答批判4: "只有Qwen3能翻译是prompt mismatch"
    """
    print(f"\n{'='*60}")
    print(f"Exp 4: 模型专属prompt验证 — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  模型: {model_name}, 层数: {n_layers}")

    # 测试实体
    test_entities = [
        ("猫", "cat"), ("狗", "dog"), ("书", "book"),
        ("水", "water"), ("火", "fire"), ("花", "flower"),
        ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
        ("马", "horse"), ("铁", "iron"), ("金", "gold"),
        ("茶", "tea"), ("米", "rice"), ("血", "blood"),
    ]

    prompt_formats = MODEL_PROMPTS.get(model_name, MODEL_PROMPTS["qwen3"])

    results = {}

    for fmt_desc, fmt_name in prompt_formats:
        print(f"\n  测试prompt格式: {fmt_name} = '{fmt_desc}'")

        correct_top5 = 0
        correct_top1 = 0
        total = 0
        en_target_probs = []

        for zh, en in test_entities:
            prompt = fmt_desc.format(zh=zh)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Top-10 (用更多候选以确保覆盖变体)
            top10_vals, top10_ids = torch.topk(probs, 10)
            top10_tokens = [tokenizer.decode([tid]).strip().lower() for tid in top10_ids.tolist()]

            # Debug: 第一个prompt打印详情
            if total == 0:
                print(f"    DEBUG: prompt='{prompt}', top1='{tokenizer.decode([top10_ids[0].item()])}' "
                      f"({top10_vals[0].item():.4f}), top3={[(tokenizer.decode([tid]), f'{probs[tid].item():.4f}') for tid in top10_ids[:3].tolist()]}")

            # 语义匹配: top-k中是否包含目标词
            en_lower = en.lower()
            top5_tokens = top10_tokens[:5]
            
            if en_lower in top5_tokens[0]:
                correct_top1 += 1
                correct_top5 += 1
            elif any(en_lower in t for t in top5_tokens):
                correct_top5 += 1

            # 概率: 找所有匹配en的token的概率之和
            en_cluster_prob = 0.0
            for i, tid in enumerate(top10_ids.tolist()):
                decoded = tokenizer.decode([tid]).strip().lower()
                if en_lower in decoded:
                    en_cluster_prob += probs[tid].item()
            en_target_probs.append(en_cluster_prob)

            total += 1
            del outputs

        accuracy_top5 = correct_top5 / total if total > 0 else 0
        accuracy_top1 = correct_top1 / total if total > 0 else 0
        mean_en_prob = np.mean(en_target_probs)

        results[fmt_name] = {
            "prompt_template": fmt_desc,
            "accuracy_top5": accuracy_top5,
            "accuracy_top1": accuracy_top1,
            "mean_en_cluster_prob": mean_en_prob,
            "n_tested": total,
        }

        print(f"    Top-5准确率: {accuracy_top5:.2%}")
        print(f"    Top-1准确率: {accuracy_top1:.2%}")
        print(f"    en_cluster平均概率: {mean_en_prob:.4f}")

    # 找最佳prompt
    best_fmt = max(results, key=lambda k: results[k]["accuracy_top5"])
    print(f"\n  最佳prompt格式: {best_fmt}")
    print(f"  Top-5准确率: {results[best_fmt]['accuracy_top5']:.2%}")
    print(f"  Top-1准确率: {results[best_fmt]['accuracy_top1']:.2%}")

    # ---- 保存 ----
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "results": results,
        "best_prompt_format": best_fmt,
    }

    outpath = f"tests/glm5_temp/phase98_exp4_{model_name}_model_prompts.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_serialize(output), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")

    release_model(model)
    return output


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_vocabulary_cluster_analysis(args.model)
    elif args.exp == 2:
        exp2_contribution_vector_analysis(args.model)
    elif args.exp == 3:
        exp3_path_patching(args.model)
    elif args.exp == 4:
        exp4_model_specific_prompts(args.model)
