"""
Phase 134: 条件激活边界分析 — 从算子几何转向计算图分析
=======================================================

Phase 133核心发现:
1. Jacobian秩被探针上限严重低估 (k=256时仍不饱和)
2. 深层Jacobian不存在稳定值 (cos<0.4)
3. "条件算子几何"在深层不适用

批评的核心洞察:
- 不应研究J_l(x), 而应研究"什么决定了哪些神经元被激活"
- 这是"条件计算理论": 输入 → 条件激活 → 稀疏路由 → 局部低秩编辑 → 输出
- 语言的数学结构可能在于"条件激活边界"

但批评也有问题:
- "放弃连续微分几何"过于绝对 — L0层Jacobian是稳定的
- "非线性路由"的跳跃需要验证 — Jacobian不稳定有多种可能原因
- "bfloat16≠数学Transformer"正确但无法解决 — 我们只能研究实际模型

Phase 134策略: 在测量可靠的地方用几何方法, 在不可靠的地方转向激活模式
- Exp 1: SwiGLU激活稀疏性 — 各层各头的激活比例, 验证条件计算假设
- Exp 2: 约束效应对激活模式的影响 — 否定/时态/被动是否改变激活集合
- Exp 3: 激活边界的几何结构 — 在L0层(几何可靠)测激活边界的局部曲率
- Exp 4: 头功能聚类 — 不同语言操作是否由不同头组合实现

方法论:
- 使用hook捕获SwiGLU gate输出, 计算激活比例
- 使用attention pattern提取头的激活模式
- 在ε=0.01-0.1(可靠范围)内测量激活边界的几何
- 大量句子(30+), 避免小数据偏差
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 句子设计 — 扩大规模, 30+句子
# ============================================================

# 句子对: 每组有 base/negation/past 三种变体
SENTENCE_PAIRS = [
    {"s": "dog", "v_base": "bites", "v_past": "bit", "o": "man"},
    {"s": "cat", "v_base": "chases", "v_past": "chased", "o": "mouse"},
    {"s": "teacher", "v_base": "helps", "v_past": "helped", "o": "student"},
    {"s": "doctor", "v_base": "treats", "v_past": "treated", "o": "patient"},
    {"s": "chef", "v_base": "cooks", "v_past": "cooked", "o": "meal"},
    {"s": "writer", "v_base": "writes", "v_past": "wrote", "o": "book"},
    {"s": "child", "v_base": "plays", "v_past": "played", "o": "game"},
    {"s": "farmer", "v_base": "grows", "v_past": "grew", "o": "crop"},
    {"s": "artist", "v_base": "paints", "v_past": "painted", "o": "picture"},
    {"s": "driver", "v_base": "drives", "v_past": "drove", "o": "car"},
    {"s": "bird", "v_base": "flies", "v_past": "flew", "o": "nest"},
    {"s": "fish", "v_base": "swims", "v_past": "swam", "o": "river"},
    {"s": "soldier", "v_base": "guards", "v_past": "guarded", "o": "fort"},
    {"s": "nurse", "v_base": "cares", "v_past": "cared", "o": "elder"},
    {"s": "scientist", "v_base": "studies", "v_past": "studied", "o": "atom"},
]

# 语义变体: 同一语法结构, 不同语义内容
SEMANTIC_PAIRS = [
    ("The dog bites the man", "The cat chases the mouse"),
    ("The teacher helps the student", "The doctor treats the patient"),
    ("The chef cooks the meal", "The writer writes the book"),
    ("The child plays the game", "The farmer grows the crop"),
    ("The artist paints the picture", "The driver drives the car"),
    ("The bird flies to the nest", "The fish swims in the river"),
    ("The soldier guards the fort", "The nurse cares for the elder"),
    ("The scientist studies the atom", "The student reads the paper"),
    ("The king rules the kingdom", "The queen leads the army"),
    ("The fire burns the forest", "The water floods the valley"),
]


def make_sentence(entry, variant="base"):
    """生成语法正确的句子变体"""
    s = entry["s"]
    v = entry["v_base"]
    v_past = entry["v_past"]
    o = entry["o"]

    # 提取动词词干
    if v.endswith("ies"):
        v_stem = v[:-3] + "y"
    elif v.endswith("shes") or v.endswith("ches") or v.endswith("xes"):
        v_stem = v[:-2]
    elif v.endswith("sses"):
        v_stem = v[:-2]
    else:
        v_stem = v[:-1] if v.endswith("s") and not v.endswith("ss") else v

    if variant == "base":
        return f"The {s} {v} the {o}"
    elif variant == "negation":
        return f"The {s} does not {v_stem} the {o}"
    elif variant == "past":
        return f"The {s} {v_past} the {o}"
    elif variant == "passive":
        return f"The {o} is {entry.get('v_pp', v_stem + 'ed')} by the {s}"
    else:
        return f"The {s} {v} the {o}"


# ============================================================
# 工具函数
# ============================================================

def get_device_for_input(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_effective_rank(singular_values, threshold=0.99):
    """计算有效秩"""
    total = np.sum(singular_values ** 2)
    if total < 1e-20:
        return 0
    cumsum = np.cumsum(singular_values ** 2)
    rank = np.searchsorted(cumsum / total, threshold) + 1
    return int(min(rank, len(singular_values)))


def jaccard_similarity(set_a, set_b):
    """计算两个集合的Jaccard相似度"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


# ============================================================
# Exp 1: SwiGLU激活稀疏性分析
# ============================================================

def exp1_activation_sparsity(model, tokenizer, device, model_info):
    """
    分析SwiGLU gate的激活模式:
    - 各层各位置的激活比例
    - 不同句子变体(base/neg/past)的激活差异
    - 激活模式的稳定性

    SwiGLU: gate = σ(W_gate · x), output = gate ⊙ (W_up · x)
    激活 = gate > threshold 的神经元集合
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # 获取MLP中间层维度
    layer0 = layers[0]
    if hasattr(layer0.mlp, 'gate_up_proj'):
        intermediate_size = layer0.mlp.gate_up_proj.weight.shape[0] // 2
    elif hasattr(layer0.mlp, 'gate_proj'):
        intermediate_size = layer0.mlp.gate_proj.weight.shape[0]
    else:
        intermediate_size = d_model * 4

    # 采样层
    sample_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    sample_names = [f"L{i}" for i in sample_indices]

    gate_threshold = 0.05  # MLP输出 |val| > 0.05 视为活跃贡献

    results = {"per_sentence": {}, "intermediate_size": intermediate_size}

    # 只用前8个句子(3变体 = 24句子)
    for si, entry in enumerate(SENTENCE_PAIRS[:8]):
        for variant in ["base", "negation", "past"]:
            sent = make_sentence(entry, variant)
            key = f"{si}_{variant}"
            print(f"  [{si+1}/8 {variant}] {sent[:40]}")

            # Hook: 捕获gate输出
            captured_gates = {}

            def make_gate_hook(layer_idx):
                def hook(module, input, output):
                    # SwiGLU: output = silu(gate) * up
                    # 我们想捕获gate值, 但hook看到的是最终输出
                    # 改为hook到gate_proj的输出(如果存在)
                    captured_gates[layer_idx] = output[0].detach().float().cpu()
                return hook

            # Hook到MLP整体, 获取中间表示
            hooks = []
            for li in sample_indices:
                if hasattr(layers[li], 'mlp'):
                    hooks.append(layers[li].mlp.register_forward_hook(make_gate_hook(li)))

            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            for h in hooks:
                h.remove()

            # 分析gate激活
            sent_result = {}
            for li in sample_indices:
                if li not in captured_gates:
                    continue

                # MLP输出: 可能是 [1, seq_len, d_model] 或 [seq_len, d_model]
                mlp_out = captured_gates[li]
                if mlp_out.dim() == 3:
                    last_token_out = mlp_out[0, -1, :].numpy()
                elif mlp_out.dim() == 2:
                    last_token_out = mlp_out[-1, :].numpy()
                else:
                    last_token_out = mlp_out.numpy()

                # 计算激活统计
                activation_ratio = float(np.mean(np.abs(last_token_out) > gate_threshold))
                activation_norm = float(np.linalg.norm(last_token_out))
                active_dims = int(np.sum(np.abs(last_token_out) > gate_threshold))

                # Top-K激活维度
                top_k = min(20, len(last_token_out))
                top_dims = np.argsort(np.abs(last_token_out))[-top_k:][::-1].tolist()
                top_vals = np.abs(last_token_out)[top_dims].tolist()

                # 稀疏度 (Gini系数)
                sorted_abs = np.sort(np.abs(last_token_out))
                n = len(sorted_abs)
                cumsum = np.cumsum(sorted_abs)
                gini = float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n) if cumsum[-1] > 0 else 0

                lk = f"L{li}"
                sent_result[lk] = {
                    "activation_ratio": activation_ratio,
                    "active_dims": active_dims,
                    "activation_norm": activation_norm,
                    "gini": gini,
                    "top_dims": top_dims,
                    "top_vals": top_vals,
                }

            results["per_sentence"][key] = {
                "sentence": sent,
                "variant": variant,
                "layers": sent_result,
            }

            # 释放GPU缓存
            del captured_gates, out
            gc.collect()

    # 汇总: 各层各变体的平均激活比例
    summary = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]
        variant_stats = defaultdict(lambda: {"ratios": [], "gini": [], "norms": []})
        for key, data in results["per_sentence"].items():
            variant = data["variant"]
            if lk in data["layers"]:
                ld = data["layers"][lk]
                variant_stats[variant]["ratios"].append(ld["activation_ratio"])
                variant_stats[variant]["gini"].append(ld["gini"])
                variant_stats[variant]["norms"].append(ld["activation_norm"])

        summary[lk] = {}
        for variant, stats in variant_stats.items():
            summary[lk][variant] = {
                "mean_ratio": float(np.mean(stats["ratios"])),
                "std_ratio": float(np.std(stats["ratios"])),
                "mean_gini": float(np.mean(stats["gini"])),
                "mean_norm": float(np.mean(stats["norms"])),
            }

    results["summary"] = summary

    # 变体间激活差异: 同一句子base vs neg/past的Jaccard
    variant_diff = {}
    for si in range(min(8, len(SENTENCE_PAIRS))):
        base_key = f"{si}_base"
        neg_key = f"{si}_negation"
        past_key = f"{si}_past"

        if base_key not in results["per_sentence"]:
            continue

        for li_idx, li in enumerate(sample_indices):
            lk = sample_names[li_idx]

            base_dims = set()
            neg_dims = set()
            past_dims = set()

            if lk in results["per_sentence"][base_key]["layers"]:
                base_dims = set(results["per_sentence"][base_key]["layers"][lk]["top_dims"])
            if neg_key in results["per_sentence"] and lk in results["per_sentence"][neg_key]["layers"]:
                neg_dims = set(results["per_sentence"][neg_key]["layers"][lk]["top_dims"])
            if past_key in results["per_sentence"] and lk in results["per_sentence"][past_key]["layers"]:
                past_dims = set(results["per_sentence"][past_key]["layers"][lk]["top_dims"])

            if lk not in variant_diff:
                variant_diff[lk] = {"base_vs_neg": [], "base_vs_past": []}

            variant_diff[lk]["base_vs_neg"].append(jaccard_similarity(base_dims, neg_dims))
            variant_diff[lk]["base_vs_past"].append(jaccard_similarity(base_dims, past_dims))

    # 平均
    for lk, vd in variant_diff.items():
        for pair_key in ["base_vs_neg", "base_vs_past"]:
            if vd[pair_key]:
                vd[f"{pair_key}_mean"] = float(np.mean(vd[pair_key]))
                vd[f"{pair_key}_std"] = float(np.std(vd[pair_key]))

    results["variant_jaccard"] = variant_diff
    return results


# ============================================================
# Exp 2: Attention Head激活模式分析
# ============================================================

def exp2_attention_patterns(model, tokenizer, device, model_info):
    """
    分析注意力头的激活模式:
    - 各头在各层的attention entropy
    - 不同变体(base/neg/past)的attention差异
    - 头的功能聚类

    方法: output_attentions=True获取attention weights
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # 获取头数
    layer0 = layers[0]
    n_heads = layer0.self_attn.config.num_attention_heads if hasattr(layer0.self_attn, 'config') else d_model // 64
    head_dim = d_model // n_heads

    sample_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    sample_names = [f"L{i}" for i in sample_indices]

    results = {"per_sentence": {}, "n_heads": n_heads, "head_dim": head_dim}

    for si, entry in enumerate(SENTENCE_PAIRS[:10]):
        for variant in ["base", "negation", "past"]:
            sent = make_sentence(entry, variant)
            key = f"{si}_{variant}"
            print(f"  [{si+1}/10 {variant}] {sent[:40]}")

            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=True)

            # out.attentions: tuple of (n_layers,), each [1, n_heads, seq_len, seq_len]
            sent_result = {}
            for li_idx, li in enumerate(sample_indices):
                lk = sample_names[li_idx]

                if li >= len(out.attentions) or out.attentions[li] is None:
                    continue

                attn = out.attentions[li][0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
                seq_len = attn.shape[1]
                last_pos = seq_len - 1

                # 各头的last token attention pattern
                head_patterns = []
                head_entropies = []
                head_max_weights = []
                head_sparsity = []

                for h in range(min(n_heads, attn.shape[0])):
                    pattern = attn[h, last_pos, :]  # [seq_len] — last token对其他token的attention

                    # 归一化(应该已经归一化了)
                    pat_sum = pattern.sum()
                    if pat_sum > 0:
                        pattern = pattern / pat_sum

                    # Entropy
                    entropy = -np.sum(pattern * np.log(pattern + 1e-10))
                    head_entropies.append(float(entropy))

                    # Max weight
                    head_max_weights.append(float(np.max(pattern)))

                    # Sparsity: 前3个位置的权重占比
                    top3 = np.sort(pattern)[-3:]
                    head_sparsity.append(float(np.sum(top3)))

                    head_patterns.append(pattern.tolist())

                # 头间相似度矩阵
                patterns_arr = np.array(head_patterns)  # [n_heads, seq_len]
                if patterns_arr.shape[0] > 1:
                    # 余弦相似度
                    norms = np.linalg.norm(patterns_arr, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    normalized = patterns_arr / norms
                    head_sim = normalized @ normalized.T  # [n_heads, n_heads]

                    # 平均off-diagonal相似度
                    mask = ~np.eye(head_sim.shape[0], dtype=bool)
                    mean_inter_head_sim = float(np.mean(head_sim[mask]))

                    # 头聚类: 用相似度矩阵的特征值
                    eigvals = np.linalg.eigvalsh(head_sim)
                    eigvals = np.sort(eigvals)[::-1]
                    head_cluster_rank = compute_effective_rank(eigvals, 0.95)
                else:
                    mean_inter_head_sim = 0
                    head_cluster_rank = 1
                    eigvals = np.array([1.0])

                sent_result[lk] = {
                    "head_entropies": head_entropies,
                    "head_max_weights": head_max_weights,
                    "head_sparsity_top3": head_sparsity,
                    "mean_entropy": float(np.mean(head_entropies)),
                    "mean_max_weight": float(np.mean(head_max_weights)),
                    "mean_inter_head_sim": mean_inter_head_sim,
                    "head_cluster_rank": head_cluster_rank,
                    "top_eigvals": eigvals[:min(8, len(eigvals))].tolist(),
                }

            results["per_sentence"][key] = {
                "sentence": sent,
                "variant": variant,
                "layers": sent_result,
            }

            del out
            gc.collect()

    # 汇总: 变体间的attention差异
    variant_diff = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]

        # 收集各变体的头模式
        base_patterns = []
        neg_patterns = []
        past_patterns = []

        for si in range(min(10, len(SENTENCE_PAIRS))):
            for variant, pattern_list in [("base", base_patterns), ("negation", neg_patterns), ("past", past_patterns)]:
                key = f"{si}_{variant}"
                if key in results["per_sentence"] and lk in results["per_sentence"][key]["layers"]:
                    ld = results["per_sentence"][key]["layers"][lk]
                    pattern_list.append({
                        "entropy": ld["mean_entropy"],
                        "cluster_rank": ld["head_cluster_rank"],
                        "inter_head_sim": ld["mean_inter_head_sim"],
                    })

        variant_diff[lk] = {
            "base_mean_entropy": float(np.mean([p["entropy"] for p in base_patterns])) if base_patterns else 0,
            "neg_mean_entropy": float(np.mean([p["entropy"] for p in neg_patterns])) if neg_patterns else 0,
            "past_mean_entropy": float(np.mean([p["entropy"] for p in past_patterns])) if past_patterns else 0,
            "base_cluster_rank": float(np.mean([p["cluster_rank"] for p in base_patterns])) if base_patterns else 0,
            "neg_cluster_rank": float(np.mean([p["cluster_rank"] for p in neg_patterns])) if neg_patterns else 0,
            "past_cluster_rank": float(np.mean([p["cluster_rank"] for p in past_patterns])) if past_patterns else 0,
        }

    results["variant_diff"] = variant_diff
    return results


# ============================================================
# Exp 3: 约束效应对hidden state的激活边界
# ============================================================

def exp3_constraint_activation_boundary(model, tokenizer, device, model_info):
    """
    在L0层(几何可靠)测量约束效应如何改变激活模式

    关键问题:
    - 否定/时态操作是否改变哪些维度被"激活"?
    - 约束效应是否落在低维子空间(rank 1-3)?
    - 不同约束的效应子空间是否正交?

    方法:
    1. 对base/negation/past句子获取L0层hidden state
    2. 计算 Δh_neg = h(neg) - h(base), Δh_past = h(past) - h(base)
    3. 分析Δh的秩和子空间结构
    4. 用ε=0.01(可靠范围)验证约束效应的稳定性
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # L0=embedding输出(无计算), L1=经过1层后, L2=经过2层后
    # 用L1(有初步计算)和L_mid(有丰富计算)
    test_layers = [1, n_layers // 2]
    test_layer_names = ["L1", f"L{n_layers//2}"]

    results = {"per_pair": {}}

    for si, entry in enumerate(SENTENCE_PAIRS[:15]):
        print(f"  [{si+1}/15] {entry['s']}/{entry['v_base']}/{entry['o']}")

        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        # 获取hidden states
        def get_hs(prompt):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            hs = [h[0, -1, :].detach().float().cpu().numpy() for h in out.hidden_states]
            del out
            return hs

        hs_base = get_hs(sent_base)
        hs_neg = get_hs(sent_neg)
        hs_past = get_hs(sent_past)

        pair_result = {}
        for li_idx, li in enumerate(test_layers):
            lk = test_layer_names[li_idx]

            h_base = hs_base[li]
            h_neg = hs_neg[li]
            h_past = hs_past[li]

            # 约束效应
            delta_neg = h_neg - h_base
            delta_past = h_past - h_base

            # 基本统计
            delta_neg_norm = float(np.linalg.norm(delta_neg))
            delta_past_norm = float(np.linalg.norm(delta_past))
            h_base_norm = float(np.linalg.norm(h_base))

            # 约束效应方向
            cos_neg_past = float(np.dot(delta_neg, delta_past) /
                                max(np.linalg.norm(delta_neg) * np.linalg.norm(delta_past), 1e-10))

            # 约束效应占base的相对大小
            rel_neg = delta_neg_norm / max(h_base_norm, 1e-10)
            rel_past = delta_past_norm / max(h_base_norm, 1e-10)

            # 约束效应在哪些维度上最强?
            top_neg_dims = np.argsort(np.abs(delta_neg))[-10:][::-1].tolist()
            top_past_dims = np.argsort(np.abs(delta_past))[-10:][::-1].tolist()

            # Top维度重叠度
            top_neg_set = set(top_neg_dims[:5])
            top_past_set = set(top_past_dims[:5])
            dim_overlap = len(top_neg_set & top_past_set) / 5.0

            pair_result[lk] = {
                "delta_neg_norm": delta_neg_norm,
                "delta_past_norm": delta_past_norm,
                "h_base_norm": h_base_norm,
                "cos_neg_past": cos_neg_past,
                "rel_neg": rel_neg,
                "rel_past": rel_past,
                "top_neg_dims": top_neg_dims,
                "top_past_dims": top_past_dims,
                "dim_overlap_ratio": dim_overlap,
            }

        results["per_pair"][si] = pair_result

        gc.collect()

    # 跨句子汇总: 约束效应的子空间结构
    summary = {}
    for li_idx, li in enumerate(test_layers):
        lk = test_layer_names[li_idx]

        # 收集所有Δh向量
        delta_negs = []
        delta_pasts = []
        for si, pair_data in results["per_pair"].items():
            if lk in pair_data:
                # 需要重新计算Δh... 但我们只存了范数
                # 改用统计
                pass

        # 统计汇总
        norms_neg = [results["per_pair"][si][lk]["delta_neg_norm"]
                     for si in results["per_pair"] if lk in results["per_pair"][si]]
        norms_past = [results["per_pair"][si][lk]["delta_past_norm"]
                      for si in results["per_pair"] if lk in results["per_pair"][si]]
        cos_vals = [results["per_pair"][si][lk]["cos_neg_past"]
                    for si in results["per_pair"] if lk in results["per_pair"][si]]
        rel_negs = [results["per_pair"][si][lk]["rel_neg"]
                    for si in results["per_pair"] if lk in results["per_pair"][si]]
        rel_pasts = [results["per_pair"][si][lk]["rel_past"]
                     for si in results["per_pair"] if lk in results["per_pair"][si]]
        overlaps = [results["per_pair"][si][lk]["dim_overlap_ratio"]
                    for si in results["per_pair"] if lk in results["per_pair"][si]]

        summary[lk] = {
            "mean_delta_neg_norm": float(np.mean(norms_neg)),
            "mean_delta_past_norm": float(np.mean(norms_past)),
            "mean_cos_neg_past": float(np.mean(cos_vals)),
            "mean_rel_neg": float(np.mean(rel_negs)),
            "mean_rel_past": float(np.mean(rel_pasts)),
            "mean_dim_overlap": float(np.mean(overlaps)),
        }

    results["summary"] = summary

    # 约束效应的子空间秩 (需要收集所有Δh向量)
    # 第二遍: 专门收集L1层的Δh向量
    print("  Collecting delta vectors for subspace analysis...")
    delta_neg_vectors = []
    delta_past_vectors = []

    for si, entry in enumerate(SENTENCE_PAIRS[:15]):
        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        def get_hs_l1(prompt):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            # hidden_states[1] = 经过第0层后的输出 = L1
            hs = out.hidden_states[1][0, -1, :].detach().float().cpu().numpy()
            del out
            return hs

        h_base = get_hs_l1(sent_base)
        h_neg = get_hs_l1(sent_neg)
        h_past = get_hs_l1(sent_past)

        delta_neg_vectors.append(h_neg - h_base)
        delta_past_vectors.append(h_past - h_base)

        gc.collect()

    # Δh_neg的子空间结构
    delta_neg_mat = np.array(delta_neg_vectors)  # [15, d_model]
    delta_past_mat = np.array(delta_past_vectors)  # [15, d_model]

    # SVD
    U_neg, S_neg, Vh_neg = np.linalg.svd(delta_neg_mat, full_matrices=False)
    U_past, S_past, Vh_past = np.linalg.svd(delta_past_mat, full_matrices=False)

    # 有效秩
    rank_neg = compute_effective_rank(S_neg, 0.99)
    rank_past = compute_effective_rank(S_past, 0.99)

    # 子空间对齐
    # Vh_neg的行是Δh_neg的主方向
    # Vh_past的行是Δh_past的主方向
    top_neg = Vh_neg[:min(rank_neg, 10)].T  # [d_model, k_neg]
    top_past = Vh_past[:min(rank_past, 10)].T  # [d_model, k_past]

    P_neg = top_neg @ top_neg.T
    P_past = top_past @ top_past.T
    trace_overlap = np.trace(P_neg @ P_past)
    norm1 = np.sqrt(np.trace(P_neg @ P_neg))
    norm2 = np.sqrt(np.trace(P_past @ P_past))
    subspace_cos = float(trace_overlap / max(norm1 * norm2, 1e-10))

    results["subspace_analysis"] = {
        "L1": {
            "neg_rank": rank_neg,
            "past_rank": rank_past,
            "neg_sv": S_neg[:min(10, len(S_neg))].tolist(),
            "past_sv": S_past[:min(10, len(S_past))].tolist(),
            "subspace_cosine": subspace_cos,
        }
    }

    return results


# ============================================================
# Exp 4: ε稳定区间内的约束投影几何
# ============================================================

def exp4_constraint_projection_stability(model, tokenizer, device, model_info):
    """
    在ε=0.01-0.1(Phase 133验证的可靠范围)内:
    测量约束投影 P_l(x, c) = J_l(x) · δh_c 的稳定性

    关键问题:
    - 约束投影P是否ε无关?
    - 不同约束的投影P是否正交?
    - 约束投影的秩是否真的是1-3?

    方法:
    1. 对base句子获取L0层h_base
    2. 用ε=0.01和ε=0.1分别扰动, 获取J_l(x)的列估计
    3. 对neg/past句子获取δh_c = h(neg) - h(base)
    4. 计算约束投影 P = J · δh_c
    5. 验证P在ε=0.01和ε=0.1下的一致性
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 64  # 扰动方向数

    eps_values = [0.01, 0.05, 0.1]  # Phase 133验证的可靠范围

    test_layers = [1, n_layers // 4]  # L1(可靠几何)和L_{1/4}(早期深层)
    test_layer_names = ["L1", f"L{n_layers//4}"]

    results = {"per_pair": {}}

    for si, entry in enumerate(SENTENCE_PAIRS[:8]):
        print(f"  [{si+1}/8] {entry['s']}/{entry['v_base']}/{entry['o']}")

        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        # 基线hidden states
        inputs_base = tokenizer(sent_base, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs_base["input_ids"].to(device)
        attention_mask = inputs_base["attention_mask"].to(device)

        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        hs_base = [h[0, -1, :].detach().float().cpu().numpy() for h in out_base.hidden_states]
        del out_base

        # 约束hidden states
        def get_hs(prompt):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            ids = inputs["input_ids"].to(device)
            mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = [h[0, -1, :].detach().float().cpu().numpy() for h in out.hidden_states]
            del out
            return hs

        hs_neg = get_hs(sent_neg)
        hs_past = get_hs(sent_past)

        # 约束向量 δh_c
        delta_c = {}
        for li in test_layers:
            delta_c[f"L{li}_neg"] = hs_neg[li] - hs_base[li]
            delta_c[f"L{li}_past"] = hs_past[li] - hs_base[li]

        # 随机扰动方向
        rng = np.random.RandomState(42 + si)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        # 各ε值的Jacobian列估计 + 约束投影
        pair_result = {}

        for eps in eps_values:
            eps_key = f"eps_{eps:.3f}"

            # 计算Jacobian列估计
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).detach().clone()

            Y = {}  # {layer_idx: [d_model, k_perturb]}
            for li in test_layers:
                Y[li] = np.zeros((d_model, k_perturb), dtype=np.float32)

            for j in range(k_perturb):
                inputs_embeds_pert = inputs_embeds_base.clone()
                v_tensor = torch.tensor(V[:, j], dtype=inputs_embeds_pert.dtype, device=device)
                inputs_embeds_pert[0, -1, :] += eps * v_tensor

                with torch.no_grad():
                    out_pert = model(inputs_embeds=inputs_embeds_pert,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True)

                for li in test_layers:
                    hs_pert = out_pert.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                    Y[li][:, j] = (hs_pert - hs_base[li]) / eps

                del out_pert

            # 约束投影 P = Y^+ δh_c (最小二乘近似)
            for li_idx, li in enumerate(test_layers):
                lk = test_layer_names[li_idx]

                Y_l = Y[li]  # [d_model, k_perturb]

                # 约束投影: P_c = (Y^T Y)^{-1} Y^T δh_c
                # 即找 Y · α ≈ δh_c 的最小二乘解
                for c_name, delta in [("neg", delta_c[f"L{li}_neg"]), ("past", delta_c[f"L{li}_past"])]:
                    # 最小二乘: α = (Y^T Y)^{-1} Y^T δh_c
                    YtY = Y_l.T @ Y_l
                    Ytd = Y_l.T @ delta
                    try:
                        alpha = np.linalg.solve(YtY + 1e-6 * np.eye(k_perturb), Ytd)
                    except np.linalg.LinAlgError:
                        alpha = np.linalg.lstsq(Y_l, delta, rcond=None)[0]

                    # 投影P = Y · α
                    P = Y_l @ alpha

                    # 投影质量
                    residual = delta - P
                    proj_quality = 1.0 - np.linalg.norm(residual) / max(np.linalg.norm(delta), 1e-10)

                    # α的稀疏度 (Gini系数 — 值越高越稀疏)
                    alpha_norm = np.linalg.norm(alpha)
                    if alpha_norm > 1e-10:
                        alpha_abs = np.abs(alpha)
                        sorted_alpha = np.sort(alpha_abs)
                        n_alpha = len(sorted_alpha)
                        cumsum_alpha = np.cumsum(sorted_alpha)
                        gini_alpha = float((n_alpha + 1 - 2 * np.sum(cumsum_alpha) / cumsum_alpha[-1]) / n_alpha) if cumsum_alpha[-1] > 0 else 0

                        # 有效参与方向数: 贡献>1%能量的方向数
                        alpha_energy = alpha_abs ** 2
                        total_energy = np.sum(alpha_energy)
                        if total_energy > 0:
                            sorted_energy = np.sort(alpha_energy)[::-1]
                            cum_energy = np.cumsum(sorted_energy) / total_energy
                            effective_alpha_rank = int(np.searchsorted(cum_energy, 0.95) + 1)
                        else:
                            effective_alpha_rank = 0
                    else:
                        gini_alpha = 0
                        effective_alpha_rank = 0

                    key = f"{lk}_{c_name}_{eps_key}"
                    pair_result[key] = {
                        "proj_quality": float(proj_quality),
                        "delta_norm": float(np.linalg.norm(delta)),
                        "proj_norm": float(np.linalg.norm(P)),
                        "alpha_top5": np.argsort(np.abs(alpha))[-5:][::-1].tolist(),
                        "alpha_top5_vals": np.abs(alpha)[np.argsort(np.abs(alpha))[-5:][::-1]].tolist(),
                        "effective_alpha_rank": int(effective_alpha_rank),
                        "alpha_gini": gini_alpha,
                    }

        results["per_pair"][si] = pair_result

        gc.collect()

    # 汇总: 各ε值的投影质量对比
    summary = {}
    for li_idx, li in enumerate(test_layers):
        lk = test_layer_names[li_idx]
        for c_name in ["neg", "past"]:
            for eps in eps_values:
                eps_key = f"eps_{eps:.3f}"
                key_pattern = f"{lk}_{c_name}_{eps_key}"

                qualities = []
                alpha_ranks = []
                for si, pr in results["per_pair"].items():
                    if key_pattern in pr:
                        qualities.append(pr[key_pattern]["proj_quality"])
                        alpha_ranks.append(pr[key_pattern]["effective_alpha_rank"])

                if qualities:
                    summary[key_pattern] = {
                        "mean_quality": float(np.mean(qualities)),
                        "std_quality": float(np.std(qualities)),
                        "mean_alpha_rank": float(np.mean(alpha_ranks)),
                    }

    results["summary"] = summary
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    print("=" * 60)
    print(f"Phase 134: 条件激活边界分析 — {model_name}")
    print("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
          f"d={model_info.d_model}")

    all_results = {"model_info": {
        "name": model_name,
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
    }}

    # === Exp 1: SwiGLU激活稀疏性 ===
    print("\n" + "=" * 40)
    print("Exp 1: SwiGLU激活稀疏性")
    print("=" * 40)
    t1 = time.time()
    r1 = exp1_activation_sparsity(model, tokenizer, device, model_info)
    all_results["exp1_activation_sparsity"] = r1
    print(f"  Exp 1 done in {time.time()-t1:.1f}s")

    if "summary" in r1:
        for lk, ld in r1["summary"].items():
            print(f"  {lk}:")
            for variant, vd in ld.items():
                print(f"    {variant}: ratio={vd['mean_ratio']:.4f}, gini={vd['mean_gini']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 2: Attention Head激活模式 ===
    print("\n" + "=" * 40)
    print("Exp 2: Attention Head激活模式")
    print("=" * 40)
    t2 = time.time()
    r2 = exp2_attention_patterns(model, tokenizer, device, model_info)
    all_results["exp2_attention_patterns"] = r2
    print(f"  Exp 2 done in {time.time()-t2:.1f}s")

    if "variant_diff" in r2:
        for lk, ld in r2["variant_diff"].items():
            print(f"  {lk}: base_entropy={ld.get('base_mean_entropy',0):.4f}, "
                  f"neg_entropy={ld.get('neg_mean_entropy',0):.4f}, "
                  f"past_entropy={ld.get('past_mean_entropy',0):.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 3: 约束激活边界 ===
    print("\n" + "=" * 40)
    print("Exp 3: 约束激活边界")
    print("=" * 40)
    t3 = time.time()
    r3 = exp3_constraint_activation_boundary(model, tokenizer, device, model_info)
    all_results["exp3_constraint_boundary"] = r3
    print(f"  Exp 3 done in {time.time()-t3:.1f}s")

    if "summary" in r3:
        for lk, ld in r3["summary"].items():
            print(f"  {lk}: cos(neg,past)={ld['mean_cos_neg_past']:.4f}, "
                  f"rel_neg={ld['mean_rel_neg']:.4f}, rel_past={ld['mean_rel_past']:.4f}, "
                  f"dim_overlap={ld['mean_dim_overlap']:.4f}")

    if "subspace_analysis" in r3 and "L0" in r3["subspace_analysis"]:
        sa = r3["subspace_analysis"]["L0"]
        print(f"  L0 subspace: neg_rank={sa['neg_rank']}, past_rank={sa['past_rank']}, "
              f"subspace_cos={sa['subspace_cosine']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 4: 约束投影稳定性 ===
    print("\n" + "=" * 40)
    print("Exp 4: 约束投影稳定性")
    print("=" * 40)
    t4 = time.time()
    r4 = exp4_constraint_projection_stability(model, tokenizer, device, model_info)
    all_results["exp4_constraint_projection"] = r4
    print(f"  Exp 4 done in {time.time()-t4:.1f}s")

    if "summary" in r4:
        for key, ld in sorted(r4["summary"].items()):
            print(f"  {key}: quality={ld['mean_quality']:.4f}±{ld['std_quality']:.4f}, "
                  f"alpha_rank={ld['mean_alpha_rank']:.1f}")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"phase134_{model_name}_activation_boundary.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, default=convert, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    release_model(model)
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
