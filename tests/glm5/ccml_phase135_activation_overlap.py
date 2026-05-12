"""
Phase 135: 激活重叠分析 — 修正统计塌缩错误
=============================================

Phase 134的核心硬伤: "统计塌缩错误"
- 测了 activation ratio (多少神经元激活), 而非 WHICH 神经元激活
- 两个完全不同的激活集合可以有相同的 ratio
- Δratio≈0 不能推出 "语法不改变激活模式"

批评的核心纠正:
1. 应该测 activation overlap (Jaccard), 不是 activation ratio
2. entropy 不是结构, 应该测 attention edge persistence
3. 不同层可能使用不同计算机制, 不应假设统一理论

Phase 135策略: 真正测量"哪些"神经元被激活
- Exp 1: MLP中间层激活重叠 — base vs neg/past的Jaccard相似度
- Exp 2: Attention Head路由重叠 — 哪些head在语法变化时改变路由
- Exp 3: Attention Edge持久性 — 哪些attention边跨句子稳定
- Exp 4: 语义vs语法对比 — 换主语(语义) vs 换时态(语法)的激活重叠差异

关键方法:
- Hook到gate_proj/up_proj输出, 获取MLP中间层激活
- 使用多个阈值(threshold)定义"激活"
- 30+句子, 大数据量
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
from typing import Dict, List, Tuple, Optional, Set

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 句子设计 — 扩大规模, 30+句子
# ============================================================

# 基础句子对: 每组有 base/negation/past 三种变体
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
    {"s": "king", "v_base": "rules", "v_past": "ruled", "o": "kingdom"},
    {"s": "queen", "v_base": "leads", "v_past": "led", "o": "army"},
    {"s": "fire", "v_base": "burns", "v_past": "burned", "o": "forest"},
    {"s": "water", "v_base": "floods", "v_past": "flooded", "o": "valley"},
    {"s": "wind", "v_base": "blows", "v_past": "blew", "o": "leaves"},
    {"s": "sun", "v_base": "warms", "v_past": "warmed", "o": "earth"},
    {"s": "moon", "v_base": "lights", "v_past": "lit", "o": "night"},
    {"s": "river", "v_base": "carries", "v_past": "carried", "o": "boat"},
    {"s": "mountain", "v_base": "blocks", "v_past": "blocked", "o": "path"},
    {"s": "cloud", "v_base": "covers", "v_past": "covered", "o": "sky"},
    {"s": "wolf", "v_base": "hunts", "v_past": "hunted", "o": "deer"},
    {"s": "bear", "v_base": "catches", "v_past": "caught", "o": "fish"},
    {"s": "fox", "v_base": "tricks", "v_past": "tricked", "o": "crow"},
    {"s": "horse", "v_base": "pulls", "v_past": "pulled", "o": "cart"},
    {"s": "eagle", "v_base": "spots", "v_past": "spotted", "o": "prey"},
]

# 语义变化对: 同语法结构, 不同主语
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
    ("The wolf hunts the deer", "The bear catches the fish"),
    ("The fox tricks the crow", "The horse pulls the cart"),
    ("The sun warms the earth", "The moon lights the night"),
    ("The river carries the boat", "The mountain blocks the path"),
    ("The cloud covers the sky", "The eagle spots the prey"),
]


def make_sentence(entry, variant="base"):
    """生成语法正确的句子变体"""
    s = entry["s"]
    v = entry["v_base"]
    v_past = entry["v_past"]
    o = entry["o"]

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

def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Jaccard相似度: |A∩B| / |A∪B|"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


def overlap_ratio(set_a: Set, set_b: Set) -> float:
    """重叠比: |A∩B| / |A| (A中有多少也在B中)"""
    if not set_a:
        return 1.0 if not set_b else 0.0
    return len(set_a & set_b) / len(set_a)


def compute_effective_rank(singular_values, threshold=0.99):
    """计算有效秩"""
    total = np.sum(singular_values ** 2)
    if total < 1e-20:
        return 0
    cumsum = np.cumsum(singular_values ** 2)
    rank = np.searchsorted(cumsum / total, threshold) + 1
    return int(min(rank, len(singular_values)))


# ============================================================
# Exp 1: MLP中间层激活重叠分析
# ============================================================

def exp1_mlp_activation_overlap(model, tokenizer, device, model_info):
    """
    核心修正: 测量WHICH神经元被激活, 而非HOW MANY

    方法:
    1. Hook到gate_proj输出, 获取MLP中间层激活
    2. 定义激活集合: gate > threshold 的神经元索引集合
    3. 计算 base vs neg/past 的Jaccard相似度
    4. 多个threshold验证稳定性
    5. 30句子, 大数据量

    关键指标:
    - Jaccard(base, neg): 否定改变了多少激活集合?
    - Jaccard(base, past): 时态改变了多少激活集合?
    - 重叠度随层变化趋势
    - 与随机句子的Jaccard对比
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    intermediate_size = model_info.intermediate_size

    # 采样层: 更密集, 看清楚趋势
    sample_indices = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-2]
    sample_names = [f"L{i}" for i in sample_indices]

    # 多个阈值
    thresholds = [0.0, 0.1, 0.5]  # gate值阈值: silu激活为正

    results = {"per_sentence": {}, "intermediate_size": intermediate_size}

    n_sentences = 20  # 使用20个句子 (3变体 = 60前向传播)

    for si, entry in enumerate(SENTENCE_PAIRS[:n_sentences]):
        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        # 对三种变体获取MLP中间层激活
        variant_activations = {}  # {variant: {layer_idx: set of active neuron indices}}

        for variant, sent in [("base", sent_base), ("negation", sent_neg), ("past", sent_past)]:
            # Hook到gate_proj输出
            captured_gates = {}

            def make_gate_hook(layer_idx):
                def hook(module, input, output):
                    # output是gate_proj的输出, 即silu之前
                    captured_gates[layer_idx] = output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()
                return hook

            # 注册hooks — hook到MLP的gate_proj
            hooks = []
            for li in sample_indices:
                mlp = layers[li].mlp
                # 优先hook gate_proj, 否则hook整个MLP
                if hasattr(mlp, 'gate_proj'):
                    hooks.append(mlp.gate_proj.register_forward_hook(make_gate_hook(li)))
                elif hasattr(mlp, 'gate_up_proj'):
                    hooks.append(mlp.gate_up_proj.register_forward_hook(make_gate_hook(li)))
                else:
                    # 最后手段: hook整个MLP
                    hooks.append(mlp.register_forward_hook(make_gate_hook(li)))

            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            for h in hooks:
                h.remove()

            # 提取激活集合
            variant_activations[variant] = {}
            for li in sample_indices:
                if li not in captured_gates:
                    continue

                gate_out = captured_gates[li]
                # gate_out: 可能是 [1, seq_len, intermediate] 或 [seq_len, intermediate]
                # 对于 gate_up_proj (GLM4), 形状是 [1, seq_len, 2*intermediate]
                # 需要取前半部分作为gate

                if gate_out.dim() == 3:
                    last_token = gate_out[0, -1, :].numpy()
                elif gate_out.dim() == 2:
                    last_token = gate_out[-1, :].numpy()
                else:
                    last_token = gate_out.numpy()

                # 对于 gate_up_proj, 输出是 [2*intermediate], 前半是gate, 后半是up
                if hasattr(layers[li].mlp, 'gate_up_proj') and len(last_token) == 2 * intermediate_size:
                    gate_vals = last_token[:intermediate_size]
                else:
                    gate_vals = last_token[:intermediate_size] if len(last_token) >= intermediate_size else last_token

                # 各阈值的激活集合
                for thresh in thresholds:
                    active_set = set(np.where(gate_vals > thresh)[0].tolist())
                    key = f"L{li}_t{thresh}"
                    variant_activations[variant][key] = active_set

            del captured_gates, out
            gc.collect()

        # 计算Jaccard相似度
        sent_result = {}
        for li_idx, li in enumerate(sample_indices):
            lk = sample_names[li_idx]

            for thresh in thresholds:
                key = f"{lk}_t{thresh}"

                base_set = variant_activations["base"].get(key, set())
                neg_set = variant_activations["negation"].get(key, set())
                past_set = variant_activations["past"].get(key, set())

                # Jaccard
                jac_base_neg = jaccard_similarity(base_set, neg_set)
                jac_base_past = jaccard_similarity(base_set, past_set)
                jac_neg_past = jaccard_similarity(neg_set, past_set)

                # 重叠比 (方向性)
                overlap_neg_in_base = overlap_ratio(neg_set, base_set)  # neg中有多少也在base中
                overlap_base_in_neg = overlap_ratio(base_set, neg_set)  # base中有多少也在neg中

                # 集合大小
                sent_result[key] = {
                    "base_size": len(base_set),
                    "neg_size": len(neg_set),
                    "past_size": len(past_set),
                    "jaccard_base_neg": jac_base_neg,
                    "jaccard_base_past": jac_base_past,
                    "jaccard_neg_past": jac_neg_past,
                    "overlap_neg_in_base": overlap_neg_in_base,
                    "overlap_base_in_neg": overlap_base_in_neg,
                    # 新增: 差异集的大小
                    "neg_only_size": len(neg_set - base_set),  # neg独有
                    "base_only_size_vs_neg": len(base_set - neg_set),  # base独有(vs neg)
                    "past_only_size": len(past_set - base_set),  # past独有
                    "base_only_size_vs_past": len(base_set - past_set),  # base独有(vs past)
                }

        results["per_sentence"][si] = sent_result

    # 汇总: 各层各阈值的平均Jaccard
    summary = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]
        for thresh in thresholds:
            key = f"{lk}_t{thresh}"

            jacs_bn = []
            jacs_bp = []
            jacs_np = []
            base_sizes = []
            neg_only_sizes = []
            past_only_sizes = []

            for si in range(n_sentences):
                if si in results["per_sentence"] and key in results["per_sentence"][si]:
                    d = results["per_sentence"][si][key]
                    jacs_bn.append(d["jaccard_base_neg"])
                    jacs_bp.append(d["jaccard_base_past"])
                    jacs_np.append(d["jaccard_neg_past"])
                    base_sizes.append(d["base_size"])
                    neg_only_sizes.append(d["neg_only_size"])
                    past_only_sizes.append(d["past_only_size"])

            if jacs_bn:
                summary[key] = {
                    "mean_jaccard_base_neg": float(np.mean(jacs_bn)),
                    "std_jaccard_base_neg": float(np.std(jacs_bn)),
                    "mean_jaccard_base_past": float(np.mean(jacs_bp)),
                    "std_jaccard_base_past": float(np.std(jacs_bp)),
                    "mean_jaccard_neg_past": float(np.mean(jacs_np)),
                    "mean_base_size": float(np.mean(base_sizes)),
                    "mean_neg_only_size": float(np.mean(neg_only_sizes)),
                    "mean_past_only_size": float(np.mean(past_only_sizes)),
                    "frac_neg_only": float(np.mean(neg_only_sizes) / max(np.mean(base_sizes), 1)),
                    "frac_past_only": float(np.mean(past_only_sizes) / max(np.mean(base_sizes), 1)),
                }

    results["summary"] = summary

    # 基线对比: 随机句子的Jaccard
    print("  Computing random sentence baseline...")
    random_jacs = []
    for i in range(min(10, n_sentences)):
        j = (i + 7) % n_sentences  # 不同的句子对
        if i == j:
            continue
        # 用句子i的base和句子j的base做Jaccard
        key_i = f"L{n_layers//2}_t0.0"
        if i in results["per_sentence"] and j < n_sentences and j in results["per_sentence"]:
            # 需要额外的forward pass来获取句子j的激活...
            # 简化: 用已有的base_size估计随机Jaccard
            pass

    # 用理论估计: 如果base_size=k, intermediate_size=d, 随机Jaccard ≈ k/(2d-k)
    # (假设两个大小为k的随机子集)
    mid_key = f"L{n_layers//2}_t0.0"
    if mid_key in summary:
        avg_size = summary[mid_key]["mean_base_size"]
        expected_random_jac = avg_size / max(2 * intermediate_size - avg_size, 1)
        results["random_baseline"] = {
            "expected_random_jaccard": float(expected_random_jac),
            "intermediate_size": intermediate_size,
            "avg_active_size": float(avg_size),
        }

    return results


# ============================================================
# Exp 2: Attention Head路由重叠
# ============================================================

def exp2_head_routing_overlap(model, tokenizer, device, model_info):
    """
    测量语法变化时哪些head的路由模式改变

    关键指标:
    - 各head的attention pattern在base vs neg/past时的余弦相似度
    - 哪些head最稳定/最不稳定
    - head的"语法敏感度"排序
    """
    n_layers = model_info.n_layers
    layers = get_layers(model)

    # 获取头数
    layer0 = layers[0]
    n_heads = layer0.self_attn.config.num_attention_heads if hasattr(layer0.self_attn, 'config') else model_info.d_model // 64

    sample_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    sample_names = [f"L{i}" for i in sample_indices]

    n_sentences = 15

    results = {"per_sentence": {}, "n_heads": n_heads}

    for si, entry in enumerate(SENTENCE_PAIRS[:n_sentences]):
        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        variant_patterns = {}

        for variant, sent in [("base", sent_base), ("negation", sent_neg), ("past", sent_past)]:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=True)

            layer_patterns = {}
            for li_idx, li in enumerate(sample_indices):
                lk = sample_names[li_idx]
                if li >= len(out.attentions) or out.attentions[li] is None:
                    continue

                attn = out.attentions[li][0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
                seq_len = attn.shape[1]
                last_pos = seq_len - 1

                # 各head的last-token attention pattern
                head_patterns = []
                for h in range(min(n_heads, attn.shape[0])):
                    pattern = attn[h, last_pos, :]  # [seq_len]
                    head_patterns.append(pattern)

                layer_patterns[lk] = np.array(head_patterns)  # [n_heads, seq_len]

            variant_patterns[variant] = layer_patterns
            del out
            gc.collect()

        # 计算各head的pattern相似度
        sent_result = {}
        for li_idx, li in enumerate(sample_indices):
            lk = sample_names[li_idx]

            if lk not in variant_patterns["base"]:
                continue

            base_pats = variant_patterns["base"][lk]  # [n_heads, seq_len]
            neg_pats = variant_patterns["negation"][lk]
            past_pats = variant_patterns["past"][lk]

            # 各head的余弦相似度
            # 注意: 不同变体的序列长度可能不同 (否定句多token)
            # 策略: 只比较公共长度部分
            base_seq_len = base_pats.shape[1]
            neg_seq_len = neg_pats.shape[1]
            past_seq_len = past_pats.shape[1]

            head_cos_neg = []
            head_cos_past = []
            head_top_pos_change = []  # top attention位置是否改变

            for h in range(min(n_heads, base_pats.shape[0])):
                bp = base_pats[h]
                np_ = neg_pats[h]
                pp = past_pats[h]

                # 只比较公共长度部分
                common_len_neg = min(len(bp), len(np_))
                common_len_past = min(len(bp), len(pp))

                bp_neg = bp[:common_len_neg]
                np_c = np_[:common_len_neg]
                bp_past = bp[:common_len_past]
                pp_c = pp[:common_len_past]

                # 归一化
                bp_neg_n = bp_neg / max(np.linalg.norm(bp_neg), 1e-10)
                np_n = np_c / max(np.linalg.norm(np_c), 1e-10)
                bp_past_n = bp_past / max(np.linalg.norm(bp_past), 1e-10)
                pp_n = pp_c / max(np.linalg.norm(pp_c), 1e-10)

                cos_neg = float(np.dot(bp_neg_n, np_n))
                cos_past = float(np.dot(bp_past_n, pp_n))

                head_cos_neg.append(cos_neg)
                head_cos_past.append(cos_past)

                # Top attention位置 (在公共长度内)
                base_top_pos = int(np.argmax(bp[:common_len_neg]))
                neg_top_pos = int(np.argmax(np_c))
                base_top_pos_past = int(np.argmax(bp[:common_len_past]))
                past_top_pos = int(np.argmax(pp_c))
                head_top_pos_change.append({
                    "neg_changed": int(base_top_pos != neg_top_pos),
                    "past_changed": int(base_top_pos_past != past_top_pos),
                })

            # 找到最敏感和最稳定的head
            head_sensitivity = np.array(head_cos_neg)  # 越低越敏感
            most_sensitive_neg = int(np.argmin(head_cos_neg))
            most_stable_neg = int(np.argmax(head_cos_neg))
            most_sensitive_past = int(np.argmin(head_cos_past))
            most_stable_past = int(np.argmax(head_cos_past))

            # Top attention位置改变率
            neg_pos_change_rate = float(np.mean([d["neg_changed"] for d in head_top_pos_change]))
            past_pos_change_rate = float(np.mean([d["past_changed"] for d in head_top_pos_change]))

            sent_result[lk] = {
                "head_cos_neg_mean": float(np.mean(head_cos_neg)),
                "head_cos_neg_std": float(np.std(head_cos_neg)),
                "head_cos_past_mean": float(np.mean(head_cos_past)),
                "head_cos_past_std": float(np.std(head_cos_past)),
                "head_cos_neg_min": float(np.min(head_cos_neg)),
                "head_cos_past_min": float(np.min(head_cos_past)),
                "most_sensitive_neg_head": most_sensitive_neg,
                "most_stable_neg_head": most_stable_neg,
                "most_sensitive_past_head": most_sensitive_past,
                "most_stable_past_head": most_stable_past,
                "neg_pos_change_rate": neg_pos_change_rate,
                "past_pos_change_rate": past_pos_change_rate,
                # 所有head的相似度分布
                "head_cos_neg_hist": np.histogram(head_cos_neg, bins=10, range=(0, 1))[0].tolist(),
                "head_cos_past_hist": np.histogram(head_cos_past, bins=10, range=(0, 1))[0].tolist(),
            }

        results["per_sentence"][si] = sent_result

    # 汇总
    summary = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]

        cos_neg_means = []
        cos_past_means = []
        cos_neg_mins = []
        neg_pos_changes = []
        past_pos_changes = []

        for si in range(n_sentences):
            if si in results["per_sentence"] and lk in results["per_sentence"][si]:
                d = results["per_sentence"][si][lk]
                cos_neg_means.append(d["head_cos_neg_mean"])
                cos_past_means.append(d["head_cos_past_mean"])
                cos_neg_mins.append(d["head_cos_neg_min"])
                neg_pos_changes.append(d["neg_pos_change_rate"])
                past_pos_changes.append(d["past_pos_change_rate"])

        if cos_neg_means:
            summary[lk] = {
                "mean_head_cos_neg": float(np.mean(cos_neg_means)),
                "mean_head_cos_past": float(np.mean(cos_past_means)),
                "mean_min_cos_neg": float(np.mean(cos_neg_mins)),
                "mean_neg_pos_change_rate": float(np.mean(neg_pos_changes)),
                "mean_past_pos_change_rate": float(np.mean(past_pos_changes)),
            }

    results["summary"] = summary
    return results


# ============================================================
# Exp 3: Attention Edge持久性
# ============================================================

def exp3_attention_edge_persistence(model, tokenizer, device, model_info):
    """
    测量哪些attention边跨句子稳定存在

    关键问题:
    - 是否存在"语法不变attention边"?
    - 否定/时态改变了哪些边?
    - 边的持久性是否与head specialization相关?

    方法:
    1. 对每个句子获取attention矩阵
    2. 定义"边": attention weight > 0.1 的(token_i, token_j)对
    3. 计算跨变体的边重叠度
    4. 计算跨句子的边持久性
    """
    n_layers = model_info.n_layers
    layers = get_layers(model)
    n_heads = layers[0].self_attn.config.num_attention_heads if hasattr(layers[0].self_attn, 'config') else model_info.d_model // 64

    sample_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    sample_names = [f"L{i}" for i in sample_indices]

    edge_threshold = 0.1  # attention weight > 0.1 视为有效边

    n_sentences = 15
    results = {"per_sentence": {}, "n_heads": n_heads}

    for si, entry in enumerate(SENTENCE_PAIRS[:n_sentences]):
        sent_base = make_sentence(entry, "base")
        sent_neg = make_sentence(entry, "negation")
        sent_past = make_sentence(entry, "past")

        variant_edges = {}

        for variant, sent in [("base", sent_base), ("negation", sent_neg), ("past", sent_past)]:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=True)

            # 获取tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

            layer_edges = {}
            for li_idx, li in enumerate(sample_indices):
                lk = sample_names[li_idx]
                if li >= len(out.attentions) or out.attentions[li] is None:
                    continue

                attn = out.attentions[li][0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
                seq_len = attn.shape[1]

                # 各head的边集合
                head_edge_sets = []
                for h in range(min(n_heads, attn.shape[0])):
                    edges = set()
                    for i in range(seq_len):
                        for j in range(seq_len):
                            if attn[h, i, j] > edge_threshold:
                                edges.add((i, j))
                    head_edge_sets.append(edges)

                # 聚合所有head的边
                all_edges = set()
                for es in head_edge_sets:
                    all_edges.update(es)

                layer_edges[lk] = {
                    "all_edges": all_edges,
                    "head_edge_sets": head_edge_sets,
                    "n_edges_total": len(all_edges),
                    "n_edges_per_head": [len(es) for es in head_edge_sets],
                    "seq_len": seq_len,  # 记录序列长度
                }

            variant_edges[variant] = layer_edges
            del out
            gc.collect()

        # 计算边的重叠度
        sent_result = {}
        for li_idx, li in enumerate(sample_indices):
            lk = sample_names[li_idx]

            if lk not in variant_edges["base"]:
                continue

            # 注意: 不同变体序列长度可能不同
            # 策略: 只比较公共位置的边
            base_e_raw = variant_edges["base"][lk]["all_edges"]
            neg_e_raw = variant_edges["negation"][lk]["all_edges"]
            past_e_raw = variant_edges["past"][lk]["all_edges"]

            base_seq = variant_edges["base"][lk].get("seq_len", 100)
            neg_seq = variant_edges["negation"][lk].get("seq_len", 100)
            past_seq = variant_edges["past"][lk].get("seq_len", 100)

            # 过滤到公共位置
            common_len_neg = min(base_seq, neg_seq)
            common_len_past = min(base_seq, past_seq)

            base_e = {(i, j) for (i, j) in base_e_raw if i < common_len_neg and j < common_len_neg}
            neg_e = {(i, j) for (i, j) in neg_e_raw if i < common_len_neg and j < common_len_neg}
            base_e_past = {(i, j) for (i, j) in base_e_raw if i < common_len_past and j < common_len_past}
            past_e = {(i, j) for (i, j) in past_e_raw if i < common_len_past and j < common_len_past}

            # Jaccard
            jac_bn = jaccard_similarity(base_e, neg_e)
            jac_bp = jaccard_similarity(base_e_past, past_e)

            # 边变化数
            neg_new = len(neg_e - base_e)
            neg_lost = len(base_e - neg_e)
            past_new = len(past_e - base_e_past)
            past_lost = len(base_e_past - past_e)

            # 各head级别的边Jaccard
            head_jacs_neg = []
            head_jacs_past = []
            for h in range(min(n_heads, len(variant_edges["base"][lk]["head_edge_sets"]))):
                bh = variant_edges["base"][lk]["head_edge_sets"][h]
                nh_full = variant_edges["negation"][lk]["head_edge_sets"][h] if h < len(variant_edges["negation"][lk]["head_edge_sets"]) else set()
                ph_full = variant_edges["past"][lk]["head_edge_sets"][h] if h < len(variant_edges["past"][lk]["head_edge_sets"]) else set()

                # 过滤到公共位置
                bh_neg = {(i, j) for (i, j) in bh if i < common_len_neg and j < common_len_neg}
                nh = {(i, j) for (i, j) in nh_full if i < common_len_neg and j < common_len_neg}
                bh_past = {(i, j) for (i, j) in bh if i < common_len_past and j < common_len_past}
                ph = {(i, j) for (i, j) in ph_full if i < common_len_past and j < common_len_past}

                head_jacs_neg.append(jaccard_similarity(bh_neg, nh))
                head_jacs_past.append(jaccard_similarity(bh_past, ph))

            sent_result[lk] = {
                "jaccard_base_neg": jac_bn,
                "jaccard_base_past": jac_bp,
                "neg_new_edges": neg_new,
                "neg_lost_edges": neg_lost,
                "past_new_edges": past_new,
                "past_lost_edges": past_lost,
                "base_n_edges": len(base_e),
                "head_jac_neg_mean": float(np.mean(head_jacs_neg)) if head_jacs_neg else 0,
                "head_jac_past_mean": float(np.mean(head_jacs_past)) if head_jacs_past else 0,
                "head_jac_neg_min": float(np.min(head_jacs_neg)) if head_jacs_neg else 0,
                "head_jac_past_min": float(np.min(head_jacs_past)) if head_jacs_past else 0,
            }

        results["per_sentence"][si] = sent_result

    # 汇总
    summary = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]

        jacs_bn = []
        jacs_bp = []
        neg_news = []
        neg_losts = []
        base_sizes = []
        head_jac_neg_means = []
        head_jac_past_means = []

        for si in range(n_sentences):
            if si in results["per_sentence"] and lk in results["per_sentence"][si]:
                d = results["per_sentence"][si][lk]
                jacs_bn.append(d["jaccard_base_neg"])
                jacs_bp.append(d["jaccard_base_past"])
                neg_news.append(d["neg_new_edges"])
                neg_losts.append(d["neg_lost_edges"])
                base_sizes.append(d["base_n_edges"])
                head_jac_neg_means.append(d["head_jac_neg_mean"])
                head_jac_past_means.append(d["head_jac_past_mean"])

        if jacs_bn:
            summary[lk] = {
                "mean_jaccard_base_neg": float(np.mean(jacs_bn)),
                "mean_jaccard_base_past": float(np.mean(jacs_bp)),
                "mean_neg_new_edges": float(np.mean(neg_news)),
                "mean_neg_lost_edges": float(np.mean(neg_losts)),
                "mean_base_n_edges": float(np.mean(base_sizes)),
                "mean_head_jac_neg": float(np.mean(head_jac_neg_means)),
                "mean_head_jac_past": float(np.mean(head_jac_past_means)),
            }

    results["summary"] = summary
    return results


# ============================================================
# Exp 4: 语义vs语法 — 激活重叠差异
# ============================================================

def exp4_semantic_vs_syntax(model, tokenizer, device, model_info):
    """
    核心对比: 语义变化 vs 语法变化 对激活模式的影响

    批评指出: 我们缺乏语义-语法对比实验
    - 语义变化: 换主语/宾语 (dog→cat, man→mouse)
    - 语法变化: 换时态/加否定 (bites→bit, bites→does not bite)

    假说:
    - 语义 → 选择激活哪些神经元 (selection)
    - 语法 → 调制注意力分散度 (modulation)

    如果这个假说正确:
    - 语义变化的Jaccard应远低于语法变化
    - 即: 语义改变激活集合, 语法不改变
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    intermediate_size = model_info.intermediate_size

    sample_indices = [n_layers//4, n_layers//2, 3*n_layers//4]
    sample_names = [f"L{i}" for i in sample_indices]

    threshold = 0.0  # gate > 0 视为激活

    n_pairs = 15  # 使用15个语义对

    results = {"per_pair": {}}

    for pi, (sent_a, sent_b) in enumerate(SEMANTIC_PAIRS[:n_pairs]):
        print(f"  [{pi+1}/{n_pairs}] {sent_a[:30]} vs {sent_b[:30]}")

        # 获取两个语义不同句子的MLP中间层激活
        pair_activations = {}

        for label, sent in [("A", sent_a), ("B", sent_b)]:
            captured_gates = {}

            def make_gate_hook(layer_idx):
                def hook(module, input, output):
                    captured_gates[layer_idx] = output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()
                return hook

            hooks = []
            for li in sample_indices:
                mlp = layers[li].mlp
                if hasattr(mlp, 'gate_proj'):
                    hooks.append(mlp.gate_proj.register_forward_hook(make_gate_hook(li)))
                elif hasattr(mlp, 'gate_up_proj'):
                    hooks.append(mlp.gate_up_proj.register_forward_hook(make_gate_hook(li)))
                else:
                    hooks.append(mlp.register_forward_hook(make_gate_hook(li)))

            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, output_attentions=True)

            for h in hooks:
                h.remove()

            layer_activations = {}
            for li in sample_indices:
                if li not in captured_gates:
                    continue

                gate_out = captured_gates[li]
                if gate_out.dim() == 3:
                    last_token = gate_out[0, -1, :].numpy()
                elif gate_out.dim() == 2:
                    last_token = gate_out[-1, :].numpy()
                else:
                    last_token = gate_out.numpy()

                if hasattr(layers[li].mlp, 'gate_up_proj') and len(last_token) == 2 * intermediate_size:
                    gate_vals = last_token[:intermediate_size]
                else:
                    gate_vals = last_token[:intermediate_size] if len(last_token) >= intermediate_size else last_token

                active_set = set(np.where(gate_vals > threshold)[0].tolist())
                layer_activations[f"L{li}"] = active_set

            # 也获取attention信息
            layer_attn = {}
            for li_idx, li in enumerate(sample_indices):
                lk = sample_names[li_idx]
                if li < len(out.attentions) and out.attentions[li] is not None:
                    attn = out.attentions[li][0].float().cpu().numpy()
                    n_h = attn.shape[0]
                    seq_len = attn.shape[1]
                    # 各head的entropy
                    entropies = []
                    for h in range(n_h):
                        pattern = attn[h, -1, :]
                        pattern = pattern / max(pattern.sum(), 1e-10)
                        ent = -np.sum(pattern * np.log(pattern + 1e-10))
                        entropies.append(float(ent))
                    layer_attn[lk] = {"entropies": entropies, "mean_entropy": float(np.mean(entropies))}

            pair_activations[label] = {
                "mlp": layer_activations,
                "attn": layer_attn,
            }

            del captured_gates, out
            gc.collect()

        # 计算语义变化的MLP Jaccard
        pair_result = {}
        for li_idx, li in enumerate(sample_indices):
            lk = sample_names[li_idx]

            a_mlp = pair_activations["A"]["mlp"].get(lk, set())
            b_mlp = pair_activations["B"]["mlp"].get(lk, set())

            # MLP Jaccard
            mlp_jac = jaccard_similarity(a_mlp, b_mlp)

            # Attention entropy差异
            a_attn = pair_activations["A"]["attn"].get(lk, {})
            b_attn = pair_activations["B"]["attn"].get(lk, {})
            entropy_diff = abs(a_attn.get("mean_entropy", 0) - b_attn.get("mean_entropy", 0))

            pair_result[lk] = {
                "mlp_jaccard_semantic": mlp_jac,
                "a_mlp_size": len(a_mlp),
                "b_mlp_size": len(b_mlp),
                "attn_entropy_diff_semantic": entropy_diff,
            }

        results["per_pair"][pi] = pair_result

    # 汇总语义Jaccard
    summary = {}
    for li_idx, li in enumerate(sample_indices):
        lk = sample_names[li_idx]

        semantic_jacs = []
        entropy_diffs = []

        for pi in range(n_pairs):
            if pi in results["per_pair"] and lk in results["per_pair"][pi]:
                d = results["per_pair"][pi][lk]
                semantic_jacs.append(d["mlp_jaccard_semantic"])
                entropy_diffs.append(d["attn_entropy_diff_semantic"])

        if semantic_jacs:
            summary[lk] = {
                "mean_semantic_jaccard": float(np.mean(semantic_jacs)),
                "std_semantic_jaccard": float(np.std(semantic_jacs)),
                "mean_semantic_entropy_diff": float(np.mean(entropy_diffs)),
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
    print(f"Phase 135: 激活重叠分析 — 修正统计塌缩错误 — {model_name}")
    print("=" * 60)

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
          f"d={model_info.d_model}, intermediate={model_info.intermediate_size}")

    all_results = {"model_info": {
        "name": model_name,
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "intermediate_size": model_info.intermediate_size,
    }}

    # === Exp 1: MLP中间层激活重叠 ===
    print("\n" + "=" * 40)
    print("Exp 1: MLP中间层激活重叠 (base vs neg/past)")
    print("=" * 40)
    t1 = time.time()
    r1 = exp1_mlp_activation_overlap(model, tokenizer, device, model_info)
    all_results["exp1_mlp_overlap"] = r1
    print(f"  Exp 1 done in {time.time()-t1:.1f}s")

    if "summary" in r1:
        print("\n  MLP激活重叠汇总:")
        for key, ld in sorted(r1["summary"].items()):
            if "t0.0" in key:  # 只打印threshold=0的结果
                print(f"    {key}: J(base,neg)={ld['mean_jaccard_base_neg']:.4f}±{ld['std_jaccard_base_neg']:.4f}, "
                      f"J(base,past)={ld['mean_jaccard_base_past']:.4f}, "
                      f"neg_only={ld['frac_neg_only']:.4f}, past_only={ld['frac_past_only']:.4f}")

    if "random_baseline" in r1:
        rb = r1["random_baseline"]
        print(f"  随机基线: expected_random_jac={rb['expected_random_jaccard']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 2: Attention Head路由重叠 ===
    print("\n" + "=" * 40)
    print("Exp 2: Attention Head路由重叠")
    print("=" * 40)
    t2 = time.time()
    r2 = exp2_head_routing_overlap(model, tokenizer, device, model_info)
    all_results["exp2_head_routing"] = r2
    print(f"  Exp 2 done in {time.time()-t2:.1f}s")

    if "summary" in r2:
        print("\n  Head路由重叠汇总:")
        for lk, ld in sorted(r2["summary"].items()):
            print(f"    {lk}: cos(neg)={ld['mean_head_cos_neg']:.4f}, cos(past)={ld['mean_head_cos_past']:.4f}, "
                  f"min_cos(neg)={ld['mean_min_cos_neg']:.4f}, "
                  f"neg_pos_change={ld['mean_neg_pos_change_rate']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 3: Attention Edge持久性 ===
    print("\n" + "=" * 40)
    print("Exp 3: Attention Edge持久性")
    print("=" * 40)
    t3 = time.time()
    r3 = exp3_attention_edge_persistence(model, tokenizer, device, model_info)
    all_results["exp3_edge_persistence"] = r3
    print(f"  Exp 3 done in {time.time()-t3:.1f}s")

    if "summary" in r3:
        print("\n  Edge持久性汇总:")
        for lk, ld in sorted(r3["summary"].items()):
            print(f"    {lk}: J(base,neg)={ld['mean_jaccard_base_neg']:.4f}, J(base,past)={ld['mean_jaccard_base_past']:.4f}, "
                  f"neg_new={ld['mean_neg_new_edges']:.1f}, neg_lost={ld['mean_neg_lost_edges']:.1f}, "
                  f"head_J(neg)={ld['mean_head_jac_neg']:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 4: 语义vs语法 ===
    print("\n" + "=" * 40)
    print("Exp 4: 语义vs语法对比")
    print("=" * 40)
    t4 = time.time()
    r4 = exp4_semantic_vs_syntax(model, tokenizer, device, model_info)
    all_results["exp4_semantic_vs_syntax"] = r4
    print(f"  Exp 4 done in {time.time()-t4:.1f}s")

    if "summary" in r4:
        print("\n  语义vs语法汇总:")
        for lk, ld in sorted(r4["summary"].items()):
            print(f"    {lk}: 语义Jaccard={ld['mean_semantic_jaccard']:.4f}±{ld['std_semantic_jaccard']:.4f}, "
                  f"语义entropy_diff={ld['mean_semantic_entropy_diff']:.4f}")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"phase135_{model_name}_activation_overlap.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, default=convert, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    release_model(model)
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
