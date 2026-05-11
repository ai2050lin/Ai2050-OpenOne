"""
Phase 132: 算子流形分析 (Operator Manifold Analysis)
====================================================

Phase 131的理论批评(用户指出):
1. "低秩扰动 ≠ 低秩语言": 我测量的是Δh(扰动), 不是h(表示).
   低秩意味着语言"操作"是低秩的(类似LoRA), 不等于语言本身低维.
2. "约束不是数学本体": "约束"是语言学概念, 计算本体是条件Jacobian J_l(x).
3. "双重否定≠逆元"不成立: tokenization混淆("does not fail to bite"≠"bites")
4. "对易子[A,B] > 交换律测试": 对易子捕获算子空间的"曲率"
5. "条件纤维丛": 基空间=输入, 纤维=局部Jacobian, 联络=层间传播, 曲率=非交换性

Phase 132核心转变:
- 从"约束效应分析" → "算子几何分析"
- 研究对象: J_l(x)的谱结构, 子空间对齐, 对易子
- 关键问题: 算子是否形成低维族? 不同输入是否共享算子结构?

5个实验:
- Exp 1: Jacobian谱结构 — 累积Jacobian J_{l:0}的奇异值谱/有效秩
- Exp 2: 算子子空间比较 — 不同句子的Jacobian主子空间是否共享?
- Exp 3: 约束对易子(修正版) — 语法正确的约束效应+对易子谱分析
- Exp 4: 低秩输运分析 — 扰动子空间如何随层传播/旋转?
- Exp 5: Tokenization控制 — "doesn't" vs "does not"的算子差异

关键设计原则:
- 用ε=1e-4小扰动避免非线性伪影
- 用语法正确的句子避免Phase 131的tokenization混淆
- 用随机扰动估计Jacobian谱, 不需要计算完整d×d矩阵
- 三个模型交叉验证
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

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 句子设计 — 语法正确! 修复Phase 131的语法错误
# ============================================================

# 基础句子: 主-谓-宾, 动词都是规则变化
BASE_SENTENCES = [
    {"s": "dog", "v_base": "bites", "v_past": "bit", "v_pp": "bitten", "o": "man"},
    {"s": "cat", "v_base": "chases", "v_past": "chased", "v_pp": "chased", "o": "mouse"},
    {"s": "teacher", "v_base": "helps", "v_past": "helped", "v_pp": "helped", "o": "student"},
    {"s": "scientist", "v_base": "discovers", "v_past": "discovered", "v_pp": "discovered", "o": "formula"},
    {"s": "artist", "v_base": "paints", "v_past": "painted", "v_pp": "painted", "o": "picture"},
    {"s": "doctor", "v_base": "treats", "v_past": "treated", "v_pp": "treated", "o": "patient"},
    {"s": "writer", "v_base": "signs", "v_past": "signed", "v_pp": "signed", "o": "letter"},
    {"s": "child", "v_base": "reads", "v_past": "read", "v_pp": "read", "o": "book"},
    {"s": "farmer", "v_base": "grows", "v_past": "grown", "v_pp": "grown", "o": "crop"},
    {"s": "driver", "v_base": "follows", "v_past": "followed", "v_pp": "followed", "o": "road"},
]


def make_sentence(entry, variant="base"):
    """生成语法正确的句子变体"""
    s = entry["s"]
    v = entry["v_base"]
    v_past = entry["v_past"]
    v_pp = entry["v_pp"]
    o = entry["o"]
    v_stem = v.rstrip("es").rstrip("s") if v.endswith("s") else v
    # 更精确的去尾: "bites" -> "bite", "chases" -> "chase"
    if v.endswith("ies"):
        v_stem = v[:-3] + "y"  # 不适用当前动词
    elif v.endswith("shes") or v.endswith("ches") or v.endswith("xes"):
        v_stem = v[:-2]  # "watches" -> "watch"... 不对
    elif v.endswith("sses"):
        v_stem = v[:-2]  # "kisses" -> "kiss"
    else:
        v_stem = v[:-1] if v.endswith("s") and not v.endswith("ss") else v

    if variant == "base":
        return f"The {s} {v} the {o}"
    elif variant == "negation":
        return f"The {s} does not {v_stem} the {o}"
    elif variant == "past":
        return f"The {s} {v_past} the {o}"
    elif variant == "passive":
        return f"The {o} is {v_pp} by the {s}"
    elif variant == "plural":
        return f"The {s}s {v_stem} the {o}"
    elif variant == "neg_past":
        return f"The {s} did not {v_stem} the {o}"
    elif variant == "past_passive":
        return f"The {o} was {v_pp} by the {s}"
    elif variant == "neg_passive":
        return f"The {o} is not {v_pp} by the {s}"
    elif variant == "neg_plural":
        return f"The {s}s do not {v_stem} the {o}"
    elif variant == "double_neg":
        # 双重否定(语法正确): "It is not true that the dog does not bite the man"
        return f"It is not true that the {s} does not {v_stem} the {o}"
    elif variant == "contraction":
        # 缩写形式(与negation对比tokenization效应)
        return f"The {s} doesn't {v_stem} the {o}"
    else:
        return f"The {s} {v} the {o}"


# ============================================================
# 核心工具函数
# ============================================================

def get_hidden_states_all(model, tokenizer, device, prompt, max_length=64):
    """获取所有层的hidden states at last token position"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hs = []
    for h in out.hidden_states:
        hs.append(h[0, -1, :].detach().float().cpu().numpy())
    return hs


def get_all_hidden_states_with_perturbation(model, tokenizer, device, prompt,
                                             perturb_pos=-1, eps=0.0, perturb_vec=None):
    """获取带扰动的所有层hidden states

    Args:
        perturb_pos: 扰动位置(-1=last token)
        eps: 扰动幅度
        perturb_vec: 扰动方向 [d_model], None则不扰动
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 获取embedding
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone()

    # 施加扰动
    if perturb_vec is not None and eps > 0:
        pos = perturb_pos if perturb_pos >= 0 else inputs_embeds.shape[1] + perturb_pos
        v_tensor = torch.tensor(perturb_vec, dtype=inputs_embeds.dtype, device=device)
        inputs_embeds[0, pos, :] += eps * v_tensor

    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    output_hidden_states=True)

    hs = []
    for h in out.hidden_states:
        hs.append(h[0, -1, :].detach().float().cpu().numpy())
    return hs


def compute_effective_rank(singular_values, threshold=0.99):
    """计算有效秩: 覆盖threshold能量所需的最小维度数"""
    total = np.sum(singular_values ** 2)
    if total < 1e-20:
        return 0
    cumsum = np.cumsum(singular_values ** 2)
    rank = np.searchsorted(cumsum / total, threshold) + 1
    return int(min(rank, len(singular_values)))


def subspace_cosine(U1, U2, k1=None, k2=None):
    """计算两个子空间的平均cosine相似度

    U1, U2: 左奇异向量矩阵 [d, k]
    """
    if k1 is not None:
        U1 = U1[:, :k1]
    if k2 is not None:
        U2 = U2[:, :k2]
    # 投影矩阵
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    # Frobenius内积归一化
    trace_overlap = np.trace(P1 @ P2)
    norm1 = np.sqrt(np.trace(P1 @ P1))
    norm2 = np.sqrt(np.trace(P2 @ P2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(trace_overlap / (norm1 * norm2))


def principal_angles(U1, U2, k1=None, k2=None):
    """计算两个子空间的主角度"""
    if k1 is not None:
        U1 = U1[:, :k1]
    if k2 is not None:
        U2 = U2[:, :k2]
    # 正交化
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    # SVD of Q1^T Q2 gives cosines of principal angles
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    # clamp to [0, 1]
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return angles


# ============================================================
# Exp 1: Jacobian谱结构 (累积Jacobian)
# ============================================================

def exp1_jacobian_spectrum(model, tokenizer, device, model_info):
    """
    通过输入embedding的随机扰动估计累积Jacobian J_{l:0}的谱结构.

    方法:
    1. 对每个句子, 生成k个随机扰动方向v_j
    2. 对每个v_j, 计算 δh_l = (h_l(perturbed) - h_l(base)) / ε
    3. 堆叠: Y_l = [δh_l^(1), ..., δh_l^(k)]
    4. SVD(Y_l) ≈ SVD(J_{l:0})的前k个分量

    测量:
    - 各层Jacobian的有效秩
    - 奇异值衰减模式
    - 谱间隙(spectral gap)
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 48  # 扰动方向数
    eps = 1e-4
    n_sentences = 5  # 用5个基础句子

    results = {"per_sentence": {}, "aggregate": {}}

    for si, entry in enumerate(BASE_SENTENCES[:n_sentences]):
        sent = make_sentence(entry, "base")
        print(f"  [{si+1}/{n_sentences}] {sent}")

        # 基线
        hs_base = get_hidden_states_all(model, tokenizer, device, sent)

        # 随机扰动方向(固定seed保证可复现)
        rng = np.random.RandomState(42 + si)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        # 归一化每列
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        # 收集各层的扰动响应
        Y_all = np.zeros((n_layers + 1, d_model, k_perturb), dtype=np.float32)

        for j in range(k_perturb):
            hs_pert = get_all_hidden_states_with_perturbation(
                model, tokenizer, device, sent,
                perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
            )
            for l in range(n_layers + 1):
                Y_all[l, :, j] = (hs_pert[l] - hs_base[l]) / eps

        # 各层SVD
        sent_result = {"singular_values": {}, "eff_ranks": {}, "spectral_gaps": {}}
        for l in range(n_layers + 1):
            Y_l = Y_all[l]
            try:
                U, S, Vh = np.linalg.svd(Y_l, full_matrices=False)
            except:
                continue

            sent_result["singular_values"][f"L{l}"] = S[:32].tolist()
            sent_result["eff_ranks"][f"L{l}"] = compute_effective_rank(S, 0.99)

            # 谱间隙: S[i]/S[i+1]的最大值
            if len(S) > 1:
                ratios = S[:-1] / np.maximum(S[1:], 1e-10)
                gap_idx = int(np.argmax(ratios))
                sent_result["spectral_gaps"][f"L{l}"] = {
                    "max_ratio": float(ratios[gap_idx]),
                    "gap_after_dim": int(gap_idx + 1),
                    "s_before": float(S[gap_idx]),
                    "s_after": float(S[gap_idx + 1]),
                }

        results["per_sentence"][sent] = sent_result

    # 汇总: 各层平均有效秩
    agg_ranks = {}
    for l in range(n_layers + 1):
        lk = f"L{l}"
        ranks = [v["eff_ranks"].get(lk, 0) for v in results["per_sentence"].values()]
        agg_ranks[lk] = {"mean": float(np.mean(ranks)), "std": float(np.std(ranks))}
    results["aggregate"]["eff_ranks"] = agg_ranks

    # 汇总: 各层平均奇异值(归一化到S[0]=1)
    agg_sv = {}
    for l in range(n_layers + 1):
        lk = f"L{l}"
        all_sv = [np.array(v["singular_values"].get(lk, [])) for v in results["per_sentence"].values()]
        if all_sv and len(all_sv[0]) > 0:
            # 归一化
            normed = [sv / max(sv[0], 1e-10) for sv in all_sv if len(sv) > 0]
            min_len = min(len(sv) for sv in normed)
            if min_len > 0:
                agg_sv[lk] = {
                    "mean_normed": np.mean([sv[:min_len] for sv in normed], axis=0).tolist(),
                    "std_normed": np.std([sv[:min_len] for sv in normed], axis=0).tolist(),
                }
    results["aggregate"]["spectral_decay"] = agg_sv

    # 汇总: 谱间隙
    agg_gaps = {}
    for l in range(n_layers + 1):
        lk = f"L{l}"
        gaps = [v["spectral_gaps"].get(lk, {}) for v in results["per_sentence"].values()]
        if gaps:
            ratios = [g.get("max_ratio", 0) for g in gaps if g]
            dims = [g.get("gap_after_dim", 0) for g in gaps if g]
            if ratios:
                agg_gaps[lk] = {
                    "mean_max_ratio": float(np.mean(ratios)),
                    "mean_gap_dim": float(np.mean(dims)),
                }
    results["aggregate"]["spectral_gaps"] = agg_gaps

    return results


# ============================================================
# Exp 2: 算子子空间比较 (跨句子Jacobian相似性)
# ============================================================

def exp2_operator_subspace_comparison(model, tokenizer, device, model_info):
    """
    比较不同句子的Jacobian主子空间是否共享.

    核心问题:
    - 不同语义的句子是否使用相同的"算子方向"?
    - 相同语义不同语法(否定/过去时)的句子是否共享算子?

    方法:
    1. 对每个句子变体, 用随机扰动估计Jacobian的左奇异向量
    2. 比较不同句子在同一层的Jacobian子空间
    3. 测量: 子空间cosine, 主角度
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 32
    eps = 1e-4

    # 3个基础句子 × 4个变体 = 12个句子
    sentences = {}
    for si, entry in enumerate(BASE_SENTENCES[:3]):
        for variant in ["base", "negation", "past", "passive"]:
            key = f"S{si}_{variant}"
            sentences[key] = make_sentence(entry, variant)

    results = {"sentences": sentences, "layer_analysis": {}}

    # 采样层
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    # 对每个句子, 收集各层的Jacobian左奇异向量
    all_U = {}  # {sent_key: {Ll: U}}
    for key, sent in sentences.items():
        print(f"  [{key}] {sent}")
        hs_base = get_hidden_states_all(model, tokenizer, device, sent)

        rng = np.random.RandomState(123)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        Y_all = np.zeros((n_layers + 1, d_model, k_perturb), dtype=np.float32)
        for j in range(k_perturb):
            hs_pert = get_all_hidden_states_with_perturbation(
                model, tokenizer, device, sent,
                perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
            )
            for l in range(n_layers + 1):
                Y_all[l, :, j] = (hs_pert[l] - hs_base[l]) / eps

        all_U[key] = {}
        for l in sample_layers:
            try:
                U, S, Vh = np.linalg.svd(Y_all[l], full_matrices=False)
                all_U[key][f"L{l}"] = U[:, :min(16, U.shape[1])]
            except:
                pass

    # 跨句子子空间比较
    keys = list(all_U.keys())
    for l in sample_layers:
        lk = f"L{l}"
        pairwise = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if lk in all_U[ki] and lk in all_U[kj]:
                    cos = subspace_cosine(all_U[ki][lk], all_U[kj][lk], k1=8, k2=8)
                    angles = principal_angles(all_U[ki][lk], all_U[kj][lk], k1=8, k2=8)
                    pairwise[f"{ki}_vs_{kj}"] = {
                        "subspace_cos_8d": cos,
                        "mean_angle_deg": float(np.mean(np.degrees(angles))),
                        "min_angle_deg": float(np.min(np.degrees(angles))),
                    }

        # 分类汇总: 同句不同语法 vs 不同句同语法 vs 不同句不同语法
        same_sem_diff_syn = []
        diff_sem_same_syn = []
        diff_sem_diff_syn = []
        for pair_key, pair_data in pairwise.items():
            ki, kj = pair_key.split("_vs_")
            si_i, vi_i = ki.split("_", 1)
            si_j, vi_j = kj.split("_", 1)

            cos_val = pair_data["subspace_cos_8d"]

            if si_i == si_j and vi_i != vi_j:
                same_sem_diff_syn.append(cos_val)
            elif si_i != si_j and vi_i == vi_j:
                diff_sem_same_syn.append(cos_val)
            else:
                diff_sem_diff_syn.append(cos_val)

        results["layer_analysis"][lk] = {
            "pairwise": {k: v for k, v in list(pairwise.items())[:20]},  # 限制输出
            "same_sem_diff_syn": {"mean": float(np.mean(same_sem_diff_syn)) if same_sem_diff_syn else 0,
                                  "n": len(same_sem_diff_syn)},
            "diff_sem_same_syn": {"mean": float(np.mean(diff_sem_same_syn)) if diff_sem_same_syn else 0,
                                  "n": len(diff_sem_same_syn)},
            "diff_sem_diff_syn": {"mean": float(np.mean(diff_sem_diff_syn)) if diff_sem_diff_syn else 0,
                                  "n": len(diff_sem_diff_syn)},
        }

    return results


# ============================================================
# Exp 3: 约束对易子(修正版 — 语法正确+对易子谱分析)
# ============================================================

def exp3_constraint_commutator(model, tokenizer, device, model_info):
    """
    修正Phase 131的对易子分析:
    1. 使用语法正确的句子(修复"does not bites"等错误)
    2. 计算"约束对易子"的谱结构, 不只是标量范数
    3. 控制tokenization效应

    对易子定义:
    - δ(A) = h_l(A) - h_l(base)  [约束A的效应]
    - δ(B) = h_l(B) - h_l(base)  [约束B的效应]
    - δ(AB) = h_l(AB) - h_l(base)  [组合约束的效应]
    - C(A,B) = δ(AB) - δ(A) - δ(B)  [对易子 ≈ 非线性交互]

    谱分析:
    - 对易子C(A,B)在各层的范数/方向
    - C(A,B)是否与δ(A)或δ(B)对齐?
    - C(A,B)的有效维度
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    results = {}

    # 对每对约束, 在多个句子上测量
    constraint_pairs = [
        ("negation", "past", "neg_past"),
        ("negation", "passive", "neg_passive"),
        ("past", "passive", "past_passive"),
        ("negation", "plural", "neg_plural"),
    ]

    for entry in BASE_SENTENCES[:6]:
        sent_key = entry["s"]
        results[sent_key] = {}

        # 获取所有变体的hidden states
        variants_needed = {"base": make_sentence(entry, "base")}
        for c1, c2, c12 in constraint_pairs:
            variants_needed[c1] = make_sentence(entry, c1)
            variants_needed[c2] = make_sentence(entry, c2)
            variants_needed[c12] = make_sentence(entry, c12)

        hs_cache = {}
        for vk, vsent in variants_needed.items():
            hs_cache[vk] = get_hidden_states_all(model, tokenizer, device, vsent)

        # 计算每对约束的对易子
        for c1, c2, c12 in constraint_pairs:
            pair_key = f"{c1}_x_{c2}"
            delta_c1 = [hs_cache[c1][l] - hs_cache["base"][l] for l in range(n_layers + 1)]
            delta_c2 = [hs_cache[c2][l] - hs_cache["base"][l] for l in range(n_layers + 1)]
            delta_c12 = [hs_cache[c12][l] - hs_cache["base"][l] for l in range(n_layers + 1)]

            # 对易子: C = δ(AB) - δ(A) - δ(B)
            commutator = [delta_c12[l] - delta_c1[l] - delta_c2[l] for l in range(n_layers + 1)]

            pair_result = {"layer_results": []}
            for l in range(n_layers + 1):
                norm_c1 = np.linalg.norm(delta_c1[l])
                norm_c2 = np.linalg.norm(delta_c2[l])
                norm_c12 = np.linalg.norm(delta_c12[l])
                norm_comm = np.linalg.norm(commutator[l])

                # 对易子相对范数
                comm_rel = norm_comm / max(norm_c12, 1e-10)

                # 对易子与各约束效应的cosine
                cos_comm_c1 = float(np.dot(commutator[l], delta_c1[l]) /
                                    max(norm_comm * norm_c1, 1e-10))
                cos_comm_c2 = float(np.dot(commutator[l], delta_c2[l]) /
                                    max(norm_comm * norm_c2, 1e-10))

                # 各约束效应之间的cosine
                cos_c1_c2 = float(np.dot(delta_c1[l], delta_c2[l]) /
                                  max(norm_c1 * norm_c2, 1e-10))

                # nl_ratio (与Phase 131一致)
                nl_ratio = norm_comm / max(norm_c12, 1e-10)

                pair_result["layer_results"].append({
                    "norm_c1": float(norm_c1),
                    "norm_c2": float(norm_c2),
                    "norm_c12": float(norm_c12),
                    "norm_commutator": float(norm_comm),
                    "comm_rel_norm": float(comm_rel),
                    "nl_ratio": float(nl_ratio),
                    "cos_commutator_c1": float(cos_comm_c1),
                    "cos_commutator_c2": float(cos_comm_c2),
                    "cos_c1_c2": float(cos_c1_c2),
                })

            results[sent_key][pair_key] = pair_result

    # 双重否定测试(修正tokenization)
    print("  [双否定修正] 使用语法正确的句子...")
    neg_neg_results = {}
    for entry in BASE_SENTENCES[:6]:
        sent_key = entry["s"]
        base_hs = get_hidden_states_all(model, tokenizer, device, make_sentence(entry, "base"))
        neg_hs = get_hidden_states_all(model, tokenizer, device, make_sentence(entry, "negation"))
        # 双重否定: "It is not true that the X does not Y the Z"
        # 注意: 这与base句子的长度不同! 需要控制
        dbl_neg_hs = get_hidden_states_all(model, tokenizer, device, make_sentence(entry, "double_neg"))

        neg_neg_results[sent_key] = {"layer_results": []}
        for l in range(n_layers + 1):
            delta_neg = neg_hs[l] - base_hs[l]
            delta_dbl = dbl_neg_hs[l] - base_hs[l]

            norm_neg = np.linalg.norm(delta_neg)
            norm_dbl = np.linalg.norm(delta_dbl)
            cos_neg_dbl = float(np.dot(delta_neg, delta_dbl) / max(norm_neg * norm_dbl, 1e-10))

            # 关键指标: 双否定是否更接近base(恢复)?
            delta_to_base = np.linalg.norm(base_hs[l] - neg_hs[l])
            dbl_to_base = np.linalg.norm(base_hs[l] - dbl_neg_hs[l])
            recovery_ratio = dbl_to_base / max(delta_to_base, 1e-10)

            neg_neg_results[sent_key]["layer_results"].append({
                "cos_neg_dblneg": float(cos_neg_dbl),
                "neg_norm": float(norm_neg),
                "dbl_neg_norm": float(norm_dbl),
                "recovery_ratio": float(recovery_ratio),  # <1表示双否定更接近base
            })

    results["double_negation_corrected"] = neg_neg_results

    return results


# ============================================================
# Exp 4: 低秩输运分析
# ============================================================

def exp4_low_rank_transport(model, tokenizer, device, model_info):
    """
    分析扰动子空间如何在层间传播/旋转.

    核心问题:
    - 第l层的Jacobian主子空间, 经过一层传播后, 旋转了多少?
    - 累积Jacobian的秩是否随层深增长/缩减?
    - 扰动信息是否被"压缩"到越来越低维的子空间?

    方法:
    1. 用随机扰动估计各层Y_l = J_{l:0} @ V
    2. 计算相邻层Y_l和Y_{l+1}的子空间角度
    3. 估计per-layer Jacobian: J_l ≈ Y_{l+1} @ Y_l^+
    4. 分析J_l的有效秩和谱结构
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 48
    eps = 1e-4

    results = {"transport_angles": {}, "per_layer_jacobian": {}, "rank_evolution": {}}

    for si, entry in enumerate(BASE_SENTENCES[:3]):
        sent = make_sentence(entry, "base")
        print(f"  [{si+1}] {sent}")

        hs_base = get_hidden_states_all(model, tokenizer, device, sent)

        rng = np.random.RandomState(99 + si)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        Y_all = np.zeros((n_layers + 1, d_model, k_perturb), dtype=np.float32)
        for j in range(k_perturb):
            hs_pert = get_all_hidden_states_with_perturbation(
                model, tokenizer, device, sent,
                perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
            )
            for l in range(n_layers + 1):
                Y_all[l, :, j] = (hs_pert[l] - hs_base[l]) / eps

        # 各层SVD
        layer_U = {}
        layer_S = {}
        for l in range(n_layers + 1):
            try:
                U, S, Vh = np.linalg.svd(Y_all[l], full_matrices=False)
                layer_U[l] = U[:, :min(16, U.shape[1])]
                layer_S[l] = S
            except:
                pass

        # 相邻层子空间角度(输运角)
        transport_result = []
        for l in range(n_layers):
            if l in layer_U and l + 1 in layer_U:
                angles = principal_angles(layer_U[l], layer_U[l + 1], k1=8, k2=8)
                cos_sim = subspace_cosine(layer_U[l], layer_U[l + 1], k1=8, k2=8)
                transport_result.append({
                    "layer": l,
                    "mean_angle_deg": float(np.mean(np.degrees(angles))),
                    "max_angle_deg": float(np.max(np.degrees(angles))),
                    "subspace_cos_8d": float(cos_sim),
                })

        results["transport_angles"][sent] = transport_result

        # 估计per-layer Jacobian: J_l ≈ Y_{l+1} @ Y_l^+
        per_layer_result = []
        for l in range(min(n_layers, len(layer_S) - 1)):
            Y_l = Y_all[l]
            Y_lp1 = Y_all[l + 1]

            # 截断SVD求伪逆
            try:
                U_l, S_l, Vh_l = np.linalg.svd(Y_l, full_matrices=False)
                # 保留显著奇异值
                threshold = max(S_l[0] * 1e-6, 1e-10)
                k_keep = max(int(np.sum(S_l > threshold)), 1)
                k_keep = min(k_keep, 32)

                # J_l ≈ Y_{l+1} @ Vh_l[:k_keep].T @ diag(1/S_l[:k_keep]) @ U_l[:, :k_keep].T
                S_inv = 1.0 / np.maximum(S_l[:k_keep], 1e-10)
                J_est = Y_lp1 @ (Vh_l[:k_keep].T * S_inv) @ U_l[:, :k_keep].T

                # J_l的SVD
                U_J, S_J, _ = np.linalg.svd(J_est, full_matrices=False)
                eff_rank = compute_effective_rank(S_J, 0.99)

                per_layer_result.append({
                    "layer": l,
                    "eff_rank": int(eff_rank),
                    "top_sv": S_J[:16].tolist(),
                    "sv_decay_ratio": float(S_J[0] / max(S_J[min(7, len(S_J) - 1)], 1e-10)),
                })
            except:
                pass

        results["per_layer_jacobian"][sent] = per_layer_result

        # 秩演化: 各层累积Jacobian的有效秩
        rank_evol = []
        for l in range(n_layers + 1):
            if l in layer_S:
                rank_evol.append({
                    "layer": l,
                    "eff_rank_99": compute_effective_rank(layer_S[l], 0.99),
                    "eff_rank_90": compute_effective_rank(layer_S[l], 0.90),
                    "top_sv_ratio": float(layer_S[l][0] / max(layer_S[l][min(7, len(layer_S[l]) - 1)], 1e-10)),
                })
        results["rank_evolution"][sent] = rank_evol

    # 汇总
    agg_transport = defaultdict(list)
    for sent, transport in results["transport_angles"].items():
        for t in transport:
            agg_transport[t["layer"]].append(t["subspace_cos_8d"])

    results["aggregate_transport"] = {
        lk: {"mean_cos": float(np.mean(v)), "std_cos": float(np.std(v))}
        for lk, v in sorted(agg_transport.items())
    }

    return results


# ============================================================
# Exp 5: Tokenization控制实验
# ============================================================

def exp5_tokenization_control(model, tokenizer, device, model_info):
    """
    控制: "doesn't" vs "does not" — 同一语义, 不同tokenization.

    核心问题:
    - 不同tokenization是否导致不同的算子结构?
    - Phase 131的"双重否定≠逆元"是否因为tokenization差异?

    方法:
    1. 对比缩写形式 vs 完整形式的hidden states差异
    2. 对比两者的Jacobian谱结构
    3. 对比两者对其他约束的响应
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    results = {}

    for entry in BASE_SENTENCES[:5]:
        sent_key = entry["s"]
        full_neg = make_sentence(entry, "negation")   # "does not"
        contract = make_sentence(entry, "contraction") # "doesn't"
        base_sent = make_sentence(entry, "base")

        hs_full = get_hidden_states_all(model, tokenizer, device, full_neg)
        hs_contract = get_hidden_states_all(model, tokenizer, device, contract)
        hs_base = get_hidden_states_all(model, tokenizer, device, base_sent)

        # 差异分析
        layer_results = []
        for l in range(n_layers + 1):
            delta_full = hs_full[l] - hs_base[l]
            delta_contract = hs_contract[l] - hs_base[l]

            norm_full = np.linalg.norm(delta_full)
            norm_contract = np.linalg.norm(delta_contract)
            cos_fc = float(np.dot(delta_full, delta_contract) /
                          max(norm_full * norm_contract, 1e-10))

            # 差异方向是否在约束子空间中?
            diff_vec = delta_full - delta_contract
            norm_diff = np.linalg.norm(diff_vec)
            rel_diff = norm_diff / max((norm_full + norm_contract) / 2, 1e-10)

            layer_results.append({
                "norm_full": float(norm_full),
                "norm_contract": float(norm_contract),
                "cos_full_contract": float(cos_fc),
                "norm_diff": float(norm_diff),
                "rel_diff": float(rel_diff),
            })

        results[sent_key] = layer_results

    # 额外: 对比"does not" vs "doesn't"的Jacobian谱
    print("  [Tokenization Jacobian] 缩写vs完整的算子谱...")
    entry = BASE_SENTENCES[0]
    full_neg = make_sentence(entry, "negation")
    contract = make_sentence(entry, "contraction")

    k_perturb = 32
    eps = 1e-4
    rng = np.random.RandomState(777)
    V = rng.randn(d_model, k_perturb).astype(np.float32)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)

    jacobian_comparison = {}
    for label, sent in [("full", full_neg), ("contract", contract)]:
        hs_base = get_hidden_states_all(model, tokenizer, device, sent)
        Y_all = np.zeros((n_layers + 1, d_model, k_perturb), dtype=np.float32)
        for j in range(k_perturb):
            hs_pert = get_all_hidden_states_with_perturbation(
                model, tokenizer, device, sent,
                perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
            )
            for l in range(n_layers + 1):
                Y_all[l, :, j] = (hs_pert[l] - hs_base[l]) / eps

        sv_dict = {}
        for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            try:
                U, S, Vh = np.linalg.svd(Y_all[l], full_matrices=False)
                sv_dict[f"L{l}"] = {
                    "eff_rank_99": compute_effective_rank(S, 0.99),
                    "top5_sv": S[:5].tolist(),
                }
            except:
                pass

        jacobian_comparison[label] = sv_dict

    results["jacobian_comparison"] = jacobian_comparison

    return results


# ============================================================
# 主函数
# ============================================================

def run_all_experiments(model_name):
    """运行所有实验"""
    print(f"\n{'='*70}")
    print(f"Phase 132: 算子流形分析 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  模型: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")

    results = {
        "model_info": {
            "name": model_name,
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
        },
    }

    t0 = time.time()

    # Exp 1
    print(f"\n--- Exp 1: Jacobian谱结构 ---")
    t1 = time.time()
    results["exp1_jacobian_spectrum"] = exp1_jacobian_spectrum(model, tokenizer, device, model_info)
    print(f"  耗时: {time.time()-t1:.1f}s")

    # Exp 2
    print(f"\n--- Exp 2: 算子子空间比较 ---")
    t2 = time.time()
    results["exp2_operator_subspace"] = exp2_operator_subspace_comparison(model, tokenizer, device, model_info)
    print(f"  耗时: {time.time()-t2:.1f}s")

    # Exp 3
    print(f"\n--- Exp 3: 约束对易子(修正版) ---")
    t3 = time.time()
    results["exp3_commutator"] = exp3_constraint_commutator(model, tokenizer, device, model_info)
    print(f"  耗时: {time.time()-t3:.1f}s")

    # Exp 4
    print(f"\n--- Exp 4: 低秩输运分析 ---")
    t4 = time.time()
    results["exp4_transport"] = exp4_low_rank_transport(model, tokenizer, device, model_info)
    print(f"  耗时: {time.time()-t4:.1f}s")

    # Exp 5
    print(f"\n--- Exp 5: Tokenization控制 ---")
    t5 = time.time()
    results["exp5_tokenization"] = exp5_tokenization_control(model, tokenizer, device, model_info)
    print(f"  耗时: {time.time()-t5:.1f}s")

    total_time = time.time() - t0
    results["total_time"] = total_time
    print(f"\n总耗时: {total_time:.1f}s")

    # 保存
    out_path = f"tests/glm5_temp/phase132_{model_name}_operator_manifold.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"结果已保存: {out_path}")

    # 打印摘要
    print_summary(results)

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_summary(results):
    """打印关键发现摘要"""
    mi = results["model_info"]
    print(f"\n{'='*70}")
    print(f"Phase 132 摘要: {mi['name']} (L={mi['n_layers']}, d={mi['d_model']})")
    print(f"{'='*70}")

    # Exp 1: Jacobian有效秩
    exp1 = results.get("exp1_jacobian_spectrum", {})
    agg_ranks = exp1.get("aggregate", {}).get("eff_ranks", {})
    if agg_ranks:
        n_layers = mi["n_layers"]
        sample = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
        print("\n[Exp 1] Jacobian有效秩 (累积Jacobian J_{l:0}):")
        for l in sample:
            lk = f"L{l}"
            if lk in agg_ranks:
                d = agg_ranks[lk]
                print(f"  {lk}: eff_rank = {d['mean']:.1f} ± {d['std']:.1f}")

    # Exp 2: 算子子空间比较
    exp2 = results.get("exp2_operator_subspace", {})
    layer_analysis = exp2.get("layer_analysis", {})
    if layer_analysis:
        print("\n[Exp 2] 跨句子Jacobian子空间cosine (8维):")
        for lk in sorted(layer_analysis.keys()):
            d = layer_analysis[lk]
            print(f"  {lk}: 同语义不同语法={d['same_sem_diff_syn']['mean']:.3f} "
                  f"(n={d['same_sem_diff_syn']['n']}), "
                  f"不同语义同语法={d['diff_sem_same_syn']['mean']:.3f} "
                  f"(n={d['diff_sem_same_syn']['n']}), "
                  f"不同语义不同语法={d['diff_sem_diff_syn']['mean']:.3f} "
                  f"(n={d['diff_sem_diff_syn']['n']})")

    # Exp 3: 对易子
    exp3 = results.get("exp3_commutator", {})
    sample_layers = [0, mi["n_layers"] // 2, mi["n_layers"]]
    if exp3:
        print("\n[Exp 3] 约束对易子 nl_ratio (关键层):")
        # 取第一个句子的结果
        for sk in list(exp3.keys()):
            if sk == "double_negation_corrected":
                continue
            if not isinstance(exp3[sk], dict):
                continue
            for pk, pd in exp3[sk].items():
                lr = pd.get("layer_results", [])
                vals = []
                for l in sample_layers:
                    if l < len(lr):
                        vals.append(f"{lr[l]['nl_ratio']:.3f}")
                    else:
                        vals.append("  -  ")
                print(f"  {sk}/{pk}: " + " / ".join(vals))
            break  # 只打印一个句子

        # 双重否定
        dn = exp3.get("double_negation_corrected", {})
        if dn:
            print("\n[Exp 3] 双重否定(修正版):")
            for sk in list(dn.keys())[:2]:
                lr = dn[sk].get("layer_results", [])
                mid = len(lr) // 2
                if mid < len(lr):
                    d = lr[mid]
                    print(f"  {sk} (L{mid}): cos(neg,dbl_neg)={d['cos_neg_dblneg']:.3f}, "
                          f"recovery_ratio={d['recovery_ratio']:.3f}")

    # Exp 4: 输运角
    exp4 = results.get("exp4_transport", {})
    agg_t = exp4.get("aggregate_transport", {})
    if agg_t:
        print("\n[Exp 4] 子空间输运cosine (8维, 相邻层):")
        n_layers = mi["n_layers"]
        sample = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        for l in sample:
            lk = str(l)
            if lk in agg_t:
                d = agg_t[lk]
                print(f"  L{l}->L{l+1}: cos={d['mean_cos']:.3f} ± {d['std_cos']:.3f}")

    # Exp 5: Tokenization
    exp5 = results.get("exp5_tokenization", {})
    if exp5:
        print("\n[Exp 5] Tokenization控制 (does not vs doesn't):")
        for sk in list(exp5.keys()):
            if sk == "jacobian_comparison":
                continue
            lr = exp5[sk]
            if isinstance(lr, list) and len(lr) > 0:
                mid = len(lr) // 2
                d = lr[mid]
                print(f"  {sk} (L{mid}): cos={d['cos_full_contract']:.3f}, "
                      f"rel_diff={d['rel_diff']:.3f}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_all_experiments(model_name)
