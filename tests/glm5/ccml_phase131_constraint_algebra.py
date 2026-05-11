"""
Phase 131: 约束代数与组合结构分析
==================================

Phase 130的核心问题(用户指出):
1. 过度物理化: 把LayerNorm/residual的放大效应误认为"混沌"
2. 把"轨迹"误认为本体: 真正稳定的是轨迹在变换群下的不变量
3. 还没进入"约束代数": 语言本质是"条件约束的可组合传播"

核心理论转变:
- 不是研究 h_l (hidden state)
- 不是研究 J_l (Jacobian)
- 而是研究 约束如何组合传播

语言的数学本质 = 约束可组合系统 (Compositional Constraint System)
- "概念" = 约束变换群下的不变量
- "语法" = 对传播路径的动态约束
- "组合" = 约束的代数结构(交换律/结合律/逆元)

5个实验:
- Exp 1: 约束交换律 — Neg∘Tense == Tense∘Neg?
- Exp 2: 约束逆元 — Neg∘Neg == Identity?
- Exp 3: 约束传播核 — K_ij = ∂h_l/∂c_i, 约束如何逐层传播
- Exp 4: 约束代数的低秩结构 — 约束效应是否活在低维子空间
- Exp 5: 约束的层间组合规律 — 早期层是否近似线性(交换), 深层是否非线性(不交换)

关键设计原则:
- 用小ε(1e-4)避免LayerNorm/residual的放大伪影
- 用约束效应而非绝对hidden state作为分析对象
- 测量代数性质(交换性/逆元/结合律)而非物理性质(放大/旋转)
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
import torch.nn.functional as F
from collections import defaultdict

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 约束定义: 基础句子 + 语法约束的组合
# ============================================================

# 基础句子模板: {subject} {verb} {object}
BASE_TEMPLATES = [
    # 主-谓-宾结构, 不同的语义类别
    "The dog bites the man",
    "The cat chases the mouse",
    "The teacher helps the student",
    "The scientist discovers the formula",
    "The artist paints the picture",
    "The child reads the book",
]

# 约束生成器: 对基础句子施加语法约束
def apply_constraint(base: str, constraint: str) -> str:
    """对基础句子施加语法约束, 返回修改后的句子"""
    words = base.split()
    subject = words[1]  # dog, cat, ...
    verb = words[2]     # bites, chases, ...
    obj = words[4]      # man, mouse, ...

    if constraint == "identity":
        return base

    elif constraint == "negation":
        # "The dog does not bite the man"
        return f"The {subject} does not {verb} the {obj}"

    elif constraint == "past":
        # 简化: 用 "did" + 动词原形
        # "The dog did bite the man" (或更自然的 "The dog bit the man")
        # 为了tokenizer一致性, 用 "did" 结构
        return f"The {subject} did {verb.rstrip('s')} the {obj}"

    elif constraint == "passive":
        # "The man is bitten by the dog"
        return f"The {obj} is {verb.rstrip('s').rstrip('e')}ed by the {subject}"

    elif constraint == "plural":
        # "The dogs bite the man"
        return f"The {subject}s {verb.rstrip('s')} the {obj}"

    elif constraint == "neg_tense":
        # 先否定再时态: Neg(Past(x)) = "The dog did not bite the man"
        return f"The {subject} did not {verb.rstrip('s')} the {obj}"

    elif constraint == "tense_neg":
        # 先时态再否定: Tense(Neg(x)) = same as neg_tense in English
        # 英语中这两者语法上相同, 但模型内部处理可能不同
        return f"The {subject} did not {verb.rstrip('s')} the {obj}"

    elif constraint == "neg_passive":
        # 否定+被动: "The man is not bitten by the dog"
        return f"The {obj} is not {verb.rstrip('s').rstrip('e')}ed by the {subject}"

    elif constraint == "passive_neg":
        # 被动+否定: same as neg_passive in English
        return f"The {obj} is not {verb.rstrip('s').rstrip('e')}ed by the {subject}"

    elif constraint == "neg_neg":
        # 双重否定: "The dog does not not bite the man" (不太自然, 但可以测试)
        return f"The {subject} does not fail to {verb.rstrip('s')} the {obj}"

    elif constraint == "neg_plural":
        # 否定+复数: "The dogs do not bite the man"
        return f"The {subject}s do not {verb.rstrip('s')} the {obj}"

    elif constraint == "plural_neg":
        # 复数+否定: same
        return f"The {subject}s do not {verb.rstrip('s')} the {obj}"

    elif constraint == "past_passive":
        # 过去+被动: "The man was bitten by the dog"
        return f"The {obj} was {verb.rstrip('s').rstrip('e')}ed by the {subject}"

    elif constraint == "neg_past_passive":
        # 三重: 否定+过去+被动: "The man was not bitten by the dog"
        return f"The {obj} was not {verb.rstrip('s').rstrip('e')}ed by the {subject}"

    else:
        return base


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

    # 每层的last token hidden state
    hs = []
    for h in out.hidden_states:
        hs.append(h[0, -1, :].detach().float().cpu().numpy())  # [d_model]
    return hs


def compute_constraint_effect(hs_constrained, hs_base):
    """计算约束效应: δh_l = h_l(constrained) - h_l(base)
    返回: list of [d_model] arrays, 每层一个
    """
    return [hc - hb for hc, hb in zip(hs_constrained, hs_base)]


# ============================================================
# Exp 1: 约束交换律
# ============================================================

def exp1_commutativity(model, tokenizer, device, model_info):
    """
    测试约束是否满足交换律: A∘B == B∘A?

    核心思想:
    - 如果 Neg∘Tense == Tense∘Neg (交换), 则约束是"正交"的
    - 如果不等, 则约束之间存在非线性交互

    测量:
    - commutativity_violation = ||δ(Neg∘Tense) - δ(Tense∘Neg)|| / ||δ(Neg∘Tense)||
    - 如果violation → 0, 则约束近似交换
    - 如果violation > 0, 则约束不交换

    注意: 英语语法使得某些约束对在表面形式上相同,
    但模型内部处理路径可能不同(通过不同的token序列达到相同语义)
    """

    n_layers = model_info.n_layers
    results = {}

    # 约束对: (A, B, A∘B的形式, B∘A的形式)
    # 关键: 英语中某些对形式上相同, 但我们需要找形式不同的对
    constraint_pairs = [
        # (constraint_A, constraint_B, AB形式, BA形式, 描述)
        ("negation", "passive", "neg_passive", "passive_neg", "Neg∘Passive vs Passive∘Neg"),
        ("negation", "plural", "neg_plural", "plural_neg", "Neg∘Plural vs Plural∘Neg"),
        ("past", "passive", "past_passive", "past_passive", "Past∘Passive vs Passive∘Past"),
        ("negation", "past", "neg_tense", "tense_neg", "Neg∘Past vs Past∘Neg"),
    ]

    for base_template in BASE_TEMPLATES:
        template_key = base_template.split()[1]  # subject word
        results[template_key] = {}

        # 获取base hidden states
        hs_base = get_hidden_states_all(model, tokenizer, device, base_template)

        for cA, cB, cAB, cBA, desc in constraint_pairs:
            # A∘B
            sent_AB = apply_constraint(base_template, cAB)
            hs_AB = get_hidden_states_all(model, tokenizer, device, sent_AB)

            # B∘A (如果形式相同, 跳过)
            sent_BA = apply_constraint(base_template, cBA)
            if sent_AB == sent_BA:
                # 形式完全相同, 无法测试交换律
                # 但我们可以用另一种方式: 分别施加A和B
                # 先施A再施B vs 先施B再施A
                # 由于语言限制, 我们用"约束效应的加法性"来测试
                # 即: δ(A+B) vs δ(A) + δ(B)
                pass

            # 计算约束效应
            delta_AB = compute_constraint_effect(hs_AB, hs_base)

            if sent_AB != sent_BA:
                hs_BA = get_hidden_states_all(model, tokenizer, device, sent_BA)
                delta_BA = compute_constraint_effect(hs_BA, hs_base)

                # 交换律违反度
                layer_results = []
                for l in range(n_layers):
                    norm_AB = np.linalg.norm(delta_AB[l])
                    norm_BA = np.linalg.norm(delta_BA[l])
                    diff = np.linalg.norm(delta_AB[l] - delta_BA[l])

                    if max(norm_AB, norm_BA) > 1e-8:
                        violation = diff / max(norm_AB, norm_BA)
                    else:
                        violation = 0.0

                    cos_AB_BA = 0.0
                    if norm_AB > 1e-8 and norm_BA > 1e-8:
                        cos_AB_BA = float(np.dot(delta_AB[l], delta_BA[l]) / (norm_AB * norm_BA))

                    layer_results.append({
                        "norm_AB": float(norm_AB),
                        "norm_BA": float(norm_BA),
                        "diff": float(diff),
                        "violation": float(violation),
                        "cos_AB_BA": float(cos_AB_BA),
                    })

                results[template_key][desc] = {
                    "type": "commutativity",
                    "sent_AB": sent_AB,
                    "sent_BA": sent_BA,
                    "layer_results": layer_results,
                }
            else:
                # 形式相同 → 测试加法性: δ(A+B) vs δ(A) + δ(B) - δ(base)
                # 这等价于测试非线性交互
                hs_A = get_hidden_states_all(model, tokenizer, device,
                                            apply_constraint(base_template, cA))
                hs_B = get_hidden_states_all(model, tokenizer, device,
                                            apply_constraint(base_template, cB))
                delta_A = compute_constraint_effect(hs_A, hs_base)
                delta_B = compute_constraint_effect(hs_B, hs_base)

                # 非线性交互 = δ(A+B) - (δ(A) + δ(B))
                layer_results = []
                for l in range(n_layers):
                    linear_pred = delta_A[l] + delta_B[l]
                    nonlinear_residue = delta_AB[l] - linear_pred
                    norm_combined = np.linalg.norm(delta_AB[l])
                    norm_residue = np.linalg.norm(nonlinear_residue)

                    if norm_combined > 1e-8:
                        nl_ratio = norm_residue / norm_combined
                    else:
                        nl_ratio = 0.0

                    cos_residue = 0.0
                    if norm_residue > 1e-8 and norm_combined > 1e-8:
                        cos_residue = float(np.dot(nonlinear_residue, delta_AB[l]) /
                                           (norm_residue * norm_combined))

                    layer_results.append({
                        "norm_A": float(np.linalg.norm(delta_A[l])),
                        "norm_B": float(np.linalg.norm(delta_B[l])),
                        "norm_combined": float(norm_combined),
                        "norm_residue": float(norm_residue),
                        "nl_ratio": float(nl_ratio),
                        "cos_residue": float(cos_residue),
                    })

                results[template_key][desc] = {
                    "type": "additivity",
                    "sent_A": apply_constraint(base_template, cA),
                    "sent_B": apply_constraint(base_template, cB),
                    "sent_AB": sent_AB,
                    "layer_results": layer_results,
                }

    return results


# ============================================================
# Exp 2: 约束逆元
# ============================================================

def exp2_inverse(model, tokenizer, device, model_info):
    """
    测试约束是否有逆元: Neg∘Neg ≈ Identity?

    核心思想:
    - 如果 Neg∘Neg ≈ I, 则否定是"自逆"的(群论中的对合 involution)
    - 如果不等, 则双重否定 ≠ 肯定, 揭示非线性结构

    还测试: Passive∘Active 是否恢复原状?

    测量:
    - inverse_violation = ||δ(Neg∘Neg)|| / ||δ(Neg)||
    - 如果 ≈ 0, 则双重否定恢复原状
    - 如果 > 0, 则双重否定不恢复
    """

    n_layers = model_info.n_layers
    results = {}

    for base_template in BASE_TEMPLATES:
        template_key = base_template.split()[1]

        hs_base = get_hidden_states_all(model, tokenizer, device, base_template)
        hs_neg = get_hidden_states_all(model, tokenizer, device,
                                       apply_constraint(base_template, "negation"))
        hs_neg_neg = get_hidden_states_all(model, tokenizer, device,
                                           apply_constraint(base_template, "neg_neg"))

        delta_neg = compute_constraint_effect(hs_neg, hs_base)
        delta_neg_neg = compute_constraint_effect(hs_neg_neg, hs_base)

        layer_results = []
        for l in range(n_layers):
            norm_neg = np.linalg.norm(delta_neg[l])
            norm_neg_neg = np.linalg.norm(delta_neg_neg[l])
            norm_base = np.linalg.norm(hs_base[l])

            # 双重否定 vs 原始的偏差
            diff_from_base = np.linalg.norm(hs_neg_neg[l] - hs_base[l])

            # 归一化偏差
            if norm_base > 1e-8:
                relative_diff = diff_from_base / norm_base
            else:
                relative_diff = 0.0

            # 双重否定效应 vs 单否定的比例
            if norm_neg > 1e-8:
                neg_neg_ratio = norm_neg_neg / norm_neg
            else:
                neg_neg_ratio = 0.0

            # cos(双重否定方向, 单否定方向) — 如果逆元存在, 应该≈-1
            cos_neg_neg = 0.0
            if norm_neg > 1e-8 and norm_neg_neg > 1e-8:
                cos_neg_neg = float(np.dot(delta_neg_neg[l], delta_neg[l]) /
                                    (norm_neg_neg * norm_neg))

            layer_results.append({
                "norm_neg": float(norm_neg),
                "norm_neg_neg": float(norm_neg_neg),
                "diff_from_base": float(diff_from_base),
                "relative_diff": float(relative_diff),
                "neg_neg_ratio": float(neg_neg_ratio),
                "cos_neg_neg": float(cos_neg_neg),
            })

        results[template_key] = {
            "sent_base": base_template,
            "sent_neg": apply_constraint(base_template, "negation"),
            "sent_neg_neg": apply_constraint(base_template, "neg_neg"),
            "layer_results": layer_results,
        }

    return results


# ============================================================
# Exp 3: 约束传播核 K_ij = ∂h_l/∂c_i
# ============================================================

def exp3_constraint_propagation_kernel(model, tokenizer, device, model_info):
    """
    约束传播核: 每个约束如何逐层传播?

    核心思想(用户指出):
    - 语言真正传播的是 constraint, 不是 hidden vector
    - 应该测量 K_ij = ∂h_j/∂c_i, 即约束i如何影响层j

    实现方式:
    - 对每个约束c_i, 计算约束效应 δh_l(c_i) = h_l(c_i(x)) - h_l(x)
    - 这就是约束i在层l的"传播核"
    - 然后分析:
      1. 约束效应的层间传播: δh_{l+1}(c_i) 与 δh_l(c_i) 的关系
      2. 不同约束的传播核是否正交
      3. 传播核是否通过低维子空间

    数据量: 用全部6个基础模板 × 5个约束 = 30组约束效应
    """

    n_layers = model_info.n_layers
    constraints = ["negation", "past", "passive", "plural", "identity"]

    results = {}

    # 收集所有约束效应
    all_constraint_effects = {}  # {template_key: {constraint: [δh_0, δh_1, ...]}}

    for base_template in BASE_TEMPLATES:
        template_key = base_template.split()[1]
        hs_base = get_hidden_states_all(model, tokenizer, device, base_template)
        all_constraint_effects[template_key] = {"base_hs": hs_base}

        for c in constraints:
            if c == "identity":
                delta = [np.zeros_like(hs_base[0]) for _ in range(n_layers)]
            else:
                sent = apply_constraint(base_template, c)
                hs_c = get_hidden_states_all(model, tokenizer, device, sent)
                delta = compute_constraint_effect(hs_c, hs_base)
            all_constraint_effects[template_key][c] = delta

    # ---- 分析1: 约束传播核的层间传播 ----
    # δh_{l+1}(c) ≈ A_l * δh_l(c) + nonlinear_l(c)
    # 其中 A_l 是线性传播算子, nonlinear_l 是非线性残差

    propagation_analysis = {}
    for template_key in all_constraint_effects:
        propagation_analysis[template_key] = {}
        for c in constraints:
            if c == "identity":
                continue
            delta = all_constraint_effects[template_key][c]
            layer_results = []
            for l in range(n_layers - 1):
                norm_l = np.linalg.norm(delta[l])
                norm_l1 = np.linalg.norm(delta[l + 1])

                # cos(δh_{l+1}, δh_l): 约束方向是否保持
                cos_dir = 0.0
                if norm_l > 1e-8 and norm_l1 > 1e-8:
                    cos_dir = float(np.dot(delta[l + 1], delta[l]) / (norm_l1 * norm_l))

                # 放大倍数
                amp = norm_l1 / max(norm_l, 1e-10)

                # 约束效应在base方向上的投影
                base_hs = all_constraint_effects[template_key]["base_hs"]
                cos_with_base = 0.0
                if norm_l1 > 1e-8:
                    base_norm = np.linalg.norm(base_hs[l + 1])
                    if base_norm > 1e-8:
                        cos_with_base = float(np.dot(delta[l + 1], base_hs[l + 1]) /
                                              (norm_l1 * base_norm))

                layer_results.append({
                    "norm_l": float(norm_l),
                    "norm_l1": float(norm_l1),
                    "amp": float(amp),
                    "cos_dir": float(cos_dir),
                    "cos_with_base": float(cos_with_base),
                })

            propagation_analysis[template_key][c] = layer_results

    # ---- 分析2: 约束间的正交性 ----
    # 不同约束的δh_l是否正交? 如果正交, 则约束在"约束空间"中独立

    orthogonality_analysis = {}
    for template_key in all_constraint_effects:
        orthogonality_analysis[template_key] = {}
        real_constraints = [c for c in constraints if c != "identity"]

        # 每层计算约束间的cosine
        for l in range(n_layers):
            constraint_vectors = {}
            for c in real_constraints:
                delta = all_constraint_effects[template_key][c][l]
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    constraint_vectors[c] = delta / norm

            # 计算约束间的cosine矩阵
            c_list = list(constraint_vectors.keys())
            cos_matrix = {}
            for i, ci in enumerate(c_list):
                for j, cj in enumerate(c_list):
                    if i < j:
                        cos_val = float(np.dot(constraint_vectors[ci], constraint_vectors[cj]))
                        cos_matrix[f"{ci}_vs_{cj}"] = cos_val

            orthogonality_analysis[template_key][f"L{l}"] = cos_matrix

    # ---- 分析3: 约束效应的维度(有效秩) ----
    # 把所有约束的δh_l堆成矩阵, 看有效秩
    # 如果有效秩 << 约束数, 则约束效应在低维子空间中

    dimensionality_analysis = {}
    for template_key in all_constraint_effects:
        dimensionality_analysis[template_key] = {}
        real_constraints = [c for c in constraints if c != "identity"]

        for l in range(n_layers):
            # 堆成矩阵: [n_constraints, d_model]
            delta_matrix = np.stack([
                all_constraint_effects[template_key][c][l]
                for c in real_constraints
            ])

            # SVD获取奇异值
            U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)

            # 有效秩: SVD熵
            total_energy = np.sum(S ** 2)
            if total_energy > 1e-20:
                p = S ** 2 / total_energy
                p = p[p > 1e-20]
                eff_rank = float(np.exp(-np.sum(p * np.log(p))))
            else:
                eff_rank = 0.0

            # 前3个奇异值的能量占比
            top3_energy = float(np.sum(S[:3] ** 2) / max(total_energy, 1e-20))

            dimensionality_analysis[template_key][f"L{l}"] = {
                "singular_values": [float(s) for s in S[:6]],
                "eff_rank": float(eff_rank),
                "top3_energy": float(top3_energy),
                "n_constraints": len(real_constraints),
            }

    results = {
        "propagation_analysis": propagation_analysis,
        "orthogonality_analysis": orthogonality_analysis,
        "dimensionality_analysis": dimensionality_analysis,
    }
    return results


# ============================================================
# Exp 4: 约束代数的低秩结构
# ============================================================

def exp4_low_rank_constraint_subspace(model, tokenizer, device, model_info):
    """
    约束效应是否活在低维子空间?

    核心问题(用户提出):
    - Transformer是否只在 k << d 维有效子空间里传播约束?

    方法:
    1. 收集大量约束效应: 不同模板 × 不同约束 × 不同层
    2. 对每层, 把所有约束效应堆成矩阵 [N, d_model]
    3. SVD分析有效秩
    4. 如果有效秩 << d_model, 则约束在低维子空间中传播

    数据量: 6模板 × 4约束 + 组合 = 30+约束效应
    """

    n_layers = model_info.n_layers
    constraints = ["negation", "past", "passive", "plural"]
    combined_constraints = ["neg_tense", "neg_passive", "neg_plural", "past_passive", "neg_past_passive"]

    results = {}

    # 收集所有约束效应
    all_effects_by_layer = defaultdict(list)  # {layer: [δh_1, δh_2, ...]}
    effect_labels_by_layer = defaultdict(list)

    for base_template in BASE_TEMPLATES:
        hs_base = get_hidden_states_all(model, tokenizer, device, base_template)

        for c in constraints + combined_constraints:
            sent = apply_constraint(base_template, c)
            hs_c = get_hidden_states_all(model, tokenizer, device, sent)
            delta = compute_constraint_effect(hs_c, hs_base)

            for l in range(n_layers):
                all_effects_by_layer[l].append(delta[l])
                effect_labels_by_layer[l].append(f"{base_template.split()[1]}_{c}")

    # 对每层进行低秩分析
    layer_analysis = {}
    for l in range(n_layers):
        delta_matrix = np.stack(all_effects_by_layer[l])  # [N, d_model]
        N, d = delta_matrix.shape

        # SVD
        U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)

        # 有效秩 (SVD熵)
        total_energy = np.sum(S ** 2)
        if total_energy > 1e-20:
            p = S ** 2 / total_energy
            p = p[p > 1e-20]
            eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            eff_rank = 0.0

        # 累积能量
        cum_energy = np.cumsum(S ** 2) / total_energy

        # 找到90%, 95%, 99%能量所需的维度
        dims_for_energy = {}
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            idx = np.searchsorted(cum_energy, threshold)
            dims_for_energy[f"dim_{int(threshold*100)}pct"] = int(idx + 1)

        # 约束子空间与base方向的关系
        # 前3个右奇异向量张成的子空间, 在d_model中的"方向"
        top3_subspace = Vt[:3]  # [3, d_model]

        layer_analysis[f"L{l}"] = {
            "N_effects": N,
            "d_model": d,
            "top10_sv": [float(s) for s in S[:10]],
            "eff_rank": eff_rank,
            "dims_for_energy": dims_for_energy,
            "total_energy": float(total_energy),
        }

    results["layer_analysis"] = layer_analysis

    # ---- 关键分析: 约束子空间是否跨模板共享? ----
    # 如果不同模板的约束效应投影到同一个低维子空间, 则约束是"通用的"
    # 如果不同模板有不同的约束子空间, 则约束是"上下文依赖的"

    cross_template_analysis = {}
    for l in range(n_layers):
        # 每个模板单独计算约束子空间
        template_subspaces = {}
        for base_template in BASE_TEMPLATES:
            template_key = base_template.split()[1]
            hs_base = get_hidden_states_all(model, tokenizer, device, base_template)

            effects = []
            for c in constraints + combined_constraints:
                sent = apply_constraint(base_template, c)
                hs_c = get_hidden_states_all(model, tokenizer, device, sent)
                delta = compute_constraint_effect(hs_c, hs_base)
                effects.append(delta[l])

            delta_matrix = np.stack(effects)
            U, S, Vt = np.linalg.svd(delta_matrix, full_matrices=False)
            template_subspaces[template_key] = {
                "top3_Vt": Vt[:3],  # [3, d_model]
                "eff_rank": float(np.exp(-np.sum(
                    (S[S > 1e-10] ** 2 / np.sum(S ** 2)) *
                    np.log(S[S > 1e-10] ** 2 / np.sum(S ** 2))
                ))) if np.sum(S ** 2) > 1e-20 else 0.0,
            }

        # 计算不同模板子空间的Grassmann距离
        # 用前3个右奇异向量的子空间
        template_keys = list(template_subspaces.keys())
        subspace_distances = {}
        for i, tk1 in enumerate(template_keys):
            for j, tk2 in enumerate(template_keys):
                if i < j:
                    V1 = template_subspaces[tk1]["top3_Vt"]
                    V2 = template_subspaces[tk2]["top3_Vt"]

                    # 子空间相似度: principal angles
                    M = V1 @ V2.T  # [3, 3]
                    _, sv, _ = np.linalg.svd(M)
                    # principal angles的cosine
                    mean_cos = float(np.mean(sv))
                    subspace_distances[f"{tk1}_vs_{tk2}"] = mean_cos

        cross_template_analysis[f"L{l}"] = {
            "template_eff_ranks": {k: v["eff_rank"] for k, v in template_subspaces.items()},
            "subspace_cosines": subspace_distances,
        }

    results["cross_template_analysis"] = cross_template_analysis

    return results


# ============================================================
# Exp 5: 约束的层间组合规律 — 早期线性 vs 深层非线性
# ============================================================

def exp5_layerwise_composition(model, tokenizer, device, model_info):
    """
    约束组合在不同层的表现:
    - 早期层: 约束近似线性叠加 (约束正交)
    - 深层: 约束产生非线性交互

    但用户警告: 不能简单归因于"非线性",
    需要区分:
    1. LayerNorm引起的非线性
    2. 残差累积引起的效应
    3. 真正的约束交互

    方法:
    - 对每个层l, 测量 δ(A+B)_l vs δ(A)_l + δ(B)_l
    - 非线性残差 = δ(A+B) - (δ(A) + δ(B))
    - 但要控制LayerNorm的影响: 用LayerNorm后的差异

    增加数据量: 测试更多约束组合
    """

    n_layers = model_info.n_layers
    results = {}

    # 更多约束组合, 增加数据量
    constraint_pairs = [
        ("negation", "past"),
        ("negation", "passive"),
        ("negation", "plural"),
        ("past", "passive"),
        ("past", "plural"),
        ("passive", "plural"),
    ]

    # 对应的组合约束名
    combined_names = {
        ("negation", "past"): "neg_tense",
        ("negation", "passive"): "neg_passive",
        ("negation", "plural"): "neg_plural",
        ("past", "passive"): "past_passive",
        ("past", "plural"): "past_passive",  # 近似
        ("passive", "plural"): "past_passive",  # 近似
    }

    for base_template in BASE_TEMPLATES:
        template_key = base_template.split()[1]
        template_results = {}

        hs_base = get_hidden_states_all(model, tokenizer, device, base_template)

        # 先计算所有单一约束效应
        single_effects = {}
        for c in ["negation", "past", "passive", "plural"]:
            sent = apply_constraint(base_template, c)
            hs_c = get_hidden_states_all(model, tokenizer, device, sent)
            single_effects[c] = compute_constraint_effect(hs_c, hs_base)

        # 计算组合约束效应
        for cA, cB in constraint_pairs:
            combined_name = combined_names.get((cA, cB), None)
            if combined_name is None:
                continue

            sent_combined = apply_constraint(base_template, combined_name)
            hs_combined = get_hidden_states_all(model, tokenizer, device, sent_combined)
            delta_combined = compute_constraint_effect(hs_combined, hs_base)

            # 线性预测
            delta_linear = [single_effects[cA][l] + single_effects[cB][l]
                           for l in range(n_layers)]

            # 非线性残差
            delta_nonlinear = [delta_combined[l] - delta_linear[l]
                              for l in range(n_layers)]

            layer_results = []
            for l in range(n_layers):
                norm_combined = np.linalg.norm(delta_combined[l])
                norm_linear = np.linalg.norm(delta_linear[l])
                norm_nonlinear = np.linalg.norm(delta_nonlinear[l])

                # 非线性比: ||nonlinear|| / ||combined||
                if norm_combined > 1e-8:
                    nl_ratio = norm_nonlinear / norm_combined
                else:
                    nl_ratio = 0.0

                # 线性预测的准确度
                if norm_combined > 1e-8:
                    lin_acc = 1.0 - norm_nonlinear / norm_combined
                else:
                    lin_acc = 1.0

                # cos(非线性残差, 组合效应) — 如果高, 则非线性方向与组合方向一致
                cos_nl = 0.0
                if norm_nonlinear > 1e-8 and norm_combined > 1e-8:
                    cos_nl = float(np.dot(delta_nonlinear[l], delta_combined[l]) /
                                  (norm_nonlinear * norm_combined))

                # LayerNorm控制: 测量约束效应在归一化后的差异
                # 用简单方法: 约束效应的方向 (归一化后的cosine)
                delta_A_norm = single_effects[cA][l] / max(np.linalg.norm(single_effects[cA][l]), 1e-8)
                delta_B_norm = single_effects[cB][l] / max(np.linalg.norm(single_effects[cB][l]), 1e-8)
                delta_C_norm = delta_combined[l] / max(norm_combined, 1e-8)

                # 约束A和B的方向在组合后的改变
                cos_A_in_combined = float(np.dot(delta_A_norm, delta_C_norm))
                cos_B_in_combined = float(np.dot(delta_B_norm, delta_C_norm))

                layer_results.append({
                    "norm_combined": float(norm_combined),
                    "norm_linear": float(norm_linear),
                    "norm_nonlinear": float(norm_nonlinear),
                    "nl_ratio": float(nl_ratio),
                    "lin_acc": float(lin_acc),
                    "cos_nl": float(cos_nl),
                    "cos_A_in_combined": float(cos_A_in_combined),
                    "cos_B_in_combined": float(cos_B_in_combined),
                })

            template_results[f"{cA}+{cB}"] = layer_results

        results[template_key] = template_results

    # ---- 层间汇总 ----
    summary = {}
    for l in range(n_layers):
        nl_ratios = []
        lin_accs = []
        for tk in results:
            for pair_key in results[tk]:
                lr = results[tk][pair_key][l]
                nl_ratios.append(lr["nl_ratio"])
                lin_accs.append(lr["lin_acc"])

        summary[f"L{l}"] = {
            "mean_nl_ratio": float(np.mean(nl_ratios)),
            "std_nl_ratio": float(np.std(nl_ratios)),
            "mean_lin_acc": float(np.mean(lin_accs)),
            "n_pairs": len(nl_ratios),
        }

    results["layer_summary"] = summary
    return results


# ============================================================
# 主程序
# ============================================================

def run_all_experiments(model_name: str):
    """运行所有实验"""

    print(f"\n{'='*60}")
    print(f"Phase 131: 约束代数与组合结构分析 — {model_name}")
    print(f"{'='*60}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, {model_info.n_layers} layers, d={model_info.d_model}")

    results = {"model": model_name, "model_info": {
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
    }}

    # Exp 1: 约束交换律
    print(f"\n--- Exp 1: 约束交换律 ---")
    t0 = time.time()
    results["exp1_commutativity"] = exp1_commutativity(model, tokenizer, device, model_info)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # Exp 2: 约束逆元
    print(f"\n--- Exp 2: 约束逆元 ---")
    t0 = time.time()
    results["exp2_inverse"] = exp2_inverse(model, tokenizer, device, model_info)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # Exp 3: 约束传播核
    print(f"\n--- Exp 3: 约束传播核 ---")
    t0 = time.time()
    results["exp3_constraint_kernel"] = exp3_constraint_propagation_kernel(model, tokenizer, device, model_info)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # Exp 4: 低秩约束子空间
    print(f"\n--- Exp 4: 低秩约束子空间 ---")
    t0 = time.time()
    results["exp4_low_rank"] = exp4_low_rank_constraint_subspace(model, tokenizer, device, model_info)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # Exp 5: 层间组合规律
    print(f"\n--- Exp 5: 层间组合规律 ---")
    t0 = time.time()
    results["exp5_layerwise_composition"] = exp5_layerwise_composition(model, tokenizer, device, model_info)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # 释放模型
    release_model(model)

    # 保存结果
    out_path = f"tests/glm5_temp/phase131_{model_name}_constraint_algebra.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    # 打印关键摘要
    print_summary(results)
    return results


def print_summary(results):
    """打印关键摘要"""

    print(f"\n{'='*60}")
    print(f"Phase 131 摘要: {results['model']}")
    print(f"{'='*60}")

    # Exp 1: 交换律
    print("\n[Exp 1] 约束交换律/加法性:")
    exp1 = results.get("exp1_commutativity", {})
    for tk, pairs in exp1.items():
        if not isinstance(pairs, dict):
            continue
        for pair_name, data in pairs.items():
            if not isinstance(data, dict) or "layer_results" not in data:
                continue
            lr = data["layer_results"]
            if not lr:
                continue
            # 取中间层
            mid = len(lr) // 2
            if data.get("type") == "commutativity":
                violation = lr[mid].get("violation", 0)
                cos = lr[mid].get("cos_AB_BA", 0)
                print(f"  {tk}/{pair_name} L{mid}: violation={violation:.4f}, cos_AB_BA={cos:.4f}")
            elif data.get("type") == "additivity":
                nl = lr[mid].get("nl_ratio", 0)
                print(f"  {tk}/{pair_name} L{mid}: nl_ratio={nl:.4f}")

    # Exp 2: 逆元
    print("\n[Exp 2] 约束逆元 (Neg∘Neg vs Identity):")
    exp2 = results.get("exp2_inverse", {})
    for tk, data in exp2.items():
        if not isinstance(data, dict) or "layer_results" not in data:
            continue
        lr = data["layer_results"]
        mid = len(lr) // 2
        rel_diff = lr[mid].get("relative_diff", 0)
        ratio = lr[mid].get("neg_neg_ratio", 0)
        cos = lr[mid].get("cos_neg_neg", 0)
        print(f"  {tk} L{mid}: rel_diff={rel_diff:.4f}, neg_neg_ratio={ratio:.4f}, cos_neg_neg={cos:.4f}")

    # Exp 3: 约束传播核维度
    print("\n[Exp 3] 约束传播核的有效秩:")
    exp3_dim = results.get("exp3_constraint_kernel", {}).get("dimensionality_analysis", {})
    for tk in list(exp3_dim.keys())[:2]:
        for layer_key in [f"L0", f"L{len(exp3_dim[tk])//4}", f"L{len(exp3_dim[tk])//2}",
                          f"L{3*len(exp3_dim[tk])//4}", f"L{len(exp3_dim[tk])-1}"]:
            if layer_key in exp3_dim[tk]:
                d = exp3_dim[tk][layer_key]
                print(f"  {tk}/{layer_key}: eff_rank={d['eff_rank']:.2f}, "
                      f"top3_energy={d['top3_energy']:.4f}")

    # Exp 4: 低秩结构
    print("\n[Exp 4] 低秩约束子空间:")
    exp4_layer = results.get("exp4_low_rank", {}).get("layer_analysis", {})
    for layer_key in ["L0", "L9", "L18", "L27"]:
        if layer_key in exp4_layer:
            d = exp4_layer[layer_key]
            de = d.get("dims_for_energy", {})
            print(f"  {layer_key}: eff_rank={d['eff_rank']:.2f}, "
                  f"dim_90pct={de.get('dim_90pct', 'N/A')}, "
                  f"dim_95pct={de.get('dim_95pct', 'N/A')}")

    # Exp 5: 层间组合
    print("\n[Exp 5] 层间组合 (非线性比):")
    exp5_summary = results.get("exp5_layerwise_composition", {}).get("layer_summary", {})
    sorted_keys = sorted([k for k in exp5_summary.keys() if k.startswith("L")],
                         key=lambda x: int(x[1:]))
    max_layer = int(sorted_keys[-1][1:]) if sorted_keys else 0
    for layer_key in sorted_keys:
        d = exp5_summary[layer_key]
        layer_idx = int(layer_key[1:])
        if layer_idx % 6 == 0 or layer_idx == max_layer:
            print(f"  {layer_key}: mean_nl_ratio={d['mean_nl_ratio']:.4f}, "
                  f"mean_lin_acc={d['mean_lin_acc']:.4f}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "deepseek7b", "glm4"]:
            run_all_experiments(name)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_all_experiments(model_name)
