"""
Phase 138: 状态稳定性实验 — 决定"程序vs动力系统"的关键分叉
==========================================================

批评的核心修正:
1. Phase 137的"约束传播"过推断 — negative recovery只证明"深层表示全局共依赖"
   不能推出"存在显式约束图", 还有三种替代解释:
   A) 高阶流形耦合: 深层在弯曲流形上, 单点patch扔出流形
   B) Attention同步性: 深层token形成mutual alignment, 单点破坏对齐
   C) Residual stream共振: 多层累计干涉结构, 单点破坏相位
2. Weighted Jaccard↓ + Binary Jaccard↑ → "连续场调制"而非"符号路由"
3. 真正的数学对象可能是"能量景观"而非"约束图"
4. 需要决定性实验: 扰动后系统是"弛豫回吸引子"(能量系统)还是"不可恢复"(程序)

Phase 138三个实验:
  Exp 1: Deep State Relaxation — 在深层扰动hidden state, 继续forward, 
         看后续层是"修复"扰动(能量系统/吸引子)还是"放大"扰动(程序/发散)
  Exp 2: Global Inconsistency Sensitivity — 测语法一致/不一致句子的
         "全局能量"(logit entropy + cross-position KL + residual norm)
  Exp 3: Multi-layer Joint Patching — 联合多层patching vs 单层,
         看是否有"超线性恢复"(分布式协同编码)

关键区分:
- 如果是"程序": 扰动后路径切换, 不可恢复, 各层独立贡献
- 如果是"动力系统/能量景观": 扰动后弛豫回吸引子, 全局能量响应, 分布式协同
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
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)


# ============================================================
# 句子设计 — 比Phase 137更丰富
# ============================================================

# 语法一致 vs 不一致句子 (Exp 2)
GRAMMAR_CONSISTENT = [
    "The dogs bark loudly in the park",
    "The children play happily outside",
    "The birds fly south for winter",
    "The rivers flow into the ocean",
    "The stars shine bright at night",
    "The leaves fall from the trees",
    "The waves crash on the shore",
    "The clouds drift across the sky",
    "The flowers bloom in spring",
    "The students study hard for exams",
    "The horses run fast on the track",
    "The wolves hunt in the forest",
]

GRAMMAR_INCONSISTENT = [
    "The dogs barks loudly in the park",   # dogs+barks (复数+单数动词)
    "The children plays happily outside",
    "The birds flies south for winter",
    "The rivers flows into the ocean",
    "The stars shines bright at night",
    "The leaves falls from the trees",
    "The waves crashes on the shore",
    "The clouds drifts across the sky",
    "The flowers blooms in spring",
    "The students studies hard for exams",
    "The horses runs fast on the track",
    "The wolves hunts in the forest",
]

# 否定/时态句子 (Exp 1 & Exp 3, 复用Phase 137的设计但增加量)
NEGATION_PAIRS = [
    ("The dog always bites the man", "The dog never bites the man"),
    ("The cat always chases the mouse", "The cat never chases the mouse"),
    ("The sun always rises early", "The sun never rises early"),
    ("The river always flows south", "The river never flows south"),
    ("The wind always blows hard", "The wind never blows hard"),
    ("The bird always sings loud", "The bird never sings loud"),
    ("The fire always burns hot", "The fire never burns hot"),
    ("The child always plays hard", "The child never plays hard"),
    ("The doctor always helps patients", "The doctor never helps patients"),
    ("The teacher always reads books", "The teacher never reads books"),
    ("The soldier always fights hard", "The soldier never fights hard"),
    ("The farmer always grows crops", "The farmer never grows crops"),
    ("The artist always paints well", "The artist never paints well"),
    ("The writer always writes clearly", "The writer never writes clearly"),
    ("The driver always drives safely", "The driver never drives safely"),
    ("The singer always sings softly", "The singer never sings softly"),
    ("The builder always builds strong", "The builder never builds strong"),
    ("The cook always makes food", "The cook never makes food"),
    ("The nurse always cares deeply", "The nurse never cares deeply"),
    ("The police always protect citizens", "The police never protect citizens"),
]

TENSE_PAIRS = [
    ("The dog bites the man", "The dog bit the man"),
    ("The cat chases the mouse", "The cat chased the mouse"),
    ("The sun rises early", "The sun rose early"),
    ("The river flows south", "The river flowed south"),
    ("The wind blows hard", "The wind blew hard"),
    ("The bird sings loud", "The bird sang loud"),
    ("The fire burns hot", "The fire burnt hot"),
    ("The child plays hard", "The child played hard"),
    ("The doctor helps patients", "The doctor helped patients"),
    ("The teacher reads books", "The teacher read books"),
    ("The soldier fights hard", "The soldier fought hard"),
    ("The farmer grows crops", "The farmer grew crops"),
    ("The artist paints well", "The artist painted well"),
    ("The writer writes clearly", "The writer wrote clearly"),
    ("The driver drives safely", "The driver drove safely"),
    ("The singer sings softly", "The singer sang softly"),
    ("The builder builds strong", "The builder built strong"),
    ("The cook makes food", "The cook made food"),
    ("The nurse cares deeply", "The nurse cared deeply"),
    ("The police protect citizens", "The police protected citizens"),
]


# ============================================================
# 工具函数
# ============================================================

def get_device_for_input(model) -> torch.device:
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_top5(tokenizer, logits):
    """获取top-5预测"""
    top5_ids = np.argsort(logits)[-5:][::-1]
    return [(tokenizer.decode([int(i)]).strip(), float(logits[i])) for i in top5_ids]


def compute_logit_entropy(logits: np.ndarray) -> float:
    """计算logits的softmax熵"""
    # 数值稳定的softmax
    logits_shifted = logits - np.max(logits)
    exp_l = np.exp(logits_shifted)
    probs = exp_l / np.sum(exp_l)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ============================================================
# Exp 1: Deep State Relaxation — 核心实验
# ============================================================

def exp1_deep_state_relaxation(model, tokenizer, device, model_info, model_name: str):
    """
    核心实验: 扰动后系统是"弛豫回吸引子"还是"放大扰动"?

    方法:
    1. 运行句子, 在层L_perturb处对hidden state加入随机扰动 ε
    2. 继续forward: L_{perturb+1} → L_{last}
    3. 在每一后续层测量:
       - 扰动传播比: ||Δh_l|| / ||ε||  (扩大=发散, 缩小=弛豫)
       - 方向保持: cos(Δh_l, ε_direction)  (同向=线性传播, 反向=修复)
       - Logit恢复: patched_logits vs original_logits 的距离

    关键预测:
    - 如果是"程序": 扰动会导致路径切换, Δh不会缩小, logit偏移大
    - 如果是"动力系统": 扰动会被后续层"修复", Δh缩小, logit接近原始

    扰动方式:
    - 随机方向扰动 (测全局稳定性)
    - 语义方向扰动 (测特定方向的稳定性)
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)

    # 采样扰动层: 浅/中/深
    perturb_layers = []
    step = max(1, n_layers // 6)
    for i in range(0, n_layers - 2, step):  # 不在最后2层扰动(没有后续层来观察)
        perturb_layers.append(i)
    if n_layers - 3 not in perturb_layers:
        perturb_layers.append(n_layers - 3)

    # 扰动强度: 相对于hidden state范数
    epsilon_scales = [0.01, 0.05, 0.1, 0.2]  # 扰动占hidden state范数的比例

    # 测试句子 (选有代表性的)
    test_sentences = [
        "The dog always bites the man",
        "The cat never chases the mouse",
        "The sun rises early every morning",
        "The children play happily outside",
        "The birds fly south for winter",
    ]

    results = {}
    n_random_dirs = 3  # 每个扰动层测3个随机方向

    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  Sentence {sent_idx+1}/{len(test_sentences)}: '{sentence}'")

        ids = tokenizer.encode(sentence, add_special_tokens=False)
        seq_len = len(ids)
        input_ids = torch.tensor([ids], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

        # 运行原始句子, 获取所有层hidden states
        with torch.no_grad():
            out_orig = model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)

        hs_orig = [hs.detach().clone() for hs in out_orig.hidden_states]
        logits_orig = out_orig.logits[0, -1].float().cpu().numpy()
        entropy_orig = compute_logit_entropy(logits_orig)

        sent_data = {"sentence": sentence, "perturbation_layers": {}}

        for perturb_li in perturb_layers:
            lk_perturb = f"L{perturb_li}"
            print(f"    Perturb at {lk_perturb}...")

            # 原始hidden state at 层perturb_li的输出
            hs_at_perturb = hs_orig[perturb_li + 1]  # [1, seq_len, d_model]
            hs_norm = float(hs_at_perturb.norm())

            layer_data = {"epsilon_scales": {}}

            for eps_scale in epsilon_scales:
                eps_abs = eps_scale * hs_norm  # 扰动绝对大小

                all_dirs_data = []

                for dir_idx in range(n_random_dirs):
                    # 生成随机扰动方向 (在最后一个token位置)
                    torch.manual_seed(42 + dir_idx + perturb_li * 100)
                    random_dir = torch.randn(1, 1, d_model, device=hs_at_perturb.device,
                                             dtype=hs_at_perturb.dtype)
                    random_dir = random_dir / random_dir.norm() * eps_abs

                    # 扰动后的hidden state
                    hs_perturbed = hs_at_perturb.clone()
                    hs_perturbed[0, -1, :] += random_dir[0, 0, :]

                    # 从扰动层开始继续forward, 用hook收集各后续层的hidden state
                    captured_hs = {}
                    capture_layers = []
                    # 采样后续层
                    for li in range(perturb_li + 1, n_layers):
                        step_cap = max(1, (n_layers - perturb_li) // 6)
                        if (li - perturb_li) % step_cap == 0 or li == n_layers - 1:
                            capture_layers.append(li)

                    def make_capture_hook(key):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                captured_hs[key] = output[0].detach().clone()
                            else:
                                captured_hs[key] = output.detach().clone()
                        return hook

                    # 注册hook
                    hooks = []
                    for cli in capture_layers:
                        hooks.append(layers[cli].register_forward_hook(
                            make_capture_hook(f"L{cli}")))

                    # 在扰动层注入扰动的hidden state, 继续forward
                    captured_perturb = {"done": False}

                    def inject_hook(module, input, output):
                        if not captured_perturb["done"]:
                            captured_perturb["done"] = True
                            if isinstance(output, tuple):
                                return (hs_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                            return hs_perturbed.to(output.device).to(output.dtype)
                        return output

                    inject_h = layers[perturb_li].register_forward_hook(inject_hook)

                    try:
                        with torch.no_grad():
                            out_perturbed = model(input_ids=input_ids,
                                                  attention_mask=attention_mask)
                    except Exception as e:
                        print(f"      Forward failed: {e}")
                        inject_h.remove()
                        for h in hooks:
                            h.remove()
                        continue

                    inject_h.remove()
                    for h in hooks:
                        h.remove()

                    # 扰动后的logits
                    logits_perturbed = out_perturbed.logits[0, -1].float().cpu().numpy()
                    logit_shift = float(np.linalg.norm(logits_perturbed - logits_orig))
                    entropy_perturbed = compute_logit_entropy(logits_perturbed)
                    entropy_change = entropy_perturbed - entropy_orig

                    # 扰动传播分析
                    propagation = {}
                    for cli in capture_layers:
                        clk = f"L{cli}"
                        if clk not in captured_hs:
                            continue

                        # Δh at this layer (perturbed vs original)
                        delta_h = captured_hs[clk] - hs_orig[cli + 1]  # [1, seq_len, d_model]
                        delta_h_last = delta_h[0, -1, :].float().cpu().numpy()  # last token
                        delta_h_norm = float(np.linalg.norm(delta_h_last))

                        # 传播比: Δh_norm / ε_norm
                        prop_ratio = delta_h_norm / max(eps_abs, 1e-10)

                        # 方向保持: cos(Δh_last, random_dir)
                        rand_dir_np = random_dir[0, 0, :].float().cpu().numpy()
                        if delta_h_norm > 1e-10 and eps_abs > 1e-10:
                            direction_preserve = float(np.dot(delta_h_last, rand_dir_np) /
                                                       (delta_h_norm * eps_abs))
                        else:
                            direction_preserve = 0.0

                        # 全序列扰动: Δh在所有位置的平均范数
                        delta_h_all_norm = float(delta_h.float().norm()) / seq_len

                        propagation[clk] = {
                            "delta_h_norm_last": delta_h_norm,
                            "prop_ratio": prop_ratio,
                            "direction_preserve": direction_preserve,
                            "delta_h_all_norm_avg": delta_h_all_norm,
                        }

                    dir_data = {
                        "logit_shift": logit_shift,
                        "entropy_change": entropy_change,
                        "propagation": propagation,
                    }
                    all_dirs_data.append(dir_data)

                # 对多个随机方向取平均
                avg_data = {
                    "logit_shift_mean": float(np.mean([d["logit_shift"] for d in all_dirs_data])),
                    "logit_shift_std": float(np.std([d["logit_shift"] for d in all_dirs_data])),
                    "entropy_change_mean": float(np.mean([d["entropy_change"] for d in all_dirs_data])),
                    "entropy_change_std": float(np.std([d["entropy_change"] for d in all_dirs_data])),
                    "propagation_avg": {},
                }

                # 对传播数据取平均
                for cli in capture_layers:
                    clk = f"L{cli}"
                    prop_list = [d["propagation"].get(clk, {}) for d in all_dirs_data
                                 if clk in d["propagation"]]
                    if prop_list:
                        avg_data["propagation_avg"][clk] = {
                            "prop_ratio_mean": float(np.mean([p.get("prop_ratio", 0) for p in prop_list])),
                            "prop_ratio_std": float(np.std([p.get("prop_ratio", 0) for p in prop_list])),
                            "direction_preserve_mean": float(np.mean([p.get("direction_preserve", 0) for p in prop_list])),
                            "direction_preserve_std": float(np.std([p.get("direction_preserve", 0) for p in prop_list])),
                            "delta_h_all_norm_mean": float(np.mean([p.get("delta_h_all_norm_avg", 0) for p in prop_list])),
                        }

                layer_data["epsilon_scales"][str(eps_scale)] = avg_data

                # 打印关键信息
                n_layers_after = len(capture_layers)
                if n_layers_after > 0:
                    last_clk = f"L{capture_layers[-1]}"
                    last_prop = avg_data["propagation_avg"].get(last_clk, {})
                    print(f"      eps={eps_scale}: logit_shift={avg_data['logit_shift_mean']:.4f}, "
                          f"entropy_Δ={avg_data['entropy_change_mean']:.4f}, "
                          f"final_prop_ratio={last_prop.get('prop_ratio_mean', 0):.4f}, "
                          f"final_dir_preserve={last_prop.get('direction_preserve_mean', 0):.4f}")

            sent_data["perturbation_layers"][lk_perturb] = layer_data

            # 清理
            del hs_at_perturb
            torch.cuda.empty_cache()

        results[sentence] = sent_data

        # 清理
        del hs_orig
        torch.cuda.empty_cache()

    # 聚合所有句子的结果
    aggregated = _aggregate_relaxation(results, perturb_layers, epsilon_scales)
    return {"per_sentence": results, "aggregated": aggregated}


def _aggregate_relaxation(per_sentence, perturb_layers, epsilon_scales):
    """聚合所有句子的弛豫结果"""
    agg = {}

    for perturb_li in perturb_layers:
        lk_perturb = f"L{perturb_li}"
        agg[lk_perturb] = {}

        for eps_scale in epsilon_scales:
            eps_key = str(eps_scale)

            logit_shifts = []
            entropy_changes = []
            # 收集每层的传播比
            prop_ratios_by_depth = defaultdict(list)
            dir_preserves_by_depth = defaultdict(list)

            for sent, sent_data in per_sentence.items():
                layer_data = sent_data.get("perturbation_layers", {}).get(lk_perturb, {})
                eps_data = layer_data.get("epsilon_scales", {}).get(eps_key, {})
                if not eps_data:
                    continue

                logit_shifts.append(eps_data.get("logit_shift_mean", 0))
                entropy_changes.append(eps_data.get("entropy_change_mean", 0))

                for clk, prop in eps_data.get("propagation_avg", {}).items():
                    prop_ratios_by_depth[clk].append(prop.get("prop_ratio_mean", 0))
                    dir_preserves_by_depth[clk].append(prop.get("direction_preserve_mean", 0))

            agg[lk_perturb][eps_key] = {
                "logit_shift_mean": float(np.mean(logit_shifts)) if logit_shifts else 0,
                "entropy_change_mean": float(np.mean(entropy_changes)) if entropy_changes else 0,
                "prop_ratio_by_depth": {clk: float(np.mean(vals))
                                        for clk, vals in prop_ratios_by_depth.items()},
                "direction_preserve_by_depth": {clk: float(np.mean(vals))
                                                for clk, vals in dir_preserves_by_depth.items()},
            }

    return agg


# ============================================================
# Exp 2: Global Inconsistency Sensitivity
# ============================================================

def exp2_inconsistency_sensitivity(model, tokenizer, device, model_info, model_name: str):
    """
    测语法一致/不一致句子的"全局能量"差异

    方法:
    对每对句子(一致 vs 不一致), 测:
    1. 每层的logit entropy (不确定性)
    2. 每层的cross-position KL (位置间不一致性)
    3. 残差流范数的层间变化率 (稳定性)
    4. Logit lens: 每层对"正确动词"和"错误动词"的概率差

    关键预测:
    - 如果是"能量系统": 不一致句子在某些层会有"高能量"(高熵, 高KL)
    - 如果是"程序": 不一致句子只是走不同分支, 不一定有全局能量差异
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    W_U = get_W_U(model, model_name)

    # 采样层
    sample_step = max(1, n_layers // 12)
    sample_indices = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))

    results = {"consistent": [], "inconsistent": [], "comparisons": []}

    for pair_idx in range(len(GRAMMAR_CONSISTENT)):
        sent_c = GRAMMAR_CONSISTENT[pair_idx]
        sent_i = GRAMMAR_INCONSISTENT[pair_idx]
        print(f"\n  Pair {pair_idx+1}/{len(GRAMMAR_CONSISTENT)}: "
              f"'{sent_c}' vs '{sent_i}'")

        pair_result = {}

        for label, sentence in [("consistent", sent_c), ("inconsistent", sent_i)]:
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            seq_len = len(ids)
            input_ids = torch.tensor([ids], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

            hs_all = [hs.detach().float().cpu().numpy() for hs in out.hidden_states]
            logits_final = out.logits[0, -1].float().cpu().numpy()

            layer_analysis = {}

            for li in sample_indices:
                lk = f"L{li}"
                h = hs_all[li + 1]  # [1, seq_len, d_model]
                h_sq = h[0]  # [seq_len, d_model]

                # 1. Logit lens: 对每层应用lm_head
                logits_at_layer = W_U @ h_sq[-1]  # [vocab_size] - 最后一个token
                entropy = compute_logit_entropy(logits_at_layer)

                # 2. Cross-position consistency: 位置间的cosine相似度
                # 归一化后的hidden state
                h_norms = np.linalg.norm(h_sq, axis=1, keepdims=True)
                h_norms = np.maximum(h_norms, 1e-10)
                h_normalized = h_sq / h_norms

                # 所有位置对的平均cosine
                cos_matrix = h_normalized @ h_normalized.T  # [seq_len, seq_len]
                # 取上三角(排除对角线)
                mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
                avg_cross_pos_cos = float(np.mean(cos_matrix[mask])) if mask.any() else 0

                # 3. 残差流范数
                residual_norm = float(np.linalg.norm(h_sq))

                # 4. 层间变化率 (稳定性)
                if li > 0:
                    h_prev = hs_all[li]  # [1, seq_len, d_model]
                    delta_h = np.linalg.norm(h_sq - h_prev[0])
                    stability = delta_h / max(residual_norm, 1e-10)
                else:
                    stability = 0.0

                layer_analysis[lk] = {
                    "logit_entropy": entropy,
                    "cross_pos_cosine": avg_cross_pos_cos,
                    "residual_norm": residual_norm,
                    "stability": stability,
                }

            # 最终logits分析
            final_entropy = compute_logit_entropy(logits_final)

            result = {
                "sentence": sentence,
                "final_entropy": final_entropy,
                "layer_analysis": layer_analysis,
            }

            if label == "consistent":
                results["consistent"].append(result)
            else:
                results["inconsistent"].append(result)

            del hs_all
            torch.cuda.empty_cache()

        # 比较: consistent vs inconsistent
        comp = _compare_consistency(
            results["consistent"][-1]["layer_analysis"],
            results["inconsistent"][-1]["layer_analysis"],
            sample_indices
        )
        comp["sentence_consistent"] = sent_c
        comp["sentence_inconsistent"] = sent_i
        results["comparisons"].append(comp)

        # 打印
        for lk in [f"L{sample_indices[0]}", f"L{sample_indices[len(sample_indices)//2]}", f"L{sample_indices[-1]}"]:
            c_data = results["consistent"][-1]["layer_analysis"].get(lk, {})
            i_data = results["inconsistent"][-1]["layer_analysis"].get(lk, {})
            print(f"    {lk}: entropy(C)={c_data.get('logit_entropy', 0):.3f} vs "
                  f"entropy(I)={i_data.get('logit_entropy', 0):.3f}, "
                  f"cos(C)={c_data.get('cross_pos_cosine', 0):.3f} vs "
                  f"cos(I)={i_data.get('cross_pos_cosine', 0):.3f}")

    return results


def _compare_consistency(consistent_layers, inconsistent_layers, sample_indices):
    """比较一致和不一致句子的各层分析"""
    comp = {}
    for li in sample_indices:
        lk = f"L{li}"
        c = consistent_layers.get(lk, {})
        i = inconsistent_layers.get(lk, {})

        comp[lk] = {
            "entropy_diff": i.get("logit_entropy", 0) - c.get("logit_entropy", 0),
            "cosine_diff": i.get("cross_pos_cosine", 0) - c.get("cross_pos_cosine", 0),
            "stability_diff": i.get("stability", 0) - c.get("stability", 0),
        }
    return comp


# ============================================================
# Exp 3: Multi-layer Joint Patching
# ============================================================

def exp3_multilayer_patching(model, tokenizer, device, model_info, model_name: str):
    """
    联合多层patching vs 单层patching

    方法:
    1. 运行 base 和 modified 句子
    2. 对单层做diff-position patching (Phase 137方法)
    3. 对2层联合做diff-position patching
    4. 对3层联合做diff-position patching
    5. 比较: 联合patching的recovery是否 > 单层recovery之和?

    关键预测:
    - 如果信息是"分布式协同编码": 联合patching出现超线性恢复
      (2层联合recovery > 单层A recovery + 单层B recovery)
    - 如果信息是"局部独立编码": 联合patching = 单层之和(线性)
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)

    # 采样层组合
    # 单层
    sample_step = max(1, n_layers // 8)
    single_layers = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))

    # 2层组合: 相邻层
    two_layer_combos = []
    for i in range(0, len(single_layers) - 1):
        two_layer_combos.append((single_layers[i], single_layers[i + 1]))

    # 3层组合: 浅+中+深
    mid_idx = len(single_layers) // 2
    three_layer_combos = [
        (single_layers[0], single_layers[mid_idx], single_layers[-1]),
    ]

    # 使用否定对(修改最清晰)
    test_pairs = NEGATION_PAIRS[:12]  # 12对

    results = {"single": {}, "two_layer": {}, "three_layer": {}}

    for pair_idx, (sent_base, sent_mod) in enumerate(test_pairs):
        print(f"\n  Pair {pair_idx+1}/{len(test_pairs)}: '{sent_base}' → '{sent_mod}'")

        ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
        ids_mod = tokenizer.encode(sent_mod, add_special_tokens=False)

        if len(ids_base) != len(ids_mod):
            print(f"    SKIP: token数不同")
            continue

        seq_len = len(ids_base)
        diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_mod[i]]
        if not diff_pos:
            continue

        input_ids_base = torch.tensor([ids_base], device=input_device)
        input_ids_mod = torch.tensor([ids_mod], device=input_device)
        attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

        # 运行两个句子
        with torch.no_grad():
            out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                             output_hidden_states=True)
            out_mod = model(input_ids=input_ids_mod, attention_mask=attention_mask,
                            output_hidden_states=True)

        hs_base_all = [hs.detach().clone() for hs in out_base.hidden_states]
        hs_mod_all = [hs.detach().clone() for hs in out_mod.hidden_states]

        logits_base = out_base.logits[0, -1].float().cpu().numpy()
        logits_mod = out_mod.logits[0, -1].float().cpu().numpy()
        logit_diff = logits_mod - logits_base
        logit_diff_norm = float(np.linalg.norm(logit_diff))

        # --- 单层patching ---
        single_recoveries = {}

        for li in single_layers:
            lk = f"L{li}"
            patched_hs = hs_base_all[li + 1].clone()
            for pos in diff_pos:
                patched_hs[0, pos, :] = hs_mod_all[li + 1][0, pos, :]

            patched_logits = _patch_layer_single(
                model, tokenizer, input_device, layers,
                input_ids_base, attention_mask,
                patched_hs, li
            )

            if patched_logits is not None:
                delta = patched_logits - logits_base
                delta_norm = float(np.linalg.norm(delta))
                if logit_diff_norm > 1e-10 and delta_norm > 1e-10:
                    cosine_recovery = float(np.dot(delta, logit_diff) /
                                            (delta_norm * logit_diff_norm))
                else:
                    cosine_recovery = 0.0
                single_recoveries[lk] = cosine_recovery
            else:
                single_recoveries[lk] = 0.0

        # --- 2层联合patching ---
        two_layer_recoveries = {}

        for combo in two_layer_combos:
            combo_key = f"L{combo[0]}+L{combo[1]}"
            # 需要同时patch两个层
            # 方法: 先patch浅层, forward到第二层, 再patch第二层
            # 简化方法: 用两次hook同时替换
            patched_logits = _patch_multilayer(
                model, tokenizer, input_device, layers,
                input_ids_base, attention_mask,
                hs_base_all, hs_mod_all,
                diff_pos, combo
            )

            if patched_logits is not None:
                delta = patched_logits - logits_base
                delta_norm = float(np.linalg.norm(delta))
                if logit_diff_norm > 1e-10 and delta_norm > 1e-10:
                    cosine_recovery = float(np.dot(delta, logit_diff) /
                                            (delta_norm * logit_diff_norm))
                else:
                    cosine_recovery = 0.0
                two_layer_recoveries[combo_key] = cosine_recovery
            else:
                two_layer_recoveries[combo_key] = 0.0

            # 超线性指标: 2层联合 - (单层A + 单层B)
            lk_a, lk_b = f"L{combo[0]}", f"L{combo[1]}"
            sum_single = single_recoveries.get(lk_a, 0) + single_recoveries.get(lk_b, 0)
            superlinear = two_layer_recoveries[combo_key] - sum_single
            two_layer_recoveries[f"{combo_key}_superlinear"] = superlinear

        # --- 3层联合patching ---
        for combo in three_layer_combos:
            combo_key = f"L{combo[0]}+L{combo[1]}+L{combo[2]}"
            patched_logits = _patch_multilayer(
                model, tokenizer, input_device, layers,
                input_ids_base, attention_mask,
                hs_base_all, hs_mod_all,
                diff_pos, combo
            )

            if patched_logits is not None:
                delta = patched_logits - logits_base
                delta_norm = float(np.linalg.norm(delta))
                if logit_diff_norm > 1e-10 and delta_norm > 1e-10:
                    cosine_recovery = float(np.dot(delta, logit_diff) /
                                            (delta_norm * logit_diff_norm))
                else:
                    cosine_recovery = 0.0

                sum_single = sum(single_recoveries.get(f"L{c}", 0) for c in combo)
                superlinear = cosine_recovery - sum_single
                results["three_layer"][combo_key] = {
                    "recovery": cosine_recovery,
                    "sum_single": sum_single,
                    "superlinear": superlinear,
                }

        # 存储单层结果
        for lk, rec in single_recoveries.items():
            if lk not in results["single"]:
                results["single"][lk] = []
            results["single"][lk].append(rec)

        for key, val in two_layer_recoveries.items():
            if key not in results["two_layer"]:
                results["two_layer"][key] = []
            results["two_layer"][key].append(val)

        # 打印
        print(f"    Single: " + ", ".join(
            [f"{lk}={rec:.3f}" for lk, rec in sorted(single_recoveries.items())[:5]]))
        print(f"    2-Layer superlinear: " + ", ".join(
            [f"{k}={v:.3f}" for k, v in two_layer_recoveries.items()
             if "superlinear" in k][:3]))

        del hs_base_all, hs_mod_all
        torch.cuda.empty_cache()

    # 聚合
    agg = _aggregate_multilayer(results)
    return {"per_pair": results, "aggregated": agg}


def _patch_layer_single(model, tokenizer, input_device, layers,
                        input_ids, attention_mask, replace_tensor, target_layer):
    """单层patching (Phase 137方法)"""
    captured = {"done": False}

    def replace_hook(module, input, output):
        if not captured["done"]:
            captured["done"] = True
            if isinstance(output, tuple):
                new_hidden = replace_tensor.to(output[0].device).to(output[0].dtype)
                return (new_hidden,) + output[1:]
            return replace_tensor.to(output.device).to(output.dtype)
        return output

    hook = layers[target_layer].register_forward_hook(replace_hook)

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        print(f"    Patch failed at L{target_layer}: {e}")
        logits = None
    finally:
        hook.remove()

    return logits


def _patch_multilayer(model, tokenizer, input_device, layers,
                      input_ids, attention_mask,
                      hs_base_all, hs_mod_all,
                      diff_pos, target_layers):
    """
    多层联合patching: 在多个层同时替换diff-position的hidden state

    关键: 需要按顺序hook多个层
    """
    captured = {li: False for li in target_layers}

    # 预计算每层的替换tensor
    replace_tensors = {}
    for li in target_layers:
        patched_hs = hs_base_all[li + 1].clone()
        for pos in diff_pos:
            patched_hs[0, pos, :] = hs_mod_all[li + 1][0, pos, :]
        replace_tensors[li] = patched_hs

    def make_hook(li):
        def hook(module, input, output):
            if not captured[li]:
                captured[li] = True
                replace_t = replace_tensors[li]
                if isinstance(output, tuple):
                    new_hidden = replace_t.to(output[0].device).to(output[0].dtype)
                    return (new_hidden,) + output[1:]
                return replace_t.to(output.device).to(output.dtype)
            return output
        return hook

    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in target_layers]

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        print(f"    Multi-patch failed at {target_layers}: {e}")
        logits = None
    finally:
        for h in hooks:
            h.remove()

    return logits


def _aggregate_multilayer(results):
    """聚合多层patching结果"""
    agg = {"single": {}, "two_layer": {}, "three_layer": {}}

    for lk, vals in results["single"].items():
        if vals:
            agg["single"][lk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    # 2层: 分离recovery和superlinear
    for key, vals in results["two_layer"].items():
        if vals:
            agg["two_layer"][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    for key, data in results["three_layer"].items():
        if isinstance(data, dict) and "recovery" in data:
            agg["three_layer"][key] = data

    return agg


# ============================================================
# 简化输出
# ============================================================

def simplify_results(results, model_name):
    """简化结果以便JSON存储"""
    import copy

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return obj

    return to_serializable(copy.deepcopy(results))


# ============================================================
# 打印结果摘要
# ============================================================

def print_summary(results, model_name):
    """打印结果摘要"""
    print(f"\n{'='*70}")
    print(f"Phase 138 结果摘要: {model_name}")
    print(f"{'='*70}")

    # Exp 1: 弛豫分析
    exp1 = results.get("exp1_relaxation", {})
    agg1 = exp1.get("aggregated", {})
    print(f"\n--- Exp 1: Deep State Relaxation ---")
    for lk_perturb in sorted(agg1.keys()):
        eps_data = agg1[lk_perturb].get("0.05", {})  # 用eps=5%的结果
        if not eps_data:
            continue
        print(f"  Perturb at {lk_perturb}:")
        print(f"    logit_shift = {eps_data.get('logit_shift_mean', 0):.4f}")
        print(f"    entropy_change = {eps_data.get('entropy_change_mean', 0):.4f}")

        # 传播比随深度变化
        prop_by_depth = eps_data.get("prop_ratio_by_depth", {})
        dir_by_depth = eps_data.get("direction_preserve_by_depth", {})
        for clk in sorted(prop_by_depth.keys()):
            print(f"      {clk}: prop_ratio={prop_by_depth[clk]:.4f}, "
                  f"dir_preserve={dir_by_depth.get(clk, 0):.4f}")

    # Exp 2: 不一致性
    exp2 = results.get("exp2_inconsistency", {})
    comparisons = exp2.get("comparisons", [])
    print(f"\n--- Exp 2: Inconsistency Sensitivity ---")
    if comparisons:
        # 平均各层的entropy_diff
        agg_comp = defaultdict(list)
        for comp in comparisons:
            for lk, data in comp.items():
                if isinstance(data, dict) and "entropy_diff" in data:
                    agg_comp[lk].append(data["entropy_diff"])
        for lk in sorted(agg_comp.keys()):
            vals = agg_comp[lk]
            print(f"  {lk}: entropy_diff(Incongruent - Congruent) = "
                  f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Exp 3: 多层patching
    exp3 = results.get("exp3_multilayer", {})
    agg3 = exp3.get("aggregated", {})
    print(f"\n--- Exp 3: Multi-layer Joint Patching ---")
    single = agg3.get("single", {})
    two_layer = agg3.get("two_layer", {})

    # 单层
    print("  Single layer:")
    for lk in sorted(single.keys()):
        d = single[lk]
        print(f"    {lk}: recovery = {d['mean']:.4f} ± {d['std']:.4f}")

    # 2层联合 - 超线性
    print("  2-Layer superlinear (joint - sum_of_singles):")
    for key in sorted(two_layer.keys()):
        if "superlinear" in key:
            d = two_layer[key]
            print(f"    {key}: {d['mean']:.4f} ± {d['std']:.4f}")


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model_name = model_name.lower()

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Phase 138: 状态稳定性实验 — {model_name}")
    print(f"{'='*70}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, {model_info.n_layers} layers, "
          f"d_model={model_info.d_model}")

    all_results = {"model_info": {
        "name": model_name,
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
    }}

    # === Exp 1: Deep State Relaxation ===
    print(f"\n{'='*70}")
    print("Exp 1: Deep State Relaxation")
    print(f"{'='*70}")
    t0 = time.time()
    exp1_results = exp1_deep_state_relaxation(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 1 done in {t1-t0:.1f}s")
    all_results["exp1_relaxation"] = exp1_results

    # === Exp 2: Inconsistency Sensitivity ===
    print(f"\n{'='*70}")
    print("Exp 2: Inconsistency Sensitivity")
    print(f"{'='*70}")
    t0 = time.time()
    exp2_results = exp2_inconsistency_sensitivity(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 2 done in {t1-t0:.1f}s")
    all_results["exp2_inconsistency"] = exp2_results

    # === Exp 3: Multi-layer Joint Patching ===
    print(f"\n{'='*70}")
    print("Exp 3: Multi-layer Joint Patching")
    print(f"{'='*70}")
    t0 = time.time()
    exp3_results = exp3_multilayer_patching(model, tokenizer, device, model_info, model_name)
    t1 = time.time()
    print(f"Exp 3 done in {t1-t0:.1f}s")
    all_results["exp3_multilayer"] = exp3_results

    # 打印摘要
    print_summary(all_results, model_name)

    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase138_{model_name}_state_stability.json")

    simplified = simplify_results(all_results, model_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
