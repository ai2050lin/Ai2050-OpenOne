"""
Phase 133: 严格算子几何 — 确立真实Jacobian秩与测量可靠性
============================================================

对Phase 132的三个关键批评(分析一+分析二):
1. k_perturb=48太小 → 测的是rank(J P_48), 不是rank(J) → 需要满秩标度律
2. 有限差分可能处于非线性区 → 需要ε收敛测试
3. LayerNorm污染Jacobian → 需要模块分解

3个实验:
- Exp 1: 满秩标度律 — k=16,32,64,128,256, 确定Jacobian内禀秩
- Exp 2: ε收敛测试 — ε=1e-1~1e-5, 验证有限差分可靠性
- Exp 3: 模块分解 — 分离J_attn, J_mlp, J_LN的贡献

关键改进:
- k从48增加到256, 消除探针上限
- ε多值比较, 确认线性区
- 子模块hook, 分离各模块对Jacobian秩的贡献

方法论说明:
- 8bit模型(DS7B/GLM4)不支持autograd, 只能用有限差分
- Qwen3(bfloat16)理论支持autograd, 但36层的计算图超出12GB显存
- 因此全部使用有限差分, 但通过Exp 2验证可靠性
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
# 句子设计 — 简化, 只用3个代表性句子
# ============================================================

BASE_SENTENCES = [
    {"s": "dog", "v_base": "bites", "v_past": "bit", "v_pp": "bitten", "o": "man"},
    {"s": "cat", "v_base": "chases", "v_past": "chased", "v_pp": "chased", "o": "mouse"},
    {"s": "teacher", "v_base": "helps", "v_past": "helped", "v_pp": "helped", "o": "student"},
]


def make_sentence(entry, variant="base"):
    """生成语法正确的句子变体"""
    s = entry["s"]
    v = entry["v_base"]
    v_past = entry["v_past"]
    v_pp = entry["v_pp"]
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


def get_hidden_states_with_perturbation(model, tokenizer, device, prompt,
                                         perturb_pos=-1, eps=0.0, perturb_vec=None):
    """获取带扰动的所有层hidden states"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone()

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
    """计算两个子空间的平均cosine相似度"""
    if k1 is not None:
        U1 = U1[:, :k1]
    if k2 is not None:
        U2 = U2[:, :k2]
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    trace_overlap = np.trace(P1 @ P2)
    norm1 = np.sqrt(np.trace(P1 @ P1))
    norm2 = np.sqrt(np.trace(P2 @ P2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(trace_overlap / (norm1 * norm2))


# ============================================================
# Exp 1: 满秩标度律 (Full-Rank Scaling Law)
# ============================================================

def exp1_full_rank_scaling(model, tokenizer, device, model_info):
    """
    测量: eff_rank(k) for k = 16, 32, 64, 128, 256

    关键问题: eff_rank ≈ 48 (k=48时) 是真实秩还是探针上限?

    方法:
    1. 生成k_max=256个随机扰动方向
    2. 计算256个Jacobian列估计
    3. 对前k个列做SVD, 计算eff_rank(k)
    4. 如果eff_rank(k)在k>64后饱和 → 内禀秩已确定
    5. 如果eff_rank(k)随k线性增长 → 真实秩远大于k_max
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_max = 256  # 最大扰动数
    eps = 1e-4   # 后续由Exp 2验证

    # 采样层: 5个代表性位置
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    sample_layer_names = ["L0", f"L{n_layers//4}", f"L{n_layers//2}",
                          f"L{3*n_layers//4}", f"L{n_layers-2}"]

    results = {"scaling_curves": {}, "per_sentence": {}}

    for si, entry in enumerate(BASE_SENTENCES):
        sent = make_sentence(entry, "base")
        print(f"  [{si+1}/{len(BASE_SENTENCES)}] {sent}")

        # 基线hidden states
        hs_base = get_hidden_states_all(model, tokenizer, device, sent)

        # 生成k_max个随机扰动方向
        rng = np.random.RandomState(42 + si)
        V = rng.randn(d_model, k_max).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        # 收集各采样层的扰动响应
        Y_all = {}  # {layer_idx: [d_model, k_max]}
        for li in sample_layers:
            Y_all[li] = np.zeros((d_model, k_max), dtype=np.float32)

        # 逐个扰动
        for j in range(k_max):
            hs_pert = get_hidden_states_with_perturbation(
                model, tokenizer, device, sent,
                perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
            )
            for li in sample_layers:
                Y_all[li][:, j] = (hs_pert[li] - hs_base[li]) / eps

            if (j + 1) % 64 == 0:
                print(f"    perturbation {j+1}/{k_max} done")

        # 各层各k值的SVD
        sent_result = {}
        for li_idx, li in enumerate(sample_layers):
            lk = sample_layer_names[li_idx]
            Y_l = Y_all[li]

            # 完整SVD (k_max x d_model)
            U_full, S_full, Vh_full = np.linalg.svd(Y_l, full_matrices=False)

            # 各k值的eff_rank
            k_values = [16, 32, 48, 64, 96, 128, 192, 256]
            scaling = {}
            for k in k_values:
                S_k = S_full[:min(k, len(S_full))]
                eff_r = compute_effective_rank(S_k, 0.99)
                # 解释方差比
                total_var = np.sum(S_full ** 2)
                k_var = np.sum(S_k ** 2)
                var_ratio = k_var / total_var if total_var > 0 else 0
                scaling[str(k)] = {
                    "eff_rank": eff_r,
                    "var_ratio": float(var_ratio),
                    "top_sv": S_full[:min(8, len(S_full))].tolist(),
                }

            # 完整奇异值谱(前64个)
            sent_result[lk] = {
                "scaling": scaling,
                "full_spectrum_top64": S_full[:64].tolist(),
                "eff_rank_full": compute_effective_rank(S_full, 0.99),
            }

        results["per_sentence"][sent] = sent_result

    # 汇总: 各层各k值的平均eff_rank
    agg = {}
    for li_idx, li in enumerate(sample_layers):
        lk = sample_layer_names[li_idx]
        k_values = [16, 32, 48, 64, 96, 128, 192, 256]
        agg[lk] = {}
        for k in k_values:
            ranks = []
            var_ratios = []
            for sent, sent_data in results["per_sentence"].items():
                if lk in sent_data:
                    ranks.append(sent_data[lk]["scaling"][str(k)]["eff_rank"])
                    var_ratios.append(sent_data[lk]["scaling"][str(k)]["var_ratio"])
            agg[lk][str(k)] = {
                "mean_eff_rank": float(np.mean(ranks)) if ranks else 0,
                "std_eff_rank": float(np.std(ranks)) if ranks else 0,
                "mean_var_ratio": float(np.mean(var_ratios)) if var_ratios else 0,
            }

    results["aggregate_scaling"] = agg
    return results


# ============================================================
# Exp 2: ε收敛测试 (Epsilon Convergence Test)
# ============================================================

def exp2_epsilon_convergence(model, tokenizer, device, model_info):
    """
    验证: 有限差分 Jv ≈ (f(x+εv) - f(x)) / ε 是否收敛到真实JVP

    方法:
    1. 固定1个句子, 1个层, 1组随机扰动方向(k=64)
    2. 对ε = 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 分别计算Jacobian列估计
    3. 比较:
       - 各ε的奇异值谱是否收敛
       - 相邻ε的子空间对齐度(cosine similarity)
       - JVP向量的逐元素相对误差

    如果收敛: 有限差分可靠, Phase 132结果可信
    如果不收敛: 需要重新审视所有基于有限差分的结论
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 64
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    # 测试2个层: 早期(L0)和中间(L_mid)
    test_layers = [0, n_layers // 2]
    test_layer_names = ["L0", f"L{n_layers//2}"]

    results = {}

    for si, entry in enumerate(BASE_SENTENCES[:2]):  # 只用2个句子
        sent = make_sentence(entry, "base")
        print(f"  [{si+1}/2] {sent}")

        # 基线
        hs_base = get_hidden_states_all(model, tokenizer, device, sent)

        # 随机扰动方向
        rng = np.random.RandomState(42 + si)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        # 各ε值的Jacobian列估计
        Y_eps = {}  # {eps_str: {layer_idx: [d_model, k]}}

        for eps in eps_values:
            eps_str = f"eps_{eps:.0e}"
            Y_eps[eps_str] = {}
            for li in test_layers:
                Y_eps[eps_str][li] = np.zeros((d_model, k_perturb), dtype=np.float32)

            for j in range(k_perturb):
                hs_pert = get_hidden_states_with_perturbation(
                    model, tokenizer, device, sent,
                    perturb_pos=-1, eps=eps, perturb_vec=V[:, j]
                )
                for li in test_layers:
                    Y_eps[eps_str][li][:, j] = (hs_pert[li] - hs_base[li]) / eps

        # 分析收敛性
        sent_result = {}

        for li_idx, li in enumerate(test_layers):
            lk = test_layer_names[li_idx]

            # 各ε的奇异值谱
            sv_spectra = {}
            for eps in eps_values:
                eps_str = f"eps_{eps:.0e}"
                Y_l = Y_eps[eps_str][li]
                U, S, Vh = np.linalg.svd(Y_l, full_matrices=False)
                sv_spectra[eps_str] = S[:32].tolist()

            # 相邻ε的子空间对齐
            alignment = {}
            for i in range(len(eps_values) - 1):
                e1 = eps_values[i]
                e2 = eps_values[i + 1]
                s1 = f"eps_{e1:.0e}"
                s2 = f"eps_{e2:.0e}"

                Y1 = Y_eps[s1][li]
                Y2 = Y_eps[s2][li]

                # SVD for subspace comparison
                U1, S1, _ = np.linalg.svd(Y1, full_matrices=False)
                U2, S2, _ = np.linalg.svd(Y2, full_matrices=False)

                # 子空间对齐 (top-16)
                cos_16 = subspace_cosine(U1, U2, k1=16, k2=16)
                # 子空间对齐 (top-32)
                cos_32 = subspace_cosine(U1, U2, k1=32, k2=32)

                # 逐向量相对误差
                rel_errors = []
                for j in range(min(k_perturb, 32)):
                    v1 = Y1[:, j]
                    v2 = Y2[:, j]
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        rel_errors.append(float(np.linalg.norm(v1 - v2) / max(norm1, norm2)))

                alignment[f"{s1}_vs_{s2}"] = {
                    "cos_top16": float(cos_16),
                    "cos_top32": float(cos_32),
                    "mean_rel_error": float(np.mean(rel_errors)) if rel_errors else -1,
                    "eff_rank_1": compute_effective_rank(S1, 0.99),
                    "eff_rank_2": compute_effective_rank(S2, 0.99),
                }

            # 最小ε(1e-5)作为"参考真值" — 与各ε比较
            ref_str = f"eps_{eps_values[-1]:.0e}"
            convergence_to_ref = {}
            for eps in eps_values[:-1]:
                s1 = f"eps_{eps:.0e}"
                Y1 = Y_eps[s1][li]
                Y_ref = Y_eps[ref_str][li]

                U1, S1, _ = np.linalg.svd(Y1, full_matrices=False)
                U_ref, S_ref, _ = np.linalg.svd(Y_ref, full_matrices=False)

                cos_16 = subspace_cosine(U1, U_ref, k1=16, k2=16)

                convergence_to_ref[s1] = {
                    "cos_to_ref_top16": float(cos_16),
                    "rank_ratio": float(compute_effective_rank(S1, 0.99) /
                                        max(compute_effective_rank(S_ref, 0.99), 1)),
                }

            sent_result[lk] = {
                "sv_spectra": sv_spectra,
                "adjacent_alignment": alignment,
                "convergence_to_ref": convergence_to_ref,
            }

        results[sent] = sent_result

    # 汇总
    summary = {}
    for li_idx, li in enumerate(test_layers):
        lk = test_layer_names[li_idx]
        # 各ε对的最小子空间对齐
        min_cos = {}
        for i in range(len(eps_values) - 1):
            e1 = eps_values[i]
            e2 = eps_values[i + 1]
            key = f"eps_{e1:.0e}_vs_eps_{e2:.0e}"
            cos_vals = []
            for sent, sent_data in results.items():
                if lk in sent_data:
                    cos_vals.append(sent_data[lk]["adjacent_alignment"]
                                    [f"eps_{e1:.0e}_vs_eps_{e2:.0e}"]["cos_top16"])
            min_cos[key] = float(np.mean(cos_vals)) if cos_vals else 0
        summary[lk] = min_cos

    results["summary"] = summary
    return results


# ============================================================
# Exp 3: 模块分解 (Module Decomposition)
# ============================================================

def exp3_module_decomposition(model, tokenizer, device, model_info):
    """
    分解: Jacobian的秩贡献来自哪些模块?

    Transformer block:
      h_{l+1} = h_l + Attn(LN1(h_l)) + MLP(LN2(h_l + Attn(LN1(h_l))))

    我们测量扰动在各子模块输出处的传播:
    1. δh_input: 输入到层的扰动
    2. δh_after_ln1: 经过input LayerNorm后的扰动
    3. δh_after_attn: 经过attention+residual后的扰动
    4. δh_after_ln2: 经过post-attention LayerNorm后的扰动
    5. δh_after_mlp: 经过MLP+residual后的扰动(=δh_output)

    方法: 用hook捕获各子模块的中间输出, 对比perturbed vs base

    注意: 这里不是精确的Jacobian分解, 而是测量扰动在各阶段的"形态变化"
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    k_perturb = 64
    eps = 1e-4

    # 只分析2个层: 早期和中间
    test_layers = [0, n_layers // 2]
    test_layer_names = ["L0", f"L{n_layers//2}"]

    results = {}

    for si, entry in enumerate(BASE_SENTENCES[:2]):
        sent = make_sentence(entry, "base")
        print(f"  [{si+1}/2] {sent}")

        # 准备输入
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds_base = embed_layer(input_ids).detach().clone()

        # 随机扰动方向
        rng = np.random.RandomState(42 + si)
        V = rng.randn(d_model, k_perturb).astype(np.float32)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        layers = get_layers(model)

        sent_result = {}

        for li_idx, li in enumerate(test_layers):
            lk = test_layer_names[li_idx]
            layer = layers[li]

            # === Hook设置: 捕获各子模块输出 ===
            captured_base = {}
            captured_pert = {}

            def make_hook(dict_obj, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        dict_obj[key] = output[0].detach().float().cpu()
                    else:
                        dict_obj[key] = output.detach().float().cpu()
                return hook

            # 需要hook的子模块 (如果存在)
            hook_points = {}

            # Input LayerNorm
            for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
                if hasattr(layer, ln_name):
                    hook_points["ln1"] = getattr(layer, ln_name)
                    break

            # Self-attention
            if hasattr(layer, "self_attn"):
                hook_points["attn"] = layer.self_attn

            # Post-attention LayerNorm
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(layer, ln_name):
                    hook_points["ln2"] = getattr(layer, ln_name)
                    break

            # MLP
            if hasattr(layer, "mlp"):
                hook_points["mlp"] = layer.mlp

            # === 基线前向 ===
            hooks_base = []
            for key, module in hook_points.items():
                hooks_base.append(module.register_forward_hook(make_hook(captured_base, key)))

            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_base, attention_mask=attention_mask,
                          output_hidden_states=True)

            for h in hooks_base:
                h.remove()

            # 扰动响应收集
            # Y_stage[stage_name] = [d_model, k_perturb]
            Y_stages = {stage: np.zeros((d_model, k_perturb), dtype=np.float32)
                        for stage in ["ln1", "attn", "ln2", "mlp"]}

            # 获取基线last-token值
            base_vals = {}
            for stage in hook_points:
                if stage in captured_base:
                    base_vals[stage] = captured_base[stage][0, -1, :].numpy()

            for j in range(k_perturb):
                # 扰动embedding
                inputs_embeds_pert = inputs_embeds_base.clone()
                v_tensor = torch.tensor(V[:, j], dtype=inputs_embeds_pert.dtype, device=device)
                inputs_embeds_pert[0, -1, :] += eps * v_tensor

                # 扰动前向
                captured_pert_j = {}
                hooks_pert = []
                for key, module in hook_points.items():
                    hooks_pert.append(module.register_forward_hook(
                        make_hook(captured_pert_j, key)))

                with torch.no_grad():
                    _ = model(inputs_embeds=inputs_embeds_pert, attention_mask=attention_mask,
                              output_hidden_states=True)

                for h in hooks_pert:
                    h.remove()

                # 收集各stage的δh / ε
                for stage in hook_points:
                    if stage in captured_pert_j and stage in base_vals:
                        pert_val = captured_pert_j[stage][0, -1, :].numpy()
                        Y_stages[stage][:, j] = (pert_val - base_vals[stage]) / eps

            # 各stage的SVD
            stage_spectra = {}
            for stage in Y_stages:
                Y_s = Y_stages[stage]
                if np.linalg.norm(Y_s) < 1e-10:
                    stage_spectra[stage] = {"eff_rank": 0, "top_sv": [], "norm": 0}
                    continue
                U, S, Vh = np.linalg.svd(Y_s, full_matrices=False)
                stage_spectra[stage] = {
                    "eff_rank": compute_effective_rank(S, 0.99),
                    "top_sv": S[:32].tolist(),
                    "norm": float(np.linalg.norm(Y_s, 'fro')),
                    "sv_ratio_s1_s32": float(S[0] / max(S[min(31, len(S)-1)], 1e-10)),
                }

            # 子空间对齐: 各stage与最终输出的子空间对齐度
            if "mlp" in Y_stages and np.linalg.norm(Y_stages["mlp"]) > 1e-10:
                U_mlp, S_mlp, _ = np.linalg.svd(Y_stages["mlp"], full_matrices=False)
                alignment_to_output = {}
                for stage in ["ln1", "attn", "ln2"]:
                    if stage in Y_stages and np.linalg.norm(Y_stages[stage]) > 1e-10:
                        U_s, S_s, _ = np.linalg.svd(Y_stages[stage], full_matrices=False)
                        cos = subspace_cosine(U_s, U_mlp, k1=min(16, U_s.shape[1]),
                                              k2=min(16, U_mlp.shape[1]))
                        alignment_to_output[stage] = float(cos)
                stage_spectra["alignment_to_output"] = alignment_to_output

            sent_result[lk] = stage_spectra

        results[sent] = sent_result

    # 汇总
    summary = {}
    for li_idx, li in enumerate(test_layers):
        lk = test_layer_names[li_idx]
        stage_ranks = defaultdict(list)
        for sent, sent_data in results.items():
            if lk in sent_data:
                for stage in ["ln1", "attn", "ln2", "mlp"]:
                    if stage in sent_data[lk]:
                        stage_ranks[stage].append(sent_data[lk][stage]["eff_rank"])
        summary[lk] = {stage: {"mean_rank": float(np.mean(ranks)), "std_rank": float(np.std(ranks))}
                       for stage, ranks in stage_ranks.items() if ranks}

    results["summary"] = summary
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    print("=" * 60)
    print(f"Phase 133: 严格算子几何 — {model_name}")
    print("=" * 60)

    # 加载模型
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

    # === Exp 1: 满秩标度律 ===
    print("\n" + "=" * 40)
    print("Exp 1: 满秩标度律 (Full-Rank Scaling Law)")
    print("=" * 40)
    t1 = time.time()
    r1 = exp1_full_rank_scaling(model, tokenizer, device, model_info)
    all_results["exp1_full_rank_scaling"] = r1
    print(f"  Exp 1 done in {time.time()-t1:.1f}s")
    # 显示关键结果
    for lk in sorted(r1["aggregate_scaling"].keys()):
        scaling = r1["aggregate_scaling"][lk]
        ranks_str = ", ".join([f"k={k}:r={scaling[k]['mean_eff_rank']:.1f}"
                               for k in ["16", "32", "48", "64", "128", "256"]
                               if k in scaling])
        print(f"  {lk}: {ranks_str}")

    # 释放GPU缓存
    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 2: ε收敛测试 ===
    print("\n" + "=" * 40)
    print("Exp 2: ε收敛测试 (Epsilon Convergence)")
    print("=" * 40)
    t2 = time.time()
    r2 = exp2_epsilon_convergence(model, tokenizer, device, model_info)
    all_results["exp2_epsilon_convergence"] = r2
    print(f"  Exp 2 done in {time.time()-t2:.1f}s")
    # 显示关键结果
    for sent, sent_data in r2.items():
        if sent == "summary":
            continue
        for lk, ld in sent_data.items():
            print(f"  {sent[:30]}... {lk}:")
            for pair, data in ld["adjacent_alignment"].items():
                print(f"    {pair}: cos_top16={data['cos_top16']:.4f}, "
                      f"rel_err={data['mean_rel_error']:.4f}")

    # 释放GPU缓存
    gc.collect()
    torch.cuda.empty_cache()

    # === Exp 3: 模块分解 ===
    print("\n" + "=" * 40)
    print("Exp 3: 模块分解 (Module Decomposition)")
    print("=" * 40)
    t3 = time.time()
    r3 = exp3_module_decomposition(model, tokenizer, device, model_info)
    all_results["exp3_module_decomposition"] = r3
    print(f"  Exp 3 done in {time.time()-t3:.1f}s")
    # 显示关键结果
    if "summary" in r3:
        for lk, stage_data in r3["summary"].items():
            print(f"  {lk}:")
            for stage, sd in stage_data.items():
                print(f"    {stage}: eff_rank={sd['mean_rank']:.1f}±{sd['std_rank']:.1f}")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"phase133_{model_name}_rigorous_operator.json")

    # 转换numpy类型
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

    # 释放模型
    release_model(model)
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
