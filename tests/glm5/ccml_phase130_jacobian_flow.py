"""
Phase 130: Jacobian传播流与约束传播分析
=========================================

Phase 129核心发现:
  1. 动词位置在L9(1/4深度)执行角色绑定, 因果效应0.42-0.56
  2. 因果效应从低层到高层递增(但可能是层深偏置)
  3. 传播流差异与语义差异成正比

Phase 129的核心不足(用户指出):
  1. 仍在"层"上思考 → 应该研究"层间算子" F_l
  2. activation patching有层深偏置 → 应该研究"扰动传播" δh_l → δh_{l+1} → ...
  3. 没有真正进入"组合性" → 应该研究"约束传播"

本阶段核心转变: 从"hidden states"转向"operators (Jacobian flow)"

5个实验:
- Exp 1: Jacobian谱分析 — 用有限差分法近似J_l(x), 分析特征值/奇异值谱
- Exp 2: 扰动传播追踪 — 在某层注入δ, 追踪它如何沿层放大/衰减/分叉
- Exp 3: 约束传播 — 否定/时态/被动如何改变Jacobian
- Exp 4: 组合传播 — 多个约束如何叠加改变传播流
- Exp 5: 轨迹吸引子验证 — 不同输入的传播流是否汇聚到"流形"
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 核心工具函数
# ============================================================

def get_hidden_states(model, tokenizer, device, prompt, max_length=64):
    """获取所有层的hidden states, 返回list of numpy arrays"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    hs = [h[0].detach().float().cpu().numpy() for h in out.hidden_states]  # list of [seq, d_model]
    return hs, input_ids, attention_mask


def compute_jacobian_finite_diff(model, tokenizer, device, prompt, layer_idx, 
                                  epsilon=1e-3, n_directions=50, max_length=64):
    """
    用有限差分法近似Jacobian: J_l(x) = ∂h_{l+1}/∂h_l
    
    方法:
    1. 正常forward, 得到 h_l 和 h_{l+1}
    2. 对 h_l 的last token位置, 沿随机方向v注入扰动 ε*v
    3. 重新forward从layer_idx开始
    4. 观察 h_{l+1} 的变化: δh_{l+1} ≈ J_l @ (ε*v)
    5. 用n_directions个方向, 重建Jacobian的近似
    
    返回: 近似Jacobian的统计量(奇异值, 特征值谱, 秩等)
    """
    # 获取所有层的hidden states
    hs, input_ids, attention_mask = get_hidden_states(model, tokenizer, device, prompt, max_length)
    
    h_l = hs[layer_idx]      # [seq, d_model]
    h_lp1 = hs[layer_idx + 1]  # [seq, d_model]
    
    seq_len = h_l.shape[0]
    d_model = h_l.shape[1]
    
    # 在last token位置注入扰动
    h_l_last = h_l[-1]       # [d_model]
    h_lp1_last = h_lp1[-1]   # [d_model]
    
    # 收集Jacobian-vector乘积
    delta_responses = []
    directions = []
    
    layers = get_layers(model)
    
    for _ in range(n_directions):
        v = np.random.randn(d_model).astype(np.float32)
        v = v / np.linalg.norm(v)
        directions.append(v)
        
        # 在layer_idx的输出上注入扰动
        h_l_perturbed = h_l.copy()
        h_l_perturbed[-1] += epsilon * v
        
        # 用hook在layer_idx处注入扰动后的hidden state
        captured_next = {}
        
        def make_inject_hook(patched_tensor, layer_i):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    patched = output[0].clone()
                    seq = min(patched.shape[1], patched_tensor.shape[0])
                    patched[0, -1, :] = torch.tensor(
                        patched_tensor[-1], dtype=patched.dtype, device=patched.device
                    )
                    return (patched,) + output[1:]
                return output
            return hook
        
        # 注入到layer_idx, 捕获layer_idx+1
        def make_capture_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_next[key] = output[0].detach().float().cpu().numpy()
                return output
            return hook
        
        h_inject = layers[layer_idx].register_forward_hook(
            make_inject_hook(h_l_perturbed, layer_idx)
        )
        h_capture = layers[layer_idx + 1].register_forward_hook(
            make_capture_hook("next")
        ) if layer_idx + 1 < len(layers) else None
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                pass
        
        h_inject.remove()
        if h_capture:
            h_capture.remove()
        
        if "next" in captured_next:
            h_lp1_perturbed_last = captured_next["next"][-1]
            delta_response = h_lp1_perturbed_last - h_lp1_last
        else:
            # fallback: 用原始forward的差分
            delta_response = np.zeros(d_model, dtype=np.float32)
        
        delta_responses.append(delta_response / epsilon)  # ≈ J @ v
    
    # 重建近似Jacobian矩阵
    # J ≈ [δr_0, δr_1, ..., δr_{n-1}] @ [v_0, v_1, ..., v_{n-1}]^+ (伪逆)
    D = np.array(delta_responses).T  # [d_model, n_directions]
    V = np.array(directions).T       # [d_model, n_directions]
    
    # SVD of V^T: V^T = U_v S_v Vt_v, then V^+ = Vt_v^T S_v^{-1} U_v^T
    # J_approx = D @ V^+ = D @ pinv(V^T)
    J_approx = D @ np.linalg.pinv(V.T)  # [d_model, d_model] (近似, 但n_directions << d_model)
    
    # 由于n_directions << d_model, 完整Jacobian无法精确重建
    # 但可以分析Jacobian的约束:
    # J_approx的有效秩和主要奇异值
    
    # 更直接: 分析Jacobian对n_directions个方向的响应
    # 奇异值分解 D = U_D S_D V_D^T
    try:
        U_D, S_D, Vt_D = np.linalg.svd(D, full_matrices=False)
    except Exception:
        S_D = np.zeros(min(d_model, n_directions))
    
    return {
        "layer_idx": layer_idx,
        "n_directions": n_directions,
        "singular_values": S_D.tolist(),
        "top5_sv": S_D[:5].tolist(),
        "effective_rank": float(np.sum(S_D > 0.01 * S_D[0]) if len(S_D) > 0 and S_D[0] > 0 else 0),
        "condition_number": float(S_D[0] / max(S_D[-1], 1e-10)) if len(S_D) > 0 else 0,
        "spectral_norm": float(S_D[0]) if len(S_D) > 0 else 0,
        "frobenius_norm": float(np.sqrt(np.sum(S_D**2))),
        "sv_entropy": float(-np.sum((S_D/np.sum(S_D)) * np.log(S_D/np.sum(S_D) + 1e-20))) if np.sum(S_D) > 0 else 0,
    }


def compute_perturbation_propagation(model, tokenizer, device, prompt, 
                                      inject_layer, n_propagate_layers=5,
                                      epsilon=1e-2, max_length=64):
    """
    扰动传播追踪:
    在inject_layer注入δ, 追踪它如何沿层传播(放大/衰减/分叉)
    
    返回: 每层的扰动范数和方向保持度
    """
    hs_base, input_ids, attention_mask = get_hidden_states(
        model, tokenizer, device, prompt, max_length
    )
    
    n_layers = len(hs_base) - 1  # hidden_states有n_layers+1个(L0是embedding)
    d_model = hs_base[0].shape[1]
    
    # 随机扰动方向
    np.random.seed(42)
    delta_v = np.random.randn(d_model).astype(np.float32)
    delta_v = delta_v / np.linalg.norm(delta_v)
    
    layers = get_layers(model)
    
    # 在inject_layer注入扰动, 追踪后续各层
    propagation = {}
    delta_inject = epsilon * delta_v  # 注入的扰动
    
    # 用hook注入
    captured_layers = {}
    
    def make_inject_hook(base_hs, inject_delta, li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                patched = output[0].clone()
                patched[0, -1, :] += torch.tensor(inject_delta, dtype=patched.dtype, device=patched.device)
                return (patched,) + output[1:]
            return output
        return hook
    
    # 逐层注入并追踪
    for target_layer_offset in range(n_propagate_layers):
        target_layer = inject_layer + target_layer_offset
        if target_layer >= n_layers:
            break
        
        # 注入inject_layer, 捕获target_layer的输出
        captured = {}
        
        h_inject = layers[inject_layer].register_forward_hook(
            make_inject_hook(hs_base[inject_layer + 1], delta_inject, inject_layer)
        )
        
        def make_capture_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0][0, -1, :].detach().float().cpu().numpy()
                return output
            return hook
        
        h_capture = layers[target_layer].register_forward_hook(
            make_capture_hook(f"L{target_layer}")
        )
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                pass
        
        h_inject.remove()
        h_capture.remove()
        
        if f"L{target_layer}" in captured:
            h_perturbed = captured[f"L{target_layer}"]
            h_base = hs_base[target_layer + 1][-1]
            delta_out = h_perturbed - h_base
            
            # 扰动范数
            delta_norm = float(np.linalg.norm(delta_out))
            base_norm = float(np.linalg.norm(h_base))
            
            # 放大/衰减比
            amplification = delta_norm / max(np.linalg.norm(delta_inject), 1e-10)
            
            # 方向保持度 (与注入方向的余弦相似度)
            if delta_norm > 1e-10:
                cos_with_inject = float(np.dot(delta_out, delta_inject) / 
                                       (delta_norm * np.linalg.norm(delta_inject)))
            else:
                cos_with_inject = 0.0
            
            # 方向变化 (与上一层的传播方向变化)
            propagation[f"L{target_layer}"] = {
                "delta_norm": round(delta_norm, 6),
                "amplification": round(amplification, 4),
                "cos_with_inject": round(cos_with_inject, 4),
                "relative_magnitude": round(delta_norm / max(base_norm, 1e-10), 6),
            }
    
    return {
        "inject_layer": inject_layer,
        "inject_epsilon": epsilon,
        "n_propagated": len(propagation),
        "propagation": propagation,
    }


# ============================================================
# Exp 1: Jacobian谱分析
# ============================================================
def exp1_jacobian_spectrum(model, tokenizer, device, model_info):
    """
    分析Jacobian J_l(x) = ∂h_{l+1}/∂h_l 的谱结构
    
    核心问题:
    - Jacobian的奇异值谱是什么形状? (幂律? 指数衰减?)
    - 有效秩如何随层变化?
    - 不同输入的Jacobian谱是否不同? (条件性)
    - 谱熵如何变化? (低熵=强约束, 高熵=自由传播)
    """
    print("\n" + "="*60)
    print("Exp 1: Jacobian谱分析")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 采样层
    sample_layers = list(range(0, n_layers - 1, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers + [n_layers // 4, n_layers // 2, 3 * n_layers // 4]))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 2)
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers - 1]))
    
    # 测试输入: 不同语义类别
    TEST_PROMPTS = [
        ("cat", "The cat sat on the mat"),
        ("dog", "The dog ran in the park"),
        ("apple", "The apple is red and sweet"),
        ("city", "The city is large and busy"),
        ("freedom", "Freedom is important for all"),
        ("hammer", "The hammer is a useful tool"),
    ]
    
    results = {}
    
    for word, prompt in TEST_PROMPTS:
        word_results = {}
        
        for li in sample_layers:
            try:
                jac_stats = compute_jacobian_finite_diff(
                    model, tokenizer, device, prompt, 
                    layer_idx=li, epsilon=1e-3, n_directions=40
                )
                word_results[f"L{li}"] = jac_stats
                
            except Exception as e:
                word_results[f"L{li}"] = {"error": str(e)}
        
        results[word] = word_results
        print(f"  {word}: {len(word_results)} layers analyzed")
    
    # 汇总: 不同层的Jacobian谱统计
    layer_stats = defaultdict(list)
    for word, wdata in results.items():
        for layer_key, jdata in wdata.items():
            if "error" not in jdata:
                layer_stats[layer_key].append(jdata)
    
    summary = {}
    for layer_key, stats_list in layer_stats.items():
        if not stats_list:
            continue
        summary[layer_key] = {
            "mean_eff_rank": round(np.mean([s["effective_rank"] for s in stats_list]), 1),
            "mean_spectral_norm": round(np.mean([s["spectral_norm"] for s in stats_list]), 4),
            "mean_sv_entropy": round(np.mean([s["sv_entropy"] for s in stats_list]), 4),
            "mean_condition": round(np.mean([s["condition_number"] for s in stats_list]), 2),
            "mean_top5_sv": [round(float(v), 4) for v in np.mean([s["top5_sv"] for s in stats_list], axis=0)],
        }
    
    # 不同语义类别的Jacobian差异
    cat_diffs = {}
    words = list(results.keys())
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            diffs = []
            for layer_key in sample_layers:
                lk = f"L{layer_key}"
                if lk in results[w1] and lk in results[w2]:
                    d1 = results[w1][lk]
                    d2 = results[w2][lk]
                    if "error" not in d1 and "error" not in d2:
                        sv1 = np.array(d1["top5_sv"])
                        sv2 = np.array(d2["top5_sv"])
                        cos = float(np.dot(sv1, sv2) / max(np.linalg.norm(sv1) * np.linalg.norm(sv2), 1e-10))
                        diffs.append(cos)
            if diffs:
                cat_diffs[f"{w1}_vs_{w2}"] = round(np.mean(diffs), 4)
    
    print(f"\n  Jacobian spectral norm by layer:")
    for layer_key in sorted(summary.keys()):
        s = summary[layer_key]
        print(f"    {layer_key}: eff_rank={s['mean_eff_rank']:.1f}, "
              f"spec_norm={s['mean_spectral_norm']:.4f}, "
              f"entropy={s['mean_sv_entropy']:.4f}")
    
    print(f"\n  Cross-word Jacobian similarity (top5 SV cosine):")
    for pair, sim in sorted(cat_diffs.items(), key=lambda x: x[1]):
        print(f"    {pair}: {sim:.4f}")
    
    return {
        "n_prompts": len(TEST_PROMPTS),
        "sample_layers": sample_layers,
        "layer_summary": summary,
        "cross_word_similarity": cat_diffs,
    }


# ============================================================
# Exp 2: 扰动传播追踪
# ============================================================
def exp2_perturbation_propagation(model, tokenizer, device, model_info):
    """
    在某层注入扰动, 追踪它如何沿层传播
    
    核心问题:
    - 扰动被放大还是衰减?
    - 方向是否保持?
    - 不同注入层的传播模式是否不同?
    - 不同输入的传播是否不同? (条件性)
    """
    print("\n" + "="*60)
    print("Exp 2: 扰动传播追踪")
    print("="*60)
    
    n_layers = model_info.n_layers
    
    # 采样注入层
    inject_layers = [0, n_layers // 6, n_layers // 4, n_layers // 3, n_layers // 2, 
                     2 * n_layers // 3, 3 * n_layers // 4]
    inject_layers = sorted(set([l for l in inject_layers if l < n_layers - 5]))
    
    # 测试输入
    TEST_PROMPTS = [
        ("dog_bites", "The dog bites the man"),
        ("man_bites", "The man bites the dog"),
        ("red_apple", "The red apple is sweet"),
        ("green_apple", "The green apple is sour"),
        ("cat", "The cat is sleeping"),
        ("city", "The city is growing"),
    ]
    
    results = {}
    
    for word, prompt in TEST_PROMPTS:
        word_results = {}
        
        for inject_l in inject_layers:
            prop = compute_perturbation_propagation(
                model, tokenizer, device, prompt,
                inject_layer=inject_l, n_propagate_layers=min(8, n_layers - inject_l),
                epsilon=1e-2
            )
            word_results[f"inject_L{inject_l}"] = prop
        
        results[word] = word_results
        print(f"  {word}: {len(word_results)} injection points")
    
    # 汇总: 每个注入层的平均传播特征
    inject_summary = defaultdict(lambda: {"amps": [], "cosines": [], "rel_mags": []})
    
    for word, wdata in results.items():
        for inject_key, prop_data in wdata.items():
            for layer_key, ldata in prop_data["propagation"].items():
                inject_summary[inject_key]["amps"].append(ldata["amplification"])
                inject_summary[inject_key]["cosines"].append(ldata["cos_with_inject"])
                inject_summary[inject_key]["rel_mags"].append(ldata["relative_magnitude"])
    
    summary = {}
    for inject_key, vals in inject_summary.items():
        summary[inject_key] = {
            "mean_amp": round(np.mean(vals["amps"]), 4),
            "mean_cos": round(np.mean(vals["cosines"]), 4),
            "mean_rel_mag": round(np.mean(vals["rel_mags"]), 6),
        }
    
    # 主宾互换的传播差异
    swap_diff = {}
    for inject_key in inject_layers:
        ik = f"inject_L{inject_key}"
        # dog_bites vs man_bites 的传播差异
        if ik in results.get("dog_bites", {}) and ik in results.get("man_bites", {}):
            prop_dog = results["dog_bites"][ik]["propagation"]
            prop_man = results["man_bites"][ik]["propagation"]
            
            for lk in prop_dog:
                if lk in prop_man:
                    amp_diff = abs(prop_dog[lk]["amplification"] - prop_man[lk]["amplification"])
                    cos_diff = abs(prop_dog[lk]["cos_with_inject"] - prop_man[lk]["cos_with_inject"])
                    swap_diff[f"L{inject_key}->{lk}"] = {
                        "amp_diff": round(amp_diff, 4),
                        "cos_diff": round(cos_diff, 4),
                    }
    
    # 属性变化的传播差异
    attr_diff = {}
    for inject_key in inject_layers:
        ik = f"inject_L{inject_key}"
        if ik in results.get("red_apple", {}) and ik in results.get("green_apple", {}):
            prop_red = results["red_apple"][ik]["propagation"]
            prop_green = results["green_apple"][ik]["propagation"]
            
            for lk in prop_red:
                if lk in prop_green:
                    amp_diff = abs(prop_red[lk]["amplification"] - prop_green[lk]["amplification"])
                    cos_diff = abs(prop_red[lk]["cos_with_inject"] - prop_green[lk]["cos_with_inject"])
                    attr_diff[f"L{inject_key}->{lk}"] = {
                        "amp_diff": round(amp_diff, 4),
                        "cos_diff": round(cos_diff, 4),
                    }
    
    print(f"\n  Propagation summary by injection point:")
    for inject_key, s in sorted(summary.items()):
        print(f"    {inject_key}: amp={s['mean_amp']:.4f}, cos={s['mean_cos']:.4f}")
    
    print(f"\n  Subject-object swap propagation diff:")
    for path, d in sorted(swap_diff.items()):
        print(f"    {path}: amp_diff={d['amp_diff']:.4f}, cos_diff={d['cos_diff']:.4f}")
    
    print(f"\n  Attribute change propagation diff:")
    for path, d in sorted(attr_diff.items()):
        print(f"    {path}: amp_diff={d['amp_diff']:.4f}, cos_diff={d['cos_diff']:.4f}")
    
    return {
        "inject_layers": inject_layers,
        "inject_summary": summary,
        "swap_propagation_diff": swap_diff,
        "attr_propagation_diff": attr_diff,
        "n_prompts": len(TEST_PROMPTS),
    }


# ============================================================
# Exp 3: 约束传播 — 否定/时态/被动如何改变Jacobian
# ============================================================
def exp3_constraint_propagation(model, tokenizer, device, model_info):
    """
    约束传播: 语法约束(否定/时态/被动)如何改变Jacobian
    
    核心思想:
    "dog bites man" vs "dog does not bite man"
    这两个输入的Jacobian应该不同 — 否定约束改变了传播结构
    
    方法:
    1. 计算base句的Jacobian谱 (每个采样层)
    2. 计算constrained句的Jacobian谱
    3. 比较Jacobian谱的差异
    """
    print("\n" + "="*60)
    print("Exp 3: 约束传播 — 否定/时态/被动如何改变Jacobian")
    print("="*60)
    
    n_layers = model_info.n_layers
    sample_layers = [n_layers // 6, n_layers // 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3]
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers - 1]))
    
    # 约束对
    CONSTRAINT_PAIRS = [
        ("base_vs_negation", "The dog bites the man", "The dog does not bite the man"),
        ("base_vs_past", "The dog bites the man", "The dog bit the man"),
        ("base_vs_future", "The dog bites the man", "The dog will bite the man"),
        ("base_vs_passive", "The dog bites the man", "The man is bitten by the dog"),
        ("base_vs_question", "The dog bites the man", "Does the dog bite the man"),
    ]
    
    results = {}
    
    for pair_name, base_prompt, constrained_prompt in CONSTRAINT_PAIRS:
        pair_results = {}
        
        # Base Jacobian
        base_jacs = {}
        for li in sample_layers:
            try:
                jac = compute_jacobian_finite_diff(
                    model, tokenizer, device, base_prompt,
                    layer_idx=li, epsilon=1e-3, n_directions=30
                )
                base_jacs[f"L{li}"] = jac
            except Exception as e:
                base_jacs[f"L{li}"] = {"error": str(e)}
        
        # Constrained Jacobian
        constr_jacs = {}
        for li in sample_layers:
            try:
                jac = compute_jacobian_finite_diff(
                    model, tokenizer, device, constrained_prompt,
                    layer_idx=li, epsilon=1e-3, n_directions=30
                )
                constr_jacs[f"L{li}"] = jac
            except Exception as e:
                constr_jacs[f"L{li}"] = {"error": str(e)}
        
        # 比较Jacobian谱
        jac_diffs = {}
        for li in sample_layers:
            lk = f"L{li}"
            if lk in base_jacs and lk in constr_jacs:
                b = base_jacs[lk]
                c = constr_jacs[lk]
                if "error" not in b and "error" not in c:
                    # SV谱差异
                    sv1 = np.array(b["top5_sv"])
                    sv2 = np.array(c["top5_sv"])
                    cos_sv = float(np.dot(sv1, sv2) / max(np.linalg.norm(sv1) * np.linalg.norm(sv2), 1e-10))
                    
                    jac_diffs[lk] = {
                        "sv_cosine": round(cos_sv, 4),
                        "eff_rank_diff": round(abs(b["effective_rank"] - c["effective_rank"]), 1),
                        "spec_norm_ratio": round(b["spectral_norm"] / max(c["spectral_norm"], 1e-10), 4),
                        "entropy_diff": round(abs(b["sv_entropy"] - c["sv_entropy"]), 4),
                        "base_eff_rank": b["effective_rank"],
                        "constr_eff_rank": c["effective_rank"],
                        "base_spec_norm": b["spectral_norm"],
                        "constr_spec_norm": c["spectral_norm"],
                    }
        
        results[pair_name] = {
            "jacobian_diffs": jac_diffs,
        }
        
        print(f"  {pair_name}: {len(jac_diffs)} layers compared")
        for lk, d in jac_diffs.items():
            print(f"    {lk}: SV_cos={d['sv_cosine']:.4f}, "
                  f"rank_diff={d['eff_rank_diff']:.1f}, "
                  f"entropy_diff={d['entropy_diff']:.4f}")
    
    # 汇总: 哪种约束对Jacobian影响最大
    constraint_impact = {}
    for pair_name, data in results.items():
        diffs = data["jacobian_diffs"]
        if diffs:
            mean_sv_cos = np.mean([d["sv_cosine"] for d in diffs.values()])
            mean_entropy_diff = np.mean([d["entropy_diff"] for d in diffs.values()])
            constraint_impact[pair_name] = {
                "mean_sv_cosine": round(mean_sv_cos, 4),
                "mean_entropy_diff": round(mean_entropy_diff, 4),
                "impact_score": round(1 - mean_sv_cos + mean_entropy_diff, 4),  # 越大=影响越大
            }
    
    print(f"\n  Constraint impact ranking:")
    for name, impact in sorted(constraint_impact.items(), key=lambda x: x[1]["impact_score"], reverse=True):
        print(f"    {name}: impact={impact['impact_score']:.4f} "
              f"(SV_cos={impact['mean_sv_cosine']:.4f}, entropy_diff={impact['mean_entropy_diff']:.4f})")
    
    return {
        "constraint_pairs": [p[0] for p in CONSTRAINT_PAIRS],
        "sample_layers": sample_layers,
        "results": results,
        "constraint_impact": constraint_impact,
    }


# ============================================================
# Exp 4: 组合传播 — 多个约束如何叠加
# ============================================================
def exp4_compositional_flow(model, tokenizer, device, model_info):
    """
    组合传播: 多个约束如何叠加改变传播流
    
    核心思想:
    "dog bites man" (base)
    "dog does not bite man" (negation)
    "dog bit the man" (past)
    "dog did not bite the man" (negation + past)
    
    否定+时态的效果 ≈ 否定效果 + 时态效果? (线性叠加)
    还是非线性交互?
    """
    print("\n" + "="*60)
    print("Exp 4: 组合传播 — 多约束叠加")
    print("="*60)
    
    n_layers = model_info.n_layers
    
    # 基础句子和约束组合
    COMPOSITIONS = [
        ("base", "The dog bites the man"),
        ("negation", "The dog does not bite the man"),
        ("past", "The dog bit the man"),
        ("neg+past", "The dog did not bite the man"),
        ("passive", "The man is bitten by the dog"),
        ("neg+passive", "The man is not bitten by the dog"),
        ("past+passive", "The man was bitten by the dog"),
        ("neg+past+passive", "The man was not bitten by the dog"),
    ]
    
    # 获取所有句子的hidden states
    all_hs = {}
    for name, prompt in COMPOSITIONS:
        hs, _, _ = get_hidden_states(model, tokenizer, device, prompt)
        all_hs[name] = hs
    
    # 计算层间Δh
    def get_delta_hs(hs_list):
        """获取层间增量"""
        deltas = []
        for i in range(len(hs_list) - 1):
            delta = hs_list[i + 1][-1] - hs_list[i][-1]  # last token的增量
            deltas.append(delta)
        return deltas
    
    # 获取各句的Δh
    all_deltas = {}
    for name, hs in all_hs.items():
        all_deltas[name] = get_delta_hs(hs)
    
    # 采样层对
    sample_pairs = [(0, 1), (n_layers//4, n_layers//4+1), 
                    (n_layers//2, n_layers//2+1), (3*n_layers//4, 3*n_layers//4+1)]
    sample_pairs = [(l1, l2) for l1, l2 in sample_pairs if l2 < n_layers]
    
    # 分析: 约束的独立效果 vs 组合效果
    composition_analysis = {}
    
    for l1, l2 in sample_pairs:
        li = l1  # Δh index
        
        base_delta = all_deltas["base"][li]
        neg_delta = all_deltas["negation"][li]
        past_delta = all_deltas["past"][li]
        neg_past_delta = all_deltas["neg+past"][li]
        
        # 否定的独立效果
        neg_effect = neg_delta - base_delta
        # 时态的独立效果
        past_effect = past_delta - base_delta
        # 组合效果
        neg_past_effect = neg_past_delta - base_delta
        # 线性预测
        linear_prediction = neg_effect + past_effect
        
        # 非线性交互项
        nonlinearity = neg_past_effect - linear_prediction
        
        # 量化
        neg_norm = np.linalg.norm(neg_effect)
        past_norm = np.linalg.norm(past_effect)
        comb_norm = np.linalg.norm(neg_past_effect)
        lin_norm = np.linalg.norm(linear_prediction)
        nl_norm = np.linalg.norm(nonlinearity)
        
        # 线性预测的准确度
        if comb_norm > 1e-10:
            linear_accuracy = float(np.dot(neg_past_effect, linear_prediction) / 
                                   (comb_norm * max(lin_norm, 1e-10)))
        else:
            linear_accuracy = 0
        
        # 非线性交互比
        nl_ratio = nl_norm / max(comb_norm, 1e-10)
        
        composition_analysis[f"L{l1}->L{l2}"] = {
            "neg_norm": round(float(neg_norm), 6),
            "past_norm": round(float(past_norm), 6),
            "combined_norm": round(float(comb_norm), 6),
            "linear_pred_norm": round(float(lin_norm), 6),
            "nonlinearity_norm": round(float(nl_norm), 6),
            "linear_accuracy": round(linear_accuracy, 4),
            "nonlinearity_ratio": round(float(nl_ratio), 4),
        }
    
    # 被动+否定的组合
    passive_composition = {}
    for l1, l2 in sample_pairs:
        li = l1
        
        base_delta = all_deltas["base"][li]
        passive_delta = all_deltas["passive"][li]
        neg_delta = all_deltas["negation"][li]
        neg_passive_delta = all_deltas["neg+passive"][li]
        
        passive_effect = passive_delta - base_delta
        neg_effect_2 = neg_delta - base_delta
        neg_passive_effect = neg_passive_delta - base_delta
        linear_prediction = passive_effect + neg_effect_2
        nonlinearity = neg_passive_effect - linear_prediction
        
        comb_norm = np.linalg.norm(neg_passive_effect)
        lin_norm = np.linalg.norm(linear_prediction)
        nl_norm = np.linalg.norm(nonlinearity)
        
        if comb_norm > 1e-10:
            linear_accuracy = float(np.dot(neg_passive_effect, linear_prediction) / 
                                   (comb_norm * max(lin_norm, 1e-10)))
        else:
            linear_accuracy = 0
        
        nl_ratio = nl_norm / max(comb_norm, 1e-10)
        
        passive_composition[f"L{l1}->L{l2}"] = {
            "combined_norm": round(float(comb_norm), 6),
            "nonlinearity_norm": round(float(nl_norm), 6),
            "linear_accuracy": round(linear_accuracy, 4),
            "nonlinearity_ratio": round(float(nl_ratio), 4),
        }
    
    # 三重组合
    triple_composition = {}
    for l1, l2 in sample_pairs:
        li = l1
        
        base_delta = all_deltas["base"][li]
        passive_delta = all_deltas["passive"][li]
        past_delta_2 = all_deltas["past"][li]
        triple_delta = all_deltas["neg+past+passive"][li]
        
        passive_effect = passive_delta - base_delta
        past_effect_2 = past_delta_2 - base_delta
        triple_effect = triple_delta - base_delta
        
        # 二阶预测: passive + past
        linear_2 = passive_effect + past_effect_2
        nonlinearity_2 = triple_effect - linear_2
        
        comb_norm = np.linalg.norm(triple_effect)
        nl_norm = np.linalg.norm(nonlinearity_2)
        nl_ratio = nl_norm / max(comb_norm, 1e-10)
        
        triple_composition[f"L{l1}->L{l2}"] = {
            "nonlinearity_ratio": round(float(nl_ratio), 4),
            "combined_norm": round(float(comb_norm), 6),
        }
    
    print(f"\n  Negation+Past composition:")
    for lk, d in composition_analysis.items():
        print(f"    {lk}: linear_acc={d['linear_accuracy']:.4f}, "
              f"nonlinearity_ratio={d['nonlinearity_ratio']:.4f}")
    
    print(f"\n  Negation+Passive composition:")
    for lk, d in passive_composition.items():
        print(f"    {lk}: linear_acc={d['linear_accuracy']:.4f}, "
              f"nonlinearity_ratio={d['nonlinearity_ratio']:.4f}")
    
    print(f"\n  Triple (Neg+Past+Passive) composition:")
    for lk, d in triple_composition.items():
        print(f"    {lk}: nonlinearity_ratio={d['nonlinearity_ratio']:.4f}")
    
    return {
        "n_compositions": len(COMPOSITIONS),
        "sample_pairs": sample_pairs,
        "neg_past_composition": composition_analysis,
        "neg_passive_composition": passive_composition,
        "triple_composition": triple_composition,
    }


# ============================================================
# Exp 5: 轨迹吸引子验证 — 不同输入的传播流是否汇聚
# ============================================================
def exp5_trajectory_attractor(model, tokenizer, device, model_info):
    """
    轨迹吸引子验证: 不同输入的传播流是否汇聚到"流形"
    
    核心思想:
    如果"动物"是一个传播流形, 那么:
    - cat, dog, wolf, horse 的传播轨迹应该逐渐汇聚
    - apple, banana 的传播轨迹应该走另一条路
    
    方法:
    1. 多个同类词的传播轨迹
    2. 计算轨迹间的层间距离变化
    3. 如果距离递减 → 存在吸引子(汇聚)
    4. 如果距离不变 → 独立传播
    5. 如果距离递增 → 分叉
    """
    print("\n" + "="*60)
    print("Exp 5: 轨迹吸引子验证")
    print("="*60)
    
    n_layers = model_info.n_layers
    
    # 每个类别8个词(加大数据量)
    WORDS_BY_CATEGORY = {
        "animal": ["cat", "dog", "wolf", "horse", "lion", "bear", "fox", "tiger"],
        "fruit": ["apple", "banana", "orange", "mango", "grape", "peach", "plum", "cherry"],
        "place": ["city", "mountain", "river", "forest", "valley", "island", "desert", "lake"],
        "tool": ["hammer", "knife", "drill", "wrench", "saw", "chisel", "pliers", "shovel"],
    }
    
    # 获取所有词的hidden states
    all_hs = {}
    for cat, words in WORDS_BY_CATEGORY.items():
        for word in words:
            prompt = f"The {word} is"
            hs, _, _ = get_hidden_states(model, tokenizer, device, prompt)
            all_hs[f"{cat}_{word}"] = hs
    
    # 计算层间距离变化
    # 对每对词: 计算它们在各层的cosine距离
    def cos_dist(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 1.0
        return 1.0 - float(np.dot(v1, v2) / (n1 * n2))
    
    # 类内距离变化 (同类别词之间的距离如何随层变化)
    intra_cat_dist = defaultdict(lambda: defaultdict(list))
    
    for cat, words in WORDS_BY_CATEGORY.items():
        word_keys = [f"{cat}_{w}" for w in words]
        for i in range(len(word_keys)):
            for j in range(i + 1, len(word_keys)):
                k1, k2 = word_keys[i], word_keys[j]
                if k1 in all_hs and k2 in all_hs:
                    for layer_idx in range(len(all_hs[k1])):
                        d = cos_dist(all_hs[k1][layer_idx][-1], all_hs[k2][layer_idx][-1])
                        intra_cat_dist[cat][layer_idx].append(d)
    
    # 跨类距离变化
    cross_cat_dist = defaultdict(list)
    cat_list = list(WORDS_BY_CATEGORY.keys())
    for ci in range(len(cat_list)):
        for cj in range(ci + 1, len(cat_list)):
            c1, c2 = cat_list[ci], cat_list[cj]
            words1 = WORDS_BY_CATEGORY[c1]
            words2 = WORDS_BY_CATEGORY[c2]
            
            for w1 in words1[:4]:
                for w2 in words2[:4]:
                    k1 = f"{c1}_{w1}"
                    k2 = f"{c2}_{w2}"
                    if k1 in all_hs and k2 in all_hs:
                        for layer_idx in range(len(all_hs[k1])):
                            d = cos_dist(all_hs[k1][layer_idx][-1], all_hs[k2][layer_idx][-1])
                            cross_cat_dist[layer_idx].append(d)
    
    # 汇总
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 10)))
    sample_layers = sorted(set(sample_layers + [0, n_layers]))
    
    intra_summary = {}
    for cat, layer_dists in intra_cat_dist.items():
        cat_data = {}
        for li in sample_layers:
            if li in layer_dists and layer_dists[li]:
                cat_data[f"L{li}"] = round(np.mean(layer_dists[li]), 4)
        intra_summary[cat] = cat_data
    
    cross_summary = {}
    for li in sample_layers:
        if li in cross_cat_dist and cross_cat_dist[li]:
            cross_summary[f"L{li}"] = round(np.mean(cross_cat_dist[li]), 4)
    
    # 分析: 距离是收敛还是发散?
    convergence = {}
    for cat, cat_data in intra_summary.items():
        layers_sorted = sorted([int(k[1:]) for k in cat_data.keys()])
        if len(layers_sorted) >= 2:
            first_d = cat_data[f"L{layers_sorted[0]}"]
            last_d = cat_data[f"L{layers_sorted[-1]}"]
            convergence[cat] = {
                "first_layer_dist": first_d,
                "last_layer_dist": last_d,
                "convergence_ratio": round(last_d / max(first_d, 1e-10), 4),
                "converging": last_d < first_d,
            }
    
    cross_convergence = {}
    if cross_summary:
        layers_sorted = sorted([int(k[1:]) for k in cross_summary.keys()])
        if len(layers_sorted) >= 2:
            first_d = cross_summary[f"L{layers_sorted[0]}"]
            last_d = cross_summary[f"L{layers_sorted[-1]}"]
            cross_convergence = {
                "first_layer_dist": first_d,
                "last_layer_dist": last_d,
                "convergence_ratio": round(last_d / max(first_d, 1e-10), 4),
                "converging": last_d > first_d,  # 跨类应该发散
            }
    
    # 类内vs跨类: 区分度如何随层变化
    discrimination = {}
    for li in sample_layers:
        lk = f"L{li}"
        intra_means = []
        for cat, cat_data in intra_summary.items():
            if lk in cat_data:
                intra_means.append(cat_data[lk])
        
        if intra_means and lk in cross_summary:
            discrimination[lk] = round(cross_summary[lk] - np.mean(intra_means), 4)
    
    print(f"\n  Intra-category distance convergence:")
    for cat, conv in convergence.items():
        print(f"    {cat}: first={conv['first_layer_dist']:.4f} → last={conv['last_layer_dist']:.4f} "
              f"(ratio={conv['convergence_ratio']:.4f}, converging={conv['converging']})")
    
    if cross_convergence:
        print(f"\n  Cross-category distance: first={cross_convergence['first_layer_dist']:.4f} → "
              f"last={cross_convergence['last_layer_dist']:.4f} "
              f"(ratio={cross_convergence['convergence_ratio']:.4f})")
    
    print(f"\n  Discrimination (cross - intra) by layer:")
    for lk, d in sorted(discrimination.items()):
        print(f"    {lk}: {d:.4f}")
    
    return {
        "n_categories": len(WORDS_BY_CATEGORY),
        "n_words_per_cat": 8,
        "intra_summary": intra_summary,
        "cross_summary": cross_summary,
        "convergence": convergence,
        "cross_convergence": cross_convergence,
        "discrimination": discrimination,
    }


# ============================================================
# Main
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        return
    
    print("="*60)
    print(f"Phase 130: Jacobian传播流与约束传播分析")
    print(f"Model: {model_name}")
    print("="*60)
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  n_layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    all_results = {}
    t0 = time.time()
    
    # Exp 1: Jacobian谱分析
    try:
        all_results["exp1_jacobian_spectrum"] = exp1_jacobian_spectrum(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"  Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_jacobian_spectrum"] = {"error": str(e)}
    
    # Exp 2: 扰动传播追踪
    try:
        all_results["exp2_perturbation_propagation"] = exp2_perturbation_propagation(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"  Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_perturbation_propagation"] = {"error": str(e)}
    
    # Exp 3: 约束传播
    try:
        all_results["exp3_constraint_propagation"] = exp3_constraint_propagation(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"  Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_constraint_propagation"] = {"error": str(e)}
    
    # Exp 4: 组合传播
    try:
        all_results["exp4_compositional_flow"] = exp4_compositional_flow(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"  Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_compositional_flow"] = {"error": str(e)}
    
    # Exp 5: 轨迹吸引子
    try:
        all_results["exp5_trajectory_attractor"] = exp5_trajectory_attractor(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"  Exp5 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_trajectory_attractor"] = {"error": str(e)}
    
    elapsed = time.time() - t0
    all_results["meta"] = {
        "model": model_name,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "elapsed_seconds": round(elapsed, 1),
    }
    
    # 保存结果
    model_short = {"qwen3": "qwen3", "deepseek7b": "deepseek7b", "glm4": "glm4"}[model_name]
    out_path = f"tests/glm5_temp/phase130_{model_short}_jacobian_flow.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
