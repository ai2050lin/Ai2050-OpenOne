"""
Subspace Topology Phase 1: 系统破解编码机制 — 残差流子空间完整拓扑
================================================================

核心目标: 搞清"特征到底怎么存在残差流里"

实验清单:
  Exp1: 全层残差流SVD — 每层的奇异值谱和本征维度
  Exp2: SVCCA层间对齐度 — 相邻层子空间如何演变
  Exp3: 特征维度手术 — 逐步切除主成分直到功能崩溃，记录临界维度
  Exp4: 子空间重叠度 — 不同特征子空间之间共享多少维度

原则:
  - 先做什么，不问为什么
  - 每个结论必须有因果/对照验证
  - 关键结论大样本量验证

Run:
  python tests/glm5/subspace_topology_phase1.py --model qwen3 --exp 1
  python tests/glm5/subspace_topology_phase1.py --model qwen3 --exp 2
  python tests/glm5/subspace_topology_phase1.py --model qwen3 --exp 3
  python tests/glm5/subspace_topology_phase1.py --model qwen3 --exp 4
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
from pathlib import Path

from model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    collect_layer_outputs, compute_cos
)

# ============================================================
# 输出目录
# ============================================================
OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 多样化测试数据 — 覆盖多种语言现象
# ============================================================
DIVERSE_PROMPTS = [
    # 简单陈述
    "The apple is red.",
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    # 否定
    "The apple is NOT red.",
    "Paris is NOT the capital of France.",
    # 疑问
    "Is the apple red?",
    "What is the capital of France?",
    # 双重否定
    "It is not true that the apple is not red.",
    # 三段论
    "All cats are animals. Whiskers is a cat. Therefore Whiskers is",
    # 翻译
    "Translate to French: The apple is red.",
    "Translate to Chinese: The apple is red.",
    # 推理
    "Step by step: If John is taller than Mary, and Mary is taller than Tom, who is the shortest?",
    # 代码
    "Python code to sort a list:",
    # 角色扮演
    "In the style of Shakespeare, explain quantum physics.",
    # 中英混杂
    "苹果apple是一种fruit。",
    # 数学
    "What is 41 + 2?",
    "What is 49 + 2?",
    # 悖论
    "This sentence is false.",
    # 抽象概念
    "Justice is a fundamental concept in philosophy.",
    "Freedom means different things to different people.",
    # 属性查询
    "The physical color of an apple is",
    "The physical color of justice is",
    # 代词
    "John gave Mary a book. She read",
    "John gave Mary a book. He read",
    # 标点差异
    "The sun is shining.",
    "The sun is shining!",
    "Is the sun shining?",
    # 词性歧义
    "I like to read a good book.",
    "I need to book a flight.",
    # 稀有字
    "饕餮是一种传说中的神兽。",
    # 数字序列
    "1, 2, 3, 4,",
    "5, 6, 7,",
]

# 按特征分类的prompt组 (用于Exp3/4)
FEATURE_GROUPS = {
    "negation": [
        "The apple is red.",
        "The apple is NOT red.",
        "The sky is blue.",
        "The sky is NOT blue.",
    ],
    "question": [
        "The apple is red.",
        "Is the apple red?",
        "Paris is the capital of France.",
        "What is the capital of France?",
    ],
    "translation": [
        "The apple is red.",
        "Translate to French: The apple is red.",
        "Translate to Chinese: The apple is red.",
    ],
    "reasoning": [
        "All cats are animals. Whiskers is a cat. Therefore Whiskers is",
        "If A=B and B=C, then A=",
        "Step by step: 5 + 3 =",
    ],
    "code": [
        "The apple is red.",
        "Python code to sort a list:",
        "Write a function in Python:",
    ],
}


def compute_participation_ratio(singular_values):
    """计算参与率 (Intrinsic Dimensionality) = (sum λ)^2 / sum(λ^2)"""
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 1e-10]  # 过滤零值
    if len(s) == 0:
        return 0.0
    return float((np.sum(s))**2 / np.sum(s**2))


def compute_effective_rank(singular_values, threshold=0.01):
    """计算有效秩: 贡献>threshold*max的奇异值数量"""
    s = np.array(singular_values, dtype=np.float64)
    if len(s) == 0:
        return 0
    max_s = np.max(s)
    if max_s < 1e-10:
        return 0
    return int(np.sum(s > threshold * max_s))


# ============================================================
# Exp1: 全层残差流SVD — 奇异值谱和本征维度
# ============================================================
def exp1_full_layer_svd(model, tokenizer, device, model_info):
    """
    对每层残差流收集多样本激活，做SVD分析
    
    输出: 每层的奇异值谱、本征维度、有效秩、方差解释比
    """
    print("\n" + "="*70)
    print("Exp1: 全层残差流SVD分析")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 收集所有prompt的各层激活
    print(f"收集 {len(DIVERSE_PROMPTS)} 个prompt的各层残差流...")
    
    # 按层存储所有token的隐藏状态 [n_total_tokens, d_model]
    layer_activations = {f"L{i}": [] for i in range(n_layers)}
    
    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 收集各层输出
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception as e:
                print(f"  Prompt {pi} forward failed: {e}")
        
        for h in hooks:
            h.remove()
        
        # 存储最后一个token的隐藏状态（预测位置）
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                # 取最后一个token
                last_tok_h = captured[key][0, -1, :].numpy()  # [d_model]
                layer_activations[key].append(last_tok_h)
        
        if (pi + 1) % 10 == 0:
            print(f"  已处理 {pi+1}/{len(DIVERSE_PROMPTS)} prompts")
        
        del captured
        gc.collect()
    
    # 对每层做SVD
    print(f"\n对每层做SVD分析...")
    results = {}
    
    for li in range(n_layers):
        key = f"L{li}"
        acts = np.array(layer_activations[key], dtype=np.float32)  # [n_prompts, d_model]
        n_samples = acts.shape[0]
        
        # 中心化
        acts_centered = acts - acts.mean(axis=0, keepdims=True)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        except Exception as e:
            print(f"  L{li} SVD failed: {e}")
            continue
        
        # 参与率 (本征维度)
        pr = compute_participation_ratio(S)
        
        # 有效秩
        erank = compute_effective_rank(S, threshold=0.01)
        erank_05 = compute_effective_rank(S, threshold=0.005)
        
        # 方差解释比
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var if total_var > 0 else np.zeros_like(S)
        n_90 = int(np.searchsorted(cumvar, 0.90)) + 1  # 解释90%方差需要的维度
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        
        # 奇异值衰减率
        if len(S) > 1 and S[0] > 1e-10:
            decay_rate = float(S[1] / S[0])  # 第二/第一奇异值比
        else:
            decay_rate = 0.0
        
        results[key] = {
            "layer": li,
            "n_samples": n_samples,
            "d_model": d_model,
            "intrinsic_dim": pr,
            "effective_rank_1pct": erank,
            "effective_rank_05pct": erank_05,
            "dim_for_90pct_var": n_90,
            "dim_for_95pct_var": n_95,
            "dim_for_99pct_var": n_99,
            "decay_rate": decay_rate,
            "top_20_singular_values": [float(x) for x in S[:20]],
            "singular_value_sum": float(np.sum(S)),
            "total_variance": float(total_var),
        }
        
        if li % 5 == 0 or li == n_layers - 1:
            print(f"  L{li:2d}: ID={pr:.2f}, eRank={erank}, "
                  f"90%var={n_90}d, 95%var={n_95}d, 99%var={n_99}d, "
                  f"decay={decay_rate:.4f}")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"exp1_svd_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    # 计算Ω压缩比
    shallow_ids = [results[f"L{i}"]["intrinsic_dim"] for i in range(min(n_layers//3, n_layers)) if f"L{i}" in results]
    deep_ids = [results[f"L{i}"]["intrinsic_dim"] for i in range(n_layers//3, n_layers) if f"L{i}" in results]
    
    if shallow_ids and deep_ids:
        max_id_shallow = max(shallow_ids)
        min_id_deep = min(deep_ids)
        omega = max_id_shallow / max(min_id_deep, 0.01)
        print(f"\nΩ压缩比: max(ID_shallow)={max_id_shallow:.2f}, min(ID_deep)={min_id_deep:.2f}, Ω={omega:.2f}")
    
    return results


# ============================================================
# Exp2: SVCCA层间对齐度
# ============================================================
def exp2_svcca_alignment(model, tokenizer, device, model_info):
    """
    SVCCA (Singular Vector CCA): 测量相邻层子空间的对齐度
    
    核心思想: 如果L_i和L_{i+1}使用相似的子空间，则CCA相关系数高
    如果层间发生了子空间旋转/跳转，则CCA相关系数低
    """
    print("\n" + "="*70)
    print("Exp2: SVCCA层间对齐度分析")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    n_components = min(50, d_model)  # CCA分量数
    
    # 收集激活
    print(f"收集 {len(DIVERSE_PROMPTS)} 个prompt的各层残差流...")
    layer_activations = {f"L{i}": [] for i in range(n_layers)}
    
    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                last_tok_h = captured[key][0, -1, :].numpy()
                layer_activations[key].append(last_tok_h)
        
        del captured
        gc.collect()
    
    # 对每层做SVD，取前n_components个奇异向量
    print(f"\n计算各层SVD...")
    layer_sv_bases = {}
    
    for li in range(n_layers):
        key = f"L{li}"
        acts = np.array(layer_activations[key], dtype=np.float32)
        acts_centered = acts - acts.mean(axis=0, keepdims=True)
        
        U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        # 取前n_components个右奇异向量 (每个是d_model维)
        k = min(n_components, Vt.shape[0])
        layer_sv_bases[li] = Vt[:k, :]  # [k, d_model]
    
    # CCA: 对相邻层计算子空间对齐度
    print(f"\n计算SVCCA层间对齐度 (n_components={n_components})...")
    results = {}
    
    for li in range(n_layers - 1):
        # L_i和L_{i+1}的SVD基
        V_i = layer_sv_bases[li].T    # [d_model, k]
        V_ip1 = layer_sv_bases[li+1].T  # [d_model, k]
        
        # CCA: 计算两个子空间的相关系数
        # 简化版SVCCA: 计算 V_i^T @ V_{i+1} 的奇异值
        cross_corr = V_i.T @ V_ip1  # [k, k]
        svd_cross = np.linalg.svd(cross_corr, compute_uv=False)
        
        # 平均SVCCA相关系数
        mean_cca = float(np.mean(svd_cross))
        
        # 子空间重叠度: |cos(θ_i)|^2的平均 (Grassmann距离)
        # 等价于 Frobenius norm of V_i^T V_{i+1} / k
        subspace_overlap = float(np.linalg.norm(cross_corr, 'fro')**2 / n_components)
        
        key = f"L{li}_L{li+1}"
        results[key] = {
            "layer_pair": [li, li+1],
            "mean_svcca": mean_cca,
            "subspace_overlap": subspace_overlap,
            "top_10_svcca_coeffs": [float(x) for x in svd_cross[:10]],
        }
        
        if li % 5 == 0:
            print(f"  L{li:2d}→L{li+1:2d}: mean_SVCCA={mean_cca:.4f}, overlap={subspace_overlap:.4f}")
    
    # 找关键转换层 (SVCCA最低 = 最大子空间跳转)
    transitions = [(k, v["mean_svcca"]) for k, v in results.items()]
    transitions.sort(key=lambda x: x[1])
    
    print(f"\n=== 关键转换层 (SVCCA最低 = 最大子空间跳转) ===")
    for key, val in transitions[:5]:
        print(f"  {key}: mean_SVCCA = {val:.4f}")
    
    out_path = OUTPUT_DIR / f"exp2_svcca_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    return results


# ============================================================
# Exp3: 特征维度手术 — 逐步切除主成分直到功能崩溃
# ============================================================
def exp3_dimension_surgery(model, tokenizer, device, model_info):
    """
    对关键特征(否定、翻译、推理等)做"维度手术":
    1. 收集有特征和无特征的激活差
    2. 对差值做SVD得到主成分
    3. 逐步切除主成分，测量功能退化(KL散度)
    4. 记录"临界维度数"——切除多少维功能才崩溃
    
    这直接回答: 每个特征用多少维编码？
    """
    print("\n" + "="*70)
    print("Exp3: 特征维度手术")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    W_U = get_W_U(model, model_info.name)  # [vocab_size, d_model]
    
    results = {}
    
    for feature_name, prompts in FEATURE_GROUPS.items():
        print(f"\n--- 特征: {feature_name} ---")
        
        # 收集每对prompt的激活差
        # 假设prompts[0]是基线，后续是有特征的
        baseline_prompt = prompts[0]
        feature_prompts = prompts[1:]
        
        # 收集基线激活
        baseline_acts = {}
        toks = tokenizer(baseline_prompt, return_tensors="pt").to(device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(toks.input_ids.shape[1], device=device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                baseline_out = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
                baseline_logits = baseline_out.logits[0, -1, :].detach().float().cpu().numpy()
            except Exception:
                baseline_logits = None
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                baseline_acts[li] = captured[key][0, -1, :].numpy()  # [d_model]
        
        del captured
        gc.collect()
        
        # 对每个feature prompt
        for fi, fprompt in enumerate(feature_prompts):
            print(f"  对比: '{baseline_prompt[:40]}...' vs '{fprompt[:40]}...'")
            
            toks_f = tokenizer(fprompt, return_tensors="pt").to(device)
            inputs_embeds_f = embed_layer(toks_f.input_ids).detach().clone().to(model.dtype)
            position_ids_f = torch.arange(toks_f.input_ids.shape[1], device=device).unsqueeze(0)
            
            captured_f = {}
            hooks_f = []
            for li in range(n_layers):
                layer = layers[li]
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_f[key] = output[0].detach().float().cpu()
                        else:
                            captured_f[key] = output.detach().float().cpu()
                    return hook
                hooks_f.append(layer.register_forward_hook(make_hook(f"L{li}")))
            
            with torch.no_grad():
                try:
                    feature_out = model(inputs_embeds=inputs_embeds_f, position_ids=position_ids_f)
                    feature_logits = feature_out.logits[0, -1, :].detach().float().cpu().numpy()
                except Exception:
                    feature_logits = None
            
            for h in hooks_f:
                h.remove()
            
            # 收集差值
            layer_deltas = {}
            for li in range(n_layers):
                key = f"L{li}"
                if key in captured_f and li in baseline_acts:
                    feature_act = captured_f[key][0, -1, :].numpy()
                    # 注意: 只取最后一个token的差值
                    # 但两个prompt长度可能不同，取各自最后一个
                    layer_deltas[li] = feature_act - baseline_acts[li]
            
            del captured_f
            gc.collect()
            
            # 找差值最大的层（特征编码最显著的层）
            delta_norms = {li: float(np.linalg.norm(d)) for li, d in layer_deltas.items()}
            if not delta_norms:
                continue
            
            peak_layer = max(delta_norms, key=delta_norms.get)
            peak_delta = layer_deltas[peak_layer]
            
            print(f"    差值峰值层: L{peak_layer}, ||delta||={delta_norms[peak_layer]:.4f}")
            
            # SVD分解差值
            # 这里差值是一个向量，无法直接SVD
            # 改用: 收集多对prompt在同一层的差值，然后SVD
            # 简化: 只分析峰值层的差值方向
            
            # 计算差值在W_U行空间中的投影（Logit影响）
            if feature_logits is not None and baseline_logits is not None:
                logit_diff = feature_logits - baseline_logits
                # KL散度
                from scipy.special import softmax, kl_div
                p = softmax(feature_logits)
                q = softmax(baseline_logits)
                kl = float(np.sum(kl_div(p, q)))
                
                # Top-1 token变化
                top1_base = int(np.argmax(baseline_logits))
                top1_feat = int(np.argmax(feature_logits))
                top1_changed = top1_base != top1_feat
                
                print(f"    KL={kl:.4f}, top1_changed={top1_changed} "
                      f"('{tokenizer.decode([top1_base])}'→'{tokenizer.decode([top1_feat])}')")
                
                # 维度手术: 在峰值层，沿差值方向逐步切除
                surgery_results = []
                peak_act = baseline_acts[peak_layer].copy()
                delta_norm = np.linalg.norm(peak_delta)
                if delta_norm < 1e-10:
                    continue
                
                delta_direction = peak_delta / delta_norm
                
                # 逐步增加"切除量" (从0到2倍delta_norm)
                for alpha in np.linspace(0, 2.0, 21):
                    # 在baseline激活上 + alpha * delta_direction
                    modified_act = baseline_acts[peak_layer] + alpha * peak_delta
                    
                    # 投影到logits (用W_U)
                    # 注意: 实际logits经过LayerNorm, 这里是近似
                    modified_logits_approx = W_U @ modified_act
                    baseline_logits_approx = W_U @ baseline_acts[peak_layer]
                    
                    # 近似KL
                    p_mod = softmax(modified_logits_approx)
                    q_base = softmax(baseline_logits_approx)
                    kl_approx = float(np.sum(kl_div(p_mod, q_base)))
                    
                    # Top-1变化
                    top1_mod = int(np.argmax(modified_logits_approx))
                    top1_base_approx = int(np.argmax(baseline_logits_approx))
                    
                    surgery_results.append({
                        "alpha": float(alpha),
                        "kl_approx": kl_approx,
                        "top1_matches_feature": top1_mod == top1_feat,
                    })
                
                result_key = f"{feature_name}_f{fi}"
                results[result_key] = {
                    "feature": feature_name,
                    "baseline": baseline_prompt[:50],
                    "feature_prompt": fprompt[:50],
                    "peak_layer": peak_layer,
                    "delta_norm": float(delta_norm),
                    "kl_exact": kl,
                    "top1_changed": top1_changed,
                    "surgery": surgery_results,
                }
    
    out_path = OUTPUT_DIR / f"exp3_surgery_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    return results


# ============================================================
# Exp4: 子空间重叠度 — 不同特征子空间共享多少维度
# ============================================================
def exp4_subspace_overlap(model, tokenizer, device, model_info):
    """
    收集不同特征在各层的激活，提取各特征的子空间基，
    计算子空间之间的重叠度
    
    这回答: 否定和疑问的子空间共享多少维？翻译和代码呢？
    """
    print("\n" + "="*70)
    print("Exp4: 子空间重叠度分析")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    n_basis = min(20, d_model)  # 每个特征提取的基向量数
    
    # 收集各特征在各层的激活
    print(f"收集各特征组在各层的激活...")
    feature_acts = {}  # {feature_name: {layer: [n_samples, d_model]}}
    
    for feature_name, prompts in FEATURE_GROUPS.items():
        feature_acts[feature_name] = {li: [] for li in range(n_layers)}
        
        for prompt in prompts:
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
            position_ids = torch.arange(toks.input_ids.shape[1], device=device).unsqueeze(0)
            
            captured = {}
            hooks = []
            for li in range(n_layers):
                layer = layers[li]
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook
                hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
                except Exception:
                    pass
            
            for h in hooks:
                h.remove()
            
            for li in range(n_layers):
                key = f"L{li}"
                if key in captured:
                    feature_acts[feature_name][li].append(captured[key][0, -1, :].numpy())
            
            del captured
            gc.collect()
    
    # 对每个特征，在每层提取子空间基
    print(f"\n提取各特征的子空间基...")
    feature_bases = {}  # {feature_name: {layer: [n_basis, d_model]}}
    
    for feature_name in FEATURE_GROUPS:
        feature_bases[feature_name] = {}
        for li in range(n_layers):
            acts = np.array(feature_acts[feature_name][li], dtype=np.float32)
            if acts.shape[0] < 2:
                continue
            acts_centered = acts - acts.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
            k = min(n_basis, Vt.shape[0])
            feature_bases[feature_name][li] = Vt[:k, :]  # [k, d_model]
    
    # 计算特征对之间的子空间重叠度
    print(f"\n计算特征对之间的子空间重叠度...")
    feature_names = list(FEATURE_GROUPS.keys())
    results = {}
    
    for li in range(n_layers):
        # 获取该层所有特征的基
        available = [fn for fn in feature_names if li in feature_bases[fn]]
        if len(available) < 2:
            continue
        
        layer_overlaps = {}
        for i in range(len(available)):
            for j in range(i+1, len(available)):
                fn_i, fn_j = available[i], available[j]
                V_i = feature_bases[fn_i][li].T  # [d_model, k]
                V_j = feature_bases[fn_j][li].T  # [d_model, k]
                
                # 子空间重叠度 = ||V_i^T V_j||_F^2 / min(k_i, k_j)
                cross = V_i.T @ V_j  # [k_i, k_j]
                overlap = float(np.linalg.norm(cross, 'fro')**2 / min(V_i.shape[1], V_j.shape[1]))
                
                # 主角度 (principal angles)
                svd_cross = np.linalg.svd(cross, compute_uv=False)
                min_angle = float(np.arccos(np.clip(svd_cross[0], -1, 1)) * 180 / np.pi)
                
                pair_key = f"{fn_i}_vs_{fn_j}"
                layer_overlaps[pair_key] = {
                    "overlap": overlap,
                    "min_angle_deg": min_angle,
                    "top_5_svcca": [float(x) for x in svd_cross[:5]],
                }
        
        if layer_overlaps:
            results[f"L{li}"] = {
                "layer": li,
                "overlaps": layer_overlaps,
            }
    
    # 打印关键层的重叠度
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    key_layers = [l for l in key_layers if f"L{l}" in results]
    
    print(f"\n=== 关键层子空间重叠度 ===")
    for lkey in key_layers:
        r = results[f"L{lkey}"]
        print(f"\n  L{lkey}:")
        for pair, data in r["overlaps"].items():
            print(f"    {pair}: overlap={data['overlap']:.4f}, min_angle={data['min_angle_deg']:.1f}°")
    
    out_path = OUTPUT_DIR / f"exp4_overlap_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Subspace Topology Phase 1")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0, help="Experiment number (0=all, 1-4)")
    args = parser.parse_args()
    
    print(f"模型: {args.model}, 实验: {args.exp}")
    
    # 加载模型
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"模型信息: {model_info.n_layers}层, d_model={model_info.d_model}, class={model_info.model_class}")
    
    try:
        if args.exp == 0 or args.exp == 1:
            exp1_full_layer_svd(model, tokenizer, device, model_info)
        
        if args.exp == 0 or args.exp == 2:
            exp2_svcca_alignment(model, tokenizer, device, model_info)
        
        if args.exp == 0 or args.exp == 3:
            exp3_dimension_surgery(model, tokenizer, device, model_info)
        
        if args.exp == 0 or args.exp == 4:
            exp4_subspace_overlap(model, tokenizer, device, model_info)
    finally:
        release_model(model)
    
    print(f"\n{'='*70}")
    print(f"阶段1实验完成! 结果保存在 {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
