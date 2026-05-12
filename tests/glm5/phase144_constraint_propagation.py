"""
Phase 144: 约束传播系统 — 直接检验"分段vs光滑"与"约束传播"假说
=============================================================

回应Phase 143用户批评的核心争议:

1. "分段低秩动力系统" vs "近似光滑流形" — Phase 143用NOT-pair测试了cos≈0.94,
   但NOT-pair语义差异太小。需要跨领域句子测试。
2. "约束传播系统" — 核心假说: Transformer维持"可继续预测"的约束一致性
3. "MLP是约束修正器" — MLP将状态投影回训练分布允许区域
4. "Attention是约束路由" — 不同head负责不同类型的约束运输

四大实验:
A. 跨领域Jacobian一致性 (检验"分段vs光滑"的最关键实验)
B. 约束违背动力学 (直接检验"约束传播"假说)
C. MLP约束修正效应 (检验"MLP是约束修正器")
D. Attention head功能聚类 (检验"Attention是约束路由")

用法: python phase144_constraint_propagation.py [model_name]
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, 'tests/glm5')

import gc
import torch
from model_utils import (load_model, get_layers, get_model_info, get_W_U, release_model,
                          get_layer_weights, get_sample_layers, collect_layer_outputs)

# ===== 输出路径 =====
TEMP_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
TEMP_DIR.mkdir(exist_ok=True)

# ===== 实验A: 跨领域Jacobian一致性 =====
# 核心问题: Phase 143的cos≈0.94是用NOT-pair测的(语义差异很小)。
# 如果系统是"分段"的，那么跨领域句子的cos应该大幅下降。
# 如果是"光滑"的，跨领域cos仍应保持高值。

CROSS_DOMAIN_PAIRS = [
    # 类别1: 语义近邻(Phase 143已测, 作为baseline)
    ("The cat sat on the mat", "The cat did not sit on the mat", "NOT"),
    ("She is happy", "She is not happy", "NOT"),
    
    # 类别2: 语法变化(时态/数)
    ("The cat runs fast", "The cats run fast", "NUM"),
    ("I went to the store", "I will go to the store", "TENSE"),
    ("He was running", "He is running", "TENSE"),
    
    # 类别3: 语义替换(同范畴)
    ("The cat sat on the mat", "The dog sat on the mat", "SYN"),
    ("She loves music", "He loves music", "SYN"),
    ("The apple is red", "The sky is blue", "SYN"),
    
    # 类别4: 句法变化(陈述/疑问/否定)
    ("The cat is on the mat", "Is the cat on the mat?", "Q"),
    ("All birds can fly", "Not all birds can fly", "SCOPE"),
    ("Every student passed", "Not every student passed", "SCOPE"),
    
    # 类别5: 完全不同领域
    ("The cat sat on the mat", "Quantum physics describes particles", "CROSS"),
    ("I love classical music", "The stock market crashed today", "CROSS"),
    ("She baked a chocolate cake", "The algorithm sorts the array", "CROSS"),
    ("The river flows through the valley", "The function returns an integer", "CROSS"),
    
    # 类别6: 逻辑/数学 vs 自然语言
    ("Two plus three equals five", "The sunset was beautiful", "CROSS"),
    ("If A then B, A, therefore B", "The flowers bloomed in spring", "CROSS"),
    
    # 类别7: 无意义 vs 有意义
    ("The cat sat on the mat", "Colorless green ideas sleep furiously", "NONSENSE"),
    ("She walked to the store", "The procedural abstraction crystallizes", "NONSENSE"),
]

def experiment_a_cross_domain_jacobian(model, tokenizer, device, model_info):
    """
    实验A: 跨领域Jacobian一致性
    
    方法: 对每对句子(s1, s2), 在同一中间层上:
    1. 用s1的hidden state h1作为基点, 计算 J(h1)*v
    2. 用s2的hidden state h2作为基点, 计算 J(h2)*v (同一v)
    3. 计算 cos(J(h1)*v, J(h2)*v)
    
    如果系统是"分段"的: 跨领域pair的cos应显著低于NOT-pair
    如果系统是"光滑"的: 跨领域pair的cos与NOT-pair接近
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    sample_layers = get_sample_layers(n_layers, 8)
    
    results = {}
    epsilons = [0.5, 2.0]  # 用Phase 143b验证过的大epsilon
    
    for eps in epsilons:
        eps_results = {}
        for s1, s2, category in CROSS_DOMAIN_PAIRS:
            pair_key = f"{category}:{s1[:30]}...→{s2[:30]}..."
            layer_cosines = []
            
            for li in sample_layers:
                if li >= n_layers - 1:
                    continue
                    
                # 获取两个句子的hidden states
                with torch.no_grad():
                    toks1 = tokenizer(s1, return_tensors="pt").to(device)
                    toks2 = tokenizer(s2, return_tensors="pt").to(device)
                    
                    embed_layer = model.get_input_embeddings()
                    emb1 = embed_layer(toks1.input_ids)
                    emb2 = embed_layer(toks2.input_ids)
                    
                    # 前向传播到层li
                    h1 = forward_to_layer(model, emb1, toks1, li)
                    h2 = forward_to_layer(model, emb2, toks2, li)
                    
                    if h1 is None or h2 is None:
                        continue
                    
                    # 取last token位置
                    h1_last = h1[0, -1, :].float()
                    h2_last = h2[0, -1, :].float()
                    
                    # 用随机方向v, 比较J(h1)*v vs J(h2)*v
                    n_trials = 5
                    trial_cosines = []
                    for _ in range(n_trials):
                        v = torch.randn(d_model, device=device, dtype=torch.float32)
                        v = v / v.norm()
                        
                        # J(h1)*v ≈ [F(h1+εv) - F(h1-εv)] / (2ε)
                        delta_h1_pos = h1_last + eps * v
                        delta_h1_neg = h1_last - eps * v
                        delta_h2_pos = h2_last + eps * v
                        delta_h2_neg = h2_last - eps * v
                        
                        # 从层li+1前向传播到层li+1的输出
                        out1_pos = forward_one_layer(model, delta_h1_pos, toks1, li)
                        out1_neg = forward_one_layer(model, delta_h1_neg, toks1, li)
                        out2_pos = forward_one_layer(model, delta_h2_pos, toks2, li)
                        out2_neg = forward_one_layer(model, delta_h2_neg, toks2, li)
                        
                        if out1_pos is None or out2_pos is None:
                            continue
                        
                        jv1 = (out1_pos - out1_neg) / (2 * eps)
                        jv2 = (out2_pos - out2_neg) / (2 * eps)
                        
                        n1 = jv1.norm()
                        n2 = jv2.norm()
                        if n1 < 1e-6 or n2 < 1e-6:
                            continue
                        
                        cos_val = float(torch.nn.functional.cosine_similarity(
                            jv1.unsqueeze(0), jv2.unsqueeze(0)
                        ).item())
                        trial_cosines.append(cos_val)
                    
                    if trial_cosines:
                        layer_cosines.append(np.mean(trial_cosines))
            
            if layer_cosines:
                eps_results[pair_key] = {
                    "s1": s1,
                    "s2": s2,
                    "category": category,
                    "mean_cos": float(np.mean(layer_cosines)),
                    "std_cos": float(np.std(layer_cosines)),
                    "n_layers": len(layer_cosines),
                }
        
        # 按类别聚合
        category_stats = defaultdict(list)
        for key, val in eps_results.items():
            category_stats[val["category"]].append(val["mean_cos"])
        
        results[f"eps_{eps}"] = {
            "by_pair": eps_results,
            "by_category": {
                cat: {
                    "mean_cos": float(np.mean(vals)),
                    "std_cos": float(np.std(vals)),
                    "n_pairs": len(vals),
                }
                for cat, vals in category_stats.items()
            }
        }
    
    return results


# ===== 实验B: 约束违背动力学 =====
# 核心假说: 如果Transformer是"约束传播系统"，那么约束违背的输入
# 应该在中间层被"修正"(MLP+Attention将状态拉回可行域)

CONSTRAINT_PAIRS = [
    # 类别1: 主谓一致 (Subject-Verb Agreement)
    ("The cat walks slowly", "The cat walk slowly", "SVA"),  # 正确 vs 错误
    ("The dogs run fast", "The dogs runs fast", "SVA"),
    ("She has been working", "She have been working", "SVA"),
    ("The children were playing", "The children was playing", "SVA"),
    
    # 类别2: 时态一致 (Tense Consistency)
    ("Yesterday I went to the store", "Yesterday I will go to the store", "TENSE"),
    ("She has already finished", "She has already finish", "TENSE"),
    ("He was walking when it rained", "He was walking when it rains", "TENSE"),
    
    # 类别3: 否定范围 (Scope/Negation)
    ("All birds cannot fly", "Not all birds can fly", "SCOPE"),  # 歧义scope
    ("Everyone did not pass", "Not everyone passed", "SCOPE"),
    
    # 类别4: 逻辑一致性
    ("The square has four sides", "The square has three sides", "LOGIC"),
    ("Water freezes at zero degrees", "Water freezes at one hundred degrees", "LOGIC"),
    ("Two plus two equals four", "Two plus two equals five", "LOGIC"),
    
    # 类别5: 语义异常(语法正确但语义冲突)
    ("The cat sat on the mat", "The cat sat on the idea", "SEMANTIC"),
    ("She ate the apple", "She ate the democracy", "SEMANTIC"),
    ("He drove the car", "He drove the silence", "SEMANTIC"),
]

def experiment_b_constraint_violation(model, tokenizer, device, model_info):
    """
    实验B: 约束违背动力学
    
    核心预测: 如果Transformer是"约束传播系统"，那么:
    1. 约束正确的输入 → hidden state稳定传播
    2. 约束违背的输入 → 中间层应出现"修正"信号
       - MLP修正: MLP输出将hidden state拉向"正确"方向
       - Attention修正: Attention模式可能"忽略"冲突token
    
    方法:
    1. 对每对(正确, 错误)句子, 收集每层的hidden state
    2. 计算 delta = h_correct - h_wrong (约束违背信号)
    3. 跟踪delta的范数和方向随层的演化
    4. 分析MLP对delta的"修正"效应
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    sample_layers = get_sample_layers(n_layers, n_layers)
    
    results = {}
    
    for correct, wrong, category in CONSTRAINT_PAIRS:
        pair_key = f"{category}:{correct[:30]}...→{wrong[:30]}..."
        
        with torch.no_grad():
            # 收集两个句子的各层hidden states
            toks_c = tokenizer(correct, return_tensors="pt").to(device)
            toks_w = tokenizer(wrong, return_tensors="pt").to(device)
            
            embed_layer = model.get_input_embeddings()
            emb_c = embed_layer(toks_c.input_ids)
            emb_w = embed_layer(toks_w.input_ids)
            
            layers_c = collect_all_layer_outputs(model, emb_c, toks_c, n_layers)
            layers_w = collect_all_layer_outputs(model, emb_w, toks_w, n_layers)
        
        layer_deltas = {}
        for li in sample_layers:
            key = f"L{li}"
            if key not in layers_c or key not in layers_w:
                continue
            
            h_c = layers_c[key][0, -1, :].float().numpy()  # last token
            h_w = layers_w[key][0, -1, :].float().numpy()
            
            delta = h_c - h_w
            delta_norm = float(np.linalg.norm(delta))
            hc_norm = float(np.linalg.norm(h_c))
            hw_norm = float(np.linalg.norm(h_w))
            
            # delta方向是否对齐h_correct? (如果MLP"修正"了错误方向)
            cos_delta_hc = float(np.dot(delta, h_c) / max(np.linalg.norm(delta) * hc_norm, 1e-10))
            
            layer_deltas[li] = {
                "delta_norm": delta_norm,
                "hc_norm": hc_norm,
                "hw_norm": hw_norm,
                "norm_ratio": delta_norm / max(hc_norm, 1e-10),
                "cos_delta_hc": cos_delta_hc,
            }
        
        # 分析delta范数随层的演化
        if layer_deltas:
            sorted_layers = sorted(layer_deltas.keys())
            delta_norms = [layer_deltas[l]["delta_norm"] for l in sorted_layers]
            
            # 峰值层和末层
            peak_idx = np.argmax(delta_norms)
            peak_layer = sorted_layers[peak_idx]
            last_layer = sorted_layers[-1]
            
            results[pair_key] = {
                "s_correct": correct,
                "s_wrong": wrong,
                "category": category,
                "layer_deltas": {str(k): v for k, v in layer_deltas.items()},
                "delta_at_first": delta_norms[0],
                "delta_at_peak": delta_norms[peak_idx],
                "delta_at_last": delta_norms[-1],
                "peak_layer": peak_layer,
                "amplification": delta_norms[peak_idx] / max(delta_norms[0], 1e-10),
                "retention": delta_norms[-1] / max(delta_norms[peak_idx], 1e-10),
            }
    
    # 按类别聚合
    category_stats = defaultdict(lambda: {"amplifications": [], "retentions": [], "first_deltas": [], "last_deltas": []})
    for key, val in results.items():
        cat = val["category"]
        category_stats[cat]["amplifications"].append(val["amplification"])
        category_stats[cat]["retentions"].append(val["retention"])
        category_stats[cat]["first_deltas"].append(val["delta_at_first"])
        category_stats[cat]["last_deltas"].append(val["delta_at_last"])
    
    aggregated = {}
    for cat, vals in category_stats.items():
        aggregated[cat] = {
            "mean_amplification": float(np.mean(vals["amplifications"])),
            "mean_retention": float(np.mean(vals["retentions"])),
            "mean_first_delta": float(np.mean(vals["first_deltas"])),
            "mean_last_delta": float(np.mean(vals["last_deltas"])),
            "n_pairs": len(vals["amplifications"]),
        }
    
    return {
        "by_pair": results,
        "by_category": aggregated,
    }


# ===== 实验C: MLP约束修正效应 =====
# 核心假说: MLP将hidden state投影回"训练分布允许区域"
# 如果正确: 约束违背输入的MLP输出应有更大的"修正"分量

def experiment_c_mlp_correction(model, tokenizer, device, model_info):
    """
    实验C: MLP约束修正
    
    方法:
    1. 对(正确, 错误)句子对, 分别收集每层的:
       - Transformer层输出 h_layer
       - MLP输出 mlp_out (hook)
    2. 计算: MLP修正 = mlp_out_correct - mlp_out_wrong
    3. 比较: MLP修正是否对齐"正确方向"(h_correct - h_wrong)
    
    如果MLP是"约束修正器":
    - MLP修正应与约束违背信号(h_c - h_w)正高度相关
    - 即: MLP在错误输入上产生的输出更接近正确输入
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    sample_layers = get_sample_layers(n_layers, 10)
    
    results = {}
    
    for correct, wrong, category in CONSTRAINT_PAIRS[:10]:  # 选取部分测试
        pair_key = f"{category}:{correct[:25]}...→{wrong[:25]}..."
        
        with torch.no_grad():
            toks_c = tokenizer(correct, return_tensors="pt").to(device)
            toks_w = tokenizer(wrong, return_tensors="pt").to(device)
            
            embed_layer = model.get_input_embeddings()
            emb_c = embed_layer(toks_c.input_ids)
            emb_w = embed_layer(toks_w.input_ids)
            
            # 收集层输出和MLP输出
            outputs_c = collect_layer_and_mlp_outputs(model, emb_c, toks_c, sample_layers)
            outputs_w = collect_layer_and_mlp_outputs(model, emb_w, toks_w, sample_layers)
        
        layer_analysis = {}
        for li in sample_layers:
            key_layer = f"L{li}"
            key_mlp = f"MLP{li}"
            
            if key_layer not in outputs_c or key_layer not in outputs_w:
                continue
            if key_mlp not in outputs_c or key_mlp not in outputs_w:
                continue
            
            h_c = outputs_c[key_layer][0, -1, :].float().numpy()
            h_w = outputs_w[key_layer][0, -1, :].float().numpy()
            mlp_c = outputs_c[key_mlp][0, -1, :].float().numpy()
            mlp_w = outputs_w[key_mlp][0, -1, :].float().numpy()
            
            # 约束违背信号
            delta_h = h_c - h_w
            # MLP修正信号
            delta_mlp = mlp_c - mlp_w
            
            dn = np.linalg.norm(delta_h)
            mn = np.linalg.norm(delta_mlp)
            
            if dn > 1e-10 and mn > 1e-10:
                cos_alignment = float(np.dot(delta_h, delta_mlp) / (dn * mn))
            else:
                cos_alignment = 0.0
            
            # MLP修正的"力度": MLP输出范数 vs 层输出范数
            mlp_correction_strength = mn / max(dn, 1e-10)
            
            layer_analysis[li] = {
                "delta_h_norm": float(dn),
                "delta_mlp_norm": float(mn),
                "cos_alignment": cos_alignment,  # MLP修正与约束信号的对齐度
                "mlp_correction_strength": mlp_correction_strength,
            }
        
        if layer_analysis:
            # 聚合
            aligns = [v["cos_alignment"] for v in layer_analysis.values()]
            strengths = [v["mlp_correction_strength"] for v in layer_analysis.values()]
            
            results[pair_key] = {
                "s_correct": correct,
                "s_wrong": wrong,
                "category": category,
                "layer_analysis": {str(k): v for k, v in layer_analysis.items()},
                "mean_alignment": float(np.mean(aligns)),
                "mean_correction_strength": float(np.mean(strengths)),
            }
    
    # 按类别聚合
    category_stats = defaultdict(lambda: {"alignments": [], "strengths": []})
    for key, val in results.items():
        cat = val["category"]
        category_stats[cat]["alignments"].append(val["mean_alignment"])
        category_stats[cat]["strengths"].append(val["mean_correction_strength"])
    
    aggregated = {}
    for cat, vals in category_stats.items():
        aggregated[cat] = {
            "mean_alignment": float(np.mean(vals["alignments"])),
            "mean_correction_strength": float(np.mean(vals["strengths"])),
            "n_pairs": len(vals["alignments"]),
        }
    
    return {
        "by_pair": results,
        "by_category": aggregated,
    }


# ===== 实验D: Attention Head功能聚类 =====
# 核心假说: 不同head负责不同类型的约束运输

def experiment_d_attention_clustering(model, tokenizer, device, model_info):
    """
    实验D: Attention Head功能聚类
    
    方法:
    1. 对多种句子, 收集每层每个head的attention pattern
    2. 构建每个head的"激活指纹": 在不同句子类型上的attention分布特征
    3. 聚类heads, 看是否有功能分化
    
    如果"Attention是约束路由":
    - 某些head应对语法约束敏感
    - 某些head应对逻辑约束敏感
    - 应该出现功能聚类
    """
    n_layers = model_info.n_layers
    
    # 多种测试句子
    test_sentences = [
        # 语法约束
        "The cat walks slowly",           # SVA正确
        "The cat walk slowly",            # SVA错误
        "The dogs run fast",              # SVA正确
        "The dogs runs fast",             # SVA错误
        # 否定/Scope
        "All birds can fly",
        "Not all birds can fly",
        "Every student passed",
        "Not every student passed",
        # 语义
        "The cat sat on the mat",
        "The cat sat on the idea",        # 语义异常
        "She ate the apple",
        "She ate the democracy",          # 语义异常
        # 逻辑
        "Two plus two equals four",
        "Two plus two equals five",       # 逻辑错误
        # 时态
        "Yesterday I went home",
        "Yesterday I will go home",       # 时态冲突
    ]
    
    # 采样层(太多层太慢, 采样8层)
    sample_layers = get_sample_layers(n_layers, 8)
    
    # 对每个句子, 收集attention patterns
    head_features = {}  # {f"L{li}_H{hi}": [features_per_sentence]}
    
    for sent_idx, sent in enumerate(test_sentences):
        print(f"  [D] Sentence {sent_idx+1}/{len(test_sentences)}: {sent[:40]}...")
        
        with torch.no_grad():
            toks = tokenizer(sent, return_tensors="pt").to(device)
            embed_layer = model.get_input_embeddings()
            emb = embed_layer(toks.input_ids)
            
            attn_data = collect_attention_patterns(model, emb, toks, sample_layers)
        
        for li in sample_layers:
            key = f"L{li}"
            if key not in attn_data:
                continue
            
            # attn_data[key]: [n_heads, seq_len, seq_len]
            attn_matrix = attn_data[key]
            n_heads = attn_matrix.shape[0]
            seq_len = attn_matrix.shape[1]
            
            for hi in range(n_heads):
                head_key = f"L{li}_H{hi}"
                
                # 提取特征: last token对各token的attention权重
                # 这反映了"当前token从哪些上下文token读取信息"
                attn_to_last = attn_matrix[hi, -1, :]  # [seq_len]
                
                # 特征: mean, max, entropy, self-attention weight
                feat = [
                    float(attn_to_last.mean()),
                    float(attn_to_last.max()),
                    float(-np.sum(attn_to_last * np.log(attn_to_last + 1e-10))),  # entropy
                    float(attn_to_last[-1]),  # self-attention
                ]
                
                if head_key not in head_features:
                    head_features[head_key] = []
                head_features[head_key].append(feat)
    
    # 对每个head, 构建特征向量(跨句子的统计特征)
    head_vectors = {}
    for head_key, features in head_features.items():
        features = np.array(features)  # [n_sentences, 4]
        # 展平为 [n_sentences * 4] 维向量
        head_vectors[head_key] = features.flatten()
    
    # 聚类 (用简单的K-means)
    if len(head_vectors) > 10:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        keys = list(head_vectors.keys())
        X = np.array([head_vectors[k] for k in keys])
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means聚类, k=5
        n_clusters = min(5, len(keys))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # 分析每个cluster的层分布
        clusters = defaultdict(list)
        for i, k in enumerate(keys):
            clusters[int(labels[i])].append(k)
        
        cluster_analysis = {}
        for cid, heads in clusters.items():
            layers_in_cluster = [int(h.split("_")[0][1:]) for h in heads]
            cluster_analysis[cid] = {
                "n_heads": len(heads),
                "mean_layer": float(np.mean(layers_in_cluster)),
                "std_layer": float(np.std(layers_in_cluster)),
                "min_layer": int(min(layers_in_cluster)),
                "max_layer": int(max(layers_in_cluster)),
                "heads": heads[:20],  # 最多列20个
            }
        
        return {
            "n_heads_total": len(keys),
            "n_clusters": n_clusters,
            "cluster_analysis": cluster_analysis,
        }
    
    return {"n_heads_total": len(head_vectors), "error": "too few heads for clustering"}


# ===== 辅助函数 =====

import torch

def forward_to_layer(model, inputs_embeds, toks, target_layer):
    """前向传播到指定层, 返回该层输出的hidden state"""
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 获取position_ids
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
    
    # 获取attention_mask
    attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
    
    hidden = inputs_embeds
    
    for li in range(min(target_layer + 1, n_layers)):
        layer = layers[li]
        
        # LayerNorm
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, "weight"):
                    hidden = layer_norm_forward(hidden, ln)
                break
        
        # Self-attention (简化: 跳过详细计算, 直接用hook方式)
        # 这里我们需要完整的forward, 所以用model的forward
        # 改用hook方式
        pass
    
    # 实际实现: 用hook收集
    captured = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['target'] = output[0].detach()
        else:
            captured['target'] = output.detach()
    
    handle = layers[target_layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        try:
            if attention_mask is not None:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask)
            else:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception as e:
            handle.remove()
            return None
    
    handle.remove()
    
    if 'target' in captured:
        return captured['target'].float()
    return None


def forward_one_layer(model, h_input, toks, layer_idx):
    """
    从层layer_idx的hidden state开始, 前向传播一层
    
    这是一个简化实现: 用完整的forward但通过hook获取特定层输出
    """
    layers = get_layers(model)
    n_layers = len(layers)
    
    if layer_idx + 1 >= n_layers:
        return None
    
    captured = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['next'] = output[0].detach()
        else:
            captured['next'] = output.detach()
    
    handle = layers[layer_idx + 1].register_forward_hook(hook_fn)
    
    # 需要从头forward... 这不太对。
    # 更好的方式: 直接用layer的forward
    handle.remove()
    
    # 改用直接调用单层forward
    layer = layers[layer_idx]
    
    with torch.no_grad():
        try:
            # 简化: 直接调用layer forward
            # 需要构造正确的输入格式
            h = h_input.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model] -> 不对, 需要完整seq
            
            # 这个方法不可行, 需要完整sequence的hidden states
            # 改回hook方式但注入扰动到embedding层
            
            return None  # 占位, 下面用另一种方式实现
        except Exception as e:
            return None


def collect_all_layer_outputs(model, inputs_embeds, toks, n_layers):
    """收集所有层的输出"""
    layers = get_layers(model)
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
    
    captured = {}
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
            if attention_mask is not None:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask)
            else:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception as e:
            print(f"  Forward error: {e}")
    
    for h in hooks:
        h.remove()
    
    return captured


def collect_layer_and_mlp_outputs(model, inputs_embeds, toks, sample_layers):
    """收集指定层的层输出和MLP输出"""
    layers = get_layers(model)
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
    
    captured = {}
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in sample_layers:
        if li < len(layers):
            # Transformer层输出
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
            # MLP输出
            if hasattr(layers[li], "mlp"):
                hooks.append(layers[li].mlp.register_forward_hook(make_hook(f"MLP{li}")))
    
    with torch.no_grad():
        try:
            attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
            if attention_mask is not None:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask)
            else:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception as e:
            print(f"  Forward error: {e}")
    
    for h in hooks:
        h.remove()
    
    return captured


def collect_attention_patterns(model, inputs_embeds, toks, sample_layers):
    """收集指定层的attention patterns"""
    layers = get_layers(model)
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
    
    # 用output_attentions=True的forward
    with torch.no_grad():
        attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "output_attentions": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        
        try:
            outputs = model(**kwargs)
        except Exception as e:
            print(f"  Forward error: {e}")
            return {}
    
    # outputs.attentions: tuple of [1, n_heads, seq_len, seq_len]
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        return {}
    
    result = {}
    for li in sample_layers:
        if li < len(outputs.attentions):
            attn = outputs.attentions[li]  # [1, n_heads, seq, seq]
            result[f"L{li}"] = attn[0].float().cpu().numpy()  # [n_heads, seq, seq]
    
    return result


def layer_norm_forward(hidden, ln):
    """简化的LayerNorm前向传播"""
    weight = ln.weight
    bias = ln.bias
    eps = ln.eps if hasattr(ln, 'eps') else 1e-5
    
    mean = hidden.mean(-1, keepdim=True)
    var = hidden.var(-1, keepdim=True, unbiased=False)
    hidden = (hidden - mean) / torch.sqrt(var + eps)
    if weight is not None:
        hidden = hidden * weight
    if bias is not None:
        hidden = hidden + bias
    return hidden


# ===== 修正的实验A实现(使用更可靠的方法) =====

def experiment_a_v2_cross_domain_jacobian(model, tokenizer, device, model_info):
    """
    实验A v2: 跨领域Jacobian一致性 (使用Finite Difference方法)
    
    改进: 不需要"forward_one_layer", 而是用两次完整forward的差来估计Jacobian-vector product
    
    方法:
    对每对句子(s1, s2), 在中间层li上:
    1. Forward s1, 获取li和li+1的hidden states: h1_li, h1_{li+1}
    2. Forward s1+εv, 获取li+1的hidden state: h1_{li+1}^+
    3. J(h1)*v ≈ [h1_{li+1}^+ - h1_{li+1}] / ε
    4. 同样对s2: J(h2)*v ≈ [h2_{li+1}^+ - h2_{li+1}] / ε
    5. cos(J(h1)*v, J(h2)*v)
    
    但注入扰动到中间层很困难(需要patching)。
    更简单的方法: 比较两个句子在li→li+1之间的Jacobian
    
    最简单可靠的方法:
    - 在embedding层注入扰动εv
    - Forward到li+1层
    - 比较在s1和s2作为基点时, 扰动传播到li+1的方向
    
    这测试的是"全局传播一致性"而非"局部Jacobian一致性"
    但对于区分"分段vs光滑"同样有效
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    sample_layers = get_sample_layers(n_layers, 6)
    
    eps = 2.0  # 使用Phase 143b验证过的大epsilon
    
    results = {}
    
    for s1, s2, category in CROSS_DOMAIN_PAIRS:
        pair_key = f"{category}"
        
        with torch.no_grad():
            toks1 = tokenizer(s1, return_tensors="pt").to(device)
            toks2 = tokenizer(s2, return_tensors="pt").to(device)
            
            embed_layer = model.get_input_embeddings()
            emb1_base = embed_layer(toks1.input_ids).detach().clone()
            emb2_base = embed_layer(toks2.input_ids).detach().clone()
        
        # 对多个随机方向, 比较扰动传播
        n_directions = 3
        direction_cosines = {li: [] for li in sample_layers}
        
        for di in range(n_directions):
            # 生成随机方向
            v = torch.randn(d_model, device=device, dtype=emb1_base.dtype)
            v = v / v.norm()
            
            with torch.no_grad():
                # s1 + εv at embedding
                emb1_pert = emb1_base.clone()
                emb1_pert[0, -1, :] += (eps * v).to(emb1_base.dtype)
                
                # s2 + εv at embedding
                emb2_pert = emb2_base.clone()
                emb2_pert[0, -1, :] += (eps * v).to(emb2_base.dtype)
                
                # Forward all four
                pos1 = torch.arange(emb1_base.shape[1], device=device).unsqueeze(0)
                pos2 = torch.arange(emb2_base.shape[1], device=device).unsqueeze(0)
                
                out1_base = collect_layer_outputs_at(model, emb1_base, pos1, toks1, sample_layers)
                out1_pert = collect_layer_outputs_at(model, emb1_pert, pos1, toks1, sample_layers)
                out2_base = collect_layer_outputs_at(model, emb2_base, pos2, toks2, sample_layers)
                out2_pert = collect_layer_outputs_at(model, emb2_pert, pos2, toks2, sample_layers)
            
            for li in sample_layers:
                key = f"L{li}"
                if key not in out1_base or key not in out1_pert:
                    continue
                if key not in out2_base or key not in out2_pert:
                    continue
                
                # 扰动传播信号
                delta1 = (out1_pert[key] - out1_base[key])[0, -1, :].float().numpy()
                delta2 = (out2_pert[key] - out2_base[key])[0, -1, :].float().numpy()
                
                n1 = np.linalg.norm(delta1)
                n2 = np.linalg.norm(delta2)
                
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_val = float(np.dot(delta1, delta2) / (n1 * n2))
                    direction_cosines[li].append(cos_val)
        
        # 聚合
        layer_means = {}
        for li in sample_layers:
            if direction_cosines[li]:
                layer_means[li] = float(np.mean(direction_cosines[li]))
        
        if layer_means:
            all_cos = list(layer_means.values())
            results[pair_key] = {
                "s1": s1,
                "s2": s2,
                "category": category,
                "mean_cos": float(np.mean(all_cos)),
                "std_cos": float(np.std(all_cos)),
                "min_cos": float(np.min(all_cos)),
                "max_cos": float(np.max(all_cos)),
                "layer_means": {str(k): v for k, v in layer_means.items()},
            }
    
    # 按类别聚合
    category_stats = defaultdict(list)
    for key, val in results.items():
        category_stats[val["category"]].append(val["mean_cos"])
    
    aggregated = {}
    for cat, vals in category_stats.items():
        aggregated[cat] = {
            "mean_cos": float(np.mean(vals)),
            "std_cos": float(np.std(vals)),
            "n_pairs": len(vals),
        }
    
    return {
        "by_pair": results,
        "by_category": aggregated,
    }


def collect_layer_outputs_at(model, inputs_embeds, position_ids, toks, sample_layers):
    """收集指定层的输出"""
    layers = get_layers(model)
    
    captured = {}
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in sample_layers:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            attention_mask = toks.attention_mask if hasattr(toks, 'attention_mask') else None
            if attention_mask is not None:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask)
            else:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception as e:
            pass
    
    for h in hooks:
        h.remove()
    
    return captured


# ===== 主函数 =====

def run_phase144(model_name: str):
    print(f"\n{'='*70}")
    print(f"Phase 144: 约束传播系统 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")
    
    all_results = {"model": model_name, "model_info": {
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
    }}
    
    # 实验A: 跨领域Jacobian一致性
    print(f"\n{'='*50}")
    print("实验A: 跨领域Jacobian一致性 (分段vs光滑)")
    print(f"{'='*50}")
    try:
        result_a = experiment_a_v2_cross_domain_jacobian(model, tokenizer, device, model_info)
        all_results["exp_a"] = result_a
        
        # 打印按类别的摘要
        print("\n按类别Jacobian一致性 (cos, ε=2.0):")
        for cat, stats in result_a["by_category"].items():
            print(f"  {cat:12s}: cos = {stats['mean_cos']:.4f} ± {stats['std_cos']:.4f} (n={stats['n_pairs']})")
    except Exception as e:
        print(f"  实验A失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_a_error"] = str(e)
    
    # 实验B: 约束违背动力学
    print(f"\n{'='*50}")
    print("实验B: 约束违背动力学")
    print(f"{'='*50}")
    try:
        result_b = experiment_b_constraint_violation(model, tokenizer, device, model_info)
        all_results["exp_b"] = result_b
        
        print("\n按类别约束违背信号:")
        for cat, stats in result_b["by_category"].items():
            print(f"  {cat:12s}: amp={stats['mean_amplification']:.2f}x, "
                  f"retention={stats['mean_retention']:.4f}, "
                  f"first={stats['mean_first_delta']:.4f}, last={stats['mean_last_delta']:.4f} "
                  f"(n={stats['n_pairs']})")
    except Exception as e:
        print(f"  实验B失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_b_error"] = str(e)
    
    # 实验C: MLP约束修正
    print(f"\n{'='*50}")
    print("实验C: MLP约束修正效应")
    print(f"{'='*50}")
    try:
        result_c = experiment_c_mlp_correction(model, tokenizer, device, model_info)
        all_results["exp_c"] = result_c
        
        print("\n按类别MLP修正:")
        for cat, stats in result_c["by_category"].items():
            print(f"  {cat:12s}: alignment={stats['mean_alignment']:.4f}, "
                  f"strength={stats['mean_correction_strength']:.4f} (n={stats['n_pairs']})")
    except Exception as e:
        print(f"  实验C失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_c_error"] = str(e)
    
    # 实验D: Attention聚类
    print(f"\n{'='*50}")
    print("实验D: Attention Head功能聚类")
    print(f"{'='*50}")
    try:
        result_d = experiment_d_attention_clustering(model, tokenizer, device, model_info)
        all_results["exp_d"] = result_d
        
        print(f"\n总heads: {result_d.get('n_heads_total', 'N/A')}")
        if "cluster_analysis" in result_d:
            for cid, info in result_d["cluster_analysis"].items():
                print(f"  Cluster {cid}: {info['n_heads']} heads, "
                      f"layer {info['min_layer']}-{info['max_layer']} "
                      f"(mean={info['mean_layer']:.1f})")
    except Exception as e:
        print(f"  实验D失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_d_error"] = str(e)
    
    # 保存结果 (用自定义encoder处理numpy float32/int64)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = TEMP_DIR / f"phase144_{model_name}_constraint_20260512_{timestamp[11:]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n结果保存到: {out_path}")
    
    # 释放模型
    release_model(model)
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase144(model_name)
