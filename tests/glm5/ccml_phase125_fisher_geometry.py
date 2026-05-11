"""
Phase 125: Fisher Geometry — Probability Modulation Theory
==========================================================

用户理论批评的核心:
1. Fisher ≠ 语义信息 — Fisher测量的是"预测敏感度", 不是语义本身
2. 真正的数学对象是条件概率流形 p(y|x), 不是激活h
3. Transformer本质是"条件概率调制系统": 残差流=载波, 微扰=信息
4. "能量几何 ≠ 信息几何" — 需要直接验证

关键可测试预测:
A. Fisher主方向 vs PCA主方向近乎正交 → 能量几何≠信息几何
B. W_U行空间与"信号子空间"(低能高Fisher方向)强对齐 → 语言读出机制
C. 删除低能高Fisher方向导致KL暴涨 → 信息确实藏在低能扰动中

实验设计:
- Exp 1: Fisher谱估计 + Fisher主方向提取
  → 对每个方向v, Fisher(v) = E[(∂log p(y*|h)/∂(v·h))²]
  → Fisher矩阵的近似本征分解

- Exp 2: Fisher主方向 vs PCA主方向的对齐度
  → cos(v_PCA_i, v_Fisher_j) 矩阵
  → 如果近正交: 彻底证明能量≠信息

- Exp 3: W_U行空间与信号子空间的对齐
  → W_U的SVD分解
  → 低能PCA方向的W_U投影
  → 如果W_U对齐低能高Fisher方向: 发现语言读出核心机制

- Exp 4: 定向消融实验
  → 消融低能高Fisher方向 vs 消融低能低Fisher方向
  → 如果前者KL暴涨而后者影响小: 信息确实在低能高Fisher方向

- Exp 5: 不同ε值的稳定性检验
  → ε = 0.001, 0.005, 0.01, 0.05, 0.1
  → 验证Fisher估计的稳定性
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

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, get_W_U, release_model, MODEL_CONFIGS
)


# ============================================================
# 配置
# ============================================================
PROMPTS = [
    "The scientist discovered a new type of crystal structure that",
    "In the field of quantum mechanics, the principle of",
    "The mathematical proof relies on the concept of",
    "The historical event changed the course of civilization by",
    "The artistic movement was characterized by its emphasis on",
    "The economic theory suggests that market forces",
    "The biological process involves the transformation of",
    "The philosophical argument centers around the nature of",
    "The technological innovation revolutionized the industry through",
    "The environmental crisis requires immediate action because",
    "The literary work explores themes of identity and",
    "The musical composition combines elements of harmony and",
    "The social phenomenon can be explained through the lens of",
    "The engineering solution addresses the challenge of",
    "The medical breakthrough offers new hope for patients with",
    "The educational reform aims to improve student outcomes by",
    "The political debate centers on the question of whether",
    "The psychological research reveals that human behavior",
    "The legal framework establishes guidelines for",
    "The architectural design integrates sustainable features such as",
    "The chemical reaction produces a compound that",
    "The astronomical observation suggests that the universe",
    "The linguistic analysis reveals patterns in how",
    "The computational algorithm efficiently solves the problem by",
    "The cultural tradition reflects values of community and",
    "The geographic feature was formed through the process of",
    "The nutritional science indicates that dietary choices",
    "The athletic performance depends on factors including",
    "The diplomatic negotiation resulted in an agreement that",
    "The ethical dilemma raises questions about the balance between",
]

MAX_SEQ_LEN = 40
N_PROMPTS = 25  # 加大数据量
N_FISHER_DIRS = 100  # Fisher谱估计: 采样100个方向
N_TOP_FISHER = 30  # 提取top-30 Fisher方向
EPSILON_DEFAULT = 0.01
EPSILON_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]  # Exp 5: 稳定性检验
K_ABLATION = 50  # 消融实验: 消融50个方向


# ============================================================
# 工具函数: 收集激活 + PCA
# ============================================================
def collect_activations_and_pca(model, tokenizer, device, model_info, prompts, target_layers):
    """收集各层隐藏状态并计算PCA"""
    print(f"\n[数据收集] 收集激活并计算PCA...")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    
    layer_activations = {l: [] for l in target_layers}
    
    for pidx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).input_ids
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        for l in target_layers:
            hs = outputs.hidden_states[l + 1].detach().cpu().float().numpy()
            layer_activations[l].append(hs[0])
        
        if (pidx + 1) % 10 == 0:
            print(f"  收集: {pidx+1}/{len(prompts)} prompts")
    
    # PCA
    pca_results = {}
    for l in target_layers:
        all_acts = np.concatenate(layer_activations[l], axis=0)
        mean = all_acts.mean(axis=0)
        centered = all_acts - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        
        pca_results[l] = {
            'components': Vt,  # (d_model, d_model)
            'eigenvalues': S,
            'mean': mean,
            'eff_rank_95': int(np.searchsorted(cumvar, 0.95)) + 1,
            'eff_rank_99': int(np.searchsorted(cumvar, 0.99)) + 1,
            'top10_variance_ratio': float(np.sum(S[:10]**2) / total_var),
            'top50_variance_ratio': float(np.sum(S[:50]**2) / total_var),
        }
        print(f"  Layer {l}: eff_rank(95%)={pca_results[l]['eff_rank_95']}, "
              f"top10方差比={pca_results[l]['top10_variance_ratio']:.4f}")
    
    print(f"  数据收集+PCA耗时: {time.time()-t0:.1f}s")
    return pca_results


# ============================================================
# Exp 1: Fisher谱估计 + Fisher主方向提取
# ============================================================
def estimate_fisher_spectrum(model, tokenizer, device, model_info,
                             prompts, pca_results, target_layers,
                             epsilon=EPSILON_DEFAULT):
    """
    Fisher谱估计: 在大量随机方向上测量Fisher敏感度
    
    方法: 对方向v添加±ε扰动, 测量|∂log p(y*|h)/∂(v·h)|
    Fisher(v) ≈ E_y [|∂log p(y*|h)/∂(v·h)|²]
    
    然后用这些方向+敏感度构造近似Fisher矩阵,
    提取Fisher主方向(高Fisher敏感度的方向).
    """
    print(f"\n[Exp 1] Fisher谱估计 (ε={epsilon})")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}
    
    for l in target_layers:
        print(f"  Layer {l}...")
        pca_comp = pca_results[l]['components']
        
        # 生成探测方向: 
        # - 前50个PCA方向
        # - 50个随机方向
        n_pca_dirs = min(50, d_model)
        n_rand_dirs = N_FISHER_DIRS - n_pca_dirs
        
        pca_dirs = pca_comp[:n_pca_dirs]  # (n_pca_dirs, d_model)
        
        # 随机正交方向
        rand_dirs = np.random.randn(n_rand_dirs, d_model)
        rand_dirs, _ = np.linalg.qr(rand_dirs.T)
        rand_dirs = rand_dirs.T[:n_rand_dirs]
        for i in range(n_rand_dirs):
            rand_dirs[i] /= np.linalg.norm(rand_dirs[i])
        
        all_dirs = np.concatenate([pca_dirs, rand_dirs], axis=0)
        n_dirs = len(all_dirs)
        
        # 测量每个方向的Fisher敏感度
        fisher_values = np.zeros(n_dirs)
        
        for pidx, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                                   max_length=MAX_SEQ_LEN).input_ids
            input_ids = input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
                base_logits = base_outputs.logits
            
            # 批量测量: 对每个方向, 同时测量+ε和-ε
            for d_idx in range(n_dirs):
                direction = all_dirs[d_idx]
                perturbation = torch.tensor(
                    direction, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(0) * epsilon
                
                def make_hook(sign):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            return (h + sign * perturbation.to(h.dtype),) + output[1:]
                        return output + sign * perturbation.to(output.dtype)
                    return hook_fn
                
                # +ε
                handle = layers[l].register_forward_hook(make_hook(+1))
                with torch.no_grad():
                    plus_logits = model(input_ids).logits
                handle.remove()
                
                # -ε
                handle = layers[l].register_forward_hook(make_hook(-1))
                with torch.no_grad():
                    minus_logits = model(input_ids).logits
                handle.remove()
                
                # 计算Fisher敏感度
                for pos in range(seq_len - 1):
                    y_star = input_ids[0, pos + 1].item()
                    log_p_plus = F.log_softmax(plus_logits[0, pos].float(), dim=-1)[y_star].item()
                    log_p_minus = F.log_softmax(minus_logits[0, pos].float(), dim=-1)[y_star].item()
                    
                    sensitivity = (log_p_plus - log_p_minus) / (2 * epsilon)
                    fisher_values[d_idx] += sensitivity ** 2
            
            if (pidx + 1) % 5 == 0:
                print(f"    Prompt {pidx+1}/{len(prompts)}")
        
        # 归一化: Fisher值 = 平均每个token
        n_measurements = len(prompts) * (MAX_SEQ_LEN - 1)
        fisher_values /= max(n_measurements, 1)
        
        # 按Fisher值排序方向
        sorted_indices = np.argsort(fisher_values)[::-1]
        
        # Fisher主方向: Fisher敏感度最高的方向
        top_fisher_dirs = all_dirs[sorted_indices[:N_TOP_FISHER]]
        top_fisher_values = fisher_values[sorted_indices[:N_TOP_FISHER]]
        
        # 分组统计
        pca_fisher = fisher_values[:n_pca_dirs]
        rand_fisher = fisher_values[n_pca_dirs:]
        
        results[l] = {
            'fisher_values': fisher_values.tolist(),
            'sorted_indices': sorted_indices.tolist(),
            'top_fisher_dirs': top_fisher_dirs.tolist(),
            'top_fisher_values': top_fisher_values.tolist(),
            'pca_fisher_mean': float(np.mean(pca_fisher)),
            'pca_fisher_max': float(np.max(pca_fisher)),
            'rand_fisher_mean': float(np.mean(rand_fisher)),
            'rand_fisher_max': float(np.max(rand_fisher)),
            'pca_fisher_top5_mean': float(np.mean(pca_fisher[:5])),
            'fisher_concentration_top5_vs_all': float(
                np.mean(top_fisher_values[:5]) / max(np.mean(fisher_values), 1e-10)
            ),
        }
        
        print(f"    PCA Fisher: mean={results[l]['pca_fisher_mean']:.4f}, max={results[l]['pca_fisher_max']:.4f}")
        print(f"    Random Fisher: mean={results[l]['rand_fisher_mean']:.4f}, max={results[l]['rand_fisher_max']:.4f}")
        print(f"    Top-5 Fisher dirs的均值: {np.mean(top_fisher_values[:5]):.4f}")
        print(f"    Fisher集中度(top5/all): {results[l]['fisher_concentration_top5_vs_all']:.3f}")
    
    print(f"  Fisher谱估计耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 2: Fisher主方向 vs PCA主方向的对齐度
# ============================================================
def fisher_vs_pca_alignment(pca_results, fisher_results, target_layers, d_model):
    """
    核心实验: Fisher主方向与PCA主方向的对齐度
    
    如果cos(v_PCA_i, v_Fisher_j) ≈ 0 对所有i,j
    → 能量几何与信息几何完全分离
    → 彻底证明Phase 124的"载波-信号分离"
    
    如果cos(v_PCA_1, v_Fisher_1) > 0.5
    → PCA主方向也携带信息, 但不是全部
    """
    print(f"\n[Exp 2] Fisher vs PCA主方向对齐度")
    t0 = time.time()
    
    results = {}
    
    for l in target_layers:
        # 兼容int和string键
        l_key = l if l in pca_results else str(l)
        if l_key not in pca_results:
            continue
        fl_key = l if l in fisher_results else str(l)
        if fl_key not in fisher_results:
            continue
        
        pca_comp = pca_results[l_key]['components']
        fisher_dirs = np.array(fisher_results[fl_key]['top_fisher_dirs'])
        
        # 计算 cos(v_PCA_i, v_Fisher_j) 矩阵
        # PCA取前30, Fisher取前30
        n_check = min(30, len(fisher_dirs))
        cos_matrix = np.zeros((n_check, n_check))
        
        for i in range(n_check):
            for j in range(n_check):
                cos_val = abs(np.dot(pca_comp[i], fisher_dirs[j]))
                cos_matrix[i, j] = cos_val
        
        # 关键指标
        # 1. PCA-top1与Fisher-top1的余弦相似度
        cos_pca1_fisher1 = abs(np.dot(pca_comp[0], fisher_dirs[0]))
        
        # 2. PCA-top1与所有Fisher方向的最大余弦相似度
        max_cos_pca1_fisher = np.max([abs(np.dot(pca_comp[0], fisher_dirs[j])) 
                                       for j in range(n_check)])
        
        # 3. Fisher-top1与所有PCA方向的最大余弦相似度
        max_cos_fisher1_pca = np.max([abs(np.dot(pca_comp[i], fisher_dirs[0])) 
                                       for i in range(n_check)])
        
        # 4. 整体对齐度: cos矩阵的均值和最大值
        mean_cos = float(np.mean(cos_matrix))
        max_cos = float(np.max(cos_matrix))
        
        # 5. 每个PCA方向的"最大Fisher对齐"
        pca_max_fisher_align = [float(np.max(cos_matrix[i, :])) for i in range(n_check)]
        # 6. 每个Fisher方向的"最大PCA对齐"
        fisher_max_pca_align = [float(np.max(cos_matrix[:, j])) for j in range(n_check)]
        
        results[l] = {
            'cos_pca1_fisher1': float(cos_pca1_fisher1),
            'max_cos_pca1_fisher': float(max_cos_pca1_fisher),
            'max_cos_fisher1_pca': float(max_cos_fisher1_pca),
            'mean_cos_matrix': mean_cos,
            'max_cos_matrix': max_cos,
            'pca_max_fisher_align_top5': pca_max_fisher_align[:5],
            'fisher_max_pca_align_top5': fisher_max_pca_align[:5],
            'cos_matrix_diag_mean': float(np.mean([cos_matrix[i, i] for i in range(n_check)])),
        }
        
        print(f"  Layer {l}:")
        print(f"    cos(PCA-1, Fisher-1) = {cos_pca1_fisher1:.4f}")
        print(f"    PCA-1的max Fisher对齐 = {max_cos_pca1_fisher:.4f}")
        print(f"    Fisher-1的max PCA对齐 = {max_cos_fisher1_pca:.4f}")
        print(f"    cos矩阵均值 = {mean_cos:.4f}, max = {max_cos:.4f}")
        print(f"    对角线均值 = {results[l]['cos_matrix_diag_mean']:.4f}")
        
        # 判断
        if cos_pca1_fisher1 < 0.3:
            print(f"    >>> PCA主方向与Fisher主方向近正交! 能量≠信息!")
        elif cos_pca1_fisher1 > 0.7:
            print(f"    >>> PCA主方向与Fisher主方向强对齐! 能量≈信息!")
        else:
            print(f"    >>> 中等对齐, 存在部分分离")
    
    print(f"  对齐度分析耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 3: W_U行空间与信号子空间的对齐
# ============================================================
def w_u_alignment_analysis(model, model_name, pca_results, fisher_results, target_layers, d_model):
    """
    核心实验: W_U行空间与"信号子空间"的对齐
    
    假说: 语言能力的数学结构 = W_U对信号子空间的选择性放大
    
    如果: W_U的行空间(即lm_head投影空间)与低能高Fisher方向强对齐
    则: 模型通过W_U读取信号子空间中的信息, 载波被自然抑制
    
    方法:
    1. W_U的SVD分解 → W_U = U Σ V^T
    2. 计算 PCA方向 v_i 在W_U行空间中的投影比
    3. 计算 Fisher方向 在W_U行空间中的投影比
    4. 比较低能PCA方向 vs 高Fisher方向的W_U投影
    """
    print(f"\n[Exp 3] W_U行空间与信号子空间的对齐")
    t0 = time.time()
    
    # W_U的SVD
    W_U = get_W_U(model, model_name)  # (vocab_size, d_model)
    print(f"  W_U shape: {W_U.shape}")
    
    # 对W_U^T做SVD: W_U^T = U S Vt, U的列是W_U行空间的基
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 2)
    k_svd = max(k_svd, 10)
    W_U_T = W_U.T.astype(np.float32)  # (d_model, vocab)
    U_wu, s_wu, Vt_wu = svds(W_U_T, k=k_svd)
    U_wu = np.asarray(U_wu, dtype=np.float64)  # (d_model, k_svd) — W_U行空间基
    
    print(f"  W_U SVD: k={k_svd}, top-5奇异值={s_wu[-5:][::-1].tolist()}")
    
    results = {}
    
    for l in target_layers:
        # 兼容int和string键
        l_key = l if l in pca_results else str(l)
        if l_key not in pca_results:
            continue
        fl_key = l if l in fisher_results else str(l)
        if fl_key not in fisher_results:
            continue
        
        pca_comp = pca_results[l_key]['components']
        pca_eigenvals = pca_results[l_key]['eigenvalues']
        fisher_dirs = np.array(fisher_results[fl_key]['top_fisher_dirs'])
        fisher_vals = np.array(fisher_results[fl_key]['top_fisher_values'])
        
        # === 3a. PCA方向在W_U行空间中的投影比 ===
        n_pca_check = min(200, d_model)
        pca_wu_proj = np.zeros(n_pca_check)
        for i in range(n_pca_check):
            v = pca_comp[i]
            # 投影到W_U行空间: proj = U_wu @ (U_wu^T @ v)
            proj = U_wu @ (U_wu.T @ v)
            pca_wu_proj[i] = np.linalg.norm(proj) ** 2  # ||proj||² / ||v||² (v已归一化)
        
        # === 3b. Fisher方向在W_U行空间中的投影比 ===
        fisher_wu_proj = np.zeros(len(fisher_dirs))
        for i in range(len(fisher_dirs)):
            v = fisher_dirs[i]
            proj = U_wu @ (U_wu.T @ v)
            fisher_wu_proj[i] = np.linalg.norm(proj) ** 2
        
        # === 3c. 低能PCA方向的W_U投影 vs 高能PCA方向 ===
        # 将PCA方向按能量分组
        # 高能组: 前10个PCA方向
        # 低能组: 第100-200个PCA方向
        high_energy_wu = float(np.mean(pca_wu_proj[:10]))
        low_energy_wu = float(np.mean(pca_wu_proj[100:200])) if n_pca_check >= 200 else float(np.mean(pca_wu_proj[50:]))
        
        # === 3d. Fisher-top方向 vs PCA-top方向的W_U投影 ===
        fisher_top5_wu = float(np.mean(fisher_wu_proj[:5]))
        pca_top5_wu = float(np.mean(pca_wu_proj[:5]))
        
        # === 3e. 信号子空间定义: 低能+高Fisher的方向 ===
        # 找出PCA能量排名>50但Fisher值在top-30的方向
        low_energy_high_fisher_dirs = []
        for idx in fisher_results[fl_key]['sorted_indices'][:N_TOP_FISHER]:
            if idx >= 50:  # PCA排名50以后
                direction = pca_comp[idx] if idx < n_pca_check else None
                if direction is not None:
                    proj = U_wu @ (U_wu.T @ direction)
                    low_energy_high_fisher_dirs.append(np.linalg.norm(proj) ** 2)
        
        signal_subspace_wu = float(np.mean(low_energy_high_fisher_dirs)) if low_energy_high_fisher_dirs else 0
        
        results[l] = {
            'pca_wu_proj_top10_mean': float(np.mean(pca_wu_proj[:10])),
            'pca_wu_proj_top50_mean': float(np.mean(pca_wu_proj[:50])),
            'pca_wu_proj_top100_mean': float(np.mean(pca_wu_proj[:100])),
            'pca_wu_proj_bottom100_mean': low_energy_wu,
            'high_energy_wu': high_energy_wu,
            'low_energy_wu': low_energy_wu,
            'fisher_top5_wu': fisher_top5_wu,
            'pca_top5_wu': pca_top5_wu,
            'signal_subspace_wu': signal_subspace_wu,
            'pca_wu_proj_all': pca_wu_proj.tolist(),
            'fisher_wu_proj_all': fisher_wu_proj.tolist(),
        }
        
        print(f"  Layer {l}:")
        print(f"    PCA top-10 W_U投影: {results[l]['pca_wu_proj_top10_mean']:.4f}")
        print(f"    PCA top-50 W_U投影: {results[l]['pca_wu_proj_top50_mean']:.4f}")
        print(f"    PCA top-100 W_U投影: {results[l]['pca_wu_proj_top100_mean']:.4f}")
        print(f"    PCA bottom W_U投影: {low_energy_wu:.4f}")
        print(f"    Fisher top-5 W_U投影: {fisher_top5_wu:.4f}")
        print(f"    PCA top-5 W_U投影: {pca_top5_wu:.4f}")
        print(f"    信号子空间(低能高Fisher) W_U投影: {signal_subspace_wu:.4f}")
        
        # 判断
        if fisher_top5_wu > pca_top5_wu * 1.5:
            print(f"    >>> Fisher方向比PCA方向更对齐W_U! 信号被选择性读出!")
        elif low_energy_wu > high_energy_wu * 1.5:
            print(f"    >>> 低能方向比高能方向更对齐W_U! 载波被抑制!")
        else:
            print(f"    >>> W_U对齐模式需要进一步分析")
    
    print(f"  W_U对齐分析耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 4: 定向消融实验 — 验证"信息藏在低能扰动里"
# ============================================================
def directional_ablation_experiment(model, tokenizer, device, model_info,
                                    prompts, pca_results, fisher_results, 
                                    target_layers, epsilon=EPSILON_DEFAULT):
    """
    核心实验: 消融不同类型的方向, 比较对预测的影响
    
    三种消融:
    A. 消融高能低Fisher方向 (载波方向): 预测应该不受影响
    B. 消融低能高Fisher方向 (信号方向): 预测应该严重受损
    C. 消融随机方向: 预测应该轻微受损
    
    如果B >> C > A → 信息确实在低能高Fisher方向!
    """
    print(f"\n[Exp 4] 定向消融实验")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}
    
    for l in target_layers:
        # 兼容int和string键
        l_key = l if l in pca_results else str(l)
        if l_key not in pca_results:
            continue
        fl_key = l if l in fisher_results else str(l)
        if fl_key not in fisher_results:
            continue
        
        print(f"  Layer {l}...")
        pca_comp = pca_results[l_key]['components']
        fisher_vals = np.array(fisher_results[fl_key]['fisher_values'])
        n_dirs = len(fisher_vals)
        
        # 分类方向
        # A: 高能低Fisher = PCA排名前K, 但Fisher值低于中位数
        # B: 低能高Fisher = PCA排名后K, 但Fisher值高于中位数
        # C: 随机K个方向
        
        fisher_median = np.median(fisher_vals[:min(50, n_dirs)])
        
        # 在PCA前50方向中找低Fisher的
        pca_low_fisher_indices = [i for i in range(min(50, n_dirs)) 
                                   if fisher_vals[i] < fisher_median][:K_ABLATION]
        # 在PCA 50-200方向中找高Fisher的  
        pca_high_fisher_indices = [i for i in range(50, min(200, n_dirs)) 
                                    if fisher_vals[i] > fisher_median][:K_ABLATION]
        # 如果不够, 扩大范围
        if len(pca_high_fisher_indices) < K_ABLATION:
            pca_high_fisher_indices = [i for i in range(50, min(n_dirs, 500)) 
                                        if fisher_vals[i] > fisher_median * 0.8][:K_ABLATION]
        
        # 随机方向
        random_indices = np.random.choice(range(min(200, n_dirs)), size=min(K_ABLATION, 200), replace=False).tolist()
        
        ablation_groups = {
            'high_energy_low_fisher': pca_low_fisher_indices,
            'low_energy_high_fisher': pca_high_fisher_indices,
            'random': random_indices,
        }
        
        for group_name, indices in ablation_groups.items():
            if len(indices) == 0:
                print(f"    {group_name}: 无可用方向, 跳过")
                continue
            
            kl_divs = []
            top1_matches = []
            cos_sims = []
            
            for pidx, prompt in enumerate(prompts):
                input_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                                       max_length=MAX_SEQ_LEN).input_ids
                input_ids = input_ids.to(device)
                
                # 基线
                with torch.no_grad():
                    base_outputs = model(input_ids, output_hidden_states=True)
                    base_logits = base_outputs.logits
                
                base_top1 = base_logits[0, -1].argmax().item()
                base_log_probs = F.log_softmax(base_logits[0, -1].float(), dim=-1)
                
                # 消融: 将h投影到这些方向的补空间
                # 即 h_ablated = h - sum_i (v_i · h) v_i
                ablation_dirs = torch.tensor(
                    pca_comp[indices], dtype=torch.float32, device=device
                )  # (K, d_model)
                
                def make_ablation_hook(dirs_matrix):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            h_float = h.float()
                            # 移除在这些方向上的投影
                            # (v_i · h) v_i = dirs^T @ h^T → (K, seq_len) → dirs @ (K, seq_len)^T
                            proj_coeffs = torch.matmul(h_float, dirs_matrix.T)  # (1, seq_len, K)
                            projections = torch.matmul(proj_coeffs, dirs_matrix)  # (1, seq_len, d_model)
                            h_ablated = h_float - projections
                            return (h_ablated.to(h.dtype),) + output[1:]
                        return output
                    return hook_fn
                
                handle = layers[l].register_forward_hook(make_ablation_hook(ablation_dirs))
                with torch.no_grad():
                    ablated_outputs = model(input_ids)
                    ablated_logits = ablated_outputs.logits
                handle.remove()
                
                ablated_log_probs = F.log_softmax(ablated_logits[0, -1].float(), dim=-1)
                
                kl_div = F.kl_div(ablated_log_probs, F.softmax(base_log_probs, dim=-1),
                                  reduction='sum').item()
                ablated_top1 = ablated_logits[0, -1].argmax().item()
                top1_match = 1 if ablated_top1 == base_top1 else 0
                cos_sim = F.cosine_similarity(
                    base_logits[0, -1].float().unsqueeze(0),
                    ablated_logits[0, -1].float().unsqueeze(0)
                ).item()
                
                kl_divs.append(kl_div)
                top1_matches.append(top1_match)
                cos_sims.append(cos_sim)
            
            results.setdefault(l, {})[group_name] = {
                'n_dirs_ablated': len(indices),
                'kl_div_mean': float(np.mean(kl_divs)),
                'kl_div_std': float(np.std(kl_divs)),
                'top1_match_rate': float(np.mean(top1_matches)),
                'cosine_sim_mean': float(np.mean(cos_sims)),
                'cosine_sim_std': float(np.std(cos_sims)),
            }
            
            print(f"    {group_name} (n={len(indices)}): "
                  f"KL={np.mean(kl_divs):.3f}±{np.std(kl_divs):.3f}, "
                  f"top1={np.mean(top1_matches):.2f}, "
                  f"cos={np.mean(cos_sims):.4f}")
    
    print(f"  消融实验耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 5: 不同ε值的Fisher稳定性
# ============================================================
def epsilon_stability_test(model, tokenizer, device, model_info,
                           prompts, pca_results, target_layers):
    """
    验证Fisher估计的ε-稳定性
    
    如果不同ε下Fisher排名一致 → 估计可靠
    如果ε太小时Fisher值发散 → 数值噪声
    如果ε太大时Fisher排名变化 → 非线性效应
    """
    print(f"\n[Exp 5] ε稳定性检验")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}
    
    # 只测试一个层的少数方向
    test_layer = target_layers[min(1, len(target_layers)-1)]
    n_test_dirs = 20
    
    pca_comp = pca_results[test_layer]['components']
    test_dirs = pca_comp[:n_test_dirs]
    
    for eps in EPSILON_VALUES:
        print(f"  ε={eps}...")
        fisher_vals = np.zeros(n_test_dirs)
        
        for pidx, prompt in enumerate(prompts[:10]):  # 少量prompt
            input_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                                   max_length=MAX_SEQ_LEN).input_ids
            input_ids = input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
                base_logits = base_outputs.logits
            
            for d_idx in range(n_test_dirs):
                direction = test_dirs[d_idx]
                perturbation = torch.tensor(
                    direction, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(0) * eps
                
                def make_hook(sign):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            return (h + sign * perturbation.to(h.dtype),) + output[1:]
                        return output + sign * perturbation.to(output.dtype)
                    return hook_fn
                
                handle = layers[test_layer].register_forward_hook(make_hook(+1))
                with torch.no_grad():
                    plus_logits = model(input_ids).logits
                handle.remove()
                
                handle = layers[test_layer].register_forward_hook(make_hook(-1))
                with torch.no_grad():
                    minus_logits = model(input_ids).logits
                handle.remove()
                
                for pos in range(min(seq_len - 1, 10)):  # 限制位置数
                    y_star = input_ids[0, pos + 1].item()
                    log_p_plus = F.log_softmax(plus_logits[0, pos].float(), dim=-1)[y_star].item()
                    log_p_minus = F.log_softmax(minus_logits[0, pos].float(), dim=-1)[y_star].item()
                    
                    sensitivity = (log_p_plus - log_p_minus) / (2 * eps)
                    fisher_vals[d_idx] += sensitivity ** 2
        
        fisher_vals /= max(len(prompts[:10]) * 10, 1)
        
        results[eps] = {
            'fisher_values': fisher_vals.tolist(),
            'fisher_mean': float(np.mean(fisher_vals)),
            'fisher_max': float(np.max(fisher_vals)),
            'fisher_ranking': np.argsort(fisher_vals)[::-1].tolist(),
        }
        
        print(f"    mean={np.mean(fisher_vals):.6f}, max={np.max(fisher_vals):.6f}, "
              f"排名前5={np.argsort(fisher_vals)[::-1][:5].tolist()}")
    
    # 检查排名稳定性
    eps_list = sorted(results.keys())
    if len(eps_list) >= 2:
        ref_ranking = np.array(results[eps_list[2]]['fisher_ranking'] if len(eps_list) > 2 
                               else results[eps_list[0]]['fisher_ranking'])
        for eps in eps_list:
            ranking = np.array(results[eps]['fisher_ranking'])
            # Kendall tau相关
            from scipy.stats import kendalltau
            tau, p = kendalltau(ref_ranking, ranking)
            results[eps]['kendall_tau_vs_ref'] = float(tau)
            results[eps]['kendall_p_vs_ref'] = float(p)
            print(f"    ε={eps}: Kendall τ vs ref = {tau:.3f} (p={p:.4f})")
    
    print(f"  ε稳定性检验耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# 主流程
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print("=" * 70)
    print(f"Phase 125: Fisher Geometry — Probability Modulation Theory")
    print(f"Model: {model_name}")
    print("=" * 70)
    
    # ===== 加载模型 =====
    print("\n" + "=" * 50)
    print("加载模型")
    print("=" * 50)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"  class={model_info.model_class}, n_layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}, vocab={model_info.vocab_size}")
    
    n_layers = model_info.n_layers
    target_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    print(f"  目标层: {target_layers}")
    
    test_prompts = PROMPTS[:N_PROMPTS]
    
    # ===== 数据收集 + PCA =====
    pca_results = collect_activations_and_pca(
        model, tokenizer, device, model_info, test_prompts, target_layers
    )
    
    # ===== Exp 1: Fisher谱估计 =====
    print("\n" + "=" * 50)
    print("Exp 1: Fisher谱估计 + Fisher主方向提取")
    print("=" * 50)
    
    fisher_results = estimate_fisher_spectrum(
        model, tokenizer, device, model_info,
        test_prompts, pca_results, target_layers
    )
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open(os.path.join(out_dir, f'phase125_exp1_{model_name}_fisher_spectrum.json'), 'w') as f:
        json.dump(fisher_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 1 结果已保存")
    
    # ===== Exp 2: Fisher vs PCA对齐 =====
    print("\n" + "=" * 50)
    print("Exp 2: Fisher vs PCA主方向对齐度")
    print("=" * 50)
    
    alignment_results = fisher_vs_pca_alignment(
        pca_results, fisher_results, target_layers, model_info.d_model
    )
    
    with open(os.path.join(out_dir, f'phase125_exp2_{model_name}_alignment.json'), 'w') as f:
        json.dump(alignment_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 2 结果已保存")
    
    # ===== Exp 3: W_U对齐 =====
    print("\n" + "=" * 50)
    print("Exp 3: W_U行空间与信号子空间的对齐")
    print("=" * 50)
    
    wu_results = w_u_alignment_analysis(
        model, model_name, pca_results, fisher_results, target_layers, model_info.d_model
    )
    
    with open(os.path.join(out_dir, f'phase125_exp3_{model_name}_wu_alignment.json'), 'w') as f:
        json.dump(wu_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 3 结果已保存")
    
    # ===== Exp 4: 定向消融 =====
    print("\n" + "=" * 50)
    print("Exp 4: 定向消融实验")
    print("=" * 50)
    
    ablation_results = directional_ablation_experiment(
        model, tokenizer, device, model_info,
        test_prompts, pca_results, fisher_results, target_layers
    )
    
    with open(os.path.join(out_dir, f'phase125_exp4_{model_name}_ablation.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 4 结果已保存")
    
    # ===== Exp 5: ε稳定性 =====
    print("\n" + "=" * 50)
    print("Exp 5: ε稳定性检验")
    print("=" * 50)
    
    stability_results = epsilon_stability_test(
        model, tokenizer, device, model_info,
        test_prompts, pca_results, target_layers
    )
    
    with open(os.path.join(out_dir, f'phase125_exp5_{model_name}_epsilon_stability.json'), 'w') as f:
        json.dump(stability_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 5 结果已保存")
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("Phase 125 汇总")
    print("=" * 70)
    
    print("\n[结论1] Fisher vs PCA对齐度:")
    for l in target_layers:
        if str(l) in alignment_results:
            r = alignment_results[str(l)]
            print(f"  Layer {l}: cos(PCA-1, Fisher-1)={r['cos_pca1_fisher1']:.4f}, "
                  f"cos矩阵均值={r['mean_cos_matrix']:.4f}")
    
    print("\n[结论2] W_U对齐:")
    for l in target_layers:
        if str(l) in wu_results:
            r = wu_results[str(l)]
            print(f"  Layer {l}: PCA-top10 W_U投影={r['pca_wu_proj_top10_mean']:.4f}, "
                  f"PCA-bottom W_U投影={r['pca_wu_proj_bottom100_mean']:.4f}, "
                  f"Fisher-top5 W_U投影={r['fisher_top5_wu']:.4f}")
    
    print("\n[结论3] 定向消融:")
    for l in target_layers:
        if str(l) in ablation_results:
            r = ablation_results[str(l)]
            for group in ['high_energy_low_fisher', 'low_energy_high_fisher', 'random']:
                if group in r:
                    print(f"  Layer {l} {group}: KL={r[group]['kl_div_mean']:.3f}, "
                          f"cos={r[group]['cosine_sim_mean']:.4f}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPhase 125 完成!")


if __name__ == "__main__":
    main()
