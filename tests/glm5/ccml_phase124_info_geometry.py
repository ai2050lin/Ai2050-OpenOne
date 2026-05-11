"""
Phase 124: Information Geometry & Mortality Experiments
======================================================

用户核心批评:
1. eff_rank≈50可能只是k_probes=50的天花板, 不是真实有效秩
2. 87-89°交叉角度可能只是高维空间随机子空间的期望角度
3. 我们知道"什么方向被输运"但不知道"被输运的是什么"
4. 缺乏不变量(invariant), 理论仍停留在统计现象学阶段

核心假说(Representation Compression Flow):
- 每层去除对next-token预测无关的自由度
- Fisher信息揭示哪些方向影响预测; PCA揭示哪些方向有能量
- 如果Fisher ≠ PCA, 则信息几何 ≠ 激活几何

实验设计:
- Exp 0: 随机基线 — 主角的Monte Carlo分布 (无需模型)
- Exp 1: Fisher敏感性谱 — 沿PCA方向扰动, 测量预测变化
- Exp 2: 预测信息损失 vs 子空间维度 — 投影+前向, 测KL散度
- Exp 3: eff_rank饱和测试 — k=50,100,200,400探针

关键问题:
- Fisher谱是否比PCA谱衰减更快? → 信息比能量更集中
- 多少维PCA子空间能保留预测信息? → 最小充分统计量
- eff_rank是否饱和? → 低维输运通道是否存在
- 87-89°是否只是随机基线? → Phase 123结论是否成立
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
from scipy.linalg import subspace_angles

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
N_PROMPTS = 20  # 使用20个prompt (加大数据量)
N_PCA_DIRS = 30  # Fisher敏感性测试: 前30个PCA方向
N_RANDOM_DIRS = 30  # Fisher敏感性测试: 30个随机方向
EPSILON = 0.01  # 扰动幅度
K_VALUES = [5, 10, 20, 50, 100, 200]  # 信息损失测试维度


# ============================================================
# Exp 0: 随机基线 — Principal Angles的Monte Carlo分布
# ============================================================
def random_baseline_principal_angles(d_model, k_subspace, n_samples=5000):
    """
    计算R^d中两个随机k维子空间的主角分布.
    
    关键: 高维空间中两个随机子空间几乎正交!
    d=2560, k=10时, 期望主角 ≈ 89°, 这意味着Phase 123的87-89°可能毫无结构意义.
    """
    print(f"\n[Exp 0] Random baseline: d={d_model}, k={k_subspace}, n_samples={n_samples}")
    t0 = time.time()
    
    all_angles = []
    min_angles = []  # 每对子空间的最小主角
    mean_angles = []  # 每对子空间的平均主角
    
    for _ in range(n_samples):
        # 两个随机正交基
        A = np.random.randn(d_model, k_subspace)
        B = np.random.randn(d_model, k_subspace)
        Q_A, _ = np.linalg.qr(A)
        Q_B, _ = np.linalg.qr(B)
        
        # Principal angles via SVD of Q_A^T Q_B
        M = Q_A.T @ Q_B
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, -1, 1)
        angles = np.degrees(np.arccos(np.clip(np.abs(svals), 0, 1)))
        angles = np.sort(angles)
        
        all_angles.extend(angles)
        min_angles.append(angles[0])
        mean_angles.append(np.mean(angles))
    
    result = {
        'd_model': d_model,
        'k_subspace': k_subspace,
        'n_samples': n_samples,
        'all_angles_mean': float(np.mean(all_angles)),
        'all_angles_std': float(np.std(all_angles)),
        'all_angles_median': float(np.median(all_angles)),
        'all_angles_p5': float(np.percentile(all_angles, 5)),
        'all_angles_p95': float(np.percentile(all_angles, 95)),
        'min_angle_mean': float(np.mean(min_angles)),
        'min_angle_std': float(np.std(min_angles)),
        'min_angle_p5': float(np.percentile(min_angles, 5)),
        'mean_angle_mean': float(np.mean(mean_angles)),
        'mean_angle_std': float(np.std(mean_angles)),
    }
    
    print(f"  随机子空间主角分布:")
    print(f"    所有角度: mean={result['all_angles_mean']:.1f}° ± {result['all_angles_std']:.1f}°, "
          f"median={result['all_angles_median']:.1f}°")
    print(f"    最小主角: mean={result['min_angle_mean']:.1f}° ± {result['min_angle_std']:.1f}°, "
          f"p5={result['min_angle_p5']:.1f}°")
    print(f"    平均主角: mean={result['mean_angle_mean']:.1f}° ± {result['mean_angle_std']:.1f}°")
    print(f"  耗时: {time.time()-t0:.1f}s")
    
    return result


def compare_with_phase123(baseline_result, phase123_cross_angle, d_model, k_subspace):
    """比较Phase 123的交叉角度与随机基线"""
    random_mean = baseline_result['mean_angle_mean']
    random_std = baseline_result['mean_angle_std']
    
    # Z-score: 测量值偏离随机基线多少个标准差
    if random_std > 0:
        z_score = (phase123_cross_angle - random_mean) / random_std
    else:
        z_score = 0
    
    is_significant = abs(z_score) > 2  # 2σ显著
    
    print(f"\n  Phase 123交叉角度 vs 随机基线:")
    print(f"    测量值: {phase123_cross_angle:.1f}°")
    print(f"    随机期望: {random_mean:.1f}° ± {random_std:.1f}°")
    print(f"    Z-score: {z_score:.2f}")
    print(f"    显著性: {'YES' if is_significant else 'NO (接近随机基线!)'}")
    
    return {
        'measured_angle': phase123_cross_angle,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_score': float(z_score),
        'is_significant': is_significant,
    }


# ============================================================
# 数据收集: 激活PCA
# ============================================================
def collect_activations_and_pca(model, tokenizer, device, model_info, prompts, target_layers):
    """收集各层隐藏状态并计算PCA"""
    print(f"\n[数据收集] 收集激活并计算PCA...")
    t0 = time.time()
    
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    layers = get_layers(model)
    
    # 收集所有prompt在所有目标层的隐藏状态
    layer_activations = {l: [] for l in target_layers}
    layer_next_tokens = {l: [] for l in target_layers}  # 下一个token的ID
    base_logits_list = []
    
    for pidx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).input_ids
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits
        
        base_logits_list.append(logits.detach().cpu())
        
        for l in target_layers:
            # hidden_states[l+1] 是第l层的输出 (hidden_states[0]是embedding)
            hs = outputs.hidden_states[l + 1].detach().cpu().float().numpy()
            layer_activations[l].append(hs[0])  # (seq_len, d_model)
            
            # 保存next token IDs
            next_tokens = input_ids[0, 1:].cpu().numpy()  # 每个位置的next token
            layer_next_tokens[l].append(next_tokens)
        
        if (pidx + 1) % 5 == 0:
            print(f"  收集: {pidx+1}/{len(prompts)} prompts")
    
    # 计算PCA
    pca_results = {}
    for l in target_layers:
        # 合并所有prompt的激活
        all_acts = np.concatenate(layer_activations[l], axis=0)  # (total_tokens, d_model)
        
        # 中心化
        mean = all_acts.mean(axis=0)
        centered = all_acts - mean
        
        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # 有效秩 (95%方差)
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        eff_rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        
        pca_results[l] = {
            'components': Vt,  # (d_model, d_model) - 每行是一个主成分
            'eigenvalues': S,  # 降序排列
            'mean': mean,
            'eff_rank_95': eff_rank_95,
            'eff_rank_99': eff_rank_99,
            'total_variance': float(total_var),
            'top10_variance_ratio': float(np.sum(S[:10]**2) / total_var),
            'top50_variance_ratio': float(np.sum(S[:50]**2) / total_var),
        }
        
        print(f"  Layer {l}: eff_rank(95%)={eff_rank_95}, eff_rank(99%)={eff_rank_99}, "
              f"top10方差比={pca_results[l]['top10_variance_ratio']:.3f}")
    
    print(f"  数据收集+PCA耗时: {time.time()-t0:.1f}s")
    
    return pca_results, layer_activations, layer_next_tokens, base_logits_list


# ============================================================
# Exp 1: Fisher敏感性谱 — 沿PCA方向扰动测量预测变化
# ============================================================
def fisher_sensitivity_experiment(model, tokenizer, device, model_info, 
                                   prompts, pca_results, target_layers):
    """
    核心实验: Fisher信息在PCA方向上的投影
    
    方法: 对h_l沿方向v添加/减去ε*v, 测量log p(y*|h)的变化
    → 如果Fisher谱比PCA谱衰减更快 → 信息比能量更集中
    → 如果Fisher谱与PCA谱一致 → 信息=能量
    
    关键指标: Fisher sensitivity_k / PCA eigenvalue_k 的比值随k的变化
    """
    print(f"\n[Exp 1] Fisher敏感性谱")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}
    
    for l in target_layers:
        print(f"  Layer {l}...")
        pca_comp = pca_results[l]['components']  # (d_model, d_model)
        pca_eigenvals = pca_results[l]['eigenvalues']
        
        # 选择方向: top-N PCA + N random
        n_pca = min(N_PCA_DIRS, d_model)
        n_random = min(N_RANDOM_DIRS, d_model)
        
        pca_dirs = pca_comp[:n_pca]  # (n_pca, d_model)
        random_dirs = np.random.randn(n_random, d_model)
        # 正交化随机方向
        random_dirs, _ = np.linalg.qr(random_dirs.T)
        random_dirs = random_dirs.T[:n_random]
        # 归一化
        for i in range(n_random):
            random_dirs[i] /= np.linalg.norm(random_dirs[i])
        
        all_dirs = np.concatenate([pca_dirs, random_dirs], axis=0)
        dir_types = ['pca'] * n_pca + ['random'] * n_random
        
        # 对每个prompt测量Fisher敏感性
        fisher_sens = np.zeros(len(all_dirs))  # 每个方向的平均敏感性
        fisher_sens_detail = {i: [] for i in range(len(all_dirs))}
        
        for pidx, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, 
                                   max_length=MAX_SEQ_LEN).input_ids
            input_ids = input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            # 基线前向: 获取原始log p(y*|h)
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
                base_logits = base_outputs.logits
            
            # 对每个方向进行扰动
            for d_idx, (direction, dir_type) in enumerate(zip(all_dirs, dir_types)):
                # 创建扰动tensor: 在所有位置添加ε*v
                perturbation = torch.tensor(
                    direction, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(0) * EPSILON  # (1, 1, d_model)
                
                # 正向扰动
                def make_perturb_hook(sign):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            return (h + sign * perturbation.to(h.dtype),) + output[1:]
                        return output + sign * perturbation.to(output.dtype)
                    return hook_fn
                
                # +ε 前向
                handle = layers[l].register_forward_hook(make_perturb_hook(+1))
                with torch.no_grad():
                    plus_outputs = model(input_ids)
                    plus_logits = plus_outputs.logits
                handle.remove()
                
                # -ε 前向
                handle = layers[l].register_forward_hook(make_perturb_hook(-1))
                with torch.no_grad():
                    minus_outputs = model(input_ids)
                    minus_logits = minus_outputs.logits
                handle.remove()
                
                # 计算每个位置的Fisher敏感性
                for pos in range(seq_len - 1):
                    y_star = input_ids[0, pos + 1].item()
                    
                    log_p_plus = F.log_softmax(plus_logits[0, pos].float(), dim=-1)[y_star].item()
                    log_p_minus = F.log_softmax(minus_logits[0, pos].float(), dim=-1)[y_star].item()
                    log_p_base = F.log_softmax(base_logits[0, pos].float(), dim=-1)[y_star].item()
                    
                    # Fisher sensitivity = (f(x+ε) - f(x-ε)) / (2ε)
                    sensitivity = (log_p_plus - log_p_minus) / (2 * EPSILON)
                    
                    fisher_sens_detail[d_idx].append({
                        'sensitivity': float(sensitivity),
                        'log_p_base': float(log_p_base),
                    })
                    fisher_sens[d_idx] += abs(sensitivity)
            
            if (pidx + 1) % 5 == 0:
                print(f"    Prompt {pidx+1}/{len(prompts)}")
        
        # 归一化
        n_measurements = len(prompts) * (MAX_SEQ_LEN - 1)  # 近似
        fisher_sens /= max(n_measurements, 1)
        
        # 构建结果
        pca_fisher = fisher_sens[:n_pca]
        random_fisher = fisher_sens[n_pca:]
        
        # 归一化PCA特征值 (与Fisher敏感性可比)
        pca_eigenvals_norm = pca_eigenvals[:n_pca] / np.max(pca_eigenvals[:n_pca])
        pca_fisher_norm = pca_fisher / np.max(pca_fisher) if np.max(pca_fisher) > 0 else pca_fisher
        
        # Fisher/PCA比值: 如果衰减, 说明信息更集中
        fisher_pca_ratio = pca_fisher_norm / np.maximum(pca_eigenvals_norm, 1e-10)
        
        results[l] = {
            'pca_fisher_sensitivity': pca_fisher.tolist(),
            'random_fisher_sensitivity': random_fisher.tolist(),
            'pca_eigenvalues_norm': pca_eigenvals_norm.tolist(),
            'pca_fisher_norm': pca_fisher_norm.tolist(),
            'fisher_pca_ratio': fisher_pca_ratio.tolist(),
            'pca_fisher_top5_mean': float(np.mean(pca_fisher[:5])),
            'pca_fisher_top30_mean': float(np.mean(pca_fisher[:n_pca])),
            'random_fisher_mean': float(np.mean(random_fisher)),
            'fisher_concentration_ratio': float(np.mean(pca_fisher[:5]) / max(np.mean(pca_fisher[:n_pca]), 1e-10)),
            'pca_energy_concentration_ratio': float(np.sum(pca_eigenvals[:5]**2) / max(np.sum(pca_eigenvals[:n_pca]**2), 1e-10)),
        }
        
        print(f"    PCA Fisher top5均值={results[l]['pca_fisher_top5_mean']:.4f}, "
              f"top30均值={results[l]['pca_fisher_top30_mean']:.4f}, "
              f"random均值={results[l]['random_fisher_mean']:.4f}")
        print(f"    Fisher集中度(前5/前30)={results[l]['fisher_concentration_ratio']:.3f}, "
              f"PCA能量集中度(前5/前30)={results[l]['pca_energy_concentration_ratio']:.3f}")
    
    print(f"  Fisher敏感性实验耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 2: 预测信息损失 vs 子空间维度
# ============================================================
def information_loss_experiment(model, tokenizer, device, model_info,
                                prompts, pca_results, target_layers):
    """
    核心实验: 测试"最小充分子空间"是否存在
    
    方法: 在第l层, 将h_l投影到前k个PCA方向, 然后继续前向传播
    → 测量KL散度: 原始logits vs 投影后logits
    → 如果k很小时KL就接近0 → 最小充分子空间存在!
    → 如果k需要很大 → 所有维度都重要
    
    这是Phase 124最关键的实验.
    """
    print(f"\n[Exp 2] 预测信息损失 vs 子空间维度")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}
    
    for l in target_layers:
        print(f"  Layer {l}...")
        pca_comp = pca_results[l]['components']  # (d_model, d_model)
        
        layer_results = {k: {'kl_divs': [], 'top1_match': [], 'cosine_sims': []} 
                        for k in K_VALUES}
        
        for pidx, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                                   max_length=MAX_SEQ_LEN).input_ids
            input_ids = input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            # 基线前向
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
                base_logits = base_outputs.logits
                base_h = base_outputs.hidden_states[l + 1]  # 第l层输出
            
            # 基线 top-1 prediction
            base_top1 = base_logits[0, -1].argmax().item()
            base_log_probs = F.log_softmax(base_logits[0, -1].float(), dim=-1)
            
            for k in K_VALUES:
                if k > d_model:
                    continue
                
                # 投影矩阵: V_k V_k^T (d_model × d_model)
                V_k = torch.tensor(pca_comp[:k], dtype=torch.float32, device=device)  # (k, d_model)
                
                # 创建hook: 将h_l替换为其在V_k子空间上的投影
                def make_projection_hook(V_k_matrix, layer_h):
                    """Hook that projects h_l onto top-k PCA directions"""
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            # 投影: h_proj = V_k^T (V_k h^T)  -- 逐位置
                            # h: (1, seq_len, d_model)
                            h_float = h.float()
                            # coeffs = h @ V_k^T → (1, seq_len, k)
                            coeffs = torch.matmul(h_float, V_k_matrix.T)
                            # proj = coeffs @ V_k → (1, seq_len, d_model)
                            h_proj = torch.matmul(coeffs, V_k_matrix)
                            return (h_proj.to(h.dtype),) + output[1:]
                        else:
                            h = output
                            h_float = h.float()
                            coeffs = torch.matmul(h_float, V_k_matrix.T)
                            h_proj = torch.matmul(coeffs, V_k_matrix)
                            return h_proj.to(h.dtype)
                    return hook_fn
                
                # 前向传播with投影
                handle = layers[l].register_forward_hook(make_projection_hook(V_k, base_h))
                with torch.no_grad():
                    proj_outputs = model(input_ids)
                    proj_logits = proj_outputs.logits
                handle.remove()
                
                # 计算指标
                proj_log_probs = F.log_softmax(proj_logits[0, -1].float(), dim=-1)
                
                # KL散度: KL(base || proj)
                kl_div = F.kl_div(proj_log_probs, F.softmax(base_log_probs, dim=-1), 
                                  reduction='sum').item()
                
                # Top-1匹配
                proj_top1 = proj_logits[0, -1].argmax().item()
                top1_match = 1 if proj_top1 == base_top1 else 0
                
                # 余弦相似度
                cos_sim = F.cosine_similarity(
                    base_logits[0, -1].float().unsqueeze(0), 
                    proj_logits[0, -1].float().unsqueeze(0)
                ).item()
                
                layer_results[k]['kl_divs'].append(kl_div)
                layer_results[k]['top1_match'].append(top1_match)
                layer_results[k]['cosine_sims'].append(cos_sim)
            
            if (pidx + 1) % 5 == 0:
                print(f"    Prompt {pidx+1}/{len(prompts)}")
        
        # 汇总
        results[l] = {}
        for k in K_VALUES:
            if k not in layer_results or len(layer_results[k]['kl_divs']) == 0:
                continue
            results[l][k] = {
                'kl_div_mean': float(np.mean(layer_results[k]['kl_divs'])),
                'kl_div_std': float(np.std(layer_results[k]['kl_divs'])),
                'top1_match_rate': float(np.mean(layer_results[k]['top1_match'])),
                'cosine_sim_mean': float(np.mean(layer_results[k]['cosine_sims'])),
                'cosine_sim_std': float(np.std(layer_results[k]['cosine_sims'])),
            }
            print(f"    k={k}: KL={results[l][k]['kl_div_mean']:.4f}±{results[l][k]['kl_div_std']:.4f}, "
                  f"top1_match={results[l][k]['top1_match_rate']:.2f}, "
                  f"cos_sim={results[l][k]['cosine_sim_mean']:.4f}")
    
    print(f"  信息损失实验耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Exp 3: eff_rank饱和测试
# ============================================================
def eff_rank_saturation_test(model, tokenizer, device, model_info,
                              prompts, pca_results, target_layers):
    """
    生死实验: eff_rank是否随k_probes增加而饱和?
    
    方法: 用投影方法代替Jacobian
    - 将h_l投影到k维PCA子空间
    - 测量投影后与原始输出的余弦相似度
    - 如果k=50时cos_sim已经≈1 → eff_rank确实≈50
    - 如果cos_sim随k持续增长 → eff_rank远大于50
    
    这比直接计算Jacobian更高效, 且对所有模型兼容.
    """
    print(f"\n[Exp 3] eff_rank饱和测试")
    t0 = time.time()
    
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 扩展k值范围
    k_test = [10, 20, 50, 100, 200, 400]
    if d_model < 400:
        k_test = [k for k in k_test if k <= d_model]
    
    results = {}
    
    for l in target_layers:
        print(f"  Layer {l}...")
        pca_comp = pca_results[l]['components']
        
        k_results = {k: [] for k in k_test}
        
        for pidx, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors='pt', truncation=True,
                                   max_length=MAX_SEQ_LEN).input_ids
            input_ids = input_ids.to(device)
            
            # 基线前向
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
                base_h = base_outputs.hidden_states[l + 1].float()  # (1, seq_len, d_model)
            
            for k in k_test:
                V_k = torch.tensor(pca_comp[:k], dtype=torch.float32, device=device)
                
                # 投影重建质量: ||V_k V_k^T h||^2 / ||h||^2
                h = base_h[0]  # (seq_len, d_model)
                coeffs = torch.matmul(h, V_k.T)  # (seq_len, k)
                h_proj = torch.matmul(coeffs, V_k)  # (seq_len, d_model)
                
                recon_ratio = float(
                    (h_proj ** 2).sum() / max((h ** 2).sum().item(), 1e-10)
                )
                k_results[k].append(recon_ratio)
            
            if (pidx + 1) % 5 == 0:
                print(f"    Prompt {pidx+1}/{len(prompts)}")
        
        results[l] = {}
        for k in k_test:
            mean_ratio = float(np.mean(k_results[k]))
            results[l][k] = {
                'reconstruction_ratio_mean': mean_ratio,
                'reconstruction_ratio_std': float(np.std(k_results[k])),
            }
            print(f"    k={k}: 重建率={mean_ratio:.4f}")
        
        # 判断是否饱和
        ratios = [results[l][k]['reconstruction_ratio_mean'] for k in k_test]
        if len(ratios) >= 2:
            # 检查: k=50→100的增益 vs k=100→200的增益
            if len(ratios) >= 3:
                gain_50_100 = ratios[2] - ratios[1] if len(ratios) > 2 else 0  # k=50→100
                gain_100_200 = ratios[3] - ratios[2] if len(ratios) > 3 else 0  # k=100→200
                saturation = gain_100_200 < gain_50_100 * 0.5  # 增益减半→开始饱和
                results[l]['saturation_analysis'] = {
                    'gain_50_100': float(gain_50_100),
                    'gain_100_200': float(gain_100_200),
                    'is_saturating': bool(saturation),
                    'estimated_eff_rank': k_test[np.searchsorted(ratios, 0.95)] if 0.95 in [r for r in ratios] else 'unknown',
                }
                print(f"    饱和分析: gain(50→100)={gain_50_100:.4f}, gain(100→200)={gain_100_200:.4f}, "
                      f"饱和={'YES' if saturation else 'NO'}")
    
    print(f"  eff_rank饱和测试耗时: {time.time()-t0:.1f}s")
    return results


# ============================================================
# 主流程
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print("=" * 70)
    print(f"Phase 124: Information Geometry & Mortality Experiments")
    print(f"Model: {model_name}")
    print("=" * 70)
    
    # ===== Exp 0: 随机基线 (不需要模型) =====
    print("\n" + "=" * 50)
    print("Exp 0: 随机基线 — Principal Angles分布")
    print("=" * 50)
    
    # 根据模型确定d_model
    d_model_map = {'qwen3': 2560, 'deepseek7b': 3584, 'glm4': 4096}
    d_model = d_model_map.get(model_name, 2560)
    
    # 测试不同k_subspace值
    baseline_results = {}
    for k_sub in [5, 10, 20, 50]:
        baseline_results[k_sub] = random_baseline_principal_angles(d_model, k_sub, n_samples=3000)
    
    # 与Phase 123比较
    phase123_cross_angles = {'qwen3': 87.5, 'deepseek7b': 88.2, 'glm4': 87.8}
    measured_angle = phase123_cross_angles.get(model_name, 87.5)
    comparison = compare_with_phase123(baseline_results[10], measured_angle, d_model, 10)
    
    # 保存Exp 0结果
    exp0_result = {
        'model': model_name,
        'd_model': d_model,
        'baselines': {str(k): v for k, v in baseline_results.items()},
        'phase123_comparison': comparison,
    }
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, f'phase124_exp0_{model_name}_random_baseline.json'), 'w') as f:
        json.dump(exp0_result, f, indent=2, ensure_ascii=False)
    print(f"\nExp 0 结果已保存")
    
    # ===== 加载模型 =====
    print("\n" + "=" * 50)
    print("加载模型")
    print("=" * 50)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"  class={model_info.model_class}, n_layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}, vocab={model_info.vocab_size}")
    
    # 目标层: 前/中/后
    n_layers = model_info.n_layers
    target_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    print(f"  目标层: {target_layers}")
    
    # 选取prompts
    test_prompts = PROMPTS[:N_PROMPTS]
    
    # ===== 数据收集 + PCA =====
    pca_results, layer_activations, layer_next_tokens, base_logits = \
        collect_activations_and_pca(model, tokenizer, device, model_info, 
                                     test_prompts, target_layers)
    
    # ===== Exp 1: Fisher敏感性 =====
    print("\n" + "=" * 50)
    print("Exp 1: Fisher敏感性谱")
    print("=" * 50)
    
    fisher_results = fisher_sensitivity_experiment(
        model, tokenizer, device, model_info, 
        test_prompts, pca_results, target_layers
    )
    
    with open(os.path.join(out_dir, f'phase124_exp1_{model_name}_fisher_sensitivity.json'), 'w') as f:
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(fisher_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 1 结果已保存")
    
    # ===== Exp 2: 信息损失 vs 维度 =====
    print("\n" + "=" * 50)
    print("Exp 2: 预测信息损失 vs 子空间维度")
    print("=" * 50)
    
    info_loss_results = information_loss_experiment(
        model, tokenizer, device, model_info,
        test_prompts, pca_results, target_layers
    )
    
    with open(os.path.join(out_dir, f'phase124_exp2_{model_name}_info_loss.json'), 'w') as f:
        json.dump(info_loss_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 2 结果已保存")
    
    # ===== Exp 3: eff_rank饱和 =====
    print("\n" + "=" * 50)
    print("Exp 3: eff_rank饱和测试")
    print("=" * 50)
    
    eff_rank_results = eff_rank_saturation_test(
        model, tokenizer, device, model_info,
        test_prompts, pca_results, target_layers
    )
    
    with open(os.path.join(out_dir, f'phase124_exp3_{model_name}_eff_rank_saturation.json'), 'w') as f:
        json.dump(eff_rank_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Exp 3 结果已保存")
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("Phase 124 汇总")
    print("=" * 70)
    
    # 关键结论
    print("\n[结论1] 随机基线 vs Phase 123交叉角度:")
    print(f"  Phase 123测量的87-89°交叉角度 vs 随机期望{comparison['random_mean']:.1f}°")
    print(f"  Z-score={comparison['z_score']:.2f} → "
          f"{'有结构意义' if comparison['is_significant'] else '可能只是随机基线!'}")
    
    print("\n[结论2] Fisher信息谱 vs PCA谱:")
    for l in target_layers:
        if l in fisher_results:
            r = fisher_results[l]
            print(f"  Layer {l}: Fisher集中度={r['fisher_concentration_ratio']:.3f}, "
                  f"PCA能量集中度={r['pca_energy_concentration_ratio']:.3f}")
    
    print("\n[结论3] 最小充分子空间:")
    for l in target_layers:
        if l in info_loss_results:
            # 找到KL<0.1的最小k
            min_k_sufficient = None
            for k in sorted(info_loss_results[l].keys(), key=lambda x: int(x)):
                if info_loss_results[l][k]['kl_div_mean'] < 0.1:
                    min_k_sufficient = k
                    break
            if min_k_sufficient:
                print(f"  Layer {l}: 最小充分子空间≈{min_k_sufficient}维 (KL<0.1)")
            else:
                print(f"  Layer {l}: 需要更多维度 (所有测试k的KL>0.1)")
    
    print("\n[结论4] eff_rank饱和:")
    for l in target_layers:
        if l in eff_rank_results and 'saturation_analysis' in eff_rank_results[l]:
            sa = eff_rank_results[l]['saturation_analysis']
            print(f"  Layer {l}: 饱和={'YES' if sa['is_saturating'] else 'NO'}, "
                  f"gain(50→100)={sa['gain_50_100']:.4f}, gain(100→200)={sa['gain_100_200']:.4f}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPhase 124 完成!")


if __name__ == "__main__":
    main()
