"""
Phase 152: 统计语言动力学 — Statistical Language Dynamics
=========================================================

核心升级: 从"向量级分析"跃迁到"统计动力学"层

用户核心批评(全部正确):
  1. cos(δ,δ')→0 ≠ 信息丢失 → 需要测二阶统计量(MI/covariance)
  2. hidden state是瞬时缓存 → 真正结构在统计量中
  3. W_U row-space ≠ semantic space → 需要独立于W_U的语义度量
  4. 欧氏距离 ≠ 决策边界灵敏度 → 需要测logit boundary
  5. 需要跨模型验证 → 所有结论可能只是架构偶然性

5个核心实验:
  Exp 1: Mutual Information Flow — I(h_ℓ; h_0) 而非 cos(δ,δ')
    - 核心问题: 方向混合后, 统计信息是否保留?
    - 方法: 线性预测性R² + 扰动协方差结构 + 离散MI
    - 若MI在cos→0后仍然显著 → "方向丢失≠信息丢失"被确认
    
  Exp 2: Logit Boundary Geometry — 决策边界灵敏度
    - 核心问题: 为什么弱欧氏耦合能产生因果效应?
    - 方法: 测量logit margin, 扰动后logit crossing概率
    - 找到"临界扰动强度"使得top-1 token切换
    
  Exp 3: Statistical Attractor — 不同初值是否收敛到相同统计量?
    - 核心问题: 语言生成长期稳定的原因?
    - 方法: 从不同perturbed起点生成, 测量统计量收敛性
    - 寻找"统计吸引子"的 basin of attraction
    
  Exp 4: Second-Order Perturbation Propagation — 二阶统计传播
    - 核心问题: 扰动协方差矩阵如何传播?
    - 方法: Cov(δ^(ℓ), δ^(0)) 的结构和秩
    - 若协方差矩阵在cos→0后仍有结构 → 二阶信息流存在
    
  Exp 5: Cross-Model Universality — 与随机初始化模型对比
    - 核心问题: 哪些性质是训练产生的? 哪些是架构固有的?
    - 方法: 同架构随机初始化模型的相同实验
    - 训练模型 - 随机模型 = "语言结构"

用法:
  python tests/glm5/phase152_statistical_dynamics.py qwen3
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

# ============================================================
# 全局参数
# ============================================================
EPSILON = 1.0
N_SENTENCES = 30    # 加大数据量
N_ROLLOUTS = 15     # 每个起点的生成次数(加大)
OUTPUT_DIR = Path("tests/glm5_temp")

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
    "The cat sat on the windowsill and watched",
    "Through the telescope, they observed a new",
    "The river flowed gently through the valley",
    "She opened the letter and read the",
    "The painting on the wall depicted a",
    "During the concert, the audience was",
    "The invention changed the way people",
    "He wrote a letter to his friend about",
    "The students in the classroom were learning",
    "The old building at the corner had",
]


# ============================================================
# 工具函数
# ============================================================
def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def get_topk_tokens(logits_np, k=5):
    return np.argsort(logits_np)[-k:][::-1].tolist()


def compute_kl(p, q):
    """KL divergence D_KL(p||q)"""
    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ============================================================
# Exp 1: Mutual Information Flow
# ============================================================
def exp1_mi_flow(model, tokenizer, model_name):
    """
    核心问题: cos(δ,δ')→0 后, 统计信息是否保留?
    
    三种方法测量 I(h_ℓ; h_0):
    
    方法A: 线性预测性 R²
    - 对每层ℓ, 用线性回归从h_0预测h_ℓ
    - R² > 0 → 线性信息保留
    - R² = 0 → 线性信息丢失
    
    方法B: 扰动协方差结构
    - Cov(δ^(ℓ)_i, δ^(0)_j) 对所有i,j
    - 协方差矩阵的秩和结构 → 二阶信息流
    
    方法C: 离散决策互信息
    - I(top1_token_at_ℓ; perturbation_direction_at_0)
    - 如果MI > 0 → 扰动方向影响了离散决策
    """
    print("\n" + "="*60)
    print("Exp 1: Mutual Information Flow")
    print("核心: cos→0 后, 统计信息是否保留?")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers + [n_layers]))
    
    # === 方法A: 线性预测性 ===
    # 收集N个输入在各层的hidden states
    print("\n  --- Method A: Linear Predictability ---")
    print(f"  Collecting hidden states from {N_SENTENCES} sentences...")
    
    all_hs = {}  # {layer_idx: [N, d_model]}
    for li in sample_layers:
        all_hs[li] = []
    
    for sent_idx in range(min(N_SENTENCES, 30)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        
        for li in sample_layers:
            vec = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_hs[li].append(vec)
    
    # 对每层, 用h_0预测h_ℓ
    h0_matrix = np.array(all_hs[0])  # [N, d_model]
    h0_centered = h0_matrix - h0_matrix.mean(axis=0)
    
    linear_r2 = {}
    for li in sample_layers:
        hl_matrix = np.array(all_hs[li])  # [N, d_model]
        hl_centered = hl_matrix - hl_matrix.mean(axis=0)
        
        # 逐维度回归: 用h0的PCA主成分预测hℓ的PCA主成分
        # 先做PCA降维
        n_pca = min(50, d_model, h0_centered.shape[0] - 1)
        
        # h0 PCA
        cov_h0 = (h0_centered.T @ h0_centered) / (h0_centered.shape[0] - 1)
        try:
            eigvals_h0, eigvecs_h0 = np.linalg.eigh(cov_h0)
            idx = np.argsort(-eigvals_h0)[:n_pca]
            h0_pca = h0_centered @ eigvecs_h0[:, idx]  # [N, n_pca]
        except:
            h0_pca = h0_centered[:, :n_pca]
        
        # hℓ PCA
        cov_hl = (hl_centered.T @ hl_centered) / (hl_centered.shape[0] - 1)
        try:
            eigvals_hl, eigvecs_hl = np.linalg.eigh(cov_hl)
            idx_hl = np.argsort(-eigvals_hl)[:n_pca]
            hl_pca = hl_centered @ eigvecs_hl[:, idx_hl]  # [N, n_pca]
        except:
            hl_pca = hl_centered[:, :n_pca]
        
        # 线性回归: hℓ_pca = W @ h0_pca + b
        # 最小二乘
        N_samples = h0_pca.shape[0]
        if N_samples < 5:
            linear_r2[li] = 0.0
            continue
        
        # 分成训练/测试
        n_train = max(5, N_samples * 2 // 3)
        h0_train = h0_pca[:n_train]
        h0_test = h0_pca[n_train:]
        hl_train = hl_pca[:n_train]
        hl_test = hl_pca[n_train:]
        
        if h0_test.shape[0] < 2:
            linear_r2[li] = 0.0
            continue
        
        # 对每个hℓ的PCA分量, 用h0的PCA分量预测
        r2_per_component = []
        for comp_j in range(min(10, hl_pca.shape[1])):
            y_train = hl_train[:, comp_j]
            y_test = hl_test[:, comp_j]
            
            # 最小二乘解
            try:
                W, residuals, rank, sv = np.linalg.lstsq(
                    np.column_stack([h0_train, np.ones(n_train)]),
                    y_train, rcond=None)
                y_pred = np.column_stack([h0_test, np.ones(h0_test.shape[0])]) @ W
                
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                r2_per_component.append(max(0, r2))
            except:
                r2_per_component.append(0.0)
        
        avg_r2 = np.mean(r2_per_component) if r2_per_component else 0
        linear_r2[li] = float(avg_r2)
    
    print("\n  Linear Predictability R² (h₀ → hℓ):")
    for li in sample_layers:
        r2 = linear_r2.get(li, 0)
        status = "STRONG" if r2 > 0.3 else "MODERATE" if r2 > 0.1 else "WEAK" if r2 > 0.01 else "~ZERO"
        print(f"    L{li:>3d}: R²={r2:.4f} [{status}]")
    
    # === 方法B: 扰动协方差结构 ===
    print("\n  --- Method B: Perturbation Covariance Structure ---")
    
    # 对同一个输入, 注入N种不同方向扰动, 收集各层的δ
    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    clean_hs = out_clean.hidden_states
    
    n_perturbations = 100  # 100种不同扰动方向
    
    # 收集各层的扰动响应
    delta_at_layer = {}  # {layer: [n_perturbations, d_model]}
    for li in sample_layers:
        delta_at_layer[li] = []
    
    for p_idx in range(n_perturbations):
        np.random.seed(42 + p_idx)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta) * EPSILON
        
        # 在L0注入
        layers = get_layers(model)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        
        def make_hook(pos, delta_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return out
            return hook
        
        last_pos = input_ids.shape[1] - 1
        hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]
        
        try:
            with torch.no_grad():
                out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                      output_hidden_states=True)
        except:
            for h in hooks:
                h.remove()
            continue
        
        for h in hooks:
            h.remove()
        
        for li in sample_layers:
            perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
            clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
            delta_prop = perturbed_vec - clean_vec
            delta_at_layer[li].append(delta_prop)
    
    # 计算各层的扰动协方差矩阵
    # 使用PCA分析协方差矩阵的秩和结构
    cov_rank = {}
    cov_top_eigenvalue_ratio = {}
    delta_correlation_with_input = {}  # δ^(ℓ) 与 δ^(0) 的平均相关
    
    delta_at_0 = np.array(delta_at_layer[0])  # [n_pert, d_model]
    
    for li in sample_layers:
        delta_matrix = np.array(delta_at_layer[li])  # [n_pert, d_model]
        
        if delta_matrix.shape[0] < 5:
            cov_rank[li] = 0
            cov_top_eigenvalue_ratio[li] = 0
            delta_correlation_with_input[li] = 0
            continue
        
        # 协方差矩阵 [d_model, d_model] — 太大, 用SVD直接分析
        # SVD of delta_matrix: δ = U S V^T
        # 协方差 = V S^2 V^T / (n-1)
        try:
            # 对delta_matrix做SVD (高效方法)
            U_d, s_d, Vt_d = np.linalg.svd(delta_matrix, full_matrices=False)
            
            # 有效秩: 95%能量
            total_energy = np.sum(s_d ** 2)
            cumulative = np.cumsum(s_d ** 2)
            k95 = np.searchsorted(cumulative, 0.95 * total_energy) + 1
            cov_rank[li] = int(k95)
            
            # top特征值占比
            top_ratio = (s_d[0] ** 2) / total_energy if total_energy > 0 else 0
            cov_top_eigenvalue_ratio[li] = float(top_ratio)
        except:
            cov_rank[li] = 0
            cov_top_eigenvalue_ratio[li] = 0
        
        # δ^(ℓ) 与 δ^(0) 的平均相关
        correlations = []
        for p_idx in range(min(50, delta_matrix.shape[0])):
            d_l = delta_matrix[p_idx]
            d_0 = delta_at_0[p_idx] if p_idx < delta_at_0.shape[0] else np.zeros(d_model)
            norm_l = np.linalg.norm(d_l)
            norm_0 = np.linalg.norm(d_0)
            if norm_l > 1e-10 and norm_0 > 1e-10:
                correlations.append(float(np.dot(d_l, d_0) / (norm_l * norm_0)))
        
        delta_correlation_with_input[li] = float(np.mean(correlations)) if correlations else 0
    
    print("\n  Perturbation Covariance Structure:")
    for li in sample_layers:
        rank = cov_rank.get(li, 0)
        top_ratio = cov_top_eigenvalue_ratio.get(li, 0)
        corr = delta_correlation_with_input.get(li, 0)
        print(f"    L{li:>3d}: effective_rank={rank}, top_eigenvalue_ratio={top_ratio:.4f}, "
              f"corr(δ_ℓ, δ_0)={corr:.4f}")
    
    # === 方法C: 离散决策MI ===
    print("\n  --- Method C: Discrete Decision MI ---")
    # 对每层注入扰动, 看最终top-1 token的分布
    # I(top1_token; perturbation_type) > 0 → 信息保留
    
    # 定义"扰动类型"为: 5种不同语义方向
    W_U = get_W_U(model, model_name)
    semantic_words = ["not", "the", "big", "science", "he", "red"]
    semantic_dirs = {}
    for word in semantic_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if tok_ids:
            direction = W_U[tok_ids[0]].copy()
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                semantic_dirs[word] = direction / norm
    
    # 对每个句子, 注入不同语义方向, 记录最终top1 token
    n_test_sents = 15
    decision_mi_data = {word: [] for word in semantic_dirs}
    decision_mi_data['random'] = []
    
    for sent_idx in range(n_test_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        last_pos = input_ids.shape[1] - 1
        
        # 每种语义方向 + 随机方向
        for word, direction in semantic_dirs.items():
            delta = direction * 5.0  # 较大扰动确保有效
            
            layers = get_layers(model)
            delta_tensor = torch.tensor(delta, dtype=torch.float32)
            
            def make_hook_inj(pos, delta_t):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return out
                return hook
            
            hooks = [layers[0].register_forward_hook(make_hook_inj(last_pos, delta_tensor))]
            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                top1 = int(np.argmax(out.logits[0, -1, :].float().cpu().numpy()))
                decision_mi_data[word].append(top1)
            except:
                pass
            for h in hooks:
                h.remove()
        
        # 随机方向
        np.random.seed(sent_idx * 100)
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir) * 5.0
        
        delta_tensor = torch.tensor(rand_dir, dtype=torch.float32)
        hooks = [layers[0].register_forward_hook(make_hook_inj(last_pos, delta_tensor))]
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            top1 = int(np.argmax(out.logits[0, -1, :].float().cpu().numpy()))
            decision_mi_data['random'].append(top1)
        except:
            pass
        for h in hooks:
            h.remove()
    
    # 计算MI: I(direction_type; top1_token)
    # 简化: 比较不同方向的top1分布差异
    print("\n  Discrete Decision MI (inject semantic directions at L0, eps=5.0):")
    for word in list(semantic_dirs.keys()) + ['random']:
        tokens = decision_mi_data[word]
        if tokens:
            unique, counts = np.unique(tokens, return_counts=True)
            entropy = -np.sum((counts / len(tokens)) * np.log2(counts / len(tokens) + 1e-10))
            print(f"    '{word}': {len(tokens)} trials, unique_top1={len(unique)}, "
                  f"entropy={entropy:.2f}")
    
    # 计算方向间的top1分布重叠
    print("\n  Cross-direction top1 overlap:")
    dir_names = list(semantic_dirs.keys()) + ['random']
    for i, w1 in enumerate(dir_names):
        for w2 in dir_names[i+1:]:
            t1 = set(decision_mi_data[w1])
            t2 = set(decision_mi_data[w2])
            if t1 and t2:
                overlap = len(t1 & t2) / min(len(t1), len(t2))
                print(f"    {w1} vs {w2}: overlap={overlap:.3f}")
    
    return {
        'linear_r2': linear_r2,
        'cov_rank': cov_rank,
        'cov_top_eigenvalue_ratio': cov_top_eigenvalue_ratio,
        'delta_correlation_with_input': delta_correlation_with_input,
        'decision_mi_data': {k: v for k, v in decision_mi_data.items()},
    }


# ============================================================
# Exp 2: Logit Boundary Geometry
# ============================================================
def exp2_logit_boundary(model, tokenizer, model_name):
    """
    核心问题: 为什么弱欧氏耦合能产生因果效应?
    
    关键: Transformer是"连续状态 + 离散读出"
    → 真正敏感的是 logit margin (top-1 vs top-2 的差距)
    → 扰动是否跨越决策边界?
    
    方法:
    1. 测量每个位置的logit margin
    2. 系统增大扰动强度, 找到"临界扰动"使top-1切换
    3. 在logit margin小的位置(接近边界) vs 大的位置, 因果灵敏度差异
    """
    print("\n" + "="*60)
    print("Exp 2: Logit Boundary Geometry")
    print("核心: 决策边界灵敏度 vs 欧氏距离")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    results = []
    
    for sent_idx in range(min(N_SENTENCES, 25)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = out.logits[0, -1, :].float().cpu().numpy()
        probs = softmax(logits)
        
        sorted_ids = np.argsort(-logits)
        top1_id = sorted_ids[0]
        top2_id = sorted_ids[1]
        
        # Logit margin: top-1 与 top-2 的差距
        logit_margin = logits[top1_id] - logits[top2_id]
        
        # Probability margin
        prob_margin = probs[top1_id] - probs[top2_id]
        
        # 系统扫描扰动强度, 找到临界点
        inject_layer = n_layers // 2  # 在中间层注入, 更接近决策
        last_pos = input_ids.shape[1] - 1
        
        np.random.seed(sent_idx * 100)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta)
        
        eps_scan = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        switching_eps = None
        
        for eps in eps_scan:
            delta_scaled = delta * eps
            
            layers = get_layers(model)
            delta_tensor = torch.tensor(delta_scaled, dtype=torch.float32)
            
            def make_hook(pos, delta_t):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return out
                return hook
            
            hooks = [layers[inject_layer].register_forward_hook(make_hook(last_pos, delta_tensor))]
            
            try:
                with torch.no_grad():
                    out_p = model(input_ids=input_ids, attention_mask=attention_mask)
                
                perturbed_logits = out_p.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))
                
                if perturbed_top1 != top1_id and switching_eps is None:
                    switching_eps = eps
            except:
                pass
            
            for h in hooks:
                h.remove()
        
        results.append({
            'sent_idx': sent_idx,
            'prompt': prompt[:40],
            'logit_margin': float(logit_margin),
            'prob_margin': float(prob_margin),
            'switching_eps': switching_eps,
            'top1_token': tokenizer.decode([top1_id]).strip(),
            'top2_token': tokenizer.decode([top2_id]).strip(),
        })
    
    # === 汇总 ===
    print("\n  === Logit Boundary Summary ===")
    
    margins = [r['logit_margin'] for r in results]
    switching = [r['switching_eps'] for r in results if r['switching_eps'] is not None]
    
    print(f"  Logit margin: mean={np.mean(margins):.3f}, std={np.std(margins):.3f}, "
          f"range=[{np.min(margins):.3f}, {np.max(margins):.3f}]")
    print(f"  Switching rate: {len(switching)}/{len(results)} = {len(switching)/len(results):.1%}")
    if switching:
        print(f"  Switching eps: mean={np.mean(switching):.3f}, "
              f"median={np.median(switching):.3f}, range=[{np.min(switching):.3f}, {np.max(switching):.3f}]")
    
    # 按 logit margin 分组
    print("\n  --- By Logit Margin ---")
    for label, lo, hi in [("Narrow(margin<1)", 0, 1), ("Medium(1-3)", 1, 3), ("Wide(margin>3)", 3, 1000)]:
        subset = [r for r in results if lo <= r['logit_margin'] < hi]
        if subset:
            switch_rate = sum(1 for r in subset if r['switching_eps'] is not None) / len(subset)
            switch_vals = [r['switching_eps'] for r in subset if r['switching_eps'] is not None]
            print(f"  {label}: N={len(subset)}, switch_rate={switch_rate:.1%}, "
                  f"avg_switching_eps={np.mean(switch_vals):.2f}" if switch_vals else
                  f"  {label}: N={len(subset)}, switch_rate={switch_rate:.1%}")
    
    # 关键: logit margin vs switching eps 的相关性
    if len(results) > 5:
        margins_with_switch = [r['logit_margin'] for r in results if r['switching_eps'] is not None]
        eps_with_switch = [r['switching_eps'] for r in results if r['switching_eps'] is not None]
        margins_no_switch = [r['logit_margin'] for r in results if r['switching_eps'] is None]
        
        if margins_with_switch and margins_no_switch:
            corr = np.corrcoef(
                [r['logit_margin'] for r in results],
                [r['switching_eps'] if r['switching_eps'] is not None else 100 for r in results]
            )[0, 1]
            print(f"\n  Correlation(logit_margin, switching_eps): {corr:.3f}")
            print(f"  Margin for switching: mean={np.mean(margins_with_switch):.3f}")
            print(f"  Margin for no-switch: mean={np.mean(margins_no_switch):.3f}")
    
    # 打印详情
    print("\n  --- Per-Sentence Details ---")
    for r in results[:15]:
        switch_str = f"eps={r['switching_eps']:.2f}" if r['switching_eps'] else "no switch"
        print(f"  [{r['sent_idx']:2d}] margin={r['logit_margin']:.2f}, "
              f"top1='{r['top1_token']}', top2='{r['top2_token']}', {switch_str}")
    
    return results


# ============================================================
# Exp 3: Statistical Attractor
# ============================================================
def exp3_statistical_attractor(model, tokenizer, model_name):
    """
    核心问题: 不同初值是否收敛到相同统计量?
    
    这直接测试"受约束的高维统计运输系统"假说:
    - 如果存在"统计吸引子", 不同起点的生成会收敛到相似的统计特征
    - 如果没有, 不同起点会产生完全不同的统计特征
    
    方法:
    1. 从同一提示, 注入不同方向扰动
    2. 从每个扰动起点生成30步
    3. 测量统计量的收敛性:
       - Token type ratio (已确认的守恒量)
       - Attention entropy profile
       - Generation "genre" statistics
    """
    print("\n" + "="*60)
    print("Exp 3: Statistical Attractor")
    print("核心: 不同初值 → 统计量收敛?")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    d_model = info.d_model
    
    n_perturb_types = 8  # 8种不同初始扰动
    n_gen_steps = 30
    n_prompts = 4
    
    results = {}
    
    for prompt_idx in range(n_prompts):
        prompt = TEST_PROMPTS[prompt_idx * 5]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 获取clean hidden state
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        
        # 从不同扰动起点生成
        rollout_data = []
        
        for p_type in range(n_perturb_types):
            np.random.seed(100 + p_type)
            delta = np.random.randn(d_model)
            delta = delta / np.linalg.norm(delta) * 2.0  # eps=2.0
            
            # 注入到embedding层
            embed_layer = model.get_input_embeddings()
            inputs_embeds_clean = embed_layer(input_ids).detach().clone()
            inputs_embeds_perturbed = inputs_embeds_clean.clone()
            inputs_embeds_perturbed[0, -1, :] += torch.tensor(delta, dtype=inputs_embeds_perturbed.dtype, device=device)
            
            # 生成
            with torch.no_grad():
                gen_ids = model.generate(
                    inputs_embeds=inputs_embeds_perturbed,
                    max_new_tokens=n_gen_steps,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gen_token_ids = gen_ids[0].cpu().numpy()
            
            # 重新forward获取hidden states
            gen_attention_mask = torch.ones_like(gen_ids)
            with torch.no_grad():
                out = model(input_ids=gen_ids, attention_mask=gen_attention_mask,
                            output_hidden_states=True, output_attentions=True)
            
            # 收集统计量
            gen_tokens = [tokenizer.decode([t]) for t in gen_token_ids]
            short_tokens = sum(1 for t in gen_tokens if len(t.strip()) <= 3)
            
            # Layer norms (采样几个关键层)
            layer_norms = {}
            for li in [0, info.n_layers//4, info.n_layers//2, 3*info.n_layers//4, info.n_layers-1]:
                if li < len(out.hidden_states):
                    layer_norms[li] = float(out.hidden_states[li][0, -1, :].float().norm().cpu())
            
            # Attention entropy profile
            attn_entropies = []
            if out.attentions:
                for li in range(min(5, len(out.attentions))):
                    if out.attentions[li] is not None:
                        attn = out.attentions[li][0].float().cpu().numpy()
                        last_pos = attn.shape[2] - 1
                        mean_attn = attn.mean(axis=0)
                        last_attn = mean_attn[last_pos, :]
                        ent = -np.sum(last_attn * np.log(last_attn + 1e-10))
                        max_ent = np.log(last_pos + 1) if last_pos > 0 else 1
                        attn_entropies.append(float(ent / max_ent))
            
            rollout_data.append({
                'perturb_type': p_type,
                'gen_text': gen_text,
                'gen_length': len(gen_token_ids),
                'token_type_ratio': short_tokens / len(gen_token_ids) if len(gen_token_ids) > 0 else 0,
                'layer_norms': layer_norms,
                'attn_entropies': attn_entropies,
            })
        
        # === 计算统计收敛性 ===
        # 跨rollout的CV → 越低越收敛
        ttr_values = [rd['token_type_ratio'] for rd in rollout_data]
        ttr_cv = np.std(ttr_values) / (np.mean(ttr_values) + 1e-10)
        
        # 最终层norm的CV
        final_norms = [rd['layer_norms'].get(info.n_layers-1, 0) for rd in rollout_data]
        norm_cv = np.std(final_norms) / (np.mean(final_norms) + 1e-10)
        
        # Attention entropy profile的相似性
        # 每对rollout之间的entropy profile correlation
        ent_corrs = []
        for i in range(len(rollout_data)):
            for j in range(i+1, len(rollout_data)):
                e1 = rollout_data[i]['attn_entropies']
                e2 = rollout_data[j]['attn_entropies']
                if len(e1) > 1 and len(e2) > 1:
                    min_len = min(len(e1), len(e2))
                    corr = np.corrcoef(e1[:min_len], e2[:min_len])[0, 1]
                    ent_corrs.append(corr)
        
        avg_ent_corr = np.mean(ent_corrs) if ent_corrs else 0
        
        results[f"prompt{prompt_idx}"] = {
            'prompt': prompt[:40],
            'n_perturb_types': n_perturb_types,
            'token_type_ratio_cv': float(ttr_cv),
            'final_norm_cv': float(norm_cv),
            'attn_entropy_avg_corr': float(avg_ent_corr),
            'mean_ttr': float(np.mean(ttr_values)),
            'rollouts': rollout_data,
        }
        
        print(f"\n  Prompt {prompt_idx}: '{prompt[:40]}...'")
        print(f"    TTR CV: {ttr_cv:.4f} ({'CONVERGED' if ttr_cv < 0.1 else 'DIVERGED'})")
        print(f"    Final norm CV: {norm_cv:.4f}")
        print(f"    Attn entropy avg corr: {avg_ent_corr:.4f}")
    
    # === 跨Prompt汇总 ===
    print("\n  === Cross-Prompt Statistical Attractor Summary ===")
    all_ttr_cv = [results[k]['token_type_ratio_cv'] for k in results]
    all_norm_cv = [results[k]['final_norm_cv'] for k in results]
    all_ent_corr = [results[k]['attn_entropy_avg_corr'] for k in results]
    
    print(f"  TTR CV (mean): {np.mean(all_ttr_cv):.4f} → "
          f"{'STATISTICAL ATTRACTOR ★' if np.mean(all_ttr_cv) < 0.1 else 'Weak attractor' if np.mean(all_ttr_cv) < 0.3 else 'No attractor'}")
    print(f"  Final norm CV (mean): {np.mean(all_norm_cv):.4f}")
    print(f"  Attn entropy corr (mean): {np.mean(all_ent_corr):.4f}")
    
    return results


# ============================================================
# Exp 4: Second-Order Perturbation Propagation
# ============================================================
def exp4_second_order_propagation(model, tokenizer, model_name):
    """
    核心问题: 扰动协方差矩阵如何传播?
    
    Phase 151发现cos(δ,δ')→0 — 这是一阶信息丢失
    但二阶统计量(协方差矩阵)可能保留结构!
    
    方法:
    1. 对同一输入, 注入N种扰动
    2. 在每层收集所有扰动响应 δ^(ℓ)
    3. 分析 δ^(ℓ) 矩阵的SVD结构
    4. 关键: δ^(ℓ) 的主成分方向是否与 δ^(0) 的主成分方向相关?
    
    如果:
    - cos(δ^(ℓ)_i, δ^(0)_i) → 0 (一阶丢失)
    - 但 PCA(δ^(ℓ)) 的主方向 与 PCA(δ^(0)) 的主方向相关 (二阶保留)
    → "方向丢失≠信息丢失" 被确认!
    """
    print("\n" + "="*60)
    print("Exp 4: Second-Order Perturbation Propagation")
    print("核心: cos→0后, 协方差结构是否保留?")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 6)))
    sample_layers = sorted(set(sample_layers + [n_layers]))
    
    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    last_pos = input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    clean_hs = out_clean.hidden_states
    
    n_perturbations = 200  # 大样本!
    
    # 收集各层的扰动响应
    delta_at_layer = {}
    for li in sample_layers:
        delta_at_layer[li] = []
    
    layers = get_layers(model)
    
    for p_idx in range(n_perturbations):
        np.random.seed(200 + p_idx)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta) * EPSILON
        
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        
        def make_hook(pos, delta_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return out
            return hook
        
        hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]
        
        try:
            with torch.no_grad():
                out_p = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            
            for li in sample_layers:
                p_vec = out_p.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                c_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_at_layer[li].append(p_vec - c_vec)
        except:
            pass
        
        for h in hooks:
            h.remove()
    
    # === 分析二阶结构 ===
    print("\n  === Second-Order Structure Analysis ===")
    
    # 对每层做PCA
    pca_results = {}
    for li in sample_layers:
        delta_matrix = np.array(delta_at_layer[li])  # [n_pert, d_model]
        if delta_matrix.shape[0] < 10:
            continue
        
        # 中心化
        delta_centered = delta_matrix - delta_matrix.mean(axis=0)
        
        # PCA via SVD
        try:
            U, s, Vt = np.linalg.svd(delta_centered, full_matrices=False)
            
            # 有效秩
            total_energy = np.sum(s ** 2)
            cumulative = np.cumsum(s ** 2)
            k50 = np.searchsorted(cumulative, 0.50 * total_energy) + 1
            k90 = np.searchsorted(cumulative, 0.90 * total_energy) + 1
            k95 = np.searchsorted(cumulative, 0.95 * total_energy) + 1
            
            # Top主成分
            top_pcs = Vt[:min(10, Vt.shape[0]), :]  # [10, d_model]
            
            pca_results[li] = {
                'k50': int(k50),
                'k90': int(k90),
                'k95': int(k95),
                'spectrum': s[:20].tolist(),
                'top_pcs': top_pcs,
            }
        except:
            continue
    
    # === 关键分析: PCA主方向的跨层相关性 ===
    # 如果Lℓ的PC1方向 与 L0的PC1方向相关 → 二阶结构保留!
    print("\n  --- PCA Principal Direction Correlation Across Layers ---")
    
    if 0 in pca_results:
        pcs_0 = pca_results[0]['top_pcs']  # [10, d_model]
        
        for li in sample_layers:
            if li not in pca_results:
                continue
            pcs_l = pca_results[li]['top_pcs']
            
            # PC1相关: cos(PC1_ℓ, PC1_0)
            pc1_corr = abs(float(np.dot(pcs_l[0], pcs_0[0])))
            
            # Subspace overlap: 前5个PC的子空间重叠
            # 用投影矩阵计算
            n_sub = min(5, pcs_0.shape[0], pcs_l.shape[0])
            if n_sub > 0:
                Q0 = pcs_0[:n_sub].T @ pcs_0[:n_sub]  # 投影矩阵
                Ql = pcs_l[:n_sub].T @ pcs_l[:n_sub]
                # 子空间重叠 = trace(Q0 @ Ql) / n_sub
                overlap = np.trace(Q0 @ Ql) / n_sub
            else:
                overlap = 0
            
            print(f"    L{li:>3d}: PC1_corr={pc1_corr:.4f}, subspace_overlap={overlap:.4f}, "
                  f"rank(50%)={pca_results[li]['k50']}, rank(90%)={pca_results[li]['k90']}")
    
    # === 对比: 一阶(cos) vs 二阶(subspace overlap) ===
    print("\n  --- First-Order vs Second-Order Decay ---")
    if 0 in pca_results:
        pcs_0 = pca_results[0]['top_pcs']
        for li in sample_layers:
            if li not in pca_results:
                continue
            
            # 一阶: 平均cos(δ^(ℓ), δ^(0))
            cos_values = []
            delta_0 = np.array(delta_at_layer[0])
            delta_l = np.array(delta_at_layer[li])
            for p in range(min(50, delta_0.shape[0], delta_l.shape[0])):
                n0 = np.linalg.norm(delta_0[p])
                nl = np.linalg.norm(delta_l[p])
                if n0 > 1e-10 and nl > 1e-10:
                    cos_values.append(float(np.dot(delta_0[p], delta_l[p]) / (n0 * nl)))
            avg_cos = np.mean(cos_values) if cos_values else 0
            
            # 二阶: subspace overlap
            n_sub = min(5, pcs_0.shape[0], pca_results[li]['top_pcs'].shape[0])
            if n_sub > 0:
                Q0 = pcs_0[:n_sub].T @ pcs_0[:n_sub]
                Ql = pca_results[li]['top_pcs'][:n_sub].T @ pca_results[li]['top_pcs'][:n_sub]
                overlap = np.trace(Q0 @ Ql) / n_sub
            else:
                overlap = 0
            
            # 判读
            status = "2ND-ORDER PRESERVED ★" if overlap > 0.3 and avg_cos < 0.1 else \
                     "1st&2nd both decayed" if overlap < 0.1 else "mixed"
            
            print(f"    L{li:>3d}: cos(1st)={avg_cos:.4f}, overlap(2nd)={overlap:.4f} [{status}]")
    
    return {
        'pca_results': {li: {k: v for k, v in data.items() if k != 'top_pcs'}
                        for li, data in pca_results.items()},
        'sample_layers': sample_layers,
    }


# ============================================================
# Exp 5: Cross-Model — 训练 vs 随机初始化
# ============================================================
def exp5_random_model_comparison(model_name):
    """
    核心问题: 哪些性质是训练产生的? 哪些是架构固有的?
    
    方法: 用同架构随机初始化模型, 运行相同实验
    训练模型 - 随机模型 = "语言结构"
    
    注意: 由于GPU内存限制, 我们不能同时加载两个模型
    所以用统计方法: 用随机矩阵模拟随机初始化模型的行为
    
    替代方法:
    1. 用随机正交矩阵模拟层权重
    2. 用随机高斯矩阵模拟W_U
    3. 比较"几何基线"(Phase 150)与"真实模型"的差异
    """
    print("\n" + "="*60)
    print("Exp 5: Trained vs Random Model Comparison")
    print("核心: 哪些性质是训练产生的?")
    print("="*60)
    
    # 方法: 对已加载的模型, 分析其权重结构
    # 与随机矩阵对比
    
    # 由于不能同时加载两个模型, 我们用以下替代方法:
    # 1. 分析模型权重的统计性质(是否偏离随机矩阵?)
    # 2. 用Phase 150的几何基线作为"随机模型"的代理
    
    # 简化: 比较W_U的SVD谱与随机矩阵的Marchenko-Pastur分布
    
    print("\n  --- W_U Spectrum vs Random Matrix ---")
    print("  (Using Marchenko-Pastur distribution as null hypothesis)")
    
    # 这个分析不需要模型, 可以用之前保存的W_U数据
    # 但我们已经在其他实验中加载了模型, 所以直接分析
    
    print("\n  NOTE: Full random model comparison requires loading a separate model.")
    print("  This is deferred to a follow-up experiment due to GPU memory constraints.")
    print("  The geometric baselines from Phase 150 serve as the 'random model' proxy.")
    
    return {"note": "Deferred to follow-up due to GPU constraints"}


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 152: Statistical Language Dynamics")
    print(f"Model: {model_name}")
    print(f"Time: {timestamp}")

    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Mode: {'8bit' if use_8bit else 'bfloat16'}")

    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")

    # 运行实验
    print("\n" + "#"*60)
    print("# Running Experiments")
    print("#"*60)

    # Exp 1: MI Flow
    exp1_results = exp1_mi_flow(model, tokenizer, model_name)
    
    # Exp 2: Logit Boundary
    exp2_results = exp2_logit_boundary(model, tokenizer, model_name)
    
    # Exp 3: Statistical Attractor
    exp3_results = exp3_statistical_attractor(model, tokenizer, model_name)
    
    # Exp 4: Second-Order Propagation
    exp4_results = exp4_second_order_propagation(model, tokenizer, model_name)
    
    # Exp 5: Cross-model (deferred)
    exp5_results = exp5_random_model_comparison(model_name)

    # 保存结果
    all_results = {
        "phase": 152,
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "exp1_mi_flow": {
            "linear_r2": exp1_results['linear_r2'],
            "cov_rank": exp1_results['cov_rank'],
            "cov_top_eigenvalue_ratio": exp1_results['cov_top_eigenvalue_ratio'],
            "delta_correlation_with_input": exp1_results['delta_correlation_with_input'],
        },
        "exp2_logit_boundary": exp2_results,
        "exp3_statistical_attractor": {
            k: {kk: vv for kk, vv in v.items() if kk != 'rollouts'}
            for k, v in exp3_results.items()
        },
        "exp4_second_order": exp4_results,
        "exp5_cross_model": exp5_results,
    }

    result_file = OUTPUT_DIR / f"phase152_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
