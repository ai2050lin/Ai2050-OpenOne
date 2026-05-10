"""
Phase 114: Marchenko-Pastur生死检验 + 传播算子 + 跨模型普适性
============================================================
核心目标:
1. 生死检验: PR≈3是真信号还是有限样本噪声? 用Marchenko-Pastur定律验证
2. 传播算子: 翻译扰动如何在层间传播? 是否存在"翻译放大模式"?
3. 跨模型验证: GLM4/DeepSeek7B是否也有MLP主导+维度塌缩?

关键理论背景:
- Marchenko-Pastur定律: 对N×P随机矩阵C = X^T X / N (X为N×P标准正态),
  特征值λ的理论分布为: p(λ) = sqrt((λ_+ - λ)(λ - λ_-)) / (2πλσ²)
  其中 λ_± = σ²(1 ± sqrt(P/N))²
- 超过λ_+的特征值 = 真信号, 在[λ_-, λ_+]内的 = 噪声
- 这是最严格的有限样本校正方法

用户批判的核心要点:
- PR≈3可能只是有限样本协方差的伪低秩结构
- 100个样本在9728维空间中, 随机矩阵也会有低维假象
- 必须区分信号谱和噪声谱
"""

import os, sys, json, gc, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import defaultdict

# 添加路径以使用model_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model

# ============================================================
# 设置
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 翻译词对 - 扩大到150个以获得更稳定的协方差估计
WORD_PAIRS = [
    # 自然
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鼠", "mouse"),
    # 物质
    ("水", "water"), ("火", "fire"), ("土", "earth"), ("风", "wind"), ("金", "gold"),
    ("木", "wood"), ("铁", "iron"), ("石", "stone"), ("沙", "sand"), ("冰", "ice"),
    # 天体
    ("月", "moon"), ("星", "star"), ("云", "cloud"), ("雨", "rain"), ("雪", "snow"),
    ("日", "sun"), ("天", "sky"), ("海", "sea"), ("河", "river"), ("山", "mountain"),
    # 颜色
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"), ("黑", "black"),
    ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"), ("棕", "brown"), ("粉", "pink"),
    # 身体
    ("手", "hand"), ("足", "foot"), ("目", "eye"), ("耳", "ear"), ("口", "mouth"),
    ("心", "heart"), ("头", "head"), ("骨", "bone"), ("血", "blood"), ("发", "hair"),
    # 人类
    ("父", "father"), ("母", "mother"), ("子", "son"), ("女", "daughter"), ("友", "friend"),
    ("王", "king"), ("师", "teacher"), ("医", "doctor"), ("兵", "soldier"), ("农", "farmer"),
    # 抽象
    ("爱", "love"), ("恨", "hate"), ("善", "good"), ("恶", "evil"), ("真", "truth"),
    ("美", "beauty"), ("智", "wisdom"), ("力", "power"), ("光", "light"), ("影", "shadow"),
    # 动作
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("吃", "eat"), ("喝", "drink"),
    ("看", "see"), ("听", "hear"), ("说", "say"), ("写", "write"), ("读", "read"),
    # 时间
    ("年", "year"), ("月", "month"), ("日", "day"), ("时", "hour"), ("分", "minute"),
    # 食物
    ("米", "rice"), ("茶", "tea"), ("酒", "wine"), ("肉", "meat"), ("盐", "salt"),
    # 交通
    ("车", "car"), ("船", "ship"), ("路", "road"), ("桥", "bridge"), ("门", "door"),
    # 自然2
    ("花", "flower"), ("草", "grass"), ("树", "tree"), ("叶", "leaf"), ("根", "root"),
    # 情感
    ("喜", "joy"), ("怒", "anger"), ("哀", "sorrow"), ("惧", "fear"), ("思", "thought"),
    # 社会制度
    ("法", "law"), ("国", "country"), ("城", "city"), ("家", "home"), ("书", "book"),
    # 科技
    ("电", "electricity"), ("网", "network"), ("数", "number"), ("算", "compute"), ("器", "device"),
    # 更多自然
    ("湖", "lake"), ("岛", "island"), ("春", "spring"), ("夏", "summer"), ("秋", "autumn"),
    ("冬", "winter"), ("晨", "morning"), ("暮", "dusk"), ("雷", "thunder"), ("雾", "fog"),
    # 更多动物
    ("龙", "dragon"), ("蛇", "snake"), ("虎", "tiger"), ("鹿", "deer"), ("兔", "rabbit"),
    # 更多抽象
    ("道", "way"), ("德", "virtue"), ("礼", "ritual"), ("义", "justice"), ("信", "trust"),
    # 器物
    ("剑", "sword"), ("笔", "pen"), ("琴", "lute"), ("画", "painting"), ("棋", "chess"),
    # 建筑材料
    ("砖", "brick"), ("瓦", "tile"), ("丝", "silk"), ("布", "cloth"), ("纸", "paper"),
]

# 去重
seen_zh = set()
UNIQUE_PAIRS = []
for zh, en in WORD_PAIRS:
    if zh not in seen_zh:
        seen_zh.add(zh)
        UNIQUE_PAIRS.append((zh, en))
WORD_PAIRS = UNIQUE_PAIRS[:150]

# 采样层
SAMPLE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]


# ============================================================
# Marchenko-Pastur定律
# ============================================================
def marchenko_pastur_bounds(N, P, sigma2=1.0):
    """
    计算Marchenko-Pastur分布的上下界
    
    对于N×P矩阵X (N个样本, P个特征), 协方差C = X^T @ X / N
    当X的元素iid ~ N(0, sigma2)时, C的特征值分布为MP分布
    
    λ_± = sigma2 * (1 ± sqrt(P/N))²
    
    Args:
        N: 样本数
        P: 特征维度
        sigma2: 噪声方差
    
    Returns:
        (lambda_minus, lambda_plus, lambda_max_expected)
    """
    ratio = P / N  # 注意: N < P 时, 会有零特征值
    lambda_minus = sigma2 * (1 - np.sqrt(ratio)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(ratio)) ** 2
    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(lam, N, P, sigma2=1.0):
    """MP分布的概率密度函数"""
    ratio = P / N
    lam_minus = sigma2 * (1 - np.sqrt(ratio)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(ratio)) ** 2
    
    if lam < lam_minus or lam > lam_plus:
        return 0.0
    
    return (np.sqrt((lam_plus - lam) * (lam - lam_minus)) / 
            (2 * np.pi * sigma2 * lam)) + 1e-30


def analyze_eigenvalue_spectrum(eigenvalues, N_samples, P_dim, sigma2_est=None):
    """
    分析特征值谱, 区分信号与噪声
    
    Args:
        eigenvalues: 协方差矩阵的特征值 (降序排列)
        N_samples: 样本数
        P_dim: 特征维度
        sigma2_est: 噪声方差估计 (若None则从中位数特征值估计)
    
    Returns:
        dict: MP分析结果
    """
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
    
    # 估计噪声方差: 使用中位数附近的特征值
    if sigma2_est is None:
        # 方法: 用最小特征值的中位数估计噪声水平
        # 在N << P时, 大部分特征值在噪声谱中
        n_noise = max(len(eigenvalues) - 10, len(eigenvalues) // 2)
        sigma2_est = np.median(eigenvalues[-n_noise:]) if n_noise > 0 else np.median(eigenvalues)
    
    # MP边界
    lam_minus, lam_plus = marchenko_pastur_bounds(N_samples, P_dim, sigma2_est)
    
    # 判断哪些特征值超过MP上界
    signal_eigs = eigenvalues[eigenvalues > lam_plus]
    noise_eigs = eigenvalues[eigenvalues <= lam_plus]
    
    # 真实维度 = 超过MP上界的特征值数
    true_dim = len(signal_eigs)
    
    # 信号解释方差比
    total_var = np.sum(eigenvalues)
    signal_var = np.sum(signal_eigs)
    signal_ratio = signal_var / total_var if total_var > 0 else 0
    
    # Participation Ratio (基于信号)
    if len(signal_eigs) > 0:
        pr_signal = (np.sum(signal_eigs))**2 / np.sum(signal_eigs**2)
    else:
        pr_signal = 0
    
    # PR (基于全部特征值)
    pr_full = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2) if total_var > 0 else 0
    
    return {
        "n_samples": N_samples,
        "n_features": P_dim,
        "sigma2_estimated": float(sigma2_est),
        "mp_lambda_minus": float(lam_minus),
        "mp_lambda_plus": float(lam_plus),
        "true_dimensionality": true_dim,
        "signal_variance_ratio": float(signal_ratio),
        "pr_signal": float(pr_signal),
        "pr_full": float(pr_full),
        "top10_eigenvalues": [float(x) for x in eigenvalues[:10]],
        "top10_explained_ratio": [float(x/total_var) for x in eigenvalues[:10]],
        "eigenvalue_at_mp_plus": float(eigenvalues[min(true_dim, len(eigenvalues)-1)]) if len(eigenvalues) > 0 else 0,
        "gap_to_noise": float(eigenvalues[true_dim-1] / lam_plus) if true_dim > 0 and lam_plus > 0 else 0,
    }


# ============================================================
# 数据收集: 隐藏状态 + MLP/Attention输出
# ============================================================
def collect_hidden_states(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    """收集各层隐藏状态 (翻译 vs 中文)"""
    layers = get_layers(model)
    
    # 存储: {层: {zh: [样本×d_model], trans: [样本×d_model]}}
    all_states = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                layer_acts = {}
                hooks = []
                
                def make_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            layer_acts[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            layer_acts[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    h = layer.register_forward_hook(make_hook(l))
                    hooks.append(h)
                
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in layer_acts:
                    all_states[task][l].append(layer_acts[l])
        
        print(f"  [collect] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)} 词对")
    
    # 转numpy
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_states[task]:
            result[task][l] = np.array(all_states[task][l])
    
    d_model = result['zh'][0].shape[1] if 0 in result['zh'] else 2560
    return result, d_model


def collect_mlp_attn_outputs(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    """收集MLP和Attention的输出 (用于算子分解)"""
    layers = get_layers(model)
    
    all_outs = {
        'mlp': {'zh': defaultdict(list), 'trans': defaultdict(list)},
        'attn': {'zh': defaultdict(list), 'trans': defaultdict(list)},
    }
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mlp_outs = {}
                attn_outs = {}
                hooks = []
                
                def make_mlp_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            mlp_outs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            mlp_outs[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                def make_attn_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            attn_outs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            attn_outs[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
                    if hasattr(layer, 'self_attn'):
                        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
                
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in mlp_outs:
                    all_outs['mlp'][task][l].append(mlp_outs[l])
                for l in attn_outs:
                    all_outs['attn'][task][l].append(attn_outs[l])
        
        print(f"  [collect_mlp_attn] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)}")
    
    # 转numpy
    result = {}
    for comp in ['mlp', 'attn']:
        result[comp] = {}
        for task in ['zh', 'trans']:
            result[comp][task] = {}
            for l in all_outs[comp][task]:
                result[comp][task][l] = np.array(all_outs[comp][task][l])
    
    return result


# ============================================================
# Exp 1: Marchenko-Pastur生死检验
# ============================================================
def exp1_marchenko_pastur(states, d_model):
    """
    生死检验: 翻译差分的PR≈3是真信号还是有限样本噪声?
    
    方法:
    1. 计算翻译差分(trans - zh)的协方差矩阵
    2. 与Marchenko-Pastur零假设比较
    3. 超过MP上界的特征值 = 真信号
    """
    print("\n" + "="*70)
    print("Exp 1: Marchenko-Pastur生死检验 — PR≈3是真信号还是噪声?")
    print("="*70)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in states['zh'] or l not in states['trans']:
            continue
        
        zh_data = states['zh'][l]  # [N, d_model]
        trans_data = states['trans'][l]  # [N, d_model]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        # 翻译差分
        diffs = trans_data - zh_data  # [N, P]
        
        # 中心化
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        
        # 协方差矩阵 (使用SVD避免直接计算大矩阵)
        # C = diffs_centered^T @ diffs_centered / N
        # SVD: diffs_centered = U S V^T → eigenvalues of C = S²/N
        
        # 只需要计算小矩阵: (diffs_centered @ diffs_centered^T) / N 是 N×N 矩阵
        # 它的非零特征值 = C的非零特征值
        small_cov = (diffs_centered @ diffs_centered.T) / N  # [N, N]
        eigenvalues_small = np.linalg.eigvalsh(small_cov)[::-1]  # 降序
        
        # 但还需要 P-N 个零特征值
        # 对于N < P, C有P-N个零特征值
        # 非零特征值 = eigenvalues_small (N个)
        all_eigenvalues = np.concatenate([eigenvalues_small, np.zeros(max(P - N, 0))])
        all_eigenvalues = np.sort(all_eigenvalues)[::-1]
        
        # 估计噪声方差
        # 方法1: 从数据的逐元素方差估计
        # 对差分数据, 每个元素的方差
        element_var = np.var(diffs_centered)  # 全局方差
        # 方法2: 从最小特征值估计
        sigma2_from_min = np.median(eigenvalues_small[-max(N//2, 1):]) if N > 2 else eigenvalues_small[-1]
        
        # MP分析 (使用两种sigma2估计)
        mp_result_elem = analyze_eigenvalue_spectrum(eigenvalues_small, N, P, sigma2_est=element_var)
        mp_result_min = analyze_eigenvalue_spectrum(eigenvalues_small, N, P, sigma2_est=sigma2_from_min)
        
        # 随机基线: 同分布随机矩阵
        # 生成N×P随机矩阵, 元素从N(0, element_var)采样
        n_null = 50
        null_dims = []
        null_prs = []
        null_signal_ratios = []
        
        for _ in range(n_null):
            rand_data = np.random.randn(N, P) * np.sqrt(element_var)
            rand_small_cov = (rand_data @ rand_data.T) / N
            rand_eigs = np.linalg.eigvalsh(rand_small_cov)[::-1]
            rand_result = analyze_eigenvalue_spectrum(rand_eigs, N, P, sigma2_est=element_var)
            null_dims.append(rand_result["true_dimensionality"])
            null_prs.append(rand_result["pr_signal"])
            null_signal_ratios.append(rand_result["signal_variance_ratio"])
        
        results[f"L{l}"] = {
            "n_samples": N,
            "n_features": P,
            "element_variance": float(element_var),
            "sigma2_from_min_eig": float(sigma2_from_min),
            "mp_analysis_element_var": mp_result_elem,
            "mp_analysis_min_eig": mp_result_min,
            "null_baseline": {
                "n_null": n_null,
                "true_dim_mean": float(np.mean(null_dims)),
                "true_dim_std": float(np.std(null_dims)),
                "true_dim_max": int(np.max(null_dims)),
                "pr_signal_mean": float(np.mean(null_prs)),
                "pr_signal_std": float(np.std(null_prs)),
                "signal_ratio_mean": float(np.mean(null_signal_ratios)),
                "signal_ratio_std": float(np.std(null_signal_ratios)),
            },
            "top10_eigenvalues": [float(x) for x in eigenvalues_small[:10]],
            "top10_explained_ratio": [float(x/np.sum(eigenvalues_small)) for x in eigenvalues_small[:10]],
            "pr_full": float(mp_result_elem["pr_full"]),
            "pr_signal": float(mp_result_elem["pr_signal"]),
            "true_dim": mp_result_elem["true_dimensionality"],
            "signal_variance_ratio": float(mp_result_elem["signal_variance_ratio"]),
            "mp_lambda_plus_elem": float(mp_result_elem["mp_lambda_plus"]),
            "gap_to_noise": float(mp_result_elem["gap_to_noise"]),
        }
        
        # 判定
        real_dim = mp_result_elem["true_dimensionality"]
        null_dim_mean = np.mean(null_dims)
        null_dim_std = np.std(null_dims)
        
        if null_dim_std > 0:
            z_dim = (real_dim - null_dim_mean) / null_dim_std
        else:
            z_dim = 0 if real_dim == null_dim_mean else float('inf')
        
        verdict = "SIGNAL_DETECTED" if z_dim > 2.0 else "CONSISTENT_WITH_NULL"
        
        results[f"L{l}"]["verdict"] = verdict
        results[f"L{l}"]["z_dimensionality"] = float(z_dim)
        
        print(f"  L{l}: N={N}, P={P}, PR_full={mp_result_elem['pr_full']:.1f}, "
              f"PR_signal={mp_result_elem['pr_signal']:.1f}, "
              f"true_dim={real_dim}, null_dim={null_dim_mean:.1f}±{null_dim_std:.1f}, "
              f"z={z_dim:.2f}, signal_ratio={mp_result_elem['signal_variance_ratio']:.3f}, "
              f"verdict={verdict}")
        
        # 详细输出前10个特征值
        print(f"    Top-10 eigenvalues: {[f'{x:.4f}' for x in eigenvalues_small[:10]]}")
        print(f"    MP lambda_plus(element_var): {mp_result_elem['mp_lambda_plus']:.4f}")
        print(f"    MP lambda_plus(min_eig): {mp_result_min['mp_lambda_plus']:.4f}")
    
    return results


# ============================================================
# Exp 2: 传播算子分析 — 翻译扰动如何跨层传播
# ============================================================
def exp2_propagation_operator(states, d_model):
    """
    传播算子分析:
    翻译扰动(trans - zh)在层间如何传播?
    
    关键问题:
    1. 差分信号是被放大还是衰减?
    2. 是否存在"翻译放大模式"?
    3. 传播是否线性? (通过residual stream近似)
    """
    print("\n" + "="*70)
    print("Exp 2: 传播算子分析 — 翻译扰动如何跨层传播")
    print("="*70)
    
    results = {}
    sorted_layers = sorted([l for l in states['zh'].keys() if l in states['trans']])
    
    # 1. 差分范数演化
    diff_norms = {}
    for l in sorted_layers:
        zh_data = states['zh'][l]
        trans_data = states['trans'][l]
        diffs = trans_data - zh_data
        diff_norms[l] = np.linalg.norm(diffs, axis=1)  # [N]
    
    # 2. 差分方向稳定性 (相邻层差分方向的相关性)
    direction_corr = {}
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        d1 = states['trans'][l1] - states['zh'][l1]  # [N, P]
        d2 = states['trans'][l2] - states['zh'][l2]  # [N, P]
        
        # 逐样本cosine相似度
        cos_sims = []
        for s in range(d1.shape[0]):
            n1, n2 = np.linalg.norm(d1[s]), np.linalg.norm(d2[s])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_sims.append(np.dot(d1[s], d2[s]) / (n1 * n2))
        
        direction_corr[f"L{l1}_L{l2}"] = {
            "mean_cos": float(np.mean(cos_sims)),
            "std_cos": float(np.std(cos_sims)),
            "median_cos": float(np.median(cos_sims)),
            "p10_cos": float(np.percentile(cos_sims, 10)),
            "p90_cos": float(np.percentile(cos_sims, 90)),
        }
    
    # 3. 线性传播算子估计
    # h_{l+1} ≈ A_l @ h_l + b_l (对翻译差分: Δ_{l+1} ≈ A_l @ Δ_l)
    # 最小二乘: A_l = Δ_{l+1}^T @ Δ_l @ (Δ_l^T @ Δ_l)^{-1}
    # 但在N << P时, 直接求解不稳定
    # 替代: 计算投影比 = ||P_{Δ_l} Δ_{l+1}||² / ||Δ_{l+1}||²
    propagation_ratios = {}
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        d1 = states['trans'][l1] - states['zh'][l1]  # [N, P]
        d2 = states['trans'][l2] - states['zh'][l2]  # [N, P]
        
        # Δ_{l+1}在Δ_l行空间上的投影比
        # 行空间由d1的SVD左奇异向量U1张成
        # 投影比 = Σ(U1^T @ d2)^2 / Σ(d2)^2
        U1, S1, _ = np.linalg.svd(d1, full_matrices=False)  # U1: [N, N], S1: [min(N,P)]
        
        proj_coeffs = U1.T @ d2  # [N, P]... 不对
        # d2: [N, P], U1: [N, N]
        # U1^T @ d2: [N, P]
        proj = U1 @ (U1.T @ d2)  # [N, P] — d2在d1行空间的投影
        
        proj_energy = np.sum(proj ** 2)
        total_energy = np.sum(d2 ** 2)
        propagation_ratio = proj_energy / total_energy if total_energy > 0 else 0
        
        # 反向: Δ_l在Δ_{l+1}行空间上的投影
        U2, S2, _ = np.linalg.svd(d2, full_matrices=False)
        proj_rev = U2 @ (U2.T @ d1)
        rev_ratio = np.sum(proj_rev ** 2) / np.sum(d1 ** 2) if np.sum(d1 ** 2) > 0 else 0
        
        propagation_ratios[f"L{l1}_L{l2}"] = {
            "forward_ratio": float(propagation_ratio),  # Δ_{l+1}有多少可以由Δ_l预测
            "reverse_ratio": float(rev_ratio),  # Δ_l有多少可以由Δ_{l+1}预测
            "norm_ratio": float(np.mean(diff_norms[l2]) / np.mean(diff_norms[l1])),
        }
    
    # 4. 主成分传播分析
    # 在每层做PCA, 看主成分在层间如何演化
    pc_propagation = {}
    prev_U = None
    prev_layer = None
    
    for l in sorted_layers:
        d = states['trans'][l] - states['zh'][l]  # [N, P]
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        
        if prev_U is not None:
            # 前k个主成分的重叠
            k = min(5, U.shape[1])
            overlap = np.sum((U[:, :k].T @ prev_U[:, :k]) ** 2) / k
            pc_propagation[f"L{prev_layer}_L{l}"] = {
                "pc_overlap_k5": float(overlap),
                "singular_value_ratio": [float(S[i] / prev_S[i]) if prev_S[i] > 0 else 0 for i in range(min(5, len(S)))],
            }
        
        prev_U = U
        prev_S = S
        prev_layer = l
    
    results = {
        "diff_norms": {f"L{l}": {"mean": float(np.mean(diff_norms[l])), "std": float(np.std(diff_norms[l]))} for l in sorted_layers},
        "direction_correlation": direction_corr,
        "propagation_ratios": propagation_ratios,
        "pc_propagation": pc_propagation,
    }
    
    # 输出关键结果
    print("\n  差分范数演化:")
    for l in sorted_layers:
        print(f"    L{l}: mean_norm={np.mean(diff_norms[l]):.4f} ± {np.std(diff_norms[l]):.4f}")
    
    print("\n  相邻层差分方向相关性:")
    for k, v in direction_corr.items():
        print(f"    {k}: cos={v['mean_cos']:.4f} ± {v['std_cos']:.4f}")
    
    print("\n  传播投影比 (Δ_{l+1}在Δ_l行空间上的投影):")
    for k, v in propagation_ratios.items():
        print(f"    {k}: forward={v['forward_ratio']:.4f}, reverse={v['reverse_ratio']:.4f}, norm_ratio={v['norm_ratio']:.4f}")
    
    print("\n  主成分传播:")
    for k, v in pc_propagation.items():
        print(f"    {k}: PC_overlap={v['pc_overlap_k5']:.4f}")
    
    return results


# ============================================================
# Exp 3: MLP差分的Marchenko-Pastur分析 (最重要!)
# ============================================================
def exp3_mlp_mp_analysis(mlp_attn_outs, d_model):
    """
    对MLP差分(trans_mlp - zh_mlp)做Marchenko-Pastur分析
    这是PR≈3最直接的检验
    
    Phase 113发现: L12/L18的MLP PR≈2.6-2.8
    关键问题: 这是否超出MP噪声谱?
    """
    print("\n" + "="*70)
    print("Exp 3: MLP差分的Marchenko-Pastur分析 — PR≈3生死检验")
    print("="*70)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in mlp_attn_outs['mlp']['zh'] or l not in mlp_attn_outs['mlp']['trans']:
            continue
        
        zh_mlp = mlp_attn_outs['mlp']['zh'][l]  # [N, d_model]
        trans_mlp = mlp_attn_outs['mlp']['trans'][l]  # [N, d_model]
        N = zh_mlp.shape[0]
        P = zh_mlp.shape[1]
        
        # MLP差分
        diffs = trans_mlp - zh_mlp  # [N, P]
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        
        # 协方差特征值
        small_cov = (diffs_centered @ diffs_centered.T) / N
        eigenvalues = np.linalg.eigvalsh(small_cov)[::-1]
        
        # 噪声方差估计
        element_var = np.var(diffs_centered)
        sigma2_min = np.median(eigenvalues[-max(N//2, 1):]) if N > 2 else eigenvalues[-1]
        
        # MP分析
        mp_result = analyze_eigenvalue_spectrum(eigenvalues, N, P, sigma2_est=element_var)
        
        # Null基线
        n_null = 50
        null_dims = []
        null_prs = []
        null_signal_ratios = []
        
        for _ in range(n_null):
            rand_data = np.random.randn(N, P) * np.sqrt(element_var)
            rand_small_cov = (rand_data @ rand_data.T) / N
            rand_eigs = np.linalg.eigvalsh(rand_small_cov)[::-1]
            rand_result = analyze_eigenvalue_spectrum(rand_eigs, N, P, sigma2_est=element_var)
            null_dims.append(rand_result["true_dimensionality"])
            null_prs.append(rand_result["pr_signal"])
            null_signal_ratios.append(rand_result["signal_variance_ratio"])
        
        real_dim = mp_result["true_dimensionality"]
        null_dim_mean = np.mean(null_dims)
        null_dim_std = np.std(null_dims)
        z_dim = (real_dim - null_dim_mean) / null_dim_std if null_dim_std > 0 else (0 if real_dim == null_dim_mean else float('inf'))
        
        verdict = "SIGNAL_DETECTED" if z_dim > 2.0 else "CONSISTENT_WITH_NULL"
        
        # 注意力差分分析
        attn_signal = False
        if l in mlp_attn_outs['attn']['zh'] and l in mlp_attn_outs['attn']['trans']:
            zh_attn = mlp_attn_outs['attn']['zh'][l]
            trans_attn = mlp_attn_outs['attn']['trans'][l]
            attn_diffs = trans_attn - zh_attn
            attn_diffs_centered = attn_diffs - attn_diffs.mean(axis=0, keepdims=True)
            attn_var = np.var(attn_diffs_centered)
            attn_norm = np.mean(np.linalg.norm(attn_diffs, axis=1))
            mlp_norm = np.mean(np.linalg.norm(diffs, axis=1))
            mlp_attn_ratio = mlp_norm / attn_norm if attn_norm > 0 else float('inf')
        else:
            attn_var = 0
            mlp_attn_ratio = 0
        
        results[f"L{l}"] = {
            "n_samples": N,
            "n_features": P,
            "element_variance": float(element_var),
            "top10_eigenvalues": [float(x) for x in eigenvalues[:10]],
            "top10_explained_ratio": [float(x/np.sum(eigenvalues)) for x in eigenvalues[:10]],
            "mp_lambda_plus": float(mp_result["mp_lambda_plus"]),
            "true_dimensionality": real_dim,
            "pr_full": float(mp_result["pr_full"]),
            "pr_signal": float(mp_result["pr_signal"]),
            "signal_variance_ratio": float(mp_result["signal_variance_ratio"]),
            "gap_to_noise": float(mp_result["gap_to_noise"]),
            "null_baseline": {
                "n_null": n_null,
                "true_dim_mean": float(null_dim_mean),
                "true_dim_std": float(null_dim_std),
                "true_dim_max": int(np.max(null_dims)),
                "pr_signal_mean": float(np.mean(null_prs)),
                "pr_signal_std": float(np.std(null_prs)),
                "signal_ratio_mean": float(np.mean(null_signal_ratios)),
                "signal_ratio_std": float(np.std(null_signal_ratios)),
            },
            "z_dimensionality": float(z_dim),
            "verdict": verdict,
            "attn_var": float(attn_var),
            "mlp_attn_norm_ratio": float(mlp_attn_ratio),
        }
        
        print(f"  L{l}: N={N}, P={P}, PR_full={mp_result['pr_full']:.1f}, "
              f"PR_signal={mp_result['pr_signal']:.1f}, true_dim={real_dim}, "
              f"null_dim={null_dim_mean:.1f}±{null_dim_std:.1f}, z={z_dim:.2f}, "
              f"signal_ratio={mp_result['signal_variance_ratio']:.3f}, "
              f"gap={mp_result['gap_to_noise']:.2f}, verdict={verdict}")
        print(f"    Top-5 eigenvalues: {[f'{x:.6f}' for x in eigenvalues[:5]]}")
        print(f"    MP lambda_plus: {mp_result['mp_lambda_plus']:.6f}")
    
    return results


# ============================================================
# Exp 4: 跨模型关键指标对比 (需要单独运行每个模型)
# ============================================================
def exp4_cross_model_summary(current_model, current_results):
    """汇总当前模型的跨模型对比数据"""
    print("\n" + "="*70)
    print("Exp 4: 跨模型对比 — 汇总当前模型数据")
    print("="*70)
    
    # 提取关键指标
    summary = {
        "model": current_model,
        "key_findings": {
            "mlp_dominant_layers": [],
            "pr_collapse_layers": [],
            "true_signal_dims": {},
        }
    }
    
    # 从Exp 3结果中提取
    for layer_key, data in current_results.items():
        if data.get("verdict") == "SIGNAL_DETECTED":
            summary["key_findings"]["pr_collapse_layers"].append(layer_key)
        summary["key_findings"]["true_signal_dims"][layer_key] = data.get("true_dimensionality", 0)
    
    print(f"  模型: {current_model}")
    print(f"  检测到信号的层: {summary['key_findings']['pr_collapse_layers']}")
    print(f"  各层真实维度: {summary['key_findings']['true_signal_dims']}")
    
    return summary


# ============================================================
# Exp 5: Fisher信息几何近似 — 哪些方向真正影响输出
# ============================================================
def exp5_fisher_geometry(states, d_model, model, tokenizer, device):
    """
    Fisher几何近似: 翻译差分方向是否真正影响输出分布?
    
    方法:
    对每个翻译词对, 在各层隐藏状态上添加小扰动,
    观察输出logit的变化, 估计Fisher信息矩阵的近似
    
    简化: 使用有限差分估计Fisher度量
    """
    print("\n" + "="*70)
    print("Exp 5: Fisher几何近似 — 翻译方向是否影响输出分布?")
    print("="*70)
    
    layers = get_layers(model)
    results = {}
    
    # 选择少量测试对 (计算密集)
    test_pairs = WORD_PAIRS[:20]
    epsilon = 0.1  # 扰动幅度
    
    for l in SAMPLE_LAYERS[:5]:  # 只测5层 (计算量大)
        fisher_ratios = []
        
        for zh, en in test_pairs:
            trans_prompt = f"Translate the following Chinese word: {zh}"
            inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
            
            # 获取基线输出
            layer_out = {}
            hooks = []
            
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        layer_out[layer_idx] = output[0].detach().clone()
                    else:
                        layer_out[layer_idx] = output.detach().clone()
                return hook_fn
            
            for li, layer in enumerate(layers):
                if li == l:
                    hooks.append(layer.register_forward_hook(make_hook(li)))
            
            with torch.no_grad():
                base_logits = model(inputs["input_ids"]).logits[0, -1, :]
            
            for h in hooks:
                h.remove()
            
            if l not in layer_out:
                continue
            
            # 添加随机扰动, 看logit变化
            h_base = layer_out[l]
            delta_logit_random = []
            
            for _ in range(10):  # 10个随机方向
                random_dir = torch.randn_like(h_base)
                random_dir = random_dir / torch.norm(random_dir) * epsilon
                h_perturbed = h_base + random_dir
                
                # 注入扰动
                hooks2 = []
                injected = {}
                
                def make_inject_hook(layer_idx, perturbation):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].detach().clone()
                            out[0, -1, :] += perturbation[0, -1, :]
                            return (out,) + output[1:]
                        else:
                            out = output.detach().clone()
                            out[0, -1, :] += perturbation[0, -1, :]
                            return out
                    return hook_fn
                
                hooks2.append(layers[l].register_forward_hook(make_inject_hook(l, random_dir)))
                
                with torch.no_grad():
                    perturbed_logits = model(inputs["input_ids"]).logits[0, -1, :]
                
                for h2 in hooks2:
                    h2.remove()
                
                logit_change = torch.norm(perturbed_logits - base_logits).item()
                delta_logit_random.append(logit_change)
            
            # 翻译方向扰动
            # 获取中文和翻译的隐藏状态差
            zh_prompt = f"翻译以下中文词：{zh}"
            zh_inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
            
            zh_layer_out = {}
            hooks3 = []
            for li, layer in enumerate(layers):
                if li == l:
                    def make_hook2(layer_idx):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                zh_layer_out[layer_idx] = output[0].detach().clone()
                            else:
                                zh_layer_out[layer_idx] = output.detach().clone()
                        return hook_fn
                    hooks3.append(layer.register_forward_hook(make_hook2(li)))
            
            with torch.no_grad():
                _ = model(zh_inputs["input_ids"])
            
            for h3 in hooks3:
                h3.remove()
            
            if l in zh_layer_out and l in layer_out:
                # 翻译差分方向 (归一化到同样幅度)
                trans_dir = layer_out[l][0, -1, :] - zh_layer_out[l][0, -1, :]
                trans_dir_norm = torch.norm(trans_dir)
                if trans_dir_norm > 1e-10:
                    trans_dir = trans_dir / trans_dir_norm * epsilon
                    
                    hooks4 = []
                    hooks4.append(layers[l].register_forward_hook(make_inject_hook(l, trans_dir.unsqueeze(0).unsqueeze(0))))
                    
                    with torch.no_grad():
                        trans_logits = model(inputs["input_ids"]).logits[0, -1, :]
                    
                    for h4 in hooks4:
                        h4.remove()
                    
                    logit_change_trans = torch.norm(trans_logits - base_logits).item()
                    
                    # Fisher ratio: 翻译方向的logit变化 / 随机方向的logit变化
                    mean_random = np.mean(delta_logit_random)
                    fisher_ratio = logit_change_trans / mean_random if mean_random > 0 else 0
                    fisher_ratios.append(fisher_ratio)
            
            del inputs, zh_inputs
            gc.collect()
            torch.cuda.empty_cache()
        
        if fisher_ratios:
            results[f"L{l}"] = {
                "mean_fisher_ratio": float(np.mean(fisher_ratios)),
                "std_fisher_ratio": float(np.std(fisher_ratios)),
                "median_fisher_ratio": float(np.median(fisher_ratios)),
                "n_samples": len(fisher_ratios),
            }
            print(f"  L{l}: Fisher ratio = {np.mean(fisher_ratios):.3f} ± {np.std(fisher_ratios):.3f} "
                  f"(>1 means translation direction has more output impact than random)")
        else:
            results[f"L{l}"] = {"mean_fisher_ratio": 0, "std_fisher_ratio": 0, "n_samples": 0}
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["all", "1", "2", "3", "4", "5"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*70}")
    print(f"Phase 114: Marchenko-Pastur生死检验 + 传播算子 + 跨模型普适性")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    print(f"模型: {model_info.model_class}, {n_layers}层, d_model={d_model}, device={device}")
    
    # 收集数据
    print("\n--- 收集隐藏状态 ---")
    states, d_model = collect_hidden_states(model, tokenizer, device, n_layers, WORD_PAIRS)
    
    # 更新采样层 (只保留有数据的层)
    available_layers = sorted([l for l in states['zh'].keys() if l in states['trans']])
    global SAMPLE_LAYERS
    SAMPLE_LAYERS = [l for l in SAMPLE_LAYERS if l in available_layers]
    print(f"可用层: {SAMPLE_LAYERS}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # Exp 1: Marchenko-Pastur生死检验
    if args.exp in ["all", "1"]:
        print("\n--- Exp 1: Marchenko-Pastur生死检验 ---")
        exp1_result = exp1_marchenko_pastur(states, d_model)
        all_results["exp1_mp_hidden_states"] = exp1_result
        
        # 保存
        out_path = os.path.join(OUT_DIR, f"phase114_exp1_{model_name}_mp_hidden.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp1_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 2: 传播算子
    if args.exp in ["all", "2"]:
        print("\n--- Exp 2: 传播算子 ---")
        exp2_result = exp2_propagation_operator(states, d_model)
        all_results["exp2_propagation"] = exp2_result
        
        out_path = os.path.join(OUT_DIR, f"phase114_exp2_{model_name}_propagation.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp2_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 3: MLP差分的MP分析 (最关键!)
    if args.exp in ["all", "3"]:
        print("\n--- Exp 3: MLP差分MP分析 ---")
        mlp_attn_outs = collect_mlp_attn_outputs(model, tokenizer, device, n_layers, WORD_PAIRS)
        exp3_result = exp3_mlp_mp_analysis(mlp_attn_outs, d_model)
        all_results["exp3_mp_mlp"] = exp3_result
        
        out_path = os.path.join(OUT_DIR, f"phase114_exp3_{model_name}_mp_mlp.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp3_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 4: 跨模型汇总
    if args.exp in ["all", "4"]:
        print("\n--- Exp 4: 跨模型汇总 ---")
        if "exp3_mp_mlp" in all_results:
            exp4_result = exp4_cross_model_summary(model_name, all_results["exp3_mp_mlp"])
            all_results["exp4_cross_model"] = exp4_result
    
    # Exp 5: Fisher几何 (计算密集, 可选)
    if args.exp in ["all", "5"]:
        print("\n--- Exp 5: Fisher几何近似 ---")
        try:
            exp5_result = exp5_fisher_geometry(states, d_model, model, tokenizer, device)
            all_results["exp5_fisher"] = exp5_result
            
            out_path = os.path.join(OUT_DIR, f"phase114_exp5_{model_name}_fisher.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(exp5_result, f, indent=2, ensure_ascii=False)
            print(f"  保存到 {out_path}")
        except Exception as e:
            print(f"  Exp 5失败 (GPU内存?): {e}")
            all_results["exp5_fisher"] = {"error": str(e)}
    
    # 保存全部结果
    all_out_path = os.path.join(OUT_DIR, f"phase114_{model_name}_all_results.json")
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n全部结果保存到 {all_out_path}")
    
    # 释放模型
    release_model(model)
    print("Phase 114 完成!")
    
    # 打印最终总结
    print("\n" + "="*70)
    print("PHASE 114 核心结论")
    print("="*70)
    
    if "exp1_mp_hidden_states" in all_results:
        print("\nExp 1 - 隐藏状态MP检验:")
        for layer_key, data in all_results["exp1_mp_hidden_states"].items():
            td = data.get('true_dimensionality', data.get('true_dim', '?'))
            pr = data.get('pr_full', 0)
            sr = data.get('signal_variance_ratio', 0)
            z = data.get('z_dimensionality', 0)
            v = data.get('verdict', '?')
            print(f"  {layer_key}: true_dim={td}, PR_full={pr:.1f}, signal_ratio={sr:.3f}, z={z:.2f}, verdict={v}")
    
    if "exp3_mp_mlp" in all_results:
        print("\nExp 3 - MLP差分MP检验 (最关键!):")
        for layer_key, data in all_results["exp3_mp_mlp"].items():
            td = data.get('true_dimensionality', '?')
            pr_f = data.get('pr_full', 0)
            pr_s = data.get('pr_signal', 0)
            sr = data.get('signal_variance_ratio', 0)
            z = data.get('z_dimensionality', 0)
            v = data.get('verdict', '?')
            print(f"  {layer_key}: true_dim={td}, PR_full={pr_f:.1f}, PR_signal={pr_s:.1f}, signal_ratio={sr:.3f}, z={z:.2f}, verdict={v}")


if __name__ == "__main__":
    main()
