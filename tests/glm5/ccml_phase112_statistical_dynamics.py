"""
Phase 112: Computation Graph Statistical Dynamics — 从neuron中心到路径系综
=========================================================================

Phase 111用户批判(系统化):
  1. "neuron不是基本对象" — 应该研究path/route, 不是单个neuron
     同一neuron在翻译/推理/数学/coding中属于完全不同子图
     "translation neuron"不是translation unit, 而是translation route上的局部节点

  2. "缺少动力学" — 只有静态快照, 没有轨迹演化
     Transformer本质是conditional dynamical routing, 不是静态图
     需要: 路由转移 G_t→G_{t+1}, 吸引子, 相变

  3. "L24不是中文抑制" — negative diff ≠ suppression
     可能是: manifold rotation / basis transformation / energy redistribution
     需要用order parameter区分

  4. "Hub top-5不稳定" — 重尾分布下top-k采样不稳定
     需要: hub overlap curve (k=5,10,20,50,100,200)
     需要: 图谱距离代替节点ID比较

  5. "Pearson不够" — 只测线性同步, 无法捕捉非线性/条件化/稀疏交互
     理想用: MI/transfer entropy/conditional dependency
     实际: 40样本估计MI噪声太大, 用spectral方法更稳健

Phase 112核心升级:
  从"neuron统计"到"计算拓扑动力学"

  关键转变:
    neuron ID → subspace geometry
    static snapshot → trajectory evolution
    Pearson correlation → spectral topology
    "哪个neuron不同" → "差分激活的维度和几何是什么"

关键实验:
  Exp 1: Spectral Topology — 从节点身份到拓扑结构
    核心: 不问"哪些neuron共享", 问"计算图的拓扑是否相同"
    方法: 图Laplacian谱比较, 谱距离, 谱熵, 谱间隙
    如果翻译和中文的图谱不同 → 拓扑重构, 不只是neuron替换
    如果翻译和中文的图谱相似 → 只换了节点, 结构不变

  Exp 2: Activation Subspace & Phase Transition
    核心: 不问"哪个neuron不同", 问"差分激活的维度和几何是什么"
    方法: SVD分解差分矩阵, 分析translation subspace维度
    Order parameters: participation ratio, routing entropy, sparsity, cosine distance
    Phase transition检测: order parameter在哪些层突变
    区分: manifold rotation vs suppression vs redistribution

  Exp 3: Hub Overlap Curve & Spectral Alignment
    核心: 解决top-5不稳定问题
    方法: k=5,10,20,50,100,200的overlap curve + 随机null
    图谱距离: ||λ_zh - λ_trans||
    度排序相关性: Spearman rank correlation

  Exp 4: Activation Trajectory — 从静态到动态
    核心: 跟踪translation和zh在激活空间中的"轨迹"
    方法: PCA降维, 跟踪两种任务的轨迹演化
    关键: 轨迹何时分离? 方向是否旋转? 有无attractor?
    Subspace rotation: 差分方向在层间如何旋转 (区分suppression vs rotation)

Run:
  python tests/glm5/ccml_phase112_statistical_dynamics.py --model qwen3
  python tests/glm5/ccml_phase112_statistical_dynamics.py --model qwen3 --exp 1
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

from model_utils import load_model, get_layers, get_model_info, release_model


# ============================================================
# 测试数据 — 扩大到70个词对
# ============================================================
ANIMAL_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鱼", "fish"), ("鸟", "bird"),
    ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"),
    ("鸡", "chicken"), ("鸭", "duck"),
]

NATURE_PAIRS = [
    ("水", "water"), ("火", "fire"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("冰", "ice"), ("雷", "thunder"), ("雾", "fog"),
    ("霜", "frost"), ("云", "cloud"),
]

OBJECT_PAIRS = [
    ("花", "flower"), ("树", "tree"), ("石", "stone"), ("铁", "iron"),
    ("金", "gold"), ("茶", "tea"), ("沙", "sand"), ("草", "grass"),
    ("血", "blood"), ("光", "light"),
]

CELESTIAL_PAIRS = [
    ("月", "moon"), ("日", "sun"), ("星", "star"), ("河", "river"),
    ("山", "mountain"), ("海", "sea"), ("天", "sky"), ("地", "earth"),
    ("夜", "night"), ("昼", "day"),
]

COLOR_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"), ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"),
]

BODY_PAIRS = [
    ("手", "hand"), ("脚", "foot"), ("头", "head"), ("眼", "eye"),
    ("耳", "ear"), ("鼻", "nose"), ("口", "mouth"), ("心", "heart"),
]

ACTION_PAIRS = [
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("游", "swim"),
    ("吃", "eat"), ("喝", "drink"), ("睡", "sleep"), ("看", "see"),
    ("听", "hear"), ("说", "say"),
]

NUMBER_PAIRS = [
    ("一", "one"), ("二", "two"), ("三", "three"), ("四", "four"),
]

ALL_PAIRS = (ANIMAL_PAIRS + NATURE_PAIRS + OBJECT_PAIRS + CELESTIAL_PAIRS
             + COLOR_PAIRS + BODY_PAIRS + ACTION_PAIRS + NUMBER_PAIRS)  # 70词对


def compute_mlp_gate_activations(model, tokenizer, device, prompt, n_layers):
    """收集MLP gate activation (SwiGLU gate_proj → SiLU)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_layers(model)

    gate_activations = {}
    hooks = []

    def make_gate_hook(l):
        def hook_fn(module, input, output):
            gate_act = torch.nn.functional.silu(output)
            gate_activations[l] = gate_act[0, -1, :].detach().float().cpu().numpy()
        return hook_fn

    for l, layer in enumerate(layers):
        if hasattr(layer.mlp, 'gate_proj'):
            h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(l))
            hooks.append(h)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    for h in hooks:
        h.remove()

    del outputs, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return gate_activations


def collect_all_activations(model, tokenizer, device, n_layers, word_pairs=None):
    """收集所有词对的gate activations — 只加载模型一次!"""
    if word_pairs is None:
        word_pairs = ALL_PAIRS

    zh_gate = defaultdict(list)
    trans_gate = defaultdict(list)

    for i, (zh, en) in enumerate(word_pairs):
        zh_prompt = f"{zh}是一种"
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)

        trans_prompt = f'"{zh}"的英文是'
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)

        for l in range(n_layers):
            if l in gate_zh:
                zh_gate[l].append(gate_zh[l])
            if l in gate_trans:
                trans_gate[l].append(gate_trans[l])  # BUG FIX: gate_trans not trans_gate!

        if (i + 1) % 10 == 0:
            print(f"    已处理 {i+1}/{len(word_pairs)} 个词对")

    # Convert to numpy arrays
    for l in zh_gate:
        zh_gate[l] = np.array(zh_gate[l])
    for l in trans_gate:
        trans_gate[l] = np.array(trans_gate[l])

    return zh_gate, trans_gate


def normalized_laplacian_eigvals(adj, n_eigvals=None):
    """计算normalized Laplacian的特征值"""
    n = adj.shape[0]
    degrees = np.sum(adj, axis=1)
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
    L_norm = np.eye(n) - (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :])
    # 对称化(消除数值误差)
    L_norm = (L_norm + L_norm.T) / 2
    eigvals = np.linalg.eigvalsh(L_norm)
    return np.sort(eigvals)


def spectral_entropy(eigvals):
    """谱熵 — 图拓扑复杂度的信息论度量"""
    eigvals_pos = np.maximum(eigvals, 0)
    total = np.sum(eigvals_pos)
    if total < 1e-10:
        return 0
    probs = eigvals_pos / total
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log(probs))


def effective_dimension(eigvals):
    """有效维度 — 图的'自由度'估计"""
    eigvals_pos = np.maximum(eigvals, 1e-10)
    return (np.sum(eigvals_pos))**2 / np.sum(eigvals_pos**2)


# ============================================================
# Exp 1: Spectral Topology — 从节点身份到拓扑结构
# ============================================================
def exp1_spectral_topology(zh_gate, trans_gate, n_layers, intermediate_size):
    print(f"\n{'='*60}")
    print("Exp 1: Spectral Topology — 图谱拓扑比较")
    print(f"{'='*60}")
    print(f"  核心: 翻译和中文的计算图拓扑是否不同?")
    print(f"  方法: normalized Laplacian谱比较 (不看节点ID, 看拓扑结构)")

    sample_layers = list(range(0, n_layers, 3)) + [n_layers - 2, n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    n_top = 200  # top-200活跃neuron做谱分析
    corr_threshold = 0.3

    results = {}

    for l in sample_layers:
        if l not in zh_gate or l not in trans_gate:
            continue

        zh_data = zh_gate[l]
        trans_data = trans_gate[l]

        # 找top活跃neuron
        all_data = np.concatenate([zh_data, trans_data], axis=0)
        mean_act = np.mean(np.abs(all_data), axis=0)
        top_indices = np.argsort(mean_act)[::-1][:n_top]

        zh_sub = zh_data[:, top_indices]
        trans_sub = trans_data[:, top_indices]

        if zh_sub.shape[0] < 3 or trans_sub.shape[0] < 3:
            continue

        # 构建相关矩阵
        zh_corr = np.corrcoef(zh_sub.T)
        trans_corr = np.corrcoef(trans_sub.T)
        zh_corr = np.nan_to_num(zh_corr, nan=0.0)
        trans_corr = np.nan_to_num(trans_corr, nan=0.0)

        # 邻接矩阵 (阈值化)
        zh_adj = (np.abs(zh_corr) > corr_threshold).astype(float)
        np.fill_diagonal(zh_adj, 0)
        trans_adj = (np.abs(trans_corr) > corr_threshold).astype(float)
        np.fill_diagonal(trans_adj, 0)

        # 计算Laplacian谱
        zh_eigvals = normalized_laplacian_eigvals(zh_adj)
        trans_eigvals = normalized_laplacian_eigvals(trans_adj)

        # === 谱距离 ===
        n_compare = min(len(zh_eigvals), len(trans_eigvals))
        spectral_dist = np.linalg.norm(zh_eigvals[:n_compare] - trans_eigvals[:n_compare])
        spectral_dist_normalized = spectral_dist / np.sqrt(n_compare)

        # === 谱熵 ===
        zh_spec_entropy = spectral_entropy(zh_eigvals)
        trans_spec_entropy = spectral_entropy(trans_eigvals)

        # === 谱间隙 (代数连通度) ===
        zh_spec_gap = zh_eigvals[1] if len(zh_eigvals) > 1 else 0
        trans_spec_gap = trans_eigvals[1] if len(trans_eigvals) > 1 else 0

        # === 有效维度 ===
        zh_eff_dim = effective_dimension(zh_eigvals)
        trans_eff_dim = effective_dimension(trans_eigvals)

        # === 图密度 ===
        n_nodes = zh_adj.shape[0]
        zh_density = np.sum(zh_adj) / (n_nodes * (n_nodes - 1))
        trans_density = np.sum(trans_adj) / (n_nodes * (n_nodes - 1))

        # === 谱分布的Wasserstein距离 ===
        # 将特征值看作分布, 计算分布距离
        zh_eig_cdf = np.cumsum(np.sort(zh_eigvals)) / max(np.sum(np.abs(zh_eigvals)), 1e-10)
        trans_eig_cdf = np.cumsum(np.sort(trans_eigvals)) / max(np.sum(np.abs(trans_eigvals)), 1e-10)
        wasserstein_dist = np.mean(np.abs(zh_eig_cdf - trans_eig_cdf))

        layer_result = {
            "spectral_distance": float(spectral_dist),
            "spectral_distance_normalized": float(spectral_dist_normalized),
            "zh_spectral_entropy": float(zh_spec_entropy),
            "trans_spectral_entropy": float(trans_spec_entropy),
            "entropy_diff": float(trans_spec_entropy - zh_spec_entropy),
            "zh_spectral_gap": float(zh_spec_gap),
            "trans_spectral_gap": float(trans_spec_gap),
            "zh_eff_dim": float(zh_eff_dim),
            "trans_eff_dim": float(trans_eff_dim),
            "zh_density": float(zh_density),
            "trans_density": float(trans_density),
            "density_ratio": float(trans_density / max(zh_density, 1e-10)),
            "wasserstein_dist": float(wasserstein_dist),
        }
        results[l] = layer_result

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}: spec_dist={spectral_dist_normalized:.4f}, "
                  f"H(zh={zh_spec_entropy:.3f}, trans={trans_spec_entropy:.3f}, Δ={trans_spec_entropy-zh_spec_entropy:+.3f}), "
                  f"gap(zh={zh_spec_gap:.4f}, trans={trans_spec_gap:.4f}), "
                  f"ρ(zh={zh_density:.4f}, trans={trans_density:.4f}), "
                  f"dim(zh={zh_eff_dim:.1f}, trans={trans_eff_dim:.1f}), "
                  f"W_dist={wasserstein_dist:.4f}")

    # === 汇总: 拓扑差异最大的层 ===
    print(f"\n  === 拓扑差异排名 (spectral distance) ===")
    sorted_layers = sorted(results.keys(), key=lambda l: results[l]["spectral_distance_normalized"], reverse=True)
    for i, l in enumerate(sorted_layers[:5]):
        r = results[l]
        print(f"    #{i+1}: L{l} — spec_dist={r['spectral_distance_normalized']:.4f}, "
              f"ΔH={r['entropy_diff']:+.3f}, ρ_ratio={r['density_ratio']:.3f}")

    return results


# ============================================================
# Exp 2: Activation Subspace & Phase Transition
# ============================================================
def exp2_activation_subspace(zh_gate, trans_gate, n_layers, intermediate_size):
    print(f"\n{'='*60}")
    print("Exp 2: Activation Subspace & Phase Transition Detection")
    print(f"{'='*60}")
    print(f"  核心: 差分激活的子空间维度和能量如何跨层演化?")
    print(f"  关键order parameters: participation ratio, routing entropy, cosine distance")
    print(f"  Phase transition: order parameter在哪些层突变?")

    results = {}

    for l in range(n_layers):
        if l not in zh_gate or l not in trans_gate:
            continue

        zh_data = zh_gate[l]  # (n_samples, intermediate)
        trans_data = trans_gate[l]

        if zh_data.shape[0] < 5:
            continue

        n_samples = zh_data.shape[0]

        # === 1. 差分矩阵 ===
        diff_matrix = trans_data - zh_data

        # === 2. SVD → translation subspace ===
        U, s, Vt = np.linalg.svd(diff_matrix, full_matrices=False)

        # === 3. Participation ratio (子空间有效维度) ===
        s_squared = s**2
        total_energy = np.sum(s_squared)
        if total_energy < 1e-10:
            pr = 0
        else:
            pr = (np.sum(s_squared))**2 / np.sum(s_squared**2)

        # 能量累积
        cum_energy = np.cumsum(s_squared) / max(total_energy, 1e-10)
        dim_50 = int(np.searchsorted(cum_energy, 0.5) + 1)
        dim_90 = int(np.searchsorted(cum_energy, 0.9) + 1)
        dim_99 = int(np.searchsorted(cum_energy, 0.99) + 1)

        # === 4. Routing entropy (激活集中度) ===
        def compute_routing_entropy(data):
            mean_act = np.mean(np.abs(data), axis=0)
            total = np.sum(mean_act)
            if total < 1e-10:
                return 0, 0
            p = mean_act / total
            p = p[p > 1e-10]
            H = -np.sum(p * np.log(p))
            H_max = np.log(data.shape[1])
            return H, H / H_max if H_max > 0 else 0

        H_trans, H_trans_norm = compute_routing_entropy(trans_data)
        H_zh, H_zh_norm = compute_routing_entropy(zh_data)

        # === 5. Sparsity (零激活比例) ===
        zh_sparsity = float(np.mean(zh_data < 0.01))
        trans_sparsity = float(np.mean(trans_data < 0.01))

        # === 6. Task discriminability (cosine distance) ===
        zh_mean = np.mean(zh_data, axis=0)
        trans_mean = np.mean(trans_data, axis=0)
        cos_sim = np.dot(zh_mean, trans_mean) / (np.linalg.norm(zh_mean) * np.linalg.norm(trans_mean) + 1e-10)
        cos_dist = 1 - cos_sim

        # === 7. Differential energy ===
        diff_energy = float(np.mean(np.sum(diff_matrix**2, axis=1)))

        # === 8. Permutation test ===
        n_perm = 50
        null_energies = []
        all_data = np.concatenate([zh_data, trans_data], axis=0)
        for _ in range(n_perm):
            perm = np.random.permutation(len(all_data))
            null_diff = all_data[perm[:n_samples]] - all_data[perm[n_samples:]]
            null_energies.append(float(np.mean(np.sum(null_diff**2, axis=1))))
        null_mean = np.mean(null_energies)
        null_std = np.std(null_energies)
        energy_z = (diff_energy - null_mean) / max(null_std, 1e-10)

        # === 9. Stable rank (有效秩) ===
        # rank_eff = (||A||_F)^2 / ||A||_2^2 = sum(s^2) / s[0]^2
        stable_rank = total_energy / max(s[0]**2, 1e-10) if len(s) > 0 else 0

        # === 10. 类内离散度 (attractor指标) ===
        # 如果translation的类内方差随层减小 → 表示在收敛到attractor
        zh_var = float(np.mean(np.var(zh_data, axis=0)))
        trans_var = float(np.mean(np.var(trans_data, axis=0)))

        layer_result = {
            "participation_ratio": float(pr),
            "stable_rank": float(stable_rank),
            "dim_50pct": dim_50,
            "dim_90pct": dim_90,
            "dim_99pct": dim_99,
            "total_diff_energy": diff_energy,
            "energy_z_score": float(energy_z),
            "routing_entropy_trans": float(H_trans_norm),
            "routing_entropy_zh": float(H_zh_norm),
            "routing_entropy_diff": float(H_trans_norm - H_zh_norm),
            "sparsity_trans": trans_sparsity,
            "sparsity_zh": zh_sparsity,
            "cosine_distance": float(cos_dist),
            "zh_variance": zh_var,
            "trans_variance": trans_var,
            "variance_ratio": float(trans_var / max(zh_var, 1e-10)),
            "top5_singular_values": s[:5].tolist(),
        }
        results[l] = layer_result

        if l % 3 == 0 or l >= n_layers - 3:
            print(f"    L{l}: PR={pr:.1f}, srank={stable_rank:.1f}, "
                  f"dim50={dim_50}, dim90={dim_90}, "
                  f"energy_z={energy_z:.1f}, cos_dist={cos_dist:.4f}, "
                  f"H_trans={H_trans_norm:.4f}, H_zh={H_zh_norm:.4f}, "
                  f"sparsity(zh={zh_sparsity:.3f}, trans={trans_sparsity:.3f}), "
                  f"var_ratio={trans_var/max(zh_var,1e-10):.3f}")

    # === Phase Transition Detection ===
    print(f"\n  === Phase Transition Detection ===")
    print(f"  检测order parameter的突变层 (jump > 3× mean_step)")

    layers_sorted = sorted(results.keys())
    for param_name in ["participation_ratio", "cosine_distance", "routing_entropy_trans",
                       "total_diff_energy", "routing_entropy_diff", "stable_rank"]:
        values = [results[l][param_name] for l in layers_sorted]
        if len(values) < 3:
            continue

        diffs = np.abs(np.diff(values))
        max_diff_idx = np.argmax(diffs)
        max_diff = diffs[max_diff_idx]
        mean_diff = np.mean(diffs)

        if max_diff > 3 * mean_diff and mean_diff > 1e-10:
            transition_l1 = layers_sorted[max_diff_idx]
            transition_l2 = layers_sorted[max_diff_idx + 1] if max_diff_idx + 1 < len(layers_sorted) else transition_l1
            print(f"    {param_name}: ★突变 L{transition_l1}→L{transition_l2}, "
                  f"jump={max_diff:.6f}, mean_step={mean_diff:.6f}, ratio={max_diff/mean_diff:.1f}x")

    return results


# ============================================================
# Exp 3: Hub Overlap Curve & Spectral Alignment
# ============================================================
def exp3_hub_overlap_curve(zh_gate, trans_gate, n_layers, intermediate_size):
    print(f"\n{'='*60}")
    print("Exp 3: Hub Overlap Curve & Spectral Alignment")
    print(f"{'='*60}")
    print(f"  核心: hub overlap在不同k下的行为 + 图谱距离")
    print(f"  解决: top-5不稳定问题, 用图谱距离代替节点ID")

    sample_layers = [0, 6, 12, 18, 24, 27, 30, 33, 35]
    n_top = 300
    k_values = [5, 10, 20, 50, 100, 200]

    results = {}

    for l in sample_layers:
        if l not in zh_gate or l not in trans_gate:
            continue

        zh_data = zh_gate[l]
        trans_data = trans_gate[l]

        all_data = np.concatenate([zh_data, trans_data], axis=0)
        mean_act = np.mean(np.abs(all_data), axis=0)
        active_indices = np.argsort(mean_act)[::-1][:n_top]

        zh_sub = zh_data[:, active_indices]
        trans_sub = trans_data[:, active_indices]

        if zh_sub.shape[0] < 3 or trans_sub.shape[0] < 3:
            continue

        zh_corr = np.corrcoef(zh_sub.T)
        trans_corr = np.corrcoef(trans_sub.T)
        zh_corr = np.nan_to_num(zh_corr, nan=0.0)
        trans_corr = np.nan_to_num(trans_corr, nan=0.0)

        # 度分布
        zh_adj = (np.abs(zh_corr) > 0.3).astype(float)
        np.fill_diagonal(zh_adj, 0)
        trans_adj = (np.abs(trans_corr) > 0.3).astype(float)
        np.fill_diagonal(trans_adj, 0)

        zh_degrees = np.sum(zh_adj, axis=1)
        trans_degrees = np.sum(trans_adj, axis=1)

        zh_ranked = np.argsort(zh_degrees)[::-1]
        trans_ranked = np.argsort(trans_degrees)[::-1]

        # === Hub overlap curve ===
        overlap_curve = {}
        for k in k_values:
            if k > len(active_indices):
                continue
            zh_hubs = set(zh_ranked[:k].tolist())
            trans_hubs = set(trans_ranked[:k].tolist())
            overlap = len(zh_hubs & trans_hubs) / k
            overlap_curve[k] = float(overlap)

        # === Random null ===
        null_overlaps = {}
        n_bootstrap = 100
        for k in k_values:
            if k > len(active_indices):
                continue
            null_vals = []
            for _ in range(n_bootstrap):
                rand1 = set(np.random.choice(len(active_indices), k, replace=False).tolist())
                rand2 = set(np.random.choice(len(active_indices), k, replace=False).tolist())
                null_vals.append(len(rand1 & rand2) / k)
            null_overlaps[k] = {
                "mean": float(np.mean(null_vals)),
                "std": float(np.std(null_vals)),
            }

        # === 谱距离 ===
        zh_eigvals = np.sort(np.linalg.eigvalsh(zh_adj))
        trans_eigvals = np.sort(np.linalg.eigvalsh(trans_adj))
        spectral_dist = float(np.linalg.norm(zh_eigvals - trans_eigvals))

        # === 度排序相关性 ===
        rank_corr, rank_p = spearmanr(zh_degrees, trans_degrees)

        # === 谱重叠 (前10个特征值的相关) ===
        n_spec = min(20, len(zh_eigvals))
        spec_corr, _ = spearmanr(zh_eigvals[:n_spec], trans_eigvals[:n_spec])

        layer_result = {
            "overlap_curve": overlap_curve,
            "null_overlaps": null_overlaps,
            "spectral_distance": spectral_dist,
            "degree_rank_correlation": float(rank_corr),
            "degree_rank_p": float(rank_p),
            "spectral_rank_correlation": float(spec_corr),
        }
        results[l] = layer_result

        overlap_str = ", ".join([f"k={k}:{overlap_curve[k]:.3f}" for k in k_values if k in overlap_curve])
        null_str = ", ".join([f"k={k}:{null_overlaps[k]['mean']:.3f}" for k in k_values if k in null_overlaps])
        print(f"    L{l}: overlap=[{overlap_str}]")
        print(f"         null=  [{null_str}]")
        print(f"         spec_dist={spectral_dist:.4f}, rank_corr={rank_corr:.3f}, spec_corr={spec_corr:.3f}")

    # === 汇总: overlap curve是否稳定 ===
    print(f"\n  === Hub Overlap Curve分析 ===")
    for l in sample_layers:
        if l not in results:
            continue
        r = results[l]
        oc = r["overlap_curve"]
        nc = r["null_overlaps"]

        # 计算overlap/null比值
        ratios = []
        for k in k_values:
            if k in oc and k in nc and nc[k]["mean"] > 0.001:
                ratios.append(oc[k] / nc[k]["mean"])
        if ratios:
            mean_ratio = np.mean(ratios)
            print(f"    L{l}: overlap/null ratio={mean_ratio:.2f}x (跨k平均), "
                  f"rank_corr={r['degree_rank_correlation']:.3f}")

    return results


# ============================================================
# Exp 4: Activation Trajectory — 从静态到动态
# ============================================================
def exp4_activation_trajectory(zh_gate, trans_gate, n_layers, intermediate_size):
    print(f"\n{'='*60}")
    print("Exp 4: Activation Trajectory — 激活轨迹动力学")
    print(f"{'='*60}")
    print(f"  核心: 翻译和中文在激活空间中的轨迹如何演化?")
    print(f"  关键: 轨迹何时分离? 方向是否旋转? 有无attractor?")
    print(f"  区分: manifold rotation vs suppression vs redistribution")

    # === 1. 收集每层的均值激活 ===
    zh_means = {}
    trans_means = {}
    for l in range(n_layers):
        if l in zh_gate and l in trans_gate:
            zh_means[l] = np.mean(zh_gate[l], axis=0)
            trans_means[l] = np.mean(trans_gate[l], axis=0)

    layer_order = sorted(zh_means.keys())
    if len(layer_order) < 3:
        print("  层数不足, 无法做轨迹分析")
        return {}

    # === 2. PCA降维 ===
    all_means = []
    labels = []
    for l in layer_order:
        all_means.append(zh_means[l])
        labels.append((l, 'zh'))
        all_means.append(trans_means[l])
        labels.append((l, 'trans'))

    all_means = np.array(all_means)

    n_components = min(10, all_means.shape[0] - 1, all_means.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(all_means)

    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_[:5]}")

    # 提取轨迹
    zh_traj = {}
    trans_traj = {}
    for i, (l, task) in enumerate(labels):
        if task == 'zh':
            zh_traj[l] = coords[i]
        else:
            trans_traj[l] = coords[i]

    common_layers = sorted(set(zh_traj.keys()) & set(trans_traj.keys()))

    # === 3. 轨迹divergence (距离) ===
    print(f"\n  Trajectory divergence (翻译vs中文的距离):")
    divergences = {}
    for l in common_layers:
        dist = float(np.linalg.norm(zh_traj[l] - trans_traj[l]))
        divergences[l] = dist
        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}: distance={dist:.4f}")

    # === 4. 轨迹velocity (变化速率) ===
    print(f"\n  Trajectory velocity (层间变化速率):")
    zh_velocity = {}
    trans_velocity = {}
    for i in range(len(common_layers) - 1):
        l1 = common_layers[i]
        l2 = common_layers[i + 1]
        zh_vel = float(np.linalg.norm(zh_traj[l2] - zh_traj[l1]))
        trans_vel = float(np.linalg.norm(trans_traj[l2] - trans_traj[l1]))
        zh_velocity[l1] = zh_vel
        trans_velocity[l1] = trans_vel
        if l1 % 6 == 0 or l1 >= n_layers - 4:
            print(f"    L{l1}→L{l2}: zh_vel={zh_vel:.4f}, trans_vel={trans_vel:.4f}, "
                  f"ratio={trans_vel/max(zh_vel, 1e-10):.2f}")

    # === 5. 方向持续性 (相邻步的方向一致性) ===
    print(f"\n  Direction persistence (相邻步的方向cosine):")
    zh_persist = {}
    trans_persist = {}
    for i in range(len(common_layers) - 2):
        l1 = common_layers[i]
        l2 = common_layers[i + 1]
        l3 = common_layers[i + 2]

        zh_d1 = zh_traj[l2] - zh_traj[l1]
        zh_d2 = zh_traj[l3] - zh_traj[l2]
        trans_d1 = trans_traj[l2] - trans_traj[l1]
        trans_d2 = trans_traj[l3] - trans_traj[l2]

        zh_cos = float(np.dot(zh_d1, zh_d2) / (np.linalg.norm(zh_d1) * np.linalg.norm(zh_d2) + 1e-10))
        trans_cos = float(np.dot(trans_d1, trans_d2) / (np.linalg.norm(trans_d1) * np.linalg.norm(trans_d2) + 1e-10))
        zh_persist[l2] = zh_cos
        trans_persist[l2] = trans_cos
        if l2 % 6 == 0 or l2 >= n_layers - 3:
            print(f"    L{l2}: zh_persist={zh_cos:.3f}, trans_persist={trans_cos:.3f}")

    # === 6. Subspace rotation (差分方向的旋转 — 区分suppression vs rotation) ===
    print(f"\n  Subspace rotation (差分方向在层间的旋转角):")
    print(f"  ★ 如果旋转角大 → manifold rotation (不是简单的suppression)")
    print(f"  ★ 如果旋转角小 + 距离大 → 纯suppression/amplification")

    diff_directions = {}
    for l in common_layers:
        diff = trans_means[l] - zh_means[l]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            diff_directions[l] = diff / norm
        else:
            diff_directions[l] = np.zeros_like(diff)

    rotation_results = {}
    for i in range(len(common_layers) - 1):
        l1 = common_layers[i]
        l2 = common_layers[i + 1]
        if l1 in diff_directions and l2 in diff_directions:
            alignment = float(np.dot(diff_directions[l1], diff_directions[l2]))
            angle = float(np.arccos(np.clip(alignment, -1, 1)) * 180 / np.pi)
            rotation_results[l1] = {"alignment": alignment, "angle_deg": angle}
            if l1 % 3 == 0 or l1 >= n_layers - 4:
                rot_type = "ROTATION" if angle > 30 else "AMPLIFICATION" if angle < 10 else "MIXED"
                print(f"    L{l1}→L{l2}: alignment={alignment:.3f}, angle={angle:.1f}° [{rot_type}]")

    # === 7. Attractor指标 (类内方差随层变化) ===
    print(f"\n  Attractor indicator (类内方差随层变化):")
    print(f"  ★ 方差递减 → 表示在收敛到attractor")
    zh_variances = {}
    trans_variances = {}
    for l in common_layers:
        if l in zh_gate and l in trans_gate:
            zh_var = float(np.mean(np.var(zh_gate[l], axis=0)))
            trans_var = float(np.mean(np.var(trans_gate[l], axis=0)))
            zh_variances[l] = zh_var
            trans_variances[l] = trans_var
            if l % 6 == 0 or l >= n_layers - 3:
                trend = ""
                if l > min(common_layers):
                    prev_l = max(ll for ll in common_layers if ll < l)
                    if prev_l in trans_variances:
                        if trans_var < trans_variances[prev_l]:
                            trend = "↓ 收敛"
                        else:
                            trend = "↑ 发散"
                print(f"    L{l}: zh_var={zh_var:.6f}, trans_var={trans_var:.6f} {trend}")

    # === Phase Transition Summary ===
    print(f"\n  === Trajectory Phase Transition Summary ===")

    # Divergence跳变
    if len(divergences) > 2:
        div_layers = sorted(divergences.keys())
        div_vals = [divergences[l] for l in div_layers]
        div_diffs = np.abs(np.diff(div_vals))
        if len(div_diffs) > 0:
            max_idx = np.argmax(div_diffs)
            print(f"    最大divergence跳变: L{div_layers[max_idx]}→L{div_layers[max_idx+1]}, "
                  f"jump={div_diffs[max_idx]:.4f}")

    # 最大旋转
    if rotation_results:
        max_rot_l = max(rotation_results.keys(), key=lambda l: rotation_results[l]["angle_deg"])
        max_rot = rotation_results[max_rot_l]
        print(f"    最大subspace旋转: L{max_rot_l}, angle={max_rot['angle_deg']:.1f}°")

    # Velocity异常
    if trans_velocity:
        vel_layers = sorted(trans_velocity.keys())
        vel_vals = [trans_velocity[l] for l in vel_layers]
        if len(vel_vals) > 2:
            mean_vel = np.mean(vel_vals)
            max_vel_idx = np.argmax(vel_vals)
            print(f"    最大translation velocity: L{vel_layers[max_vel_idx]}, "
                  f"vel={vel_vals[max_vel_idx]:.4f} (mean={mean_vel:.4f}, "
                  f"ratio={vel_vals[max_vel_idx]/mean_vel:.1f}x)")

    results = {
        "divergences": divergences,
        "zh_velocity": zh_velocity,
        "trans_velocity": trans_velocity,
        "zh_direction_persist": zh_persist,
        "trans_direction_persist": trans_persist,
        "subspace_rotation": rotation_results,
        "zh_variances": zh_variances,
        "trans_variances": trans_variances,
        "pca_explained_variance": pca.explained_variance_ratio_[:5].tolist(),
    }

    return results


# ============================================================
# Main: Collect once, analyze 4 times
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--exp", type=int, default=0)  # 0=all, 1-4=individual
    args = parser.parse_args()

    model_name = args.model
    t0 = time.time()

    print(f"\n{'#'*60}")
    print(f"Phase 112: Computation Graph Statistical Dynamics")
    print(f"{'#'*60}")

    # === Step 1: Collect all activations (only once!) ===
    print(f"\n  [Step 1] 收集所有激活 ({len(ALL_PAIRS)}个词对)...")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    intermediate_size = model_info.intermediate_size

    zh_gate, trans_gate = collect_all_activations(model, tokenizer, device, n_layers, ALL_PAIRS)

    # 立即释放模型!
    release_model(model)
    print(f"  模型已释放, 后续分析在CPU上进行")

    # === Step 2: Run experiments ===
    all_results = {"n_layers": n_layers, "intermediate_size": intermediate_size,
                   "n_pairs": len(ALL_PAIRS), "model": model_name}

    if args.exp in [0, 1]:
        t1 = time.time()
        all_results["exp1_spectral_topology"] = exp1_spectral_topology(
            zh_gate, trans_gate, n_layers, intermediate_size)
        print(f"  Exp 1 耗时: {time.time()-t1:.1f}s")

    if args.exp in [0, 2]:
        t1 = time.time()
        all_results["exp2_activation_subspace"] = exp2_activation_subspace(
            zh_gate, trans_gate, n_layers, intermediate_size)
        print(f"  Exp 2 耗时: {time.time()-t1:.1f}s")

    if args.exp in [0, 3]:
        t1 = time.time()
        all_results["exp3_hub_overlap_curve"] = exp3_hub_overlap_curve(
            zh_gate, trans_gate, n_layers, intermediate_size)
        print(f"  Exp 3 耗时: {time.time()-t1:.1f}s")

    if args.exp in [0, 4]:
        t1 = time.time()
        all_results["exp4_activation_trajectory"] = exp4_activation_trajectory(
            zh_gate, trans_gate, n_layers, intermediate_size)
        print(f"  Exp 4 耗时: {time.time()-t1:.1f}s")

    # === Step 3: Save ===
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    out_path = f"tests/glm5_temp/phase112_{model_name}_statistical_dynamics.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n  结果保存到 {out_path}")
    print(f"  总耗时: {time.time()-t0:.1f}s")
