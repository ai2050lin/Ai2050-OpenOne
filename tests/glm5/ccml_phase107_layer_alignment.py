"""
Phase 107: 层间对齐几何与Probe可靠性验证
==========================================

Phase 106的硬伤 (用户批判):
  1. "Probe 100% 是幻觉"
     2560维空间中50个样本的100%探针几乎无意义
     高维空间中几乎任何东西都线性可分
     必须做: random label控制, 跨分布泛化, 低数据缩放

  2. "Fisher rank=2 根本性错误"
     F = E[hh^T] W_U^T W_U 不是标准Fisher信息矩阵
     effective rank=2可能只是top-2 variance modes
     不能推出"decoder只读2个方向"

  3. "还没真正证明规范不变量"
     只观察到坐标旋转, 没有严格不变量
     需要证明: principal angles, Procrustes alignment, holonomy

  4. "理论化过快"
     用了gauge/transport/invariant大词但没有严格证明
     目前只能说"类规范旋转表示", 不能说"规范动力学"

Phase 107核心升级:
  从"声称发现规范结构"到"严格验证对齐几何"

关键实验:
  Exp 1: Probe可靠性验证 — 三个控制实验
    A. Random label probe: 打乱标签, 看100%是否是幻觉
    B. Cross-distribution generalization: 训练动物词, 测试自然现象词
    C. Low-data scaling: N=5,10,20,50 的probe accuracy curve

  Exp 2: 层间Principal Angles — 翻译子空间的主角分析
    核心问题: 翻译子空间在层间如何对齐?
    - 在每层用PCA提取翻译差分子空间(k维)
    - 计算相邻层之间的principal angles
    - principal angles小 → 子空间被有结构地传输
    - principal angles大 → 子空间被重参数化

  Exp 3: Procrustes Alignment — 是否存在正交映射对齐跨层子空间
    寻找正交矩阵Q使得Q*S_l ≈ S_{l+1}
    如果Q存在且residual小 → 有结构的传输
    如果Q不存在 → 确实是重参数化

  Exp 4: Holonomy Accumulation — 闭环旋转结构
    36层旋转后: R_36*...*R_1是什么?
    是否有闭环几何结构?
    旋转的累积是否形成群?

  Exp 5: Decoder Alignment Manifold — 翻译子空间何时进入decoder可读流形
    翻译子空间与W_U主方向的principal angles随层如何变化?
    关键量: 翻译子空间与Fisher前k方向的alignment

Run:
  python tests/glm5/ccml_phase107_layer_alignment.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase107_layer_alignment.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase107_layer_alignment.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase107_layer_alignment.py --model qwen3 --exp 4
  python tests/glm5/ccml_phase107_layer_alignment.py --model qwen3 --exp 5
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.linalg import subspace_angles, orthogonal_procrustes
from scipy.spatial.transform import Rotation

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据 — 按语义域分组, 用于跨分布测试
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

ALL_PAIRS = ANIMAL_PAIRS + NATURE_PAIRS + OBJECT_PAIRS + CELESTIAL_PAIRS  # 40词对

# 额外词对用于低数据缩放测试
EXTRA_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"), ("大", "big"), ("小", "small"), ("长", "long"),
    ("短", "short"), ("新", "new"), ("旧", "old"), ("快", "fast"),
    ("慢", "slow"), ("高", "tall"), ("低", "low"), ("热", "hot"),
    ("冷", "cold"), ("甜", "sweet"), ("苦", "bitter"), ("酸", "sour"),
]


def get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def apply_layer_norm(h, eps=1e-5):
    mean = np.mean(h)
    std = np.std(h)
    return (h - mean) / (std + eps)


def collect_hidden_states(model, tokenizer, device, pairs, prompt_types=None):
    """收集所有词对在各层的hidden states"""
    if prompt_types is None:
        prompt_types = ["zh", "trans"]

    model_info = get_model_info(model, model.name if hasattr(model, 'name') else 'qwen3')
    n_layers = model_info.n_layers

    layer_states = defaultdict(lambda: {"zh": [], "trans": []})

    for zh, en in pairs:
        prompts = {
            "zh": f"{zh}是一种",
            "trans": f'"{zh}"的英文是',
        }
        for ptype in prompt_types:
            prompt = prompts[ptype]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                layer_states[l][ptype].append(h)

    return dict(layer_states)


# ============================================================
# Exp 1: Probe可靠性验证 — 三个控制实验
# ============================================================
def exp1_probe_reliability(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 1: Probe可靠性验证 — 是否probe illusion?")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 收集hidden states (60个词对)
    all_pairs = ALL_PAIRS + EXTRA_PAIRS  # 60词对
    print(f"\n  收集{len(all_pairs)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, all_pairs)

    # 选取关键层
    sample_layers = [0, 6, 12, 21, 27, 33, 35, 36]
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    results = {}

    # ========================================
    # A. Random Label Probe
    # ========================================
    print(f"\n  === A. Random Label Probe ===")
    print(f"  打乱标签, 看probe是否仍然100%")

    n_random_trials = 10
    random_results = {}

    for l in sample_layers:
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float32)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float32)
        X = np.vstack([zh_data, trans_data])
        X_ln = np.array([apply_layer_norm(x) for x in X], dtype=np.float32)
        y_true = np.array([0]*len(zh_data) + [1]*len(trans_data))

        # True label accuracy
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_ln, y_true)
            true_acc = accuracy_score(y_true, lr.predict(X_ln))
        except:
            true_acc = 0.5

        # Random label accuracy (多次试验)
        random_accs = []
        for trial in range(n_random_trials):
            y_random = np.random.permutation(y_true)
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_ln, y_random)
                rand_acc = accuracy_score(y_random, lr.predict(X_ln))
            except:
                rand_acc = 0.5
            random_accs.append(rand_acc)

        random_results[l] = {
            "true_label_acc": float(true_acc),
            "random_label_acc_mean": float(np.mean(random_accs)),
            "random_label_acc_std": float(np.std(random_accs)),
            "random_label_acc_max": float(np.max(random_accs)),
            "n_random_trials": n_random_trials,
        }
        print(f"    L{l}: true={true_acc:.3f}, random_mean={np.mean(random_accs):.3f}±{np.std(random_accs):.3f}, random_max={np.max(random_accs):.3f}")

    results["random_label_probe"] = random_results

    # ========================================
    # B. Cross-Distribution Generalization
    # ========================================
    print(f"\n  === B. Cross-Distribution Generalization ===")
    print(f"  训练: 动物词, 测试: 自然现象词")

    domain_groups = {
        "train": ANIMAL_PAIRS,  # 10个动物词
        "test_nature": NATURE_PAIRS,  # 10个自然现象词
        "test_object": OBJECT_PAIRS,  # 10个物体词
        "test_celestial": CELESTIAL_PAIRS,  # 10个天体词
    }

    # 收集各组的hidden states
    domain_states = {}
    for group_name, pairs in domain_groups.items():
        group_states = defaultdict(lambda: {"zh": [], "trans": []})
        for zh, en in pairs:
            prompts = {
                "zh": f"{zh}是一种",
                "trans": f'"{zh}"的英文是',
            }
            for ptype, prompt in prompts.items():
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(inputs["input_ids"], output_hidden_states=True)
                for l in range(n_layers + 1):
                    h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                    group_states[l][ptype].append(h)
        domain_states[group_name] = dict(group_states)

    cross_dist_results = {}
    for l in sample_layers:
        # 训练数据
        train_zh = np.array(domain_states["train"][l]["zh"], dtype=np.float32)
        train_trans = np.array(domain_states["train"][l]["trans"], dtype=np.float32)
        X_train = np.vstack([train_zh, train_trans])
        X_train_ln = np.array([apply_layer_norm(x) for x in X_train], dtype=np.float32)
        y_train = np.array([0]*len(train_zh) + [1]*len(train_trans))

        # 训练探针
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_train_ln, y_train)
            train_acc = accuracy_score(y_train, lr.predict(X_train_ln))
        except:
            train_acc = 0.5
            lr = None

        # 在各测试域测试
        test_accs = {}
        for test_name in ["test_nature", "test_object", "test_celestial"]:
            test_zh = np.array(domain_states[test_name][l]["zh"], dtype=np.float32)
            test_trans = np.array(domain_states[test_name][l]["trans"], dtype=np.float32)
            X_test = np.vstack([test_zh, test_trans])
            X_test_ln = np.array([apply_layer_norm(x) for x in X_test], dtype=np.float32)
            y_test = np.array([0]*len(test_zh) + [1]*len(test_trans))

            if lr is not None:
                test_acc = accuracy_score(y_test, lr.predict(X_test_ln))
            else:
                test_acc = 0.5
            test_accs[test_name] = float(test_acc)

        cross_dist_results[l] = {
            "train_acc": float(train_acc),
            **test_accs,
        }
        print(f"    L{l}: train={train_acc:.3f}, nature={test_accs['test_nature']:.3f}, "
              f"object={test_accs['test_object']:.3f}, celestial={test_accs['test_celestial']:.3f}")

    results["cross_distribution"] = cross_dist_results

    # ========================================
    # C. Low-Data Scaling
    # ========================================
    print(f"\n  === C. Low-Data Scaling ===")
    print(f"  N=5,10,20,40,60的probe accuracy curve")

    # 使用所有60个词对
    all_zh = np.array(layer_states[21]["zh"], dtype=np.float32)  # 用L21做代表
    all_trans = np.array(layer_states[21]["trans"], dtype=np.float32)

    n_total = len(all_zh)
    sample_sizes = [5, 10, 20, 40, 60]
    n_trials = 5

    scaling_results = {}
    for n in sample_sizes:
        if n > n_total:
            continue
        accs = []
        for trial in range(n_trials):
            idx = np.random.choice(n_total, n, replace=False)
            X = np.vstack([all_zh[idx], all_trans[idx]])
            X_ln = np.array([apply_layer_norm(x) for x in X], dtype=np.float32)
            y = np.array([0]*n + [1]*n)
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_ln, y)
                acc = accuracy_score(y, lr.predict(X_ln))
            except:
                acc = 0.5
            accs.append(acc)
        scaling_results[n] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        }
        print(f"    N={n}: acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    # 同时做各层的N=10缩放 (最有诊断力的)
    print(f"\n  各层N=10缩放:")
    layer_scaling = {}
    for l in sample_layers:
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float32)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float32)
        n_total_l = len(zh_data)

        accs = []
        for trial in range(n_trials):
            idx = np.random.choice(n_total_l, 10, replace=False)
            X = np.vstack([zh_data[idx], trans_data[idx]])
            X_ln = np.array([apply_layer_norm(x) for x in X], dtype=np.float32)
            y = np.array([0]*10 + [1]*10)
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_ln, y)
                acc = accuracy_score(y, lr.predict(X_ln))
            except:
                acc = 0.5
            accs.append(acc)
        layer_scaling[l] = {
            "N10_mean": float(np.mean(accs)),
            "N10_std": float(np.std(accs)),
        }
        print(f"    L{l}: N=10 acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    results["low_data_scaling"] = scaling_results
    results["layer_N10_scaling"] = layer_scaling

    # 保存
    out_path = f"tests/glm5_temp/phase107_exp1_{model_name}_probe_reliability.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: 层间Principal Angles — 翻译子空间主角分析
# ============================================================
def exp2_principal_angles(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 2: 层间Principal Angles — 翻译子空间对齐")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # 收集40个词对的hidden states
    print(f"\n  收集{len(ALL_PAIRS)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, ALL_PAIRS)

    # 在每层构建翻译差分子空间
    # 方法: 翻译prompt和中文prompt的差分矩阵, PCA提取前k维
    print(f"\n  构建翻译差分子空间 (k=5)...")

    k = 5  # 子空间维度
    subspaces = {}  # {layer: basis_matrix (d_model x k)}

    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)

        # 差分矩阵
        diff = trans_data - zh_data  # (n_pairs, d_model)

        # LN后
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)

        # PCA
        mean_diff = np.mean(diff_ln, axis=0)
        centered = diff_ln - mean_diff

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Vt[:k] 是前k个主方向
        subspaces[l] = Vt[:k].T  # (d_model, k), 列向量是基

    # 计算相邻层的principal angles
    print(f"\n  计算相邻层principal angles...")
    pa_results = {}

    for l in range(n_layers):
        # subspaces[l] 和 subspaces[l+1] 之间的principal angles
        # scipy.linalg.subspace_angles: 计算两个子空间之间的主角
        A = subspaces[l]      # (d_model, k)
        B = subspaces[l + 1]  # (d_model, k)

        angles_rad = subspace_angles(A, B)  # 返回min(k,k)个角度
        angles_deg = np.degrees(angles_rad)

        pa_results[f"{l}_{l+1}"] = {
            "angles_rad": angles_rad.tolist(),
            "angles_deg": angles_deg.tolist(),
            "max_angle_deg": float(np.max(angles_deg)),
            "mean_angle_deg": float(np.mean(angles_deg)),
            "min_angle_deg": float(np.min(angles_deg)),
        }

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}→L{l+1}: max={np.max(angles_deg):.1f}°, "
                  f"mean={np.mean(angles_deg):.1f}°, "
                  f"min={np.min(angles_deg):.1f}°")

    # 也计算跨多层的principal angles (全局结构)
    print(f"\n  跨层principal angles (L0 vs L_l):")
    global_pa = {}
    for l in [6, 12, 21, 27, 33, 36]:
        if l > n_layers:
            continue
        A = subspaces[0]
        B = subspaces[l]
        angles_rad = subspace_angles(A, B)
        angles_deg = np.degrees(angles_rad)
        global_pa[f"0_{l}"] = {
            "max_angle_deg": float(np.max(angles_deg)),
            "mean_angle_deg": float(np.mean(angles_deg)),
        }
        print(f"    L0→L{l}: max={np.max(angles_deg):.1f}°, mean={np.mean(angles_deg):.1f}°")

    # 不同k值的主角
    print(f"\n  子空间维度k的影响:")
    k_values = [1, 2, 3, 5, 10]
    k_results = {}
    for k_val in k_values:
        # 重建子空间
        sub_k = {}
        for l in [0, 12, 21, 33, 36]:
            if l > n_layers:
                continue
            zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
            trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
            diff = trans_data - zh_data
            diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)
            mean_diff = np.mean(diff_ln, axis=0)
            centered = diff_ln - mean_diff
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            sub_k[l] = Vt[:min(k_val, len(Vt))].T

        k_pa = {}
        for l in [12, 21, 33, 36]:
            if l > n_layers or 0 not in sub_k or l not in sub_k:
                continue
            if sub_k[0].shape[1] == 0 or sub_k[l].shape[1] == 0:
                continue
            min_dim = min(sub_k[0].shape[1], sub_k[l].shape[1])
            A = sub_k[0][:, :min_dim]
            B = sub_k[l][:, :min_dim]
            angles_rad = subspace_angles(A, B)
            angles_deg = np.degrees(angles_rad)
            k_pa[f"0_{l}"] = {
                "max_angle_deg": float(np.max(angles_deg)),
                "mean_angle_deg": float(np.mean(angles_deg)),
            }
            print(f"    k={k_val}, L0→L{l}: max={np.max(angles_deg):.1f}°, mean={np.mean(angles_deg):.1f}°")

        k_results[k_val] = k_pa

    results = {
        "adjacent_principal_angles": pa_results,
        "global_principal_angles": global_pa,
        "k_scaling": k_results,
        "subspace_dim_k": k,
    }

    out_path = f"tests/glm5_temp/phase107_exp2_{model_name}_principal_angles.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: Procrustes Alignment — 正交映射对齐
# ============================================================
def exp3_procrustes(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 3: Procrustes Alignment — 正交映射对齐跨层子空间")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"\n  收集hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, ALL_PAIRS)

    k = 5  # 子空间维度

    # 在每层构建翻译差分子空间
    print(f"\n  构建翻译差分子空间 (k={k})...")
    subspaces = {}
    sv_spectrum = {}

    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)
        mean_diff = np.mean(diff_ln, axis=0)
        centered = diff_ln - mean_diff
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        subspaces[l] = Vt[:k].T  # (d_model, k)
        sv_spectrum[l] = S[:10].tolist()  # 前10个奇异值

    # Procrustes analysis: 找正交矩阵Q使得Q*A ≈ B
    print(f"\n  Procrustes Analysis (adjacent layers):")

    procrustes_results = {}
    for l in range(n_layers):
        A = subspaces[l]      # (d_model, k)
        B = subspaces[l + 1]  # (d_model, k)

        # orthogonal_procrustes: 找正交R(k,k)使得||A@R^T - B||_F最小
        R, scale = orthogonal_procrustes(A, B)

        # 计算对齐后的残差: A @ R^T ≈ B
        aligned = A @ R.T
        residual = np.linalg.norm(aligned - B, 'fro')
        norm_B = np.linalg.norm(B, 'fro')
        relative_residual = residual / norm_B if norm_B > 0 else 0

        procrustes_results[f"{l}_{l+1}"] = {
            "scale": float(scale),
            "frobenius_residual": float(residual),
            "relative_residual": float(relative_residual),
        }

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}→L{l+1}: residual={residual:.4f}, "
                  f"rel_residual={relative_residual:.4f}, scale={scale:.4f}")

    # 全局Procrustes: L0 → L_l
    print(f"\n  Global Procrustes (L0 vs L_l):")
    global_proc = {}
    for l in [6, 12, 21, 27, 33, 36]:
        if l > n_layers:
            continue
        A = subspaces[0]
        B = subspaces[l]
        R, scale = orthogonal_procrustes(A, B)
        aligned = A @ R.T
        residual = np.linalg.norm(aligned - B, 'fro')
        norm_B = np.linalg.norm(B, 'fro')
        rel_res = residual / norm_B if norm_B > 0 else 0
        global_proc[f"0_{l}"] = {
            "relative_residual": float(rel_res),
            "scale": float(scale),
        }
        print(f"    L0→L{l}: rel_residual={rel_res:.4f}")

    # 累积Procrustes: 逐层对齐
    # 从L0出发, 逐层应用最优正交映射, 看最终与L36的差距
    print(f"\n  Cumulative Procrustes (层间逐步对齐):")
    cumulative_A = subspaces[0].copy()
    cumulative_residuals = {}
    for l in range(n_layers):
        R, scale = orthogonal_procrustes(cumulative_A, subspaces[l + 1])
        cumulative_A = cumulative_A @ R.T
        res = np.linalg.norm(cumulative_A - subspaces[l + 1], 'fro')
        norm_b = np.linalg.norm(subspaces[l + 1], 'fro')
        rel_res = res / norm_b if norm_b > 0 else 0
        cumulative_residuals[l + 1] = float(rel_res)

    for l in [6, 12, 21, 27, 33, 36]:
        if l in cumulative_residuals:
            print(f"    L0→L{l} (cumulative): rel_residual={cumulative_residuals[l]:.4f}")

    # 也在原始hidden state空间做Procrustes (不只是子空间)
    # 用差分向量的均值方向做Procrustes
    print(f"\n  Raw difference Procrustes (原始差分向量):")
    raw_proc = {}
    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)

        # 差分矩阵
        diff_curr = (trans_data - zh_data).T  # (d_model, n_pairs)
        diff_next = (trans_next - zh_next).T  # (d_model, n_pairs)

        R, scale = orthogonal_procrustes(diff_curr, diff_next)
        aligned = diff_curr @ R.T
        residual = np.linalg.norm(aligned - diff_next, 'fro')
        norm_next = np.linalg.norm(diff_next, 'fro')
        rel_res = residual / norm_next if norm_next > 0 else 0

        raw_proc[f"{l}_{l+1}"] = {
            "relative_residual": float(rel_res),
        }

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}→L{l+1}: rel_residual={rel_res:.4f}")

    results = {
        "adjacent_procrustes": procrustes_results,
        "global_procrustes": global_proc,
        "cumulative_procrustes": cumulative_residuals,
        "raw_diff_procrustes": raw_proc,
        "sv_spectrum": sv_spectrum,
        "subspace_dim_k": k,
    }

    out_path = f"tests/glm5_temp/phase107_exp3_{model_name}_procrustes.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Holonomy Accumulation — 闭环旋转结构
# ============================================================
def exp4_holonomy(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 4: Holonomy Accumulation — 36层旋转的闭环结构")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"\n  收集hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, ALL_PAIRS)

    k = 5
    print(f"\n  构建翻译差分子空间 (k={k})...")
    subspaces = {}
    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)
        mean_diff = np.mean(diff_ln, axis=0)
        centered = diff_ln - mean_diff
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        subspaces[l] = Vt[:k].T  # (d_model, k)

    # 逐层累积正交变换, 看总旋转是什么
    # Procrustes返回R(k,k): 找正交R使得||A@R^T - B||_F最小
    # 所以: A@R^T ≈ B, 即R^T将A的坐标变换到B的坐标
    print(f"\n  计算逐层正交变换R_l (k={k}维子空间)...")
    per_layer_R = []
    per_layer_residual = []
    for l in range(n_layers):
        A = subspaces[l]       # (d_model, k)
        B = subspaces[l + 1]   # (d_model, k)
        R, scale = orthogonal_procrustes(A, B)  # R: (k, k)
        aligned = A @ R.T
        residual = np.linalg.norm(aligned - B, 'fro') / np.linalg.norm(B, 'fro')
        per_layer_R.append(R)
        per_layer_residual.append(float(residual))

    # 累积旋转: R_cum = R_0 * R_1 * ... * R_l
    # 作用: subspace[0] @ R_cum^T 约等于 subspace[l+1]
    print(f"\n  累积旋转分析:")
    holonomy_results = {}
    R_cum = np.eye(k)

    for l in range(n_layers):
        R_cum = R_cum @ per_layer_R[l]  # (k, k) 正交矩阵累积

        if l in [5, 11, 17, 23, 29, 35]:
            # 检查R_cum是否接近正交
            orth_dev = np.linalg.norm(R_cum @ R_cum.T - np.eye(k), 'fro')
            det = np.linalg.det(R_cum)

            # 用累积R变换subspace[0], 看与subspace[l+1]的对齐度
            transported = subspaces[0] @ R_cum.T  # (d_model, k)
            target = subspaces[l + 1]
            R_align, _ = orthogonal_procrustes(transported, target)
            aligned = transported @ R_align.T
            alignment_residual = np.linalg.norm(aligned - target, 'fro')
            alignment_norm = np.linalg.norm(target, 'fro')
            rel_alignment = alignment_residual / alignment_norm if alignment_norm > 0 else 0

            # R_cum的特征值 (旋转角度)
            eigenvalues = np.linalg.eigvals(R_cum)
            angles = np.degrees(np.angle(eigenvalues))
            sorted_angles = np.sort(np.abs(angles))[::-1]

            holonomy_results[l + 1] = {
                "orthogonality_deviation": float(orth_dev),
                "det": float(det),
                "alignment_residual": float(rel_alignment),
                "rotation_angles_deg": sorted_angles.tolist()[:5],
            }
            print(f"    L0→L{l+1}: orth_dev={orth_dev:.4f}, det={det:.4f}, "
                  f"alignment={rel_alignment:.4f}, top_angles={sorted_angles[:3]}")

    # 核心Holonomy问题: R_total是否接近单位阵?
    # 如果是, 则说明36层后子空间"回到"了原点
    print(f"\n  Holonomy test: R_total vs I:")
    print(f"    det(R_total) = {np.linalg.det(R_cum):.4f}")
    print(f"    ||R_total - I||_F = {np.linalg.norm(R_cum - np.eye(k), 'fro'):.4f}")
    print(f"    R_total eigenvalues: {np.sort(np.abs(np.linalg.eigvals(R_cum)))[::-1][:5]}")

    # 全局Procrustes: L0 → L36
    R_0_36, _ = orthogonal_procrustes(subspaces[0], subspaces[n_layers])
    transported_0 = subspaces[0] @ R_0_36.T
    holonomy_residual = np.linalg.norm(transported_0 - subspaces[n_layers], 'fro')
    holonomy_norm = np.linalg.norm(subspaces[n_layers], 'fro')
    holonomy_rel = holonomy_residual / holonomy_norm if holonomy_norm > 0 else 0

    # R_0_36的特征值分析 (旋转角度)
    eigenvalues_036 = np.linalg.eigvals(R_0_36)
    angles_036 = np.degrees(np.angle(eigenvalues_036))
    rotation_angles_deg = np.sort(np.abs(angles_036))[::-1]

    print(f"    L0→L36 Procrustes residual: {holonomy_rel:.4f}")
    print(f"    R_0_36 rotation angles (deg): {rotation_angles_deg[:5]}")

    # 分段Holonomy: 前半(L0-L18) vs 后半(L18-L36)
    print(f"\n  分段Holonomy:")
    mid = n_layers // 2
    R_first, _ = orthogonal_procrustes(subspaces[0], subspaces[mid])
    R_second, _ = orthogonal_procrustes(subspaces[mid], subspaces[n_layers])

    # 前半和后半的旋转角度
    ev_first = np.linalg.eigvals(R_first)
    ev_second = np.linalg.eigvals(R_second)
    angles_first = np.sort(np.abs(np.degrees(np.angle(ev_first))))[::-1]
    angles_second = np.sort(np.abs(np.degrees(np.angle(ev_second))))[::-1]

    print(f"    前半 L0→L{mid}: top rotation angles = {angles_first}")
    print(f"    后半 L{mid}→L{n_layers}: top rotation angles = {angles_second}")

    results = {
        "holonomy_per_layer": holonomy_results,
        "R_0_36_residual": float(holonomy_rel),
        "R_0_36_rotation_angles_deg": sorted(np.abs(rotation_angles_deg).tolist(), reverse=True)[:10],
        "first_half_angles": angles_first.tolist(),
        "second_half_angles": angles_second.tolist(),
    }

    out_path = f"tests/glm5_temp/phase107_exp4_{model_name}_holonomy.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 5: Decoder Alignment Manifold — 翻译子空间与decoder对齐
# ============================================================
def exp5_decoder_alignment(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 5: Decoder Alignment Manifold — 子空间何时进入decoder可读流形")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    print(f"\n  收集hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, ALL_PAIRS)

    k = 5

    # 构建翻译差分子空间
    print(f"\n  构建翻译差分子空间 (k={k})...")
    subspaces = {}
    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)
        mean_diff = np.mean(diff_ln, axis=0)
        centered = diff_ln - mean_diff
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        subspaces[l] = Vt[:k].T  # (d_model, k)

    # W_U的SVD分解 (decoder判别子空间)
    # W_U shape: (vocab_size, d_model)
    # 我们需要d_model维的判别方向 → 对W_U.T做SVD, 或者取W_U的右奇异向量
    print(f"\n  W_U SVD分析 (decoder判别子空间):")
    print(f"    W_U shape: {W_U.shape}")
    # W_U = U @ diag(S) @ Vt, U: (vocab, vocab), Vt: (d_model, d_model)
    # Vt的行是d_model维空间的基向量 — 这是我们需要的
    U_dec, S_dec, Vt_dec = np.linalg.svd(W_U, full_matrices=False)
    # Vt_dec: (d_model, d_model), 行向量是d_model维的基
    # 我们需要列形式: dec_directions = Vt_dec.T → (d_model, d_model)
    dec_directions = Vt_dec.T  # (d_model, min(vocab, d_model)), 列是d_model维主方向
    print(f"    dec_directions shape: {dec_directions.shape}")
    print(f"    Top-10 singular values: {S_dec[:10].tolist()}")
    print(f"    Top-100 singular values range: [{S_dec[99]:.2f}, {S_dec[0]:.2f}]")

    # W_U的有效秩 (用合理的阈值)
    total_energy = np.sum(S_dec**2)
    cumulative = np.cumsum(S_dec**2)
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        rank = np.searchsorted(cumulative / total_energy, thresh) + 1
        print(f"    {thresh*100:.0f}% energy rank: {rank}")

    # 关键分析: 翻译差分子空间与W_U主方向的principal angles
    # dec_directions的前k_dec列构成decoder的"读出子空间"
    print(f"\n  翻译差分子空间 vs Decoder读出子空间 (principal angles):")
    decoder_align = {}

    for k_dec in [2, 5, 10, 20]:
        dec_subspace = dec_directions[:, :k_dec]  # (d_model, k_dec)

        pa_per_layer = {}
        for l in range(n_layers + 1):
            # 翻译差分子空间 vs decoder子空间的principal angles
            # 两者维度可能不同, subspace_angles要求维度相同
            # 用较小的维度
            min_k = min(k, k_dec)
            A = subspaces[l][:, :min_k]
            B = dec_subspace[:, :min_k]

            angles_rad = subspace_angles(A, B)
            angles_deg = np.degrees(angles_rad)

            pa_per_layer[l] = {
                "max_angle_deg": float(np.max(angles_deg)),
                "mean_angle_deg": float(np.mean(angles_deg)),
            }

        decoder_align[k_dec] = pa_per_layer

        # 打印关键层
        for l in [0, 6, 12, 21, 27, 33, 35, 36]:
            if l in pa_per_layer:
                print(f"    k_dec={k_dec}, L{l}: max={pa_per_layer[l]['max_angle_deg']:.1f}°, "
                      f"mean={pa_per_layer[l]['mean_angle_deg']:.1f}°")

    # 更精细的分析: 翻译差分子空间投影到W_U各主方向上的能量
    print(f"\n  翻译差分信号在W_U主方向上的能量分布:")
    energy_per_layer = {}
    for l in [0, 6, 12, 21, 27, 33, 35, 36]:
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)

        # 投影到W_U的前20个主方向(dec_directions的列)
        projections = diff_ln @ dec_directions[:, :20]  # (n_pairs, 20)
        energy = np.mean(projections**2, axis=0)  # 每个主方向的平均能量

        total_e = np.sum(energy)
        top2_frac = np.sum(energy[:2]) / total_e if total_e > 0 else 0
        top5_frac = np.sum(energy[:5]) / total_e if total_e > 0 else 0
        top10_frac = np.sum(energy[:10]) / total_e if total_e > 0 else 0

        energy_per_layer[l] = {
            "top2_fraction": float(top2_frac),
            "top5_fraction": float(top5_frac),
            "top10_fraction": float(top10_frac),
            "energy_per_direction": energy.tolist()[:20],
        }
        print(f"    L{l}: top2={top2_frac:.3f}, top5={top5_frac:.3f}, top10={top10_frac:.3f}")

    # margin方向(W_U[en]-W_U[zh])在W_U主方向上的分解
    print(f"\n  Margin方向(W_U[en]-W_U[zh])在W_U主方向上的分解:")
    margin_decomp = {}
    for zh, en in ALL_PAIRS[:10]:  # 用前10个词对
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue
        margin_dir = W_U[en_id] - W_U[zh_id]
        # 投影到W_U主方向(dec_directions的列)
        proj = margin_dir @ dec_directions[:, :20]
        proj_energy = proj**2
        total_e = np.sum(proj_energy)
        top2_frac = np.sum(proj_energy[:2]) / total_e if total_e > 0 else 0
        top5_frac = np.sum(proj_energy[:5]) / total_e if total_e > 0 else 0
        margin_decomp[f"{zh}_{en}"] = {
            "top2_fraction": float(top2_frac),
            "top5_fraction": float(top5_frac),
            "norm": float(np.linalg.norm(margin_dir)),
        }
        print(f"    {zh}→{en}: top2={top2_frac:.3f}, top5={top5_frac:.3f}, norm={np.linalg.norm(margin_dir):.2f}")

    results = {
        "decoder_alignment": decoder_align,
        "energy_per_layer": energy_per_layer,
        "margin_decomposition": margin_decomp,
        "W_U_sv_spectrum": S_dec[:100].tolist(),
        "W_U_energy_ranks": {
            "50%": int(np.searchsorted(cumulative / total_energy, 0.5) + 1),
            "80%": int(np.searchsorted(cumulative / total_energy, 0.8) + 1),
            "90%": int(np.searchsorted(cumulative / total_energy, 0.9) + 1),
            "95%": int(np.searchsorted(cumulative / total_energy, 0.95) + 1),
            "99%": int(np.searchsorted(cumulative / total_energy, 0.99) + 1),
        },
    }

    out_path = f"tests/glm5_temp/phase107_exp5_{model_name}_decoder_alignment.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--exp", type=int, required=True)
    args = parser.parse_args()

    if args.exp == 1:
        exp1_probe_reliability(args)
    elif args.exp == 2:
        exp2_principal_angles(args)
    elif args.exp == 3:
        exp3_procrustes(args)
    elif args.exp == 4:
        exp4_holonomy(args)
    elif args.exp == 5:
        exp5_decoder_alignment(args)
    else:
        print(f"Unknown exp: {args.exp}")
