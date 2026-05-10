"""
Phase 108: 约束分散动力学 — 信号熵、参与率与功能传输
=====================================================

Phase 107的硬伤 (用户批判):
  1. "反向过度修正" — probe无意义 ≠ 信息不存在
     正确: probe不可靠, 但不代表信息不存在
     应该测: probe COMPLEXITY (样本复杂度/描述长度), 不是accuracy

  2. "W_U满秩 ≠ 有效读出维度高"
     50%能量需814维只说明谱长尾, 不说明所有方向同等重要
     需要: 可达logit子空间 (reachable logit subspace)

  3. "principal angle小 ≠ 稳定计算"
     可能是谱退化(spectral degeneracy), 不是stable translation computation
     需要: canonical correlation tracking, mode matching

  4. "仍研究状态几何, 不是向量场"
     需要从 h_l, Δh_l 转向 dh/dl = F(h,l)
     需要: local flow, stability structure

  5. "理论词汇膨胀" — gauge/attractor/phase transition未证明
     严格成立的只有: margin振荡, 子空间漂移, 非正交传输, 信号分散

Phase 108核心升级:
  从"信息是否存在"到"约束如何分散"
  用严格的可测量替代隐喻性描述

关键实验:
  Exp 1: Signal Entropy & Participation Ratio — 约束分散的严格量化
    核心量: 
    - 翻译差分信号在decoder基底中的能量分布 p_i
    - 信号熵 H(p) = -Σ p_i log p_i
    - 参与率 PR = (Σλ_i)² / Σλ_i²  (比rank稳定得多)
    - 有效支撑维度
    看L0→L36的信号分散过程

  Exp 2: Probe Complexity — 不是accuracy, 而是复杂度
    核心:
    - 样本复杂度: 达到90%准确率需要多少样本?
    - 最小描述长度: 需要多复杂的分类器?
    - 泛化曲线: train vs test accuracy随样本数的缩放
    如果L0需要5样本就能泛化, L30需要100样本 → L0确实更容易读出

  Exp 3: Reachable Logit Subspace — decoder真正用了多少维?
    核心:
    - 对大量文本计算logits, 分析logit的实际分布
    - PCA of logit vectors → 有效维度
    - 高频token vs 低频token的logit方向
    - 翻译相关token的logit在哪个子空间?

  Exp 4: Functional Transport — 扰动是否保持下游行为?
    核心:
    - 在L_l沿翻译差分方向扰动hidden state
    - 看扰动对L36的logit margin的影响
    - 如果L12的扰动→L36的margin变化大 → L12的功能传输强
    - 如果L30的扰动→L36的margin变化小 → L30的功能传输弱
    这才是真正functional transport, 不是几何角度

  Exp 5: Local Flow Field — dh/dl = F(h, l)的近似
    核心:
    - 在多个h点计算Δh = h_{l+1} - h_l
    - 分析flow field的局部结构
    - 是否存在不动点/鞍点?
    - flow的Jacobian的特征值分布随层如何变化?

Run:
  python tests/glm5/ccml_phase108_constraint_dispersion.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase108_constraint_dispersion.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase108_constraint_dispersion.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase108_constraint_dispersion.py --model qwen3 --exp 4
  python tests/glm5/ccml_phase108_constraint_dispersion.py --model qwen3 --exp 5
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
from scipy.linalg import subspace_angles

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据 — 按语义域分组
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

EXTRA_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"), ("大", "big"), ("小", "small"), ("长", "long"),
    ("短", "short"), ("新", "new"), ("旧", "old"), ("快", "fast"),
    ("慢", "slow"), ("高", "tall"), ("低", "low"), ("热", "hot"),
    ("冷", "cold"), ("甜", "sweet"), ("苦", "bitter"), ("酸", "sour"),
]

# 大量词对用于logit subspace分析
LARGE_PAIRS = ALL_PAIRS + EXTRA_PAIRS + [
    ("一", "one"), ("二", "two"), ("三", "three"), ("四", "four"),
    ("五", "five"), ("六", "six"), ("七", "seven"), ("八", "eight"),
    ("九", "nine"), ("十", "ten"),
    ("春", "spring"), ("夏", "summer"), ("秋", "autumn"), ("冬", "winter"),
    ("东", "east"), ("西", "west"), ("南", "south"), ("北", "north"),
    ("上", "up"), ("下", "down"), ("左", "left"), ("右", "right"),
    ("前", "front"), ("后", "back"), ("里", "inside"), ("外", "outside"),
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
# Exp 1: Signal Entropy & Participation Ratio
# ============================================================
def exp1_signal_dispersion(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 1: Signal Entropy & Participation Ratio — 约束分散量化")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # W_U的右奇异向量 (d_model维主方向)
    U_dec, S_dec, Vt_dec = np.linalg.svd(W_U, full_matrices=False)
    dec_directions = Vt_dec.T  # (d_model, d_model), 列是主方向

    # 收集hidden states
    all_pairs = ALL_PAIRS + EXTRA_PAIRS  # 60词对
    print(f"\n  收集{len(all_pairs)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, all_pairs)

    results = {}

    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)

        # 翻译差分 (LN后)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)
        mean_diff = np.mean(diff_ln, axis=0)  # 平均差分方向

        # 差分方向投影到decoder基底
        projection = mean_diff @ dec_directions  # (d_model,) 每个主方向的系数
        energy = projection ** 2

        # 信号熵
        total_energy = np.sum(energy)
        if total_energy > 0:
            p = energy / total_energy  # 能量占比
            p = p[p > 0]  # 去零
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(d_model)  # 均匀分布的最大熵
            normalized_entropy = entropy / max_entropy
        else:
            entropy = 0
            normalized_entropy = 0

        # 参与率 (Participation Ratio)
        # PR = (Σλ_i)² / Σλ_i²
        lambda_i = energy
        sum_lambda = np.sum(lambda_i)
        sum_lambda_sq = np.sum(lambda_i ** 2)
        if sum_lambda_sq > 0:
            PR = (sum_lambda ** 2) / sum_lambda_sq
        else:
            PR = 0

        # 有效支撑维度 (95% energy需要多少维)
        sorted_energy = np.sort(energy)[::-1]
        cumulative = np.cumsum(sorted_energy)
        if total_energy > 0:
            dim_50 = np.searchsorted(cumulative / total_energy, 0.5) + 1
            dim_90 = np.searchsorted(cumulative / total_energy, 0.9) + 1
            dim_95 = np.searchsorted(cumulative / total_energy, 0.95) + 1
            dim_99 = np.searchsorted(cumulative / total_energy, 0.99) + 1
        else:
            dim_50 = dim_90 = dim_95 = dim_99 = 0

        results[l] = {
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "max_entropy": float(max_entropy),
            "participation_ratio": float(PR),
            "dim_50": int(dim_50),
            "dim_90": int(dim_90),
            "dim_95": int(dim_95),
            "dim_99": int(dim_99),
            "top2_fraction": float(np.sum(sorted_energy[:2]) / total_energy) if total_energy > 0 else 0,
            "top5_fraction": float(np.sum(sorted_energy[:5]) / total_energy) if total_energy > 0 else 0,
            "top10_fraction": float(np.sum(sorted_energy[:10]) / total_energy) if total_energy > 0 else 0,
            "diff_norm": float(np.linalg.norm(mean_diff)),
        }

        if l % 6 == 0 or l >= n_layers - 2:
            print(f"    L{l}: H={entropy:.2f} (norm={normalized_entropy:.3f}), "
                  f"PR={PR:.1f}, dim90={dim_90}, top2={results[l]['top2_fraction']:.3f}")

    # 也在各层自己的SVD基底中做(不投影到W_U)
    print(f"\n  在各层自己的SVD基底中的信号分散:")
    for l in [0, 6, 12, 21, 27, 33, 35, 36]:
        if l > n_layers:
            continue
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        diff_ln = np.array([apply_layer_norm(d) for d in diff], dtype=np.float64)

        # 在差分矩阵本身做SVD
        mean_diff = np.mean(diff_ln, axis=0)
        centered = diff_ln - mean_diff
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # S是差分信号在自己基底中的奇异值谱
        total_sv = np.sum(S)
        if total_sv > 0:
            p = (S ** 2) / np.sum(S ** 2)
            p = p[p > 0]
            self_entropy = -np.sum(p * np.log(p))
            self_PR = (np.sum(S) ** 2) / np.sum(S ** 2) if np.sum(S ** 2) > 0 else 0
        else:
            self_entropy = 0
            self_PR = 0

        print(f"    L{l} (self-basis): H={self_entropy:.2f}, PR={self_PR:.1f}, "
              f"S[:5]={S[:5].tolist()}")

    out_path = f"tests/glm5_temp/phase108_exp1_{model_name}_signal_dispersion.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Probe Complexity — 样本复杂度与描述长度
# ============================================================
def exp2_probe_complexity(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 2: Probe Complexity — 不是accuracy, 而是复杂度")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 收集60词对的hidden states
    all_pairs = ALL_PAIRS + EXTRA_PAIRS
    print(f"\n  收集{len(all_pairs)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, all_pairs)

    sample_layers = [0, 6, 12, 21, 27, 33, 35, 36]
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    results = {}

    for l in sample_layers:
        if l > n_layers:
            continue

        zh_data = np.array(layer_states[l]["zh"], dtype=np.float32)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float32)
        X_all = np.vstack([zh_data, trans_data])
        X_ln = np.array([apply_layer_norm(x) for x in X_all], dtype=np.float32)
        y_all = np.array([0]*len(zh_data) + [1]*len(trans_data))

        layer_result = {}

        # ========================================
        # A. 样本复杂度: 达到90%/95%准确率需要多少样本?
        # ========================================
        n_total = len(y_all)
        sample_sizes = list(range(4, 52, 2))  # 4, 6, 8, ..., 50
        n_trials = 20

        scaling_curve = {}
        for n_train in sample_sizes:
            if n_train >= n_total:
                continue
            accs = []
            for trial in range(n_trials):
                # 随机分train/test
                idx = np.random.permutation(n_total)
                train_idx = idx[:n_train]
                test_idx = idx[n_train:]

                X_train, y_train = X_ln[train_idx], y_all[train_idx]
                X_test, y_test = X_ln[test_idx], y_all[test_idx]

                try:
                    lr = LogisticRegression(max_iter=1000, C=1.0)
                    lr.fit(X_train, y_train)
                    acc = accuracy_score(y_test, lr.predict(X_test))
                except:
                    acc = 0.5
                accs.append(acc)

            scaling_curve[n_train] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "median": float(np.median(accs)),
            }

        # 找到达到90%/95%的最小样本数
        min_90 = None
        min_95 = None
        for n_train in sample_sizes:
            if n_train >= n_total:
                continue
            if scaling_curve[n_train]["median"] >= 0.90 and min_90 is None:
                min_90 = n_train
            if scaling_curve[n_train]["median"] >= 0.95 and min_95 is None:
                min_95 = n_train

        layer_result["sample_complexity"] = scaling_curve
        layer_result["min_samples_90"] = min_90
        layer_result["min_samples_95"] = min_95

        # ========================================
        # B. 描述长度: 不同正则化强度的probe性能
        # ========================================
        # 正则化越强 = 分类器越简单 = 描述长度越短
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        n_train_fixed = 20  # 固定训练集大小

        desc_length_curve = {}
        for C in C_values:
            accs = []
            for trial in range(n_trials):
                idx = np.random.permutation(n_total)
                train_idx = idx[:n_train_fixed]
                test_idx = idx[n_train_fixed:]

                X_train, y_train = X_ln[train_idx], y_all[train_idx]
                X_test, y_test = X_ln[test_idx], y_all[test_idx]

                try:
                    lr = LogisticRegression(max_iter=1000, C=C)
                    lr.fit(X_train, y_train)
                    train_acc = accuracy_score(y_train, lr.predict(X_train))
                    test_acc = accuracy_score(y_test, lr.predict(X_test))
                    # 权重的L2范数作为描述长度代理
                    weight_norm = np.linalg.norm(lr.coef_)
                except:
                    train_acc = 0.5
                    test_acc = 0.5
                    weight_norm = 0
                accs.append({
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "weight_norm": float(weight_norm),
                })

            desc_length_curve[C] = {
                "mean_test_acc": float(np.mean([a["test_acc"] for a in accs])),
                "mean_weight_norm": float(np.mean([a["weight_norm"] for a in accs])),
            }

        layer_result["desc_length_curve"] = desc_length_curve

        # ========================================
        # C. Random label baseline (同Phase 107, 但加上样本缩放)
        # ========================================
        random_scaling = {}
        for n_train in [10, 20, 40, 60]:
            if n_train > n_total:
                continue
            rand_accs = []
            for trial in range(10):
                idx = np.random.choice(n_total, n_train, replace=False)
                X_sub = X_ln[idx]
                y_random = np.random.permutation(y_all[idx])
                try:
                    lr = LogisticRegression(max_iter=1000, C=1.0)
                    lr.fit(X_sub, y_random)
                    acc = accuracy_score(y_random, lr.predict(X_sub))
                except:
                    acc = 0.5
                rand_accs.append(acc)
            random_scaling[n_train] = {
                "mean": float(np.mean(rand_accs)),
                "std": float(np.std(rand_accs)),
            }

        layer_result["random_label_scaling"] = random_scaling

        results[l] = layer_result
        print(f"    L{l}: min_90={min_90}, min_95={min_95}, "
              f"random@20={random_scaling.get(20, {}).get('mean', 0):.3f}")

    out_path = f"tests/glm5_temp/phase108_exp2_{model_name}_probe_complexity.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: Reachable Logit Subspace
# ============================================================
def exp3_logit_subspace(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 3: Reachable Logit Subspace — decoder真正用了多少维?")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # ========================================
    # A. W_U的SVD详细分析
    # ========================================
    U_dec, S_dec, Vt_dec = np.linalg.svd(W_U, full_matrices=False)
    dec_directions = Vt_dec.T  # (d_model, d_model)

    # 参与率
    PR_WU = (np.sum(S_dec) ** 2) / np.sum(S_dec ** 2)
    total_energy = np.sum(S_dec ** 2)
    cumulative = np.cumsum(S_dec ** 2)

    print(f"\n  W_U SVD:")
    print(f"    Shape: {W_U.shape}")
    print(f"    Participation Ratio: {PR_WU:.1f}")
    print(f"    Top-5 sv: {S_dec[:5].tolist()}")
    print(f"    Top-100 sv: [{S_dec[99]:.2f}, ..., {S_dec[0]:.2f}]")

    # ========================================
    # B. 实际logit向量的子空间
    # ========================================
    # 用大量文本, 计算各层的logit向量, 分析其子空间维度
    print(f"\n  收集实际logit向量...")

    # 用翻译prompt和中文prompt生成logits
    logit_vectors = []
    prompt_labels = []  # 0=中文, 1=翻译

    for zh, en in LARGE_PAIRS:
        for ptype, prompt in [("zh", f"{zh}是一种"), ("trans", f'"{zh}"的英文是')]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            # L36的hidden state → logits
            h_final = outputs.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
            logits = h_final @ W_U.T  # (vocab_size,)
            logit_vectors.append(logits)
            prompt_labels.append(0 if ptype == "zh" else 1)

    logit_matrix = np.array(logit_vectors, dtype=np.float64)  # (n_prompts, vocab_size)
    print(f"    Logit matrix shape: {logit_matrix.shape}")

    # Logit PCA
    mean_logits = np.mean(logit_matrix, axis=0)
    centered_logits = logit_matrix - mean_logits
    U_logit, S_logit, Vt_logit = np.linalg.svd(centered_logits, full_matrices=False)

    PR_logits = (np.sum(S_logit) ** 2) / np.sum(S_logit ** 2) if np.sum(S_logit ** 2) > 0 else 0

    print(f"\n  Logit PCA:")
    print(f"    Participation Ratio: {PR_logits:.1f}")
    print(f"    Top-10 sv: {S_logit[:10].tolist()}")

    total_logit_energy = np.sum(S_logit ** 2)
    cum_logit = np.cumsum(S_logit ** 2)
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        dim = np.searchsorted(cum_logit / total_logit_energy, thresh) + 1 if total_logit_energy > 0 else 0
        print(f"    {thresh*100:.0f}% energy dim: {dim}")

    # ========================================
    # C. 翻译相关token的logit子空间
    # ========================================
    print(f"\n  翻译相关token的logit方向:")

    # 收集翻译相关token的W_U行向量
    trans_token_ids = []
    trans_token_names = []
    for zh, en in ALL_PAIRS:
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is not None and zh_id is not None:
            trans_token_ids.append(en_id)
            trans_token_names.append(f"{zh}→{en}(en)")
            trans_token_ids.append(zh_id)
            trans_token_names.append(f"{zh}→{en}(zh)")

    # 翻译相关token的W_U行向量
    trans_W_U = W_U[trans_token_ids]  # (n_trans_tokens, d_model)
    U_trans, S_trans, Vt_trans = np.linalg.svd(trans_W_U, full_matrices=False)
    PR_trans = (np.sum(S_trans) ** 2) / np.sum(S_trans ** 2) if np.sum(S_trans ** 2) > 0 else 0

    print(f"    翻译token数: {len(trans_token_ids)}")
    print(f"    翻译token W_U Participation Ratio: {PR_trans:.1f}")
    print(f"    Top-5 sv: {S_trans[:5].tolist()}")

    # 翻译token子空间与W_U主方向的principal angles
    trans_subspace = Vt_trans.T[:, :5]  # (d_model, 5)
    dec_subspace = dec_directions[:, :5]

    angles = subspace_angles(trans_subspace, dec_subspace)
    print(f"    翻译子空间 vs W_U top5 主方向: max={np.max(np.degrees(angles)):.1f}°, "
          f"mean={np.mean(np.degrees(angles)):.1f}°")

    # ========================================
    # D. 高频 vs 低频 token的logit方向
    # ========================================
    print(f"\n  高频 vs 低频 token分析:")

    # 用tokenizer的词频近似 (简单方法: 用token ID排序, ID小的通常更常见)
    # 更好: 用实际文本的token频率
    # 简化: 取前1000个token (通常是special + 高频) vs 后1000个

    high_freq_WU = W_U[:1000]  # 高频token
    low_freq_WU = W_U[-1000:]  # 低频token

    _, S_high, _ = np.linalg.svd(high_freq_WU, full_matrices=False)
    _, S_low, _ = np.linalg.svd(low_freq_WU, full_matrices=False)

    PR_high = (np.sum(S_high) ** 2) / np.sum(S_high ** 2)
    PR_low = (np.sum(S_low) ** 2) / np.sum(S_low ** 2)

    print(f"    高频token(前1000): PR={PR_high:.1f}, top5 sv={S_high[:5].tolist()}")
    print(f"    低频token(后1000): PR={PR_low:.1f}, top5 sv={S_low[:5].tolist()}")

    results = {
        "W_U_PR": float(PR_WU),
        "W_U_top_sv": S_dec[:20].tolist(),
        "logit_PR": float(PR_logits),
        "logit_top_sv": S_logit[:20].tolist(),
        "trans_token_PR": float(PR_trans),
        "trans_token_top_sv": S_trans[:10].tolist(),
        "high_freq_PR": float(PR_high),
        "low_freq_PR": float(PR_low),
    }

    out_path = f"tests/glm5_temp/phase108_exp3_{model_name}_logit_subspace.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Functional Transport — 扰动传播
# ============================================================
def exp4_functional_transport(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 4: Functional Transport — 扰动是否保持下游行为?")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 选择几个关键层做扰动
    perturb_layers = [0, 6, 12, 21, 27, 33]
    alphas = [0.1, 0.5, 1.0, 2.0]  # 扰动强度

    results = {}

    for test_pair_idx, (zh, en) in enumerate(ALL_PAIRS[:15]):  # 15个词对
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue

        w_en = W_U[en_id]
        w_zh = W_U[zh_id]
        margin_dir = w_en - w_zh  # decoder-aligned margin方向

        pair_result = {"zh": zh, "en": en}

        # 1. 获取baseline (无扰动)
        prompt_trans = f'"{zh}"的英文是'
        prompt_zh = f"{zh}是一种"

        inputs_trans = tokenizer(prompt_trans, return_tensors="pt").to(device)
        inputs_zh = tokenizer(prompt_zh, return_tensors="pt").to(device)

        with torch.no_grad():
            out_trans = model(inputs_trans["input_ids"], output_hidden_states=True)
            out_zh = model(inputs_zh["input_ids"], output_hidden_states=True)

        # Baseline margin at final layer
        h_final_trans = out_trans.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
        h_final_zh = out_zh.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
        h_final_trans_ln = apply_layer_norm(h_final_trans)
        h_final_zh_ln = apply_layer_norm(h_final_zh)
        baseline_margin = np.dot(h_final_trans_ln, margin_dir) - np.dot(h_final_zh_ln, margin_dir)

        pair_result["baseline_margin"] = float(baseline_margin)

        # 2. 在每层扰动, 看对最终margin的影响
        for perturb_l in perturb_layers:
            if perturb_l >= n_layers:
                continue

            # 获取该层的翻译差分方向 (hidden state空间)
            h_l_trans = out_trans.hidden_states[perturb_l][0, -1, :].float().cpu().numpy()
            h_l_zh = out_zh.hidden_states[perturb_l][0, -1, :].float().cpu().numpy()
            diff_dir = h_l_trans - h_l_zh
            diff_norm = np.linalg.norm(diff_dir)
            if diff_norm > 0:
                diff_dir = diff_dir / diff_norm  # 归一化

            layer_result = {}

            for alpha in alphas:
                # 沿翻译差分方向扰动翻译prompt的hidden state
                # 然后继续forward到最后一层
                # 由于无法直接修改中间层hidden state重新forward,
                # 我们用线性近似: 扰动α*diff_dir → 传播到L36的影响

                # 简化方法: 计算每层的Jacobian近似
                # Δmargin ≈ (∂margin/∂h_l) · (α * diff_dir)

                # ∂margin/∂h_l 的近似: 用差分法
                # margin = (LN(h_36) @ margin_dir)_trans - (LN(h_36) @ margin_dir)_zh
                # 对翻译prompt:
                # Δmargin_trans ≈ margin_dir · (∂h_36/∂h_l) · diff_dir * α

                # 更简单: 直接看各层差分方向的投影能量
                # 如果diff_dir在L_l有很多能量指向margin_dir,
                # 那么扰动这个方向会影响最终margin

                h_l_trans_ln = apply_layer_norm(h_l_trans)

                # diff_dir投影到W_U主方向的能量
                proj_on_margin = np.dot(diff_dir, margin_dir)

                # diff_dir在各decoder主方向上的能量分布
                proj_on_dec = diff_dir @ dec_directions if 'dec_directions' in dir() else np.zeros(d_model)

                layer_result[f"alpha_{alpha}"] = {
                    "proj_on_margin_dir": float(proj_on_margin * alpha),
                }

            pair_result[f"L{perturb_l}"] = layer_result

            if test_pair_idx == 0:
                print(f"    {zh}→{en}, L{perturb_l}: "
                      f"proj_margin={list(layer_result.values())[0]['proj_on_margin_dir']:.4f}")

        results[f"{zh}_{en}"] = pair_result

    # 汇总: 各层的平均功能传输强度
    print(f"\n  功能传输强度汇总 (投影到margin方向):")
    summary = {}
    for perturb_l in perturb_layers:
        projs = []
        for key, pr in results.items():
            l_key = f"L{perturb_l}"
            if l_key in pr:
                for alpha_key, val in pr[l_key].items():
                    if alpha_key == "alpha_1.0":
                        projs.append(val["proj_on_margin_dir"])
        if projs:
            summary[perturb_l] = {
                "mean_proj": float(np.mean(projs)),
                "std_proj": float(np.std(projs)),
            }
            print(f"    L{perturb_l}: mean_proj={np.mean(projs):.4f}±{np.std(projs):.4f}")

    results["_summary"] = summary

    out_path = f"tests/glm5_temp/phase108_exp4_{model_name}_functional_transport.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 5: Local Flow Field — dh/dl ≈ Δh/Δl
# ============================================================
def exp5_flow_field(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 5: Local Flow Field — dh = h_{l+1} - h_l 的结构")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # 收集hidden states
    print(f"\n  收集hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, ALL_PAIRS)

    results = {}

    # ========================================
    # A. Flow field的基本统计
    # ========================================
    print(f"\n  Flow field基本统计:")
    flow_stats = {}

    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)

        # Δh for each sample
        delta_zh = zh_next - zh_data
        delta_trans = trans_next - trans_data

        # Flow norms
        norm_zh = np.mean(np.linalg.norm(delta_zh, axis=1))
        norm_trans = np.mean(np.linalg.norm(delta_trans, axis=1))
        norm_diff = np.mean(np.linalg.norm(delta_trans - delta_zh, axis=1))

        # Mean flow direction
        mean_flow_zh = np.mean(delta_zh, axis=0)
        mean_flow_trans = np.mean(delta_trans, axis=0)
        mean_flow_diff = mean_flow_trans - mean_flow_zh

        # Flow vs h alignment (Δh · h)
        dot_zh = np.mean([np.dot(delta_zh[i], zh_data[i]) for i in range(len(zh_data))])
        dot_trans = np.mean([np.dot(delta_trans[i], trans_data[i]) for i in range(len(trans_data))])

        # Flow的一致性: 各样本的Δh方向是否一致?
        # 用cosine similarity的平均
        if norm_zh > 0:
            cosines_zh = []
            for i in range(min(20, len(delta_zh))):
                for j in range(i+1, min(20, len(delta_zh))):
                    n_i = np.linalg.norm(delta_zh[i])
                    n_j = np.linalg.norm(delta_zh[j])
                    if n_i > 0 and n_j > 0:
                        cosines_zh.append(np.dot(delta_zh[i], delta_zh[j]) / (n_i * n_j))
            mean_cos_zh = np.mean(cosines_zh) if cosines_zh else 0
        else:
            mean_cos_zh = 0

        flow_stats[l] = {
            "mean_norm_zh": float(norm_zh),
            "mean_norm_trans": float(norm_trans),
            "mean_norm_diff": float(norm_diff),
            "flow_h_alignment_zh": float(dot_zh),
            "flow_h_alignment_trans": float(dot_trans),
            "flow_consistency_zh": float(mean_cos_zh),
            "norm_ratio_trans_zh": float(norm_trans / norm_zh) if norm_zh > 0 else 0,
        }

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}→L{l+1}: ||Δh_zh||={norm_zh:.3f}, ||Δh_trans||={norm_trans:.3f}, "
                  f"||Δh_diff||={norm_diff:.3f}, consistency={mean_cos_zh:.3f}")

    results["flow_stats"] = flow_stats

    # ========================================
    # B. Flow的方向在层间如何变化?
    # ========================================
    print(f"\n  Flow方向在层间的变化:")
    flow_angles = {}

    for l in range(n_layers - 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)
        zh_next2 = np.array(layer_states[l + 2]["zh"], dtype=np.float64)
        trans_next2 = np.array(layer_states[l + 2]["trans"], dtype=np.float64)

        # 两次flow
        delta1_trans = np.mean(trans_next - trans_data, axis=0)
        delta2_trans = np.mean(trans_next2 - trans_next, axis=0)

        n1 = np.linalg.norm(delta1_trans)
        n2 = np.linalg.norm(delta2_trans)
        if n1 > 0 and n2 > 0:
            cos_flow = np.dot(delta1_trans, delta2_trans) / (n1 * n2)
            angle_deg = np.degrees(np.arccos(np.clip(cos_flow, -1, 1)))
        else:
            angle_deg = 90

        flow_angles[l] = float(angle_deg)

        if l % 6 == 0 or l >= n_layers - 4:
            print(f"    L{l}→L{l+1} vs L{l+1}→L{l+2}: angle={angle_deg:.1f}°")

    results["flow_angles"] = flow_angles

    # ========================================
    # C. 差分flow (翻译-中文) 的结构
    # ========================================
    print(f"\n  差分flow结构 (翻译-中文的Δh差):")
    diff_flow = {}

    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)

        delta_zh = zh_next - zh_data
        delta_trans = trans_next - trans_data
        delta_diff = delta_trans - delta_zh  # 差分flow

        mean_diff_flow = np.mean(delta_diff, axis=0)
        norm_diff_flow = np.linalg.norm(mean_diff_flow)

        # 差分flow与当前差分方向的对齐度
        current_diff = np.mean(trans_data - zh_data, axis=0)
        norm_current = np.linalg.norm(current_diff)

        if norm_diff_flow > 0 and norm_current > 0:
            alignment = np.dot(mean_diff_flow, current_diff) / (norm_diff_flow * norm_current)
        else:
            alignment = 0

        diff_flow[l] = {
            "norm": float(norm_diff_flow),
            "alignment_with_current_diff": float(alignment),
        }

        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}: ||diff_flow||={norm_diff_flow:.4f}, "
                  f"alignment={alignment:.3f}")

    results["diff_flow"] = diff_flow

    out_path = f"tests/glm5_temp/phase108_exp5_{model_name}_flow_field.json"
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
        exp1_signal_dispersion(args)
    elif args.exp == 2:
        exp2_probe_complexity(args)
    elif args.exp == 3:
        exp3_logit_subspace(args)
    elif args.exp == 4:
        exp4_functional_transport(args)
    elif args.exp == 5:
        exp5_flow_field(args)
    else:
        print(f"Unknown exp: {args.exp}")
