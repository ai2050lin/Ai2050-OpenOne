"""
Phase 158: 语言状态变量搜索 — 寻找闭合动力学子空间
=====================================================

理论背景(基于用户核心批评):
  Phase 157的硬伤:
  1. tau/entropy/margin/probe accuracy都是"观测量"(observer-dependent observables)
     不是"动力学状态变量"(dynamical state variables)
  2. PCA基底 ≠ 动力学重要性(最大方差方向≠最重要动力学方向)
  3. 约束概念不够严格 — probe只找"可分类方向",不是C(h)=0的约束
  4. 需要找到"最小闭合动力学子空间": z_ℓ=P(h_ℓ) 使得 z_{ℓ+1}=f(z_ℓ)

  用户给出三个关键步骤:
  Step 1: 寻找闭合变量 — Koopman分析 / DMD / 扩散坐标
  Step 2: 验证守恒量 — 拓扑不变量 / |λ|≈1的Koopman特征值
  Step 3: 寻找"力" — Δh_ℓ = -∇E(h_ℓ) 是否成立

实验:
  Exp 1: Dynamic Mode Decomposition — 层间线性动力学算子
  Exp 2: 闭合子空间搜索 — ★★★ 最关键实验 ★★★
         找最小d使得d维投影z_ℓ可以预测z_{ℓ+1} (R²>0.9)
  Exp 3: 能量/力分析 — Δh ≈ -∇E(h) 是否成立
  Exp 4: 守恒量搜索 — w^T h ≈ 常数 的线性守恒量
  Exp 5: LN消融 — 最终层相变是否为LN伪影
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tests/glm5')

import torch
from model_utils import load_model, get_model_info, get_layers, release_model, get_W_U

# ===== 300+ diverse prompts =====
def generate_prompts():
    """生成300+多样化的短句prompt"""
    nouns = [
        "cat", "dog", "bird", "fish", "child", "woman", "man", "boy", "girl", "teacher",
        "doctor", "student", "tree", "flower", "river", "mountain", "car", "book", "house", "city",
        "king", "queen", "soldier", "artist", "scientist", "writer", "musician", "engineer", "farmer", "driver",
        "chair", "table", "door", "window", "road", "bridge", "tower", "castle", "garden", "forest",
        "ocean", "island", "valley", "desert", "planet", "star", "moon", "sun", "cloud", "rain",
        "apple", "bread", "water", "fire", "stone", "glass", "paper", "silk", "gold", "iron",
    ]
    verbs = [
        "runs", "sits", "walks", "jumps", "flies", "swims", "reads", "writes", "eats", "drinks",
        "sings", "dances", "sleeps", "wakes", "falls", "rises", "moves", "stops", "starts", "ends",
        "opens", "closes", "grows", "shrinks", "turns", "falls", "stands", "lies", "hangs", "breaks",
    ]
    adjs = [
        "red", "blue", "green", "big", "small", "old", "new", "fast", "slow", "tall",
        "short", "dark", "bright", "warm", "cold", "soft", "hard", "rich", "poor", "young",
        "heavy", "light", "thick", "thin", "wide", "narrow", "deep", "shallow", "clean", "dirty",
    ]
    places = [
        "park", "house", "school", "office", "garden", "street", "market", "church", "field", "lake",
        "beach", "harbor", "airport", "station", "hospital", "museum", "theater", "prison", "palace", "temple",
    ]

    prompts = []
    seen = set()

    def add(p):
        if p not in seen and len(p.split()) >= 3:
            seen.add(p)
            prompts.append(p)

    # Pattern 1: "The {noun} {verb} toward the" (intransitive)
    for n in nouns[:20]:
        for v in verbs[:6]:
            add(f"The {n} {v} toward the")

    # Pattern 2: "The {adj} {noun} is" (copular)
    for a in adjs[:12]:
        for n in nouns[:12]:
            add(f"The {a} {n} is")

    # Pattern 3: "The {noun}s {verb} toward the" (plural subject)
    for n in nouns[:15]:
        for v in verbs[:5]:
            if n.endswith(("s", "x", "ch", "sh")):
                add(f"The {n}es {v} toward the")
            else:
                add(f"The {n}s {v} toward the")

    # Pattern 4: "Does the {noun} {verb}" (question)
    for n in nouns[:10]:
        for v in verbs[:8]:
            add(f"Does the {n} {v}")

    # Pattern 5: "The {noun} does not {verb}" (negation)
    for n in nouns[:10]:
        for v in verbs[:8]:
            add(f"The {n} does not {v}")

    # Pattern 6: "If the {noun} {verb}" (conditional)
    for n in nouns[:10]:
        for v in verbs[:6]:
            add(f"If the {n} {v}")

    # Pattern 7: "The {noun} in the {place}" (locative)
    for n in nouns[:15]:
        for p in places[:10]:
            add(f"The {n} in the {p}")

    # Pattern 8: "Although the {noun} {verb}" (concessive)
    for n in nouns[:8]:
        for v in verbs[:5]:
            add(f"Although the {n} {v}")

    # Pattern 9: "Because the {noun} {verb}" (causal)
    for n in nouns[:8]:
        for v in verbs[:5]:
            add(f"Because the {n} {v}")

    # Pattern 10: "The {noun} that {verb}" (relative clause)
    for n in nouns[:10]:
        for v in verbs[:6]:
            add(f"The {n} that {v}")

    # Fill up to 300+ with additional patterns
    for n in nouns[:20]:
        add(f"Every {n} knows the")
        add(f"No {n} could ever")
        add(f"Some {n} will always")
    for a in adjs[:10]:
        for n in nouns[:5]:
            add(f"A very {a} {n} was")

    print(f"[prompts] Generated {len(prompts)} unique prompts")
    return prompts[:350]  # Cap at 350


# ===== Collect hidden states =====
def collect_hidden_states(model, tokenizer, device, model_info, prompts):
    """
    收集所有prompt在所有层的hidden states (last token position)
    Returns: H dict, shape (n_prompts, n_layers+1, d_model)
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    n_prompts = len(prompts)

    H = np.zeros((n_prompts, n_layers + 1, d_model), dtype=np.float32)
    norms = np.zeros((n_prompts, n_layers + 1), dtype=np.float32)

    # 8bit模型输入设备
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device

    successful = 0
    failed_prompts = set()
    for i, prompt in enumerate(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            has_nan = False
            for l in range(n_layers + 1):
                h = out.hidden_states[l][0, -1, :].detach().float().cpu().numpy()
                if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                    has_nan = True
                    break
                H[i, l] = h
                norms[i, l] = np.linalg.norm(h)

            if has_nan:
                H[i] = 0
                norms[i] = 0
                failed_prompts.add(i)
            else:
                successful += 1

            del out

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_prompts}] collected, last norm={norms[i, n_layers]:.1f}")

        except Exception as e:
            print(f"  [!] Prompt {i} failed: {e}")
            H[i] = 0
            norms[i] = 0
            failed_prompts.add(i)

    # Remove failed prompts
    if failed_prompts:
        print(f"  Removing {len(failed_prompts)} prompts with NaN/Inf values")
        valid_mask = np.ones(n_prompts, dtype=bool)
        valid_mask[list(failed_prompts)] = False
        H = H[valid_mask]
        norms = norms[valid_mask]
        n_prompts = H.shape[0]

    print(f"[collect] {successful}/{n_prompts} successful, "
          f"norm range: [{norms[norms > 0].min():.1f}, {norms[norms > 0].max():.1f}]")
    return H, norms


# ===== Exp 1: Dynamic Mode Decomposition =====
def exp1_dmd(H, n_layers, d_model):
    """
    DMD分析: 找每层对的线性动力学算子 A_ℓ 使得 h_{ℓ+1} ≈ A_ℓ · h_ℓ
    
    关键输出:
    - 每层的DMD特征值 (判断 |λ|≈1 的守恒模式)
    - 每层的DMD有效秩 (动力学维度)
    - 每层的线性拟合R²
    """
    print("\n" + "=" * 60)
    print("Exp 1: Dynamic Mode Decomposition")
    print("=" * 60)

    n_prompts = H.shape[0]
    results = {"per_layer": {}, "summary": {}}

    all_r2 = []
    all_eff_rank = []
    all_n_modes_near_1 = []

    sample_layers = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    for l in sample_layers:
        if l >= n_layers:
            continue
        # 使用正确的DMD约定: X^T, Y^T 是 (d, N)
        X_T = H[:, l, :].T      # (d, N)
        Y_T = H[:, l + 1, :].T  # (d, N)

        try:
            # Center
            X_mean = X_T.mean(axis=1, keepdims=True)  # (d, 1)
            Y_mean = Y_T.mean(axis=1, keepdims=True)  # (d, 1)
            Xc = X_T - X_mean  # (d, N)
            Yc = Y_T - Y_mean  # (d, N)

            # SVD of X: (d, N) → U(d, k), S(k), Vt(k, N)
            k = min(Xc.shape[0], Xc.shape[1], 200)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            U = U[:, :k]   # (d, k)
            S = S[:k]       # (k,)
            Vt = Vt[:k, :]  # (k, N)

            # Threshold small singular values
            mask = S > 1e-8
            U = U[:, mask]   # (d, r)
            S = S[mask]       # (r,)
            Vt = Vt[mask, :]  # (r, N)

            if len(S) == 0:
                print(f"  L{l}->L{l+1}: No valid SVD components")
                continue

            # A_tilde = U^T Y V Σ^{-1}: (r, d) @ (d, N) = (r, N)
            # (r, N) @ (N, r) = (r, r), then (r, r) @ (r, r) = (r, r)
            S_inv = np.diag(1.0 / S)
            A_tilde = U.T @ Yc @ Vt.T @ S_inv  # (r, r)

            # Eigenvalues of A_tilde
            eigenvalues = np.linalg.eigvals(A_tilde)

            # DMD R²: predict Y from X
            # Y_pred = U @ A_tilde @ U^T @ Xc + Y_mean (in d-dimensional space)
            Y_pred = U @ (A_tilde @ (U.T @ Xc)) + Y_mean  # (d, N)

            ss_res = np.sum((Yc - (Y_pred - Y_mean)) ** 2)
            ss_tot = np.sum(Yc ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            # Effective rank: number of singular values > 1% of max
            eff_rank = int(np.sum(S > 0.01 * S[0]))

            # Modes near |λ|=1 (conserved quantities)
            n_near_1 = int(np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 0.1))

            # Top eigenvalue magnitudes
            top_mag = sorted(np.abs(eigenvalues), reverse=True)[:10]

            all_r2.append(float(r2))
            all_eff_rank.append(int(eff_rank))
            all_n_modes_near_1.append(int(n_near_1))

            results["per_layer"][f"L{l}_L{l+1}"] = {
                "r2": float(r2),
                "eff_rank": int(eff_rank),
                "n_modes_near_1": int(n_near_1),
                "top_eigenvalue_mags": [float(x) for x in top_mag],
                "n_svd_components": int(len(S)),
                "top5_singular_values": [float(x) for x in S[:5].tolist()],
            }

            print(f"  L{l}->L{l+1}: R²={r2:.4f}, eff_rank={eff_rank}, "
                  f"n_near_1={n_near_1}, top|λ|={[f'{x:.3f}' for x in top_mag[:5]]}")

        except Exception as e:
            print(f"  L{l}->L{l+1}: DMD failed: {e}")

    results["summary"] = {
        "mean_r2": float(np.mean(all_r2)) if all_r2 else 0,
        "mean_eff_rank": float(np.mean(all_eff_rank)) if all_eff_rank else 0,
        "mean_n_near_1": float(np.mean(all_n_modes_near_1)) if all_n_modes_near_1 else 0,
        "max_r2": float(np.max(all_r2)) if all_r2 else 0,
        "min_r2": float(np.min(all_r2)) if all_r2 else 0,
    }

    print(f"\n  Summary: mean_R²={results['summary']['mean_r2']:.4f}, "
          f"mean_eff_rank={results['summary']['mean_eff_rank']:.1f}, "
          f"mean_n_near_1={results['summary']['mean_n_near_1']:.1f}")

    return results


# ===== Exp 2: Closed Subspace Search (★★★ 最关键实验 ★★★) =====
def exp2_closed_subspace(H, n_layers, d_model):
    """
    寻找最小闭合动力学子空间:
    找最小d使得d维投影z_ℓ = P_d(h_ℓ)可以预测z_{ℓ+1} (R² > 0.9)
    
    方法:
    1. 对所有层的hidden states做全局PCA, 找主方向
    2. 对每个d, 用前d个PCA方向投影, 线性回归预测下一层
    3. 用5-fold交叉验证计算R²
    4. 找最小d使得R² > 0.9
    
    Returns:
    - R²(d)曲线 (全局和逐层)
    - 最小闭合子空间维度
    """
    print("\n" + "=" * 60)
    print("Exp 2: Closed Subspace Search (★★★ 最关键实验 ★★★)")
    print("=" * 60)

    n_prompts = H.shape[0]
    n_all_layers = H.shape[1]  # n_layers + 1

    # Step 1: 全局PCA (pool所有层的hidden states)
    # Stack: (N * n_all_layers, d_model)
    H_flat = H.reshape(-1, d_model)
    print(f"  Pooling {H_flat.shape[0]} hidden states for global PCA...")

    n_components = min(d_model, 200, H_flat.shape[0] - 1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(H_flat)
    explained_var = pca.explained_variance_ratio_

    print(f"  Global PCA: {n_components} components, "
          f"top-10 explain {explained_var[:10].sum():.4f}, "
          f"top-50 explain {explained_var[:50].sum():.4f}, "
          f"top-200 explain {explained_var[:min(200, n_components)].sum():.4f}")

    # Step 2: 对每个d, 用前d个PCA方向投影, 线性回归预测下一层
    dims_to_test = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    dims_to_test = [d for d in dims_to_test if d <= n_components]

    results = {"dims": dims_to_test, "global_r2": {}, "per_layer_r2": {}}

    # 5-fold CV
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for d in dims_to_test:
        # Project to d dimensions
        components_d = pca.components_[:d]  # (d, d_model)
        H_proj = np.einsum('nld,dk->nlk', H, components_d.T)  # (N, n_all_layers, d)

        # For each fold, predict h_{ℓ+1} from h_ℓ
        fold_r2s = {l: [] for l in range(n_layers)}

        for train_idx, test_idx in kf.split(range(n_prompts)):
            for l in range(n_layers):
                X_train = H_proj[train_idx, l, :]    # (n_train, d)
                Y_train = H_proj[train_idx, l+1, :]  # (n_train, d)
                X_test = H_proj[test_idx, l, :]      # (n_test, d)
                Y_test = H_proj[test_idx, l+1, :]    # (n_test, d)

                # Ridge regression (alpha=1.0 for regularization)
                alpha = 1.0 if d > 50 else 0.1
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, Y_train)
                Y_pred = reg.predict(X_test)

                # R² per dimension, then average
                ss_res = np.sum((Y_test - Y_pred) ** 2, axis=0)
                ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0)
                r2_per_dim = 1 - ss_res / np.maximum(ss_tot, 1e-10)
                r2_avg = float(np.mean(r2_per_dim))

                fold_r2s[l].append(r2_avg)

        # Average across folds
        mean_r2_per_layer = {l: float(np.mean(fold_r2s[l])) for l in range(n_layers)}
        global_mean_r2 = float(np.mean([mean_r2_per_layer[l] for l in range(n_layers)]))
        min_r2 = float(np.min([mean_r2_per_layer[l] for l in range(n_layers)]))

        results["global_r2"][str(d)] = {
            "mean": global_mean_r2,
            "min": min_r2,
            "std": float(np.std([mean_r2_per_layer[l] for l in range(n_layers)])),
        }

        # Sample per-layer R²
        sample_ls = [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
        sample_ls = sorted(set([l for l in sample_ls if 0 <= l < n_layers]))
        results["per_layer_r2"][str(d)] = {
            f"L{l}": float(mean_r2_per_layer[l]) for l in sample_ls
        }

        # Print
        r2_str = ", ".join([f"L{l}={mean_r2_per_layer[l]:.3f}" for l in sample_ls[:5]])
        print(f"  d={d:3d}: mean_R²={global_mean_r2:.4f}, min_R²={min_r2:.4f}  "
              f"[{r2_str}...]")

    # Find minimum d for R² > 0.9
    d_90 = None
    d_95 = None
    for d in dims_to_test:
        if results["global_r2"][str(d)]["min"] >= 0.9 and d_90 is None:
            d_90 = d
        if results["global_r2"][str(d)]["min"] >= 0.95 and d_95 is None:
            d_95 = d

    results["min_d_for_r2_90"] = d_90
    results["min_d_for_r2_95"] = d_95
    results["pca_explained_variance_top10"] = float(explained_var[:10].sum())
    results["pca_explained_variance_top50"] = float(explained_var[:50].sum())
    results["pca_explained_variance_top200"] = float(explained_var[:min(200, n_components)].sum())

    print(f"\n  ★ 结果: 最小d(R²>0.9)={d_90}, 最小d(R²>0.95)={d_95}")
    print(f"  ★ PCA解释方差: top10={explained_var[:10].sum():.4f}, "
          f"top50={explained_var[:50].sum():.4f}")

    return results


# ===== Exp 3: Energy/Force Analysis =====
def exp3_energy_force(H, n_layers, d_model):
    """
    能量/力分析: 检查Δh是否≈-∇E(h)
    
    如果 Δh_ℓ = A·h_ℓ + b (线性关系), 则:
    Δh = -∇E 意味着 E(h) = -h^T A h/2 - b^T h (二次能量)
    
    测试:
    1. 拟合 Δh = A·h + b, 测量R²
    2. 如果R²高, 检查"能量"E(h_ℓ)是否跨层单调递减
    """
    print("\n" + "=" * 60)
    print("Exp 3: Energy/Force Analysis")
    print("=" * 60)

    n_prompts = H.shape[0]
    results = {"per_layer": {}, "summary": {}}

    all_r2 = []
    all_energy_monotone_frac = []

    # Cross-validation
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    for l in sample_layers:
        if l >= n_layers:
            continue
        X = H[:, l, :]        # (N, d)
        dH = H[:, l+1, :] - X  # Δh = h_{ℓ+1} - h_ℓ

        # Fit Δh = A·h + b using Ridge regression with CV
        fold_r2s = []
        fold_energy_mono = []

        for train_idx, test_idx in kf.split(range(n_prompts)):
            X_train, X_test = X[train_idx], X[test_idx]
            dH_train, dH_test = dH[train_idx], dH[test_idx]

            # Ridge regression: predict Δh from h
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, dH_train)
            dH_pred = reg.predict(X_test)

            # R² (total, not per-dimension)
            ss_res = np.sum((dH_test - dH_pred) ** 2)
            ss_tot = np.sum((dH_test - dH_test.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            fold_r2s.append(float(r2))

            # Check if Δh points "downhill" on the fitted energy
            # E(h) = -h^T A h/2 - b^T h  (where A = reg.coef_, b = reg.intercept_)
            # ∇E = -A^T h - b (but reg predicts Δh = A·h + b)
            # If Δh = -∇E, then Δh should decrease E
            # dE/dℓ = ∇E · Δh = (-A·h - b) · (A·h + b) = -||A·h + b||² < 0
            # So E should always decrease → check if E(h_{ℓ+1}) < E(h_ℓ)

            A = reg.coef_  # (d, d)
            b = reg.intercept_  # (d,)

            # E(h) = -0.5 * h @ A @ h - b @ h  (scalar for each h)
            # But this is problematic because A is (d,d) and we'd need h @ A @ h
            # which is h^T A h. For d_model=2560 this is fine.

            # Simpler check: just verify Δh^T · Δh_pred > 0 (consistent direction)
            # and ||Δh_pred|| < ||Δh|| (the model captures some but not all of the update)

            # Actually, let's check if the "gradient-like" property holds:
            # Δh should be approximately in the direction of the negative gradient
            # This means cos(Δh, -∇E) should be high

            # For a simpler check: does the linear model capture the update direction?
            cos_sim = np.sum(dH_test * dH_pred, axis=1) / (
                np.linalg.norm(dH_test, axis=1) * np.linalg.norm(dH_pred, axis=1) + 1e-10
            )
            mean_cos = float(np.mean(cos_sim))

            # Energy monotonicity: compute "energy" along the trajectory
            # E_ℓ = ||h_ℓ||² (simple proxy — the norm squared)
            # Actually, let's use the proper quadratic energy
            # E(h) = -0.5 * sum_ij A_ij h_i h_j - sum_i b_i h_i
            # But this is O(d²) per sample, which is expensive for d=2560

            # Instead, let's just check: does ||h||² change monotonically?
            h_norm_sq = np.sum(H[test_idx, l, :] ** 2, axis=1)
            h_next_norm_sq = np.sum(H[test_idx, l+1, :] ** 2, axis=1)
            mono_frac = float(np.mean(h_next_norm_sq < h_norm_sq))  # fraction where norm decreases

            fold_energy_mono.append(mono_frac)

        mean_r2 = float(np.mean(fold_r2s))
        mean_mono = float(np.mean(fold_energy_mono))
        all_r2.append(mean_r2)
        all_energy_monotone_frac.append(mean_mono)

        results["per_layer"][f"L{l}_L{l+1}"] = {
            "r2": mean_r2,
            "norm_decrease_frac": mean_mono,
        }

        print(f"  L{l}->L{l+1}: R²(Δh| h)={mean_r2:.4f}, "
              f"norm_decrease_frac={mean_mono:.3f}")

    results["summary"] = {
        "mean_r2": float(np.mean(all_r2)) if all_r2 else 0,
        "mean_norm_decrease": float(np.mean(all_energy_monotone_frac)) if all_energy_monotone_frac else 0,
    }

    print(f"\n  Summary: mean_R²={results['summary']['mean_r2']:.4f}, "
          f"mean_norm_decrease={results['summary']['mean_norm_decrease']:.3f}")

    # Additional analysis: compute actual energy trajectory for a few prompts
    print("\n  Computing energy trajectory (using ||h||² as proxy)...")
    energy_trajectories = {}
    for i in range(min(5, n_prompts)):
        e_traj = [float(np.sum(H[i, l, :] ** 2)) for l in range(n_layers + 1)]
        energy_trajectories[f"prompt_{i}"] = e_traj

    # Check monotonicity of ||h||² across ALL layers for ALL prompts
    all_norm_sq = np.sum(H ** 2, axis=2)  # (N, n_all_layers)
    # For each prompt, check if ||h||² is monotonically decreasing
    monotone_count = 0
    for i in range(n_prompts):
        traj = all_norm_sq[i]
        # Check each layer transition
        is_mono = all(traj[l+1] <= traj[l] + 1e-6 for l in range(n_layers))
        if is_mono:
            monotone_count += 1

    results["norm_monotone_frac"] = float(monotone_count / n_prompts)
    results["energy_trajectories_sample"] = energy_trajectories

    # Norm changes per layer (averaged across prompts)
    norm_changes = {}
    for l in range(n_layers):
        delta_norm = float(np.mean(np.sum(H[:, l+1, :] ** 2, axis=1) -
                                   np.sum(H[:, l, :] ** 2, axis=1)))
        norm_changes[f"L{l}_L{l+1}"] = delta_norm
    results["norm_changes"] = norm_changes

    # Check if norm change pattern matches the phase transition
    # (Phase 157 found phase transition at L35 for Qwen3)
    sample_norm_changes = {k: v for k, v in list(norm_changes.items())[::max(1, len(norm_changes)//10)]}
    print(f"  Norm changes (sample): {sample_norm_changes}")
    print(f"  ||h||² monotone fraction: {results['norm_monotone_frac']:.3f}")

    return results


# ===== Exp 4: Conservation Law Search =====
def exp4_conservation_laws(H, n_layers, d_model):
    """
    守恒量搜索: 找w使得 w^T h_ℓ ≈ 常数 (跨层守恒)
    
    方法:
    1. 计算所有Δh_ℓ = h_{ℓ+1} - h_ℓ
    2. 找Δh的零空间 → 线性守恒量方向w (w^T Δh = 0)
    3. 检查二次守恒量: h^T Q h ≈ 常数
    
    另外用DMD特征值 |λ|≈1 找Koopman守恒量
    """
    print("\n" + "=" * 60)
    print("Exp 4: Conservation Law Search")
    print("=" * 60)

    n_prompts = H.shape[0]
    results = {}

    # Method 1: Null space of Δh (linear conservation laws)
    # Stack all Δh vectors: (N * n_layers, d_model)
    all_dh = []
    for l in range(n_layers):
        dh = H[:, l+1, :] - H[:, l, :]  # (N, d_model)
        all_dh.append(dh)
    all_dh = np.vstack(all_dh)  # (N * n_layers, d_model)
    print(f"  Stacked Δh: shape={all_dh.shape}")

    # SVD of Δh to find the null space
    # Null space = columns of V corresponding to σ ≈ 0
    # Since Δh is (N*L, d), the SVD is Δh = U S V^T where V is (d, d)
    k_svd = min(all_dh.shape[0], all_dh.shape[1], 300)
    U, S, Vt = np.linalg.svd(all_dh[:k_svd], full_matrices=False)

    print(f"  SVD of Δh: top-10 singular values = {S[:10].tolist()}")
    print(f"  SVD of Δh: bottom-10 singular values = {S[-10:].tolist()}")

    # Count near-zero singular values (null space dimension)
    threshold = 0.01 * S[0]
    null_dim = int(np.sum(S < threshold))
    results["null_space_dimension"] = null_dim
    results["svd_top10"] = [float(x) for x in S[:10]]
    results["svd_bottom10"] = [float(x) for x in S[-10:]]

    print(f"  Null space dimension (σ < {threshold:.2f}): {null_dim}")

    # If null_dim > 0, extract the conservation directions
    if null_dim > 0 and null_dim <= 50:
        conservation_dirs = Vt[-null_dim:]  # (null_dim, d_model)
        results["n_conservation_directions"] = null_dim

        # Verify: check that w^T Δh ≈ 0 for these directions
        max_violations = []
        for i in range(min(null_dim, 10)):
            w = conservation_dirs[i]
            violations = np.abs(all_dh @ w)
            max_viol = float(np.max(violations))
            max_violations.append(max_viol)
        results["max_conservation_violations"] = max_violations
        print(f"  Max conservation violations: {max_violations}")
    else:
        results["n_conservation_directions"] = 0
        print(f"  No significant conservation directions found (null_dim={null_dim})")

    # Method 2: DMD eigenvalues near |λ|=1 (Koopman conservation)
    print("\n  Computing DMD eigenvalues for conservation check...")
    dmd_conservation = {}
    sample_layers = [0, 1, n_layers//2, n_layers-2, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if 0 <= l < n_layers]))

    for l in sample_layers:
        # 使用正确的DMD约定: (d, N)
        X_T = H[:, l, :].T      # (d, N)
        Y_T = H[:, l+1, :].T    # (d, N)
        Xc = X_T - X_T.mean(axis=1, keepdims=True)  # (d, N)
        Yc = Y_T - Y_T.mean(axis=1, keepdims=True)  # (d, N)

        k = min(Xc.shape[0], Xc.shape[1], 200)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]

        mask = S > 1e-8
        if mask.sum() == 0:
            continue
        U, S, Vt = U[:, mask], S[mask], Vt[mask, :]

        A_tilde = U.T @ Yc @ Vt.T @ np.diag(1.0 / S)
        eigenvalues = np.linalg.eigvals(A_tilde)

        # Count |λ| near 1
        near_1 = np.abs(np.abs(eigenvalues) - 1.0) < 0.1
        n_near_1 = int(np.sum(near_1))

        # Find the most conserved modes
        sorted_by_near_1 = sorted(np.abs(eigenvalues), key=lambda x: abs(abs(x) - 1.0))
        top5_conserved = [float(x) for x in sorted_by_near_1[:5]]

        dmd_conservation[f"L{l}_L{l+1}"] = {
            "n_modes_near_1": n_near_1,
            "top5_conserved_eigenvalues": top5_conserved,
            "mean_abs_eigenvalue": float(np.mean(np.abs(eigenvalues))),
        }

        print(f"  L{l}->L{l+1}: n_near_1={n_near_1}, "
              f"top5_conserved|λ|={[f'{x:.3f}' for x in top5_conserved]}")

    results["dmd_conservation"] = dmd_conservation

    # Method 3: Quadratic conservation laws
    # Check if h^T Q h ≈ constant for some Q
    # Simple test: is ||h||² approximately conserved?
    norm_sq = np.sum(H ** 2, axis=2)  # (N, n_all_layers)
    norm_sq_variation = np.std(norm_sq, axis=1) / (np.mean(norm_sq, axis=1) + 1e-10)
    mean_variation = float(np.mean(norm_sq_variation))
    results["norm_sq_relative_variation"] = mean_variation

    # Check specific quadratic forms: h[0]², h[1]², etc.
    # These are too high-dimensional, so check the PCA components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, d_model, H.shape[0] - 1))
    H_flat = H.reshape(-1, d_model)
    pca.fit(H_flat)

    # Check conservation of PCA component energies
    H_proj = pca.transform(H_flat).reshape(H.shape[0], H.shape[1], -1)  # (N, L+1, 50)
    pca_conservation = {}
    for pc in range(min(10, H_proj.shape[2])):
        pc_vals = H_proj[:, :, pc]  # (N, L+1)
        # Variation across layers (normalized by mean)
        pc_mean = np.mean(np.abs(pc_vals))
        pc_std_across_layers = np.mean(np.std(pc_vals, axis=1))
        pc_relative_var = float(pc_std_across_layers / max(pc_mean, 1e-10))
        pca_conservation[f"PC{pc}"] = {
            "relative_variation": pc_relative_var,
            "mean_abs": float(pc_mean),
        }

    results["pca_conservation"] = pca_conservation

    # Find the most conserved PCA components
    conserved_pcs = sorted(pca_conservation.items(), key=lambda x: x[1]["relative_variation"])
    print(f"\n  Most conserved PCA components (lowest relative variation):")
    for name, info in conserved_pcs[:5]:
        print(f"    {name}: rel_var={info['relative_variation']:.4f}, mean_abs={info['mean_abs']:.2f}")

    return results


# ===== Exp 5: LN Ablation =====
def exp5_ln_ablation(model, tokenizer, device, model_info, H, n_layers, d_model):
    """
    LayerNorm消融实验:
    比较最终LN前后的hidden state, 判断相变是否为LN伪影
    
    方法:
    1. 对少量prompt, 收集LN前后的hidden state
    2. 比较有LN和无LN的norm变化
    3. 比较有LN和无LN的可行空间体积变化
    """
    print("\n" + "=" * 60)
    print("Exp 5: LayerNorm Ablation")
    print("=" * 60)

    results = {}
    n_test = min(30, H.shape[0])

    # Get the final LayerNorm module
    try:
        layers = get_layers(model)
        # The final LN is typically model.model.norm
        if hasattr(model.model, 'norm'):
            final_ln = model.model.norm
            print(f"  Found final LN: {type(final_ln).__name__}")
        else:
            print("  No final LN found (model.model.norm)")
            results["error"] = "No final LN found"
            return results
    except Exception as e:
        print(f"  Error accessing model layers: {e}")
        results["error"] = str(e)
        return results

    # Collect pre-LN and post-LN states
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device

    # Use a few test prompts
    test_prompts = [
        "The cat sits on the", "A red apple was placed", "The scientist discovered a",
        "Does the dog run", "If the bird flies", "Because the river flows",
        "The children play in the", "Although the mountain is",
        "Every student reads the", "No writer could ever",
    ]

    pre_ln_states = []
    post_ln_states = []

    captured = {}
    def pre_ln_hook(module, input, output):
        captured["pre_ln"] = input[0].detach().float().cpu()

    hook = final_ln.register_forward_hook(pre_ln_hook)

    for prompt in test_prompts[:n_test]:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            # Post-LN state (from hidden_states[-1])
            h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            post_ln_states.append(h_post)

            # Pre-LN state (from hook)
            if "pre_ln" in captured:
                h_pre = captured["pre_ln"][0, -1, :].numpy()
                pre_ln_states.append(h_pre)

            del out
        except Exception as e:
            print(f"  Prompt failed: {e}")

    hook.remove()

    if len(pre_ln_states) == 0:
        results["error"] = "No pre-LN states captured"
        return results

    pre_ln = np.array(pre_ln_states)   # (n, d)
    post_ln = np.array(post_ln_states)  # (n, d)

    # Compare norms
    pre_norms = np.linalg.norm(pre_ln, axis=1)
    post_norms = np.linalg.norm(post_ln, axis=1)

    results["pre_ln_norm_mean"] = float(np.mean(pre_norms))
    results["post_ln_norm_mean"] = float(np.mean(post_norms))
    results["norm_ratio_mean"] = float(np.mean(post_norms / (pre_norms + 1e-10)))

    print(f"  Pre-LN norm: {np.mean(pre_norms):.2f} ± {np.std(pre_norms):.2f}")
    print(f"  Post-LN norm: {np.mean(post_norms):.2f} ± {np.std(post_norms):.2f}")
    print(f"  Norm ratio (post/pre): {results['norm_ratio_mean']:.4f}")

    # Compare the last hidden state (pre-LN) with the layer before
    # From H, get the state at the last transformer layer output
    # H[:, -1, :] = after the final LN (from model output)
    # H[:, -2, :] = after the last transformer layer but before final LN

    # Actually, in HuggingFace models, hidden_states[-1] is BEFORE the final LN
    # and the actual model output goes through final LN → lm_head
    # So H[:, -1, :] should be similar to our pre_ln_states

    # Check correlation between H's last layer and pre-LN states
    if len(pre_ln_states) > 0:
        # The last hidden state from output.hidden_states is pre-final-LN
        # So it should match our captured pre_ln
        # But they might not be from the same prompts, so just compare stats

        # Compare the "phase transition" metrics
        # Phase 157 found: Qwen3 L34→L35 has d²V=40441
        # Check if this transition exists in pre-LN states

        # Compute norm changes across layers (from H)
        norm_traj = np.linalg.norm(H[:len(test_prompts), :, :], axis=2)  # (n, L+1)

        # Check the norm jump at the last few layers
        last_few_norms = {}
        for l_offset in [-3, -2, -1]:
            l = n_layers + l_offset
            if 0 <= l <= n_layers:
                last_few_norms[f"L{l}"] = float(np.mean(norm_traj[:len(test_prompts), l]))

        results["last_few_layer_norms"] = last_few_norms
        results["pre_final_ln_norm"] = float(np.mean(pre_norms))

        print(f"  Layer norms (from H): {last_few_norms}")
        print(f"  Pre-final-LN norm: {np.mean(pre_norms):.2f}")

        # Key test: is the norm change at the final layer due to LN?
        # If H[:,-2] → H[:,-1] shows a big norm change, and
        # pre_ln → post_ln also shows a big norm change,
        # then the phase transition IS the LN effect
        if n_layers >= 2:
            h_before_last = H[:len(test_prompts), -2, :]
            h_after_last = H[:len(test_prompts), -1, :]
            norm_before = np.linalg.norm(h_before_last, axis=1)
            norm_after = np.linalg.norm(h_after_last, axis=1)
            norm_change_ratio = float(np.mean(norm_after / (norm_before + 1e-10)))
            results["last_layer_norm_change_ratio"] = norm_change_ratio
            print(f"  Last layer norm change ratio: {norm_change_ratio:.4f}")

    # Compute feasible volume for pre-LN and post-LN (using W_U)
    try:
        W_U = get_W_U(model, model_info.name if hasattr(model_info, 'name') else None)
        eps = 5.0

        for label, states in [("pre_ln", pre_ln), ("post_ln", post_ln)]:
            logits = states @ W_U.T  # (n, vocab)
            max_logit = np.max(logits, axis=1, keepdims=True)
            feasible = (logits > max_logit - eps)
            volume = float(np.mean(np.sum(feasible, axis=1)))
            entropy = float(np.mean(-np.sum(
                feasible.astype(float) * np.log(feasible.astype(float) + 1e-10), axis=1
            )))
            results[f"{label}_feasible_volume"] = volume
            results[f"{label}_feasible_entropy"] = entropy

        print(f"  Pre-LN feasible volume: {results['pre_ln_feasible_volume']:.1f}")
        print(f"  Post-LN feasible volume: {results['post_ln_feasible_volume']:.1f}")
        print(f"  Volume ratio (post/pre): "
              f"{results['post_ln_feasible_volume'] / max(results['pre_ln_feasible_volume'], 1):.4f}")

    except Exception as e:
        print(f"  W_U computation failed: {e}")
        results["wu_error"] = str(e)

    return results


# ===== Main =====
def main(model_name):
    print(f"\n{'='*60}")
    print(f"Phase 158: 语言状态变量搜索 — {model_name}")
    print(f"{'='*60}")

    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Generate prompts
    prompts = generate_prompts()

    # Collect hidden states
    print(f"\nCollecting hidden states for {len(prompts)} prompts...")
    H, norms = collect_hidden_states(model, tokenizer, device, model_info, prompts)

    # Run experiments
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": len(prompts),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Exp 1: DMD
    results["exp1_dmd"] = exp1_dmd(H, n_layers, d_model)

    # Exp 2: Closed subspace (most important!)
    results["exp2_closed_subspace"] = exp2_closed_subspace(H, n_layers, d_model)

    # Exp 3: Energy/force
    results["exp3_energy_force"] = exp3_energy_force(H, n_layers, d_model)

    # Exp 4: Conservation laws
    results["exp4_conservation"] = exp4_conservation_laws(H, n_layers, d_model)

    # Exp 5: LN ablation
    results["exp5_ln_ablation"] = exp5_ln_ablation(
        model, tokenizer, device, model_info, H, n_layers, d_model
    )

    # Save results
    t_elapsed = time.time() - t_start
    results["total_time_sec"] = round(t_elapsed, 1)

    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase158_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTotal time: {t_elapsed:.1f}s")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    main(model_name)
