"""
Phase 159: Language Phase Space Reconstruction — Koopman Eigenfunction Search
=============================================================================

Theoretical basis (from user's Phase 158 review):
  Phase 158 found d≈30 closed subspace using PCA, but:
  1. PCA ≠ optimal dynamics basis (max variance ≠ max dynamics importance)
  2. Need TRUE Koopman eigenfunctions: φ(h_{ℓ+1}) = λ φ(h_ℓ)
  3. These are genuine state variables of the language dynamical system
  4. Language = "low-dimensional phase space trajectory + high-dimensional readout"
  5. Key shift: from "constraint analysis" to "attractor dynamics / phase space theory"

  User's 4 key directions:
  - Phase space reconstruction (attractors, bifurcations)
  - Koopman spectrum (true closed variables, NOT PCA)
  - Manifold topology (persistent homology, connectivity)
  - Dynamical decomposition (slow/fast, parallel/perpendicular)

Critical insight:
  d≈30 R² peak-then-decline means there ARE dynamically irrelevant directions.
  But PCA is not the right basis to find them. Koopman eigenfunctions are.

Experiments:
  Exp 1: EDMD — Koopman Eigenfunction Search (★★★)
  Exp 2: Koopman vs PCA Prediction Comparison (★★★ CORE ★★★)
  Exp 3: Slow/Fast Manifold Decomposition
  Exp 4: Attractor Structure & Trajectory Analysis
  Exp 5: Koopman Spectral Evolution (Bifurcation Analysis)
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


# ===== 500+ diverse prompts (increased for robustness) =====
def generate_prompts():
    """生成500+多样化的短句prompt"""
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
        "opens", "closes", "grows", "shrinks", "turns", "stands", "lies", "hangs", "breaks", "builds",
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

    # Pattern 1-10: Same as Phase 158 but more
    for n in nouns[:25]:
        for v in verbs[:8]:
            add(f"The {n} {v} toward the")
    for a in adjs[:15]:
        for n in nouns[:15]:
            add(f"The {a} {n} is")
    for n in nouns[:20]:
        for v in verbs[:6]:
            if n.endswith(("s", "x", "ch", "sh")):
                add(f"The {n}es {v} toward the")
            else:
                add(f"The {n}s {v} toward the")
    for n in nouns[:12]:
        for v in verbs[:10]:
            add(f"Does the {n} {v}")
    for n in nouns[:12]:
        for v in verbs[:10]:
            add(f"The {n} does not {v}")
    for n in nouns[:10]:
        for v in verbs[:8]:
            add(f"If the {n} {v}")
    for n in nouns[:15]:
        for p in places[:12]:
            add(f"The {n} in the {p}")
    for n in nouns[:10]:
        for v in verbs[:6]:
            add(f"Although the {n} {v}")
    for n in nouns[:10]:
        for v in verbs[:6]:
            add(f"Because the {n} {v}")
    for n in nouns[:12]:
        for v in verbs[:8]:
            add(f"The {n} that {v}")

    # Additional patterns for diversity
    for n in nouns[:20]:
        add(f"Every {n} knows the")
        add(f"No {n} could ever")
        add(f"Some {n} will always")
    for a in adjs[:12]:
        for n in nouns[:8]:
            add(f"A very {a} {n} was")
    for n in nouns[:15]:
        add(f"When the {n} arrived")
        add(f"While the {n} rested")
        add(f"After the {n} disappeared")
    for n in nouns[:10]:
        for v in verbs[:5]:
            add(f"Nobody {v} the {n}")
    for a in adjs[:8]:
        for n in nouns[:5]:
            add(f"The most {a} {n}")

    print(f"[prompts] Generated {len(prompts)} unique prompts")
    return prompts[:550]


# ===== Collect hidden states =====
def collect_hidden_states(model, tokenizer, device, model_info, prompts):
    """收集所有prompt在所有层的hidden states (last token position)"""
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    n_prompts = len(prompts)

    H = np.zeros((n_prompts, n_layers + 1, d_model), dtype=np.float32)

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

            if has_nan:
                H[i] = 0
                failed_prompts.add(i)
            else:
                successful += 1

            del out

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{n_prompts}] collected")

        except Exception as e:
            H[i] = 0
            failed_prompts.add(i)

    if failed_prompts:
        print(f"  Removing {len(failed_prompts)} prompts with NaN/Inf")
        valid_mask = np.ones(n_prompts, dtype=bool)
        valid_mask[list(failed_prompts)] = False
        H = H[valid_mask]

    print(f"[collect] {successful}/{H.shape[0]} successful")
    return H


# ===== EDMD Dictionary =====
def build_edmd_dictionary(Z, n_linear=20, n_cross=10):
    """
    Build EDMD dictionary from PCA-projected states Z: (N, d_pca)
    
    Dictionary = [linear(z_1..z_n), squares(z_1²..z_n²), cross(z_i*z_j)]
    
    Args:
        Z: (N, d_pca) PCA-projected states
        n_linear: number of linear components (use top n_linear PCA components)
        n_cross: number of components for cross terms
    
    Returns:
        Psi: (N, M) dictionary evaluation
        dict_info: metadata about the dictionary
    """
    N, d = Z.shape
    n_linear = min(n_linear, d)
    n_cross = min(n_cross, n_linear)
    
    Z_use = Z[:, :n_linear]  # (N, n_linear)
    
    # Linear terms
    parts = [Z_use]
    labels = [f"z{i}" for i in range(n_linear)]
    
    # Square terms
    parts.append(Z_use ** 2)
    labels += [f"z{i}²" for i in range(n_linear)]
    
    # Cross terms (among top n_cross components)
    for i in range(n_cross):
        for j in range(i + 1, n_cross):
            parts.append((Z_use[:, i] * Z_use[:, j]).reshape(-1, 1))
            labels.append(f"z{i}*z{j}")
    
    Psi = np.hstack(parts)  # (N, M)
    dict_info = {
        "M": Psi.shape[1],
        "n_linear": n_linear,
        "n_cross": n_cross,
        "n_square": n_linear,
        "n_cross_terms": n_cross * (n_cross - 1) // 2,
    }
    
    return Psi, dict_info


def compute_edmd(Psi_X, Psi_Y, alpha=1.0):
    """
    Compute EDMD Koopman matrix K via ridge regression.
    
    Psi_Y ≈ Psi_X @ K
    K = (Psi_X^T Psi_X + αI)^{-1} Psi_X^T Psi_Y
    
    Returns:
        K: (M, M) Koopman matrix
        eigenvalues: (M,) Koopman eigenvalues
        eigenvectors: (M, M) Koopman eigenvectors (columns)
    """
    M = Psi_X.shape[1]
    
    # Ridge regression
    G = Psi_X.T @ Psi_X + alpha * np.eye(M)
    A = Psi_X.T @ Psi_Y
    K = np.linalg.solve(G, A)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(K)
    
    return K, eigenvalues, eigenvectors


# ===== Exp 1: EDMD — Koopman Eigenfunction Search =====
def exp1_koopman_spectrum(H, n_layers, d_model):
    """
    对关键层对计算EDMD, 获取Koopman谱
    
    Returns:
        per-layer Koopman eigenvalues, mode counts, etc.
    """
    print("\n" + "=" * 60)
    print("Exp 1: EDMD — Koopman Eigenfunction Search")
    print("=" * 60)

    n_prompts = H.shape[0]
    
    # Step 1: Global PCA projection to d_pca=30
    from sklearn.decomposition import PCA
    d_pca = 30
    H_flat = H.reshape(-1, d_model)
    pca = PCA(n_components=d_pca)
    pca.fit(H_flat)
    H_proj = np.zeros((n_prompts, n_layers + 1, d_pca), dtype=np.float32)
    for l in range(n_layers + 1):
        H_proj[:, l, :] = pca.transform(H[:, l, :])
    
    print(f"  PCA projection: {d_model} → {d_pca}, "
          f"explained var (top 30) = {pca.explained_variance_ratio_[:30].sum():.4f}")
    
    # Step 2: EDMD for key layer pairs
    sample_layers = [0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if 0 <= l < n_layers]))
    
    results = {"per_layer": {}, "pca_explained_var_top30": float(pca.explained_variance_ratio_[:30].sum())}
    
    for l in sample_layers:
        # Build dictionary for layer ℓ and ℓ+1
        Z_X = H_proj[:, l, :]      # (N, d_pca)
        Z_Y = H_proj[:, l + 1, :]  # (N, d_pca)
        
        Psi_X, dict_info = build_edmd_dictionary(Z_X, n_linear=20, n_cross=10)
        Psi_Y, _ = build_edmd_dictionary(Z_Y, n_linear=20, n_cross=10)
        
        # Center
        Psi_X_mean = Psi_X.mean(axis=0)
        Psi_Y_mean = Psi_Y.mean(axis=0)
        Psi_Xc = Psi_X - Psi_X_mean
        Psi_Yc = Psi_Y - Psi_Y_mean
        
        # Compute EDMD
        K, eigenvalues, eigenvectors = compute_edmd(Psi_Xc, Psi_Yc, alpha=1.0)
        
        # EDMD R² (in dictionary space)
        Psi_Y_pred = Psi_Xc @ K + Psi_Y_mean
        ss_res = np.sum((Psi_Y - Psi_Y_pred) ** 2)
        ss_tot = np.sum((Psi_Y - Psi_Y_mean) ** 2)
        r2_edmd = 1 - ss_res / max(ss_tot, 1e-10)
        
        # Classify modes
        abs_eig = np.abs(eigenvalues)
        n_slow = int(np.sum(np.abs(abs_eig - 1.0) < 0.1))  # |λ| ≈ 1
        n_decaying = int(np.sum(abs_eig < 0.9))              # |λ| < 0.9
        n_growing = int(np.sum(abs_eig > 1.1))               # |λ| > 1.1
        n_marginal = int(np.sum((abs_eig >= 0.9) & (abs_eig <= 1.1)))  # near-unitary
        
        # Also compute linear DMD for comparison (using (d, N) convention)
        X_mean = Z_X.mean(axis=0)  # (d,)
        Y_mean = Z_Y.mean(axis=0)  # (d,)
        Xc_T = Z_X.T - X_mean.reshape(-1, 1)  # (d, N)
        Yc_T = Z_Y.T - Y_mean.reshape(-1, 1)  # (d, N)
        k_svd = min(Xc_T.shape[0], Xc_T.shape[1], d_pca)
        U, S, Vt = np.linalg.svd(Xc_T, full_matrices=False)
        U, S, Vt = U[:, :k_svd], S[:k_svd], Vt[:k_svd, :]
        mask = S > 1e-8
        U, S, Vt = U[:, mask], S[mask], Vt[mask, :]
        if len(S) > 0:
            A_tilde = U.T @ Yc_T @ Vt.T @ np.diag(1.0 / S)
            dmd_eigenvalues = np.linalg.eigvals(A_tilde)
            dmd_n_slow = int(np.sum(np.abs(np.abs(dmd_eigenvalues) - 1.0) < 0.1))
            # DMD R²
            Y_pred_T = U @ (A_tilde @ (U.T @ Xc_T)) + Y_mean.reshape(-1, 1)  # (d, N)
            ss_res_dmd = np.sum((Yc_T - (Y_pred_T - Y_mean.reshape(-1, 1))) ** 2)
            ss_tot_dmd = np.sum(Yc_T ** 2)
            r2_dmd = 1 - ss_res_dmd / max(ss_tot_dmd, 1e-10)
        else:
            dmd_eigenvalues = np.array([])
            dmd_n_slow = 0
            r2_dmd = 0
        
        results["per_layer"][f"L{l}_L{l+1}"] = {
            "edmd_r2": float(r2_edmd),
            "dmd_r2": float(r2_dmd),
            "edmd_n_slow": n_slow,
            "edmd_n_marginal": n_marginal,
            "edmd_n_decaying": n_decaying,
            "edmd_n_growing": n_growing,
            "dmd_n_slow": dmd_n_slow,
            "dict_size_M": dict_info["M"],
            "top5_eigenvalue_mags": sorted([float(x) for x in abs_eig], reverse=True)[:5],
            "slow_eigenvalues_real": [float(x.real) for x in eigenvalues[np.abs(abs_eig - 1.0) < 0.1][:10]],
            "slow_eigenvalues_imag": [float(x.imag) for x in eigenvalues[np.abs(abs_eig - 1.0) < 0.1][:10]],
        }
        
        print(f"  L{l}->L{l+1}: EDMD R²={r2_edmd:.4f}, DMD R²={r2_dmd:.4f}, "
              f"slow={n_slow}, marginal={n_marginal}, decaying={n_decaying}, growing={n_growing}, M={dict_info['M']}")
    
    return results, pca, d_pca


# ===== Exp 2: Koopman vs PCA Prediction Comparison (★★★ CORE ★★★) =====
def exp2_koopman_vs_pca(H, n_layers, d_model, pca, d_pca):
    """
    核心实验: 对比Koopman特征函数 vs PCA作为动力学基底
    
    方法:
    1. 训练EDMD获取Koopman特征函数(在早期层)
    2. 对每个维度d:
       - PCA: 投影到前d个PCA方向 → 线性预测下一层 → R²
       - Koopman: 投影到前d个Koopman特征函数 → 线性预测下一层 → R²
    3. 5-fold交叉验证
    
    ★★★ 核心假设: 如果Koopman R² > PCA R², 说明Koopman特征函数是更好的状态变量 ★★★
    """
    print("\n" + "=" * 60)
    print("Exp 2: Koopman vs PCA Prediction Comparison (★★★ CORE ★★★)")
    print("=" * 60)

    n_prompts = H.shape[0]
    
    # Project all layers to PCA space
    H_proj = np.zeros((n_prompts, n_layers + 1, d_pca), dtype=np.float32)
    for l in range(n_layers + 1):
        H_proj[:, l, :] = pca.transform(H[:, l, :])
    
    # Step 1: Train EDMD on early layers (L2→L3, L3→L4, L4→L5) 
    # This gives us the Koopman eigenfunctions
    print("  Training EDMD on early layers (L2→L5)...")
    train_layers = [2, 3, 4]  # Use L2→L3, L3→L4, L4→L5
    
    all_Psi_X = []
    all_Psi_Y = []
    for l in train_layers:
        if l >= n_layers:
            continue
        Z_X = H_proj[:, l, :]
        Z_Y = H_proj[:, l + 1, :]
        Psi_X, dict_info = build_edmd_dictionary(Z_X, n_linear=20, n_cross=10)
        Psi_Y, _ = build_edmd_dictionary(Z_Y, n_linear=20, n_cross=10)
        all_Psi_X.append(Psi_X)
        all_Psi_Y.append(Psi_Y)
    
    Psi_X_train = np.vstack(all_Psi_X)  # (3N, M)
    Psi_Y_train = np.vstack(all_Psi_Y)  # (3N, M)
    
    # Center and compute EDMD
    Psi_X_mean = Psi_X_train.mean(axis=0)
    Psi_Y_mean = Psi_Y_train.mean(axis=0)
    Psi_Xc = Psi_X_train - Psi_X_mean
    Psi_Yc = Psi_Y_train - Psi_Y_mean
    
    K, eigenvalues, eigvecs = compute_edmd(Psi_Xc, Psi_Yc, alpha=1.0)
    
    # Sort Koopman modes by |λ| closeness to 1 (slow modes first)
    abs_eig = np.abs(eigenvalues)
    # Two sorting criteria: by slow (|λ|≈1) and by variance explained
    slow_order = np.argsort(np.abs(abs_eig - 1.0))  # Slowest first
    var_order = np.argsort(-abs_eig)  # Largest |λ| first
    
    print(f"  EDMD trained: M={dict_info['M']}, top 10 |λ| = "
          f"{[round(x, 4) for x in sorted(abs_eig, reverse=True)[:10]]}")
    print(f"  Top 5 slow modes: |λ| = {abs_eig[slow_order[:5]].tolist()}")
    
    # ★★★ Key: Koopman eigenvectors are COMPLEX (complex eigenvalues)
    # For fair comparison with PCA (real-valued), we use Re(φ_i(h)) as the real coordinate
    # This is valid because Re(φ(h_{ℓ+1})) = Re(λ) Re(φ(h_ℓ)) - Im(λ) Im(φ(h_ℓ))
    # i.e., the real part evolves linearly in terms of both real and imaginary parts
    
    # Step 2: For each dimension d, compare Koopman vs PCA prediction
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    dims_to_test = [2, 5, 10, 15, 20, 30, 50, 75, 100]
    dims_to_test = [d for d in dims_to_test if d <= d_pca]
    
    test_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    results = {"dims": dims_to_test, "koopman_r2": {}, "pca_r2": {}, "koopman_r2_slow": {}}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Pre-compute Koopman coordinates for all layers (complex, then take real part)
    # φ_i(h) = v_i^T (Ψ(h) - Ψ_mean) — complex-valued
    # We use |φ_i(h)| (magnitude) as the real-valued coordinate
    M = eigvecs.shape[0]
    koop_all_layers_mag = {}   # magnitude |φ_i(h)|
    koop_all_layers_real = {}  # real part Re(φ_i(h))
    
    # Also precompute for slow ordering
    V_var = eigvecs[:, var_order]   # (M, M) sorted by |λ|
    V_slow = eigvecs[:, slow_order]  # (M, M) sorted by slowness
    
    print("  Pre-computing Koopman coordinates for all layers...")
    for l in range(n_layers + 1):
        Psi_l, _ = build_edmd_dictionary(H_proj[:, l, :], n_linear=20, n_cross=10)
        Psi_l_c = Psi_l - Psi_X_mean  # (N, M)
        
        # Koopman coordinates (complex)
        koop_complex_var = Psi_l_c @ V_var   # (N, M) complex
        koop_complex_slow = Psi_l_c @ V_slow  # (N, M) complex
        
        # Store real part and magnitude for both orderings
        koop_all_layers_real[l] = {
            "var": koop_complex_var.real.astype(np.float32),   # (N, M) real part
            "slow": koop_complex_slow.real.astype(np.float32), # (N, M) real part
        }
    
    print("  Pre-computation done.")
    
    for d in dims_to_test:
        pca_r2_per_layer = {l: [] for l in test_layers}
        koopman_r2_per_layer = {l: [] for l in test_layers}
        koopman_slow_r2_per_layer = {l: [] for l in test_layers}
        
        for train_idx, test_idx in kf.split(range(n_prompts)):
            for l in test_layers:
                if l >= n_layers:
                    continue
                
                # === PCA prediction ===
                X_train_pca = H_proj[train_idx, l, :d]
                Y_train_pca = H_proj[train_idx, l + 1, :d]
                X_test_pca = H_proj[test_idx, l, :d]
                Y_test_pca = H_proj[test_idx, l + 1, :d]
                
                reg_pca = Ridge(alpha=0.1)
                reg_pca.fit(X_train_pca, Y_train_pca)
                Y_pred_pca = reg_pca.predict(X_test_pca)
                
                ss_res = np.sum((Y_test_pca - Y_pred_pca) ** 2)
                ss_tot = np.sum((Y_test_pca - Y_test_pca.mean(axis=0)) ** 2)
                r2_pca = 1 - ss_res / max(ss_tot, 1e-10)
                pca_r2_per_layer[l].append(float(r2_pca))
                
                # === Koopman prediction (sorted by |λ|, use real part) ===
                kX_train = koop_all_layers_real[l]["var"][train_idx, :d]
                kY_train = koop_all_layers_real[l + 1]["var"][train_idx, :d]
                kX_test = koop_all_layers_real[l]["var"][test_idx, :d]
                kY_test = koop_all_layers_real[l + 1]["var"][test_idx, :d]
                
                reg_koop = Ridge(alpha=0.1)
                reg_koop.fit(kX_train, kY_train)
                kY_pred = reg_koop.predict(kX_test)
                
                ss_res = np.sum((kY_test - kY_pred) ** 2)
                ss_tot = np.sum((kY_test - kY_test.mean(axis=0)) ** 2)
                r2_koop = 1 - ss_res / max(ss_tot, 1e-10)
                koopman_r2_per_layer[l].append(float(r2_koop))
                
                # === Koopman slow prediction (sorted by slowness) ===
                ksX_train = koop_all_layers_real[l]["slow"][train_idx, :d]
                ksY_train = koop_all_layers_real[l + 1]["slow"][train_idx, :d]
                ksX_test = koop_all_layers_real[l]["slow"][test_idx, :d]
                ksY_test = koop_all_layers_real[l + 1]["slow"][test_idx, :d]
                
                reg_slow = Ridge(alpha=0.1)
                reg_slow.fit(ksX_train, ksY_train)
                ksY_pred = reg_slow.predict(ksX_test)
                
                ss_res = np.sum((ksY_test - ksY_pred) ** 2)
                ss_tot = np.sum((ksY_test - ksY_test.mean(axis=0)) ** 2)
                r2_slow = 1 - ss_res / max(ss_tot, 1e-10)
                koopman_slow_r2_per_layer[l].append(float(r2_slow))
        
        # Average across folds
        pca_mean = float(np.mean([np.mean(pca_r2_per_layer[l]) for l in test_layers if l < n_layers]))
        koop_mean = float(np.mean([np.mean(koopman_r2_per_layer[l]) for l in test_layers if l < n_layers]))
        koop_slow_mean = float(np.mean([np.mean(koopman_slow_r2_per_layer[l]) for l in test_layers if l < n_layers]))
        
        pca_min = float(np.min([np.mean(pca_r2_per_layer[l]) for l in test_layers if l < n_layers]))
        koop_min = float(np.min([np.mean(koopman_r2_per_layer[l]) for l in test_layers if l < n_layers]))
        
        results["pca_r2"][str(d)] = {"mean": pca_mean, "min": pca_min}
        results["koopman_r2"][str(d)] = {"mean": koop_mean, "min": koop_min}
        results["koopman_r2_slow"][str(d)] = {"mean": koop_slow_mean}
        
        # Sample per-layer R² for d=10 and d=30
        if d in [10, 30]:
            per_layer = {f"L{l}": {
                "pca": float(np.mean(pca_r2_per_layer[l])),
                "koopman": float(np.mean(koopman_r2_per_layer[l])),
                "koopman_slow": float(np.mean(koopman_slow_r2_per_layer[l])),
            } for l in test_layers if l < n_layers}
            results[f"per_layer_d{d}"] = per_layer
        
        print(f"  d={d:3d}: PCA mean={pca_mean:.4f}, Koopman mean={koop_mean:.4f}, "
              f"Koopman-slow mean={koop_slow_mean:.4f} | "
              f"PCA min={pca_min:.4f}, Koopman min={koop_min:.4f}")
    
    # Find optimal d
    best_d_pca = max(results["pca_r2"].items(), key=lambda x: x[1]["mean"])
    best_d_koop = max(results["koopman_r2"].items(), key=lambda x: x[1]["mean"])
    print(f"\n  ★ Best PCA d={best_d_pca[0]} (mean R²={best_d_pca[1]['mean']:.4f})")
    print(f"  ★ Best Koopman d={best_d_koop[0]} (mean R²={best_d_koop[1]['mean']:.4f})")
    
    return results


# ===== Exp 3: Slow/Fast Manifold Decomposition =====
def exp3_slow_fast_decomposition(H, n_layers, d_model, pca, d_pca):
    """
    将Δh分解为慢流形分量(Δh_slow)和快瞬态分量(Δh_fast)
    
    慢流形 = Koopman modes with |λ| ≈ 1
    快瞬态 = Koopman modes with |λ| << 1
    
    关键问题:
    1. Δh_slow占Δh的多少比例?
    2. Δh_fast是噪声还是有结构?
    3. 慢流形维度是多少?
    """
    print("\n" + "=" * 60)
    print("Exp 3: Slow/Fast Manifold Decomposition")
    print("=" * 60)

    n_prompts = H.shape[0]
    
    # Project to PCA space
    H_proj = np.zeros((n_prompts, n_layers + 1, d_pca), dtype=np.float32)
    for l in range(n_layers + 1):
        H_proj[:, l, :] = pca.transform(H[:, l, :])
    
    # Train EDMD on all data from L2→L3
    Z_X = H_proj[:, 2, :]
    Z_Y = H_proj[:, 3, :]
    Psi_X, dict_info = build_edmd_dictionary(Z_X, n_linear=20, n_cross=10)
    Psi_Y, _ = build_edmd_dictionary(Z_Y, n_linear=20, n_cross=10)
    
    Psi_X_mean = Psi_X.mean(axis=0)
    Psi_Y_mean = Psi_Y.mean(axis=0)
    K, eigenvalues, eigvecs = compute_edmd(Psi_X - Psi_X_mean, Psi_Y - Psi_Y_mean, alpha=1.0)
    
    abs_eig = np.abs(eigenvalues)
    M = len(eigenvalues)
    
    # Classify modes
    slow_mask = np.abs(abs_eig - 1.0) < 0.15  # |λ| ≈ 1 (slow)
    fast_mask = abs_eig < 0.85                  # |λ| << 1 (fast)
    mid_mask = ~slow_mask & ~fast_mask           # intermediate
    
    n_slow = int(np.sum(slow_mask))
    n_fast = int(np.sum(fast_mask))
    n_mid = int(np.sum(mid_mask))
    
    print(f"  Mode classification: slow={n_slow}, mid={n_mid}, fast={n_fast} (M={M})")
    
    # Project Δh onto slow and fast modes for each layer
    results = {"per_layer": {}, "mode_stats": {
        "n_slow": n_slow, "n_fast": n_fast, "n_mid": n_mid,
        "slow_eigenvalues": [complex(round(e.real, 4), round(e.imag, 4)) 
                            for e in eigenvalues[slow_mask][:10]],
    }}
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    for l in sample_layers:
        if l >= n_layers:
            continue
        
        # Compute Δh in PCA space
        dH = H_proj[:, l + 1, :] - H_proj[:, l, :]  # (N, d_pca)
        
        # Build dictionary for this layer's states
        Psi_l, _ = build_edmd_dictionary(H_proj[:, l, :], n_linear=20, n_cross=10)
        Psi_l_next, _ = build_edmd_dictionary(H_proj[:, l + 1, :], n_linear=20, n_cross=10)
        
        dPsi = (Psi_l_next - Psi_l)  # (N, M) - dictionary-level change
        dPsi_c = dPsi  # Already centered approximately
        
        # Project onto slow and fast eigenmodes
        V = eigvecs  # (M, M)
        dPsi_slow = dPsi_c @ V[:, slow_mask]  # (N, n_slow)
        dPsi_fast = dPsi_c @ V[:, fast_mask]   # (N, n_fast)
        
        # Energy fractions
        total_energy = np.mean(np.sum(dPsi_c ** 2, axis=1))
        slow_energy = np.mean(np.sum(dPsi_slow ** 2, axis=1))
        fast_energy = np.mean(np.sum(dPsi_fast ** 2, axis=1))
        
        slow_frac = slow_energy / max(total_energy, 1e-10)
        fast_frac = fast_energy / max(total_energy, 1e-10)
        
        # Check if Δh_fast has structure: compute its correlation with Δh_slow
        # Use the full dictionary change dPsi_c, project onto slow and fast subspaces
        # Then measure angle between the slow and fast RECONSTRUCTIONS
        if n_slow > 0 and n_fast > 0 and M > 0:
            # Reconstruct slow and fast components in the full dictionary space
            dPsi_slow_full = dPsi_slow @ V[:, slow_mask].T  # (N, M) 
            dPsi_fast_full = dPsi_fast @ V[:, fast_mask].T   # (N, M)
            # Cosine similarity between slow and fast reconstructions
            cos_vals = np.sum(dPsi_slow_full * dPsi_fast_full, axis=1) / (
                np.linalg.norm(dPsi_slow_full, axis=1) * np.linalg.norm(dPsi_fast_full, axis=1) + 1e-10
            )
            cos_sf = float(np.mean(cos_vals))
        else:
            cos_sf = 0
        
        # Check autocorrelation of Δh_fast (if structured, should have some)
        if n_fast > 5:
            # Sample autocorrelation
            dPsi_fast_sample = dPsi_fast[:50]  # (50, n_fast)
            acf = np.mean([np.corrcoef(dPsi_fast_sample[i], dPsi_fast_sample[min(i+1, 49)])[0, 1]
                          for i in range(49) if not np.isnan(np.corrcoef(dPsi_fast_sample[i], dPsi_fast_sample[min(i+1, 49)])[0, 1])])
        else:
            acf = 0
        
        results["per_layer"][f"L{l}_L{l+1}"] = {
            "slow_frac": float(slow_frac),
            "fast_frac": float(fast_frac),
            "slow_fast_cos": float(cos_sf),
            "fast_autocorr": float(acf),
        }
        
        print(f"  L{l}->L{l+1}: slow={slow_frac:.3f}, fast={fast_frac:.3f}, "
              f"cos(slow,fast)={cos_sf:.3f}, fast_acf={acf:.3f}")
    
    return results


# ===== Exp 4: Attractor Structure & Trajectory Analysis =====
def exp4_attractor_structure(H, n_layers, d_model, pca, d_pca):
    """
    在Koopman坐标系中分析吸引子结构
    
    核心问题:
    1. 不同句子类型(疑问/陈述/条件/否定)是否流向不同吸引子?
    2. 所有轨迹是否收敛到单一吸引子?
    3. 是否存在多个盆地(basins)?
    """
    print("\n" + "=" * 60)
    print("Exp 4: Attractor Structure & Trajectory Analysis")
    print("=" * 60)

    n_prompts = H.shape[0]
    
    # Project to PCA space
    H_proj = np.zeros((n_prompts, n_layers + 1, d_pca), dtype=np.float32)
    for l in range(n_layers + 1):
        H_proj[:, l, :] = pca.transform(H[:, l, :])
    
    # Train EDMD on L2→L3 for Koopman coordinates
    Z_X = H_proj[:, 2, :]
    Z_Y = H_proj[:, 3, :]
    Psi_X, dict_info = build_edmd_dictionary(Z_X, n_linear=20, n_cross=10)
    Psi_Y, _ = build_edmd_dictionary(Z_Y, n_linear=20, n_cross=10)
    K, eigenvalues, eigvecs = compute_edmd(Psi_X - Psi_X.mean(axis=0), 
                                            Psi_Y - Psi_Y.mean(axis=0), alpha=1.0)
    abs_eig = np.abs(eigenvalues)
    
    # Use top 3 slow Koopman modes for trajectory visualization
    slow_order = np.argsort(np.abs(abs_eig - 1.0))
    V_slow3 = eigvecs[:, slow_order[:3]]  # (M, 3)
    
    # Compute Koopman coordinates for all layers
    Psi_X_mean = Psi_X.mean(axis=0)
    
    koop_coords = np.zeros((n_prompts, n_layers + 1, 3), dtype=np.float32)
    for l in range(n_layers + 1):
        Psi_l, _ = build_edmd_dictionary(H_proj[:, l, :], n_linear=20, n_cross=10)
        koop_coords[:, l, :] = (Psi_l - Psi_X_mean) @ V_slow3
    
    # Analysis 1: Trajectory convergence
    # Do all trajectories converge to the same region at late layers?
    # Measure: std of coordinates at each layer
    coord_std = np.std(koop_coords, axis=0)  # (n_layers+1, 3)
    coord_mean_std = np.mean(coord_std, axis=1)  # (n_layers+1,)
    
    # Analysis 2: Trajectory curvature (do they spiral?)
    # Compute angle between successive steps
    if n_layers > 2:
        dkoop = np.diff(koop_coords, axis=1)  # (N, n_layers, 3)
        # Angle between successive steps
        angles = []
        for l in range(n_layers - 1):
            cos_vals = np.sum(dkoop[:, l, :] * dkoop[:, l + 1, :], axis=1) / (
                np.linalg.norm(dkoop[:, l, :], axis=1) * np.linalg.norm(dkoop[:, l + 1, :], axis=1) + 1e-10
            )
            cos_vals = np.clip(cos_vals, -1, 1)
            angles.append(float(np.mean(np.arccos(cos_vals))))
    
    # Analysis 3: Cluster structure at late layers
    # Use k-means on late-layer coordinates
    from sklearn.cluster import KMeans
    
    # Cluster at the last layer (before LN)
    last_koop = koop_coords[:, n_layers - 1, :]  # (N, 3) - before final LN
    n_clusters = 4
    
    if n_prompts >= n_clusters * 5:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(last_koop)
        
        # Cluster sizes
        cluster_sizes = [int(np.sum(labels == k)) for k in range(n_clusters)]
        
        # Inter-cluster vs intra-cluster distance
        inter_dist = 0
        intra_dist = 0
        n_inter = 0
        n_intra = 0
        for i in range(min(100, n_prompts)):
            for j in range(i + 1, min(100, n_prompts)):
                d = np.linalg.norm(last_koop[i] - last_koop[j])
                if labels[i] == labels[j]:
                    intra_dist += d
                    n_intra += 1
                else:
                    inter_dist += d
                    n_inter += 1
        
        avg_intra = intra_dist / max(n_intra, 1)
        avg_inter = inter_dist / max(n_inter, 1)
        separation = avg_inter / max(avg_intra, 1e-10)
    else:
        cluster_sizes = [n_prompts]
        avg_intra = 0
        avg_inter = 0
        separation = 0
        labels = np.zeros(n_prompts, dtype=int)
    
    # Analysis 4: Convergence rate
    # Measure how quickly trajectories converge
    # At each layer, compute mean pairwise distance
    mean_pairwise_dist = []
    sample_size = min(200, n_prompts)
    for l in range(n_layers + 1):
        sample_koop = koop_coords[:sample_size, l, :]
        # Subsample for efficiency
        if sample_size > 50:
            idx = np.random.choice(sample_size, 50, replace=False)
            sample_koop = sample_koop[idx]
        dists = []
        for i in range(len(sample_koop)):
            for j in range(i + 1, len(sample_koop)):
                dists.append(np.linalg.norm(sample_koop[i] - sample_koop[j]))
        mean_pairwise_dist.append(float(np.mean(dists)))
    
    results = {
        "trajectory_convergence": {
            "coord_std_early": [float(x) for x in coord_std[1:4].flatten()],
            "coord_std_mid": [float(x) for x in coord_std[n_layers//2-1:n_layers//2+2].flatten()],
            "coord_std_late": [float(x) for x in coord_std[-4:-1].flatten()],
            "mean_std_per_layer": [float(x) for x in coord_mean_std[::max(1, n_layers//10)]],
        },
        "trajectory_curvature": {
            "mean_angle_rad": float(np.mean(angles)) if n_layers > 2 else 0,
            "angles_per_layer": [float(a) for a in angles[::max(1, len(angles)//6)]],
        },
        "cluster_structure": {
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "avg_intra_dist": float(avg_intra),
            "avg_inter_dist": float(avg_inter),
            "separation_ratio": float(separation),
        },
        "convergence": {
            "mean_pairwise_dist_per_layer": mean_pairwise_dist[::max(1, (n_layers+1)//10)],
            "convergence_ratio": float(mean_pairwise_dist[-1] / max(mean_pairwise_dist[0], 1e-10)),
        },
    }
    
    print(f"  Convergence: early_std={coord_mean_std[1]:.2f}, "
          f"mid_std={coord_mean_std[n_layers//2]:.2f}, "
          f"late_std={coord_mean_std[-2]:.2f}")
    print(f"  Curvature: mean_angle={np.mean(angles):.3f} rad "
          f"({np.mean(angles)*180/np.pi:.1f}°)" if n_layers > 2 else "")
    print(f"  Clusters: {cluster_sizes}, separation={separation:.2f}")
    print(f"  Convergence ratio (late/early dist): {results['convergence']['convergence_ratio']:.4f}")
    
    return results


# ===== Exp 5: Koopman Spectral Evolution (Bifurcation Analysis) =====
def exp5_spectral_evolution(H, n_layers, d_model, pca, d_pca):
    """
    跟踪Koopman谱跨层演化 — 寻找真正的动力学相变(不是LN伪影!)
    
    关键指标:
    1. 慢模式数(|λ|≈1)如何跨层变化?
    2. 是否存在特征值交叉(模式分裂)?
    3. 谱宽(最大|λ|-最小|λ|)如何变化?
    """
    print("\n" + "=" * 60)
    print("Exp 5: Koopman Spectral Evolution (Bifurcation Analysis)")
    print("=" * 60)

    n_prompts = H.shape[0]
    
    # Project to PCA space
    H_proj = np.zeros((n_prompts, n_layers + 1, d_pca), dtype=np.float32)
    for l in range(n_layers + 1):
        H_proj[:, l, :] = pca.transform(H[:, l, :])
    
    # Compute EDMD for EVERY layer (full spectral evolution)
    results = {"per_layer": {}, "spectral_evolution": {}}
    
    all_n_slow = []
    all_n_growing = []
    all_spectral_radius = []
    all_spectral_width = []
    all_mean_abs_eig = []
    
    for l in range(n_layers):
        Z_X = H_proj[:, l, :]
        Z_Y = H_proj[:, l + 1, :]
        
        Psi_X, dict_info = build_edmd_dictionary(Z_X, n_linear=20, n_cross=10)
        Psi_Y, _ = build_edmd_dictionary(Z_Y, n_linear=20, n_cross=10)
        
        K, eigenvalues, eigvecs = compute_edmd(
            Psi_X - Psi_X.mean(axis=0), Psi_Y - Psi_Y.mean(axis=0), alpha=1.0
        )
        
        abs_eig = np.abs(eigenvalues)
        
        n_slow = int(np.sum(np.abs(abs_eig - 1.0) < 0.1))
        n_growing = int(np.sum(abs_eig > 1.1))
        spectral_radius = float(np.max(abs_eig))
        spectral_width = float(np.max(abs_eig) - np.min(abs_eig))
        mean_abs = float(np.mean(abs_eig))
        
        all_n_slow.append(n_slow)
        all_n_growing.append(n_growing)
        all_spectral_radius.append(spectral_radius)
        all_spectral_width.append(spectral_width)
        all_mean_abs_eig.append(mean_abs)
        
        # Store per-layer details for key layers
        if l in [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]:
            results["per_layer"][f"L{l}_L{l+1}"] = {
                "n_slow": n_slow,
                "n_growing": n_growing,
                "spectral_radius": spectral_radius,
                "mean_abs_eig": mean_abs,
                "top5_eigenvalue_mags": sorted([float(x) for x in abs_eig], reverse=True)[:5],
            }
            print(f"  L{l}->L{l+1}: n_slow={n_slow}, n_grow={n_growing}, "
                  f"ρ={spectral_radius:.3f}, <|λ|>={mean_abs:.3f}")
    
    # Detect bifurcation points: where n_slow changes rapidly
    n_slow_changes = [abs(all_n_slow[l+1] - all_n_slow[l]) for l in range(len(all_n_slow) - 1)]
    max_change_layer = int(np.argmax(n_slow_changes))
    
    results["spectral_evolution"] = {
        "n_slow_per_layer": all_n_slow,
        "n_growing_per_layer": all_n_growing,
        "spectral_radius_per_layer": all_spectral_radius,
        "spectral_width_per_layer": all_spectral_width,
        "mean_abs_eig_per_layer": all_mean_abs_eig,
        "max_n_slow_change_layer": max_change_layer,
        "max_n_slow_change": float(max(n_slow_changes)) if n_slow_changes else 0,
    }
    
    # Compute spectral evolution statistics
    print(f"\n  Spectral evolution summary:")
    print(f"    n_slow: L0={all_n_slow[0]}, L{n_layers//2}={all_n_slow[n_layers//2]}, "
          f"L{n_layers-1}={all_n_slow[-1]}")
    print(f"    Max n_slow change at L{max_change_layer}→L{max_change_layer+1}: "
          f"Δ={n_slow_changes[max_change_layer] if max_change_layer < len(n_slow_changes) else 'N/A'}")
    print(f"    Spectral width: early={all_spectral_width[0]:.3f}, "
          f"mid={all_spectral_width[n_layers//2]:.3f}, late={all_spectral_width[-1]:.3f}")
    
    return results


# ===== Main =====
def main(model_name):
    print(f"\n{'='*60}")
    print(f"Phase 159: Language Phase Space Reconstruction — {model_name}")
    print(f"{'='*60}")

    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Generate prompts (500+ for robustness)
    prompts = generate_prompts()

    # Collect hidden states
    print(f"\nCollecting hidden states for {len(prompts)} prompts...")
    H = collect_hidden_states(model, tokenizer, device, model_info, prompts)

    # Run experiments
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_prompts": H.shape[0],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Exp 1: Koopman Spectrum
    exp1_results, pca, d_pca = exp1_koopman_spectrum(H, n_layers, d_model)
    results["exp1_koopman_spectrum"] = exp1_results

    # Exp 2: Koopman vs PCA Comparison (★★★ CORE ★★★)
    exp2_results = exp2_koopman_vs_pca(H, n_layers, d_model, pca, d_pca)
    results["exp2_koopman_vs_pca"] = exp2_results

    # Exp 3: Slow/Fast Decomposition
    exp3_results = exp3_slow_fast_decomposition(H, n_layers, d_model, pca, d_pca)
    results["exp3_slow_fast"] = exp3_results

    # Exp 4: Attractor Structure
    exp4_results = exp4_attractor_structure(H, n_layers, d_model, pca, d_pca)
    results["exp4_attractor"] = exp4_results

    # Exp 5: Spectral Evolution
    exp5_results = exp5_spectral_evolution(H, n_layers, d_model, pca, d_pca)
    results["exp5_spectral_evolution"] = exp5_results

    # Save results
    t_elapsed = time.time() - t_start
    results["total_time_sec"] = round(t_elapsed, 1)

    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase159_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
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
