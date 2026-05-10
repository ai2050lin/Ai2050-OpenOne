"""
Phase 92: 计算流形的拓扑与代数结构 — 大规模多模型测试
=========================================================
目标:
  1. 扩大数据量: 50+实体 × 5+关系 = 250+轨迹
  2. F的雅可比矩阵 J_F(h) 的谱结构
  3. 非线性自治性检验
  4. 流形拓扑分析: 不动点邻域、曲率
  5. 代数约束: F 是否满足某种代数律

Run:
  python tests/glm5/ccml_phase92_manifold_algebra.py --model qwen3
  python tests/glm5/ccml_phase92_manifold_algebra.py --model deepseek7b
  python tests/glm5/ccml_phase92_manifold_algebra.py --model glm4
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS

# ============================================================
# 大规模数据
# ============================================================

RELATIONS = {
    "capital": "The capital of {entity} is",
    "currency": "The currency of {entity} is",
    "language": "The official language of {entity} is",
    "continent": "The continent of {entity} is",
    "population": "The population of {entity} is",
}

ENTITIES = [
    # Europe
    "France", "Germany", "Italy", "Spain", "Portugal",
    "Poland", "Netherlands", "Belgium", "Sweden", "Norway",
    # Asia
    "Japan", "China", "India", "Thailand", "Vietnam",
    "Korea", "Indonesia", "Philippines", "Malaysia", "Pakistan",
    # Americas
    "Brazil", "Mexico", "Argentina", "Colombia", "Chile",
    "Canada", "Peru", "Ecuador", "Cuba", "Jamaica",
    # Africa
    "Egypt", "Nigeria", "Kenya", "South Africa", "Morocco",
    "Ethiopia", "Ghana", "Tanzania", "Algeria", "Tunisia",
    # Oceania
    "Australia", "New Zealand", "Fiji", "Samoa", "Tonga",
    # Middle East
    "Turkey", "Iran", "Iraq", "Saudi Arabia", "Israel",
]

# ============================================================
# Representation extraction
# ============================================================

def get_all_layer_reprs(model, tokenizer, device, prompt, n_layers):
    """Get representations at all layers using forward hook."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    layers = get_layers(model)
    
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0][0, -1, :].detach().cpu().float()
            else:
                captured[key] = output[0, -1, :].detach().cpu().float()
        return hook
    
    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    for h in hooks:
        h.remove()
    
    reprs = [out.hidden_states[0][0, -1, :].detach().cpu().float()]
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            reprs.append(captured[key])
        else:
            reprs.append(out.hidden_states[li+1][0, -1, :].detach().cpu().float())
    
    return reprs


# ============================================================
# Experiment 1: 大规模速度场维度与自治性
# ============================================================

def experiment_velocity_field(model, tokenizer, device, model_name):
    """大规模速度场分析"""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*70}")
    print(f"Experiment 1: Velocity Field Structure ({model_name}, {n_layers}L, d={d_model})")
    print(f"{'='*70}")
    
    # Collect trajectories
    print(f"Collecting {len(ENTITIES)}×{len(RELATIONS)} = {len(ENTITIES)*len(RELATIONS)} trajectories...")
    t0 = time.time()
    all_reprs = {}
    errors = 0
    
    for entity in ENTITIES:
        for relation, template in RELATIONS.items():
            prompt = template.format(entity=entity)
            try:
                reprs = get_all_layer_reprs(model, tokenizer, device, prompt, n_layers)
                all_reprs[(entity, relation)] = reprs
            except Exception as e:
                errors += 1
    
    print(f"  Collected {len(all_reprs)} trajectories ({errors} errors) in {time.time()-t0:.1f}s")
    
    if len(all_reprs) < 10:
        print("  ERROR: Too few trajectories!")
        return None
    
    # Compute velocities
    all_velocities = {}
    for key, reprs in all_reprs.items():
        vels = [reprs[l+1] - reprs[l] for l in range(len(reprs) - 1)]
        all_velocities[key] = vels
    
    relations_list = list(RELATIONS.keys())
    
    # --- 1a. Velocity dimensionality profile ---
    print("\n--- 1a: Velocity dimensionality profile ---")
    dim_profile = {}
    max_vel_len = len(next(iter(all_velocities.values())))
    
    for l in range(max_vel_len):
        vels_at_l = np.array([v[l].numpy() for v in all_velocities.values() if l < len(v)])
        if len(vels_at_l) < 10:
            continue
        
        pca = PCA()
        pca.fit(vels_at_l)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = int(np.searchsorted(cumvar, 0.9) + 1)
        n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        n_99 = int(np.searchsorted(cumvar, 0.99) + 1)
        
        dim_profile[l] = {
            "n_90": n_90, "n_95": n_95, "n_99": n_99,
            "top1_var": float(pca.explained_variance_ratio_[0]),
            "top3_var": float(pca.explained_variance_ratio_[:3].sum()),
            "mean_norm": float(np.mean(np.linalg.norm(vels_at_l, axis=1))),
        }
    
    # Print sampled layers
    sample_layers = sorted(set([0] + list(range(0, n_layers, max(1, n_layers//8))) + [n_layers-1]))
    for l in sample_layers:
        if l in dim_profile:
            d = dim_profile[l]
            print(f"  L{l:2d}: dim(90%)={d['n_90']:3d}, dim(95%)={d['n_95']:3d}, "
                  f"dim(99%)={d['n_99']:3d}, top1={d['top1_var']:.4f}, |v|={d['mean_norm']:.2f}")
    
    # --- 1b. Autonomy test (linear + nonlinear) ---
    print("\n--- 1b: Autonomy Test ---")
    
    all_h, all_v, all_l = [], [], []
    for key, reprs in all_reprs.items():
        vels = all_velocities.get(key, [])
        for l in range(len(reprs) - 1):
            all_h.append(reprs[l].numpy())
            all_v.append(vels[l].numpy() if l < len(vels) else (reprs[l+1] - reprs[l]).numpy())
            all_l.append(l)
    
    all_h = np.array(all_h)
    all_v = np.array(all_v)
    all_l_arr = np.array(all_l, dtype=float)
    l_norm = all_l_arr / max(all_l_arr.max(), 1.0)
    
    # Linear: v = f(h)
    ridge_h = Ridge(alpha=1.0)
    ridge_h.fit(all_h, all_v)
    pred_h = ridge_h.predict(all_h)
    
    # Linear: v = f(h, l)
    h_with_l = np.column_stack([all_h, l_norm])
    ridge_hl = Ridge(alpha=1.0)
    ridge_hl.fit(h_with_l, all_v)
    pred_hl = ridge_hl.predict(h_with_l)
    
    # Cosine similarities
    def batch_cosine(pred, true):
        cosines = []
        for i in range(len(true)):
            p_n, t_n = np.linalg.norm(pred[i]), np.linalg.norm(true[i])
            if p_n > 1e-8 and t_n > 1e-8:
                cosines.append(np.dot(pred[i], true[i]) / (p_n * t_n))
        return cosines
    
    cos_h = batch_cosine(pred_h, all_v)
    cos_hl = batch_cosine(pred_hl, all_v)
    
    print(f"  Linear v = f(h):    mean_cos = {np.mean(cos_h):.4f}, median = {np.median(cos_h):.4f}")
    print(f"  Linear v = f(h, l): mean_cos = {np.mean(cos_hl):.4f}, median = {np.median(cos_hl):.4f}")
    print(f"  Layer improvement: {np.mean(cos_hl) - np.mean(cos_h):.4f}")
    
    # Per-layer autonomy
    print("\n  Per-layer autonomy:")
    for l in sample_layers:
        mask = all_l_arr == l
        if mask.sum() < 5:
            continue
        cos_at_l = [cos_h[i] for i in range(len(mask)) if mask[i]]
        cos_hl_at_l = [cos_hl[i] for i in range(len(mask)) if mask[i]]
        improvement = np.mean(cos_hl_at_l) - np.mean(cos_at_l)
        print(f"    L{l:2d}: f(h)={np.mean(cos_at_l):.4f}, f(h,l)={np.mean(cos_hl_at_l):.4f}, "
              f"Δ={improvement:.4f}")
    
    # --- 1c. Relation-specific velocity structure ---
    print("\n--- 1c: Relation-specific velocity cosine matrix ---")
    
    for l in [n_layers//4, n_layers//2, 3*n_layers//4]:
        if l >= max_vel_len:
            continue
        
        mean_vels = {}
        for relation in relations_list:
            vels = [all_velocities[(e, relation)][l].numpy() 
                    for e in ENTITIES if (e, relation) in all_velocities and l < len(all_velocities[(e, relation)])]
            if vels:
                mean_vels[relation] = np.mean(vels, axis=0)
        
        print(f"\n  L{l}->{l+1}:")
        for i, r1 in enumerate(relations_list):
            for j, r2 in enumerate(relations_list):
                if i >= j:
                    continue
                if r1 in mean_vels and r2 in mean_vels:
                    v1, v2 = mean_vels[r1], mean_vels[r2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    cos = np.dot(v1, v2) / (n1 * n2) if n1 > 1e-8 and n2 > 1e-8 else 0
                    print(f"    {r1:12s} <-> {r2:12s}: cos={cos:.4f}")
    
    # --- 1d. Velocity PCA subspace overlap across layers ---
    print("\n--- 1d: Velocity PCA subspace overlap across layers ---")
    
    pca_per_layer = {}
    for l in range(max_vel_len):
        vels_at_l = np.array([v[l].numpy() for v in all_velocities.values() if l < len(v)])
        if len(vels_at_l) < 10:
            continue
        pca = PCA(n_components=min(50, vels_at_l.shape[0]-1, vels_at_l.shape[1]))
        pca.fit(vels_at_l)
        pca_per_layer[l] = pca.components_  # [k, d_model]
    
    test_layers_overlap = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    test_layers_overlap = [l for l in test_layers_overlap if l in pca_per_layer]
    
    print(f"  Subspace overlap (principal angles):")
    for i, l1 in enumerate(test_layers_overlap):
        for j, l2 in enumerate(test_layers_overlap):
            if i >= j:
                continue
            B1 = pca_per_layer[l1]  # [k1, d]
            B2 = pca_per_layer[l2]  # [k2, d]
            # Subspace overlap = ||B1 B1^T B2 B2^T||_F / sqrt(k1 * k2)
            k = min(B1.shape[0], B2.shape[0])
            overlap = np.linalg.norm(B1[:k] @ B1[:k].T @ B2[:k] @ B2[:k].T, 'fro') / k
            print(f"    L{l1} <-> L{l2}: overlap={overlap:.4f}")
    
    # --- 1e. Trajectory divergence detailed profile ---
    print("\n--- 1e: Trajectory divergence detailed profile ---")
    
    for l in sample_layers:
        if l > len(next(iter(all_reprs.values()))) - 1:
            continue
        
        # Same entity, different relations
        same_ent_diff_rel = []
        for entity in ENTITIES:
            reprs_list = [all_reprs.get((entity, r)) for r in relations_list if (entity, r) in all_reprs]
            valid = [r[l] for r in reprs_list if l < len(r)]
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    c = F.cosine_similarity(valid[i].unsqueeze(0), valid[j].unsqueeze(0)).item()
                    same_ent_diff_rel.append(c)
        
        # Same relation, different entities
        same_rel_diff_ent = []
        for relation in relations_list:
            reprs_list = [all_reprs.get((e, relation)) for e in ENTITIES if (e, relation) in all_reprs]
            valid = [r[l] for r in reprs_list if l < len(r)]
            # Sample to avoid O(n^2) for 50 entities
            indices = np.random.choice(len(valid), min(20, len(valid)), replace=False) if len(valid) > 20 else range(len(valid))
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx+1, len(indices)):
                    ii, jj = indices[i_idx], indices[j_idx]
                    c = F.cosine_similarity(valid[ii].unsqueeze(0), valid[jj].unsqueeze(0)).item()
                    same_rel_diff_ent.append(c)
        
        if same_ent_diff_rel and same_rel_diff_ent:
            print(f"  L{l:2d}: ent_diff_rel={np.mean(same_ent_diff_rel):.4f}, "
                  f"rel_diff_ent={np.mean(same_rel_diff_ent):.4f}, "
                  f"ratio={np.mean(same_rel_diff_ent)/max(np.mean(same_ent_diff_rel), 0.01):.2f}")
    
    results = {
        "model": model_name,
        "n_trajectories": len(all_reprs),
        "dim_profile": {str(k): v for k, v in dim_profile.items()},
        "autonomy": {
            "f_h_mean": float(np.mean(cos_h)),
            "f_hl_mean": float(np.mean(cos_hl)),
            "improvement": float(np.mean(cos_hl) - np.mean(cos_h)),
        },
    }
    
    return results, all_reprs, all_velocities


# ============================================================
# Experiment 2: Jacobian of velocity field F
# ============================================================

def experiment_jacobian(model, tokenizer, device, model_name, all_reprs=None):
    """分析速度场 F 的雅可比矩阵"""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*70}")
    print(f"Experiment 2: Jacobian of Velocity Field F ({model_name})")
    print(f"{'='*70}")
    
    # Collect a subset of h vectors at key layers
    if all_reprs is None:
        # Quick collection with few entities
        quick_entities = ENTITIES[:10]
        quick_relations = ["capital", "currency", "language"]
        all_reprs = {}
        for entity in quick_entities:
            for relation in quick_relations:
                template = RELATIONS[relation]
                prompt = template.format(entity=entity)
                try:
                    reprs = get_all_layer_reprs(model, tokenizer, device, prompt, n_layers)
                    all_reprs[(entity, relation)] = reprs
                except:
                    pass
    
    # Numerical Jacobian at selected points
    # J_F(h) ≈ [F(h + ε*e_i) - F(h - ε*e_i)] / (2ε)
    # But this requires model forward passes which is expensive
    # Instead, use the collected velocities to estimate local linearization
    
    print("\n--- 2a: Local velocity linearization (using collected data) ---")
    
    max_repr_len = len(next(iter(all_reprs.values())))
    
    for target_layer in [0, n_layers//3, 2*n_layers//3, n_layers-2]:
        if target_layer >= max_repr_len - 1:
            continue
        
        # Collect h and v at this layer
        h_list = []
        v_list = []
        for key, reprs in all_reprs.items():
            if target_layer + 1 < len(reprs):
                h_list.append(reprs[target_layer].numpy())
                v_list.append((reprs[target_layer + 1] - reprs[target_layer]).numpy())
        
        if len(h_list) < 20:
            continue
        
        H = np.array(h_list)  # [N, d]
        V = np.array(v_list)  # [N, d]
        
        # Ridge regression: V ≈ J @ H^T + b  (linear approximation of F)
        # This gives us the best linear approximation J of F
        ridge = Ridge(alpha=1.0)
        ridge.fit(H, V)
        J_approx = ridge.coef_  # [d, d]
        
        # Spectral analysis of J
        try:
            eigenvalues = np.linalg.eigvals(J_approx)
            real_eigs = np.real(eigenvalues)
            abs_eigs = np.abs(eigenvalues)
            
            # Sort by absolute value
            sorted_idx = np.argsort(abs_eigs)[::-1]
            top_eigs = real_eigs[sorted_idx[:20]]
            top_abs = abs_eigs[sorted_idx[:20]]
            
            print(f"\n  L{target_layer}: J spectral analysis")
            print(f"    Top-5 eigenvalues (real): {top_eigs[:5].round(4)}")
            print(f"    Top-5 |eigenvalues|: {top_abs[:5].round(4)}")
            print(f"    Max |λ|={top_abs[0]:.4f}, spectral radius={np.max(abs_eigs):.4f}")
            
            # Effective rank
            sorted_abs = np.sort(abs_eigs)[::-1]
            total_energy = np.sum(sorted_abs)
            cum_energy = np.cumsum(sorted_abs) / total_energy
            eff_rank_90 = np.searchsorted(cum_energy, 0.9) + 1
            eff_rank_99 = np.searchsorted(cum_energy, 0.99) + 1
            print(f"    Effective rank(90%)={eff_rank_90}, rank(99%)={eff_rank_99}")
            
            # Symmetry: how close is J to symmetric?
            J_sym = (J_approx + J_approx.T) / 2
            J_anti = (J_approx - J_approx.T) / 2
            sym_norm = np.linalg.norm(J_sym)
            anti_norm = np.linalg.norm(J_anti)
            print(f"    Sym/Anti ratio: {sym_norm:.2f}/{anti_norm:.2f} = {sym_norm/(anti_norm+1e-10):.2f}")
            
            # Prediction quality
            pred_v = ridge.predict(H)
            cosines = []
            for i in range(len(V)):
                p_n, t_n = np.linalg.norm(pred_v[i]), np.linalg.norm(V[i])
                if p_n > 1e-8 and t_n > 1e-8:
                    cosines.append(np.dot(pred_v[i], V[i]) / (p_n * t_n))
            print(f"    Linear fit quality: mean_cos={np.mean(cosines):.4f}")
            
        except Exception as e:
            print(f"  L{target_layer}: J spectral analysis failed: {e}")
    
    # --- 2b: Velocity field continuity check ---
    print("\n--- 2b: Velocity field continuity ---")
    print("  If F is smooth, nearby h should have similar v.")
    
    for target_layer in [n_layers//2]:
        h_list = []
        v_list = []
        for key, reprs in all_reprs.items():
            if target_layer + 1 < len(reprs):
                h_list.append(reprs[target_layer].numpy())
                v_list.append((reprs[target_layer + 1] - reprs[target_layer]).numpy())
        
        if len(h_list) < 20:
            continue
        
        H = np.array(h_list)
        V = np.array(v_list)
        
        # Compute pairwise distances in h-space and v-space
        n = min(30, len(H))  # Sample to avoid O(N^2)
        idx = np.random.choice(len(H), n, replace=False)
        H_sub, V_sub = H[idx], V[idx]
        
        h_dists = squareform(pdist(H_sub, 'cosine'))
        v_dists = squareform(pdist(V_sub, 'cosine'))
        
        # Correlation: nearby h → nearby v?
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        h_flat = h_dists[mask]
        v_flat = v_dists[mask]
        
        if len(h_flat) > 10:
            corr = np.corrcoef(h_flat, v_flat)[0, 1]
            print(f"  L{target_layer}: h-distance vs v-distance correlation = {corr:.4f}")
            if corr > 0.5:
                print(f"    → F is roughly continuous (nearby h → nearby v)")
            else:
                print(f"    → F may be discontinuous or highly nonlinear")
    
    return


# ============================================================
# Experiment 3: Nonlinear autonomy test
# ============================================================

def experiment_nonlinear_autonomy(model, tokenizer, device, model_name, all_reprs=None):
    """用简单MLP检验非线性自治性"""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*70}")
    print(f"Experiment 3: Nonlinear Autonomy Test ({model_name})")
    print(f"{'='*70}")
    
    if all_reprs is None:
        print("  Need all_reprs from Experiment 1!")
        return
    
    # Collect data
    all_h, all_v, all_l = [], [], []
    for key, reprs in all_reprs.items():
        vels = [reprs[l+1] - reprs[l] for l in range(len(reprs) - 1)]
        for l in range(len(vels)):
            all_h.append(reprs[l].numpy())
            all_v.append(vels[l].numpy())
            all_l.append(l)
    
    all_h = np.array(all_h, dtype=np.float32)
    all_v = np.array(all_v, dtype=np.float32)
    all_l_arr = np.array(all_l, dtype=np.float32)
    l_norm = all_l_arr / max(all_l_arr.max(), 1.0)
    
    N = len(all_h)
    
    # Use PCA to reduce dimensionality for MLP training
    print(f"  Total samples: {N}, reducing h from d={d_model} to d=64 via PCA")
    pca_h = PCA(n_components=64)
    pca_v = PCA(n_components=64)
    
    h_reduced = pca_h.fit_transform(all_h)  # [N, 64]
    v_reduced = pca_v.fit_transform(all_v)  # [N, 64]
    
    print(f"  PCA explained variance: h={pca_h.explained_variance_ratio_[:5].sum():.4f}, "
          f"v={pca_v.explained_variance_ratio_[:5].sum():.4f}")
    
    # Split train/test
    perm = np.random.permutation(N)
    n_train = int(0.8 * N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    
    h_train, h_test = h_reduced[train_idx], h_reduced[test_idx]
    v_train, v_test = v_reduced[train_idx], v_reduced[test_idx]
    l_train, l_test = l_norm[train_idx], l_norm[test_idx]
    
    # PyTorch MLP
    import torch.nn as nn
    
    class VelocityMLP(nn.Module):
        def __init__(self, in_dim, hidden_dim=128, out_dim=64, use_layer=False):
            super().__init__()
            self.use_layer = use_layer
            actual_in = in_dim + (1 if use_layer else 0)
            self.net = nn.Sequential(
                nn.Linear(actual_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        def forward(self, h, l=None):
            if self.use_layer and l is not None:
                x = torch.cat([h, l], dim=-1)
            else:
                x = h
            return self.net(x)
    
    device_t = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train v = F(h) - no layer info
    model_h = VelocityMLP(64, 128, 64, use_layer=False).to(device_t)
    optimizer = torch.optim.Adam(model_h.parameters(), lr=1e-3)
    
    h_train_t = torch.tensor(h_train, dtype=torch.float32, device=device_t)
    v_train_t = torch.tensor(v_train, dtype=torch.float32, device=device_t)
    h_test_t = torch.tensor(h_test, dtype=torch.float32, device=device_t)
    v_test_t = torch.tensor(v_test, dtype=torch.float32, device=device_t)
    
    print(f"\n  Training MLP: v = F(h) [no layer info]...")
    for epoch in range(200):
        pred = model_h(h_train_t)
        loss = F.mse_loss(pred, v_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_test = model_h(h_test_t)
                cos = F.cosine_similarity(pred_test, v_test_t, dim=-1).mean()
                print(f"    Epoch {epoch+1}: loss={loss.item():.6f}, test_cos={cos.item():.4f}")
    
    # Train v = F(h, l) - with layer info
    model_hl = VelocityMLP(64, 128, 64, use_layer=True).to(device_t)
    optimizer2 = torch.optim.Adam(model_hl.parameters(), lr=1e-3)
    
    l_train_t = torch.tensor(l_train, dtype=torch.float32, device=device_t).unsqueeze(-1)
    l_test_t = torch.tensor(l_test, dtype=torch.float32, device=device_t).unsqueeze(-1)
    
    print(f"\n  Training MLP: v = F(h, l) [with layer info]...")
    for epoch in range(200):
        pred = model_hl(h_train_t, l_train_t)
        loss = F.mse_loss(pred, v_train_t)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_test = model_hl(h_test_t, l_test_t)
                cos = F.cosine_similarity(pred_test, v_test_t, dim=-1).mean()
                print(f"    Epoch {epoch+1}: loss={loss.item():.6f}, test_cos={cos.item():.4f}")
    
    # Final comparison
    with torch.no_grad():
        pred_h = model_h(h_test_t)
        pred_hl = model_hl(h_test_t, l_test_t)
        
        cos_h = F.cosine_similarity(pred_h, v_test_t, dim=-1).mean().item()
        cos_hl = F.cosine_similarity(pred_hl, v_test_t, dim=-1).mean().item()
    
    improvement = cos_hl - cos_h
    print(f"\n  === Nonlinear Autonomy Result ===")
    print(f"  MLP v = F(h):    test_cos = {cos_h:.4f}")
    print(f"  MLP v = F(h, l): test_cos = {cos_hl:.4f}")
    print(f"  Nonlinear layer improvement: {improvement:.4f}")
    
    if improvement < 0.02:
        print(f"  → NONLINEAR AUTONOMY CONFIRMED (layer adds <2%)")
    elif improvement < 0.05:
        print(f"  → WEAK layer dependence (2-5% improvement)")
    else:
        print(f"  → SIGNIFICANT layer dependence (>5% improvement)")
    
    # Cleanup MLP from GPU
    del model_h, model_hl
    gc.collect()
    torch.cuda.empty_cache()
    
    return {"nonlinear_f_h": cos_h, "nonlinear_f_hl": cos_hl, "improvement": improvement}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model loaded: {args.model}, class={info.model_class}, "
          f"n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_reprs = None
    
    if args.exp in ["1", "all"]:
        result = experiment_velocity_field(model, tokenizer, device, args.model)
        if result is not None:
            summary_dict, all_reprs_raw, all_velocities_raw = result
            # Save summary
            out_path = f"tests/glm5_temp/phase92_exp1_{args.model}_results.json"
            with open(out_path, 'w') as f:
                json.dump(summary_dict if isinstance(summary_dict, dict) else {}, f, indent=2)
            print(f"\nExp1 results saved to {out_path}")
            all_reprs = all_reprs_raw
    
    if args.exp in ["2", "all"]:
        experiment_jacobian(model, tokenizer, device, args.model, all_reprs)
    
    if args.exp in ["3", "all"]:
        result3 = experiment_nonlinear_autonomy(model, tokenizer, device, args.model, all_reprs)
        if result3:
            out_path3 = f"tests/glm5_temp/phase92_exp3_{args.model}_results.json"
            with open(out_path3, 'w') as f:
                json.dump(result3, f, indent=2)
            print(f"\nExp3 results saved to {out_path3}")
    
    # Cleanup
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"PHASE 92 COMPLETE ({args.model})")
    print(f"{'='*70}")
