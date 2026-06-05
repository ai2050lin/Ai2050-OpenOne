"""
Phase 382: 多因子残差分解 & PC1语义解码
=======================================

核心目标：
1. 分解dh_proper中各因子的方差贡献（category, norm_ratio, object_id, binding_strength等）
2. 解码DS7B PC1的真实语义（是绑定强度？对象显著性？还是纯粹的范数比？）
3. PC-factor相关性矩阵（每个PC编码什么？）
4. 类别swap因果测试（比add/remove更干净）

方法：
Part 1: 收集所有必要数据（residual states + logit信息）
Part 2: 因子R²分解 — 每个因子解释dh_proper多少方差
Part 3: PC语义解码 — PC1-PC10与各因子的相关性
Part 4: DS7B PC1回归 — 多候选因子回归确定PC1语义
Part 5: 类别swap因果测试 — logit lens方式

用法:
  python tests/glm5/phase382_factor_decomposition.py qwen3
  python tests/glm5/phase382_factor_decomposition.py deepseek7b
  python tests/glm5/phase382_factor_decomposition.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS
from phase381_norm_matched_category_test import (
    ALL_PAIRS, PAIR_CATEGORIES, ALL_CATEGORIES, N_CATEGORIES,
    CORRUPTED_BASELINE, TEMPLATE, rms_norm_single, cosine_sim,
    load_model_bf16, load_mlp_weights, _load_ln_weight,
)


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Classification utilities =====
def loo_centroid_accuracy(X, labels):
    """Leave-one-out nearest centroid classification."""
    n = X.shape[0]
    unique_labels = sorted(set(labels))
    correct = 0
    for i in range(n):
        centroids = {}
        for lab in unique_labels:
            mask = [j for j in range(n) if j != i and labels[j] == lab]
            if len(mask) > 0:
                centroids[lab] = np.mean(X[mask], axis=0)
            else:
                centroids[lab] = np.zeros(X.shape[1])
        dists = {lab: np.linalg.norm(X[i] - c) for lab, c in centroids.items()}
        pred = min(dists, key=dists.get)
        if pred == labels[i]:
            correct += 1
    return correct / n


# ===== Part 1: Data collection =====
def collect_all_data(model, tokenizer, model_name, target_layers):
    """
    Collect residual states and logit information for all pairs.
    Returns dict with h_clean, h_corrupt, logits, token_ids for each layer.
    """
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    W_U = None  # Load once, cache
    
    all_data = {}
    for l in target_layers:
        log(f"  Collecting Layer {l}...")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        h_post_clean_list = []
        h_post_corrupt_list = []
        logit_clean_list = []
        logit_corrupt_list = []
        target_token_ids = []
        competitor_token_ids = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs} (layer {l})")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # Tokenize
            clean_toks = tokenizer(clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=64)
            corrupt_toks = tokenizer(corrupt_prompt, return_tensors="pt",
                                     truncation=True, max_length=64)
            
            # Get token IDs for target and competitor
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)
            
            # Clean forward pass
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_toks["input_ids"].to(input_device),
                    attention_mask=clean_toks["attention_mask"].to(input_device),
                    output_hidden_states=True,
                )
            
            last_pos = clean_toks["input_ids"].shape[1] - 1
            h_clean = clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
            logits_clean = clean_out.logits[0, -1].float().cpu().numpy()
            h_post_clean_list.append(h_clean)
            logit_clean_list.append(logits_clean)
            del clean_out
            
            # Corrupt forward pass
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=corrupt_toks["input_ids"].to(input_device),
                    attention_mask=corrupt_toks["attention_mask"].to(input_device),
                    output_hidden_states=True,
                )
            
            last_pos_r = corrupt_toks["input_ids"].shape[1] - 1
            h_corrupt = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            logits_corrupt = corrupt_out.logits[0, -1].float().cpu().numpy()
            h_post_corrupt_list.append(h_corrupt)
            logit_corrupt_list.append(logits_corrupt)
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        # Load W_U for logit lens
        if W_U is None:
            W_U = get_W_U(model, model_name)
        
        all_data[str(l)] = {
            "h_post_clean": np.array(h_post_clean_list),     # (n_pairs, d)
            "h_post_corrupt": np.array(h_post_corrupt_list),  # (n_pairs, d)
            "logit_clean": np.array(logit_clean_list),        # (n_pairs, vocab)
            "logit_corrupt": np.array(logit_corrupt_list),    # (n_pairs, vocab)
            "target_token_ids": np.array(target_token_ids),   # (n_pairs,)
            "competitor_token_ids": np.array(competitor_token_ids),  # (n_pairs,)
            "ln_weight": ln_weight,
            "W_U": W_U,
        }
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


# ===== Part 2: Factor R² decomposition =====
def factor_r2_decomposition(dh_proper, all_factor_data):
    """
    Compute R² for each factor's ability to explain dh_proper variance.
    
    For scalar factors: R² = 1 - ||dh - regression_prediction||² / ||dh - mean||²
    For categorical factors: R² = 1 - ||dh - category_means||² / ||dh - mean||²
    """
    n, d = dh_proper.shape
    M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
    total_var = np.sum(M ** 2)
    
    results = {}
    
    for fname, fdata in all_factor_data.items():
        if fname == "category":
            # One-way ANOVA R²: category means explain variance
            labels = fdata  # list of category labels
            unique_labels = sorted(set(labels))
            pred = np.zeros_like(M)
            for lab in unique_labels:
                mask = [i for i, l in enumerate(labels) if l == lab]
                centroid = np.mean(M[mask], axis=0)
                for i in mask:
                    pred[i] = centroid
            residual = M - pred
            r2 = 1.0 - np.sum(residual ** 2) / max(total_var, 1e-10)
            
        elif fname == "object_identity":
            # One-way ANOVA R²: object means explain variance
            labels = fdata
            unique_labels = sorted(set(labels))
            pred = np.zeros_like(M)
            for lab in unique_labels:
                mask = [i for i, l in enumerate(labels) if l == lab]
                if len(mask) > 0:
                    centroid = np.mean(M[mask], axis=0)
                    for i in mask:
                        pred[i] = centroid
            residual = M - pred
            r2 = 1.0 - np.sum(residual ** 2) / max(total_var, 1e-10)
            
        elif fname.startswith("scalar_"):
            # Scalar regression: regress dh onto centered scalar
            scalar = fdata
            s_centered = (scalar - scalar.mean())[:, None]  # (n, 1)
            # Solve: dh ≈ s_centered @ beta  => beta = (s^T s)^-1 s^T dh
            sts = np.sum(s_centered ** 2)
            if sts < 1e-10:
                r2 = 0.0
            else:
                beta = (s_centered.T @ M) / sts  # (1, d)
                pred = s_centered @ beta
                residual = M - pred
                r2 = 1.0 - np.sum(residual ** 2) / max(total_var, 1e-10)
        else:
            continue
        
        results[fname] = float(r2)
    
    return results


# ===== Part 3: PC-factor correlation matrix =====
def pc_factor_correlation(dh_proper, all_factor_data, n_pcs=10):
    """
    Compute correlation between PC scores and each factor.
    
    Returns: dict {factor_name: {PC1: corr, PC2: corr, ...}}
    """
    M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return {}
    
    n_pcs = min(n_pcs, S.shape[0])
    pc_scores = U[:, :n_pcs] * S[:n_pcs]  # (n, n_pcs)
    
    results = {}
    for fname, fdata in all_factor_data.items():
        if fname in ("category", "object_identity"):
            # For categorical: compute F-statistic equivalent
            # One-way ANOVA F-stat per PC
            labels = fdata
            unique_labels = sorted(set(labels))
            pc_corrs = {}
            for k in range(n_pcs):
                scores = pc_scores[:, k]
                # Compute eta² (effect size) for each PC across categories
                grand_mean = scores.mean()
                ss_between = sum(len([j for j, l in enumerate(labels) if l == lab]) *
                                 (np.mean([scores[j] for j, l in enumerate(labels) if l == lab]) - grand_mean)**2
                                 for lab in unique_labels)
                ss_total = np.sum((scores - grand_mean)**2)
                eta_sq = ss_between / max(ss_total, 1e-10)
                pc_corrs[f"PC{k+1}"] = float(np.sqrt(eta_sq))  # sqrt(eta²) ≈ correlation
            results[fname] = pc_corrs
            
        elif fname.startswith("scalar_"):
            scalar = fdata
            s_centered = scalar - scalar.mean()
            pc_corrs = {}
            for k in range(n_pcs):
                scores = pc_scores[:, k]
                if np.std(scores) < 1e-10 or np.std(s_centered) < 1e-10:
                    pc_corrs[f"PC{k+1}"] = 0.0
                else:
                    corr = np.corrcoef(scores, s_centered)[0, 1]
                    pc_corrs[f"PC{k+1}"] = float(corr)
            results[fname] = pc_corrs
    
    # Also add variance explained per PC
    variance_explained = (S[:n_pcs] ** 2) / np.sum(S ** 2) * 100
    results["__variance_explained_pct__"] = {f"PC{k+1}": float(variance_explained[k]) for k in range(n_pcs)}
    
    return results


# ===== Part 4: DS7B PC1 multi-factor regression =====
def pc1_semantic_regression(dh_proper, all_factor_data):
    """
    Multi-factor regression to decode PC1 semantics.
    Regress PC1 scores against all candidate predictors simultaneously.
    """
    M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return {}
    
    pc1_scores = U[:, 0] * S[0]  # (n,)
    
    # Build design matrix from all factors
    predictors = {}
    for fname, fdata in all_factor_data.items():
        if fname in ("category", "object_identity"):
            # One-hot encode
            labels = fdata
            unique_labels = sorted(set(labels))
            for lab in unique_labels:
                predictors[f"{fname}_{lab}"] = np.array([1.0 if l == lab else 0.0 for l in labels])
        elif fname.startswith("scalar_"):
            predictors[fname] = fdata
    
    if not predictors:
        return {}
    
    # Design matrix: each column is a predictor
    pred_names = sorted(predictors.keys())
    X = np.column_stack([predictors[name] for name in pred_names])
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # Remove collinear columns
    # Simple approach: keep only if std > 0
    keep_mask = np.std(X_centered, axis=0) > 1e-10
    if not np.any(keep_mask):
        return {"pc1_r2_total": 0.0}
    
    X_centered = X_centered[:, keep_mask]
    pred_names_kept = [name for name, k in zip(pred_names, keep_mask) if k]
    
    # Add intercept
    n = X_centered.shape[0]
    X_design = np.column_stack([np.ones(n), X_centered])
    
    # OLS regression
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_design, pc1_scores, rcond=None)
    except:
        return {"pc1_r2_total": 0.0}
    
    # R²
    pred_values = X_design @ beta
    ss_res = np.sum((pc1_scores - pred_values)**2)
    ss_tot = np.sum((pc1_scores - pc1_scores.mean())**2)
    r2_total = 1.0 - ss_res / max(ss_tot, 1e-10)
    
    # Individual beta coefficients (standardized)
    beta_standardized = {}
    for i, name in enumerate(pred_names_kept):
        idx = i + 1  # +1 for intercept
        # Standardized beta = beta * std(X) / std(y)
        std_x = np.std(X_centered[:, i])
        std_y = np.std(pc1_scores)
        if std_y > 1e-10:
            beta_standardized[name] = float(beta[idx] * std_x / std_y)
        else:
            beta_standardized[name] = 0.0
    
    # Also compute individual R² for each predictor
    individual_r2 = {}
    for i, name in enumerate(pred_names_kept):
        x_col = X_centered[:, i:i+1]
        try:
            b_ind = np.linalg.lstsq(x_col, pc1_scores, rcond=None)[0]
            pred_ind = x_col @ b_ind
            ss_res_ind = np.sum((pc1_scores - pred_ind)**2)
            individual_r2[name] = float(1.0 - ss_res_ind / max(ss_tot, 1e-10))
        except:
            individual_r2[name] = 0.0
    
    return {
        "pc1_r2_total": float(r2_total),
        "pc1_individual_r2": individual_r2,
        "pc1_standardized_betas": beta_standardized,
        "n_predictors": len(pred_names_kept),
    }


# ===== Part 5: Category swap causal test (logit lens) =====
def category_swap_causal_test(dh_proper, all_factor_data, W_U, ln_weight, h_clean, h_corrupt):
    """
    Test whether swapping category component between pairs causes
    logit differences to switch.
    
    Method:
    1. Compute category subspace (centroid directions)
    2. For each cross-category pair, swap category components
    3. Measure logit difference change via logit lens (RMSNorm + W_U projection)
    """
    n, d = dh_proper.shape
    labels = all_factor_data["category"]
    unique_labels = sorted(set(labels))
    
    # Compute category subspace
    cat_centroids = {}
    for cat in unique_labels:
        idx = [i for i, c in enumerate(labels) if c == cat]
        cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
    
    overall = np.mean(dh_proper, axis=0)
    cat_dirs = np.array([cat_centroids[cat] - overall for cat in unique_labels])
    
    # QR decomposition for category subspace basis
    Q, _ = np.linalg.qr(cat_dirs.T)  # Q: (d, n_cat)
    
    # Project dh_proper onto category subspace
    cat_components = dh_proper @ Q  # (n, n_cat)
    cat_projections = cat_components @ Q.T  # (n, d) - category part of dh
    noncat_projections = dh_proper - cat_projections  # (n, d) - non-category part
    
    results = {
        "same_category_swap": [],
        "cross_category_swap": [],
    }
    
    # For efficiency, only test a subset of pairs
    np.random.seed(42)
    n_test = min(40, n)
    test_indices = np.random.choice(n, n_test, replace=False)
    
    target_token_ids = all_factor_data.get("target_token_ids", None)
    competitor_token_ids = all_factor_data.get("competitor_token_ids", None)
    
    if target_token_ids is None or W_U is None:
        log("    WARNING: Missing token IDs or W_U, skipping swap test")
        return results
    
    for idx in test_indices:
        t_id = int(target_token_ids[idx])
        c_id = int(competitor_token_ids[idx])
        if t_id < 0 or c_id < 0:
            continue
        
        cat_i = labels[idx]
        h_c = h_clean[idx]
        
        # Original logit diff (via logit lens on clean)
        h_c_norm = rms_norm_single(h_c, ln_weight)
        logit_orig = W_U @ h_c_norm
        orig_diff = float(logit_orig[t_id] - logit_orig[c_id])
        
        # === Same-category swap ===
        # Find another pair in same category
        same_cat_idx = [i for i, c in enumerate(labels) if c == cat_i and i != idx]
        if len(same_cat_idx) > 0:
            j = same_cat_idx[np.random.randint(len(same_cat_idx))]
            # Swap category components: use j's category component but idx's non-cat component
            h_swap_same = h_corrupt[idx] + noncat_projections[idx] + cat_projections[j]
            h_swap_same_norm = rms_norm_single(h_swap_same, ln_weight)
            logit_swap = W_U @ h_swap_same_norm
            swap_diff = float(logit_swap[t_id] - logit_swap[c_id])
            
            results["same_category_swap"].append({
                "orig_diff": orig_diff,
                "swap_diff": swap_diff,
                "change": swap_diff - orig_diff,
            })
        
        # === Cross-category swap ===
        diff_cats = [c for c in unique_labels if c != cat_i]
        if len(diff_cats) > 0:
            target_cat = diff_cats[np.random.randint(len(diff_cats))]
            target_cat_idx = [i for i, c in enumerate(labels) if c == target_cat]
            if len(target_cat_idx) > 0:
                j = target_cat_idx[np.random.randint(len(target_cat_idx))]
                # Replace idx's category component with j's
                h_swap_cross = h_corrupt[idx] + noncat_projections[idx] + cat_projections[j]
                h_swap_cross_norm = rms_norm_single(h_swap_cross, ln_weight)
                logit_swap = W_U @ h_swap_cross_norm
                swap_diff = float(logit_swap[t_id] - logit_swap[c_id])
                
                results["cross_category_swap"].append({
                    "orig_diff": orig_diff,
                    "swap_diff": swap_diff,
                    "change": swap_diff - orig_diff,
                    "from_cat": cat_i,
                    "to_cat": target_cat,
                })
    
    return results


# ===== Main analysis function =====
def run_full_analysis(all_data, model_name):
    """Run all analysis parts on collected data."""
    log("\n" + "="*60)
    log("Phase 382: Multi-factor Residual Decomposition")
    log("="*60)
    
    full_results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        n_pairs = d["h_post_clean"].shape[0]
        W_U = d.get("W_U", None)
        
        log(f"\n--- Layer {l} ---")
        
        h_clean = d["h_post_clean"]
        h_corrupt = d["h_post_corrupt"]
        t_ids = d["target_token_ids"]
        c_ids = d["competitor_token_ids"]
        
        # Compute PROPER post-RMSNorm difference
        h_clean_norm = np.zeros_like(h_clean)
        h_corrupt_norm = np.zeros_like(h_corrupt)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Norm stats
        norm_clean = np.linalg.norm(h_clean, axis=1)
        norm_corrupt = np.linalg.norm(h_corrupt, axis=1)
        norm_ratio = norm_clean / (norm_corrupt + 1e-10)
        norm_diff = norm_clean - norm_corrupt
        
        # Compute logit-based factors
        logit_diff_clean = np.array([
            float(d["logit_clean"][i, t_ids[i]] - d["logit_clean"][i, c_ids[i]])
            if t_ids[i] >= 0 and c_ids[i] >= 0 else 0.0
            for i in range(n_pairs)
        ])
        
        logit_target_clean = np.array([
            float(d["logit_clean"][i, t_ids[i]]) if t_ids[i] >= 0 else 0.0
            for i in range(n_pairs)
        ])
        
        logit_target_corrupt = np.array([
            float(d["logit_corrupt"][i, t_ids[i]]) if t_ids[i] >= 0 else 0.0
            for i in range(n_pairs)
        ])
        
        # Binding strength: probability of target given clean prompt
        # P(target) = softmax(logit)[target] ≈ exp(logit_target) / sum(exp(logit))
        # Use logit as proxy (monotonic)
        logit_entropy_clean = np.array([
            -np.sum(np.exp(d["logit_clean"][i]) / np.sum(np.exp(d["logit_clean"][i])) * 
                    np.log(np.exp(d["logit_clean"][i]) / np.sum(np.exp(d["logit_clean"][i])) + 1e-10))
            for i in range(n_pairs)
        ])
        
        # Build factor data dictionary
        category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
        object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
        
        all_factor_data = {
            "category": category_labels,
            "object_identity": object_labels,
            "scalar_norm_ratio": norm_ratio,
            "scalar_norm_diff": norm_diff,
            "scalar_norm_clean": norm_clean,
            "scalar_norm_corrupt": norm_corrupt,
            "scalar_logit_diff": logit_diff_clean,
            "scalar_logit_target_clean": logit_target_clean,
            "scalar_logit_target_corrupt": logit_target_corrupt,
            "scalar_entropy_clean": logit_entropy_clean,
            "target_token_ids": t_ids,
            "competitor_token_ids": c_ids,
        }
        
        # === Part 2: Factor R² decomposition ===
        log("  Part 2: Factor R² decomposition...")
        r2_results = factor_r2_decomposition(dh_proper, all_factor_data)
        for fname, r2 in sorted(r2_results.items(), key=lambda x: -x[1]):
            log(f"    {fname:40s}: R² = {r2:.4f}")
        
        # === Part 3: PC-factor correlation matrix ===
        log("  Part 3: PC-factor correlation matrix...")
        pc_corr_results = pc_factor_correlation(dh_proper, all_factor_data, n_pcs=10)
        
        if "__variance_explained_pct__" in pc_corr_results:
            ve = pc_corr_results["__variance_explained_pct__"]
            log(f"    Variance explained: " + ", ".join(f"PC{k+1}={ve[f'PC{k+1}']:.1f}%" for k in range(min(5, len(ve)))))
        
        for fname in sorted(pc_corr_results.keys()):
            if fname.startswith("__"):
                continue
            corrs = pc_corr_results[fname]
            top_pc = max(corrs.keys(), key=lambda k: abs(corrs[k]))
            log(f"    {fname:40s}: top align={top_pc} (corr={corrs[top_pc]:.3f})")
        
        # === Part 4: PC1 semantic regression ===
        log("  Part 4: PC1 semantic regression...")
        pc1_reg = pc1_semantic_regression(dh_proper, all_factor_data)
        log(f"    PC1 R² total: {pc1_reg.get('pc1_r2_total', 0):.4f}")
        
        if pc1_reg.get("pc1_individual_r2"):
            top_factors = sorted(pc1_reg["pc1_individual_r2"].items(), key=lambda x: -x[1])[:5]
            log(f"    Top individual R²: " + ", ".join(f"{name}={r2:.4f}" for name, r2 in top_factors))
        
        if pc1_reg.get("pc1_standardized_betas"):
            top_betas = sorted(pc1_reg["pc1_standardized_betas"].items(), key=lambda x: -abs(x[1]))[:5]
            log(f"    Top standardized β: " + ", ".join(f"{name}={beta:.3f}" for name, beta in top_betas))
        
        # === Part 5: Category swap causal test ===
        log("  Part 5: Category swap causal test (logit lens)...")
        swap_results = category_swap_causal_test(
            dh_proper, all_factor_data, W_U, ln_weight, h_clean, h_corrupt)
        
        if swap_results["same_category_swap"]:
            same_changes = [s["change"] for s in swap_results["same_category_swap"]]
            log(f"    Same-cat swap: mean Δ = {np.mean(same_changes):.4f}, "
                f"std = {np.std(same_changes):.4f}, n = {len(same_changes)}")
        
        if swap_results["cross_category_swap"]:
            cross_changes = [s["change"] for s in swap_results["cross_category_swap"]]
            log(f"    Cross-cat swap: mean Δ = {np.mean(cross_changes):.4f}, "
                f"std = {np.std(cross_changes):.4f}, n = {len(cross_changes)}")
        
        # Store results
        layer_result = {
            "layer": l,
            "n_pairs": n_pairs,
            "factor_r2": r2_results,
            "pc_factor_correlation": pc_corr_results,
            "pc1_regression": pc1_reg,
            "category_swap": swap_results,
        }
        
        # Summary stats for swap
        if swap_results["same_category_swap"]:
            sc = [s["change"] for s in swap_results["same_category_swap"]]
            layer_result["swap_summary"] = {
                "same_cat_mean_change": float(np.mean(sc)),
                "same_cat_std_change": float(np.std(sc)),
            }
        if swap_results["cross_category_swap"]:
            cc = [s["change"] for s in swap_results["cross_category_swap"]]
            if "swap_summary" not in layer_result:
                layer_result["swap_summary"] = {}
            layer_result["swap_summary"]["cross_cat_mean_change"] = float(np.mean(cc))
            layer_result["swap_summary"]["cross_cat_std_change"] = float(np.std(cc))
        
        full_results[l_str] = layer_result
    
    return full_results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 382: Multi-factor Residual Decomposition — {model_name}")
    log(f"=" * 60)
    
    # Target layers
    if model_name == "deepseek7b":
        target_layers = [4, 8, 12, 16, 20, 24]
    elif model_name == "qwen3":
        target_layers = [4, 12, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 20, 30]
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    # Part 1: Collect data
    log("\nPart 1: Collecting residual states and logit data...")
    all_data = collect_all_data(model, tokenizer, model_name, target_layers)
    
    # Run analysis
    log("\nRunning multi-factor analysis...")
    results = run_full_analysis(all_data, model_name)
    
    # Save results
    out_dir = "results/phase382_factor_decomposition"
    os.makedirs(out_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Remove large arrays from results to save space
    clean_results = {}
    for k, v in results.items():
        clean_v = {}
        for kk, vv in v.items():
            if kk in ("pc_factor_correlation",):
                # Keep structure but remove __variance_explained_pct__ for brevity
                clean_v[kk] = {kkk: convert(vvv) for kkk, vvv in vv.items()}
            elif kk == "category_swap":
                # Only keep summary stats, not individual swap results
                clean_v[kk] = {
                    "same_cat_n": len(vv.get("same_category_swap", [])),
                    "cross_cat_n": len(vv.get("cross_category_swap", [])),
                    "same_cat_changes": [convert(s["change"]) for s in vv.get("same_category_swap", [])],
                    "cross_cat_changes": [convert(s["change"]) for s in vv.get("cross_category_swap", [])],
                }
            else:
                clean_v[kk] = convert(vv)
        clean_results[k] = clean_v
    
    full_output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_pairs": len(ALL_PAIRS),
        "n_categories": N_CATEGORIES,
        "test": "phase382_factor_decomposition",
        "results": clean_results,
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase382.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 382 complete for {model_name}!")


if __name__ == "__main__":
    main()
