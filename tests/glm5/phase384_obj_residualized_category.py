"""
Phase 384: 对象身份去除 + Partial R²方差分割 + 净化类别因果测试
=================================================================

核心目标（解决Phase 382-383b的两大硬伤）：
1. 先回归掉object_identity，在residualized dh上提取category subspace
   → 解决"category subspace被object identity污染"问题
2. 做partial R²方差分割（unique R², shared R²）
   → 解决"object_identity R²=99%是否过拟合"问题
3. 用净化后的category分量做因果patch
   → 验证"去除污染后类别因果方向是否正常"

方法：
Part 1: 收集所有residual states（同Phase 382）
Part 2: Partial R²方差分割
  - 对每个因子，计算unique R²（排除其他因子后的独立贡献）
  - 计算shared R²（因子间的共享方差）
  - 用Type III SS (方差分析)方法
Part 3: 对象身份residualization
  - dh_resid = dh - predict(dh, object_identity_dummies)
  - 在dh_resid上提取category subspace
  - 对比: raw category R² vs residualized category R²
Part 4: 净化类别因果测试
  - 用residualized category subspace做add/remove/swap
  - 对比: 污染category vs 净化category的因果效应

用法:
  python tests/glm5/phase384_obj_residualized_category.py qwen3
  python tests/glm5/phase384_obj_residualized_category.py deepseek7b
  python tests/glm5/phase384_obj_residualized_category.py glm4
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
    load_model_bf16, _load_ln_weight,
)


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Part 1: Data collection (same as Phase 382) =====
def collect_all_data(model, tokenizer, model_name, target_layers):
    """Collect residual states and logit info for all pairs."""
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    W_U = None

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

            clean_toks = tokenizer(clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=64)
            corrupt_toks = tokenizer(corrupt_prompt, return_tensors="pt",
                                     truncation=True, max_length=64)

            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)

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

        if W_U is None:
            W_U = get_W_U(model, model_name)

        all_data[str(l)] = {
            "h_post_clean": np.array(h_post_clean_list),
            "h_post_corrupt": np.array(h_post_corrupt_list),
            "logit_clean": np.array(logit_clean_list),
            "logit_corrupt": np.array(logit_corrupt_list),
            "target_token_ids": np.array(target_token_ids),
            "competitor_token_ids": np.array(competitor_token_ids),
            "ln_weight": ln_weight,
            "W_U": W_U,
        }
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")

    return all_data


# ===== Part 2: Partial R² variance partition =====
def partial_r2_variance_partition(dh, category_labels, object_labels, norm_ratios, n_perm=500):
    """
    Compute partial (unique) R² for each factor using Type III SS approach.
    
    For each factor, unique R² = R²(full_model) - R²(model_without_factor)
    
    Also computes:
    - Individual R² (single factor)
    - Shared R² (by subtraction)
    
    For categorical factors, uses one-hot encoding.
    For scalar factors, uses simple regression.
    
    Returns dict with:
    - individual_r2: R² when only this factor is used
    - unique_r2: R² unique to this factor (controlling for others)
    - shared_r2: R² shared between factors
    """
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    total_var = np.sum(M ** 2)
    
    if total_var < 1e-10:
        return {"individual_r2": {}, "unique_r2": {}, "shared_r2": {}, "total_r2": 0.0}
    
    # Build design matrices for each factor
    factor_matrices = {}
    
    # Category: one-hot encoding (k-1 dummies to avoid collinearity)
    unique_cats = sorted(set(category_labels))
    cat_onehot = np.zeros((n, len(unique_cats)))
    for i, cat in enumerate(unique_cats):
        cat_onehot[:, i] = [1.0 if c == cat else 0.0 for c in category_labels]
    # Remove last column (reference category) for identifiability
    if cat_onehot.shape[1] > 1:
        factor_matrices["category"] = cat_onehot[:, :-1]
    else:
        factor_matrices["category"] = cat_onehot
    
    # Object identity: one-hot encoding (k-1 dummies)
    unique_objs = sorted(set(object_labels))
    obj_onehot = np.zeros((n, len(unique_objs)))
    for i, obj in enumerate(unique_objs):
        obj_onehot[:, i] = [1.0 if o == obj else 0.0 for o in object_labels]
    if obj_onehot.shape[1] > 1:
        factor_matrices["object_identity"] = obj_onehot[:, :-1]
    else:
        factor_matrices["object_identity"] = obj_onehot
    
    # Norm ratio: scalar
    nr_centered = (norm_ratios - norm_ratios.mean())[:, None]
    if np.std(nr_centered) > 1e-10:
        factor_matrices["norm_ratio"] = nr_centered
    
    # Compute individual R² for each factor
    individual_r2 = {}
    for fname, X_f in factor_matrices.items():
        X_design = np.column_stack([np.ones(n), X_f])
        try:
            beta = np.linalg.lstsq(X_design, M, rcond=None)[0]
            pred = X_design @ beta
            ss_res = np.sum((M - pred)**2)
            r2 = 1.0 - ss_res / total_var
            individual_r2[fname] = max(0.0, float(r2))
        except:
            individual_r2[fname] = 0.0
    
    # Compute full model R²
    all_X = np.column_stack([X_f for X_f in factor_matrices.values()])
    X_full = np.column_stack([np.ones(n), all_X])
    try:
        beta_full = np.linalg.lstsq(X_full, M, rcond=None)[0]
        pred_full = X_full @ beta_full
        ss_res_full = np.sum((M - pred_full)**2)
        r2_full = 1.0 - ss_res_full / total_var
    except:
        r2_full = 0.0
    
    # Compute unique R² for each factor (leave-one-out)
    unique_r2 = {}
    for fname, X_f in factor_matrices.items():
        # Model without this factor
        other_factors = {k: v for k, v in factor_matrices.items() if k != fname}
        if not other_factors:
            # This is the only factor
            unique_r2[fname] = individual_r2.get(fname, 0.0)
            continue
        
        other_X = np.column_stack([v for v in other_factors.values()])
        X_reduced = np.column_stack([np.ones(n), other_X])
        try:
            beta_red = np.linalg.lstsq(X_reduced, M, rcond=None)[0]
            pred_red = X_reduced @ beta_red
            ss_res_red = np.sum((M - pred_red)**2)
            r2_red = 1.0 - ss_res_red / total_var
            unique_r2[fname] = max(0.0, float(r2_full - r2_red))
        except:
            unique_r2[fname] = 0.0
    
    # Shared R² = individual R² - unique R² (for each factor)
    shared_r2 = {}
    for fname in factor_matrices:
        shared_r2[fname] = max(0.0, individual_r2.get(fname, 0.0) - unique_r2.get(fname, 0.0))
    
    # Permutation test for unique R² significance
    perm_pvalues = {}
    rng = np.random.RandomState(42)
    for fname in factor_matrices:
        obs_unique = unique_r2.get(fname, 0.0)
        if obs_unique < 1e-6:
            perm_pvalues[fname] = 1.0
            continue
        
        count_geq = 0
        for _ in range(n_perm):
            perm_labels = rng.permutation(n)
            # Permute this factor's design matrix
            X_f_perm = factor_matrices[fname][perm_labels]
            
            # Full model with permuted factor
            all_X_perm = []
            for k, v in factor_matrices.items():
                if k == fname:
                    all_X_perm.append(X_f_perm)
                else:
                    all_X_perm.append(v)
            all_X_perm = np.column_stack(all_X_perm)
            X_full_perm = np.column_stack([np.ones(n), all_X_perm])
            
            try:
                beta_perm = np.linalg.lstsq(X_full_perm, M, rcond=None)[0]
                pred_perm = X_full_perm @ beta_perm
                r2_full_perm = 1.0 - np.sum((M - pred_perm)**2) / total_var
            except:
                r2_full_perm = 0.0
            
            # Reduced model (without this factor, same as before)
            other_factors = {k: v for k, v in factor_matrices.items() if k != fname}
            if other_factors:
                other_X = np.column_stack([v for v in other_factors.values()])
                X_red_perm = np.column_stack([np.ones(n), other_X])
                try:
                    beta_red_p = np.linalg.lstsq(X_red_perm, M, rcond=None)[0]
                    pred_red_p = X_red_perm @ beta_red_p
                    r2_red_perm = 1.0 - np.sum((M - pred_red_p)**2) / total_var
                except:
                    r2_red_perm = 0.0
            else:
                r2_red_perm = 0.0
            
            perm_unique = max(0.0, r2_full_perm - r2_red_perm)
            if perm_unique >= obs_unique:
                count_geq += 1
        
        perm_pvalues[fname] = (count_geq + 1) / (n_perm + 1)
    
    return {
        "individual_r2": individual_r2,
        "unique_r2": unique_r2,
        "shared_r2": shared_r2,
        "total_r2": float(r2_full),
        "perm_pvalues": perm_pvalues,
        "n_perm": n_perm,
    }


# ===== Part 3: Object identity residualization =====
def residualize_object_identity(dh, object_labels):
    """
    Remove object identity component from dh.
    
    Method: regress dh onto object identity dummies, keep residual.
    This removes all variance linearly attributable to which object it is.
    """
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    
    # Build one-hot encoding for objects
    unique_objs = sorted(set(object_labels))
    obj_onehot = np.zeros((n, len(unique_objs)))
    for i, obj in enumerate(unique_objs):
        obj_onehot[:, i] = [1.0 if o == obj else 0.0 for o in object_labels]
    
    # Remove reference category
    if obj_onehot.shape[1] > 1:
        X_obj = obj_onehot[:, :-1]
    else:
        X_obj = obj_onehot
    
    # Regression: M = X_obj @ beta + residual
    X_design = np.column_stack([np.ones(n), X_obj])
    try:
        beta = np.linalg.lstsq(X_design, M, rcond=None)[0]
        predicted = X_design @ beta
        residual = M - predicted
    except:
        residual = M
    
    return residual, predicted


def extract_category_subspace(dh, category_labels):
    """Extract category subspace using centroid differences."""
    unique_labels = sorted(set(category_labels))
    cat_centroids = {}
    for cat in unique_labels:
        idx = [i for i, c in enumerate(category_labels) if c == cat]
        cat_centroids[cat] = np.mean(dh[idx], axis=0)
    
    overall = np.mean(dh, axis=0)
    cat_dirs = np.array([cat_centroids[cat] - overall for cat in unique_labels])
    Q, _ = np.linalg.qr(cat_dirs.T)
    return Q, cat_centroids, overall


def compute_category_r2(dh, category_labels):
    """Compute R² of category labels explaining dh variance."""
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    total_var = np.sum(M**2)
    if total_var < 1e-10:
        return 0.0
    
    unique_labels = sorted(set(category_labels))
    pred = np.zeros_like(M)
    for cat in unique_labels:
        idx = [i for i, c in enumerate(category_labels) if c == cat]
        centroid = np.mean(M[idx], axis=0)
        for i in idx:
            pred[i] = centroid
    
    residual = M - pred
    r2 = 1.0 - np.sum(residual**2) / total_var
    return max(0.0, float(r2))


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


# ===== Part 4: Clean category causal patching =====
def run_model_with_patch(model, tokenizer, device, prompt, layer_idx,
                          patch_delta, target_token_id, competitor_token_id):
    """Run model with delta added to residual at layer l (last token)."""
    if target_token_id < 0 or competitor_token_id < 0:
        return None

    layers = get_layers(model)
    delta_tensor = torch.tensor(patch_delta, dtype=torch.bfloat16, device=device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h_patched = h.clone()
        h_patched[0, -1, :] += delta_tensor
        if isinstance(output, tuple):
            return (h_patched,) + output[1:]
        return h_patched

    hook = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            out = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )
            logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        log(f"    Forward failed: {str(e)[:60]}")
        hook.remove()
        return None

    hook.remove()
    return float(logits[target_token_id] - logits[competitor_token_id])


def clean_category_causal_test(model, tokenizer, device, model_name, target_layers,
                                h_clean_dict, h_corrupt_dict, ln_weights, dh_proper_dict,
                                dh_resid_dict, n_test=60):
    """
    Compare raw category subspace vs residualized category subspace for causal patching.
    
    For each layer:
    1. Extract category subspace from dh_proper (raw, contaminated)
    2. Extract category subspace from dh_resid (residualized, clean)
    3. Do add/remove/swap for both
    4. Compare causal effects
    """
    # Get n_all from first layer's data
    first_l = target_layers[0]
    n_all = h_clean_dict[first_l].shape[0]
    labels = [PAIR_CATEGORIES[i] for i in range(n_all)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_all)]
    
    np.random.seed(789)
    test_indices = sorted(np.random.choice(n_all, min(n_test, n_all), replace=False).tolist())
    
    results = {}
    
    for l in target_layers:
        log(f"  Causal test Layer {l}...")
        t_l = time.time()
        ln_weight = ln_weights.get(l)
        
        dh_proper = dh_proper_dict[l]
        dh_resid = dh_resid_dict[l]
        h_clean = h_clean_dict[l]
        h_corrupt = h_corrupt_dict[l]
        
        # === Category subspace from raw dh_proper ===
        Q_raw, cat_centroids_raw, overall_raw = extract_category_subspace(dh_proper, labels)
        cat_comp_raw = dh_proper @ Q_raw
        cat_proj_raw = cat_comp_raw @ Q_raw.T
        
        # === Category subspace from residualized dh ===
        Q_clean, cat_centroids_clean, overall_clean = extract_category_subspace(dh_resid, labels)
        cat_comp_clean = dh_resid @ Q_clean
        cat_proj_clean = cat_comp_clean @ Q_clean.T
        
        # R² comparison
        r2_raw = compute_category_r2(dh_proper, labels)
        r2_resid = compute_category_r2(dh_resid, labels)
        
        # Classification accuracy comparison
        acc_raw = loo_centroid_accuracy(dh_proper, labels)
        acc_resid = loo_centroid_accuracy(dh_resid, labels)
        
        log(f"    R²: raw={r2_raw:.4f}, resid={r2_resid:.4f}")
        log(f"    Accuracy: raw={acc_raw:.4f}, resid={acc_resid:.4f}")
        
        # === Causal patching ===
        ca_data = {
            "raw": {"add_cat": [], "remove_cat": [], "cross_swap": [], "same_swap": []},
            "clean": {"add_cat": [], "remove_cat": [], "cross_swap": [], "same_swap": []},
        }
        
        baselines = {"clean_ld": [], "corrupt_ld": []}
        
        np.random.seed(321)
        
        for pidx in test_indices:
            obj, target, competitor = ALL_PAIRS[pidx]
            cat_i = labels[pidx]
            
            t_ids_tok = tokenizer.encode(target, add_special_tokens=False)
            c_ids_tok = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids_tok[0] if len(t_ids_tok) > 0 else -1
            c_id = c_ids_tok[0] if len(c_ids_tok) > 0 else -1
            if t_id < 0 or c_id < 0:
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # Baselines
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            clean_logits = clean_out.logits[0, -1].float().cpu().numpy()
            baselines["clean_ld"].append(float(clean_logits[t_id] - clean_logits[c_id]))
            del clean_out
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            corrupt_logits = corrupt_out.logits[0, -1].float().cpu().numpy()
            baselines["corrupt_ld"].append(float(corrupt_logits[t_id] - corrupt_logits[c_id]))
            del corrupt_out
            
            n_base = len(baselines["clean_ld"])
            
            # === Test both raw and clean category projections ===
            for subspace_type, Q_sub, cat_proj_sub in [
                ("raw", Q_raw, cat_proj_raw),
                ("clean", Q_clean, cat_proj_clean),
            ]:
                # Get category component for this pair
                cat_proj_i = cat_proj_sub[pidx]
                
                # Add category to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                           cat_proj_i, t_id, c_id)
                if ld is not None and n_base <= len(ca_data[subspace_type]["add_cat"]) + 1:
                    ca_data[subspace_type]["add_cat"].append(ld)
                
                # Remove category from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           -cat_proj_i, t_id, c_id)
                if ld is not None and n_base <= len(ca_data[subspace_type]["remove_cat"]) + 1:
                    ca_data[subspace_type]["remove_cat"].append(ld)
                
                # Cross-category swap
                diff_cat_idx = [j for j in range(n_all) if labels[j] != cat_i]
                if len(diff_cat_idx) > 0:
                    j_cross = diff_cat_idx[np.random.randint(len(diff_cat_idx))]
                    cat_proj_j = cat_proj_sub[j_cross]
                    swap_delta = cat_proj_j - cat_proj_i
                    ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                               swap_delta, t_id, c_id)
                    if ld is not None:
                        ca_data[subspace_type]["cross_swap"].append(ld)
                
                # Same-category swap
                same_cat_idx = [j for j in range(n_all) if labels[j] == cat_i and j != pidx]
                if len(same_cat_idx) > 0:
                    j_same = same_cat_idx[np.random.randint(len(same_cat_idx))]
                    cat_proj_j = cat_proj_sub[j_same]
                    swap_delta = cat_proj_j - cat_proj_i
                    ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                               swap_delta, t_id, c_id)
                    if ld is not None:
                        ca_data[subspace_type]["same_swap"].append(ld)
            
            if pidx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Compute causal effects
        layer_result = {
            "r2_raw": r2_raw,
            "r2_resid": r2_resid,
            "acc_raw": acc_raw,
            "acc_resid": acc_resid,
        }
        
        for subspace_type in ["raw", "clean"]:
            d = ca_data[subspace_type]
            n_eff = min(len(d["add_cat"]), len(baselines["corrupt_ld"]))
            
            if n_eff > 0:
                add_effect = np.array(d["add_cat"][:n_eff]) - np.array(baselines["corrupt_ld"][:n_eff])
                layer_result[f"{subspace_type}_add_effect"] = {
                    "mean": float(np.mean(add_effect)),
                    "std": float(np.std(add_effect)),
                    "t": float(np.mean(add_effect) / (np.std(add_effect) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
            
            n_eff2 = min(len(d["remove_cat"]), len(baselines["clean_ld"]))
            if n_eff2 > 0:
                rem_effect = np.array(d["remove_cat"][:n_eff2]) - np.array(baselines["clean_ld"][:n_eff2])
                layer_result[f"{subspace_type}_remove_effect"] = {
                    "mean": float(np.mean(rem_effect)),
                    "std": float(np.std(rem_effect)),
                    "t": float(np.mean(rem_effect) / (np.std(rem_effect) / np.sqrt(n_eff2) + 1e-10)),
                    "n": n_eff2,
                }
            
            # Swap: cross vs same
            n_swap = min(len(d["cross_swap"]), len(d["same_swap"]), len(baselines["clean_ld"]))
            if n_swap > 0:
                cross_eff = np.array(d["cross_swap"][:n_swap]) - np.array(baselines["clean_ld"][:n_swap])
                same_eff = np.array(d["same_swap"][:n_swap]) - np.array(baselines["clean_ld"][:n_swap])
                layer_result[f"{subspace_type}_swap_effect"] = {
                    "cross_mean": float(np.mean(cross_eff)),
                    "same_mean": float(np.mean(same_eff)),
                    "diff": float(np.mean(cross_eff) - np.mean(same_eff)),
                    "diff_t": float((np.mean(cross_eff) - np.mean(same_eff)) /
                                    (np.std(cross_eff - same_eff) / np.sqrt(n_swap) + 1e-10)),
                    "n": n_swap,
                }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} causal test done in {time.time()-t_l:.1f}s")
    
    return results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")

    log(f"Phase 384: Object-Residualized Category + Partial R² — {model_name}")
    log(f"=" * 60)

    if model_name == "deepseek7b":
        target_layers = [4, 12, 24]
    elif model_name == "qwen3":
        target_layers = [4, 28]
    elif model_name == "glm4":
        target_layers = [4, 30]

    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers")

    # Part 1: Collect data
    log("\nPart 1: Collecting residual states...")
    all_data = collect_all_data(model, tokenizer, model_name, target_layers)
    
    # Prepare data structures
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    
    dh_proper_all = {}
    dh_resid_all = {}
    h_clean_all = {}
    h_corrupt_all = {}
    ln_weights = {}
    norm_ratios = {}
    
    partial_r2_results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        l = int(l_str)
        d = all_data[l_str]
        ln_weight = d["ln_weight"]
        ln_weights[l] = ln_weight
        
        h_clean = d["h_post_clean"]
        h_corrupt = d["h_post_corrupt"]
        h_clean_all[l] = h_clean
        h_corrupt_all[l] = h_corrupt
        
        # Compute post-RMSNorm
        h_clean_norm = np.zeros_like(h_clean)
        h_corrupt_norm = np.zeros_like(h_corrupt)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        dh_proper_all[l] = dh_proper
        
        # Norm ratios
        norm_clean = np.linalg.norm(h_clean, axis=1)
        norm_corrupt = np.linalg.norm(h_corrupt, axis=1)
        norm_ratios[l] = norm_clean / (norm_corrupt + 1e-10)
        
        # === Part 2: Partial R² variance partition ===
        log(f"\nPart 2: Partial R² for Layer {l}...")
        pR2 = partial_r2_variance_partition(
            dh_proper, category_labels, object_labels, norm_ratios[l], n_perm=500)
        
        partial_r2_results[l_str] = pR2
        
        log(f"  Individual R²: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["individual_r2"].items(), key=lambda x: -x[1])))
        log(f"  Unique R²:     " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["unique_r2"].items(), key=lambda x: -x[1])))
        log(f"  Shared R²:     " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["shared_r2"].items(), key=lambda x: -x[1])))
        log(f"  Total R²: {pR2['total_r2']:.4f}")
        log(f"  Perm p-values: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["perm_pvalues"].items())))
        
        # === Part 3: Object identity residualization ===
        log(f"\nPart 3: Object-residualization for Layer {l}...")
        dh_resid, dh_obj_predicted = residualize_object_identity(dh_proper, object_labels)
        dh_resid_all[l] = dh_resid
        
        # Category R² comparison
        r2_before = compute_category_r2(dh_proper, category_labels)
        r2_after = compute_category_r2(dh_resid, category_labels)
        
        # Classification accuracy comparison
        acc_before = loo_centroid_accuracy(dh_proper, category_labels)
        acc_after = loo_centroid_accuracy(dh_resid, category_labels)
        
        log(f"  Category R²: before={r2_before:.4f}, after={r2_after:.4f}, "
            f"change={r2_after-r2_before:+.4f}")
        log(f"  LOO Accuracy: before={acc_before:.4f}, after={acc_after:.4f}, "
            f"change={acc_after-acc_before:+.4f}")
        
        # How much of category R² was shared with object identity?
        # Category individual R² in original vs residualized
        # If most category R² was shared → r2_after << r2_before
        # If category has unique info → r2_after ≈ r2_before
        ratio = r2_after / max(r2_before, 1e-10)
        log(f"  Residualization ratio (after/before): {ratio:.4f}")
        
        if ratio > 0.5:
            log(f"  → Category has SUBSTANTIAL unique variance (>{ratio:.0%} survives)")
        elif ratio > 0.1:
            log(f"  → Category has MODERATE unique variance ({ratio:.0%} survives)")
        else:
            log(f"  → Category has LITTLE unique variance (only {ratio:.0%} survives)")
    
    # === Part 4: Clean category causal test ===
    log(f"\nPart 4: Clean category causal test...")
    ca_results = clean_category_causal_test(
        model, tokenizer, device, model_name, target_layers,
        h_clean_all, h_corrupt_all, ln_weights, dh_proper_all, dh_resid_all, n_test=60)
    
    # Save results
    out_dir = "results/phase384_obj_residualized_category"
    os.makedirs(out_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    full_output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_pairs": n_pairs,
        "n_categories": N_CATEGORIES,
        "n_test_causal": 60,
        "test": "phase384_obj_residualized_category",
        "partial_r2": convert(partial_r2_results),
        "causal_test": convert(ca_results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase384.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 384 Summary — {model_name}")
    log(f"{'='*60}")
    
    for l_str in sorted(partial_r2_results.keys(), key=int):
        pR2 = partial_r2_results[l_str]
        log(f"\nLayer {l_str}:")
        log(f"  Unique R²: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["unique_r2"].items(), key=lambda x: -x[1])))
        log(f"  Shared R²: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["shared_r2"].items(), key=lambda x: -x[1])))
        log(f"  Perm p:    " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(pR2["perm_pvalues"].items())))
    
    for l_str in sorted(ca_results.keys(), key=int):
        r = ca_results[l_str]
        log(f"\nLayer {l_str} Causal:")
        log(f"  R²: raw={r['r2_raw']:.4f}, resid={r['r2_resid']:.4f}")
        log(f"  Acc: raw={r['acc_raw']:.4f}, resid={r['acc_resid']:.4f}")
        
        for stype in ["raw", "clean"]:
            ae = r.get(f"{stype}_add_effect", {})
            re = r.get(f"{stype}_remove_effect", {})
            se = r.get(f"{stype}_swap_effect", {})
            log(f"  {stype:5s} add:  mean={ae.get('mean', 0):+.4f} t={ae.get('t', 0):.2f}")
            log(f"  {stype:5s} rem:  mean={re.get('mean', 0):+.4f} t={re.get('t', 0):.2f}")
            if se:
                log(f"  {stype:5s} swap: diff={se.get('diff', 0):+.4f} t={se.get('diff_t', 0):.2f}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 384 complete for {model_name}!")


if __name__ == "__main__":
    main()
