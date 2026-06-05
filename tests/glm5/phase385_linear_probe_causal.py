"""
Phase 385: 线性探针因果验证 + 跨层传递追踪
============================================

Phase 384的核心困境：
- add/remove任何向量都伤害logit_diff（不是category的因果效应，而是扰动的一般效应）
- category unique R²只有0.7-2.5%，permutation test不显著
- 原因：centroid-based category subspace不够"纯"，且patch方向与残差流不兼容

Phase 385的新方法：
Part 1: 线性探针提取category方向
  - 在dh_resid（去除object identity后）上训练linear SVM / logistic regression
  - 得到W_cat矩阵（n_categories-1 × d），每行是一个category判别方向
  - 比较linear probe vs centroid方法的category分类准确率和R²

Part 2: 探针方向的因果验证
  - 用W_cat的方向做add/remove/swap，但限制patch的范数
  - 关键创新：**逐步增加patch强度**，观察因果效应是否单调递增
  - 如果category方向真的因果有效，那么小patch也应该有正确方向的效应
  - 如果只是扰动效应，小patch不会有方向特异性

Part 3: 跨层category传递追踪
  - 在每层提取category探针方向
  - 计算相邻层的探针方向的余弦相似度
  - 验证category信息是否在层间被"传递"和"增强"
  - 这是比add/remove更鲁棒的因果证据：如果L4的category方向能预测L24的category方向，
    说明category信息确实在层间流动

Part 4: Counterfactual验证
  - 在L层用探针做patch，观察对最终输出（而非L层的logit lens）的效应
  - 直接patch到模型的中间层，前向传播到最后一层
  - 比较logit lens预测和真实输出的因果效应差异

用法:
  python tests/glm5/phase385_linear_probe_causal.py qwen3
  python tests/glm5/phase385_linear_probe_causal.py deepseek7b
  python tests/glm5/phase385_linear_probe_causal.py glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

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


# ===== Part 1: Linear probe for category direction =====
def train_linear_probe(dh, category_labels, method='logistic'):
    """
    Train a linear probe on dh to predict category labels.
    Returns the weight matrix W (n_classes × d) and bias b.
    
    method: 'logistic' or 'svm'
    """
    n, d = dh.shape
    X = dh - dh.mean(axis=0, keepdims=True)  # center
    
    if method == 'logistic':
        clf = LogisticRegression(
            solver='lbfgs',
            max_iter=2000,
            C=1.0,
            fit_intercept=True,
        )
    elif method == 'svm':
        clf = LinearSVC(
            max_iter=5000,
            C=1.0,
            fit_intercept=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clf.fit(X, category_labels)
    
    # Weight matrix: (n_classes, d)
    W = clf.coef_
    b = clf.intercept_
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(
        LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        if method == 'logistic' else LinearSVC(max_iter=5000, C=1.0),
        X, category_labels, cv=5, scoring='accuracy'
    )
    
    return W, b, clf, float(np.mean(cv_scores))


def compute_probe_r2(dh, category_labels, W):
    """Compute how much variance in dh is explained by probe directions."""
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    total_var = np.sum(M ** 2)
    if total_var < 1e-10:
        return 0.0
    
    # Project onto probe subspace
    Q, _ = np.linalg.qr(W.T)  # (d, n_classes-1) orthonormal basis
    proj = (M @ Q) @ Q.T
    residual = M - proj
    r2 = 1.0 - np.sum(residual ** 2) / total_var
    return max(0.0, float(r2))


def residualize_object_identity(dh, object_labels):
    """Remove object identity component from dh."""
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    unique_objs = sorted(set(object_labels))
    obj_onehot = np.zeros((n, len(unique_objs)))
    for i, obj in enumerate(unique_objs):
        obj_onehot[:, i] = [1.0 if o == obj else 0.0 for o in object_labels]
    if obj_onehot.shape[1] > 1:
        X_obj = obj_onehot[:, :-1]
    else:
        X_obj = obj_onehot
    X_design = np.column_stack([np.ones(n), X_obj])
    try:
        beta = np.linalg.lstsq(X_design, M, rcond=None)[0]
        predicted = X_design @ beta
        residual = M - predicted
    except:
        residual = M
    return residual


# ===== Part 2: Probe-based causal validation with scaling =====
def run_model_with_patch(model, tokenizer, device, prompt, layer_idx,
                          patch_delta, target_token_id, competitor_token_id):
    """Run model with delta added to residual at layer l (last token position)."""
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


def scaled_causal_test(model, tokenizer, device, model_name, target_layers,
                        probe_directions, dh_resid_dict, n_test=80):
    """
    Causal test with scaled patch intensities.
    
    For each test pair:
    1. Get the category direction from the linear probe
    2. Add/remove at multiple scale factors (0.1x, 0.3x, 0.5x, 1.0x, 2.0x)
    3. Observe whether the causal effect is:
       a. Monotonically increasing with scale (supports real causal effect)
       b. Always negative regardless of direction (supports perturbation artifact)
       c. No consistent pattern (supports noise)
    """
    n_all = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_all)]
    
    np.random.seed(42)
    test_indices = sorted(np.random.choice(n_all, min(n_test, n_all), replace=False).tolist())
    
    scale_factors = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    results = {}
    
    for l in target_layers:
        log(f"  Layer {l}: Scaled causal test...")
        t_l = time.time()
        
        W_probe = probe_directions[l]  # (n_classes, d)
        dh_resid = dh_resid_dict[l]
        
        # Build orthonormal probe subspace
        Q_probe, _ = np.linalg.qr(W_probe.T)  # (d, n_dirs)
        
        # Project each sample's residualized dh onto probe subspace
        cat_projections = (dh_resid @ Q_probe) @ Q_probe.T  # (n, d)
        
        # Get per-sample category direction
        # Use the projection onto the category-specific direction from probe
        label_to_idx = {cat: i for i, cat in enumerate(sorted(set(category_labels)))}
        
        # Also compute random control directions (same dimensionality, random orientation)
        n_dirs = Q_probe.shape[1]
        d_dim = Q_probe.shape[0]
        np.random.seed(123)
        Q_random, _ = np.linalg.qr(np.random.randn(d_dim, n_dirs))
        rand_projections = (dh_resid @ Q_random) @ Q_random.T  # (n, d)
        
        ca_data = {
            "probe": {f"scale_{s}": {"add": [], "remove": []} for s in scale_factors},
            "random": {f"scale_{s}": {"add": [], "remove": []} for s in scale_factors},
            "probe_swap": {"cross": [], "same": []},
        }
        baselines = {"clean_ld": [], "corrupt_ld": []}
        
        np.random.seed(456)
        
        for cnt, pidx in enumerate(test_indices):
            if cnt % 20 == 0:
                log(f"    Test pair {cnt+1}/{len(test_indices)}")
            
            obj, target, competitor = ALL_PAIRS[pidx]
            cat_i = category_labels[pidx]
            
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
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
            
            # Get category projection for this sample
            cat_proj_i = cat_projections[pidx]
            rand_proj_i = rand_projections[pidx]
            
            # Scaled add/remove tests
            for scale in scale_factors:
                # Probe direction: add to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                           cat_proj_i * scale, t_id, c_id)
                if ld is not None:
                    ca_data["probe"][f"scale_{scale}"]["add"].append(ld)
                
                # Probe direction: remove from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           -cat_proj_i * scale, t_id, c_id)
                if ld is not None:
                    ca_data["probe"][f"scale_{scale}"]["remove"].append(ld)
                
                # Random direction: add to corrupt (control)
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                           rand_proj_i * scale, t_id, c_id)
                if ld is not None:
                    ca_data["random"][f"scale_{scale}"]["add"].append(ld)
                
                # Random direction: remove from clean (control)
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           -rand_proj_i * scale, t_id, c_id)
                if ld is not None:
                    ca_data["random"][f"scale_{scale}"]["remove"].append(ld)
            
            # Swap tests (only at scale 1.0)
            diff_cat_idx = [j for j in range(n_all) if category_labels[j] != cat_i]
            same_cat_idx = [j for j in range(n_all) if category_labels[j] == cat_i and j != pidx]
            
            if len(diff_cat_idx) > 0:
                j_cross = diff_cat_idx[np.random.randint(len(diff_cat_idx))]
                swap_delta = cat_projections[j_cross] - cat_proj_i
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    ca_data["probe_swap"]["cross"].append(ld)
            
            if len(same_cat_idx) > 0:
                j_same = same_cat_idx[np.random.randint(len(same_cat_idx))]
                swap_delta = cat_projections[j_same] - cat_proj_i
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    ca_data["probe_swap"]["same"].append(ld)
            
            if cnt % 5 == 0:
                torch.cuda.empty_cache()
        
        # Compute effects for each scale
        layer_result = {"layer": l, "n_test": len(test_indices), "scale_factors": scale_factors}
        
        for direction_type in ["probe", "random"]:
            for scale in scale_factors:
                key = f"scale_{scale}"
                
                # Add effect
                n_eff = min(len(ca_data[direction_type][key]["add"]),
                           len(baselines["corrupt_ld"]))
                if n_eff > 0:
                    add_eff = np.array(ca_data[direction_type][key]["add"][:n_eff]) - \
                              np.array(baselines["corrupt_ld"][:n_eff])
                    layer_result[f"{direction_type}_{key}_add"] = {
                        "mean": float(np.mean(add_eff)),
                        "std": float(np.std(add_eff)),
                        "t": float(np.mean(add_eff) / (np.std(add_eff) / np.sqrt(n_eff) + 1e-10)),
                        "n": n_eff,
                    }
                
                # Remove effect
                n_eff2 = min(len(ca_data[direction_type][key]["remove"]),
                            len(baselines["clean_ld"]))
                if n_eff2 > 0:
                    rem_eff = np.array(ca_data[direction_type][key]["remove"][:n_eff2]) - \
                              np.array(baselines["clean_ld"][:n_eff2])
                    layer_result[f"{direction_type}_{key}_remove"] = {
                        "mean": float(np.mean(rem_eff)),
                        "std": float(np.std(rem_eff)),
                        "t": float(np.mean(rem_eff) / (np.std(rem_eff) / np.sqrt(n_eff2) + 1e-10)),
                        "n": n_eff2,
                    }
        
        # Swap effect
        n_swap = min(len(ca_data["probe_swap"]["cross"]),
                     len(ca_data["probe_swap"]["same"]),
                     len(baselines["clean_ld"]))
        if n_swap > 0:
            cross_eff = np.array(ca_data["probe_swap"]["cross"][:n_swap]) - \
                        np.array(baselines["clean_ld"][:n_swap])
            same_eff = np.array(ca_data["probe_swap"]["same"][:n_swap]) - \
                       np.array(baselines["clean_ld"][:n_swap])
            diff = cross_eff - same_eff
            layer_result["probe_swap"] = {
                "cross_mean": float(np.mean(cross_eff)),
                "same_mean": float(np.mean(same_eff)),
                "diff": float(np.mean(diff)),
                "diff_std": float(np.std(diff)),
                "diff_t": float(np.mean(diff) / (np.std(diff) / np.sqrt(n_swap) + 1e-10)),
                "n": n_swap,
            }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
        
        # Print summary for this layer
        for direction_type in ["probe", "random"]:
            log(f"  {direction_type} direction causal effects:")
            for scale in scale_factors:
                key = f"scale_{scale}"
                ae = layer_result.get(f"{direction_type}_{key}_add", {})
                re = layer_result.get(f"{direction_type}_{key}_remove", {})
                log(f"    scale={scale:.1f}: add_mean={ae.get('mean',0):+.4f} t={ae.get('t',0):.2f}, "
                    f"rem_mean={re.get('mean',0):+.4f} t={re.get('t',0):.2f}")
        
        se = layer_result.get("probe_swap", {})
        if se:
            log(f"  swap: diff={se.get('diff',0):+.4f} t={se.get('diff_t',0):.2f}")
    
    return results


# ===== Part 3: Cross-layer category transfer tracking =====
def cross_layer_transfer(probe_directions_dict, target_layers, category_labels):
    """
    Track how category probe directions transfer across layers.
    
    If category information is genuinely transmitted through layers,
    the probe directions at adjacent layers should be more similar
    than random (higher cosine similarity).
    """
    results = {}
    
    layer_list = sorted(probe_directions_dict.keys())
    
    for i, l1 in enumerate(layer_list):
        for l2 in layer_list[i+1:]:
            W1 = probe_directions_dict[l1]  # (n_classes, d)
            W2 = probe_directions_dict[l2]  # (n_classes, d)
            
            # Compute subspace similarity via canonical correlation
            Q1, _ = np.linalg.qr(W1.T)  # (d, n_dirs1)
            Q2, _ = np.linalg.qr(W2.T)  # (d, n_dirs2)
            
            # Subspace cosine similarity: ||Q1^T Q2||_F / sqrt(n_dirs1 * n_dirs2)
            M_sub = Q1.T @ Q2
            subspace_sim = float(np.linalg.norm(M_sub, 'fro') / 
                                np.sqrt(min(Q1.shape[1], Q2.shape[1])))
            
            # Per-category direction similarity
            cat_sims = {}
            unique_cats = sorted(set(category_labels))
            for c_idx, cat in enumerate(unique_cats):
                if c_idx < W1.shape[0] and c_idx < W2.shape[0]:
                    sim = float(cosine_sim(W1[c_idx], W2[c_idx]))
                    cat_sims[cat] = sim
            
            # Random baseline: compare W1 to shuffled W2
            n_perm = 200
            rng = np.random.RandomState(42)
            perm_sims = []
            for _ in range(n_perm):
                perm_idx = rng.permutation(W2.shape[1])
                W2_perm = W2[:, perm_idx]
                Q2_perm, _ = np.linalg.qr(W2_perm.T)
                M_perm = Q1.T @ Q2_perm
                perm_sims.append(float(np.linalg.norm(M_perm, 'fro') / 
                                      np.sqrt(min(Q1.shape[1], Q2_perm.shape[1]))))
            
            perm_mean = float(np.mean(perm_sims))
            perm_std = float(np.std(perm_sims))
            z_score = (subspace_sim - perm_mean) / (perm_std + 1e-10)
            
            results[f"L{l1}_L{l2}"] = {
                "subspace_sim": subspace_sim,
                "perm_mean": perm_mean,
                "perm_std": perm_std,
                "z_score": z_score,
                "cat_sims": cat_sims,
            }
            
            log(f"  L{l1}→L{l2}: subspace_sim={subspace_sim:.4f}, "
                f"perm={perm_mean:.4f}±{perm_std:.4f}, z={z_score:.2f}")
    
    return results


# ===== Part 4: Counterfactual validation (real forward pass) =====
def counterfactual_validation(model, tokenizer, device, model_name, target_layers,
                               probe_directions, dh_resid_dict, n_test=40):
    """
    Counterfactual test: patch at layer l, observe effect on FINAL output (not logit lens).
    This is the gold standard for causality.
    
    Compare with logit lens prediction at the same layer.
    """
    n_all = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_all)]
    
    np.random.seed(789)
    test_indices = sorted(np.random.choice(n_all, min(n_test, n_all), replace=False).tolist())
    
    results = {}
    
    for l in target_layers:
        log(f"  Layer {l}: Counterfactual validation...")
        t_l = time.time()
        
        W_probe = probe_directions[l]
        dh_resid = dh_resid_dict[l]
        Q_probe, _ = np.linalg.qr(W_probe.T)
        cat_projections = (dh_resid @ Q_probe) @ Q_probe.T
        
        ca_data = {
            "forward_add": [],
            "forward_remove": [],
            "logit_lens_add": [],
            "logit_lens_remove": [],
        }
        baselines = {"clean_ld": [], "corrupt_ld": []}
        
        ln_weight = _load_ln_weight(model, model_name, l)
        W_U = get_W_U(model, model_name)
        
        np.random.seed(321)
        
        for cnt, pidx in enumerate(test_indices):
            if cnt % 10 == 0:
                log(f"    Counterfactual pair {cnt+1}/{len(test_indices)}")
            
            obj, target, competitor = ALL_PAIRS[pidx]
            
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            if t_id < 0 or c_id < 0:
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # Clean baseline (real forward)
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
            
            # Corrupt baseline (real forward)
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
            
            cat_proj_i = cat_projections[pidx]
            
            # === Forward add: add category to corrupt, observe final output ===
            ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                       cat_proj_i, t_id, c_id)
            if ld is not None:
                ca_data["forward_add"].append(ld)
                
                # Logit lens prediction at the same layer
                # Get h_corrupt for logit lens
                with torch.no_grad():
                    corrupt_h_out = model(
                        input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                           truncation=True, max_length=64)["input_ids"].to(device),
                        attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                                truncation=True, max_length=64)["attention_mask"].to(device),
                        output_hidden_states=True,
                    )
                last_pos = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
                h_corrupt_l = corrupt_h_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
                
                # Add category in post-RMSNorm space (logit lens)
                h_corrupt_norm = rms_norm_single(h_corrupt_l, ln_weight)
                h_patched_norm = h_corrupt_norm + cat_proj_i
                logit_lens_pred = W_U @ h_patched_norm
                ca_data["logit_lens_add"].append(float(logit_lens_pred[t_id] - logit_lens_pred[c_id]))
                del corrupt_h_out
            
            # === Forward remove: remove category from clean, observe final output ===
            ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                       -cat_proj_i, t_id, c_id)
            if ld is not None:
                ca_data["forward_remove"].append(ld)
            
            if cnt % 5 == 0:
                torch.cuda.empty_cache()
        
        # Compute effects
        n_base = len(baselines["clean_ld"])
        layer_result = {"layer": l, "n_test": len(test_indices)}
        
        # Forward add effect
        n_eff = min(len(ca_data["forward_add"]), len(baselines["corrupt_ld"]))
        if n_eff > 0:
            fwd_add = np.array(ca_data["forward_add"][:n_eff]) - np.array(baselines["corrupt_ld"][:n_eff])
            layer_result["forward_add"] = {
                "mean": float(np.mean(fwd_add)),
                "std": float(np.std(fwd_add)),
                "t": float(np.mean(fwd_add) / (np.std(fwd_add) / np.sqrt(n_eff) + 1e-10)),
                "n": n_eff,
            }
        
        # Forward remove effect
        n_eff2 = min(len(ca_data["forward_remove"]), n_base)
        if n_eff2 > 0:
            fwd_rem = np.array(ca_data["forward_remove"][:n_eff2]) - np.array(baselines["clean_ld"][:n_eff2])
            layer_result["forward_remove"] = {
                "mean": float(np.mean(fwd_rem)),
                "std": float(np.std(fwd_rem)),
                "t": float(np.mean(fwd_rem) / (np.std(fwd_rem) / np.sqrt(n_eff2) + 1e-10)),
                "n": n_eff2,
            }
        
        # Logit lens add effect
        n_eff3 = min(len(ca_data["logit_lens_add"]), len(baselines["corrupt_ld"]))
        if n_eff3 > 0:
            ll_add = np.array(ca_data["logit_lens_add"][:n_eff3]) - np.array(baselines["corrupt_ld"][:n_eff3])
            layer_result["logit_lens_add"] = {
                "mean": float(np.mean(ll_add)),
                "std": float(np.std(ll_add)),
                "t": float(np.mean(ll_add) / (np.std(ll_add) / np.sqrt(n_eff3) + 1e-10)),
                "n": n_eff3,
            }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
        
        # Print summary
        fa = layer_result.get("forward_add", {})
        fr = layer_result.get("forward_remove", {})
        la = layer_result.get("logit_lens_add", {})
        log(f"  Forward add:  mean={fa.get('mean',0):+.4f} t={fa.get('t',0):.2f}")
        log(f"  Forward rem:  mean={fr.get('mean',0):+.4f} t={fr.get('t',0):.2f}")
        log(f"  LogitLens add: mean={la.get('mean',0):+.4f} t={la.get('t',0):.2f}")
    
    return results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 385: Linear Probe Causal + Cross-Layer Transfer — {model_name}")
    log(f"=" * 60)
    
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
    device = next(model.parameters()).device
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    
    # ===== Collect all residual states =====
    log("\n=== Collecting residual states ===")
    
    dh_proper_dict = {}
    dh_resid_dict = {}
    probe_directions = {}
    probe_results = {}
    
    for l in target_layers:
        log(f"  Layer {l}: collecting data...")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        h_clean_all = []
        h_corrupt_all = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs} (layer {l})")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_clean_all.append(clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy())
            del clean_out
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            
            last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_corrupt_all.append(corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_clean_all = np.array(h_clean_all)
        h_corrupt_all = np.array(h_corrupt_all)
        
        # Compute post-RMSNorm difference
        h_clean_norm = np.zeros_like(h_clean_all)
        h_corrupt_norm = np.zeros_like(h_corrupt_all)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean_all[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt_all[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        dh_proper_dict[l] = dh_proper
        
        # Residualize object identity
        dh_resid = residualize_object_identity(dh_proper, object_labels)
        dh_resid_dict[l] = dh_resid
        
        # ===== Part 1: Train linear probes =====
        log(f"  Layer {l}: Training linear probes...")
        
        # Train on dh_resid (clean)
        W_logistic, b_logistic, clf_logistic, cv_acc_logistic = train_linear_probe(
            dh_resid, category_labels, method='logistic')
        
        # Also train on raw dh_proper for comparison
        W_logistic_raw, b_logistic_raw, clf_logistic_raw, cv_acc_raw = train_linear_probe(
            dh_proper, category_labels, method='logistic')
        
        probe_directions[l] = W_logistic
        
        # Compute R²
        r2_probe_resid = compute_probe_r2(dh_resid, category_labels, W_logistic)
        r2_probe_raw = compute_probe_r2(dh_proper, category_labels, W_logistic_raw)
        
        # Also compute centroid-based R² for comparison
        unique_labels = sorted(set(category_labels))
        cat_centroids = {}
        for cat in unique_labels:
            idx = [i for i, c in enumerate(category_labels) if c == cat]
            cat_centroids[cat] = np.mean(dh_resid[idx], axis=0)
        overall = np.mean(dh_resid, axis=0)
        W_centroid = np.array([cat_centroids[cat] - overall for cat in unique_labels])
        r2_centroid = compute_probe_r2(dh_resid, category_labels, W_centroid)
        
        probe_results[str(l)] = {
            "cv_acc_resid": cv_acc_logistic,
            "cv_acc_raw": cv_acc_raw,
            "r2_probe_resid": r2_probe_resid,
            "r2_probe_raw": r2_probe_raw,
            "r2_centroid_resid": r2_centroid,
            "n_probe_dirs": int(W_logistic.shape[0]),
        }
        
        log(f"    CV accuracy: resid={cv_acc_logistic:.4f}, raw={cv_acc_raw:.4f}")
        log(f"    Probe R²:    resid={r2_probe_resid:.4f}, raw={r2_probe_raw:.4f}")
        log(f"    Centroid R²: resid={r2_centroid:.4f}")
        log(f"    Layer {l} data collection done in {time.time()-t_l:.1f}s")
    
    # ===== Part 2: Scaled causal test =====
    log("\n=== Part 2: Scaled causal test ===")
    causal_results = scaled_causal_test(
        model, tokenizer, device, model_name, target_layers,
        probe_directions, dh_resid_dict, n_test=80)
    
    # ===== Part 3: Cross-layer transfer =====
    log("\n=== Part 3: Cross-layer transfer tracking ===")
    transfer_results = cross_layer_transfer(
        probe_directions, target_layers, category_labels)
    
    # ===== Part 4: Counterfactual validation =====
    log("\n=== Part 4: Counterfactual validation ===")
    counterfactual_results = counterfactual_validation(
        model, tokenizer, device, model_name, target_layers,
        probe_directions, dh_resid_dict, n_test=40)
    
    # ===== Save results =====
    out_dir = "results/phase385_linear_probe_causal"
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
        "test": "phase385_linear_probe_causal",
        "probe_results": convert(probe_results),
        "causal_results": convert(causal_results),
        "transfer_results": convert(transfer_results),
        "counterfactual_results": convert(counterfactual_results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase385.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # ===== Print summary =====
    log(f"\n{'='*60}")
    log(f"Phase 385 Summary — {model_name}")
    log(f"{'='*60}")
    
    log("\n--- Part 1: Linear Probe Quality ---")
    for l_str in sorted(probe_results.keys(), key=int):
        r = probe_results[l_str]
        log(f"  L{l_str}: CV_acc(resid)={r['cv_acc_resid']:.4f}, "
            f"R²_probe(resid)={r['r2_probe_resid']:.4f}, "
            f"R²_centroid(resid)={r['r2_centroid_resid']:.4f}")
    
    log("\n--- Part 2: Scaled Causal Effects ---")
    for l_str in sorted(causal_results.keys(), key=int):
        r = causal_results[l_str]
        log(f"  L{l_str}:")
        for scale in [0.1, 0.5, 1.0, 2.0]:
            key = f"scale_{scale}"
            pa = r.get(f"probe_{key}_add", {})
            ra = r.get(f"random_{key}_add", {})
            log(f"    scale={scale}: probe_add={pa.get('mean',0):+.4f}(t={pa.get('t',0):.2f}), "
                f"random_add={ra.get('mean',0):+.4f}(t={ra.get('t',0):.2f})")
    
    log("\n--- Part 3: Cross-Layer Transfer ---")
    for key in sorted(transfer_results.keys()):
        r = transfer_results[key]
        log(f"  {key}: sim={r['subspace_sim']:.4f}, z={r['z_score']:.2f}")
    
    log("\n--- Part 4: Counterfactual vs Logit Lens ---")
    for l_str in sorted(counterfactual_results.keys(), key=int):
        r = counterfactual_results[l_str]
        fa = r.get("forward_add", {})
        la = r.get("logit_lens_add", {})
        fr = r.get("forward_remove", {})
        log(f"  L{l_str}: fwd_add={fa.get('mean',0):+.4f}(t={fa.get('t',0):.2f}), "
            f"fwd_rem={fr.get('mean',0):+.4f}(t={fr.get('t',0):.2f}), "
            f"ll_add={la.get('mean',0):+.4f}(t={la.get('t',0):.2f})")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 385 complete for {model_name}!")


if __name__ == "__main__":
    main()
