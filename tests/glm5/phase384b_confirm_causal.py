"""
Phase 384b: 确认测试 — 更大样本量的净化类别因果测试
===================================================

Phase 384发现：
- Category unique R²只有0.7-2.5%
- Clean category的remove方向全部正确
- Clean category的因果效应极弱（mean~0.01-0.03）

本确认测试：
- 样本量增加到120对
- 只测试clean (residualized) category subspace
- 增加LDA方向作为对比（LDA最大化类间/类内比，可能更纯）
- 重点验证：DS7B和GLM4的深层因果方向

用法:
  python tests/glm5/phase384b_confirm_causal.py qwen3
  python tests/glm5/phase384b_confirm_causal.py deepseek7b
  python tests/glm5/phase384b_confirm_causal.py glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS
from phase381_norm_matched_category_test import (
    ALL_PAIRS, PAIR_CATEGORIES, ALL_CATEGORIES, N_CATEGORIES,
    CORRUPTED_BASELINE, TEMPLATE, rms_norm_single,
    load_model_bf16, _load_ln_weight,
)


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


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


def extract_lda_directions(dh, category_labels, n_dirs=6):
    """
    Extract LDA directions that maximize between-class / within-class variance ratio.
    This gives the most discriminative directions for category separation.
    """
    unique_labels = sorted(set(category_labels))
    n_classes = len(unique_labels)
    n, d = dh.shape
    M = dh - dh.mean(axis=0, keepdims=True)
    
    # Compute class means and scatter matrices
    class_means = {}
    class_counts = {}
    for cat in unique_labels:
        idx = [i for i, c in enumerate(category_labels) if c == cat]
        class_means[cat] = np.mean(M[idx], axis=0)
        class_counts[cat] = len(idx)
    
    # Within-class scatter
    S_w = np.zeros((d, d))
    for cat in unique_labels:
        idx = [i for i, c in enumerate(category_labels) if c == cat]
        for i in idx:
            diff = (M[i] - class_means[cat])[:, None]
            S_w += diff @ diff.T
    
    # Between-class scatter
    grand_mean = np.mean(M, axis=0)
    S_b = np.zeros((d, d))
    for cat in unique_labels:
        diff = (class_means[cat] - grand_mean)[:, None]
        S_b += class_counts[cat] * diff @ diff.T
    
    # Regularize S_w for numerical stability
    S_w_reg = S_w + 1e-6 * np.eye(d) * np.trace(S_w) / d
    
    # Solve generalized eigenvalue problem: S_b w = lambda S_w w
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(S_b, S_w_reg)
        # Sort by eigenvalue (descending)
        idx_sort = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx_sort]
        n_dirs = min(n_dirs, n_classes - 1, d)
        Q_lda = eigenvectors[:, :n_dirs]
    except:
        # Fallback: use centroid-based directions
        Q_lda, _ = extract_category_subspace(dh, category_labels)
    
    return Q_lda


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


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 384b: Confirm Clean Category Causal (n=120) — {model_name}")
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
    
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    
    all_results = {}
    
    for l in target_layers:
        log(f"\n--- Layer {l} ---")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # Collect ALL residual states
        log(f"  Collecting residual states for all {n_pairs} pairs...")
        h_clean_all = []
        h_corrupt_all = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                    output_hidden_states=True)
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_clean_all.append(clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy())
            del clean_out
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                    output_hidden_states=True)
            
            last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_corrupt_all.append(corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_clean_all = np.array(h_clean_all)
        h_corrupt_all = np.array(h_corrupt_all)
        
        # Compute post-RMSNorm
        h_clean_norm = np.zeros_like(h_clean_all)
        h_corrupt_norm = np.zeros_like(h_corrupt_all)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean_all[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt_all[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Residualize object identity
        dh_resid = residualize_object_identity(dh_proper, object_labels)
        
        # Extract category subspaces
        Q_centroid, _, _ = extract_category_subspace(dh_resid, category_labels)
        Q_lda = extract_lda_directions(dh_resid, category_labels, n_dirs=N_CATEGORIES-1)
        
        # Project
        cat_proj_centroid = (dh_resid @ Q_centroid) @ Q_centroid.T
        cat_proj_lda = (dh_resid @ Q_lda) @ Q_lda.T
        
        # R²
        r2_resid_centroid = compute_category_r2(dh_resid, category_labels)
        log(f"  Category R² (residualized): {r2_resid_centroid:.4f}")
        
        # === Causal test with 120 pairs ===
        n_test = min(120, n_pairs)
        np.random.seed(999)
        test_indices = sorted(np.random.choice(n_pairs, n_test, replace=False).tolist())
        
        log(f"  Running causal test on {n_test} pairs...")
        
        ca_data = {
            "centroid": {"add": [], "remove": [], "cross_swap": [], "same_swap": []},
            "lda": {"add": [], "remove": [], "cross_swap": [], "same_swap": []},
        }
        baselines = {"clean_ld": [], "corrupt_ld": []}
        
        np.random.seed(888)
        
        for cnt, pidx in enumerate(test_indices):
            if cnt % 20 == 0:
                log(f"    Test pair {cnt+1}/{n_test}")
            
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
            
            # Test both centroid and LDA projections
            for method, cat_proj_sub in [("centroid", cat_proj_centroid), ("lda", cat_proj_lda)]:
                cat_proj_i = cat_proj_sub[pidx]
                
                # Add category to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                           cat_proj_i, t_id, c_id)
                if ld is not None:
                    ca_data[method]["add"].append(ld)
                
                # Remove category from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           -cat_proj_i, t_id, c_id)
                if ld is not None:
                    ca_data[method]["remove"].append(ld)
                
                # Cross-category swap
                diff_cat_idx = [j for j in range(n_pairs) if category_labels[j] != cat_i]
                if len(diff_cat_idx) > 0:
                    j_cross = diff_cat_idx[np.random.randint(len(diff_cat_idx))]
                    swap_delta = cat_proj_sub[j_cross] - cat_proj_i
                    ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                               swap_delta, t_id, c_id)
                    if ld is not None:
                        ca_data[method]["cross_swap"].append(ld)
                
                # Same-category swap
                same_cat_idx = [j for j in range(n_pairs) if category_labels[j] == cat_i and j != pidx]
                if len(same_cat_idx) > 0:
                    j_same = same_cat_idx[np.random.randint(len(same_cat_idx))]
                    swap_delta = cat_proj_sub[j_same] - cat_proj_i
                    ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                               swap_delta, t_id, c_id)
                    if ld is not None:
                        ca_data[method]["same_swap"].append(ld)
            
            if cnt % 5 == 0:
                torch.cuda.empty_cache()
        
        # Compute effects
        layer_result = {"layer": l, "n_test": n_test}
        
        for method in ["centroid", "lda"]:
            d = ca_data[method]
            
            n_eff = min(len(d["add"]), len(baselines["corrupt_ld"]))
            if n_eff > 0:
                add_eff = np.array(d["add"][:n_eff]) - np.array(baselines["corrupt_ld"][:n_eff])
                layer_result[f"{method}_add"] = {
                    "mean": float(np.mean(add_eff)),
                    "std": float(np.std(add_eff)),
                    "t": float(np.mean(add_eff) / (np.std(add_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
            
            n_eff2 = min(len(d["remove"]), len(baselines["clean_ld"]))
            if n_eff2 > 0:
                rem_eff = np.array(d["remove"][:n_eff2]) - np.array(baselines["clean_ld"][:n_eff2])
                layer_result[f"{method}_remove"] = {
                    "mean": float(np.mean(rem_eff)),
                    "std": float(np.std(rem_eff)),
                    "t": float(np.mean(rem_eff) / (np.std(rem_eff) / np.sqrt(n_eff2) + 1e-10)),
                    "n": n_eff2,
                }
            
            n_swap = min(len(d["cross_swap"]), len(d["same_swap"]), len(baselines["clean_ld"]))
            if n_swap > 0:
                cross_eff = np.array(d["cross_swap"][:n_swap]) - np.array(baselines["clean_ld"][:n_swap])
                same_eff = np.array(d["same_swap"][:n_swap]) - np.array(baselines["clean_ld"][:n_swap])
                diff = cross_eff - same_eff
                layer_result[f"{method}_swap"] = {
                    "cross_mean": float(np.mean(cross_eff)),
                    "same_mean": float(np.mean(same_eff)),
                    "diff": float(np.mean(diff)),
                    "diff_std": float(np.std(diff)),
                    "diff_t": float(np.mean(diff) / (np.std(diff) / np.sqrt(n_swap) + 1e-10)),
                    "n": n_swap,
                }
        
        all_results[str(l)] = layer_result
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
        
        # Print results
        for method in ["centroid", "lda"]:
            ae = layer_result.get(f"{method}_add", {})
            re = layer_result.get(f"{method}_remove", {})
            se = layer_result.get(f"{method}_swap", {})
            log(f"  {method:8s} add:  mean={ae.get('mean',0):+.4f} t={ae.get('t',0):.2f} n={ae.get('n',0)}")
            log(f"  {method:8s} rem:  mean={re.get('mean',0):+.4f} t={re.get('t',0):.2f} n={re.get('n',0)}")
            if se:
                log(f"  {method:8s} swap: diff={se.get('diff',0):+.4f} t={se.get('diff_t',0):.2f} n={se.get('n',0)}")
    
    # Save
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
        "n_test_causal": 120,
        "test": "phase384b_confirm_causal",
        "results": convert(all_results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase384b.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 384b complete for {model_name}!")


if __name__ == "__main__":
    main()
