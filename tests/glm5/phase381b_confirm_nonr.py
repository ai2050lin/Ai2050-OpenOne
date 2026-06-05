"""
Phase 381b: 确认测试 — DS7B NoNR准确率大幅提升的验证
=====================================================

Phase 381的核心发现：DS7B移除norm_ratio回归后centroid分类准确率从39%→80%。
这需要确认：

1. PC1移除 vs norm_ratio回归 vs PC1-PC3移除的效果
2. 不同分类器（centroid, KNN, 线性SVM）的交叉验证
3. 增加数据量验证

重点模型：deepseek7b（其他模型作为参照）

用法:
  python tests/glm5/phase381b_confirm_nonr.py deepseek7b
  python tests/glm5/phase381b_confirm_nonr.py qwen3
  python tests/glm5/phase381b_confirm_nonr.py glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS
# Reuse pairs from phase381
from phase381_norm_matched_category_test import (
    ALL_PAIRS, PAIR_CATEGORIES, ALL_CATEGORIES, N_CATEGORIES,
    CORRUPTED_BASELINE, TEMPLATE, rms_norm_single, cosine_sim,
    load_model_bf16, load_mlp_weights, _load_ln_weight,
)


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


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


def loo_knn_accuracy(X, labels, k=5):
    """Leave-one-out KNN classification."""
    n = X.shape[0]
    unique_labels = sorted(set(labels))
    correct = 0
    for i in range(n):
        dists = []
        for j in range(n):
            if j == i:
                continue
            d = np.linalg.norm(X[i] - X[j])
            dists.append((d, labels[j]))
        dists.sort()
        top_k = [d[1] for d in dists[:k]]
        votes = {}
        for lab in top_k:
            votes[lab] = votes.get(lab, 0) + 1
        pred = max(votes, key=votes.get)
        if pred == labels[i]:
            correct += 1
    return correct / n


def collect_residual_states(model, tokenizer, model_name, target_layers):
    """Reuse data collection from Phase 381."""
    act_fn = "gelu" if model_name == "glm4" else "silu"
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    
    all_data = {}
    for l in target_layers:
        log(f"  Collecting Layer {l}...")
        t_l = time.time()
        
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None:
            log(f"    SKIP: Could not load MLP weights for layer {l}")
            continue
        
        mlp_module = layers[l].mlp
        ln_weight = _load_ln_weight(model, model_name, l)
        
        h_post_clean_list = []
        h_post_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            captured = {}
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            h_hook = mlp_module.register_forward_pre_hook(mlp_input_hook)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_post_clean_list.append(clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy())
            del clean_out
            captured.clear()
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_post_corrupt_list.append(corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            del corrupt_out
            h_hook.remove()
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        all_data[str(l)] = {
            "h_post_clean": np.array(h_post_clean_list),
            "h_post_corrupt": np.array(h_post_corrupt_list),
            "ln_weight": ln_weight,
        }
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


def run_confirm_tests(all_data, model_name):
    """Run multiple classification tests to confirm the NoNR result."""
    log("\n" + "="*60)
    log("Confirmatory Tests: NoNR Accuracy Boost")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        n_pairs = d["h_post_clean"].shape[0]
        
        h_clean = d["h_post_clean"]
        h_corrupt = d["h_post_corrupt"]
        
        # PROPER post-RMSNorm
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
        
        labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
        
        # PCA
        M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            continue
        pc1_dir = Vt[0]  # (d,)
        pc2_dir = Vt[1]
        
        # === Test 1: Remove PC1, PC1-3, PC1-5 ===
        # Project out PC1: dh_no_pc1 = dh - (dh @ pc1) * pc1
        proj_pc1 = (M @ pc1_dir)[:, None] * pc1_dir[None, :]
        dh_no_pc1 = dh_proper - proj_pc1
        
        proj_pc13 = sum((M @ Vt[k])[:, None] * Vt[k][None, :] for k in range(3))
        dh_no_pc13 = dh_proper - proj_pc13
        
        proj_pc15 = sum((M @ Vt[k])[:, None] * Vt[k][None, :] for k in range(5))
        dh_no_pc15 = dh_proper - proj_pc15
        
        # Regress out norm_ratio
        nr_centered = (norm_ratio - norm_ratio.mean())[:, None]
        beta = np.linalg.lstsq(nr_centered, dh_proper, rcond=None)[0]
        dh_no_nr = dh_proper - nr_centered @ beta
        
        # Regress out norm_diff
        norm_diff = norm_clean - norm_corrupt
        nd_centered = (norm_diff - norm_diff.mean())[:, None]
        beta_nd = np.linalg.lstsq(nd_centered, dh_proper, rcond=None)[0]
        dh_no_nd = dh_proper - nd_centered @ beta_nd
        
        # === Classification with centroid ===
        variants = {
            "original": dh_proper,
            "no_pc1": dh_no_pc1,
            "no_pc13": dh_no_pc13,
            "no_pc15": dh_no_pc15,
            "no_norm_ratio": dh_no_nr,
            "no_norm_diff": dh_no_nd,
        }
        
        variant_accs = {}
        for vname, dh_v in variants.items():
            # Centroid direction classification (5d)
            cat_centroids = {}
            for cat in ALL_CATEGORIES:
                idx = [j for j, c in enumerate(labels) if c == cat]
                cat_centroids[cat] = np.mean(dh_v[idx], axis=0)
            overall = np.mean(dh_v, axis=0)
            cd = np.array([cat_centroids[cat] - overall for cat in ALL_CATEGORIES])
            Q, _ = np.linalg.qr(cd.T)
            dh_proj = dh_v @ Q[:, :5]
            acc_c = loo_centroid_accuracy(dh_proj, labels)
            
            # KNN-5 classification (5d projection)
            acc_k = loo_knn_accuracy(dh_proj, labels, k=5)
            
            # Centroid classification (10d)
            dh_proj10 = dh_v @ Q[:, :10]
            acc_c10 = loo_centroid_accuracy(dh_proj10, labels)
            
            variant_accs[vname] = {
                "centroid_5d": float(acc_c),
                "centroid_10d": float(acc_c10),
                "knn5_5d": float(acc_k),
            }
        
        # === Test 2: PC1 score correlation with norm_ratio ===
        pc1_scores = U[:, 0] * S[0]
        pc1_nr_corr = cosine_sim(norm_ratio - norm_ratio.mean(), pc1_scores)
        
        # === Test 3: Per-PC correlation with categories (one-hot) ===
        pc_cat_corr = {}
        for k in range(min(10, S.shape[0])):
            pc_scores = U[:, k] * S[k]
            cat_corrs = {}
            for cat in ALL_CATEGORIES:
                binary = np.array([1 if c == cat else 0 for c in labels], dtype=float)
                corr = np.corrcoef(pc_scores, binary)[0, 1]
                cat_corrs[cat] = float(corr)
            max_abs_corr = max(abs(v) for v in cat_corrs.values())
            pc_cat_corr[f"PC{k+1}"] = {"max_abs_cat_corr": float(max_abs_corr), "details": cat_corrs}
        
        # === Test 4: PCA on no_pc1 data - what becomes the new PC1? ===
        M_no1 = dh_no_pc1 - dh_no_pc1.mean(axis=0, keepdims=True)
        try:
            U2, S2, Vt2 = np.linalg.svd(M_no1, full_matrices=False)
            new_pc1_dir = Vt2[0]
            # Correlate new PC1 with original PC2
            new_pc1_scores = U2[:, 0] * S2[0]
            orig_pc2_scores = U[:, 1] * S[1]
            new_pc1_orig_pc2_corr = cosine_sim(new_pc1_scores, orig_pc2_scores)
            new_pc1_nr_corr = cosine_sim(new_pc1_scores, norm_ratio - norm_ratio.mean())
            
            # New PC1 category correlation
            max_cat_corr_new_pc1 = 0
            for cat in ALL_CATEGORIES:
                binary = np.array([1 if c == cat else 0 for c in labels], dtype=float)
                corr = abs(np.corrcoef(new_pc1_scores, binary)[0, 1])
                max_cat_corr_new_pc1 = max(max_cat_corr_new_pc1, corr)
        except:
            new_pc1_orig_pc2_corr = 0
            new_pc1_nr_corr = 0
            max_cat_corr_new_pc1 = 0
        
        res = {
            "layer": l,
            "n_pairs": n_pairs,
            "pc1_norm_ratio_corr": float(pc1_nr_corr),
            "classification_variants": variant_accs,
            "pc_category_correlations": pc_cat_corr,
            "after_pc1_removal": {
                "new_pc1_vs_orig_pc2": float(new_pc1_orig_pc2_corr),
                "new_pc1_vs_norm_ratio": float(new_pc1_nr_corr),
                "new_pc1_max_cat_corr": float(max_cat_corr_new_pc1),
            }
        }
        
        results[l_str] = res
        
        log(f"\n  Layer {l}:")
        log(f"    PC1~norm_ratio: {pc1_nr_corr:.3f}")
        log(f"    Classification (centroid 5d):")
        for vname, accs in variant_accs.items():
            log(f"      {vname:20s}: centroid={accs['centroid_5d']:.3f}, knn5={accs['knn5_5d']:.3f}, centroid10d={accs['centroid_10d']:.3f}")
        log(f"    After PC1 removal:")
        log(f"      new PC1 vs orig PC2: {new_pc1_orig_pc2_corr:.3f}")
        log(f"      new PC1 vs norm_ratio: {new_pc1_nr_corr:.3f}")
        log(f"      new PC1 max cat corr: {max_cat_corr_new_pc1:.3f}")
        # Show top PC category correlations
        for k in range(min(5, len(pc_cat_corr))):
            pc_key = f"PC{k+1}"
            log(f"      {pc_key} max|cat_corr|={pc_cat_corr[pc_key]['max_abs_cat_corr']:.3f}")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 381b: NoNR Accuracy Boost Confirmation — {model_name}")
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
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers")
    
    # Collect data
    log("\nCollecting residual states...")
    all_data = collect_residual_states(model, tokenizer, model_name, target_layers)
    
    # Run confirmation tests
    log("\nRunning confirmation tests...")
    results = run_confirm_tests(all_data, model_name)
    
    # Save
    out_dir = "results/phase381_norm_matched_category"
    os.makedirs(out_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    full_results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_pairs": len(ALL_PAIRS),
        "test": "phase381b_confirm_nonr",
        "results": {k: convert(v) for k, v in results.items()},
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase381b.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 381b complete for {model_name}!")


if __name__ == "__main__":
    main()
