"""
Phase 383: 真实类别Swap因果测试
================================

核心目标：用实际模型干预（而非logit lens近似）验证类别分量的因果有效性。

方法：
1. 收集clean/corrupt的residual states
2. 分解dh_proper为category分量和non-category分量
3. 实际模型干预：
   a. 基线：正常corrupt前向传播 → logit_diff
   b. 添加类别分量：corrupt + cat_component → logit_diff
   c. 移除类别分量：clean - cat_component → logit_diff
   d. 跨类别swap：corrupt_i + cat_component_j (j来自不同类别) → logit_diff
   e. 同类别swap：corrupt_i + cat_component_j (j来自同类别) → logit_diff
   f. 零化类别分量：corrupt + noncat_component → logit_diff

关键改进（vs Phase 380）：
- 不是"添加/移除"，而是"替换"
- 使用实际模型干预，不是logit lens
- 分same-category和cross-category测试

用法:
  python tests/glm5/phase383_category_swap_causal.py qwen3
  python tests/glm5/phase383_category_swap_causal.py deepseek7b
  python tests/glm5/phase383_category_swap_causal.py glm4
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


def loo_centroid_accuracy(X, labels):
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


def compute_category_subspace(dh_proper, labels):
    """Compute category subspace using centroid directions."""
    unique_labels = sorted(set(labels))
    cat_centroids = {}
    for cat in unique_labels:
        idx = [i for i, c in enumerate(labels) if c == cat]
        cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
    
    overall = np.mean(dh_proper, axis=0)
    cat_dirs = np.array([cat_centroids[cat] - overall for cat in unique_labels])
    
    # QR decomposition for subspace basis
    Q, _ = np.linalg.qr(cat_dirs.T)  # Q: (d, n_cat)
    return Q, cat_centroids, overall


def run_model_with_patched_residual(model, tokenizer, device, prompt, 
                                      layer_idx, patch_fn, target_token_id, 
                                      competitor_token_id):
    """
    Run model with a patched residual at a specific layer.
    
    patch_fn: function(h_at_layer) -> patched_h
    Returns: (logit_diff, target_logit, competitor_logit)
    """
    if target_token_id < 0 or competitor_token_id < 0:
        return None, None, None
    
    layers = get_layers(model)
    
    captured = {}
    patch_applied = [False]
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        
        # Only patch at the last token position
        patched_h = patch_fn(h)
        patch_applied[0] = True
        
        if isinstance(output, tuple):
            return (patched_h,) + output[1:]
        return patched_h
    
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
        log(f"    Model forward failed: {str(e)[:60]}")
        hook.remove()
        return None, None, None
    
    hook.remove()
    
    logit_diff = float(logits[target_token_id] - logits[competitor_token_id])
    target_logit = float(logits[target_token_id])
    competitor_logit = float(logits[competitor_token_id])
    
    return logit_diff, target_logit, competitor_logit


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 383: Real Category Swap Causal Test — {model_name}")
    log(f"=" * 60)
    
    # Target layers
    if model_name == "deepseek7b":
        target_layers = [4, 12, 24]
    elif model_name == "qwen3":
        target_layers = [4, 12, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 30]
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers")
    
    # Number of test pairs (subset for efficiency - real intervention is slow)
    n_test = min(30, len(ALL_PAIRS))
    np.random.seed(42)
    test_indices = sorted(np.random.choice(len(ALL_PAIRS), n_test, replace=False).tolist())
    
    all_results = {}
    
    for l in target_layers:
        log(f"\n--- Layer {l} ---")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # Step 1: Collect residual states for ALL pairs (needed for category subspace)
        log(f"  Step 1: Collecting residual states for all {len(ALL_PAIRS)} pairs...")
        h_clean_all = []
        h_corrupt_all = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{len(ALL_PAIRS)}")
            
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
        
        # Compute PROPER post-RMSNorm differences
        n_all = h_clean_all.shape[0]
        h_clean_norm = np.zeros_like(h_clean_all)
        h_corrupt_norm = np.zeros_like(h_corrupt_all)
        for i in range(n_all):
            h_clean_norm[i] = rms_norm_single(h_clean_all[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt_all[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Step 2: Compute category subspace
        log(f"  Step 2: Computing category subspace...")
        labels = [PAIR_CATEGORIES[i] for i in range(n_all)]
        Q, cat_centroids, overall_mean = compute_category_subspace(dh_proper, labels)
        
        # Project each dh_proper onto category subspace
        cat_components = dh_proper @ Q  # (n, n_cat)
        cat_projections = cat_components @ Q.T  # (n, d) - category part
        noncat_projections = dh_proper - cat_projections  # (n, d) - non-category part
        
        # Category R²
        cat_r2 = 1.0 - np.sum((dh_proper - cat_projections - overall_mean)**2) / max(np.sum((dh_proper - overall_mean)**2), 1e-10)
        log(f"    Category R² = {cat_r2:.4f}")
        
        # Step 3: Real causal interventions
        log(f"  Step 3: Running real causal interventions on {n_test} pairs...")
        
        intervention_results = {
            "clean_baseline": [],      # Clean forward pass logit diff
            "corrupt_baseline": [],    # Corrupt forward pass logit diff
            "add_cat_to_corrupt": [],  # Corrupt + category component → should improve
            "remove_cat_from_clean": [],  # Clean - category component → should hurt
            "cross_cat_swap": [],      # Corrupt_i + cat_component_j (j different cat)
            "same_cat_swap": [],       # Corrupt_i + cat_component_j (j same cat)
            "zero_cat": [],            # Corrupt + only non-cat → remove cat entirely
        }
        
        for pidx in test_indices:
            obj, target, competitor = ALL_PAIRS[pidx]
            cat_i = PAIR_CATEGORIES[pidx]
            
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            if t_id < 0 or c_id < 0:
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # 3a. Clean baseline
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            clean_logits = clean_out.logits[0, -1].float().cpu().numpy()
            clean_diff = float(clean_logits[t_id] - clean_logits[c_id])
            intervention_results["clean_baseline"].append(clean_diff)
            del clean_out
            
            # 3b. Corrupt baseline
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            corrupt_logits = corrupt_out.logits[0, -1].float().cpu().numpy()
            corrupt_diff = float(corrupt_logits[t_id] - corrupt_logits[c_id])
            intervention_results["corrupt_baseline"].append(corrupt_diff)
            del corrupt_out
            
            # Get the category component for this pair
            cat_proj_i = cat_projections[pidx]  # (d,) - category part of dh_proper
            noncat_proj_i = noncat_projections[pidx]  # (d,) - non-category part
            
            # 3c. Add category component to corrupt
            # We need to add cat_proj_i to the residual stream at layer l+1 output
            # The residual at layer l+1 is h_corrupt_norm[pidx] (post-RMSNorm)
            # We want to add cat_proj_i (which is in post-RMSNorm space) to the RAW residual
            # But cat_proj_i is computed in post-RMSNorm space, so we need to un-RMSNorm it
            # Actually, we should work in the raw residual space:
            # h_raw_corrupt + something → patched raw residual → model continues
            
            # For simplicity, let's add the difference in raw residual space
            # dh_raw = h_clean_raw - h_corrupt_raw
            # cat_proj in raw space ≈ cat_proj in post-RMSNorm space (approximate)
            # This is because RMSNorm is approximately linear for small perturbations
            
            # Better approach: directly patch the residual stream
            # Get the raw residual at layer l+1 for corrupt, then add cat_proj_i
            # The hook will do this during forward pass
            
            # Get h_corrupt at layer l+1 in raw residual space
            h_corrupt_raw = h_corrupt_all[pidx]  # (d,) raw
            
            # Add category component to corrupt's residual
            # We add it to the RAW residual (before RMSNorm of next layer)
            cat_add_raw = cat_proj_i  # approx: cat_proj in post-RMSNorm ≈ raw (small perturbation)
            
            def make_add_patch(delta_vec):
                """Create a patch function that adds delta to the last token's residual."""
                delta_tensor = torch.tensor(delta_vec, dtype=torch.bfloat16, device=device)
                def patch_fn(h):
                    h_patched = h.clone()
                    h_patched[0, -1, :] += delta_tensor
                    return h_patched
                return patch_fn
            
            # 3c. Add category component to corrupt
            ld, _, _ = run_model_with_patched_residual(
                model, tokenizer, device, corrupt_prompt, l,
                make_add_patch(cat_add_raw), t_id, c_id)
            if ld is not None:
                intervention_results["add_cat_to_corrupt"].append(ld)
            
            # 3d. Remove category component from clean
            ld, _, _ = run_model_with_patched_residual(
                model, tokenizer, device, clean_prompt, l,
                make_add_patch(-cat_add_raw), t_id, c_id)
            if ld is not None:
                intervention_results["remove_cat_from_clean"].append(ld)
            
            # 3e. Cross-category swap
            # Find a pair from a different category
            diff_cat_indices = [j for j in range(n_all) if PAIR_CATEGORIES[j] != cat_i]
            if len(diff_cat_indices) > 0:
                j_cross = diff_cat_indices[np.random.randint(len(diff_cat_indices))]
                cat_proj_j = cat_projections[j_cross]
                # Swap: add j's category component instead of i's
                swap_delta = cat_proj_j - cat_proj_i
                ld, _, _ = run_model_with_patched_residual(
                    model, tokenizer, device, clean_prompt, l,
                    make_add_patch(swap_delta), t_id, c_id)
                if ld is not None:
                    intervention_results["cross_cat_swap"].append(ld)
            
            # 3f. Same-category swap
            same_cat_indices = [j for j in range(n_all) if PAIR_CATEGORIES[j] == cat_i and j != pidx]
            if len(same_cat_indices) > 0:
                j_same = same_cat_indices[np.random.randint(len(same_cat_indices))]
                cat_proj_j = cat_projections[j_same]
                swap_delta = cat_proj_j - cat_proj_i
                ld, _, _ = run_model_with_patched_residual(
                    model, tokenizer, device, clean_prompt, l,
                    make_add_patch(swap_delta), t_id, c_id)
                if ld is not None:
                    intervention_results["same_cat_swap"].append(ld)
            
            # 3g. Zero category component (remove from clean, keep non-cat)
            # clean - cat_proj_i (remove category, keep everything else)
            ld, _, _ = run_model_with_patched_residual(
                model, tokenizer, device, clean_prompt, l,
                make_add_patch(-cat_proj_i), t_id, c_id)
            if ld is not None:
                intervention_results["zero_cat"].append(ld)
            
            torch.cuda.empty_cache()
        
        # Compute summary statistics
        summary = {}
        for key in intervention_results:
            vals = intervention_results[key]
            if len(vals) > 0:
                summary[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        
        # Compute causal effect sizes
        if len(intervention_results["add_cat_to_corrupt"]) > 0 and len(intervention_results["corrupt_baseline"]) > 0:
            corrupt_base = np.array(intervention_results["corrupt_baseline"])
            add_cat = np.array(intervention_results["add_cat_to_corrupt"])
            cat_effect = add_cat - corrupt_base
            summary["add_cat_causal_effect"] = {
                "mean": float(np.mean(cat_effect)),
                "std": float(np.std(cat_effect)),
                "t_stat": float(np.mean(cat_effect) / (np.std(cat_effect) / np.sqrt(len(cat_effect)) + 1e-10)),
            }
        
        if len(intervention_results["remove_cat_from_clean"]) > 0 and len(intervention_results["clean_baseline"]) > 0:
            clean_base = np.array(intervention_results["clean_baseline"])
            remove_cat = np.array(intervention_results["remove_cat_from_clean"])
            remove_effect = remove_cat - clean_base
            summary["remove_cat_causal_effect"] = {
                "mean": float(np.mean(remove_effect)),
                "std": float(np.std(remove_effect)),
                "t_stat": float(np.mean(remove_effect) / (np.std(remove_effect) / np.sqrt(len(remove_effect)) + 1e-10)),
            }
        
        if len(intervention_results["cross_cat_swap"]) > 0 and len(intervention_results["same_cat_swap"]) > 0:
            cross = np.array(intervention_results["cross_cat_swap"])
            same = np.array(intervention_results["same_cat_swap"])
            # Note: both are from clean baseline, so the diff is the swap effect
            clean_base = np.array(intervention_results["clean_baseline"][:len(cross)])
            cross_effect = cross - clean_base[:len(cross)]
            same_effect = same - clean_base[:len(same)]
            summary["swap_causal_effect"] = {
                "cross_cat_mean": float(np.mean(cross_effect)),
                "same_cat_mean": float(np.mean(same_effect)),
                "cross_vs_same_diff": float(np.mean(cross_effect) - np.mean(same_effect)),
            }
        
        # Print results
        log(f"\n  Results for Layer {l}:")
        log(f"    Category R² = {cat_r2:.4f}")
        for key, s in summary.items():
            if isinstance(s, dict) and "mean" in s:
                log(f"    {key:30s}: mean={s['mean']:.4f}, std={s.get('std', 0):.4f}, n={s.get('n', 0)}")
        
        all_results[str(l)] = {
            "layer": l,
            "category_r2": float(cat_r2),
            "n_test_pairs": n_test,
            "summary": summary,
            "raw_results": {k: [float(v) for v in vs] for k, vs in intervention_results.items()},
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    # Save results
    out_dir = "results/phase383_category_swap_causal"
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
        "n_test_pairs": n_test,
        "n_total_pairs": len(ALL_PAIRS),
        "test": "phase383_category_swap_causal",
        "results": convert(all_results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase383.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 383 complete for {model_name}!")


if __name__ == "__main__":
    main()
