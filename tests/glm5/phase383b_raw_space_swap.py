"""
Phase 383b: 确认测试 — Raw空间类别Swap + 更大样本量
===================================================

修复Phase 383的硬伤1（空间不匹配）：
- 在raw residual空间（而非post-RMSNorm空间）计算category subspace
- 增加测试样本量到60对
- 重点关注DS7B和GLM4的对比

方法改进：
1. 使用dh_raw = h_clean - h_corrupt（raw residual difference）计算category subspace
2. 所有patch操作都在raw residual空间进行
3. 对比raw-space vs post-RMSNorm-space的category R²

用法:
  python tests/glm5/phase383b_raw_space_swap.py qwen3
  python tests/glm5/phase383b_raw_space_swap.py deepseek7b
  python tests/glm5/phase383b_raw_space_swap.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime

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


def compute_category_subspace_raw(dh_raw, labels):
    """Compute category subspace in RAW residual space."""
    unique_labels = sorted(set(labels))
    cat_centroids = {}
    for cat in unique_labels:
        idx = [i for i, c in enumerate(labels) if c == cat]
        cat_centroids[cat] = np.mean(dh_raw[idx], axis=0)
    
    overall = np.mean(dh_raw, axis=0)
    cat_dirs = np.array([cat_centroids[cat] - overall for cat in unique_labels])
    Q, _ = np.linalg.qr(cat_dirs.T)
    return Q, cat_centroids, overall


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
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 383b: Raw-Space Category Swap Confirmation — {model_name}")
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
    
    # Increased test sample size
    n_test = min(60, len(ALL_PAIRS))
    np.random.seed(123)
    test_indices = sorted(np.random.choice(len(ALL_PAIRS), n_test, replace=False).tolist())
    
    all_results = {}
    
    for l in target_layers:
        log(f"\n--- Layer {l} ---")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # Collect ALL residual states
        log(f"  Collecting residual states for all {len(ALL_PAIRS)} pairs...")
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
        n_all = h_clean_all.shape[0]
        
        # === KEY CHANGE: Use RAW residual space ===
        dh_raw = h_clean_all - h_corrupt_all  # raw residual difference
        
        # Compute category subspace in RAW space
        labels = [PAIR_CATEGORIES[i] for i in range(n_all)]
        Q_raw, cat_centroids_raw, overall_raw = compute_category_subspace_raw(dh_raw, labels)
        
        # Project each dh_raw onto category subspace
        cat_components_raw = dh_raw @ Q_raw
        cat_projections_raw = cat_components_raw @ Q_raw.T
        noncat_projections_raw = dh_raw - cat_projections_raw
        
        # Category R² in raw space
        M_raw = dh_raw - dh_raw.mean(axis=0, keepdims=True)
        total_var_raw = np.sum(M_raw**2)
        pred_raw = np.zeros_like(dh_raw)
        for i in range(n_all):
            cat = labels[i]
            pred_raw[i] = cat_centroids_raw[cat]
        residual_raw = dh_raw - pred_raw
        cat_r2_raw = 1.0 - np.sum((residual_raw - residual_raw.mean(axis=0))**2) / max(total_var_raw, 1e-10)
        
        log(f"    Category R² (raw space) = {cat_r2_raw:.4f}")
        
        # Also compute in post-RMSNorm space for comparison
        h_clean_norm = np.zeros_like(h_clean_all)
        h_corrupt_norm = np.zeros_like(h_corrupt_all)
        for i in range(n_all):
            h_clean_norm[i] = rms_norm_single(h_clean_all[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt_all[i], ln_weight)
        dh_proper = h_clean_norm - h_corrupt_norm
        M_proper = dh_proper - dh_proper.mean(axis=0, keepdims=True)
        total_var_proper = np.sum(M_proper**2)
        pred_proper = np.zeros_like(dh_proper)
        for cat in ALL_CATEGORIES:
            idx = [i for i, c in enumerate(labels) if c == cat]
            centroid = np.mean(dh_proper[idx], axis=0)
            for i in idx:
                pred_proper[i] = centroid
        residual_proper = dh_proper - pred_proper
        cat_r2_proper = 1.0 - np.sum((residual_proper - residual_proper.mean(axis=0))**2) / max(total_var_proper, 1e-10)
        
        log(f"    Category R² (post-RMSNorm) = {cat_r2_proper:.4f}")
        
        # === Real causal interventions (in RAW space) ===
        log(f"  Running raw-space causal interventions on {n_test} pairs...")
        
        results = {
            "clean_baseline": [],
            "corrupt_baseline": [],
            "add_cat_raw": [],         # corrupt + cat_proj_raw → should help
            "remove_cat_raw": [],      # clean - cat_proj_raw → should hurt
            "cross_cat_swap_raw": [],  # clean + (cat_j - cat_i) from different cat
            "same_cat_swap_raw": [],   # clean + (cat_j - cat_i) from same cat
        }
        
        np.random.seed(456)
        
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
            
            # Baselines
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            clean_logits = clean_out.logits[0, -1].float().cpu().numpy()
            results["clean_baseline"].append(float(clean_logits[t_id] - clean_logits[c_id]))
            del clean_out
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(device),
                )
            corrupt_logits = corrupt_out.logits[0, -1].float().cpu().numpy()
            results["corrupt_baseline"].append(float(corrupt_logits[t_id] - corrupt_logits[c_id]))
            del corrupt_out
            
            # Get category component for this pair (RAW space)
            cat_proj_i = cat_projections_raw[pidx]
            
            # Add category component to corrupt (raw space)
            ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                       cat_proj_i, t_id, c_id)
            if ld is not None:
                results["add_cat_raw"].append(ld)
            
            # Remove category component from clean (raw space)
            ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                       -cat_proj_i, t_id, c_id)
            if ld is not None:
                results["remove_cat_raw"].append(ld)
            
            # Cross-category swap (raw space)
            diff_cat_idx = [j for j in range(n_all) if PAIR_CATEGORIES[j] != cat_i]
            if len(diff_cat_idx) > 0:
                j_cross = diff_cat_idx[np.random.randint(len(diff_cat_idx))]
                cat_proj_j = cat_projections_raw[j_cross]
                swap_delta = cat_proj_j - cat_proj_i
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    results["cross_cat_swap_raw"].append(ld)
            
            # Same-category swap (raw space)
            same_cat_idx = [j for j in range(n_all) if PAIR_CATEGORIES[j] == cat_i and j != pidx]
            if len(same_cat_idx) > 0:
                j_same = same_cat_idx[np.random.randint(len(same_cat_idx))]
                cat_proj_j = cat_projections_raw[j_same]
                swap_delta = cat_proj_j - cat_proj_i
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    results["same_cat_swap_raw"].append(ld)
            
            torch.cuda.empty_cache()
        
        # Summary
        summary = {}
        for key in results:
            vals = results[key]
            if len(vals) > 0:
                summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        
        # Causal effects
        if len(results["add_cat_raw"]) > 0 and len(results["corrupt_baseline"]) > 0:
            n_match = min(len(results["add_cat_raw"]), len(results["corrupt_baseline"]))
            effect = np.array(results["add_cat_raw"][:n_match]) - np.array(results["corrupt_baseline"][:n_match])
            summary["add_cat_effect"] = {
                "mean": float(np.mean(effect)),
                "std": float(np.std(effect)),
                "t": float(np.mean(effect) / (np.std(effect) / np.sqrt(len(effect)) + 1e-10)),
            }
        
        if len(results["remove_cat_raw"]) > 0 and len(results["clean_baseline"]) > 0:
            n_match = min(len(results["remove_cat_raw"]), len(results["clean_baseline"]))
            effect = np.array(results["remove_cat_raw"][:n_match]) - np.array(results["clean_baseline"][:n_match])
            summary["remove_cat_effect"] = {
                "mean": float(np.mean(effect)),
                "std": float(np.std(effect)),
                "t": float(np.mean(effect) / (np.std(effect) / np.sqrt(len(effect)) + 1e-10)),
            }
        
        if len(results["cross_cat_swap_raw"]) > 0 and len(results["same_cat_swap_raw"]) > 0:
            n_match = min(len(results["cross_cat_swap_raw"]), len(results["clean_baseline"]),
                         len(results["same_cat_swap_raw"]))
            cross_eff = np.array(results["cross_cat_swap_raw"][:n_match]) - np.array(results["clean_baseline"][:n_match])
            same_eff = np.array(results["same_cat_swap_raw"][:n_match]) - np.array(results["clean_baseline"][:n_match])
            summary["swap_effect"] = {
                "cross_mean": float(np.mean(cross_eff)),
                "same_mean": float(np.mean(same_eff)),
                "diff": float(np.mean(cross_eff) - np.mean(same_eff)),
                "diff_t": float((np.mean(cross_eff) - np.mean(same_eff)) / 
                               (np.std(cross_eff - same_eff) / np.sqrt(len(cross_eff)) + 1e-10)),
            }
        
        log(f"\n  Results for Layer {l}:")
        log(f"    Cat R²: raw={cat_r2_raw:.4f}, post-RMSNorm={cat_r2_proper:.4f}")
        for key, s in summary.items():
            if isinstance(s, dict) and "mean" in s:
                t_str = f", t={s.get('t', 0):.2f}" if 't' in s else ""
                log(f"    {key:30s}: mean={s['mean']:.4f}, std={s.get('std', 0):.4f}, n={s.get('n', 0)}{t_str}")
        
        all_results[str(l)] = {
            "layer": l,
            "cat_r2_raw": float(cat_r2_raw),
            "cat_r2_post_rmsnorm": float(cat_r2_proper),
            "n_test_pairs": n_test,
            "summary": summary,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    # Save
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
        "test": "phase383b_raw_space_swap",
        "results": convert(all_results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase383b.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 383b complete for {model_name}!")


if __name__ == "__main__":
    main()
