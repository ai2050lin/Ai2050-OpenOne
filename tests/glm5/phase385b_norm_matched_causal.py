"""
Phase 385b: 确认测试 — 范数匹配 + 大样本DS7B forward验证
=========================================================

Phase 385的关键发现需要确认：
1. Probe vs Random效应差异可能来自范数差异（probe投影范数更大）
2. DS7B的forward_add方向正确但微弱（t<1.5），需要更大样本量
3. 需要在raw residual空间（而非post-RMSNorm空间）提取category方向

确认测试设计：
Part 1: 范数匹配的probe vs random因果比较
  - 对probe和random投影，归一化到相同范数
  - 比较归一化后的因果效应差异
  - 如果差异消失→之前的效应来自范数差异
  - 如果差异保留→确实是方向特异效应

Part 2: 大样本DS7B forward验证（n=150）
  - 用全部151对样本做forward patch测试
  - 累积t统计量，看是否能达到显著

Part 3: Raw空间category方向提取
  - 在raw residual空间（而非post-RMSNorm空间）提取probe方向
  - 用raw空间方向做forward patch
  - 消除空间不匹配问题

用法:
  python tests/glm5/phase385b_norm_matched_causal.py qwen3
  python tests/glm5/phase385b_norm_matched_causal.py deepseek7b
  python tests/glm5/phase385b_norm_matched_causal.py glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
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


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 385b: Norm-Matched Confirm — {model_name}")
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
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    
    # ===== Collect all residual states =====
    log("\n=== Collecting residual states ===")
    
    results = {}
    
    for l in target_layers:
        log(f"\n--- Layer {l} ---")
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
        
        # Raw dh (without RMSNorm)
        dh_raw = h_clean_all - h_corrupt_all
        
        # Residualize object identity (both spaces)
        dh_resid_post = residualize_object_identity(dh_proper, object_labels)
        dh_resid_raw = residualize_object_identity(dh_raw, object_labels)
        
        # ===== Part 1: Train probes in both spaces =====
        log(f"  Training probes (post-RMSNorm + raw)...")
        
        # Post-RMSNorm probe
        clf_post = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        clf_post.fit(dh_resid_post, category_labels)
        W_post = clf_post.coef_  # (7, d)
        Q_post, _ = np.linalg.qr(W_post.T)  # (d, 6)
        cat_proj_post = (dh_resid_post @ Q_post) @ Q_post.T  # (n, d)
        
        # Raw space probe
        clf_raw = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        clf_raw.fit(dh_resid_raw, category_labels)
        W_raw = clf_raw.coef_
        Q_raw, _ = np.linalg.qr(W_raw.T)
        cat_proj_raw = (dh_resid_raw @ Q_raw) @ Q_raw.T  # (n, d)
        
        # Random directions (norm-matched)
        n_dirs = Q_post.shape[1]
        d_dim = Q_post.shape[0]
        np.random.seed(123)
        Q_rand, _ = np.linalg.qr(np.random.randn(d_dim, n_dirs))
        rand_proj = (dh_resid_post @ Q_rand) @ Q_rand.T
        
        # ===== Norm statistics =====
        probe_norms = np.linalg.norm(cat_proj_post, axis=1)
        raw_norms = np.linalg.norm(cat_proj_raw, axis=1)
        rand_norms = np.linalg.norm(rand_proj, axis=1)
        
        log(f"  Projection norms: probe={np.mean(probe_norms):.4f}±{np.std(probe_norms):.4f}, "
            f"raw_probe={np.mean(raw_norms):.4f}±{np.std(raw_norms):.4f}, "
            f"random={np.mean(rand_norms):.4f}±{np.std(rand_norms):.4f}")
        
        # ===== Norm-matched versions =====
        # Scale random projections to match probe norm per sample
        rand_proj_matched = np.zeros_like(rand_proj)
        for i in range(n_pairs):
            probe_n = np.linalg.norm(cat_proj_post[i])
            rand_n = np.linalg.norm(rand_proj[i])
            if rand_n > 1e-10:
                rand_proj_matched[i] = rand_proj[i] * (probe_n / rand_n)
            else:
                rand_proj_matched[i] = rand_proj[i]
        
        matched_rand_norms = np.linalg.norm(rand_proj_matched, axis=1)
        log(f"  Norm-matched random: {np.mean(matched_rand_norms):.4f}±{np.std(matched_rand_norms):.4f}")
        
        # ===== Causal tests =====
        n_test = min(n_pairs, n_pairs)  # Use ALL pairs
        test_indices = list(range(n_test))
        
        log(f"  Running causal tests on {n_test} pairs (3 conditions: probe, raw_probe, norm-matched random)...")
        
        scale = 1.0
        
        ca_data = {
            "probe": {"add": [], "remove": []},
            "raw_probe": {"add": [], "remove": []},
            "matched_random": {"add": [], "remove": []},
            "probe_swap": {"cross": [], "same": []},
        }
        baselines = {"clean_ld": [], "corrupt_ld": []}
        
        np.random.seed(456)
        
        for cnt, pidx in enumerate(test_indices):
            if cnt % 30 == 0:
                log(f"    Pair {cnt+1}/{n_test}")
            
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
            
            # Probe direction (post-RMSNorm space) — add to corrupt
            ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                       cat_proj_post[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["probe"]["add"].append(ld)
            
            # Probe direction — remove from clean
            ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                       -cat_proj_post[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["probe"]["remove"].append(ld)
            
            # Raw probe direction (raw residual space) — add to corrupt
            ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                       cat_proj_raw[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["raw_probe"]["add"].append(ld)
            
            # Raw probe direction — remove from clean
            ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                       -cat_proj_raw[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["raw_probe"]["remove"].append(ld)
            
            # Norm-matched random direction — add to corrupt
            ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                       rand_proj_matched[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["matched_random"]["add"].append(ld)
            
            # Norm-matched random direction — remove from clean
            ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                       -rand_proj_matched[pidx] * scale, t_id, c_id)
            if ld is not None:
                ca_data["matched_random"]["remove"].append(ld)
            
            # Swap test
            diff_cat_idx = [j for j in range(n_pairs) if category_labels[j] != cat_i]
            same_cat_idx = [j for j in range(n_pairs) if category_labels[j] == cat_i and j != pidx]
            
            if len(diff_cat_idx) > 0:
                j_cross = diff_cat_idx[np.random.randint(len(diff_cat_idx))]
                swap_delta = cat_proj_post[j_cross] - cat_proj_post[pidx]
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    ca_data["probe_swap"]["cross"].append(ld)
            
            if len(same_cat_idx) > 0:
                j_same = same_cat_idx[np.random.randint(len(same_cat_idx))]
                swap_delta = cat_proj_post[j_same] - cat_proj_post[pidx]
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                           swap_delta, t_id, c_id)
                if ld is not None:
                    ca_data["probe_swap"]["same"].append(ld)
            
            if cnt % 5 == 0:
                torch.cuda.empty_cache()
        
        # Compute effects
        layer_result = {"layer": l, "n_test": n_test}
        
        for cond in ["probe", "raw_probe", "matched_random"]:
            # Add effect
            n_eff = min(len(ca_data[cond]["add"]), len(baselines["corrupt_ld"]))
            if n_eff > 0:
                add_eff = np.array(ca_data[cond]["add"][:n_eff]) - np.array(baselines["corrupt_ld"][:n_eff])
                layer_result[f"{cond}_add"] = {
                    "mean": float(np.mean(add_eff)),
                    "std": float(np.std(add_eff)),
                    "t": float(np.mean(add_eff) / (np.std(add_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
            
            # Remove effect
            n_eff2 = min(len(ca_data[cond]["remove"]), len(baselines["clean_ld"]))
            if n_eff2 > 0:
                rem_eff = np.array(ca_data[cond]["remove"][:n_eff2]) - np.array(baselines["clean_ld"][:n_eff2])
                layer_result[f"{cond}_remove"] = {
                    "mean": float(np.mean(rem_eff)),
                    "std": float(np.std(rem_eff)),
                    "t": float(np.mean(rem_eff) / (np.std(rem_eff) / np.sqrt(n_eff2) + 1e-10)),
                    "n": n_eff2,
                }
        
        # Swap
        n_swap = min(len(ca_data["probe_swap"]["cross"]),
                     len(ca_data["probe_swap"]["same"]),
                     len(baselines["clean_ld"]))
        if n_swap > 0:
            cross_eff = np.array(ca_data["probe_swap"]["cross"][:n_swap]) - \
                        np.array(baselines["clean_ld"][:n_swap])
            same_eff = np.array(ca_data["probe_swap"]["same"][:n_swap]) - \
                       np.array(baselines["clean_ld"][:n_swap])
            diff = cross_eff - same_eff
            layer_result["swap"] = {
                "cross_mean": float(np.mean(cross_eff)),
                "same_mean": float(np.mean(same_eff)),
                "diff": float(np.mean(diff)),
                "diff_std": float(np.std(diff)),
                "diff_t": float(np.mean(diff) / (np.std(diff) / np.sqrt(n_swap) + 1e-10)),
                "n": n_swap,
            }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
        
        # Print summary
        for cond in ["probe", "raw_probe", "matched_random"]:
            ae = layer_result.get(f"{cond}_add", {})
            re = layer_result.get(f"{cond}_remove", {})
            log(f"  {cond:16s} add: mean={ae.get('mean',0):+.4f} t={ae.get('t',0):.2f} n={ae.get('n',0)}")
            log(f"  {cond:16s} rem: mean={re.get('mean',0):+.4f} t={re.get('t',0):.2f} n={re.get('n',0)}")
        
        se = layer_result.get("swap", {})
        if se:
            log(f"  swap: diff={se.get('diff',0):+.4f} t={se.get('diff_t',0):.2f} n={se.get('n',0)}")
    
    # Save results
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
        "test": "phase385b_norm_matched_confirm",
        "results": convert(results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase385b.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # ===== Summary =====
    log(f"\n{'='*60}")
    log(f"Phase 385b Summary — {model_name}")
    log(f"{'='*60}")
    
    for l_str in sorted(results.keys(), key=int):
        r = results[l_str]
        log(f"\nLayer {l_str}:")
        for cond in ["probe", "raw_probe", "matched_random"]:
            ae = r.get(f"{cond}_add", {})
            re = r.get(f"{cond}_remove", {})
            log(f"  {cond:16s} add={ae.get('mean',0):+.4f}(t={ae.get('t',0):.2f}), "
                f"rem={re.get('mean',0):+.4f}(t={re.get('t',0):.2f})")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 385b complete for {model_name}!")


if __name__ == "__main__":
    main()
