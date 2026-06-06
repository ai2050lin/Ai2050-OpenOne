"""
Phase 386b: 确认测试 — ANOVA A分量稳定性 + 多尺度验证
=====================================================

Phase 386的核心发现需要确认：
1. ANOVA A分量(category centroid)因果效力高度显著(t=3.46~7.61)
   - 需要确认：这是否稳定？还是特定于probe训练方式？
2. A分量比raw_probe更有效——是因为centroid方向更优，还是因为其他原因？
3. 需要测试：A分量在不同scale下的效应是否单调递增？

确认测试设计：
Part 1: 多随机种子centroid vs probe对比
  - 用5个不同随机种子训练probe
  - 对比probe方向 vs centroid方向的因果效力
  - 如果centroid始终更优，说明centroid确实更接近真实信号

Part 2: 多尺度因果测试
  - A分量在0.5x, 1.0x, 2.0x scale下的因果效应
  - 如果效应随scale单调递增，说明方向确实正确

Part 3: I+A联合效应
  - 测试I+A联合添加的因果效力
  - 对比I单独、A单独、I+A、full Δh

Usage:
  python tests/glm5/phase386b_confirm_hierarchy.py qwen3
  python tests/glm5/phase386b_confirm_hierarchy.py deepseek7b
  python tests/glm5/phase386b_confirm_hierarchy.py glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

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


def run_model_with_patch(model, tokenizer, device, prompt, layer_idx,
                         patch_delta, target_token_id, competitor_token_id):
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
        log(f"    Forward failed: {str(e)[:80]}")
        hook.remove()
        return None
    hook.remove()
    return float(logits[target_token_id] - logits[competitor_token_id])


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 386b: Confirm Hierarchy — {model_name}")
    log(f"=" * 70)
    
    # Focus on key layers
    if model_name == "qwen3":
        target_layers = [4, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 20, 30]
    elif model_name == "deepseek7b":
        target_layers = [4, 12, 24]
    
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    log(f"  Model loaded: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    
    results = {}
    
    for l in target_layers:
        log(f"\n--- Layer {l} ---")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # ===== Collect residual states =====
        log(f"  Collecting residual states...")
        
        h_clean_raw = []
        h_corrupt_raw = []
        target_token_ids = []
        competitor_token_ids = []
        baseline_clean_ld = []
        baseline_corrupt_ld = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)
            
            with torch.no_grad():
                clean_toks = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
                clean_out = model(
                    input_ids=clean_toks["input_ids"].to(device),
                    attention_mask=clean_toks["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            last_pos = clean_toks["input_ids"].shape[1] - 1
            h_raw_c = clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
            h_clean_raw.append(h_raw_c)
            cl = clean_out.logits[0, -1].float().cpu().numpy()
            if t_id >= 0 and c_id >= 0:
                baseline_clean_ld.append(float(cl[t_id] - cl[c_id]))
            else:
                baseline_clean_ld.append(0.0)
            del clean_out
            
            with torch.no_grad():
                corrupt_toks = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
                corrupt_out = model(
                    input_ids=corrupt_toks["input_ids"].to(device),
                    attention_mask=corrupt_toks["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            last_pos_r = corrupt_toks["input_ids"].shape[1] - 1
            h_raw_r = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            h_corrupt_raw.append(h_raw_r)
            crl = corrupt_out.logits[0, -1].float().cpu().numpy()
            if t_id >= 0 and c_id >= 0:
                baseline_corrupt_ld.append(float(crl[t_id] - crl[c_id]))
            else:
                baseline_corrupt_ld.append(0.0)
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_clean_raw = np.array(h_clean_raw)
        h_corrupt_raw = np.array(h_corrupt_raw)
        dh_raw = h_clean_raw - h_corrupt_raw
        baseline_clean_ld = np.array(baseline_clean_ld)
        baseline_corrupt_ld = np.array(baseline_corrupt_ld)
        
        # ===== ANOVA Decomposition =====
        mu = np.mean(dh_raw, axis=0)
        
        # Object centroids
        unique_objs = sorted(set(object_labels))
        obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
        c_obj = np.zeros((len(unique_objs), dh_raw.shape[1]))
        obj_counts = np.zeros(len(unique_objs))
        for i in range(n_pairs):
            oi = obj_to_idx[object_labels[i]]
            c_obj[oi] += dh_raw[i]
            obj_counts[oi] += 1
        for j in range(len(unique_objs)):
            if obj_counts[j] > 0:
                c_obj[j] /= obj_counts[j]
        
        I_comp = np.zeros_like(dh_raw)
        for i in range(n_pairs):
            I_comp[i] = c_obj[obj_to_idx[object_labels[i]]] - mu
        
        dh_resid_I = dh_raw - mu - I_comp
        
        # Category centroids (on I-residualized data)
        unique_cats = sorted(set(category_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        c_cat = np.zeros((len(unique_cats), dh_raw.shape[1]))
        cat_counts = np.zeros(len(unique_cats))
        for i in range(n_pairs):
            ci = cat_to_idx[category_labels[i]]
            c_cat[ci] += dh_resid_I[i]
            cat_counts[ci] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                c_cat[j] /= cat_counts[j]
        
        A_comp = np.zeros_like(dh_raw)
        for i in range(n_pairs):
            A_comp[i] = c_cat[cat_to_idx[category_labels[i]]]
        
        eps_comp = dh_raw - mu - I_comp - A_comp
        I_plus_A = I_comp + A_comp
        
        # ===== Part 1: Centroid vs Probe (5 seeds) =====
        log(f"  Part 1: Centroid vs Probe (5 seeds)...")
        
        # Residualize for probe training
        obj_onehot = np.zeros((n_pairs, len(unique_objs)))
        for i in range(n_pairs):
            obj_onehot[i, obj_to_idx[object_labels[i]]] = 1.0
        if obj_onehot.shape[1] > 1:
            X_obj = obj_onehot[:, :-1]
        else:
            X_obj = obj_onehot
        X_design = np.column_stack([np.ones(n_pairs), X_obj])
        beta_obj = np.linalg.lstsq(X_design, dh_raw, rcond=None)[0]
        dh_raw_resid = dh_raw - X_design @ beta_obj
        
        probe_results = []
        for seed in [42, 123, 456, 789, 1024]:
            clf = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0, random_state=seed)
            clf.fit(dh_raw_resid, category_labels)
            W = clf.coef_
            Q, _ = np.linalg.qr(W.T)
            probe_proj = (dh_raw_resid @ Q) @ Q.T
            
            # Test on a random subset of 60 pairs for speed
            np.random.seed(seed)
            test_idx = np.random.choice(n_pairs, size=min(60, n_pairs), replace=False)
            
            add_effs = []
            for cnt, pidx in enumerate(test_idx):
                obj, target, competitor = ALL_PAIRS[pidx]
                t_id = target_token_ids[pidx]
                c_id = competitor_token_ids[pidx]
                if t_id < 0 or c_id < 0:
                    continue
                
                corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          probe_proj[pidx], t_id, c_id)
                if ld is not None:
                    add_effs.append(ld - baseline_corrupt_ld[pidx])
                
                if cnt % 10 == 0:
                    torch.cuda.empty_cache()
            
            if len(add_effs) > 0:
                probe_results.append({
                    'seed': seed,
                    'mean': float(np.mean(add_effs)),
                    't': float(np.mean(add_effs) / (np.std(add_effs) / np.sqrt(len(add_effs)) + 1e-10)),
                    'n': len(add_effs),
                })
        
        # Centroid test on same 60-pair subsets
        centroid_results = []
        for seed in [42, 123, 456, 789, 1024]:
            np.random.seed(seed)
            test_idx = np.random.choice(n_pairs, size=min(60, n_pairs), replace=False)
            
            add_effs = []
            for cnt, pidx in enumerate(test_idx):
                obj, target, competitor = ALL_PAIRS[pidx]
                t_id = target_token_ids[pidx]
                c_id = competitor_token_ids[pidx]
                if t_id < 0 or c_id < 0:
                    continue
                
                corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          A_comp[pidx], t_id, c_id)
                if ld is not None:
                    add_effs.append(ld - baseline_corrupt_ld[pidx])
                
                if cnt % 10 == 0:
                    torch.cuda.empty_cache()
            
            if len(add_effs) > 0:
                centroid_results.append({
                    'seed': seed,
                    'mean': float(np.mean(add_effs)),
                    't': float(np.mean(add_effs) / (np.std(add_effs) / np.sqrt(len(add_effs)) + 1e-10)),
                    'n': len(add_effs),
                })
        
        # ===== Part 2: Multi-scale A test =====
        log(f"  Part 2: Multi-scale A test...")
        
        np.random.seed(42)
        test_idx_scale = np.random.choice(n_pairs, size=min(60, n_pairs), replace=False)
        
        scale_results = {}
        for scale in [0.5, 1.0, 2.0]:
            add_effs = []
            for cnt, pidx in enumerate(test_idx_scale):
                obj, target, competitor = ALL_PAIRS[pidx]
                t_id = target_token_ids[pidx]
                c_id = competitor_token_ids[pidx]
                if t_id < 0 or c_id < 0:
                    continue
                
                corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          A_comp[pidx] * scale, t_id, c_id)
                if ld is not None:
                    add_effs.append(ld - baseline_corrupt_ld[pidx])
                
                if cnt % 10 == 0:
                    torch.cuda.empty_cache()
            
            if len(add_effs) > 0:
                scale_results[str(scale)] = {
                    'mean': float(np.mean(add_effs)),
                    't': float(np.mean(add_effs) / (np.std(add_effs) / np.sqrt(len(add_effs)) + 1e-10)),
                    'n': len(add_effs),
                }
        
        # ===== Part 3: I+A combined test =====
        log(f"  Part 3: I+A combined test...")
        
        np.random.seed(789)
        test_idx_combo = np.random.choice(n_pairs, size=min(60, n_pairs), replace=False)
        
        combo_components = {
            'I': I_comp,
            'A': A_comp,
            'I+A': I_plus_A,
            'full': dh_raw,
        }
        
        combo_results = {}
        for comp_name, comp_vec in combo_components.items():
            add_effs = []
            rem_effs = []
            for cnt, pidx in enumerate(test_idx_combo):
                obj, target, competitor = ALL_PAIRS[pidx]
                t_id = target_token_ids[pidx]
                c_id = competitor_token_ids[pidx]
                if t_id < 0 or c_id < 0:
                    continue
                
                clean_prompt = TEMPLATE.format(obj=obj, attr=target)
                corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                
                # Add to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          comp_vec[pidx], t_id, c_id)
                if ld is not None:
                    add_effs.append(ld - baseline_corrupt_ld[pidx])
                
                # Remove from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                          -comp_vec[pidx], t_id, c_id)
                if ld is not None:
                    rem_effs.append(ld - baseline_clean_ld[pidx])
                
                if cnt % 5 == 0:
                    torch.cuda.empty_cache()
            
            combo_results[comp_name] = {
                'add': {'mean': float(np.mean(add_effs)), 't': float(np.mean(add_effs) / (np.std(add_effs) / np.sqrt(len(add_effs)) + 1e-10))} if add_effs else {},
                'remove': {'mean': float(np.mean(rem_effs)), 't': float(np.mean(rem_effs) / (np.std(rem_effs) / np.sqrt(len(rem_effs)) + 1e-10))} if rem_effs else {},
            }
        
        # ===== Store results =====
        layer_result = {
            "layer": l,
            "probe_vs_centroid": {
                "probe": probe_results,
                "centroid": centroid_results,
            },
            "scale_test": scale_results,
            "combo_test": combo_results,
        }
        results[str(l)] = layer_result
        
        # Print summary
        log(f"\n  Layer {l} results:")
        
        # Probe vs Centroid
        probe_means = [p['mean'] for p in probe_results]
        centroid_means = [c['mean'] for c in centroid_results]
        probe_ts = [p['t'] for p in probe_results]
        centroid_ts = [c['t'] for c in centroid_results]
        log(f"  Probe:    mean={np.mean(probe_means):+.4f}±{np.std(probe_means):.4f}, "
            f"t={np.mean(probe_ts):.2f}±{np.std(probe_ts):.2f}")
        log(f"  Centroid: mean={np.mean(centroid_means):+.4f}±{np.std(centroid_means):.4f}, "
            f"t={np.mean(centroid_ts):.2f}±{np.std(centroid_ts):.2f}")
        
        # Scale
        for s in ['0.5', '1.0', '2.0']:
            sr = scale_results.get(s, {})
            log(f"  A×{s}: add={sr.get('mean',0):+.4f}(t={sr.get('t',0):.2f})")
        
        # Combo
        for cn in ['I', 'A', 'I+A', 'full']:
            cr = combo_results.get(cn, {})
            ae = cr.get('add', {})
            re = cr.get('remove', {})
            log(f"  {cn:6s}: add={ae.get('mean',0):+.4f}(t={ae.get('t',0):.2f}), "
                f"rem={re.get('mean',0):+.4f}(t={re.get('t',0):.2f})")
        
        log(f"  Layer {l} done in {time.time()-t_l:.1f}s")
    
    # Save
    out_dir = "results/phase386_factor_causal_hierarchy"
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
        "test": "phase386b_confirm_hierarchy",
        "results": convert(results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase386b.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 386b complete for {model_name}!")


if __name__ == "__main__":
    main()
