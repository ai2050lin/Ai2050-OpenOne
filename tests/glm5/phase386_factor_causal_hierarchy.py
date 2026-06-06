"""
Phase 386: Factor Decomposition Causal Hierarchy + RMSNorm Jacobian Mapping
============================================================================

Core Question:
  Which factor component has the strongest raw-space causal effect?
  - I (object identity main effect)?
  - A (category main effect, residualized)?
  - epsilon (residual = interaction + value + noise)?
  - Full Δh?

Hypothesis (from Phase 384-385b analysis):
  If epsilon (interaction+value+noise) is more causally effective than A (pure category),
  then language encodes relations, not independent categories.

Method:
  Part 1: ANOVA decomposition of raw Δh
    - Grand mean: mu = mean(dh_raw)
    - Object centroids: c_obj[o] = mean(dh_raw | object=o)
    - I_component(p) = c_obj[p.object] - mu
    - Category centroids (residualized): c_cat[c] = mean(dh_raw - c_obj[p.object] | category=c)
    - A_component(p) = c_cat[p.category]
    - epsilon(p) = dh_raw(p) - I_component(p) - A_component(p) - mu
    - Full dh_raw = mu + I + A + epsilon

  Part 2: Raw-space causal test for each component
    - Add each component to corrupt h_raw -> forward -> measure logit_diff

  Part 3: RMSNorm Jacobian mapping
    - J = d(RMSNorm)/d(h_raw) at clean h_raw
    - Map post-RMSNorm category direction: u_raw_J = J^+ @ u_post
    - Compare: raw_probe vs J_mapped vs post_probe causal effects

  Part 4: Cross-model factor hierarchy comparison

Usage:
  python tests/glm5/phase386_factor_causal_hierarchy.py qwen3
  python tests/glm5/phase386_factor_causal_hierarchy.py deepseek7b
  python tests/glm5/phase386_factor_causal_hierarchy.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
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
        log(f"    Forward failed: {str(e)[:80]}")
        hook.remove()
        return None
    hook.remove()
    return float(logits[target_token_id] - logits[competitor_token_id])


def run_model_get_logits(model, tokenizer, device, prompt, target_token_id, competitor_token_id):
    """Run model without patch, return logit_diff."""
    if target_token_id < 0 or competitor_token_id < 0:
        return None
    with torch.no_grad():
        toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        out = model(
            input_ids=toks["input_ids"].to(device),
            attention_mask=toks["attention_mask"].to(device),
        )
        logits = out.logits[0, -1].float().cpu().numpy()
    return float(logits[target_token_id] - logits[competitor_token_id])


def anova_decomposition(dh_raw, object_labels, category_labels):
    """
    Two-way ANOVA decomposition of dh_raw.
    
    Returns: I_comp, A_comp, eps_comp, mu
      mu: grand mean
      I_comp[i]: object identity component for sample i
      A_comp[i]: category component for sample i (residualized after I)
      eps_comp[i]: residual = dh_raw - mu - I - A
    """
    n, d = dh_raw.shape
    mu = np.mean(dh_raw, axis=0)  # (d,)
    
    # Object centroids
    unique_objs = sorted(set(object_labels))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    c_obj = np.zeros((len(unique_objs), d))
    obj_counts = np.zeros(len(unique_objs))
    for i in range(n):
        oi = obj_to_idx[object_labels[i]]
        c_obj[oi] += dh_raw[i]
        obj_counts[oi] += 1
    for j in range(len(unique_objs)):
        if obj_counts[j] > 0:
            c_obj[j] /= obj_counts[j]
    
    # I_component: object centroid - grand mean
    I_comp = np.zeros_like(dh_raw)
    for i in range(n):
        oi = obj_to_idx[object_labels[i]]
        I_comp[i] = c_obj[oi] - mu
    
    # Residual after removing I
    dh_resid_I = dh_raw - mu - I_comp  # = dh_raw - c_obj[oi]
    
    # Category centroids (on I-residualized data)
    unique_cats = sorted(set(category_labels))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    c_cat = np.zeros((len(unique_cats), d))
    cat_counts = np.zeros(len(unique_cats))
    for i in range(n):
        ci = cat_to_idx[category_labels[i]]
        c_cat[ci] += dh_resid_I[i]
        cat_counts[ci] += 1
    for j in range(len(unique_cats)):
        if cat_counts[j] > 0:
            c_cat[j] /= cat_counts[j]
    
    # A_component: category centroid (on residualized data)
    A_comp = np.zeros_like(dh_raw)
    for i in range(n):
        ci = cat_to_idx[category_labels[i]]
        A_comp[i] = c_cat[ci]
    
    # Epsilon: residual after removing I and A
    eps_comp = dh_raw - mu - I_comp - A_comp
    
    # R² computation
    ss_total = np.sum((dh_raw - mu) ** 2)
    ss_I = np.sum(I_comp ** 2)
    ss_A = np.sum(A_comp ** 2)
    ss_eps = np.sum(eps_comp ** 2)
    
    r2_I = ss_I / ss_total if ss_total > 0 else 0
    r2_A = ss_A / ss_total if ss_total > 0 else 0
    r2_eps = ss_eps / ss_total if ss_total > 0 else 0
    
    return I_comp, A_comp, eps_comp, mu, {
        'r2_I': float(r2_I),
        'r2_A': float(r2_A),
        'r2_eps': float(r2_eps),
        'ss_total': float(ss_total),
        'ss_I': float(ss_I),
        'ss_A': float(ss_A),
        'ss_eps': float(ss_eps),
        'n_objs': len(unique_objs),
        'n_cats': len(unique_cats),
    }


def compute_rmsnorm_jacobian_pseudoinv(h_raw_vec, ln_weight, eps=1e-6):
    """
    Compute the pseudo-inverse of RMSNorm Jacobian.
    
    RMSNorm: y = g * x / sqrt(mean(x^2) + eps)
    Jacobian: J = (g/rms) * (I - x*x^T / (rms^2 * d))
    
    J has null space along x. Pseudo-inverse maps:
    u_raw = J^+ @ u_post = (rms/g) * (u_post - (u_post . x / ||x||^2) * x)
    
    Returns: function that maps u_post -> u_raw
    """
    d = len(h_raw_vec)
    rms = np.sqrt(np.mean(h_raw_vec ** 2) + eps)
    g = ln_weight  # (d,) vector
    
    # For component-wise: u_raw[i] = (rms/g[i]) * (u_post[i] - (u_post . x / ||x||^2) * x[i])
    # But g is a vector, so we need element-wise scaling
    # J = diag(g/rms) * (I - x*x^T/(rms^2*d))
    # J^+ = (I - x*x^T/||x||^2) * diag(rms/g)
    
    x_norm_sq = np.sum(h_raw_vec ** 2)
    
    def map_post_to_raw(u_post):
        """Map a post-RMSNorm direction to raw space via J^+."""
        # Project out the component along x
        proj = np.dot(u_post, h_raw_vec) / (x_norm_sq + 1e-12)
        u_orth = u_post - proj * h_raw_vec
        # Scale by rms/g (element-wise)
        u_raw = u_orth * (rms / (g + 1e-12))
        return u_raw
    
    return map_post_to_raw


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 386: Factor Decomposition Causal Hierarchy — {model_name}")
    log(f"=" * 70)
    
    # Target layers
    if model_name == "qwen3":
        target_layers = [4, 12, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 20, 30]
    elif model_name == "deepseek7b":
        target_layers = [4, 8, 12, 20, 24]
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    n_pairs = len(ALL_PAIRS)
    category_labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
    object_labels = [ALL_PAIRS[i][0] for i in range(n_pairs)]
    value_labels = [ALL_PAIRS[i][1] for i in range(n_pairs)]  # target attribute
    
    log(f"  Data: {n_pairs} pairs, {len(set(object_labels))} objects, "
        f"{len(set(category_labels))} categories, {len(set(value_labels))} values")
    
    # Object-category distribution
    obj_cat_pairs = set()
    for i in range(n_pairs):
        obj_cat_pairs.add((object_labels[i], category_labels[i]))
    log(f"  Object-category combinations: {len(obj_cat_pairs)}")
    
    # Objects appearing in multiple categories
    obj_cats = defaultdict(set)
    for i in range(n_pairs):
        obj_cats[object_labels[i]].add(category_labels[i])
    multi_cat_objs = sum(1 for o, cs in obj_cats.items() if len(cs) > 1)
    log(f"  Objects in multiple categories: {multi_cat_objs}/{len(obj_cats)}")
    
    results = {}
    
    for l in target_layers:
        log(f"\n{'='*70}")
        log(f"Layer {l}")
        log(f"{'='*70}")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # ===== Step 1: Collect all residual states =====
        log(f"  Step 1: Collecting residual states...")
        
        h_clean_raw = []
        h_corrupt_raw = []
        h_clean_norm = []
        h_corrupt_norm = []
        clean_logits_list = []
        corrupt_logits_list = []
        target_token_ids = []
        competitor_token_ids = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs} (layer {l})")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            t_ids = tokenizer.encode(target, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)
            
            # Clean
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
            h_clean_norm.append(rms_norm_single(h_raw_c, ln_weight))
            clean_logits_list.append(clean_out.logits[0, -1].float().cpu().numpy())
            del clean_out
            
            # Corrupt
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
            h_corrupt_norm.append(rms_norm_single(h_raw_r, ln_weight))
            corrupt_logits_list.append(corrupt_out.logits[0, -1].float().cpu().numpy())
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_clean_raw = np.array(h_clean_raw)
        h_corrupt_raw = np.array(h_corrupt_raw)
        h_clean_norm = np.array(h_clean_norm)
        h_corrupt_norm = np.array(h_corrupt_norm)
        
        # Compute differences
        dh_raw = h_clean_raw - h_corrupt_raw
        dh_norm = h_clean_norm - h_corrupt_norm
        
        # Baseline logit_diff
        baseline_clean_ld = []
        baseline_corrupt_ld = []
        for i in range(n_pairs):
            t_id, c_id = target_token_ids[i], competitor_token_ids[i]
            if t_id >= 0 and c_id >= 0:
                baseline_clean_ld.append(float(clean_logits_list[i][t_id] - clean_logits_list[i][c_id]))
                baseline_corrupt_ld.append(float(corrupt_logits_list[i][t_id] - corrupt_logits_list[i][c_id]))
            else:
                baseline_clean_ld.append(0.0)
                baseline_corrupt_ld.append(0.0)
        baseline_clean_ld = np.array(baseline_clean_ld)
        baseline_corrupt_ld = np.array(baseline_corrupt_ld)
        
        log(f"  Baseline: clean_ld={np.mean(baseline_clean_ld):.3f}±{np.std(baseline_clean_ld):.3f}, "
            f"corrupt_ld={np.mean(baseline_corrupt_ld):.3f}±{np.std(baseline_corrupt_ld):.3f}")
        
        # ===== Step 2: ANOVA Decomposition =====
        log(f"  Step 2: ANOVA decomposition...")
        
        I_comp, A_comp, eps_comp, mu, anova_stats = anova_decomposition(
            dh_raw, object_labels, category_labels
        )
        
        log(f"  ANOVA R²: I={anova_stats['r2_I']:.4f}, A={anova_stats['r2_A']:.4f}, "
            f"eps={anova_stats['r2_eps']:.4f}")
        
        # Component norms
        I_norms = np.linalg.norm(I_comp, axis=1)
        A_norms = np.linalg.norm(A_comp, axis=1)
        eps_norms = np.linalg.norm(eps_comp, axis=1)
        dh_norms = np.linalg.norm(dh_raw, axis=1)
        
        log(f"  Component norms: I={np.mean(I_norms):.3f}±{np.std(I_norms):.3f}, "
            f"A={np.mean(A_norms):.3f}±{np.std(A_norms):.3f}, "
            f"eps={np.mean(eps_norms):.3f}±{np.std(eps_norms):.3f}, "
            f"full={np.mean(dh_norms):.3f}±{np.std(dh_norms):.3f}")
        
        # ===== Step 3: Post-RMSNorm category probe + Jacobian mapping =====
        log(f"  Step 3: Post-RMSNorm probe + Jacobian mapping...")
        
        # Residualize dh_norm for category probe
        unique_objs = sorted(set(object_labels))
        obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
        obj_onehot = np.zeros((n_pairs, len(unique_objs)))
        for i in range(n_pairs):
            obj_onehot[i, obj_to_idx[object_labels[i]]] = 1.0
        if obj_onehot.shape[1] > 1:
            X_obj = obj_onehot[:, :-1]
        else:
            X_obj = obj_onehot
        X_design = np.column_stack([np.ones(n_pairs), X_obj])
        beta_obj = np.linalg.lstsq(X_design, dh_norm, rcond=None)[0]
        dh_norm_resid = dh_norm - X_design @ beta_obj
        
        # Post-RMSNorm category probe
        clf_post = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        clf_post.fit(dh_norm_resid, category_labels)
        W_post = clf_post.coef_  # (n_cats-1, d) or (n_cats, d)
        Q_post, _ = np.linalg.qr(W_post.T)  # (d, rank)
        cat_proj_post = (dh_norm_resid @ Q_post) @ Q_post.T
        
        # Raw-space category probe
        dh_raw_resid = dh_raw - X_design @ np.linalg.lstsq(X_design, dh_raw, rcond=None)[0]
        clf_raw = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        clf_raw.fit(dh_raw_resid, category_labels)
        W_raw = clf_raw.coef_
        Q_raw, _ = np.linalg.qr(W_raw.T)
        cat_proj_raw = (dh_raw_resid @ Q_raw) @ Q_raw.T
        
        # Jacobian mapping: for each sample, map post-RMSNorm direction to raw
        cat_proj_J = np.zeros_like(dh_raw)
        for i in range(n_pairs):
            map_fn = compute_rmsnorm_jacobian_pseudoinv(h_clean_raw[i], ln_weight)
            cat_proj_J[i] = map_fn(cat_proj_post[i])
        
        J_norms = np.linalg.norm(cat_proj_J, axis=1)
        post_norms = np.linalg.norm(cat_proj_post, axis=1)
        raw_probe_norms = np.linalg.norm(cat_proj_raw, axis=1)
        
        log(f"  Projection norms: post={np.mean(post_norms):.3f}±{np.std(post_norms):.3f}, "
            f"raw_probe={np.mean(raw_probe_norms):.3f}±{np.std(raw_probe_norms):.3f}, "
            f"J_mapped={np.mean(J_norms):.3f}±{np.std(J_norms):.3f}")
        
        # Cosine similarity between J_mapped and raw_probe
        cos_J_raw = []
        for i in range(n_pairs):
            n1 = np.linalg.norm(cat_proj_J[i])
            n2 = np.linalg.norm(cat_proj_raw[i])
            if n1 > 1e-8 and n2 > 1e-8:
                cos_J_raw.append(float(np.dot(cat_proj_J[i], cat_proj_raw[i]) / (n1 * n2)))
        if cos_J_raw:
            log(f"  Cosine(J_mapped, raw_probe): mean={np.mean(cos_J_raw):.4f}, "
                f"std={np.std(cos_J_raw):.4f}")
        
        # ===== Step 4: Causal Tests =====
        log(f"  Step 4: Causal tests (5 components: I, A, eps, full, J_mapped, raw_probe)...")
        
        n_test = n_pairs
        test_indices = list(range(n_test))
        
        # Components to test
        components = {
            'I': I_comp,
            'A': A_comp,
            'eps': eps_comp,
            'full': dh_raw,
            'J_mapped': cat_proj_J,
            'raw_probe': cat_proj_raw,
        }
        
        ca_results = {}
        for comp_name in components:
            ca_results[comp_name] = {'add': [], 'remove': []}
        
        for cnt, pidx in enumerate(test_indices):
            if cnt % 20 == 0:
                log(f"    Pair {cnt+1}/{n_test}")
            
            obj, target, competitor = ALL_PAIRS[pidx]
            t_id = target_token_ids[pidx]
            c_id = competitor_token_ids[pidx]
            if t_id < 0 or c_id < 0:
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            for comp_name, comp_vec in components.items():
                delta_add = comp_vec[pidx]
                
                # Add to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          delta_add, t_id, c_id)
                if ld is not None:
                    ca_results[comp_name]['add'].append(ld)
                else:
                    ca_results[comp_name]['add'].append(None)
                
                # Remove from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                          -delta_add, t_id, c_id)
                if ld is not None:
                    ca_results[comp_name]['remove'].append(ld)
                else:
                    ca_results[comp_name]['remove'].append(None)
            
            if cnt % 3 == 0:
                torch.cuda.empty_cache()
        
        # ===== Step 5: Compute Effects =====
        log(f"  Step 5: Computing effects...")
        
        layer_result = {
            "layer": l,
            "n_test": n_test,
            "anova": anova_stats,
            "component_norms": {
                "I": {"mean": float(np.mean(I_norms)), "std": float(np.std(I_norms))},
                "A": {"mean": float(np.mean(A_norms)), "std": float(np.std(A_norms))},
                "eps": {"mean": float(np.mean(eps_norms)), "std": float(np.std(eps_norms))},
                "full": {"mean": float(np.mean(dh_norms)), "std": float(np.std(dh_norms))},
                "J_mapped": {"mean": float(np.mean(J_norms)), "std": float(np.std(J_norms))},
                "raw_probe": {"mean": float(np.mean(raw_probe_norms)), "std": float(np.std(raw_probe_norms))},
            },
        }
        
        for comp_name in components:
            add_vals = [v for v in ca_results[comp_name]['add'] if v is not None]
            rem_vals = [v for v in ca_results[comp_name]['remove'] if v is not None]
            
            # Add effect: (patched_corrupt_ld - baseline_corrupt_ld)
            if len(add_vals) > 0:
                n_eff = min(len(add_vals), len(baseline_corrupt_ld))
                add_eff = np.array(add_vals[:n_eff]) - baseline_corrupt_ld[:n_eff]
                layer_result[f"{comp_name}_add"] = {
                    "mean": float(np.mean(add_eff)),
                    "std": float(np.std(add_eff)),
                    "t": float(np.mean(add_eff) / (np.std(add_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
            
            # Remove effect: (patched_clean_ld - baseline_clean_ld)
            if len(rem_vals) > 0:
                n_eff = min(len(rem_vals), len(baseline_clean_ld))
                rem_eff = np.array(rem_vals[:n_eff]) - baseline_clean_ld[:n_eff]
                layer_result[f"{comp_name}_remove"] = {
                    "mean": float(np.mean(rem_eff)),
                    "std": float(np.std(rem_eff)),
                    "t": float(np.mean(rem_eff) / (np.std(rem_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
        
        results[str(l)] = layer_result
        
        # Print summary
        log(f"\n  Layer {l} results:")
        log(f"  ANOVA: R²_I={anova_stats['r2_I']:.4f}, R²_A={anova_stats['r2_A']:.4f}, "
            f"R²_eps={anova_stats['r2_eps']:.4f}")
        log(f"  {'Component':12s} {'Add_mean':>10s} {'Add_t':>8s} {'Rem_mean':>10s} {'Rem_t':>8s}")
        log(f"  {'-'*52}")
        for comp_name in ['I', 'A', 'eps', 'full', 'J_mapped', 'raw_probe']:
            ae = layer_result.get(f"{comp_name}_add", {})
            re = layer_result.get(f"{comp_name}_remove", {})
            log(f"  {comp_name:12s} {ae.get('mean',0):+10.4f} {ae.get('t',0):8.2f} "
                f"{re.get('mean',0):+10.4f} {re.get('t',0):8.2f}")
        
        log(f"  Layer {l} done in {time.time()-t_l:.1f}s")
    
    # ===== Save Results =====
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
        "n_categories": len(set(category_labels)),
        "n_objects": len(set(object_labels)),
        "test": "phase386_factor_causal_hierarchy",
        "results": convert(results),
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase386.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # ===== Final Summary =====
    log(f"\n{'='*70}")
    log(f"Phase 386 Summary — {model_name}")
    log(f"{'='*70}")
    
    for l_str in sorted(results.keys(), key=int):
        r = results[l_str]
        anova = r.get('anova', {})
        log(f"\nLayer {l_str}: R²_I={anova.get('r2_I',0):.4f}, R²_A={anova.get('r2_A',0):.4f}, "
            f"R²_eps={anova.get('r2_eps',0):.4f}")
        for comp_name in ['I', 'A', 'eps', 'full', 'J_mapped', 'raw_probe']:
            ae = r.get(f"{comp_name}_add", {})
            re = r.get(f"{comp_name}_remove", {})
            log(f"  {comp_name:12s} add={ae.get('mean',0):+.4f}(t={ae.get('t',0):.2f}), "
                f"rem={re.get('mean',0):+.4f}(t={re.get('t',0):.2f})")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 386 complete for {model_name}!")


if __name__ == "__main__":
    main()
