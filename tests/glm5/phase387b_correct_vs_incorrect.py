"""
Phase 387b: Correct Value vs Incorrect Value — Conditional Centroid Causal Test
================================================================================

Key finding from Phase 387:
  The three-way ANOVA is INVALID because:
  - Only 48/192 cells filled (unbalanced design)
  - R² sums to >1.0 (effects are NOT orthogonal)
  - This makes interaction estimates unreliable

  However, Phase 387 revealed a critical OBSERVATION:
  - correct vs incorrect value conditions produce different Δh patterns
  - This V factor (value correctness) was never tested in Phase 386

Method:
  Use Phase 386's proven two-way ANOVA (I+A+ε) approach, but separately for:
  1. correct-value pairs: Δh_correct = h_clean(correct) - h_corrupt
  2. incorrect-value pairs: Δh_incorrect = h_clean(incorrect) - h_corrupt
  
  Then compare:
  - A_correct vs A_incorrect: does category centroid differ by value correctness?
  - I_correct vs I_incorrect: does object identity differ by value correctness?
  - full_correct vs full_incorrect: does complete binding signal differ?

  Also test: swapping correct↔incorrect centroids
  - Add A_incorrect to a correct-value corrupt → should reduce output
  - Add A_correct to an incorrect-value corrupt → should improve output

Data: Same as Phase 387 (12 objects × 2 categories × 2 values = 48 stimuli)
  24 correct + 24 incorrect

Usage:
  python tests/glm5/phase387b_correct_vs_incorrect.py qwen3
  python tests/glm5/phase387b_correct_vs_incorrect.py deepseek7b
  python tests/glm5/phase387b_correct_vs_incorrect.py glm4
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


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Data (same as Phase 387) =====
MULTI_CATEGORY_OBJECTS = {
    "apple": {"color": ["red", "blue"], "taste": ["sweet", "sour"]},
    "snow": {"color": ["white", "black"], "temperature": ["cold", "hot"]},
    "fire": {"temperature": ["hot", "cold"], "brightness": ["bright", "dark"]},
    "ocean": {"color": ["blue", "red"], "moisture": ["wet", "dry"]},
    "desert": {"temperature": ["hot", "cold"], "moisture": ["dry", "wet"]},
    "elephant": {"size": ["big", "small"], "weight": ["heavy", "light"]},
    "feather": {"weight": ["light", "heavy"], "size": ["small", "big"]},
    "coal": {"color": ["black", "white"], "temperature": ["hot", "cold"]},
    "cheetah": {"speed": ["fast", "slow"], "size": ["big", "small"]},
    "star": {"brightness": ["bright", "dark"], "size": ["big", "small"]},
    "cloud": {"color": ["white", "black"], "weight": ["light", "heavy"]},
    "lava": {"temperature": ["hot", "cold"], "brightness": ["bright", "dark"]},
}

TEMPLATE = "The {obj} is {attr}."
CORRUPTED_BASELINE = "The item"

STIMULI = []
for obj, cats in MULTI_CATEGORY_OBJECTS.items():
    for cat, values in cats.items():
        correct_v = values[0]
        for inc_v in values[1:]:
            STIMULI.append({"object": obj, "category": cat, "value": correct_v,
                           "value_type": "correct", "competitor": inc_v})
            STIMULI.append({"object": obj, "category": cat, "value": inc_v,
                           "value_type": "incorrect", "competitor": correct_v})


# ===== Model Loading =====
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2",
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True, attn_implementation="eager",
        )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  {model_name} loaded: GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def _load_ln_weight(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    for attr_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
        ln = getattr(layers[layer_idx], attr_name, None)
        if ln is not None:
            try:
                w = ln.weight.detach().cpu().float().numpy()
                if w is not None and len(w) > 0:
                    return w
            except:
                pass
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    for ln_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
                        if f"layers.{layer_idx}.{ln_name}.weight" in key:
                            return sf.get_tensor(key).float().numpy()
        except:
            continue
    return None


def run_model_with_patch(model, tokenizer, device, prompt, layer_idx,
                         patch_delta, target_token_id, competitor_token_id):
    if target_token_id < 0 or competitor_token_id < 0:
        return None
    layers = get_layers(model)
    delta_tensor = torch.tensor(patch_delta, dtype=torch.bfloat16, device=device)
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h_patched = h.clone()
        h_patched[0, -1, :] += delta_tensor
        return (h_patched,) + output[1:] if isinstance(output, tuple) else h_patched
    hook = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            out = model(input_ids=toks["input_ids"].to(device), attention_mask=toks["attention_mask"].to(device))
            logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        log(f"    Forward failed: {str(e)[:60]}")
        hook.remove()
        return None
    hook.remove()
    return float(logits[target_token_id] - logits[competitor_token_id])


# ===== Two-way ANOVA (proven from Phase 386) =====
def two_way_anova(dh_raw, object_labels, category_labels):
    n, d = dh_raw.shape
    mu = np.mean(dh_raw, axis=0)
    
    unique_objs = sorted(set(object_labels))
    unique_cats = sorted(set(category_labels))
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    
    # Object centroids
    c_obj = np.zeros((len(unique_objs), d))
    obj_counts = np.zeros(len(unique_objs))
    for i in range(n):
        oi = obj_to_idx[object_labels[i]]
        c_obj[oi] += dh_raw[i]
        obj_counts[oi] += 1
    for j in range(len(unique_objs)):
        if obj_counts[j] > 0:
            c_obj[j] /= obj_counts[j]
    
    I_comp = np.zeros_like(dh_raw)
    for i in range(n):
        I_comp[i] = c_obj[obj_to_idx[object_labels[i]]] - mu
    
    dh_resid_I = dh_raw - mu - I_comp
    
    # Category centroids (residualized)
    c_cat = np.zeros((len(unique_cats), d))
    cat_counts = np.zeros(len(unique_cats))
    for i in range(n):
        ci = cat_to_idx[category_labels[i]]
        c_cat[ci] += dh_resid_I[i]
        cat_counts[ci] += 1
    for j in range(len(unique_cats)):
        if cat_counts[j] > 0:
            c_cat[j] /= cat_counts[j]
    
    A_comp = np.zeros_like(dh_raw)
    for i in range(n):
        A_comp[i] = c_cat[cat_to_idx[category_labels[i]]]
    
    eps_comp = dh_raw - mu - I_comp - A_comp
    
    ss_total = np.sum((dh_raw - mu) ** 2)
    r2_I = np.sum(I_comp ** 2) / ss_total if ss_total > 0 else 0
    r2_A = np.sum(A_comp ** 2) / ss_total if ss_total > 0 else 0
    r2_eps = np.sum(eps_comp ** 2) / ss_total if ss_total > 0 else 0
    
    return I_comp, A_comp, eps_comp, mu, {
        'r2_I': float(r2_I), 'r2_A': float(r2_A), 'r2_eps': float(r2_eps),
        'n_obj': len(unique_objs), 'n_cat': len(unique_cats),
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 387b: Correct vs Incorrect Value Conditional Test — {model_name}")
    log(f"=" * 70)
    
    n_stim = len(STIMULI)
    correct_stim = [s for s in STIMULI if s["value_type"] == "correct"]
    incorrect_stim = [s for s in STIMULI if s["value_type"] == "incorrect"]
    log(f"Stimuli: {n_stim} total ({len(correct_stim)} correct, {len(incorrect_stim)} incorrect)")
    
    if model_name == "qwen3":
        target_layers = [4, 12, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 20, 30]
    elif model_name == "deepseek7b":
        target_layers = [4, 8, 12, 20, 24]
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    results = {}
    
    for l in target_layers:
        log(f"\n{'='*70}")
        log(f"Layer {l}")
        log(f"{'='*70}")
        t_l = time.time()
        
        # ===== Step 1: Collect all residual states =====
        h_clean_raw = []
        h_corrupt_raw = []
        clean_logits_list = []
        corrupt_logits_list = []
        target_token_ids = []
        competitor_token_ids = []
        valid_indices = []
        
        for sidx, stim in enumerate(STIMULI):
            if sidx % 10 == 0:
                log(f"    Stimulus {sidx+1}/{n_stim}")
            
            obj, value, competitor = stim["object"], stim["value"], stim["competitor"]
            clean_prompt = TEMPLATE.format(obj=obj, attr=value)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=value)
            
            t_ids = tokenizer.encode(value, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            if t_id < 0 or c_id < 0:
                continue
            
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)
            valid_indices.append(sidx)
            
            with torch.no_grad():
                toks = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
                out = model(input_ids=toks["input_ids"].to(device), attention_mask=toks["attention_mask"].to(device), output_hidden_states=True)
            last_pos = toks["input_ids"].shape[1] - 1
            h_clean_raw.append(out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy())
            clean_logits_list.append(out.logits[0, -1].float().cpu().numpy())
            del out
            
            with torch.no_grad():
                toks = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
                out = model(input_ids=toks["input_ids"].to(device), attention_mask=toks["attention_mask"].to(device), output_hidden_states=True)
            last_pos_r = toks["input_ids"].shape[1] - 1
            h_corrupt_raw.append(out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            corrupt_logits_list.append(out.logits[0, -1].float().cpu().numpy())
            del out
            
            if sidx % 3 == 0:
                torch.cuda.empty_cache()
        
        h_clean_raw = np.array(h_clean_raw)
        h_corrupt_raw = np.array(h_corrupt_raw)
        dh_raw = h_clean_raw - h_corrupt_raw
        
        n_valid = len(valid_indices)
        valid_stim = [STIMULI[i] for i in valid_indices]
        obj_labels = [s["object"] for s in valid_stim]
        cat_labels = [s["category"] for s in valid_stim]
        vtype_labels = [s["value_type"] for s in valid_stim]
        
        # Baseline logit_diff
        baseline_clean_ld = np.array([
            float(clean_logits_list[i][target_token_ids[i]] - clean_logits_list[i][competitor_token_ids[i]])
            for i in range(n_valid)
        ])
        baseline_corrupt_ld = np.array([
            float(corrupt_logits_list[i][target_token_ids[i]] - corrupt_logits_list[i][competitor_token_ids[i]])
            for i in range(n_valid)
        ])
        
        log(f"  Baseline: clean_ld={np.mean(baseline_clean_ld):.3f}, corrupt_ld={np.mean(baseline_corrupt_ld):.3f}")
        
        # ===== Step 2: Separate ANOVA for correct vs incorrect =====
        correct_idx = [i for i in range(n_valid) if vtype_labels[i] == "correct"]
        incorrect_idx = [i for i in range(n_valid) if vtype_labels[i] == "incorrect"]
        
        log(f"  Correct: {len(correct_idx)}, Incorrect: {len(incorrect_idx)}")
        
        # All-data ANOVA
        I_all, A_all, eps_all, mu_all, stats_all = two_way_anova(dh_raw, obj_labels, cat_labels)
        
        # Correct-only ANOVA
        dh_correct = dh_raw[correct_idx]
        obj_correct = [obj_labels[i] for i in correct_idx]
        cat_correct = [cat_labels[i] for i in correct_idx]
        I_corr, A_corr, eps_corr, mu_corr, stats_corr = two_way_anova(dh_correct, obj_correct, cat_correct)
        
        # Incorrect-only ANOVA
        dh_incorrect = dh_raw[incorrect_idx]
        obj_incorrect = [obj_labels[i] for i in incorrect_idx]
        cat_incorrect = [cat_labels[i] for i in incorrect_idx]
        I_inc, A_inc, eps_inc, mu_inc, stats_inc = two_way_anova(dh_incorrect, obj_incorrect, cat_incorrect)
        
        log(f"  R²_all: I={stats_all['r2_I']:.4f}, A={stats_all['r2_A']:.4f}, eps={stats_all['r2_eps']:.4f}")
        log(f"  R²_correct: I={stats_corr['r2_I']:.4f}, A={stats_corr['r2_A']:.4f}, eps={stats_corr['r2_eps']:.4f}")
        log(f"  R²_incorrect: I={stats_inc['r2_I']:.4f}, A={stats_inc['r2_A']:.4f}, eps={stats_inc['r2_eps']:.4f}")
        
        # ===== Step 3: Causal Tests =====
        log(f"  Step 3: Causal tests...")
        
        # Components to test for each sample (matched to its condition)
        comp_names = ['I', 'A', 'eps', 'full']
        
        # Build component vectors for each sample using matched-condition ANOVA
        comp_vecs = {}
        for cn in comp_names:
            comp_vecs[cn] = np.zeros_like(dh_raw)
        
        for i in range(n_valid):
            local_i = None
            if vtype_labels[i] == "correct":
                local_i = correct_idx.index(i)
                comp_vecs['I'][i] = I_corr[local_i]
                comp_vecs['A'][i] = A_corr[local_i]
                comp_vecs['eps'][i] = eps_corr[local_i]
            else:
                local_i = incorrect_idx.index(i)
                comp_vecs['I'][i] = I_inc[local_i]
                comp_vecs['A'][i] = A_inc[local_i]
                comp_vecs['eps'][i] = eps_inc[local_i]
            comp_vecs['full'][i] = dh_raw[i]
        
        # Also test cross-condition: add incorrect centroid to correct corrupt
        # This tests whether "wrong category direction" hurts
        # A_incorrect applied to correct-value samples
        A_cross_wrong = np.zeros_like(dh_raw)
        for i in range(n_valid):
            if vtype_labels[i] == "correct":
                # For correct samples, apply incorrect centroid of same category
                local_i = correct_idx.index(i)
                cat = cat_labels[i]
                # Find the A_inc for this category
                unique_cats_inc = sorted(set(cat_incorrect))
                cat_idx_inc = unique_cats_inc.index(cat) if cat in unique_cats_inc else -1
                if cat_idx_inc >= 0:
                    A_cross_wrong[i] = A_inc[cat_idx_inc]  # This is the category centroid from incorrect condition
                else:
                    A_cross_wrong[i] = A_corr[local_i]  # fallback
            else:
                # For incorrect samples, apply correct centroid
                local_i = incorrect_idx.index(i)
                cat = cat_labels[i]
                unique_cats_corr = sorted(set(cat_correct))
                cat_idx_corr = unique_cats_corr.index(cat) if cat in unique_cats_corr else -1
                if cat_idx_corr >= 0:
                    A_cross_wrong[i] = A_corr[cat_idx_corr]
                else:
                    A_cross_wrong[i] = A_inc[local_i]
        
        # Actually, let me simplify: use the A centroid from the OTHER condition
        # For each sample, its cross-A is the centroid of the same category but from opposite value_type
        # We need to map categories to centroids in each condition
        
        # Build category centroid maps for correct and incorrect
        unique_cats_c = sorted(set(cat_correct))
        unique_cats_i = sorted(set(cat_incorrect))
        
        # Centroid maps
        cat_centroid_correct = {}
        for ci, cat in enumerate(unique_cats_c):
            cat_centroid_correct[cat] = A_corr[ci]  # This is wrong - A_comp uses index mapping
        
        # Let me redo this properly: compute category centroids directly
        cat_centroid_correct_direct = {}
        for cat in unique_cats_c:
            mask = [j for j in range(len(correct_idx)) if cat_correct[j] == cat]
            if mask:
                cat_centroid_correct_direct[cat] = np.mean(dh_correct[mask], axis=0) - mu_corr
        
        cat_centroid_incorrect_direct = {}
        for cat in unique_cats_i:
            mask = [j for j in range(len(incorrect_idx)) if cat_incorrect[j] == cat]
            if mask:
                cat_centroid_incorrect_direct[cat] = np.mean(dh_incorrect[mask], axis=0) - mu_inc
        
        # Build cross-condition A vectors
        A_cross = np.zeros_like(dh_raw)
        for i in range(n_valid):
            cat = cat_labels[i]
            if vtype_labels[i] == "correct" and cat in cat_centroid_incorrect_direct:
                A_cross[i] = cat_centroid_incorrect_direct[cat]
            elif vtype_labels[i] == "incorrect" and cat in cat_centroid_correct_direct:
                A_cross[i] = cat_centroid_correct_direct[cat]
            else:
                A_cross[i] = comp_vecs['A'][i]
        
        all_comp_names = comp_names + ['A_cross']
        comp_vecs['A_cross'] = A_cross
        
        # Run causal tests
        ca_results = {cn: {'add': [], 'remove': []} for cn in all_comp_names}
        
        for cnt in range(n_valid):
            if cnt % 10 == 0:
                log(f"    Sample {cnt+1}/{n_valid}")
            
            stim = valid_stim[cnt]
            obj, value = stim["object"], stim["value"]
            t_id, c_id = target_token_ids[cnt], competitor_token_ids[cnt]
            if t_id < 0 or c_id < 0:
                for cn in all_comp_names:
                    ca_results[cn]['add'].append(None)
                    ca_results[cn]['remove'].append(None)
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=value)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=value)
            
            for cn in all_comp_names:
                delta = comp_vecs[cn][cnt]
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l, delta, t_id, c_id)
                ca_results[cn]['add'].append(ld)
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l, -delta, t_id, c_id)
                ca_results[cn]['remove'].append(ld)
            
            if cnt % 2 == 0:
                torch.cuda.empty_cache()
        
        # ===== Step 4: Compute Effects =====
        log(f"  Step 4: Computing effects...")
        
        layer_result = {"layer": l, "n_valid": n_valid,
                       "stats_all": stats_all, "stats_correct": stats_corr, "stats_incorrect": stats_inc}
        
        for cn in all_comp_names:
            add_vals = [v for v in ca_results[cn]['add'] if v is not None]
            rem_vals = [v for v in ca_results[cn]['remove'] if v is not None]
            
            if add_vals:
                n_eff = min(len(add_vals), len(baseline_corrupt_ld))
                add_eff = np.array(add_vals[:n_eff]) - baseline_corrupt_ld[:n_eff]
                layer_result[f"{cn}_add"] = {
                    "mean": float(np.mean(add_eff)), "std": float(np.std(add_eff)),
                    "t": float(np.mean(add_eff) / (np.std(add_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
            if rem_vals:
                n_eff = min(len(rem_vals), len(baseline_clean_ld))
                rem_eff = baseline_clean_ld[:n_eff] - np.array(rem_vals[:n_eff])
                layer_result[f"{cn}_remove"] = {
                    "mean": float(np.mean(rem_eff)), "std": float(np.std(rem_eff)),
                    "t": float(np.mean(rem_eff) / (np.std(rem_eff) / np.sqrt(n_eff) + 1e-10)),
                    "n": n_eff,
                }
        
        # Also compute effects separately for correct and incorrect subsets
        for vtype, idx_list in [("correct", correct_idx), ("incorrect", incorrect_idx)]:
            for cn in comp_names:
                add_key = f"{cn}_add"
                rem_key = f"{cn}_remove"
                if add_key in ca_results:
                    vals = [ca_results[cn]['add'][i] for i in idx_list if i < len(ca_results[cn]['add']) and ca_results[cn]['add'][i] is not None]
                    if vals:
                        base = baseline_corrupt_ld[idx_list[:len(vals)]]
                        eff = np.array(vals) - base
                        layer_result[f"{cn}_{vtype}_add"] = {
                            "mean": float(np.mean(eff)), "std": float(np.std(eff)),
                            "t": float(np.mean(eff) / (np.std(eff) / np.sqrt(len(eff)) + 1e-10)),
                        }
        
        # Print summary
        log(f"\n  Layer {l} Summary:")
        log(f"  {'Comp':>8s} | {'Add mean':>10s} {'t':>8s} | {'Rem mean':>10s} {'t':>8s}")
        log(f"  {'-'*55}")
        for cn in all_comp_names:
            ai = layer_result.get(f"{cn}_add", {})
            ri = layer_result.get(f"{cn}_remove", {})
            a_str = f"{ai.get('mean', 0):+.4f} {ai.get('t', 0):+6.2f}" if ai else "    N/A"
            r_str = f"{ri.get('mean', 0):+.4f} {ri.get('t', 0):+6.2f}" if ri else "    N/A"
            log(f"  {cn:>8s} | {a_str} | {r_str}")
        
        # Separate correct/incorrect
        log(f"\n  Separate by value_type:")
        log(f"  {'Comp':>8s} | {'Correct add':>20s} | {'Incorrect add':>20s}")
        log(f"  {'-'*60}")
        for cn in comp_names:
            ci = layer_result.get(f"{cn}_correct_add", {})
            ii = layer_result.get(f"{cn}_incorrect_add", {})
            c_str = f"{ci.get('mean', 0):+.4f} t={ci.get('t', 0):+5.2f}" if ci else "N/A"
            i_str = f"{ii.get('mean', 0):+.4f} t={ii.get('t', 0):+5.2f}" if ii else "N/A"
            log(f"  {cn:>8s} | {c_str:>20s} | {i_str:>20s}")
        
        results[f"L{l}"] = layer_result
        log(f"  Layer {l} done in {time.time()-t_l:.1f}s")
    
    # Save
    os.makedirs("results/phase387b_correct_vs_incorrect", exist_ok=True)
    out_path = f"results/phase387b_correct_vs_incorrect/{model_name}_phase387b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"\nPhase 387b complete for {model_name}!")


if __name__ == "__main__":
    main()
