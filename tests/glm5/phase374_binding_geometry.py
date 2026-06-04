"""
Phase 374: Binding Geometry — Algebraic Structure of Binding Signals
====================================================================

Phase 370-373 analyzed the 1D collapse mechanism in DS7B. Now we shift to the
fundamental question: What is the UNIVERSAL mathematical structure of binding
across models?

Key hypotheses to test:
1. Factorization: Δh(obj, attr) ≈ f(obj) + g(attr) + interaction
2. Additivity: Δh(obj1, attr1) - Δh(obj1, attr2) ≈ Δh(obj2, attr1) - Δh(obj2, attr2)
   (The "attribute effect" should be independent of the object)
3. Binding subspace: Do all binding Δh vectors live in a low-dimensional subspace?
4. Cross-model consistency: Is the binding geometry similar across DS7B/Qwen3/GLM4?

We collect Δh for all 84 binding pairs and analyze the geometric structure.

Models: deepseek7b, qwen3, glm4
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

# Color binding pairs (obj, correct_attr, wrong_attr)
COLOR_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
]

# Temperature binding pairs
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
]

# Moisture binding pairs
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
]

# Texture binding pairs
TEXTURE_PAIRS = [
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]

# Size pairs
SIZE_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
]

# Weight pairs
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
]

# Speed pairs
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
]

# Brightness pairs
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
]

ALL_PAIRS = COLOR_PAIRS + TEMP_PAIRS + MOISTURE_PAIRS + TEXTURE_PAIRS + SIZE_PAIRS + WEIGHT_PAIRS + SPEED_PAIRS + BRIGHT_PAIRS

# Category labels for each pair
PAIR_CATEGORIES = (
    ["color"] * len(COLOR_PAIRS) +
    ["temperature"] * len(TEMP_PAIRS) +
    ["moisture"] * len(MOISTURE_PAIRS) +
    ["texture"] * len(TEXTURE_PAIRS) +
    ["size"] * len(SIZE_PAIRS) +
    ["weight"] * len(WEIGHT_PAIRS) +
    ["speed"] * len(SPEED_PAIRS) +
    ["brightness"] * len(BRIGHT_PAIRS)
)

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded with attn_impl={impl}")
            break
        except:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def collect_binding_vectors(model, tokenizer, device, model_name, target_layers, n_layers, d_model):
    """
    Collect Δh vectors for all binding pairs at target layers.
    
    Also collect:
    - Δh for wrong_attr (competitor) to test attribute specificity
    - Corrupt baseline for subtraction
    """
    log("\n--- Collecting binding Δh vectors ---")
    
    n_pairs = len(ALL_PAIRS)
    input_device = next(model.parameters()).device
    
    results = {}
    
    for l in target_layers:
        log(f"\n  Layer {l}:")
        
        dh_correct_list = []   # Δh for correct binding
        dh_wrong_list = []     # Δh for wrong binding (competitor)
        
        # Also collect h for corrupted baseline
        h_corrupt_cache = {}
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 20 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            wrong_prompt = TEMPLATE.format(obj=obj, attr=competitor)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            wrong_inputs = tokenizer(wrong_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_inputs["input_ids"].to(input_device),
                    attention_mask=clean_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
                wrong_out = model(
                    input_ids=wrong_inputs["input_ids"].to(input_device),
                    attention_mask=wrong_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
                corrupt_out = model(
                    input_ids=corrupt_inputs["input_ids"].to(input_device),
                    attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_c = clean_inputs["input_ids"].shape[1] - 1
            last_pos_w = wrong_inputs["input_ids"].shape[1] - 1
            last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1
            
            h_clean = clean_out.hidden_states[l+1][0, last_pos_c].detach().cpu().float().numpy()
            h_wrong = wrong_out.hidden_states[l+1][0, last_pos_w].detach().cpu().float().numpy()
            h_corrupt = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            
            dh_correct_list.append(h_clean - h_corrupt)
            dh_wrong_list.append(h_wrong - h_corrupt)
            
            del clean_out, wrong_out, corrupt_out
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        dh_correct = np.array(dh_correct_list)  # (n_pairs, d_model)
        dh_wrong = np.array(dh_wrong_list)        # (n_pairs, d_model)
        
        # ===== Analysis 1: PCA structure of binding subspace =====
        log(f"\n    === Analysis 1: Binding subspace PCA ===")
        
        M_centered = dh_correct - dh_correct.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        total_var = np.sum(S**2)
        explained = (S**2) / total_var
        
        # Effective rank
        eff_rank_95 = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
        eff_rank_99 = int(np.searchsorted(np.cumsum(explained), 0.99) + 1)
        
        log(f"    PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, PC5={explained[4]:.4f}")
        log(f"    eff_rank_95={eff_rank_95}, eff_rank_99={eff_rank_99}")
        
        # How many PCs to explain different variance levels
        for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
            n_pc = int(np.searchsorted(np.cumsum(explained), thresh) + 1)
            log(f"    {thresh*100:.0f}% variance: {n_pc} PCs")
        
        # ===== Analysis 2: Attribute effect independence test =====
        log(f"\n    === Analysis 2: Attribute effect independence ===")
        
        # For pairs sharing the same attribute category, test if:
        # Δh(obj1, attr1) - Δh(obj1, attr2) ≈ Δh(obj2, attr1) - Δh(obj2, attr2)
        # This tests if the "attribute difference" is independent of the object
        
        # Use color pairs: all have (obj, correct_color, wrong_color)
        # Compare: Δh(apple, red) - Δh(apple, blue) vs Δh(cherry, red) - Δh(cherry, blue)
        
        attr_independence_errors = []
        
        # For each category with enough pairs
        for cat_name, cat_pairs in [("color", COLOR_PAIRS), ("temperature", TEMP_PAIRS)]:
            cat_indices = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat_name]
            if len(cat_indices) < 4:
                continue
            
            # For each pair of objects in this category, compare attribute effects
            cat_dh_correct = dh_correct[cat_indices]  # (n_cat, d_model)
            cat_dh_wrong = dh_wrong[cat_indices]        # (n_cat, d_model)
            
            # Attribute effect for each object: correct - wrong
            attr_effects = cat_dh_correct - cat_dh_wrong  # (n_cat, d_model)
            
            # Pairwise comparison of attribute effects
            cos_sims = []
            for i in range(len(attr_effects)):
                for j in range(i+1, len(attr_effects)):
                    cos_sim = np.dot(attr_effects[i], attr_effects[j]) / (
                        np.linalg.norm(attr_effects[i]) * np.linalg.norm(attr_effects[j]) + 1e-10)
                    cos_sims.append(cos_sim)
            
            mean_cos = np.mean(cos_sims) if cos_sims else 0
            log(f"    {cat_name}: attribute effect pairwise cos = {mean_cos:.4f} (n_pairs={len(cos_sims)})")
            attr_independence_errors.append({
                "category": cat_name,
                "mean_cos_attr_effects": float(mean_cos),
                "n_comparisons": len(cos_sims),
            })
        
        # ===== Analysis 3: Factorization test =====
        log(f"\n    === Analysis 3: Factorization test ===")
        
        # Test: Δh(obj, attr) ≈ f(obj) + g(attr) + bias
        # If factorizable, then:
        # Δh(obj1, attr1) - Δh(obj1, attr2) = g(attr1) - g(attr2) (independent of obj)
        # Δh(obj1, attr1) - Δh(obj2, attr1) = f(obj1) - f(obj2) (independent of attr)
        
        # Use color pairs where we have multiple objects with same attributes
        # Test 1: Row differences should be constant
        # Test 2: Column differences should be constant
        
        # Create a matrix for color pairs: rows=objects, cols=attributes
        # We need pairs that share attributes. Use a subset.
        
        # Simple factorization test using ANOVA-like decomposition
        # Group by category and test variance explained by object vs attribute
        
        # For color pairs, group objects by their correct attribute
        from collections import defaultdict
        attr_groups = defaultdict(list)
        obj_groups = defaultdict(list)
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            cat = PAIR_CATEGORIES[pidx]
            if cat == "color":
                attr_groups[target].append(pidx)
                obj_groups[obj].append(pidx)
        
        # Test: For same attribute, do different objects produce similar Δh?
        if len(attr_groups) >= 2:
            within_attr_cos = []
            for attr, indices in attr_groups.items():
                if len(indices) >= 2:
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            v1 = dh_correct[indices[i]]
                            v2 = dh_correct[indices[j]]
                            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                            within_attr_cos.append(cos)
            
            mean_within = np.mean(within_attr_cos) if within_attr_cos else 0
            log(f"    Same-attribute pairwise cos: {mean_within:.4f} (n={len(within_attr_cos)})")
        
        # Test: For same object, do different attributes produce similar Δh?
        if len(obj_groups) >= 2:
            within_obj_cos = []
            for obj, indices in obj_groups.items():
                if len(indices) >= 2:
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            v1 = dh_correct[indices[i]]
                            v2 = dh_correct[indices[j]]
                            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                            within_obj_cos.append(cos)
            
            mean_within_obj = np.mean(within_obj_cos) if within_obj_cos else 0
            log(f"    Same-object pairwise cos: {mean_within_obj:.4f} (n={len(within_obj_cos)})")
        
        # ===== Analysis 4: Category clustering in binding subspace =====
        log(f"\n    === Analysis 4: Category clustering ===")
        
        # Project Δh onto top PCs and check if categories cluster
        n_proj = min(10, Vt.shape[0])
        projections = M_centered @ Vt[:n_proj].T  # (n_pairs, n_proj)
        
        # For each pair of categories, compute average pairwise cosine
        categories = list(set(PAIR_CATEGORIES))
        category_cos = {}
        
        for i, cat1 in enumerate(categories):
            idx1 = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat1]
            cent1 = dh_correct[idx1].mean(axis=0)
            
            for j, cat2 in enumerate(categories):
                if j <= i:
                    continue
                idx2 = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat2]
                cent2 = dh_correct[idx2].mean(axis=0)
                
                cos = np.dot(cent1, cent2) / (np.linalg.norm(cent1) * np.linalg.norm(cent2) + 1e-10)
                category_cos[f"{cat1}_vs_{cat2}"] = float(cos)
        
        # Within-category cosine
        within_cat_cos = {}
        for cat in categories:
            idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
            if len(idx) >= 2:
                cat_vecs = dh_correct[idx]
                cent = cat_vecs.mean(axis=0)
                cos_vals = [np.dot(v, cent) / (np.linalg.norm(v) * np.linalg.norm(cent) + 1e-10) 
                           for v in cat_vecs]
                within_cat_cos[cat] = float(np.mean(cos_vals))
        
        log(f"    Within-category cos (avg): {np.mean(list(within_cat_cos.values())):.4f}")
        for cat, cos in sorted(within_cat_cos.items(), key=lambda x: -x[1]):
            log(f"      {cat}: {cos:.4f}")
        
        # ===== Analysis 5: Correct vs Wrong binding geometry =====
        log(f"\n    === Analysis 5: Correct vs Wrong binding ===")
        
        # Compare Δh_correct and Δh_wrong
        # If binding is about object-attribute association, then:
        # - Δh_correct should encode the true binding
        # - Δh_wrong should encode a different (wrong) binding
        
        # Compute PCA of wrong binding Δh
        M_wrong_centered = dh_wrong - dh_wrong.mean(axis=0, keepdims=True)
        _, S_w, Vt_w = np.linalg.svd(M_wrong_centered, full_matrices=False)
        total_var_w = np.sum(S_w**2)
        explained_w = (S_w**2) / total_var_w
        
        log(f"    Δh_correct PC1={explained[0]:.4f}, eff_rank_95={eff_rank_95}")
        log(f"    Δh_wrong PC1={explained_w[0]:.4f}, eff_rank_95_w={int(np.searchsorted(np.cumsum(explained_w), 0.95)+1)}")
        
        # Alignment between correct and wrong binding subspaces
        # Do the top PCs of correct and wrong binding align?
        n_align = min(5, Vt.shape[0], Vt_w.shape[0])
        cos_pc_subspace = []
        for k in range(n_align):
            cos_k = abs(np.dot(Vt[k], Vt_w[k]))
            cos_pc_subspace.append(float(cos_k))
        
        log(f"    |cos(PC_correct_k, PC_wrong_k)| for k=1..5: {[f'{c:.3f}' for c in cos_pc_subspace]}")
        
        # ===== Analysis 6: Algebraic structure — additive decomposition =====
        log(f"\n    === Analysis 6: Additive decomposition ===")
        
        # Test: Δh(obj, attr) ≈ mean + f(obj) + g(attr)
        # If additive, then:
        # Δh(obj, attr) - mean(Δh for same obj) ≈ g(attr) - mean(g(attr))
        # Δh(obj, attr) - mean(Δh for same attr) ≈ f(obj) - mean(f(obj))
        
        # Use color pairs for this test
        color_indices = [i for i, c in enumerate(PAIR_CATEGORIES) if c == "color"]
        
        if len(color_indices) >= 10:
            # Compute mean Δh
            mean_dh = dh_correct.mean(axis=0)
            
            # Object factor: average Δh for each object minus global mean
            obj_factors = {}
            for pidx in color_indices:
                obj = ALL_PAIRS[pidx][0]
                if obj not in obj_factors:
                    obj_factors[obj] = []
                obj_factors[obj].append(dh_correct[pidx])
            
            # Attribute factor: average Δh for each attribute minus global mean
            attr_factors = {}
            for pidx in color_indices:
                attr = ALL_PAIRS[pidx][1]
                if attr not in attr_factors:
                    attr_factors[attr] = []
                attr_factors[attr].append(dh_correct[pidx])
            
            # Compute additive reconstruction error
            obj_means = {k: np.mean(v, axis=0) for k, v in obj_factors.items()}
            attr_means = {k: np.mean(v, axis=0) for k, v in attr_factors.items()}
            
            recon_errors = []
            for pidx in color_indices:
                obj, target, _ = ALL_PAIRS[pidx]
                if obj in obj_means and target in attr_means:
                    predicted = mean_dh + (obj_means[obj] - mean_dh) + (attr_means[target] - mean_dh)
                    actual = dh_correct[pidx]
                    error = np.linalg.norm(predicted - actual) / (np.linalg.norm(actual) + 1e-10)
                    recon_errors.append(error)
            
            if recon_errors:
                log(f"    Additive model recon error (color): mean={np.mean(recon_errors):.4f}, "
                    f"median={np.median(recon_errors):.4f}")
        
        # ===== Store results =====
        layer_result = {
            "n_pairs": n_pairs,
            "n_categories": len(categories),
            # Analysis 1
            "pca_explained_top10": [float(e) for e in explained[:10]],
            "eff_rank_95": eff_rank_95,
            "eff_rank_99": eff_rank_99,
            "n_pc_for_50pct": int(np.searchsorted(np.cumsum(explained), 0.5) + 1),
            "n_pc_for_90pct": int(np.searchsorted(np.cumsum(explained), 0.9) + 1),
            # Analysis 2
            "attr_independence": attr_independence_errors,
            # Analysis 3
            "same_attr_cos": float(mean_within) if 'mean_within' in dir() else None,
            "same_obj_cos": float(mean_within_obj) if 'mean_within_obj' in dir() else None,
            # Analysis 4
            "within_category_cos": within_cat_cos,
            "between_category_cos": category_cos,
            # Analysis 5
            "wrong_pca_pc1": float(explained_w[0]),
            "wrong_eff_rank_95": int(np.searchsorted(np.cumsum(explained_w), 0.95) + 1),
            "cos_correct_wrong_pc": cos_pc_subspace,
            # Analysis 6
            "additive_recon_error_mean": float(np.mean(recon_errors)) if 'recon_errors' in dir() and recon_errors else None,
            "additive_recon_error_median": float(np.median(recon_errors)) if 'recon_errors' in dir() and recon_errors else None,
        }
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done")
    
    return results


def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 374: {model_name}")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    if model_name == "deepseek7b":
        target_layers = [4, 5, 8, 12, 18, 24]
    elif model_name == "qwen3":
        target_layers = [4, 8, 16, 28]
    else:
        target_layers = [4, 10, 20, 30]
    
    results = collect_binding_vectors(
        model, tokenizer, device, model_name, target_layers, n_layers, d_model)
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(ALL_PAIRS),
        "phase": "374",
        "target_layers": target_layers,
        "pair_categories": PAIR_CATEGORIES,
        "layer_results": results,
    }
    
    os.makedirs("results/phase374_binding_geometry", exist_ok=True)
    out_path = f"results/phase374_binding_geometry/{model_name}_phase374.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 374 Summary: {model_name}")
    log(f"{'='*60}")
    log(f"{'Layer':>5} | {'PC1':>6} | {'PC2':>6} | {'rank95':>6} | {'rank99':>6} | "
        f"{'same_attr':>9} | {'same_obj':>9} | {'within_cat':>10} | "
        f"{'wrongPC1':>8} | {'add_err':>8}")
    log("-" * 100)
    
    for l in sorted(results.keys(), key=int):
        r = results[l]
        log(f"  L{int(l):>3} | {r['pca_explained_top10'][0]:>6.3f} | "
            f"{r['pca_explained_top10'][1]:>6.3f} | "
            f"{r['eff_rank_95']:>6} | {r['eff_rank_99']:>6} | "
            f"{r.get('same_attr_cos',0) or 0:>9.3f} | "
            f"{r.get('same_obj_cos',0) or 0:>9.3f} | "
            f"{np.mean(list(r.get('within_category_cos',{}).values())):>10.3f} | "
            f"{r.get('wrong_pca_pc1',0):>8.3f} | "
            f"{r.get('additive_recon_error_mean',0) or 0:>8.3f}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released")
    
    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if model_name == "all":
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                run_model(name)
            except Exception as e:
                log(f"!!! {name} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_model(model_name)
    
    log("\nPhase 374 complete!")


if __name__ == "__main__":
    main()
