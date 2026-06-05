"""Phase 382 supplementary: Pure category signal after removing object_identity & norm_ratio."""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

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

models = ["qwen3", "deepseek7b", "glm4"]
for model in models:
    f = f"results/phase382_factor_decomposition/{model}_phase382.json"
    d = json.load(open(f, 'r', encoding='utf-8'))
    
    print(f"\n{'='*70}")
    print(f"{model} — Incremental R² decomposition")
    print(f"{'='*70}")
    
    for l_str in sorted(d['results'].keys(), key=int):
        r = d['results'][l_str]
        l = r['layer']
        r2 = r['factor_r2']
        pc1 = r['pc1_regression']
        pc_corr = r['pc_factor_correlation']
        
        # Key R² values
        obj_r2 = r2.get('object_identity', 0)
        cat_r2 = r2.get('category', 0)
        nr_r2 = r2.get('scalar_norm_ratio', 0)
        nd_r2 = r2.get('scalar_norm_diff', 0)
        nc_r2 = r2.get('scalar_norm_clean', 0)
        lt_r2 = r2.get('scalar_logit_target_clean', 0)
        ld_r2 = r2.get('scalar_logit_diff', 0)
        ent_r2 = r2.get('scalar_entropy_clean', 0)
        
        # PC1 top factors
        pc1_ind = pc1.get('pc1_individual_r2', {})
        top3_pc1 = sorted(pc1_ind.items(), key=lambda x: -x[1])[:3]
        
        # Category PC alignment
        cat_pc = pc_corr.get('category', {})
        cat_top_pc = max(cat_pc.keys(), key=lambda k: abs(cat_pc[k])) if cat_pc else "N/A"
        cat_top_corr = cat_pc.get(cat_top_pc, 0) if cat_pc else 0
        
        # Norm_ratio PC alignment
        nr_pc = pc_corr.get('scalar_norm_ratio', {})
        nr_top_pc = max(nr_pc.keys(), key=lambda k: abs(nr_pc[k])) if nr_pc else "N/A"
        nr_top_corr = nr_pc.get(nr_top_pc, 0) if nr_pc else 0
        
        print(f"\n  L{l}:")
        print(f"    R²: obj={obj_r2:.3f}, cat={cat_r2:.3f}, nr={nr_r2:.3f}, nd={nd_r2:.3f}, "
              f"nc={nc_r2:.3f}, lt={lt_r2:.3f}, ld={ld_r2:.3f}, ent={ent_r2:.3f}")
        print(f"    PC1 top3: " + ", ".join(f"{n}={v:.3f}" for n, v in top3_pc1))
        print(f"    Cat→{cat_top_pc}({cat_top_corr:.3f}), NR→{nr_top_pc}({nr_top_corr:.3f})")
        
        # Compute incremental: after removing norm_ratio, what's the remaining R² for category?
        # This is approximate: cat_R²_after_nr_removal ≈ cat_R² / (1 - nr_R²)
        # Better: cat_R² - overlap. But we don't have overlap directly.
        # Use formula: R²(cat|no_nr) ≈ (cat_R² - |cat∩nr|) / (1 - nr_R²)
        # Where |cat∩nr| ≈ 0 if they're independent. 
        # But they're NOT independent in DS7B (PC1=nr, cat is on PC3)
        # So the incremental is: category explains cat_R² of the total, and (1-nr_R²) is the non-nr part.
        # The fraction of non-nr variance that category explains ≈ cat_R² / (1 - nr_R²)
        if nr_r2 < 0.99:
            cat_r2_non_nr = cat_r2 / (1 - nr_r2)
        else:
            cat_r2_non_nr = float('inf')
        print(f"    Cat R² in non-NR space: {cat_r2_non_nr:.3f} (= {cat_r2:.3f} / {1-nr_r2:.3f})")
