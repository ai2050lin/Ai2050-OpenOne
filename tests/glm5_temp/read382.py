"""Read Phase 382 results."""
import sys, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

f = f"results/phase382_factor_decomposition/{model}_phase382.json"
d = json.load(open(f, 'r', encoding='utf-8'))

print(f"\n{'='*70}")
print(f"Phase 382 Results: {model}")
print(f"{'='*70}")

for l_str in sorted(d['results'].keys(), key=int):
    r = d['results'][l_str]
    l = r['layer']
    print(f"\n--- Layer {l} ---")
    
    # Factor R²
    print("  Factor R² decomposition:")
    r2 = r['factor_r2']
    for fname in sorted(r2.keys(), key=lambda x: -r2[x]):
        print(f"    {fname:40s}: {r2[fname]:.4f}")
    
    # PC1 regression
    pc1 = r['pc1_regression']
    print(f"\n  PC1 semantic regression:")
    print(f"    Total R² = {pc1['pc1_r2_total']:.4f}")
    if pc1.get('pc1_individual_r2'):
        top5 = sorted(pc1['pc1_individual_r2'].items(), key=lambda x: -x[1])[:5]
        print(f"    Top 5 individual R²:")
        for name, r2v in top5:
            print(f"      {name:40s}: {r2v:.4f}")
    if pc1.get('pc1_standardized_betas'):
        top5 = sorted(pc1['pc1_standardized_betas'].items(), key=lambda x: -abs(x[1]))[:5]
        print(f"    Top 5 standardized β:")
        for name, beta in top5:
            print(f"      {name:40s}: β={beta:.3f}")
    
    # PC-factor correlations (top alignment per factor)
    pc_corr = r['pc_factor_correlation']
    if '__variance_explained_pct__' in pc_corr:
        ve = pc_corr['__variance_explained_pct__']
        print(f"\n  PC variance explained: " + ", ".join(f"PC{k+1}={ve[f'PC{k+1}']:.1f}%" for k in range(min(5, len(ve)))))
    
    print(f"\n  PC-factor top alignments:")
    for fname in sorted(pc_corr.keys()):
        if fname.startswith('__'):
            continue
        corrs = pc_corr[fname]
        top_pc = max(corrs.keys(), key=lambda k: abs(corrs[k]))
        print(f"    {fname:40s}: {top_pc} (corr={corrs[top_pc]:.3f})")
    
    # Category swap results
    swap = r.get('category_swap', {})
    same = swap.get('same_cat_changes', [])
    cross = swap.get('cross_cat_changes', [])
    if same:
        print(f"\n  Category swap (logit lens):")
        print(f"    Same-cat:  mean Δ = {np.mean(same):.4f}, std = {np.std(same):.4f}, n={len(same)}")
        print(f"    Cross-cat: mean Δ = {np.mean(cross):.4f}, std = {np.std(cross):.4f}, n={len(cross)}")
