import json, sys

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
suffix = sys.argv[2] if len(sys.argv) > 2 else "381b"

d = json.load(open(f'results/phase381_norm_matched_category/{model}_phase{suffix}.json','r',encoding='utf-8'))

results = d.get('results', d)
print(f"=== Phase {suffix} Confirmation: {model} ===")
for l, r in sorted(results.items(), key=lambda x: int(x[0])):
    print(f"\nLayer {r['layer']}:")
    print(f"  PC1~norm_ratio: {r['pc1_norm_ratio_corr']:.3f}")
    print(f"  Classification (centroid 5d):")
    for vname, accs in r['classification_variants'].items():
        print(f"    {vname:20s}: centroid5d={accs['centroid_5d']:.3f}, knn5={accs['knn5_5d']:.3f}, centroid10d={accs['centroid_10d']:.3f}")
    print(f"  After PC1 removal:")
    print(f"    new PC1 vs orig PC2: {r['after_pc1_removal']['new_pc1_vs_orig_pc2']:.3f}")
    print(f"    new PC1 vs norm_ratio: {r['after_pc1_removal']['new_pc1_vs_norm_ratio']:.3f}")
    print(f"    new PC1 max cat corr: {r['after_pc1_removal']['new_pc1_max_cat_corr']:.3f}")
    print(f"  PC category correlations (top 5):")
    for k in range(min(5, len(r['pc_category_correlations']))):
        pc_key = f"PC{k+1}"
        if pc_key in r['pc_category_correlations']:
            print(f"    {pc_key}: max|cat_corr|={r['pc_category_correlations'][pc_key]['max_abs_cat_corr']:.3f}")
