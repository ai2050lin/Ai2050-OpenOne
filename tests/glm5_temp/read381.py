import json, sys

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
d = json.load(open(f'results/phase381_norm_matched_category/{model}_phase381.json','r',encoding='utf-8'))

print(f"=== Part 1: Norm-Matched Classification ({model}) ===")
for l, r in sorted(d['part1_norm_matched_classification'].items(), key=lambda x: int(x[0])):
    cl = r['classification']
    print(f"  L{r['layer']}: PROPER={cl['acc_proper_centroid5']:.3f}, "
          f"NM={cl['acc_norm_matched_centroid5']:.3f}, "
          f"NoNR={cl['acc_no_norm_ratio_centroid5']:.3f}, "
          f"PC1~nr={r['pc1_norm_ratio_corr']:.3f}, "
          f"drop_NM={r['norm_matched_vs_original_drop']:+.3f}, "
          f"drop_NR={r['norm_regressed_vs_original_drop']:+.3f}")

print(f"\n=== Part 2: Norm vs Direction Causal ({model}) ===")
for l, r in sorted(d['part2_norm_vs_direction_causal'].items(), key=lambda x: int(x[0])):
    md = r['mean_delta']
    gf = r['gap_fraction']
    print(f"  L{r['layer']}: d_norm={md['pure_norm_vs_corrupt']:.3f}, "
          f"d_dir={md['pure_dir_vs_corrupt']:.3f}, "
          f"d_clean={md['clean_vs_corrupt']:.3f}, "
          f"norm_frac={gf['norm']:.3f}, dir_frac={gf['direction']:.3f}")

print(f"\n=== Part 3: Deep Layer Tracking ({model}) ===")
for l, r in sorted(d['part3_deep_layer_tracking'].items(), key=lambda x: int(x[0])):
    print(f"  L{r['layer']}: PC1var={r['pc1_variance']:.3f}, "
          f"eff_rank={r['effective_rank']}, "
          f"PC1~nr={r['pc1_norm_ratio_corr']:.3f}, "
          f"acc={r['acc_proper_centroid5']:.3f}, "
          f"acc_nm={r['acc_norm_matched_centroid5']:.3f}, "
          f"acc_no_nr={r['acc_no_norm_ratio_centroid5']:.3f}")
