"""Phase 276 cross-model summary"""
import json, numpy as np

print("=" * 80)
print("Phase 276: Jacobian Spectrum & Conditional Operator — CROSS-MODEL COMPARISON")
print("=" * 80)

# === Exp A: Spectral Feature Similarity ===
print("\n=== Exp A: Spectral Feature Similarity ===")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase276_jspectrum/{model}_exp_a_summary.json') as f:
        data = json.load(f)
    sfs = data['spectral_feature_similarity']
    svcs = data['sv_cosine_similarity']
    print(f"\n{model}:")
    print(f"  Spectral feature sim: within={sfs['within_mean']:.8f}, between={sfs['between_mean']:.8f}, delta={sfs['delta']:+.2e}")
    print(f"  SV cosine sim:        within={svcs['within_mean']:.8f}, between={svcs['between_mean']:.8f}, delta={svcs['delta']:+.2e}")
    
    print("  Per-layer spectral averages:")
    for layer_idx in data['sampled_layers']:
        pl = data['per_layer_avg_features'][str(layer_idx)]
        print(f"    L{layer_idx}: eff_rank={pl['effective_rank']['mean']:.1f}, "
              f"entropy={pl['spectral_entropy']['mean']:.4f}, "
              f"top1_sv={pl['top1_sv']['mean']:.2f}, "
              f"unstable%={pl['unstable_ratio']['mean']:.4f}, "
              f"stable%={pl['stable_ratio']['mean']:.4f}")

# === Exp B: Operator Distance ===
print("\n\n=== Exp B: Conditional Operator Distance ===")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase276_jspectrum/{model}_exp_b_summary.json') as f:
        data = json.load(f)
    fd = data['frobenius_distance']
    sc = data['subspace_cosine']
    print(f"\n{model}:")
    print(f"  Frobenius dist:  within={fd['within_mean']:.4f}, between={fd['between_mean']:.4f}, delta={fd['delta']:+.4f}")
    print(f"  Subspace cosine: within={sc['within_mean']:.4f}, between={sc['between_mean']:.4f}, delta={sc['delta']:+.4f}")
    
    print("  Per-layer:")
    for layer_idx in data['sampled_layers']:
        pl = data['per_layer'][str(layer_idx)]
        fd_l = pl['frobenius_distance']
        sc_l = pl['subspace_cosine']
        pa_l = pl['principal_angle_cos']
        print(f"    L{layer_idx}: frob_delta={fd_l.get('delta', 'N/A'):+.4f}, "
              f"subspace_delta={sc_l.get('delta', 'N/A'):+.4f}, "
              f"principal_delta={pa_l.get('delta', 'N/A'):+.4f}")

# === Exp C: Clustering ===
print("\n\n=== Exp C: Dynamical Clustering ===")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase276_jspectrum/{model}_exp_c_clustering.json') as f:
        data = json.load(f)
    sari = data.get('spectral_clustering', {}).get('ari', 'N/A')
    eari = data.get('embedding_clustering', {}).get('ari', 'N/A')
    print(f"\n{model}: spectral_ARI={sari}, embedding_ARI={eari}")
    pl_ari = data.get('per_layer_spectral_ari', {})
    if pl_ari:
        best_layer = max(pl_ari.items(), key=lambda x: x[1])
        print(f"  Best per-layer ARI: L{best_layer[0]} = {best_layer[1]:.4f}")

# === Exp D: Critical Layers ===
print("\n\n=== Exp D: Critical Layer Search ===")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase276_jspectrum/{model}_exp_d_critical.json') as f:
        data = json.load(f)
    ts = data['transition_similarity']
    print(f"\n{model}: transition_sim within={ts['within_mean']:.4f}, "
          f"between={ts['between_mean']:.4f}, delta={ts['delta']:+.4f}")
    # Find largest transition
    ta = data['transition_avg']
    if ta:
        max_key = max(ta.items(), key=lambda x: x[1]['mean_delta_norm'])
        print(f"  Largest transition: {max_key[0]} = {max_key[1]['mean_delta_norm']:.1f}")

# === Cross-method comparison ===
print("\n\n=== CROSS-METHOD: Phase 275 Jacobian cosine vs Phase 276 Subspace cosine ===")
for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase275_cdre/{model}_exp_a_summary.json') as f:
        p275 = json.load(f)
    with open(f'results/phase276_jspectrum/{model}_exp_b_summary.json') as f:
        p276 = json.load(f)
    
    js275 = p275['jacobian_similarity']
    sc276 = p276['subspace_cosine']
    fd276 = p276['frobenius_distance']
    
    print(f"\n{model}:")
    print(f"  Phase 275 Jacobian cosine delta: {js275['delta']:+.4f}")
    print(f"  Phase 276 Subspace cosine delta:  {sc276['delta']:+.4f}")
    print(f"  Phase 276 Frobenius dist delta:   {fd276['delta']:+.4f}")
    print(f"  Phase 276 Spectral feature delta: {p276.get('spectral_feature_similarity', {}).get('delta', 'N/A')}")
