import json
import numpy as np

with open('tests/glm5_temp/phase217_3_normalized_kl_results.json') as f:
    data = json.load(f)

trained = {d['layer']: d for d in data['trained']}
random = {d['layer']: d for d in data['random']}

print("=== Strict Component Decomposition ===")
print()

for l in [0, 7, 14, 21, 27]:
    t = trained[l]
    r = random[l]
    
    kl_orig_t = t['kl_original']
    kl_norm_t = t['kl_normalized']
    kl_norm_r = r['kl_normalized']
    
    # Component A: h_norm scale effect
    scale_effect_t = kl_orig_t - kl_norm_t
    scale_frac_t = scale_effect_t / max(kl_orig_t, 1e-10) * 100
    
    # Component B: W_U anisotropy (normalized trained - normalized random)
    wu_effect = kl_norm_t - kl_norm_r
    wu_frac = wu_effect / max(kl_orig_t, 1e-10) * 100
    
    # Component C: structural routing (normalized random baseline)
    baseline_frac = kl_norm_r / max(kl_orig_t, 1e-10) * 100
    
    print(f"Layer {l}:")
    print(f"  Trained: KL_orig={kl_orig_t:.4f}, KL_norm={kl_norm_t:.6f}")
    print(f"  Random:  KL_norm={kl_norm_r:.6f}")
    print(f"  A. Scale effect: {scale_effect_t:.4f} ({scale_frac_t:.1f}%)")
    print(f"  B. W_U anisotropy: {wu_effect:.6f} ({wu_frac:.1f}%)")
    print(f"  C. Baseline (random): {kl_norm_r:.6f} ({baseline_frac:.1f}%)")
    print()

# Precise decomposition of trained KL decrease L0->L27
print("=== Trained KL Decrease Decomposition ===")
t0 = trained[0]
t27 = trained[27]
r0 = random[0]
r27 = random[27]

total_decrease = t0['kl_original'] - t27['kl_original']
print(f"Total KL decrease: {t0['kl_original']:.4f} -> {t27['kl_original']:.10f} = {total_decrease:.4f}")
print()

# Component 1: h_norm effect (difference between original and normalized at L0)
norm_decrease_L0 = t0['kl_original'] - t0['kl_normalized']
print(f"h_norm effect at L0: {norm_decrease_L0:.4f} ({norm_decrease_L0/total_decrease*100:.1f}% of total decrease)")

# Component 2: W_U anisotropy
wu_L0 = t0['kl_normalized'] - r0['kl_normalized']
wu_L27 = t27['kl_normalized'] - r27['kl_normalized']
print(f"W_U anisotropy at L0: {wu_L0:.6f} ({wu_L0/total_decrease*100:.2f}% of total decrease)")

# Component 3: structural routing (normalized KL decrease, minus random baseline)
norm_decrease_total = t0['kl_normalized'] - t27['kl_normalized']
random_norm_decrease = r0['kl_normalized'] - r27['kl_normalized']
structural_net = norm_decrease_total - random_norm_decrease
print(f"Normalized KL decrease (structural+random): {norm_decrease_total:.6f} ({norm_decrease_total/total_decrease*100:.2f}%)")
print(f"Random baseline decrease: {random_norm_decrease:.6f}")
print(f"Net structural routing: {structural_net:.6f} ({structural_net/total_decrease*100:.2f}%)")
print()

# Key numbers
print("=== Key Numbers ===")
print(f"Trained L0 KL_orig  = {t0['kl_original']:.4f}")
print(f"Trained L0 KL_norm  = {t0['kl_normalized']:.6f}")
print(f"Random  L0 KL_norm  = {r0['kl_normalized']:.6f}")
print(f"Trained L27 KL_norm = {t27['kl_normalized']:.8f}")
print(f"Random  L27 KL_norm = {r27['kl_normalized']:.8f}")
print()

# W_U anisotropy ratio
print(f"W_U anisotropy ratio at L0: trained/random = {t0['kl_normalized']/r0['kl_normalized']:.1f}x")
print(f"W_U anisotropy ratio at L27: trained/random = {t27['kl_normalized']/r27['kl_normalized']:.1f}x")
print()

# Numerical check for L27
print("=== L27 Numerical Check ===")
with open('tests/glm5_temp/phase217_2_verification_results.json') as f:
    data2 = json.load(f)

for item in data2['trained_layer_analysis']:
    if item['layer'] == 27:
        print(f"  logits_max = {item['logits_max']:.1f}")
        print(f"  logits_std = {item['logits_std']:.1f}")
        print(f"  p_max = {item['p_max']}")
        print(f"  logits_max / logits_std = {item['logits_max']/item['logits_std']:.1f}")
        print(f"  -> softmax collapse: logits_max >> logits_std => near one-hot")
        break

# Check which layers have NaN entropy (numerical issues)
print()
print("=== Layers with NaN entropy (numerical instability) ===")
for item in data2['trained_layer_analysis']:
    if str(item.get('entropy', '')) == 'nan' or (isinstance(item.get('entropy'), float) and np.isnan(item.get('entropy', 0))):
        print(f"  Layer {item['layer']}: p_max={item['p_max']:.4f}, logits_max={item['logits_max']:.1f}")

# Random model entropy check
print()
print("=== Random Model Output Entropy ===")
for l in [0, 7, 14, 21, 27]:
    for item in data2['random_layer_analysis']:
        if item['layer'] == l:
            print(f"  L{l}: entropy={item['entropy']:.2f}, max_entropy={item['max_entropy']:.2f}, ratio={item['entropy_ratio']:.3f}")
            break

print()
print("=== CORRECTED Component Decomposition ===")
print()
print("Original 70/25/5 claim was NOT mathematically rigorous.")
print("Correct decomposition relative to total KL decrease (5.304):")
print(f"  A. h_norm scale effect:   {norm_decrease_L0:.4f} = {norm_decrease_L0/total_decrease*100:.1f}%")
print(f"  B. W_U anisotropy:         {wu_L0:.6f} = {wu_L0/total_decrease*100:.2f}%")
print(f"  C. Structural (net):       {structural_net:.6f} = {structural_net/total_decrease*100:.2f}%")
print(f"  D. Random baseline:        {random_norm_decrease:.6f} = {random_norm_decrease/total_decrease*100:.2f}%")
print()
print("Note: Components B+C+D don't sum to total because of nonlinear interactions")
print("The '70%' was a rough estimate. The true scale effect is ~99.5% at L0 level.")
print()
print("But this decomposition is misleading because:")
print("  1. h_norm effect (99.5%) is calculated at L0 only")
print("  2. The real question is: at each layer, what fraction of KL")
print("     is due to W_U anisotropy vs structural routing?")
print()

# Per-layer decomposition relative to that layer's KL
print("=== Per-layer: Fraction of normalized KL due to W_U anisotropy ===")
for l in [0, 7, 14, 21, 27]:
    t = trained[l]
    r = random[l]
    wu_frac_of_norm = (t['kl_normalized'] - r['kl_normalized']) / max(t['kl_normalized'], 1e-10) * 100
    print(f"  L{l}: norm_KL_train={t['kl_normalized']:.6f}, norm_KL_rand={r['kl_normalized']:.6f}, W_U fraction={wu_frac_of_norm:.1f}%")
