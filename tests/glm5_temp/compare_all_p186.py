"""Phase 186 Cross-Model Comparison"""
import json, glob, numpy as np

models = ['qwen3', 'glm4', 'deepseek7b']
data = {}
for m in models:
    files = glob.glob(f"tests/glm5_temp/phase186_{m}_*.json")
    if files:
        data[m] = json.load(open(files[-1], 'r', encoding='utf-8'))

print("="*80)
print("PHASE 186: CROSS-MODEL COMPARISON")
print("="*80)

# ===== Exp1: Equivalence Class Contraction =====
print("\n" + "="*80)
print("TABLE 1: Equivalence Class Contraction (Intra-class cosine distance)")
print("="*80)
print(f"{'Layer':<8}", end="")
for m in models:
    print(f"{'  '+m.upper()+' intra':>14}{'  inter':>10}{'  sep':>8}", end="")
print()
print("-"*80)

# Get common layers
all_layers = set()
for m in models:
    e1 = data[m]['exp1_equivalence_class_contraction']
    for k in e1:
        try:
            all_layers.add(int(k))
        except (ValueError, TypeError):
            pass

for li in sorted(all_layers):
    if li > 40:
        continue
    print(f"L{li:<6}", end="")
    for m in models:
        e1 = data[m]['exp1_equivalence_class_contraction']
        key = str(li)
        if key in e1:
            intra = e1[key].get('intra_cos_mean', 0)
            inter = e1[key].get('inter_cos_mean', 0)
            sep = e1[key].get('separability_cos', 0)
            print(f"  {intra:>8.4f}{inter:>10.4f}{sep:>8.2f}", end="")
        else:
            print(f"  {'N/A':>8}{'':>10}{'':>8}", end="")
    print()

# Meta comparison
print("\n--- Exp1 Summary ---")
for m in models:
    e1 = data[m]['exp1_equivalence_class_contraction']
    meta = e1.get('_meta', {})
    print(f"  {m}: intra_slope={meta.get('intra_slope',0):.5f} [{meta.get('intra_verdict','N/A')}], "
          f"inter_slope={meta.get('inter_slope',0):.5f} [{meta.get('inter_verdict','N/A')}], "
          f"sep={meta.get('separability_first',0):.2f}→{meta.get('separability_last',0):.2f}")

# ===== Exp2: Distinguishability =====
print("\n" + "="*80)
print("TABLE 2: Distinguishability (Distance at last layer)")
print("="*80)
pair_names = ['apple_vs_pear', 'dog_vs_cat', 'apple_vs_banana', 'apple_vs_car', 'dog_vs_book']
print(f"{'Pair':<20} {'Sim':>5}", end="")
for m in models:
    print(f"  {m.upper():>8}", end="")
print()
print("-"*50)
for pn in pair_names:
    for m in models:
        e2 = data[m]['exp2_distinguishability_emergence']
        if pn in e2:
            meta = e2[pn].get('_meta', {})
            exp_sim = meta.get('expected_similarity', 0)
            last_d = meta.get('last_dist', 0)
            if pn == pair_names[0]:
                print(f"{pn:<20} {exp_sim:>5.1f}", end="")
            print(f"  {last_d:>8.4f}", end="")
        else:
            if pn == pair_names[0]:
                print(f"{pn:<20} {'':>5}", end="")
            print(f"  {'N/A':>8}", end="")
    print()

# ===== Exp3: Cross-Lingual =====
print("\n" + "="*80)
print("TABLE 3: Cross-Lingual Semantic Orbit")
print("="*80)
print(f"{'Layer':<8}", end="")
for m in models:
    print(f"  {m.upper()+' CL':>10}{' near':>8}{' ratio':>8}", end="")
print()
print("-"*70)

all_layers3 = set()
for m in models:
    e3 = data[m]['exp3_cross_lingual_orbit']
    for k in e3:
        try:
            all_layers3.add(int(k))
        except (ValueError, TypeError):
            pass

for li in sorted(all_layers3):
    if li > 40:
        continue
    print(f"L{li:<6}", end="")
    for m in models:
        e3 = data[m]['exp3_cross_lingual_orbit']
        key = str(li)
        if key in e3:
            d = e3[key]
            cl = d.get('cross_lingual', {}).get('mean', 0)
            sn = d.get('same_lang_near', {}).get('mean', 0)
            ratio = d.get('ratio_cross_to_near', 0)
            print(f"  {cl:>10.4f}{sn:>8.4f}{ratio:>8.2f}", end="")
        else:
            print(f"  {'':>10}{'':>8}{'':>8}", end="")
    print()

# Meta
print("\n--- Exp3 Summary ---")
for m in models:
    e3 = data[m]['exp3_cross_lingual_orbit']
    meta = e3.get('_meta', {})
    print(f"  {m}: CL_slope={meta.get('cross_lingual_slope',0):.5f} [{meta.get('orbit_verdict','N/A')[:30]}]")

# ===== Exp4: Jacobian Context Dependence =====
print("\n" + "="*80)
print("TABLE 4: Jacobian Context Dependence")
print("="*80)
for m in models:
    e4 = data[m]['exp4_trained_vs_random_jacobian']
    comp = e4.get('_comparison', {})
    meaningful_g = comp.get('meaningful_g_mean', 0)
    random_g = comp.get('random_g_mean', 0)
    p_val = comp.get('p_value', 1)
    verdict = comp.get('verdict', 'N/A')
    print(f"  {m}: meaningful_g={meaningful_g:.3f}, random_g={random_g:.3f}, p={p_val:.4f}")
    print(f"    → {verdict}")

# ===== KEY INSIGHTS =====
print("\n" + "="*80)
print("★★★ KEY INSIGHTS ★★★")
print("="*80)

print("""
1. ALL models show EQUIVALENCE CLASS CONTRACTION (intra < 0 after deep layers)
   - This is the strongest evidence yet that models form semantic equivalence classes

2. ALL models show CROSS-LINGUAL SEMANTIC ORBITS
   - CL/near ratio approaches 1.0 in deep layers
   - Cross-lingual distances shrink faster than same-language distances
   - This is STRONG EVIDENCE for language-invariant semantic representation

3. Separability index is roughly CONSTANT across layers
   - This means: compression is ISOMETRIC (same factor for intra and inter)
   - NOT "intra shrinks + inter expands" as predicted by simple theory
   - Instead: ALL distances shrink, but RATIO is preserved
   - This is "information manifold compression" not "differential separation"

4. QWEN3 is UNIQUE: λ>1 is context-dependent (p=0.018)
   - Qwen3's Jacobian amplification is a LEARNED encoding mechanism
   - GLM4 and DS7B: λ>1 is likely architectural (p=0.71, p=0.19)

5. DS7B's repeated token has huge g (6.2) — architectural instability for degenerate input
""")
