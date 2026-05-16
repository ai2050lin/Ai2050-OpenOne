"""Read Phase 186 results for all models"""
import json, glob, numpy as np
from collections import defaultdict

def read_p186(model_name):
    files = glob.glob(f"tests/glm5_temp/phase186_{model_name}_*.json")
    if not files:
        print(f"  {model_name}: No results yet")
        return None
    d = json.load(open(files[-1], 'r', encoding='utf-8'))
    print(f"\n{'='*70}")
    print(f"PHASE 186: {model_name.upper()} (from {files[-1].split('\\\\')[-1]})")
    print(f"{'='*70}")
    
    # ===== Exp1: Equivalence Class Contraction =====
    print("\n--- Exp1: Equivalence Class Contraction ---")
    e1 = d['exp1_equivalence_class_contraction']
    meta1 = e1.get('_meta', {})
    print(f"  Intra-class slope: {meta1.get('intra_slope',0):.5f} [{meta1.get('intra_verdict','N/A')}]")
    print(f"  Inter-class slope: {meta1.get('inter_slope',0):.5f} [{meta1.get('inter_verdict','N/A')}]")
    sep_first = meta1.get('separability_first', 0)
    sep_last = meta1.get('separability_last', 0)
    print(f"  Separability index: first={sep_first:.3f} → last={sep_last:.3f}")
    
    # Per-layer separability
    print("\n  Per-layer:")
    for k in sorted([int(x) for x in e1.keys() if x not in ('_meta', '_per_class') and not x.startswith('_')]):
        li = str(k)
        if li in e1:
            data = e1[li]
            intra = data.get('intra_cos_mean', 0)
            inter = data.get('inter_cos_mean', 0)
            sep = data.get('separability_cos', 0)
            print(f"    L{k}: intra={intra:.4f}, inter={inter:.4f}, sep={sep:.2f}")
    
    # Per-class contraction
    print("\n  Per-class intra-distance (first→last layer):")
    per_class = e1.get('_per_class', {})
    for cname in per_class:
        pc = per_class[cname]
        layers_s = sorted([int(x) for x in pc.keys()])
        if len(layers_s) >= 2:
            first_d = pc[str(layers_s[0])].get('intra_cos_mean', 0)
            last_d = pc[str(layers_s[-1])].get('intra_cos_mean', 0)
            change = last_d - first_d
            print(f"    {cname}: {first_d:.4f} → {last_d:.4f} (Δ={change:.4f})")
    
    # ===== Exp2: Distinguishability Emergence =====
    print("\n--- Exp2: Distinguishability Emergence ---")
    e2 = d['exp2_distinguishability_emergence']
    for pair_name in ['apple_vs_pear', 'dog_vs_cat', 'apple_vs_banana', 
                      'apple_vs_car', 'dog_vs_book', 'hot_vs_cold']:
        if pair_name in e2:
            meta = e2[pair_name].get('_meta', {})
            slope = meta.get('emergence_slope', 0)
            first_d = meta.get('first_dist', 0)
            last_d = meta.get('last_dist', 0)
            verdict = meta.get('verdict', 'N/A')
            exp_sim = meta.get('expected_similarity', 0)
            print(f"  {pair_name} (sim~{exp_sim}): {first_d:.4f}→{last_d:.4f}, slope={slope:.5f} [{verdict}]")
    
    corr = e2.get('_correlation', {})
    if corr:
        print(f"  ★ Correlation: ρ={corr.get('spearman_rho',0):.3f}, p={corr.get('p_value',1):.4f}")
        print(f"    → {corr.get('verdict','N/A')}")
    
    # ===== Exp3: Cross-Lingual Semantic Orbit =====
    print("\n--- Exp3: Cross-Lingual Semantic Orbit ---")
    e3 = d['exp3_cross_lingual_orbit']
    meta3 = e3.get('_meta', {})
    print(f"  Cross-lingual slope: {meta3.get('cross_lingual_slope',0):.5f} [{meta3.get('orbit_verdict','N/A')}]")
    print(f"  Same-lang near slope: {meta3.get('same_lang_near_slope',0):.5f}")
    print(f"  Same-lang far slope: {meta3.get('same_lang_far_slope',0):.5f}")
    
    # Per-layer cross-lingual vs same-lang
    print("\n  Per-layer distances:")
    for k in sorted([int(x) for x in e3.keys() if x != '_meta' and not x.startswith('_')]):
        li = str(k)
        if li in e3:
            data = e3[li]
            cl = data.get('cross_lingual', {}).get('mean', 0)
            sn = data.get('same_lang_near', {}).get('mean', 0)
            sf = data.get('same_lang_far', {}).get('mean', 0)
            ratio = data.get('ratio_cross_to_near', 0)
            verdict = data.get('orbit_verdict', 'N/A')[:20]
            print(f"    L{k}: CL={cl:.4f}, near={sn:.4f}, far={sf:.4f}, CL/near={ratio:.2f} [{verdict}]")
    
    # ===== Exp4: Trained vs Random Jacobian =====
    print("\n--- Exp4: Trained vs Random Jacobian ---")
    e4 = d['exp4_trained_vs_random_jacobian']
    comp = e4.get('_comparison', {})
    print(f"  Meaningful g_mean: {comp.get('meaningful_g_mean',0):.3f}")
    print(f"  Random g_mean: {comp.get('random_g_mean',0):.3f}")
    print(f"  p-value: {comp.get('p_value',1):.4f}")
    print(f"  → {comp.get('verdict','N/A')}")
    
    # Per-type averages
    for input_type in ['meaningful', 'random_order', 'repeated']:
        if input_type in e4:
            meta = e4[input_type].get('_meta', {})
            print(f"  {input_type}: g={meta.get('overall_g_mean',0):.3f}±{meta.get('overall_g_std',0):.3f}")
    
    return d

# Read all available
for m in ['qwen3', 'glm4', 'deepseek7b']:
    read_p186(m)
