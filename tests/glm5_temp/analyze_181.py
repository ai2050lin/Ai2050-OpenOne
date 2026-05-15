"""Analyze Phase 181 results across models"""
import json
import numpy as np
import sys
import glob

def load_results(model_name):
    """Load latest Phase 181 results for a model"""
    pattern = f"tests/glm5_temp/phase181_{model_name}_*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No results found for {model_name}")
        return None
    with open(files[-1], 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_model(model_name):
    print(f"\n{'='*70}")
    print(f"  Phase 181 Analysis: {model_name.upper()}")
    print(f"{'='*70}")
    
    data = load_results(model_name)
    if data is None:
        return
    
    n_layers = data['n_layers']
    d_model = data['d_model']
    
    # ===== Exp1: Transport Ratio =====
    print(f"\n  === Exp1: Transport Ratio σ(l) ===")
    print(f"  {'Layer':<6}", end='')
    for cat in ['grammar', 'physical', 'animacy', 'causal', 'control']:
        print(f"  {cat:<10}", end='')
    print()
    
    exp1 = data['exp1_transport_ratio']
    key_layers = [0, 1, 5, 10, 15, 20, 25, 30, 35, n_layers-1]
    key_layers = [l for l in key_layers if l < n_layers]
    
    for li in key_layers:
        print(f"  L{li:<5}", end='')
        for cat in ['grammar', 'physical', 'animacy', 'causal', 'control']:
            if cat in exp1 and str(li) in exp1[cat]:
                sigma = exp1[cat][str(li)].get('transport_ratio_mean', 0)
                print(f"  {sigma:>8.3f}  ", end='')
            else:
                print(f"  {'N/A':>8}  ", end='')
        print()
    
    # Key comparison: constraint vs control
    print(f"\n  === Constraint vs Control: Δσ = σ(constraint) - σ(control) ===")
    print(f"  {'Layer':<6}", end='')
    for cat in ['grammar', 'physical', 'animacy', 'causal']:
        print(f"  {cat:<10}", end='')
    print()
    
    for li in key_layers:
        print(f"  L{li:<5}", end='')
        ctrl_sigma = 0
        if 'control' in exp1 and str(li) in exp1['control']:
            ctrl_sigma = exp1['control'][str(li)].get('transport_ratio_mean', 0)
        
        for cat in ['grammar', 'physical', 'animacy', 'causal']:
            if cat in exp1 and str(li) in exp1[cat]:
                sigma = exp1[cat][str(li)].get('transport_ratio_mean', 0)
                diff = sigma - ctrl_sigma
                print(f"  {diff:>+8.3f}  ", end='')
            else:
                print(f"  {'N/A':>8}  ", end='')
        print()
    
    # ===== Δ norm trend =====
    print(f"\n  === ||Δ|| Trend (Constraint Signal Magnitude) ===")
    print(f"  {'Layer':<6}", end='')
    for cat in ['grammar', 'physical', 'animacy', 'causal', 'control']:
        print(f"  {cat:<10}", end='')
    print()
    
    for li in key_layers:
        print(f"  L{li:<5}", end='')
        for cat in ['grammar', 'physical', 'animacy', 'causal', 'control']:
            if cat in exp1 and str(li) in exp1[cat]:
                dn = exp1[cat][str(li)].get('delta_norm_mean', 0)
                print(f"  {dn:>8.2f}  ", end='')
            else:
                print(f"  {'N/A':>8}  ", end='')
        print()
    
    # ===== Exp2: W_U Decomposition =====
    print(f"\n  === Exp2: W_U Decomposition ===")
    exp2 = data['exp2_wu_decomposition']
    
    print(f"\n  --- Parallel Ratio ||Δ_∥|| / ||Δ|| (in decoder space) ---")
    print(f"  {'Layer':<6}", end='')
    for cat in ['grammar', 'physical', 'animacy', 'causal']:
        print(f"  {cat:<10}", end='')
    print()
    
    for li in [0, 5, 10, 15, 20, 25, 30, n_layers]:
        print(f"  L{li:<5}", end='')
        for cat in ['grammar', 'physical', 'animacy', 'causal']:
            if cat in exp2 and str(li) in exp2[cat]:
                pr = exp2[cat][str(li)].get('parallel_ratio_mean', 0)
                print(f"  {pr:>8.3f}  ", end='')
            else:
                print(f"  {'N/A':>8}  ", end='')
        print()
    
    print(f"\n  --- Orthogonal Ratio ||Δ_⊥|| / ||Δ|| (NOT in decoder space) ---")
    print(f"  {'Layer':<6}", end='')
    for cat in ['grammar', 'physical', 'animacy', 'causal']:
        print(f"  {cat:<10}", end='')
    print()
    
    for li in [0, 5, 10, 15, 20, 25, 30, n_layers]:
        print(f"  L{li:<5}", end='')
        for cat in ['grammar', 'physical', 'animacy', 'causal']:
            if cat in exp2 and str(li) in exp2[cat]:
                ortho = exp2[cat][str(li)].get('orthogonal_ratio_mean', 0)
                print(f"  {ortho:>8.3f}  ", end='')
            else:
                print(f"  {'N/A':>8}  ", end='')
        print()
    
    # ===== Exp3: Phase Analysis =====
    print(f"\n  === Exp3: Phase Analysis Summary ===")
    exp3 = data['exp3_phase_analysis']
    for cat in ['grammar', 'physical', 'animacy', 'causal', 'control']:
        if cat in exp3:
            pd = exp3[cat]
            print(f"    {cat}: avg σ={pd.get('avg_transport_ratio',0):.3f}, "
                  f"max σ={pd.get('max_transport_ratio',0):.3f} at L{pd.get('max_sigma_layer','?')}, "
                  f"min σ={pd.get('min_transport_ratio',0):.3f}, "
                  f"phase transition at L{pd.get('phase_transition_layer','N/A')}, "
                  f"prop={pd.get('n_propagation_layers',0)} contr={pd.get('n_contraction_layers',0)}")
    
    # ===== Key constraint vs control differences =====
    print(f"\n  === Constraint vs Control: Transport Difference ===")
    for cat in ['grammar', 'physical', 'animacy', 'causal']:
        key = f'{cat}_vs_control'
        if key in exp3:
            kd = exp3[key]
            print(f"    {cat} vs control: avg Δσ = {kd.get('avg_diff',0):.4f}, "
                  f"max Δσ = {kd.get('max_diff',0):.4f} at L{kd.get('max_diff_layer','?')}")

# Run analysis
for model in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        analyze_model(model)
    except Exception as e:
        print(f"Error analyzing {model}: {e}")
