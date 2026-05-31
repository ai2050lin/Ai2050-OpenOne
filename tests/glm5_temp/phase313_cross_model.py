import json
import numpy as np

print("=" * 80)
print("Phase 313: W_U READOUT ANALYSIS - CROSS-MODEL COMPARISON")
print("=" * 80)

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase313_WU_readout/{mn}_WU_readout.json') as f:
            d = json.load(f)
    except:
        continue
    
    print(f"\n{'='*70}")
    print(f"  {mn.upper()}")
    print(f"{'='*70}")
    
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        
        # Key W_U gains
        random_wu = lr.get('random_WU_gain', 0)
        O_not_wu = lr['WU_gains'].get('O_not', 0)
        O_clean_R_wu = lr['WU_gains'].get('O_clean_R', 0)
        O_clean_RC_wu = lr['WU_gains'].get('O_clean_RC', 0)
        R_wu = lr['WU_gains'].get('R', 0)
        C_pc1_wu = lr['WU_gains'].get('C_pc1', 0)
        A_wu = lr['WU_gains'].get('A', 0)
        
        # Key Jacobian gains
        O_not_jac = lr.get('jacobian_gains', {}).get('O_not', {}).get('gain', 0)
        O_clean_R_jac = lr.get('jacobian_gains', {}).get('O_clean_R', {}).get('gain', 0)
        O_clean_RC_jac = lr.get('jacobian_gains', {}).get('O_clean_RC', {}).get('gain', 0)
        R_jac = lr.get('jacobian_gains', {}).get('R', {}).get('gain', 0)
        C_pc1_jac = lr.get('jacobian_gains', {}).get('C_pc1', {}).get('gain', 0)
        A_jac = lr.get('jacobian_gains', {}).get('A', {}).get('gain', 0)
        random_jac = lr.get('random_jacobian_gain', 0)
        
        # Jac/WU ratio (intermediate amplification)
        O_not_amp = O_not_jac / O_not_wu if O_not_wu > 0 else 0
        O_clean_R_amp = O_clean_R_jac / O_clean_R_wu if O_clean_R_wu > 0 else 0
        O_clean_RC_amp = O_clean_RC_jac / O_clean_RC_wu if O_clean_RC_wu > 0 else 0
        R_amp = R_jac / R_wu if R_wu > 0 else 0
        C_pc1_amp = C_pc1_jac / C_pc1_wu if C_pc1_wu > 0 else 0
        A_amp = A_jac / A_wu if A_wu > 0 else 0
        
        # Direction norms
        O_not_norm = lr['direction_norms'].get('O_not', 0)
        O_clean_R_norm = lr['direction_norms'].get('O_clean_R', 0)
        R_norm = lr['direction_norms'].get('R', 0)
        O_clean_R_ratio = O_clean_R_norm / O_not_norm if O_not_norm > 0 else 0
        
        # Delta from shared - key intermediate amplification
        delta_names = [k for k in lr.get('jacobian_gains', {}) if '_delta_from_shared' in k]
        delta_jacs = {k: lr['jacobian_gains'][k].get('gain', 0) for k in delta_names}
        delta_ratios = {k: lr['jacobian_gains'][k].get('ratio_to_random', 0) for k in delta_names}
        
        print(f"\n  L{li}: O_clean_R/O_not norm ratio = {O_clean_R_ratio:.3f}")
        print(f"    W_U gains (||WU@v||/||v||): O_not={O_not_wu:.2f}, O_clean_R={O_clean_R_wu:.2f}, R={R_wu:.2f}, A={A_wu:.2f}")
        print(f"    W_U gain ratio to random: O_not={O_not_wu/random_wu:.2f}x, O_clean_R={O_clean_R_wu/random_wu:.2f}x, R={R_wu/random_wu:.2f}x, A={A_wu/random_wu:.2f}x")
        print(f"    Jacobian gains: O_not={O_not_jac:.1f}, O_clean_R={O_clean_R_jac:.1f}, R={R_jac:.1f}, A={A_jac:.1f}")
        print(f"    Jacobian ratio to random: O_not={O_not_jac/random_jac:.2f}x, O_clean_R={O_clean_R_jac/random_jac:.2f}x, R={R_jac/random_jac:.2f}x, A={A_jac/random_jac:.2f}x")
        print(f"    INTERMEDIATE AMPLIFICATION (jac/wu): O_not={O_not_amp:.2f}, O_clean_R={O_clean_R_amp:.2f}, R={R_amp:.2f}, A={A_amp:.2f}")
        print(f"    O_clean_R/O_not amp ratio: {O_clean_R_amp/O_not_amp:.3f}" if O_not_amp > 0 else "    O_clean_R/O_not amp ratio: N/A")
        
        # Delta from shared amplification
        if delta_names:
            print(f"    DELTA FROM SHARED amplification:")
            for k in sorted(delta_names):
                print(f"      {k}: jac={delta_jacs[k]:.1f}, ratio={delta_ratios[k]:.2f}x random")

# KEY COMPARISON TABLE
print("\n\n" + "=" * 80)
print("KEY COMPARISON: O_clean_R vs O_not INTERMEDIATE AMPLIFICATION (jac/wu ratio)")
print("=" * 80)
print(f"{'Model':<12} {'Layer':>6} {'O_not_amp':>10} {'O_clean_R_amp':>14} {'Ratio':>8} {'O_clean_R/O_not norm':>22}")
print("-" * 80)

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase313_WU_readout/{mn}_WU_readout.json') as f:
            d = json.load(f)
    except:
        continue
    
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        O_not_wu = lr['WU_gains'].get('O_not', 0)
        O_clean_R_wu = lr['WU_gains'].get('O_clean_R', 0)
        O_not_jac = lr.get('jacobian_gains', {}).get('O_not', {}).get('gain', 0)
        O_clean_R_jac = lr.get('jacobian_gains', {}).get('O_clean_R', {}).get('gain', 0)
        
        O_not_amp = O_not_jac / O_not_wu if O_not_wu > 0 else 0
        O_clean_R_amp = O_clean_R_jac / O_clean_R_wu if O_clean_R_wu > 0 else 0
        amp_ratio = O_clean_R_amp / O_not_amp if O_not_amp > 0 else 0
        
        O_not_norm = lr['direction_norms'].get('O_not', 0)
        O_clean_R_norm = lr['direction_norms'].get('O_clean_R', 0)
        norm_ratio = O_clean_R_norm / O_not_norm if O_not_norm > 0 else 0
        
        print(f"{mn:<12} {li:>6} {O_not_amp:>10.2f} {O_clean_R_amp:>14.2f} {amp_ratio:>8.3f} {norm_ratio:>22.3f}")

# C_pc1_delta_from_shared analysis
print("\n\n" + "=" * 80)
print("DELTA FROM SHARED: KEY DIRECTIONS WITH HIGHEST JACOBIAN GAIN")
print("=" * 80)

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase313_WU_readout/{mn}_WU_readout.json') as f:
            d = json.load(f)
    except:
        continue
    
    print(f"\n  {mn.upper()}:")
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        random_jac = lr.get('random_jacobian_gain', 0)
        
        # Collect all directions with their jacobian ratios
        all_jacs = {}
        for name, data in lr.get('jacobian_gains', {}).items():
            if isinstance(data, dict) and 'gain' in data:
                all_jacs[name] = data.get('ratio_to_random', 0)
        
        # Sort by ratio
        sorted_jacs = sorted(all_jacs.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_jacs[:3]
        bottom3 = sorted_jacs[-3:]
        
        print(f"    L{li}: Top3 by jac/random ratio: {[(n, f'{r:.2f}x') for n, r in top3]}")
        print(f"          Bottom3: {[(n, f'{r:.2f}x') for n, r in bottom3]}")
