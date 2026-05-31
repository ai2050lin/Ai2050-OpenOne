import json
for mn in ['qwen3']:
    with open(f'results/phase313_WU_readout/{mn}_WU_readout.json') as f:
        d = json.load(f)
    print(f"=== {mn} ===")
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        print(f"\n--- L{li} ---")
        # W_U gains
        random_wu = lr.get('random_WU_gain', 0)
        print(f"  W_U gains (ratio to random={random_wu:.2f}):")
        for name in ['O_not', 'O_clean_R', 'O_clean_RC', 'R', 'C_pc1', 'A', 'shared_pc1']:
            g = lr['WU_gains'].get(name, 0)
            r = g / random_wu if random_wu > 0 else 0
            print(f"    {name}: gain={g:.2f}, ratio={r:.2f}x")
        
        # Target logits
        print(f"  Target logits (W_U@v[target]/||v||):")
        for name in ['O_not', 'O_clean_R', 'R', 'A']:
            tl = lr.get('target_logits', {}).get(name, {})
            not_v = tl.get('not', 0)
            happy_v = tl.get('happy', 0)
            sad_v = tl.get('sad', 0)
            print(f"    {name}: not={not_v:.4f}, happy={happy_v:.4f}, sad={sad_v:.4f}")
        
        # Jacobian
        random_jac = lr.get('random_jacobian_gain', 0)
        print(f"  Jacobian gains (ratio to random={random_jac:.2f}):")
        for name in ['O_not', 'O_clean_R', 'O_clean_RC', 'R', 'C_pc1', 'A']:
            jg = lr.get('jacobian_gains', {}).get(name, {}).get('gain', 0)
            rj = lr.get('jacobian_gains', {}).get(name, {}).get('ratio_to_random', 0)
            dn = lr.get('jacobian_gains', {}).get(name, {}).get('delta_not', 0)
            print(f"    {name}: jac={jg:.2f}, ratio={rj:.2f}x, delta_not={dn:.4f}")
        
        # Amplification
        if 'amplification_source' in lr:
            amp = lr['amplification_source']
            print(f"  AMPLIFICATION: O_not jac/wu={amp['O_not_jac_over_wu']:.4f}, "
                  f"O_clean_R jac/wu={amp['O_clean_R_jac_over_wu']:.4f}, "
                  f"delta_amplified={amp['delta_amplified_more']}, "
                  f"ratio={amp['amplification_ratio']:.4f}")
