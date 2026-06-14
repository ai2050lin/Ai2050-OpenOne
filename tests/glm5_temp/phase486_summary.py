"""Phase 486 结果摘要"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

SEP = "=" * 70

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase486_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{SEP}")
    print(f"  {model.upper()} - Phase 486 Results")
    print(f"{SEP}")
    
    # Exp1
    print(f"\n--- Exp1: Cross-Layer Accumulation Profile ---")
    exp1 = data.get('exp1_cross_layer_accumulation', {})
    if 'error' in exp1:
        print(f"  ERROR: {exp1['error'][:100]}")
    else:
        for cat, d in exp1.items():
            if isinstance(d, dict) and 'layer_profile' in d:
                profile = d['layer_profile']
                attn_peak = d.get('attn_peak_layer', '?')
                mlp_peak = d.get('mlp_peak_layer', '?')
                resid_peak = d.get('resid_peak_layer', '?')
                print(f"  {cat} L{d['best_layer']}: attn_peak=L{attn_peak}, mlp_peak=L{mlp_peak}, resid_peak=L{resid_peak}")
                # Show all layers sorted by mlp_proj_diff absolute
                sorted_layers = sorted(profile.keys(), key=lambda l: abs(profile[l]['mlp_proj_diff']), reverse=True)
                for l in sorted_layers[:8]:
                    p = profile[l]
                    print(f"    L{l:>2s}: attn={p['attn_proj_diff']:+.2f}, mlp={p['mlp_proj_diff']:+.2f}, resid={p['resid_proj_diff']:+.2f}")
    
    # Exp2
    print(f"\n--- Exp2: Multi-Layer MLP Ablation ---")
    exp2 = data.get('exp2_multi_layer_ablation', {})
    if 'error' in exp2:
        print(f"  ERROR: {exp2['error'][:100]}")
    else:
        for cat, d in exp2.items():
            if isinstance(d, dict) and 'ablation_results' in d:
                dir_target = d.get('direction_remove_target', 0)
                print(f"  {cat}: direction_remove_target={dir_target:.2f}")
                abl = d['ablation_results']
                for abl_name in ['single', 'pm5', 'pm10', 'full_span']:
                    if abl_name in abl:
                        a = abl[abl_name]
                        print(f"    {abl_name:12s}: target_D={a['target_delta']:.2f}, amp={a['amplitude_ratio']:.1%}, cos={a['cos_with_direction_remove']:.3f}")
    
    # Exp3
    print(f"\n--- Exp3: Format Subspace Per Layer ---")
    exp3 = data.get('exp3_format_subspace_per_layer', {})
    for cat, d in exp3.items():
        if isinstance(d, dict) and 'format_profile' in d:
            fp = d['format_profile']
            print(f"  {cat} L{d['best_layer']}:")
            for l in sorted(fp.keys(), key=int):
                p = fp[l]
                print(f"    L{l:>2s}: cos_top1={p['cos_bc_format_top1']:.3f}, cos_top2={p['cos_bc_format_top2']:.3f}, energy_top3={p['format_energy_top3']:.3f}")
    
    # Exp4
    print(f"\n--- Exp4: Cross-Layer Relation Invariance ---")
    exp4 = data.get('exp4_cross_layer_relation_invariance', {})
    for cat, d in exp4.items():
        if isinstance(d, dict) and 'layer_invariance' in d:
            li = d['layer_invariance']
            print(f"  {cat} L{d['best_layer']} scale={d['scale']}:")
            for l in sorted(li.keys(), key=int):
                p = li[l]
                print(f"    L{l:>2s}: mean_delta={p['delta_mean']:.2f}, range={p['delta_range']:.2f}, rel_range={p['relative_range']:.2%}")
