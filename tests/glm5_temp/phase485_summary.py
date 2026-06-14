"""Phase 485结果摘要"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase485_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{"="*70}')
    print(f'  {model.upper()} - Phase 485 Results')
    print(f'{"="*70}')
    
    # Exp1
    print(f'\n--- Exp1: Attention Writer ---')
    exp1 = data.get('exp1_attn_writer', {})
    if 'error' in exp1:
        print(f'  ERROR: {exp1["error"][:100]}')
    else:
        for cat, d in exp1.items():
            attn = d.get('full_attn_ablation', {})
            mlp = d.get('full_mlp_ablation', {})
            cov = d.get('coverage_ratio', 0)
            print(f'  {cat} L{d["best_layer"]}:')
            print(f'    Attn: target_D={attn.get("target_delta",0):.2f}, cos_remove={attn.get("cos_with_direction_remove",0):.3f}')
            print(f'    MLP:  target_D={mlp.get("target_delta",0):.2f}, cos_remove={mlp.get("cos_with_direction_remove",0):.3f}')
            print(f'    Coverage: attn+mlp/direction = {cov:.1%}')
            print(f'    cos(diff_attn, Bc) = {d.get("cos_diff_attn_bc",0):.3f}')
    
    # Exp2
    print(f'\n--- Exp2: MLP Amplitude Closure ---')
    exp2 = data.get('exp2_mlp_amplitude', {})
    for cat, d in exp2.items():
        dir_target = d.get('direction_remove_target', 0)
        abl = d.get('ablation_results', {})
        print(f'  {cat}: direction_target={dir_target:.2f}')
        for k_name in ['k=5', 'k=10', 'k=50', 'k=100', 'k=200', 'k=500']:
            if k_name in abl:
                kd = abl[k_name]
                print(f'    {k_name}: target_D={kd["target_delta"]:.2f}, amp={kd["amplitude_ratio"]:.1%}, cos={kd["cos_with_direction_remove"]:.3f}')
    
    # Exp3
    print(f'\n--- Exp3: Relation Small Scale ---')
    exp3 = data.get('exp3_relation_small_scale', {})
    for cat, d in exp3.items():
        cons = d.get('cross_relation_consistency', {})
        print(f'  {cat}:')
        for scale_key in ['scale_0.05', 'scale_0.1', 'scale_0.2', 'scale_0.5', 'scale_1.0']:
            if scale_key in cons:
                c = cons[scale_key]
                print(f'    {scale_key}: mean_delta={c["delta_mean"]:.2f}, range={c["delta_range"]:.2f}, rel_range={c["relative_range"]:.2%}')
    
    # Exp4
    print(f'\n--- Exp4: Format Removal ---')
    exp4 = data.get('exp4_format_removal', {})
    for cat, d in exp4.items():
        orig = d.get('original_remove', {})
        clean = d.get('clean_remove', {})
        imp = d.get('improvement', {})
        print(f'  {cat}: cos(Bc,fmt)={d.get("cos_bc_format",0):.3f}')
        print(f'    Orig: target_D={orig.get("target_delta",0):.2f}, sel={orig.get("selectivity",0):.2f}, release={orig.get("max_competitor_release",0):.2f}')
        print(f'    Clean: target_D={clean.get("target_delta",0):.2f}, sel={clean.get("selectivity",0):.2f}, release={clean.get("max_competitor_release",0):.2f}')
        print(f'    Improvement: sel_ratio={imp.get("selectivity_ratio",0):.2f}, target_pres={imp.get("target_preservation",0):.2f}')
