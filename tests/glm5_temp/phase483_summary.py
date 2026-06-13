"""Phase 483 结果汇总分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase483_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'  {model.upper()} - Phase 483 Results')
    print(sep)
    
    # Exp1: Boundary Writers
    print(f'\n--- Exp1: Boundary Writers ---')
    for cat, d in data.get('exp1_boundary_writers', {}).items():
        print(f'  {cat}: top50_conc={d["concentration_top50"]:.3f}, top10_conc={d["concentration_top10"]:.3f}, '
              f'cos(Bc)={d["cos_top50_dir_with_bc"]:.3f}, pos={d["n_positive_neurons"]}, neg={d["n_negative_neurons"]}')
    
    # Exp2: Competition Release Matrix
    print(f'\n--- Exp2: Competition Release ---')
    for cat, d in data.get('exp2_competition_release', {}).get('release_detail', {}).items():
        target_d = d['target_delta']
        releases = d.get('competitor_releases', [])
        top = releases[0] if releases else {'category': 'none', 'delta': 0}
        print(f'  remove {cat}: target_D={target_d:.2f}, top_release={top["category"]}+{top["delta"]:.2f}, sel={d["selectivity"]:.2f}')
    
    # Exp3: Layer Formation
    print(f'\n--- Exp3: Layer Formation ---')
    for cat, d in data.get('exp3_layer_formation', {}).items():
        ne = d.get('norm_emergence_layer', '?')
        ms = d.get('max_sel_layer', '?')
        mr = d.get('max_removal_layer', '?')
        mc = d.get('max_competitor_layer', '?')
        print(f'  {cat}: norm_emergence=L{ne}, max_sel=L{ms}, max_removal=L{mr}, max_competitor=L{mc}')

print('\n\nDone.')
