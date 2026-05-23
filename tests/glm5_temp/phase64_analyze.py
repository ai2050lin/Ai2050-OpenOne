import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

R = Path('results/subspace_topology')

# ===== Part 1: 12轴正交性 =====
print('='*70)
print('Part 1: 12-AXIS ORTHOGONALITY')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part1_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    last_layer = str(max(int(k) for k in d['results_by_layer']))
    lr = d['results_by_layer'][last_layer]
    
    print(f'\n--- {model} (L{last_layer}) ---')
    
    min_angles = []
    low_orth_pairs = []
    for key, val in lr['orthogonality_matrix'].items():
        ma = val['min_angle']
        min_angles.append(ma)
        if ma < 60:
            low_orth_pairs.append((key, ma))
    
    print(f'  Orthogonality: {len(min_angles)} pairs, min_angle range=[{min(min_angles):.1f}, {max(min_angles):.1f}]')
    print(f'  Mean min_angle: {sum(min_angles)/len(min_angles):.1f} deg')
    print(f'  Pairs with min_angle < 60: {len(low_orth_pairs)}/{len(min_angles)}')
    if low_orth_pairs:
        for p, a in sorted(low_orth_pairs, key=lambda x: x[1])[:10]:
            print(f'    {p}: {a:.1f} deg')
    
    print(f'  Axis k90:')
    for axis, kd in sorted(lr['axis_k90'].items()):
        print(f'    {axis}: k90={kd["k90"]}')

# ===== Part 2: 协方差稳定性 =====
print('\n' + '='*70)
print('Part 2: COVARIANCE STABILITY')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part2_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    
    print(f'\n--- {model} ---')
    
    # 协方差稳定性 (split-half)
    for group, results in d['cov_stability'].items():
        if not results: continue
        cos_sims = [r['cov_cosine'] for r in results if 'cov_cosine' in r]
        sub_angles = [r.get('subspace_mean_angle', None) for r in results if 'subspace_mean_angle' in r]
        print(f'  {group}: cov_cosine={sum(cos_sims)/len(cos_sims):.3f}, subspace_angle={sum(sub_angles)/len(sub_angles):.1f}deg' if cos_sims and sub_angles else f'  {group}: insufficient data')
    
    # 跨组协方差相似度
    print(f'  Cross-group covariance similarity:')
    for key, val in d['cross_group_cov_similarity'].items():
        is_same = val.get('is_same_axis', False)
        ma = val.get('mean_angle', 'N/A')
        tag = 'SAME-AXIS' if is_same else ''
        print(f'    {key}: mean_angle={ma:.1f}deg {tag}' if isinstance(ma, float) else f'    {key}: {val}')

# ===== Part 3: 共享成分 =====
print('\n' + '='*70)
print('Part 3: SHARED COMPONENT DECOMPOSITION')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part3_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    
    print(f'\n--- {model} ---')
    
    for cat, res in d['results'].items():
        shared_top3 = [t['token'] for t in res['shared_vs_global'][:3]]
        specific_top3 = [t['token'] for t in res['specific_vs_highlevel'][:3]]
        print(f'  {cat}: shared={shared_top3}, specific={specific_top3}, '
              f'shared_norm={res["shared_norm"]:.1f}, specific_norm={res["specific_norm"]:.1f}')
    
    # 共享成分稳定性
    print(f'  Shared component stability:')
    for cat, stab in d['shared_stability'].items():
        print(f'    {cat}: mean_cos_sim={stab["mean_cos_sim"]:.3f}, min_cos_sim={stab["min_cos_sim"]:.3f}')

# ===== Part 4: 维度估算 =====
print('\n' + '='*70)
print('Part 4: DIMENSION ESTIMATION')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part4_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    
    print(f'\n--- {model} (d={d["d_model"]}) ---')
    for layer_key in sorted(d['results_by_layer'].keys(), key=lambda x: int(x)):
        lr = d['results_by_layer'][layer_key]
        tdu = lr['total_dim_used']
        gd = lr.get('global_dimensionality', {})
        or_k90 = gd.get('overlap_ratio_k90', 'N/A')
        or_str = f'{or_k90:.2f}' if isinstance(or_k90, (int, float)) else 'N/A'
        print(f'  L{layer_key}: sum_k90={tdu["sum_k90"]}, global_k90={gd.get("global_k90","N/A")}, '
              f'overlap_ratio={or_str}, util={tdu["utilization_k90"]:.1%}')
        
        po = lr.get('pairwise_overlap', {})
        if po:
            top5 = sorted(po.items(), key=lambda x: -x[1])[:5]
            print(f'    Top overlap pairs: {[(k, round(v,3)) for k,v in top5]}')
