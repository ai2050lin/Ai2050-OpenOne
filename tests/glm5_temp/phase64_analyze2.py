import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

R = Path('results/subspace_topology')

# ===== Part 2: 协方差稳定性详细 =====
print('='*70)
print('Part 2: COVARIANCE STABILITY (detailed)')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part2_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    
    print(f'\n--- {model} ---')
    
    # Split-half协方差稳定性
    print('  Split-half covariance stability:')
    for group, results in d['cov_stability'].items():
        if not results: continue
        cos_sims = [r['cov_cosine'] for r in results if 'cov_cosine' in r]
        sub_angles = [r.get('subspace_mean_angle', None) for r in results if 'subspace_mean_angle' in r]
        if cos_sims:
            cs_mean = sum(cos_sims)/len(cos_sims)
            sa_mean = sum(sub_angles)/len(sub_angles) if sub_angles else 'N/A'
            print(f'    {group}: cov_cos={cs_mean:.3f}, subspace_angle={sa_mean:.1f}deg' if isinstance(sa_mean, float) else f'    {group}: cov_cos={cs_mean:.3f}')
    
    # 跨组协方差
    print('  Cross-group covariance (same-axis vs cross-axis):')
    same_axis = []
    cross_axis = []
    for key, val in d['cross_group_cov_similarity'].items():
        ma = val.get('mean_angle', None)
        is_same = val.get('is_same_axis', False)
        if isinstance(ma, (int, float)):
            if is_same:
                same_axis.append(ma)
            else:
                cross_axis.append(ma)
    
    if same_axis:
        print(f'    Same-axis pairs: mean_angle={sum(same_axis)/len(same_axis):.1f}deg (n={len(same_axis)})')
    if cross_axis:
        print(f'    Cross-axis pairs: mean_angle={sum(cross_axis)/len(cross_axis):.1f}deg (n={len(cross_axis)})')

# ===== Part 3: 共享成分详细 =====
print('\n' + '='*70)
print('Part 3: SHARED COMPONENT (detailed)')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part3_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    
    print(f'\n--- {model} ---')
    
    for cat, res in d['results'].items():
        shared_toks = [t['token'] for t in res['shared_vs_global'][:5]]
        specific_toks = [t['token'] for t in res['specific_vs_highlevel'][:5]]
        # 个体差异解码
        ind_toks = {}
        for word, wd in res['individual_deltas'].items():
            ind_toks[word] = [t['token'] for t in wd['decode'][:3]]
        
        print(f'  {cat}:')
        print(f'    shared(vs global): {shared_toks}')
        print(f'    specific(vs highlevel): {specific_toks}')
        print(f'    individual deltas: {ind_toks}')
    
    # 稳定性
    print(f'  Shared stability:')
    for cat, stab in d['shared_stability'].items():
        print(f'    {cat}: mean_cos={stab["mean_cos_sim"]:.3f}')

# ===== Part 1: 哪些轴对正交？哪些重叠？ =====
print('\n' + '='*70)
print('Part 1: ORTHOGONALITY CLASSIFICATION')
print('='*70)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    f = R / f'phase64_part1_{model}.json'
    if not f.exists(): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    last_layer = str(max(int(k) for k in d['results_by_layer']))
    lr = d['results_by_layer'][last_layer]
    
    print(f'\n--- {model} (L{last_layer}) ---')
    
    # 分类: 正交(>75), 部分正交(45-75), 重叠(<45)
    orthogonal = []
    partial = []
    overlapping = []
    
    for key, val in lr['orthogonality_matrix'].items():
        ma = val['min_angle']
        if ma > 75:
            orthogonal.append((key, ma))
        elif ma > 45:
            partial.append((key, ma))
        else:
            overlapping.append((key, ma))
    
    print(f'  Orthogonal (>75deg): {len(orthogonal)} pairs')
    print(f'  Partial orth (45-75deg): {len(partial)} pairs')
    print(f'  Overlapping (<45deg): {len(overlapping)} pairs')
    
    if overlapping:
        print(f'  Top overlapping pairs:')
        for p, a in sorted(overlapping, key=lambda x: x[1])[:8]:
            print(f'    {p}: {a:.1f}deg')
    
    if orthogonal:
        print(f'  Top orthogonal pairs:')
        for p, a in sorted(orthogonal, key=lambda x: -x[1])[:5]:
            print(f'    {p}: {a:.1f}deg')
