import json

with open('results/phase389_per_pair_analysis/qwen3_phase389.json') as f:
    data = json.load(f)

print('=== Condition Comparison ===')
for li in ['4', '20']:
    cc = data['condition_comparison'][li]
    cm = cc['correct_mean']
    ct = cc['correct_t']
    cp = cc['correct_pos_pct']
    im = cc['incorrect_mean']
    it = cc['incorrect_t']
    ip = cc['incorrect_pos_pct']
    bc = cc['baseline_add_corr']
    print(f'L{li}:')
    print(f'  Correct:   mean={cm:+.4f}, t={ct:+.2f}, pos={cp:.0f}%')
    print(f'  Incorrect: mean={im:+.4f}, t={it:+.2f}, pos={ip:.0f}%')
    print(f'  Corr(baseline,add)={bc:.3f}')

print()
print('=== Category Summary ===')
for li in ['4', '20']:
    print(f'L{li}:')
    cs = data['category_summary'][li]
    for cat in sorted(cs.keys()):
        c = cs[cat]
        cm = c['correct_mean']
        cp = c['correct_pos_pct']
        nc = c['n_correct']
        im = c['incorrect_mean']
        ip = c['incorrect_pos_pct']
        ni = c['n_incorrect']
        print(f'  {cat:12s}: correct={cm:+.4f}({cp:.0f}%pos,n={nc}) incorrect={im:+.4f}({ip:.0f}%pos,n={ni})')

# Also analyze per-pair distribution
print()
print('=== Per-Pair Effect Distribution ===')
for li in ['4', '20']:
    pp = data['per_pair'][li]
    correct_effs = [r['add_effect'] for r in pp if r['condition'] == 'correct']
    incorrect_effs = [r['add_effect'] for r in pp if r['condition'] == 'incorrect']
    
    # Distribution
    for name, effs in [('correct', correct_effs), ('incorrect', incorrect_effs)]:
        arr = sorted(effs)
        n = len(arr)
        p10 = arr[int(n*0.1)]
        p25 = arr[int(n*0.25)]
        p50 = arr[int(n*0.5)]
        p75 = arr[int(n*0.75)]
        p90 = arr[int(n*0.9)]
        print(f'  L{li} {name}: p10={p10:+.4f} p25={p25:+.4f} p50={p50:+.4f} p75={p75:+.4f} p90={p90:+.4f}')

    # Effect by baseline_ld
    import numpy as np
    for name, effs_pairs in [('correct', [r for r in pp if r['condition']=='correct']), 
                              ('incorrect', [r for r in pp if r['condition']=='incorrect'])]:
        bl = np.array([r['baseline_compat_ld'] for r in effs_pairs])
        ae = np.array([r['add_effect'] for r in effs_pairs])
        if len(bl) > 2 and np.std(bl) > 0:
            corr = np.corrcoef(bl, ae)[0,1]
        else:
            corr = 0
        # Split by baseline positive vs negative
        pos_bl = ae[bl > 0]
        neg_bl = ae[bl <= 0]
        pos_mean = np.mean(pos_bl) if len(pos_bl) > 0 else 0
        neg_mean = np.mean(neg_bl) if len(neg_bl) > 0 else 0
        print(f'  L{li} {name}: corr(baseline,add)={corr:.3f}, pos_baseline_add={pos_mean:+.4f}(n={len(pos_bl)}), neg_baseline_add={neg_mean:+.4f}(n={len(neg_bl)})')
