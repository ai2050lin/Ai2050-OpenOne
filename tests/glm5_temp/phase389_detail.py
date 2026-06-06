import json
import numpy as np

with open('results/phase389_per_pair_analysis/qwen3_phase389.json') as f:
    data = json.load(f)

for li in ['4', '20']:
    if li not in data['condition_comparison']:
        continue
    cc = data['condition_comparison'][li]
    cm = cc['correct_mean']
    ct = cc['correct_t']
    cp = cc['correct_pos_pct']
    im = cc['incorrect_mean']
    it2 = cc['incorrect_t']
    ip = cc['incorrect_pos_pct']
    bc = cc['baseline_add_corr']
    print(f'L{li}:')
    print(f'  Correct:   mean={cm:+.4f}, t={ct:+.2f}, pos={cp:.0f}%')
    print(f'  Incorrect: mean={im:+.4f}, t={it2:+.2f}, pos={ip:.0f}%')
    print(f'  Corr(baseline,add)={bc:.3f}')
    
    if li in data['category_summary']:
        cs = data['category_summary'][li]
        cats_sorted = sorted(cs.items(), key=lambda x: abs(x[1]['correct_mean']), reverse=True)
        for cat, c in cats_sorted:
            cm2 = c['correct_mean']
            cp2 = c['correct_pos_pct']
            im2 = c['incorrect_mean']
            ip2 = c['incorrect_pos_pct']
            print(f'    {cat:12s}: C={cm2:+.4f}({cp2:.0f}%) I={im2:+.4f}({ip2:.0f}%)')

# Check: for categories where correct effect is positive, is incorrect always negative?
print('\n=== Correct/Incorrect Symmetry Check ===')
for li in ['4', '20']:
    if li not in data['category_summary']:
        continue
    cs = data['category_summary'][li]
    for cat, c in cs.items():
        cm2 = c['correct_mean']
        im2 = c['incorrect_mean']
        sym = 'SYMMETRIC' if (cm2 > 0 and im2 < 0) or (cm2 < 0 and im2 > 0) else 'ASYMMETRIC'
        zero = 'ZERO' if abs(cm2) < 0.001 and abs(im2) < 0.001 else ''
        print(f'  L{li} {cat:12s}: C={cm2:+.4f} I={im2:+.4f} -> {sym} {zero}')
