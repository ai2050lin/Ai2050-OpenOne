import json
patch = json.load(open('results/phase286_head_patching/glm4_head_patching.json'))
print('GLM4 Per-Category Top-3 Causal Heads:')
if 'per_category_top_heads' in patch:
    for cat, heads in sorted(patch['per_category_top_heads'].items()):
        top3 = sorted(heads.items(), key=lambda x: -x[1])[:3]
        ts = ", ".join([f"{h}={v:.4f}" for h, v in top3])
        print(f"  {cat:<18}: {ts}")
print()
print('GLM4 Top 15 Causal Effects:')
chs = patch['head_causal_effects']
for rank, (hk, hv) in enumerate(sorted(chs.items(), key=lambda x: -x[1]['mean'])[:15]):
    print(f"  {rank+1:>3} {hk:>12} mean={hv['mean']:.4f} std={hv['std']:.4f} n={hv['n']}")

print()
print('GLM4 Diff-vs-Causal:')
d = json.load(open('results/phase286_head_patching/glm4_head_diff.json'))
import numpy as np
dmap, cmap = {}, {}
for li, hi, md, sd, n in d['global_ranking'][:100]:
    dmap[f"L{li}_H{hi}"] = md
for hk, hv in chs.items():
    cmap[hk] = hv['mean']
common = [(dmap[hk], cmap[hk]) for hk in dmap if hk in cmap]
diffs = np.array([x[0] for x in common])
causals = np.array([x[1] for x in common])
corr = np.corrcoef(diffs, causals)[0, 1]
print(f"N={len(common)}, Pearson r={corr:.4f}")
print(f"Diff range: [{diffs.min():.3f}, {diffs.max():.3f}]")
print(f"Causal range: [{causals.min():.6f}, {causals.max():.6f}]")
