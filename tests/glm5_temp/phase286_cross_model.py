import json, numpy as np
for model in ['qwen3', 'glm4', 'deepseek7b']:
    d = json.load(open(f'results/phase286_head_patching/{model}_head_diff.json'))
    p = json.load(open(f'results/phase286_head_patching/{model}_head_patching.json'))
    dmap, cmap = {}, {}
    for li, hi, md, sd, n in d['global_ranking'][:100]:
        dmap[f"L{li}_H{hi}"] = md
    for hk, hv in p['head_causal_effects'].items():
        cmap[hk] = hv['mean']
    common = [(dmap[hk], cmap[hk]) for hk in dmap if hk in cmap]
    diffs = np.array([x[0] for x in common])
    causals = np.array([x[1] for x in common])
    corr = np.corrcoef(diffs, causals)[0, 1]
    print(f"{model:<12}: N={len(common):>3}, r={corr:+.4f}, diff=[{diffs.min():.1f},{diffs.max():.1f}], causal=[{causals.min():.4f},{causals.max():.4f}]")
    
    # Also: per-category head count
    cats = p.get('per_category_top_heads', {})
    n_cats = len(cats)
    all_vals = []
    for cat, heads in cats.items():
        all_vals.extend(heads.values())
    all_vals = np.array(all_vals)
    print(f"  Categories={n_cats}, head_effects: mean={np.mean(all_vals):.4f}, std={np.std(all_vals):.4f}, max={np.max(all_vals):.4f}, min={np.min(all_vals):.4f}")
    
    # Count heads with effect >0.1
    n_strong = int(np.sum(all_vals > 0.1))
    print(f"  Heads with effect>0.1: {n_strong}/{len(all_vals)}")
    print()
