"""Read and summarize Phase 286 results for all three models."""
import json
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        diff_path = f'results/phase286_head_patching/{model}_head_diff.json'
        patch_path = f'results/phase286_head_patching/{model}_head_patching.json'
        
        diff = json.load(open(diff_path))
        patch = json.load(open(patch_path))
        
        print(f'\n{"="*60}')
        print(f'MODEL: {model}')
        print(f'{"="*60}')
        
        # Top 15 heads by diff norm
        print(f'\n  Top 15 Heads by Diff Norm:')
        print(f'  {"Rank":>4} {"Layer":>5} {"Head":>5} {"Mean_Diff":>10} {"Std":>10} {"N":>6}')
        ranking = diff['global_ranking']
        for rank, item in enumerate(ranking[:15]):
            li, hi, md, sd, n = item
            print(f'  {rank+1:>4} L{li:>4} H{hi:>4} {md:10.3f} {sd:10.3f} {n:>6}')
        
        # Top 15 heads by causal effect
        print(f'\n  Top 15 Heads by Causal Effect:')
        print(f'  {"Rank":>4} {"Head":>12} {"Mean_Eff":>10} {"Std":>10} {"N":>6}')
        head_causal = patch['head_causal_effects']
        causal_sorted = sorted(head_causal.items(), key=lambda x: -x[1]['mean'])
        for rank, (hk, hv) in enumerate(causal_sorted[:15]):
            print(f'  {rank+1:>4} {hk:>12} {hv["mean"]:10.4f} {hv["std"]:10.4f} {hv["n"]:>6}')
        
        # Per layer summary
        print(f'\n  Layer-wise Summary (mean causal effect):')
        layer_effects = {}
        for hk, hv in head_causal.items():
            li = int(hk.split('_')[0][1:])
            if li not in layer_effects:
                layer_effects[li] = []
            layer_effects[li].append(hv['mean'])
        
        for li in sorted(layer_effects.keys()):
            effects = layer_effects[li]
            print(f'  L{li:>4}: mean={np.mean(effects):.4f}, max={np.max(effects):.4f}, n_heads={len(effects)}')
        
        # Per-category top heads
        print(f'\n  Per-Category Top-3 Causal Heads:')
        if 'per_category_top_heads' in patch:
            for cat, heads in sorted(patch['per_category_top_heads'].items()):
                top3 = sorted(heads.items(), key=lambda x: -x[1])[:3]
                ts = ", ".join([f"{h}={v:.4f}" for h, v in top3])
                print(f'    {cat:<18}: {ts}')
        
        # Correlation: diff norm vs causal effect
        print(f'\n  Diff-vs-Causal Correlation:')
        diff_map = {}
        for li, hi, md, sd, n in ranking[:100]:
            diff_map[f"L{li}_H{hi}"] = md
        causal_map = {hk: hv['mean'] for hk, hv in head_causal.items()}
        
        common = []
        for hk in diff_map:
            if hk in causal_map:
                common.append((diff_map[hk], causal_map[hk]))
        
        if common:
            diffs = np.array([x[0] for x in common])
            causals = np.array([x[1] for x in common])
            corr = np.corrcoef(diffs, causals)[0, 1]
            print(f'    N={len(common)}, Pearson r={corr:.4f}')
            print(f'    Diff range: [{diffs.min():.3f}, {diffs.max():.3f}]')
            print(f'    Causal range: [{causals.min():.6f}, {causals.max():.6f}]')
        
    except Exception as e:
        print(f'{model} ERROR: {e}')
        import traceback
        traceback.print_exc()
