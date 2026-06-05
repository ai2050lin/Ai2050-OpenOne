"""Read Phase 383b results."""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
suffix = sys.argv[2] if len(sys.argv) > 2 else "383b"

f = f"results/phase383_category_swap_causal/{model}_phase{suffix}.json"
d = json.load(open(f, 'r', encoding='utf-8'))

for l_str in sorted(d['results'].keys(), key=int):
    r = d['results'][l_str]
    s = r['summary']
    print(f'L{r["layer"]}: cat_R2_raw={r.get("cat_r2_raw",0):.4f}, cat_R2_norm={r.get("cat_r2_post_rmsnorm",0):.4f}')
    for key in ['add_cat_effect', 'remove_cat_effect', 'swap_effect',
                'add_cat_causal_effect', 'remove_cat_causal_effect', 'swap_causal_effect']:
        if key in s:
            print(f'  {key}: ', end='')
            for k, v in s[key].items():
                if isinstance(v, float):
                    print(f'{k}={v:.3f} ', end='')
            print()
