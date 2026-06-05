"""Read Phase 383 results."""
import sys, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

f = f"results/phase383_category_swap_causal/{model}_phase383.json"
d = json.load(open(f, 'r', encoding='utf-8'))

print(f"\n{'='*70}")
print(f"Phase 383 Results: {model}")
print(f"{'='*70}")

for l_str in sorted(d['results'].keys(), key=int):
    r = d['results'][l_str]
    l = r['layer']
    s = r['summary']
    
    print(f"\n--- Layer {l} (cat_R2={r['category_r2']:.4f}) ---")
    
    for key in ["clean_baseline", "corrupt_baseline", "add_cat_to_corrupt", 
                "remove_cat_from_clean", "cross_cat_swap", "same_cat_swap", "zero_cat"]:
        if key in s:
            print(f"  {key:30s}: mean={s[key]['mean']:.4f}, std={s[key]['std']:.4f}, n={s[key]['n']}")
    
    for key in ["add_cat_causal_effect", "remove_cat_causal_effect", "swap_causal_effect"]:
        if key in s:
            print(f"  {key:30s}: ", end="")
            for k, v in s[key].items():
                if isinstance(v, float):
                    print(f"{k}={v:.4f} ", end="")
            print()
