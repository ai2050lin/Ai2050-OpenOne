import json
import numpy as np

for model in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'tests/glm5_temp/phase232_{model}_results.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        print(f'{model}: NO FILE')
        continue
    
    c = d['expC']
    lr = c['layer_results']
    best_layer = c.get('best_patch_layer')
    best_kl = c.get('best_patch_kl', 0)
    
    print(f'\n=== {model} ExpC ===')
    print(f'Best patch layer: L{best_layer}, KL={best_kl:.4f}')
    
    # Each layer's value is a list of KL values (one per pair)
    for k in sorted(lr.keys(), key=int):
        v = lr[k]
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            mean_kl = np.mean(v)
            if mean_kl > 0.01:
                print(f'  L{k}: mean_KL={mean_kl:.4f} (n={len(v)})')
        elif isinstance(v, dict):
            mean_kl = v.get('mean_kl', 0)
            if mean_kl > 0.01:
                print(f'  L{k}: KL={mean_kl:.4f}')
    
    # Try first few raw values
    print(f'  Sample raw data:')
    for k in ['0', '1', '2', '3']:
        if k in lr:
            v = lr[k]
            print(f'    L{k}: type={type(v).__name__}, len={len(v) if isinstance(v, (list, dict)) else "N/A"}, first={v[0] if isinstance(v, list) and len(v) > 0 else "N/A"}')
