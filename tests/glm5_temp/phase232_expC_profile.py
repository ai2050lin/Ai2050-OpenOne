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
    
    print(f'\n=== {model} ExpC: Activation Patching Profile ===')
    print(f'Best layer: L{best_layer} (KL={best_kl:.4f})')
    print(f'{"Layer":<6} {"Mean_KL":<10} {"Mean_Cos":<10} {"Top1_Chg%":<10}')
    print('-' * 36)
    
    for k in sorted(lr.keys(), key=int):
        v = lr[k]
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            kl_vals = [x.get('kl_vs_affirm', 0) for x in v]
            cos_vals = [x.get('cosine_vs_negated', 0) for x in v]
            top1_chg = [x.get('top1_changed', 0) for x in v]
            
            mean_kl = np.mean(kl_vals)
            mean_cos = np.mean(cos_vals)
            pct_top1 = np.mean(top1_chg) * 100
            
            marker = " <<< BEST" if k == str(best_layer) else ""
            print(f'L{k:<4} {mean_kl:<10.4f} {mean_cos:<10.4f} {pct_top1:<10.1f}{marker}')
