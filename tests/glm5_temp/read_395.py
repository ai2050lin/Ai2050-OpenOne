"""Quick reader for Phase 395 results"""
import json
for model in ['qwen3','deepseek7b','glm4']:
    with open(f'results/phase395_denoised_l2/{model}_phase395.json') as f:
        data = json.load(f)
    print(f'\n=== {model} ===')
    for li, vr in data['per_layer'].items():
        print(f'  Layer {li}:')
        for ver in ['L1_category','L2_original','L2_crossfit','L2_cf_lam0.2','L2_cf_lam0.5','L2_cf_lam2.0']:
            if ver not in vr:
                continue
            r = vr[ver]
            cats = r.get('category_breakdown',{})
            parts = []
            for cat in ['moisture','color','size']:
                if cat in cats:
                    cb = cats[cat]
                    parts.append(f'{cat}={cb["mechanism"]}(T{cb["target_delta_mean"]:+.3f}C{cb["competitor_delta_mean"]:+.3f}dmg{cb.get("damage_ratio",0):.2f})')
            print(f'    {ver:18s}: ' + ' | '.join(parts))
