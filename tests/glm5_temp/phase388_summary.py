import json

for model in ['qwen3', 'deepseek7b', 'glm4']:
    with open(f'results/phase388_centroid_bootstrap/{model}_phase388.json') as f:
        data = json.load(f)
    
    print(f'=== {model} ===')
    for li in data['layers']:
        for n in data['sample_sizes']:
            s = data['summary'][str(li)][str(n)]
            marker = '+' if s['all_positive'] else ('-' if s['all_negative'] else '~')
            a = s['A_add_mean_of_means']
            sd = s['A_add_std_of_means']
            t = s['A_t_mean']
            d = s['direction_consistency']
            c = s['avg_cosine_with_full']
            print(f'  L{li} n={n}: A_add={a:+.4f}(sd={sd:.4f}) t={t:+.2f} dir={d} cos={c:.3f} {marker}')
    print()
