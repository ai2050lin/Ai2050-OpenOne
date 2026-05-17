"""Phase 217 结果分析脚本"""
import json
import numpy as np

with open('d:/Ai2050/TransformerLens-Project/tests/glm5_temp/phase217_architecture_vs_language_results.json') as f:
    data = json.load(f)

# 1B: 每层的约束KL对比
print('=== 1B: 约束方向KL (逐层对比) ===')
trained_kls_by_layer = {}
random_kls_by_layer = {}

for pair_key in data['1B_trained']:
    for lr in data['1B_trained'][pair_key]:
        l = lr['layer']
        if l not in trained_kls_by_layer:
            trained_kls_by_layer[l] = []
        trained_kls_by_layer[l].append(lr['kl_constraint'])

for pair_key in data['1B_random']:
    for lr in data['1B_random'][pair_key]:
        l = lr['layer']
        if l not in random_kls_by_layer:
            random_kls_by_layer[l] = []
        random_kls_by_layer[l].append(lr['kl_constraint'])

print('Layer | Trained_KL | Random_KL | Ratio(T/R)')
for l in sorted(trained_kls_by_layer.keys()):
    t = np.mean(trained_kls_by_layer[l])
    r = np.mean(random_kls_by_layer[l]) if l in random_kls_by_layer else 0
    ratio = t / max(r, 1e-10)
    print(f'  {l:2d}  |  {t:.4f}   |  {r:.4f}  | {ratio:.2f}')

# 1A: 等价类宽度
print('\n=== 1A: 等价类宽度 (KL随层变化) ===')
for model_key in ['1A_trained', '1A_random']:
    label = model_key.replace('1A_', '')
    print(f'\n{label} model:')
    for layer_key in sorted(data[model_key].keys()):
        for sr in data[model_key][layer_key]:
            s = sr['scale']
            kl = sr['avg_kl']
            if s in [0.1, 0.5, 1.0]:
                print(f'  {layer_key}: scale={s:.1f} -> KL={kl:.6f}')

# Row space proxy
print('\n=== Row space proxy ===')
trained_rsp = {}
random_rsp = {}
for pair_key in data['1B_trained']:
    for lr in data['1B_trained'][pair_key]:
        l = lr['layer']
        if l not in trained_rsp:
            trained_rsp[l] = []
        trained_rsp[l].append(lr['row_space_proxy'])
for pair_key in data['1B_random']:
    for lr in data['1B_random'][pair_key]:
        l = lr['layer']
        if l not in random_rsp:
            random_rsp[l] = []
        random_rsp[l].append(lr['row_space_proxy'])

print('Layer | Trained_RSP | Random_RSP')
for l in [0, 7, 14, 21, 27]:
    if l in trained_rsp and l in random_rsp:
        t = np.mean(trained_rsp[l])
        r = np.mean(random_rsp[l])
        print(f'  {l:2d}  |  {t:.4f}    |  {r:.4f}')

# 1C
print('\n=== 1C: W_U结构 ===')
for mk in ['1C_trained', '1C_random']:
    r = data[mk]
    label = mk.replace('1C_', '')
    print(f'{label}: eff_rank={r["effective_rank"]}, null_frac={r["null_space_frac"]:.4f}, rank_95={r["rank_95"]}, top5_sv={[round(x,2) for x in r["top_20_sv"][:5]]}')
