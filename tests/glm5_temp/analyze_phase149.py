import json, numpy as np

with open('tests/glm5_temp/phase149_qwen3_20260513_0923.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== Exp 2: Three-Level Direction Summary ===')
for dir_type in ['null', 'row']:
    print(f'\n--- {dir_type.upper()}-space input ---')
    layer_avgs = {}
    for key, ld in data['exp2_three_level_direction'].items():
        if f'_{dir_type}' not in key:
            continue
        for i, li in enumerate(ld['layers']):
            if li not in layer_avgs:
                layer_avgs[li] = {'gc': [], 'nr': [], 'nic': [], 'ric': []}
            layer_avgs[li]['gc'].append(ld['global_cos'][i])
            layer_avgs[li]['nr'].append(ld['null_ratio'][i])
            layer_avgs[li]['nic'].append(ld['null_internal_cos'][i])
            layer_avgs[li]['ric'].append(ld['row_internal_cos'][i])
    
    header = f"  {'Layer':>5}  {'global_cos':>10}  {'null_ratio':>10}  {'null_int_cos':>13}  {'row_int_cos':>12}"
    print(header)
    for li in sorted(layer_avgs.keys()):
        a = layer_avgs[li]
        print(f"  L{li:>4d}  {np.mean(a['gc']):>10.4f}  {np.mean(a['nr']):>10.4f}  "
              f"{np.mean(a['nic']):>13.4f}  {np.mean(a['ric']):>12.4f}")

# Exp1 汇总
print('\n=== Exp 1: Token Coupling Summary ===')
for measure_key in ['mL1', 'mL5', 'mL32', 'mL36']:
    self_resps = []
    cross_resps = []
    for key, cd in data['exp1_token_coupling'].items():
        if measure_key not in key:
            continue
        self_resps.append(cd['self_response'])
        cross_resps.append(cd['avg_cross_response'])
    if self_resps:
        ratio = np.mean(self_resps) / (np.mean(cross_resps) + 1e-10)
        print(f"  Measure {measure_key}: avg_self={np.mean(self_resps):.3f}, avg_cross={np.mean(cross_resps):.3f}, ratio={ratio:.1f}x (n={len(self_resps)})")

# 关键发现
print('\n=== KEY FINDINGS ===')
print('1. global_cos drops RAPIDLY: 1.0 at L1 -> ~0.2 by L5 -> ~0.03 by L33')
print('2. null_int_cos tracks global_cos closely -> null-space internal direction ALSO lost')
print('3. null_ratio stays ~0.92 throughout -> consistent with Phase 148')
print('4. row_int_cos stays near 0 for null-input, ~0.3-0.7 early then decays for row-input')
print('5. CONTRADICTS Phase 147 cos~0.999 -> eps=2.0 causes nonlinear mixing')
