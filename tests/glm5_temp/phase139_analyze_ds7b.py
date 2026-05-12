#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(encoding='utf-8')
import json

fn = 'tests/glm5_temp/phase139_deepseek7b_jacobian_geometry_20260512_1440.json'
with open(fn, 'r', encoding='utf-8') as f:
    data = json.load(f)

mi = data['model_info']
print(f"Model: {data['model_name']}")
print(f"Info: class={mi['class']}, n_layers={mi['n_layers']}, d_model={mi['d_model']}")

# Exp A
print('\n--- Exp A: 传播比 ---')
expA = data['expA']
for pk in sorted(expA.get('perturbation_layers', {}).keys()):
    pdata = expA['perturbation_layers'][pk]
    if 'error' in pdata: continue
    eff_rank = pdata['mean_effective_rank_ratio']
    last_obs = None
    for ok in sorted(pdata.get('propagation_ratios_summary', {}).keys()):
        last_obs = ok
    if last_obs:
        ls = pdata['propagation_ratios_summary'][last_obs]
        print(f"  {pk} -> {last_obs}: mean_prop={ls['mean']:.1f}, pct>1={ls['pct_above_1']:.2f}, eff_rank={eff_rank:.3f}")

# Exp B
print('\n--- Exp B: 语义vs随机 ---')
expB = data['expB']
for op_name in ['negation_analysis', 'tense_analysis']:
    print(f'  {op_name}:')
    for layer in sorted(expB.get(op_name, {}).keys()):
        d = expB[op_name][layer]
        sem = d['semantic_logit_shift_mean']
        rand = d['random_logit_shift_mean']
        ratio = d['shift_ratio']
        print(f"    {layer}: sem={sem:.2f}, rand={rand:.2f}, ratio={ratio:.2f}")

# Exp C
print('\n--- Exp C: 归一化传播比 ---')
expC = data['expC']
for pk in sorted(expC.get('layer_results', {}).keys()):
    ld = expC['layer_results'][pk]
    print(f'  perturb@{pk}:')
    for ok in sorted(ld.keys()):
        s = ld[ok]
        print(f"    -> {ok}: norm_ratio={s['norm_ratio_mean']:.3f}, dir_preserve={s['direction_preserve_mean']:.3f}, top10%={s['top10pct_energy_ratio_mean']:.3f}")
