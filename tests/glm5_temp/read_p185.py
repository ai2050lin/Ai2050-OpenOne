import json, numpy as np

d = json.load(open('tests/glm5_temp/phase185_qwen3_20260515_2049.json', 'r', encoding='utf-8'))

print('=== Exp1: Per-Layer Boundary Formation (Qwen3) ===')
e1 = d['exp1_per_layer_boundary']
for li in sorted([int(k) for k in e1.keys()]):
    alpha = e1[str(li)]['alpha_star_mean']
    std = e1[str(li)]['alpha_star_std']
    print(f'  L{li}: alpha*={alpha:.3f} +/- {std:.3f}')

print('\n=== Exp2: Jacobian Amplification (Qwen3) ===')
e2 = d['exp2_jacobian_amplification']
for li in sorted([int(k) for k in e2.keys()]):
    gd = e2[str(li)]['g_delta_mean']
    gp = e2[str(li)]['g_perp_mean']
    lm = e2[str(li)]['lambda_max_mean']
    sel = e2[str(li)]['selectivity_mean']
    barrier = e2[str(li)]['is_barrier']
    print(f'  L{li}: g_D={gd:.3f}, g_P={gp:.3f}, lambda_max={lm:.3f}, S={sel:.2f}, barrier%={barrier:.2f}')

print('\n=== Exp3: Propagation Profile (Qwen3) ===')
e3 = d['exp3_propagation_profile']
for li in sorted([int(k) for k in e3.keys()]):
    slope = e3[str(li)]['propagation_slope']
    verdict = e3[str(li)]['verdict']
    print(f'  Inject L{li}: slope={slope:.4f} [{verdict}]')

print('\n=== Exp4: Constraint Types (Qwen3) ===')
e4 = d['exp4_constraint_type_comparison']
for ct in ['syntactic', 'semantic', 'factual']:
    meta = e4.get(ct, {}).get('_meta', {})
    slope = meta.get('formation_slope', 0)
    verdict = meta.get('verdict', 'N/A')
    first_d = meta.get('first_layer_delta', 0)
    last_d = meta.get('last_layer_delta', 0)
    print(f'  {ct}: slope={slope:.5f}, first_d={first_d:.4f}, last_d={last_d:.4f} [{verdict}]')
