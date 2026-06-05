"""Read Phase 384 results."""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
f = f'results/phase384_obj_residualized_category/{model}_phase384.json'
d = json.load(open(f, 'r', encoding='utf-8'))
print('Model:', d['model'])
print('N pairs:', d['n_pairs'])
print()

# Partial R2
for l_str in sorted(d['partial_r2'].keys(), key=int):
    p = d['partial_r2'][l_str]
    print(f'Layer {l_str}:')
    ind = p.get('individual_r2', {})
    uni = p.get('unique_r2', {})
    shr = p.get('shared_r2', {})
    ppv = p.get('perm_pvalues', {})
    for k in sorted(ind.keys(), key=lambda x: -ind.get(x, 0)):
        print(f'  {k:20s}: indiv={ind.get(k,0):.4f}  unique={uni.get(k,0):.4f}  shared={shr.get(k,0):.4f}  p={ppv.get(k,1):.4f}')
    print(f'  Total R2: {p.get("total_r2",0):.4f}')
    print()

# Causal test
for l_str in sorted(d['causal_test'].keys(), key=int):
    r = d['causal_test'][l_str]
    print(f'Layer {l_str} Causal:')
    print(f'  R2: raw={r["r2_raw"]:.4f}, resid={r["r2_resid"]:.4f}')
    print(f'  Acc: raw={r["acc_raw"]:.4f}, resid={r["acc_resid"]:.4f}')
    ratio = r["r2_resid"] / max(r["r2_raw"], 1e-10)
    print(f'  Residualization ratio: {ratio:.4f}')
    for stype in ['raw', 'clean']:
        ae = r.get(f'{stype}_add_effect', {})
        re = r.get(f'{stype}_remove_effect', {})
        se = r.get(f'{stype}_swap_effect', {})
        print(f'  {stype:5s} add:  mean={ae.get("mean",0):+.4f} t={ae.get("t",0):.2f} n={ae.get("n",0)}')
        print(f'  {stype:5s} rem:  mean={re.get("mean",0):+.4f} t={re.get("t",0):.2f} n={re.get("n",0)}')
        if se:
            print(f'  {stype:5s} swap: cross={se.get("cross_mean",0):+.4f} same={se.get("same_mean",0):+.4f} diff={se.get("diff",0):+.4f} t={se.get("diff_t",0):.2f}')
