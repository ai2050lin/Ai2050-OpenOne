import json, sys
model = sys.argv[1]
f = open(f'd:/AI2050/Ai2050-OpenOne/results/phase946_operator_algebra/{model}_operator_algebra.json','r',encoding='utf-8')
d = json.load(f)
mid = str(d['mid_layer'])
print(f'Model: {d["model"]}, mid={mid}')
nb = d.get('neg_baseline',{})
print(f'NEG LOO: {nb.get("loo_cos",{}).get(mid,{})}')
print(f'NEG PCA: {nb.get("pca",{}).get(mid,{})}')
dn = d.get('double_neg_closure',{}).get(mid,{})
if dn:
    print(f'DN closure: cos={dn["avg_cos"]:.4f}, pos={dn["pos_rate"]:.0%}')
aff = d.get('op_affinity_matrix',{}).get(mid,{})
if aff:
    names = aff['op_names']
    mat = aff['cosine_matrix']
    print(f'\nAffinity Matrix ({len(names)} ops):')
    for i, ni in enumerate(names):
        others = [(mat[i][j], names[j]) for j in range(len(names)) if j != i]
        others.sort(reverse=True)
        print(f'  {ni:22s}: top3={[(f"{c:+.3f}",n) for c,n in others[:3]]}')
comp = d.get('composition',{}).get(mid,{})
if comp:
    print(f'\nComposition:')
    for k, v in comp.items():
        print(f'  {k}: {v}')
