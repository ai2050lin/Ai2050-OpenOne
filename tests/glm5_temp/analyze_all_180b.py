import json

models = {
    'qwen3': 'tests/glm5_temp/phase180_qwen3_20260515_1312.json',
    'glm4': 'tests/glm5_temp/phase180_glm4_20260515_1327.json',
    'deepseek7b': 'tests/glm5_temp/phase180_deepseek7b_20260515_1336.json',
}

cats = ['grammar', 'physical', 'animacy', 'causal']

for mname, path in models.items():
    d = json.load(open(path, 'r', encoding='utf-8'))
    n_layers = d['n_layers']
    
    # Choose sample layers based on model
    if n_layers <= 28:
        sl = [1, 5, 10, 15, 20, 25, 28]
    elif n_layers <= 36:
        sl = [1, 5, 10, 15, 20, 25, 30, 36]
    else:
        sl = [1, 5, 10, 15, 20, 25, 30, 40]
    
    print(f"\n{'='*60}")
    print(f"  {mname} (n_layers={n_layers})")
    print(f"{'='*60}")
    
    for cat in cats:
        cd = d['exp1_feasible_region'].get(cat, {})
        print(f"\n  {cat}:")
        for li in sl:
            s = str(li)
            if s in cd:
                c = cd[s]
                print(f"    L{li:3d}: H_corr={c['correct_entropy']:5.2f} H_inc={c['incorrect_entropy']:5.2f} "
                      f"feas={c['correct_feasible']:5.0f} margin={c['correct_margin']:6.2f}")
    
    # Trajectory topology
    topo = d.get('exp3_topology', {})
    para_cos = topo.get('paraphrase_cosine', {})
    rand_cos = topo.get('random_cosine', {})
    
    print(f"\n  Trajectory Topology:")
    for li in sl:
        s = str(li)
        if s in para_cos and s in rand_cos:
            diff = float(para_cos[s]) - float(rand_cos[s])
            print(f"    L{li:3d}: para={float(para_cos[s]):.3f} rand={float(rand_cos[s]):.3f} diff={diff:.3f}")
