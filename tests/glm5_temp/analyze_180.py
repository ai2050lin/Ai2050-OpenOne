import json
d = json.load(open('tests/glm5_temp/phase180_qwen3_20260515_1312.json','r',encoding='utf-8'))

sample_layers = [1,5,10,15,20,25,30,36]
for cat in ['grammar','physical','animacy','causal']:
    cd = d['exp1_feasible_region'][cat]
    print(f'\n=== {cat} ===')
    print('  Layer | Corr_H | Incorr_H | Feasible | Margin  | Exp_P')
    for li in sample_layers:
        s = str(li)
        if s in cd:
            c = cd[s]
            print(f'  L{li:3d} | {c["correct_entropy"]:6.2f} | {c["incorrect_entropy"]:8.2f} | {c["correct_feasible"]:7.1f} | {c["correct_margin"]:6.2f} | {c["correct_expected_prob"]:.6f}')

print('\n=== Trajectory Topology ===')
topo = d['exp3_topology']
para_cos = topo['paraphrase_cosine']
rand_cos = topo['random_cosine']
print(f'  Paraphrase cosine: L1={para_cos.get("1",0):.3f} L18={para_cos.get("18",0):.3f} L36={para_cos.get("36",0):.3f}')
print(f'  Random cosine: L1={rand_cos.get("1",0):.3f} L18={rand_cos.get("18",0):.3f} L36={rand_cos.get("36",0):.3f}')

print('\n=== Bifurcation ===')
bif = d['exp2_bifurcation']
for cat in ['grammar','physical','animacy','causal']:
    b = bif[cat]
    print(f'  {cat}: PhaseTransL={b["phase_transition_layer"]}, Sharpness={b["bifurcation_sharpness"]:.4f}')
