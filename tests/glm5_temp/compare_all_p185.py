import json, numpy as np, glob

print("="*70)
print("PHASE 185: ALL MODELS COMPARISON")
print("="*70)

all_data = {}
for model_name in ['qwen3', 'glm4', 'deepseek7b']:
    files = glob.glob(f'tests/glm5_temp/phase185_{model_name}_*.json')
    if not files:
        continue
    d = json.load(open(files[-1], 'r', encoding='utf-8'))
    all_data[model_name] = d

# === KEY TABLE: Jacobian Amplification ===
print("\n" + "="*70)
print("TABLE 1: Jacobian Amplification (g_delta, lambda_max)")
print("="*70)
print(f"{'Layer':<6} | {'Qwen3 g_D':>10} {'lam':>6} | {'GLM4 g_D':>10} {'lam':>6} | {'DS7B g_D':>10} {'lam':>6}")
print("-"*70)

e2_all = {m: all_data[m]['exp2_jacobian_amplification'] for m in all_data}
all_layers = set()
for m in e2_all:
    all_layers.update([int(k) for k in e2_all[m].keys()])
all_layers = sorted(all_layers)

for li in all_layers:
    vals = []
    for m in ['qwen3', 'glm4', 'deepseek7b']:
        if m in e2_all and str(li) in e2_all[m]:
            gd = e2_all[m][str(li)]['g_delta_mean']
            lm = e2_all[m][str(li)]['lambda_max_mean']
            vals.append(f"{gd:>10.3f} {lm:>6.3f}")
        else:
            vals.append(f"{'N/A':>10} {'N/A':>6}")
    print(f"L{li:<5} | {vals[0]} | {vals[1]} | {vals[2]}")

# === KEY TABLE: Propagation Slope ===
print("\n" + "="*70)
print("TABLE 2: Propagation Slope (perturbation growth rate)")
print("="*70)
print(f"{'Inject':<8} | {'Qwen3':>10} | {'GLM4':>10} | {'DS7B':>10}")
print("-"*50)

e3_all = {m: all_data[m]['exp3_propagation_profile'] for m in all_data}
inject_layers = set()
for m in e3_all:
    inject_layers.update([int(k) for k in e3_all[m].keys()])
inject_layers = sorted(inject_layers)

for li in inject_layers:
    vals = []
    for m in ['qwen3', 'glm4', 'deepseek7b']:
        if m in e3_all and str(li) in e3_all[m]:
            slope = e3_all[m][str(li)]['propagation_slope']
            vals.append(f"{slope:>10.4f}")
        else:
            vals.append(f"{'N/A':>10}")
    print(f"L{li:<7} | {vals[0]} | {vals[1]} | {vals[2]}")

# === KEY TABLE: Constraint Type ===
print("\n" + "="*70)
print("TABLE 3: Constraint Type Comparison")
print("="*70)
for ct in ['syntactic', 'semantic', 'factual']:
    print(f"\n  {ct.upper()}:")
    for m in ['qwen3', 'glm4', 'deepseek7b']:
        if m in all_data:
            meta = all_data[m]['exp4_constraint_type_comparison'].get(ct, {}).get('_meta', {})
            slope = meta.get('formation_slope', 0)
            verdict = meta.get('verdict', 'N/A')
            first_d = meta.get('first_layer_delta', 0)
            last_d = meta.get('last_layer_delta', 0)
            print(f"    {m}: slope={slope:.5f}, first={first_d:.4f}, last={last_d:.4f} [{verdict}]")

# === CRITICAL COMPARISON: Shallow vs Deep lambda_max ===
print("\n" + "="*70)
print("CRITICAL: Shallow-layer instability (L1-L3 lambda_max)")
print("="*70)
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if m in e2_all:
        shallow_lam = []
        for li in [1, 2, 3]:
            if str(li) in e2_all[m]:
                shallow_lam.append(e2_all[m][str(li)]['lambda_max_mean'])
        if shallow_lam:
            print(f"  {m}: lambda_max(L1-L3) = {shallow_lam} mean={np.mean(shallow_lam):.3f}")
