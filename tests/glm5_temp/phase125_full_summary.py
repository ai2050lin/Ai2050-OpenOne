"""Phase 125 三模型完整汇总"""
import json, numpy as np

models = ['qwen3', 'deepseek7b', 'glm4']
model_labels = {'qwen3': 'Qwen3-4B', 'deepseek7b': 'DS7B', 'glm4': 'GLM4-9B'}

print('=' * 80)
print('PHASE 125 完整汇总: Fisher信息几何 vs PCA能量几何 (3模型)')
print('=' * 80)

# ============ Exp 2: PCA vs Fisher对齐度 ============
print('\n' + '='*60)
print('Exp 2: PCA主方向 vs Fisher主方向对齐度 (cos值)')
print('='*60)
for model in models:
    d = json.load(open(f'tests/glm5_temp/phase125_exp2_{model}_alignment.json'))
    print(f'\n  {model_labels[model]}:')
    for l in sorted(d.keys(), key=lambda x: int(x)):
        r = d[l]
        cos_val = r['cos_pca1_fisher1']
        diag = r.get('cos_matrix_diag_mean', 0)
        print(f'    L{l}: cos(PCA1,Fisher1)={cos_val:.6f}, diag_mean={diag:.4f}')

# ============ Exp 3: W_U对齐 ============
print('\n' + '='*60)
print('Exp 3: W_U对齐度')
print('='*60)
for model in models:
    d = json.load(open(f'tests/glm5_temp/phase125_exp3_{model}_wu_alignment.json'))
    if not d:
        print(f'\n  {model_labels[model]}: NO DATA')
        continue
    print(f'\n  {model_labels[model]}:')
    for l in sorted(d.keys(), key=lambda x: int(x)):
        r = d[l]
        pca5 = r.get('pca_top5_wu', 0)
        fish5 = r.get('fisher_top5_wu', 0)
        sig = r.get('signal_subspace_wu', 0)
        pca10 = r.get('pca_wu_proj_top10_mean', 0)
        print(f'    L{l}: PCA5_WU={pca5:.4f}, Fisher5_WU={fish5:.4f}, Signal_WU={sig:.4f}, PCA10_WU={pca10:.4f}')

# ============ Exp 4: 消融 ============
print('\n' + '='*60)
print('Exp 4: 定向消融 (KL散度)')
print('='*60)
for model in models:
    d = json.load(open(f'tests/glm5_temp/phase125_exp4_{model}_ablation.json'))
    if not d:
        print(f'\n  {model_labels[model]}: NO DATA')
        continue
    print(f'\n  {model_labels[model]}:')
    for l in sorted(d.keys(), key=lambda x: int(x)):
        r = d[l]
        kl_helf = r.get('high_energy_low_fisher', {}).get('kl_div_mean', 'N/A')
        kl_lehf = r.get('low_energy_high_fisher', {}).get('kl_div_mean', 'N/A')
        kl_rand = r.get('random', {}).get('kl_div_mean', 'N/A')
        n_helf = r.get('high_energy_low_fisher', {}).get('n_dirs_ablated', '?')
        n_lehf = r.get('low_energy_high_fisher', {}).get('n_dirs_ablated', '?')
        n_rand = r.get('random', {}).get('n_dirs_ablated', '?')
        if isinstance(kl_helf, float):
            print(f'    L{l}: HighE-LowF(n={n_helf}) KL={kl_helf:.3f}, LowE-HighF(n={n_lehf}) KL={kl_lehf:.3f}, Random(n={n_rand}) KL={kl_rand:.3f}')

# ============ Exp 5: ε稳定性 ============
print('\n' + '='*60)
print('Exp 5: Fisher ε稳定性 (Kendall tau)')
print('='*60)
for model in models:
    d = json.load(open(f'tests/glm5_temp/phase125_exp5_{model}_epsilon_stability.json'))
    print(f'\n  {model_labels[model]}:')
    for eps in sorted(d.keys(), key=float):
        r = d[eps]
        tau = r.get('kendall_tau_vs_ref', 'N/A')
        if isinstance(tau, float):
            print(f'    eps={eps}: tau={tau:.3f}')
