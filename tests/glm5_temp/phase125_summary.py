"""Phase 125 全模型汇总分析"""
import json, numpy as np

models = ['qwen3', 'deepseek7b']
model_labels = {'qwen3': 'Qwen3-4B', 'deepseek7b': 'DS7B'}

print('=' * 80)
print('PHASE 125 全模型汇总: Fisher信息几何 vs PCA能量几何')
print('=' * 80)

for model in models:
    d2 = json.load(open(f'tests/glm5_temp/phase125_exp2_{model}_alignment.json'))
    d3 = json.load(open(f'tests/glm5_temp/phase125_exp3_{model}_wu_alignment.json'))
    d4 = json.load(open(f'tests/glm5_temp/phase125_exp4_{model}_ablation.json'))
    d5 = json.load(open(f'tests/glm5_temp/phase125_exp5_{model}_epsilon_stability.json'))
    d1 = json.load(open(f'tests/glm5_temp/phase125_exp1_{model}_fisher_spectrum.json'))
    
    print(f'\n### {model_labels[model]} ###')
    
    print('\n--- Exp 1: Fisher谱集中度 ---')
    for l in sorted(d1.keys(), key=int):
        r = d1[l]
        pca_top5 = r.get('pca_fisher_top5_mean', 0)
        rand_fish_mean = r.get('rand_fisher_mean', 0)
        fish_conc = r.get('fisher_concentration_top5_vs_all', 0)
        ratio = pca_top5 / rand_fish_mean if rand_fish_mean > 0 else 0
        fish_vals = np.array(r.get('fisher_values', []))
        if len(fish_vals) > 0 and fish_vals.sum() > 0:
            p = fish_vals / fish_vals.sum()
            eff_rank = 1.0 / np.sum(p**2) if np.sum(p**2) > 0 else 0
        else:
            eff_rank = 0
        print(f'  L{l}: eff_rank={eff_rank:.1f}, conc={fish_conc:.3f}, PCA_top5/Rand={ratio:.2f}')
    
    print('\n--- Exp 2: PCA vs Fisher对齐度 ---')
    for l in sorted(d2.keys(), key=int):
        r = d2[l]
        cos_val = r['cos_pca1_fisher1']
        diag_mean = r.get('cos_matrix_diag_mean', 0)
        print(f'  L{l}: cos(PCA1,Fisher1)={cos_val:.6f}, diag_mean={diag_mean:.4f}')
    
    print('\n--- Exp 3: W_U对齐 ---')
    for l in sorted(d3.keys(), key=int):
        r = d3[l]
        pca_top10 = r.get('pca_wu_proj_top10_mean', 0)
        pca_bot = r.get('pca_wu_proj_bottom100_mean', 0)
        fish_top5 = r.get('fisher_top5_wu', 0)
        pca_top5 = r.get('pca_top5_wu', 0)
        sig_wu = r.get('signal_subspace_wu', 0)
        print(f'  L{l}: PCA-top10_WU={pca_top10:.4f}, PCA-bot100_WU={pca_bot:.4f}, '
              f'Fisher-top5_WU={fish_top5:.4f}, PCA-top5_WU={pca_top5:.4f}, '
              f'Signal_WU={sig_wu:.4f}')
    
    print('\n--- Exp 4: 定向消融 ---')
    for l in sorted(d4.keys(), key=int):
        r = d4[l]
        for g in ['high_energy_low_fisher', 'low_energy_high_fisher', 'random']:
            if g in r:
                kl = r[g]['kl_div_mean']
                cos = r[g]['cosine_sim_mean']
                n = r[g]['n_dirs_ablated']
                print(f'  L{l} {g:25s}: KL={kl:.3f}, cos={cos:.4f}, n={n}')
    
    print('\n--- Exp 5: e稳定性 ---')
    for eps in sorted(d5.keys(), key=float):
        r = d5[eps]
        layer_keys = [k for k in r.keys() if k.startswith('L')]
        taus = []
        for lk in sorted(layer_keys):
            tau = r[lk].get('kendall_tau', 0)
            taus.append(tau)
        mean_tau = np.mean(taus) if taus else 0
        print(f'  eps={eps}: mean_tau={mean_tau:.3f}, layers={len(taus)}')

print('\n' + '=' * 80)
print('关键发现')
print('=' * 80)
