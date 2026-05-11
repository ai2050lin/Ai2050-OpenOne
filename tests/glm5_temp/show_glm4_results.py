import json

# GLM4 Exp 3
d3 = json.load(open('tests/glm5_temp/phase125_exp3_glm4_wu_alignment.json'))
print('=== GLM4 W_U Alignment ===')
for l in sorted(d3.keys(), key=int):
    r = d3[l]
    pca10 = r.get('pca_wu_proj_top10_mean', 0)
    pca_bot = r.get('pca_wu_proj_bottom100_mean', 0)
    fish5 = r.get('fisher_top5_wu', 0)
    pca5 = r.get('pca_top5_wu', 0)
    sig = r.get('signal_subspace_wu', 0)
    print(f'L{l}: PCA-top10={pca10:.4f}, PCA-bot={pca_bot:.4f}, Fisher5={fish5:.4f}, PCA5={pca5:.4f}, Signal={sig:.4f}')

# GLM4 Exp 4
d4 = json.load(open('tests/glm5_temp/phase125_exp4_glm4_ablation.json'))
print('\n=== GLM4 Ablation ===')
for l in sorted(d4.keys(), key=int):
    r = d4[l]
    for g in ['high_energy_low_fisher', 'low_energy_high_fisher', 'random']:
        if g in r:
            kl = r[g]['kl_div_mean']
            cos = r[g]['cosine_sim_mean']
            n = r[g]['n_dirs_ablated']
            print(f'L{l} {g:25s}: KL={kl:.3f}, cos={cos:.4f}, n={n}')

# GLM4 Exp 5
d5 = json.load(open('tests/glm5_temp/phase125_exp5_glm4_epsilon_stability.json'))
print('\n=== GLM4 Epsilon Stability ===')
for eps in sorted(d5.keys(), key=float):
    r = d5[eps]
    tau = r.get('kendall_tau_vs_ref', 'N/A')
    print(f'eps={eps}: tau={tau}')
