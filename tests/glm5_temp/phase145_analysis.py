"""分析Phase 145 Qwen3结果"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

with open('tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

n_layers = data['model_info']['n_layers']

# === Exp A ===
print('='*60)
print('Exp A: 吸引子恢复')
print('='*60)
exp_a = data['exp_a']

# 按注入层和eps分组
from collections import defaultdict
groups = defaultdict(lambda: {'random': [], 'semantic': [], 'constraint': []})
for key, val in exp_a.items():
    gkey = f"L{val['inject_layer']}_eps{val['eps']}"
    groups[gkey]['random'].append(val['recovery_random'])
    groups[gkey]['semantic'].append(val['recovery_semantic'])
    groups[gkey]['constraint'].append(val['recovery_constraint'])

for gkey in sorted(groups.keys()):
    print(f'\n{gkey}:')
    for dtype in ['random', 'semantic', 'constraint']:
        trajs = groups[gkey][dtype]
        # 归一化: 每个轨迹除以其初始值
        normed = []
        for t in trajs:
            init = t[0]
            if init > 1e-10:
                normed.append([x/init for x in t])
            else:
                normed.append(t)
        mean_normed = np.mean(normed, axis=0)
        # 找peak和final
        peak_idx = np.argmax(mean_normed)
        peak_val = mean_normed[peak_idx]
        final_val = mean_normed[-1]
        print(f'  {dtype}: peak@L+{peak_idx}={peak_val:.1f}x, final={final_val:.1f}x')
        # 打印轨迹(每5层)
        print(f'    轨迹: {[f"{x:.1f}" for x in mean_normed[::5]]}')

# === Exp B ===
print('\n' + '='*60)
print('Exp B: 稳定/不稳定模态谱')
print('='*60)
exp_b = data['exp_b']
for key in sorted(exp_b.keys()):
    val = exp_b[key]
    sv = val['top_singular_values']
    print(f'\n{key} (L{val["layer"]}):')
    print(f'  Top-5 SV: {[f"{x:.3f}" for x in sv[:5]]}')
    print(f'  Min-5 SV: {[f"{x:.3f}" for x in sv[-5:]]}')
    print(f'  PR={val["participation_ratio"]:.1f}')
    print(f'  contract(<0.5)={val["n_contract"]}, expand(>1.5)={val["n_expand"]}, neutral={val["n_neutral"]}/{val["total_dim"]}')
    print(f'  cum_energy 50%: top-{val["cum_energy_50"]} dims, 90%: top-{val["cum_energy_90"]} dims')
    print(f'  expand-semantic align: {val["mean_expand_semantic_align"]:.3f}')
    print(f'  contract-semantic align: {val["mean_contract_semantic_align"]:.3f}')

# === Exp C ===
print('\n' + '='*60)
print('Exp C: 约束修复动力学')
print('='*60)
exp_c = data['exp_c']
for c_type in ['SVA', 'TENSE', 'SCOPE', 'LOGIC', 'SEMANTIC']:
    val = exp_c[c_type]
    traj = val['mean_delta_trajectory']
    peak_idx = np.argmax(traj)
    peak_val = traj[peak_idx]
    final_val = traj[-1]
    init_val = traj[0]
    print(f'\n{c_type}:')
    print(f'  init={init_val:.3f}, peak@L{peak_idx}={peak_val:.2f}, final={final_val:.2f}')
    amp = f'{peak_val/init_val:.1f}x' if init_val > 1e-10 else 'inf'
    decay = f'{final_val/peak_val:.3f}' if peak_val > 1e-10 else '0'
    print(f'  amplification: {amp}, decay: {decay}')
    # 打印轨迹(关键区间)
    if peak_idx > 0 and peak_idx < len(traj) - 1:
        start = max(0, peak_idx - 3)
        end = min(len(traj), peak_idx + 4)
        print(f'  around peak: {[f"L{l}:{traj[l]:.2f}" for l in range(start, end)]}')
    # 最后5层
    print(f'  last 5 layers: {[f"{x:.2f}" for x in traj[-5:]]}')

# === Exp D ===
print('\n' + '='*60)
print('Exp D: 语义vs随机扰动')
print('='*60)
exp_d = data['exp_d']
for dir_name in ['random', 'semantic_pc1', 'semantic_pc5', 'semantic_pc20', 'orthogonal']:
    if dir_name in exp_d['per_direction']:
        d = exp_d['per_direction'][dir_name]
        ratios = d['remaining_ratios']
        print(f'\n{dir_name}:')
        if ratios:
            print(f'  mean remaining: {np.mean(ratios):.6f} +/- {np.std(ratios):.6f}')
            # 看原始轨迹
            if d.get('trajectory_means'):
                traj = d['trajectory_means']
                init = traj[0] if traj[0] > 0 else 1.0
                normed = [x/init for x in traj]
                print(f'  normed trajectory: {[f"{x:.4f}" for x in normed]}')
        else:
            print('  no data')
