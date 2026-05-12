"""深入分析Phase 145 Qwen3的Exp A原始数据"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

with open('tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

n_layers = data['model_info']['n_layers']  # 36

# === Exp A: 看原始轨迹 (不归一化) ===
print('='*60)
print('Exp A: 吸引子恢复 - 原始轨迹')
print('='*60)
exp_a = data['exp_a']

# 看几个代表性case的原始值
sample_keys = [k for k in exp_a.keys() if exp_a[k]['eps'] == 2.0 and exp_a[k]['sentence_idx'] == 0]

for key in sample_keys:
    val = exp_a[key]
    il = val['inject_layer']
    eps = val['eps']
    
    print(f'\nL{il}, eps={eps}, sent={val["sentence_idx"]}:')
    
    for dtype in ['random', 'semantic', 'constraint']:
        traj = val[f'recovery_{dtype}']
        init = val[f'initial_dist_{dtype}']
        # 注意: 第一个entry是inject层的clean output (hook修改的是下一层的输入)
        # 所以traj[0]应该是0 (embedding或上一层输出不受影响)
        # traj[1]才是扰动首次出现的位置
        print(f'  {dtype}:')
        print(f'    initial_dist (at inject_l) = {init:.4f}')
        if len(traj) > 2:
            print(f'    traj[0]={traj[0]:.4f} (before hook)')
            print(f'    traj[1]={traj[1]:.4f} (first affected)')
        else:
            print(f'    traj: {[f"{x:.4f}" for x in traj]}')
        
        # 从traj[1]开始归一化
        if len(traj) > 1 and traj[1] > 1e-10:
            normed = [x / traj[1] for x in traj]
            peak_idx = np.argmax(normed)
            print(f'    normed from traj[1]: peak@+{peak_idx}={normed[peak_idx]:.2f}x, final={normed[-1]:.2f}x')
            # 详细轨迹 (每隔5层)
            print(f'    detailed: {[f"L{l+il}:{normed[l]:.2f}" for l in range(0, len(normed), max(1, len(normed)//8))]}')

# === Exp C: 详细轨迹 ===
print('\n' + '='*60)
print('Exp C: 约束修复 - 详细轨迹')
print('='*60)
exp_c = data['exp_c']
for c_type in ['SVA', 'TENSE', 'SCOPE', 'LOGIC', 'SEMANTIC']:
    val = exp_c[c_type]
    traj = val['mean_delta_trajectory']
    # 关键: 从traj[0]开始是embedding层
    # traj的长度 = n_layers + 1 = 37
    print(f'\n{c_type}:')
    # 前5层
    print(f'  early: {[f"L{l}:{traj[l]:.3f}" for l in range(min(5, len(traj)))]}')
    # 中间层
    mid = len(traj)//2
    print(f'  mid:   {[f"L{l}:{traj[l]:.3f}" for l in range(max(0,mid-2), min(len(traj),mid+3))]}')
    # 最后5层
    print(f'  late:  {[f"L{l}:{traj[l]:.3f}" for l in range(max(0,len(traj)-5), len(traj))]}')
    
    # 关键指标
    peak_idx = np.argmax(traj)
    peak_val = traj[peak_idx]
    final_val = traj[-1]
    print(f'  peak@L{peak_idx}={peak_val:.2f}, final@L{len(traj)-1}={final_val:.2f}')
    print(f'  decay from peak to final: {(1 - final_val/peak_val)*100:.1f}%')

# === Exp B: 奇异值谱 ===
print('\n' + '='*60)
print('Exp B: 奇异值谱')
print('='*60)
exp_b = data['exp_b']
for key in sorted(exp_b.keys()):
    val = exp_b[key]
    sv = val['top_singular_values']
    print(f'\n{key} (L{val["layer"]}):')
    print(f'  top SV: {[f"{x:.4f}" for x in sv[:10]]}')
    print(f'  PR={val["participation_ratio"]:.1f}')
    total = val['total_dim']
    print(f'  modes: contract(<0.5)={val["n_contract"]}/{total}, expand(>1.5)={val["n_expand"]}/{total}, neutral={val["n_neutral"]}/{total}')
    print(f'  energy: 50% in top-{val["cum_energy_50"]}, 90% in top-{val["cum_energy_90"]}')
    print(f'  expand-semantic: {val["mean_expand_semantic_align"]:.4f}, contract-semantic: {val["mean_contract_semantic_align"]:.4f}')
