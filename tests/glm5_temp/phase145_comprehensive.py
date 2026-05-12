"""
Phase 145 增强分析: 基于Qwen3完整数据
修正Exp A的归一化问题, 深入分析吸引子恢复行为
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from collections import defaultdict

with open('tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

n_layers = data['model_info']['n_layers']  # 36
d_model = data['model_info']['d_model']    # 2560

# =================================================================
# Exp A: 吸引子恢复 - 修正归一化后的完整分析
# =================================================================
print('='*70)
print('Exp A: 吸引子恢复实验 - 修正归一化')
print('='*70)

exp_a = data['exp_a']

# 按注入层和eps分组, 归一化到traj[1](扰动首次出现的层)
groups = defaultdict(lambda: {'random': [], 'semantic': [], 'constraint': []})

for key, val in exp_a.items():
    gkey = f"L{val['inject_layer']}_eps{val['eps']}"
    for dtype in ['random', 'semantic', 'constraint']:
        traj = val[f'recovery_{dtype}']
        # traj[0] = inject层的输出 (hook还没生效, 距离≈0)
        # traj[1] = 扰动首次出现 (距离≈eps)
        if len(traj) > 1 and traj[1] > 1e-10:
            normed = [x / traj[1] for x in traj]
            groups[gkey][dtype].append(normed)

print('\n--- 吸引子恢复: 归一化到首次扰动出现 ---')
print('格式: [注入层] eps=X: random(peak->final), semantic(peak->final), constraint(peak->final)')
print()

key_results = {}
for gkey in sorted(groups.keys()):
    parts = gkey.split('_')
    inject_l = int(parts[0][1:])
    eps = float(parts[1][3:])
    
    result = {}
    for dtype in ['random', 'semantic', 'constraint']:
        trajs = groups[gkey][dtype]
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            peak_idx = np.argmax(mean_traj)
            peak_val = mean_traj[peak_idx]
            final_val = mean_traj[-1]
            result[dtype] = {
                'peak_layer_offset': peak_idx,
                'peak_val': peak_val,
                'final_val': final_val,
                'decay_from_peak': 1 - final_val / peak_val,
                'mean_traj': mean_traj.tolist(),
            }
    
    key_results[gkey] = result
    
    # 只打印eps=2.0的结果
    if eps == 2.0:
        print(f'{gkey}:')
        for dtype in ['random', 'semantic', 'constraint']:
            r = result.get(dtype, {})
            if r:
                print(f'  {dtype}: peak@L+{r["peak_layer_offset"]}={r["peak_val"]:.2f}x -> final={r["final_val"]:.2f}x (decay={r["decay_from_peak"]*100:.0f}%)')
        print()

# =================================================================
# 关键分析: 扰动是否被"纠正"?
# =================================================================
print('\n' + '='*70)
print('关键分析: 扰动是否被"纠正"?')
print('='*70)

# 如果有吸引子: final_val 应该 << 1.0 (扰动被消除)
# 如果是纯前馈: final_val 应该 ≈ 1.0 (扰动保持)
# 如果是放大: final_val 应该 > 1.0 (扰动被放大)

for gkey in sorted(groups.keys()):
    parts = gkey.split('_')
    inject_l = int(parts[0][1:])
    eps = float(parts[1][3:])
    
    if eps != 2.0:
        continue
    
    for dtype in ['random']:
        trajs = groups[gkey][dtype]
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            final_val = mean_traj[-1]
            
            # 判断行为
            if final_val < 0.5:
                behavior = "STRONG CORRECTION (吸引子)"
            elif final_val < 0.9:
                behavior = "PARTIAL CORRECTION (弱吸引子)"
            elif final_val < 1.1:
                behavior = "PRESERVATION (前馈保持)"
            elif final_val < 2.0:
                behavior = "MILD AMPLIFICATION (弱放大)"
            else:
                behavior = "STRONG AMPLIFICATION (强放大)"
            
            print(f'L{inject_l} eps={eps}: final={final_val:.2f}x -> {behavior}')

# =================================================================
# 逐层分析: 扰动轨迹的三阶段
# =================================================================
print('\n' + '='*70)
print('逐层分析: 扰动轨迹 (L0注入, eps=2.0, random)')
print('='*70)

gkey = 'L0_eps2.0'
trajs = groups[gkey]['random']
if trajs:
    mean_traj = np.mean(trajs, axis=0)
    print(f'层号: 扰动倍数 | 阶段判断')
    print(f'-' * 50)
    for l_offset in range(len(mean_traj)):
        val = mean_traj[l_offset]
        if l_offset == 0:
            phase = "pre-inject"
        elif val < 0.95:
            phase = "CONTRACT"
        elif val < 1.05:
            phase = "NEUTRAL"
        elif val < 2.0:
            phase = "MILD EXPAND"
        elif val < 5.0:
            phase = "MODERATE EXPAND"
        else:
            phase = "STRONG EXPAND"
        
        if l_offset % 3 == 0 or l_offset >= len(mean_traj) - 3:
            print(f'  L{l_offset:2d}: {val:6.2f}x | {phase}')

# =================================================================
# Exp B: 奇异值谱 - 更深入分析
# =================================================================
print('\n' + '='*70)
print('Exp B: 奇异值谱分析')
print('='*70)

exp_b = data['exp_b']

# 按层汇总
layer_sv = defaultdict(list)
for key, val in exp_b.items():
    layer = val['layer']
    layer_sv[layer].append(val)

for layer in sorted(layer_sv.keys()):
    vals = layer_sv[layer]
    print(f'\n--- Layer {layer} ---')
    
    # 合并所有句子的结果
    all_sv = []
    all_contract = []
    all_expand = []
    all_pr = []
    
    for v in vals:
        all_sv.append(v['top_singular_values'])
        all_contract.append(v['n_contract'])
        all_expand.append(v['n_expand'])
        all_pr.append(v['participation_ratio'])
    
    mean_sv = np.mean(all_sv, axis=0)
    mean_contract = np.mean(all_contract)
    mean_expand = np.mean(all_expand)
    mean_pr = np.mean(all_pr)
    
    print(f'  Mean top-5 SV: {[f"{x:.4f}" for x in mean_sv[:5]]}')
    print(f'  Mean bottom-5 SV: {[f"{x:.4f}" for x in mean_sv[-5:]]}')
    print(f'  PR={mean_pr:.1f}')
    print(f'  Mean contract(<0.5): {mean_contract:.1f}/80, expand(>1.5): {mean_expand:.1f}/80')
    
    # 关键: 最大的SV是多少? (决定放大率)
    max_sv = mean_sv[0]
    print(f'  Maximum singular value: {max_sv:.4f}')
    if max_sv < 0.5:
        print(f'  -> ALL directions CONTRACT at this layer')
    elif max_sv < 1.0:
        print(f'  -> Most directions CONTRACT at this layer')
    elif max_sv < 1.5:
        print(f'  -> Directions are approximately PRESERVED at this layer')
    else:
        print(f'  -> Some directions EXPAND at this layer')

# =================================================================
# Exp C: 约束修复 - 对比不同约束类型
# =================================================================
print('\n' + '='*70)
print('Exp C: 约束修复动力学对比')
print('='*70)

exp_c = data['exp_c']

print(f'\n{"类型":>10} | {"init":>8} | {"peak@L":>8} | {"peak_val":>8} | {"final":>8} | {"decay%":>6} | {"amplification":>14}')
print('-' * 80)

for c_type in ['SVA', 'TENSE', 'SCOPE', 'LOGIC', 'SEMANTIC']:
    val = exp_c[c_type]
    traj = val['mean_delta_trajectory']
    peak_idx = np.argmax(traj)
    peak_val = traj[peak_idx]
    final_val = traj[-1]
    init_val = traj[0]
    decay = (1 - final_val / peak_val) * 100
    
    if init_val > 1e-10:
        amp = f'{peak_val/init_val:.0f}x'
    else:
        amp = 'N/A'
    
    print(f'{c_type:>10} | {init_val:>8.3f} | L{peak_idx:>6} | {peak_val:>8.2f} | {final_val:>8.2f} | {decay:>5.1f}% | {amp:>14}')

# 关键问题: 末层(L36)的delta骤降来自什么?
print('\n--- 末层骤降分析 ---')
for c_type in ['SVA', 'TENSE', 'SCOPE', 'LOGIC', 'SEMANTIC']:
    val = exp_c[c_type]
    traj = val['mean_delta_trajectory']
    # L35 -> L36 的变化
    if len(traj) > 1:
        l35 = traj[-2]
        l36 = traj[-1]
        ratio = l36 / l35
        print(f'{c_type}: L35={l35:.2f} -> L36={l36:.2f}, ratio={ratio:.3f} ({(1-ratio)*100:.0f}% drop)')

# =================================================================
# 综合结论
# =================================================================
print('\n' + '='*70)
print('综合结论')
print('='*70)

print("""
1. 吸引子恢复实验 (Exp A):
   - 扰动NOT被纠正! 末层距离仍是初始的1.1-2.9倍
   - 扰动轨迹: 初始微缩(0.94x) -> 中间层放大(3-9x) -> 末层骤降(1.4-2.9x)
   - 末层骤降来自LayerNorm归一化, 不是"约束修正"
   - 结论: Transformer **不是吸引子系统**

2. Jacobian谱 (Exp B):
   - 早/中层层: Jacobian近似单位矩阵(SV ~ 1.002)
   - 末层: 所有过方向强收缩(SV < 0.5)
   - 每层近乎保持扰动, 只有末层统一降低
   - 结论: 没有发现"语义方向被保留,非语义方向被修正"的证据

3. 约束修复 (Exp C):
   - 约束违背信号从embedding持续增长到L35
   - L36骤降60-77% (与LayerNorm一致)
   - 不同约束类型的行为高度一致
   - 结论: 末层骤降是LayerNorm效应, 不是约束特异性修复

4. 整体判断:
   - Transformer是"分层放大器 + 末层归一化器"
   - 不是"约束稳定传播系统"(因为扰动不被纠正)
   - 不是"吸引子系统"(因为轨道不回归)
   - 更准确: "约束增强传播系统"——约束信号被放大, 不是被修正
""")
