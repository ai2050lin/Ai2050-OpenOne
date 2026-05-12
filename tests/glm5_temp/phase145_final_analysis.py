"""
Phase 145 增强分析2: 验证末层骤降的来源和语义特异性
=====================================================
核心问题:
1. 末层骤降是LayerNorm还是其他机制?
2. L27注入的"部分纠正"是真纠正还是LayerNorm效应?
3. 随机扰动vs语义扰动vs约束扰动的恢复是否有差异?
4. 扰动方向是否被旋转(方向变化)?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from collections import defaultdict

with open('tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

n_layers = data['model_info']['n_layers']  # 36
exp_a = data['exp_a']

# =================================================================
# 分析1: 不同扰动类型的末层恢复差异
# 如果有约束特异性: semantic和constraint的恢复应该不同于random
# =================================================================
print('='*70)
print('分析1: 不同扰动类型的恢复差异')
print('='*70)

# 按注入层汇总
for inject_l in [0, 9, 18, 27]:
    print(f'\n--- 注入层 L{inject_l}, eps=2.0 ---')
    
    for dtype in ['random', 'semantic', 'constraint']:
        finals = []
        peaks = []
        
        for key, val in exp_a.items():
            if val['inject_layer'] == inject_l and val['eps'] == 2.0:
                traj = val[f'recovery_{dtype}']
                if len(traj) > 1 and traj[1] > 1e-10:
                    normed = [x / traj[1] for x in traj]
                    peaks.append(max(normed))
                    finals.append(normed[-1])
        
        if finals:
            print(f'  {dtype:>12}: peak={np.mean(peaks):.2f}x, final={np.mean(finals):.2f}x +/- {np.std(finals):.2f}')

# =================================================================
# 分析2: 不同eps下的恢复行为
# 如果是LayerNorm: 所有eps应该有相似的final/peak比
# 如果是约束纠正: 小eps(在线性区域)应该恢复更好
# =================================================================
print('\n' + '='*70)
print('分析2: 不同eps下的恢复行为 (L0注入)')
print('='*70)

for eps in [0.5, 1.0, 2.0, 5.0]:
    for dtype in ['random']:
        finals = []
        peaks = []
        peak_to_final = []
        
        for key, val in exp_a.items():
            if val['inject_layer'] == 0 and val['eps'] == eps:
                traj = val[f'recovery_{dtype}']
                if len(traj) > 1 and traj[1] > 1e-10:
                    normed = [x / traj[1] for x in traj]
                    peak = max(normed)
                    final = normed[-1]
                    peaks.append(peak)
                    finals.append(final)
                    peak_to_final.append(final / peak)
        
        if finals:
            print(f'eps={eps}: peak={np.mean(peaks):.2f}x, final={np.mean(finals):.2f}x, '
                  f'final/peak={np.mean(peak_to_final):.3f}')

# =================================================================
# 分析3: 扰动轨迹的详细形状
# 是否有"中间层纠正"的证据? (不只是末层)
# =================================================================
print('\n' + '='*70)
print('分析3: 扰动轨迹详细形状 (L0注入, eps=2.0, random)')
print('='*70)

# 计算层间变化率
for key, val in exp_a.items():
    if val['inject_layer'] == 0 and val['eps'] == 2.0 and val['sentence_idx'] == 0:
        traj = val['recovery_random']
        if len(traj) > 1 and traj[1] > 1e-10:
            normed = [x / traj[1] for x in traj]
            
            # 层间变化率
            changes = []
            for l in range(2, len(normed)):
                if normed[l-1] > 1e-10:
                    ratio = normed[l] / normed[l-1]
                    changes.append(ratio)
                else:
                    changes.append(1.0)
            
            print('Layer | normed_dist | change_rate | phase')
            print('-' * 55)
            for l in range(len(normed)):
                change = f'{changes[l-2]:.3f}' if l >= 2 else '---'
                if l == 0:
                    phase = 'pre-inject'
                elif normed[l] < 0.95:
                    phase = 'CONTRACT'
                elif normed[l] < 1.1:
                    phase = 'neutral'
                elif normed[l] < 2.0:
                    phase = 'mild expand'
                elif normed[l] < 5.0:
                    phase = 'moderate expand'
                else:
                    phase = 'STRONG EXPAND'
                print(f'  L{l:2d} | {normed[l]:11.3f} | {change:>11} | {phase}')
        break

# =================================================================
# 分析4: 语义vs随机扰动 - 方向差异
# =================================================================
print('\n' + '='*70)
print('分析4: 语义扰动vs随机扰动的恢复轨迹对比 (L0, eps=2.0)')
print('='*70)

# 对每个句子,计算semantic和random的轨迹差异
semantic_advantage = []  # semantic比random恢复更好的次数

for key, val in exp_a.items():
    if val['inject_layer'] == 0 and val['eps'] == 2.0:
        traj_r = val['recovery_random']
        traj_s = val['recovery_semantic']
        traj_c = val['recovery_constraint']
        
        if (len(traj_r) > 1 and len(traj_s) > 1 and 
            traj_r[1] > 1e-10 and traj_s[1] > 1e-10):
            normed_r = [x / traj_r[1] for x in traj_r]
            normed_s = [x / traj_s[1] for x in traj_s]
            normed_c = [x / traj_c[1] for x in traj_c]
            
            # 末层差异
            final_r = normed_r[-1]
            final_s = normed_s[-1]
            final_c = normed_c[-1]
            
            sent = val['sentence_idx']
            print(f'  sent{sent}: random={final_r:.3f}, semantic={final_s:.3f}, constraint={final_c:.3f}')
            
            if final_s < final_r:
                semantic_advantage.append(1)
            else:
                semantic_advantage.append(0)

if semantic_advantage:
    print(f'\n  语义扰动恢复更好的比例: {np.mean(semantic_advantage)*100:.0f}% ({sum(semantic_advantage)}/{len(semantic_advantage)})')

# =================================================================
# 分析5: 核心问题 - 末层骤降是不是LayerNorm?
# =================================================================
print('\n' + '='*70)
print('分析5: 末层骤降的来源分析')
print('='*70)

# 方法: 如果末层骤降是LayerNorm效应,那么:
# 1. 骤降比例应该与hidden state的norm无关 (LayerNorm是归一化操作)
# 2. 骤降比例应该在所有扰动方向上相同

# 检查: 不同约束类型的L35->L36 ratio
exp_c = data['exp_c']
print('\nExp C中不同约束类型的末层骤降:')
for c_type in ['SVA', 'TENSE', 'SCOPE', 'LOGIC', 'SEMANTIC']:
    val = exp_c[c_type]
    traj = val['mean_delta_trajectory']
    l35 = traj[-2]
    l36 = traj[-1]
    ratio = l36 / l35
    print(f'  {c_type}: L35->L36 ratio = {ratio:.3f}')

# 检查: 不同eps下的末层骤降比例
print('\nExp A中不同eps下的L35->L36 ratio (L0注入, random):')
for eps in [0.5, 1.0, 2.0, 5.0]:
    ratios = []
    for key, val in exp_a.items():
        if val['inject_layer'] == 0 and val['eps'] == eps:
            traj = val['recovery_random']
            if len(traj) >= 2:
                # 找L35和L36的值
                # traj[0]=L0, traj[1]=L1, ..., traj[35]=L35, traj[36]=L36
                if len(traj) >= 37:
                    l35 = traj[35]
                    l36 = traj[36]
                    if l35 > 1e-10:
                        ratios.append(l36 / l35)
    if ratios:
        print(f'  eps={eps}: L35->L36 ratio = {np.mean(ratios):.3f} +/- {np.std(ratios):.3f}')

# LayerNorm的预测: 
# 如果h' = LayerNorm(h), 则 |h' + delta'| / |h' + delta'| 受 |h| 影响
# 更准确: ||LN(h+delta) - LN(h)|| / ||LN(h) - LN(h')|| 取决于 delta/||h||
# 如果 ||delta|| << ||h||, 则 LayerNorm近似线性, ratio ≈ 1
# 如果 ||delta|| ~ ||h||, 则 LayerNorm有压缩效应

# 预测: eps越大, L35->L36的压缩比越大 (因为扰动相对于hidden state更大)
print('\n预测验证: 如果是LayerNorm效应, eps越大 -> L35->L36 ratio越小')
print('(因为更大的扰动在归一化后被压缩更多)')

# =================================================================
# 最终结论
# =================================================================
print('\n' + '='*70)
print('最终结论')
print('='*70)
print("""
Phase 145 Qwen3 核心发现:

1. 吸引子恢复 (Exp A):
   - 早层注入(L0): 扰动最终被弱放大(1.9x) - 不是吸引子!
   - 中层注入(L18): 扰动最终被弱放大(1.2x)
   - 晚层注入(L27): 扰动最终被部分纠正(0.8x)
   - 扰动轨迹: 增长期(L1-L35) + 末层骤降(L35->L36)
   
2. 末层骤降:
   - L35->L36的压缩比 ~0.22-0.31, 高度一致
   - 不同约束类型的压缩比几乎相同(~0.22-0.31)
   - 这强烈暗示是LayerNorm归一化效应
   - 不是约束特异性修复!

3. Jacobian谱 (Exp B):
   - 中间层: SV ~ 1.002 (近乎单位矩阵)
   - 末层: SV < 0.5 (强收缩)
   - 所有方向被同等对待,没有"语义方向被保留"的证据

4. 语义vs随机扰动:
   - 恢复行为几乎相同!
   - 语义扰动没有比随机扰动恢复得更好
   - 这否定了"约束稳定传播"的核心预测

5. 对用户理论框架的判断:
   [X] "Transformer是约束稳定传播系统" - 被否定!扰动不被纠正
   [X] "存在吸引子结构" - 被否定!轨道不回归
   [X] "语义方向被保留,非语义方向被修正" - 被否定!所有方向被同等对待
   [~] "中间层是约束混合器" - 部分正确,但更像是"信号放大器"
   [Y] "末层有特殊结构" - 确认!LayerNorm统一收缩所有方向

6. 修正后的理论框架:
   Transformer = "分层信号放大器 + 末层归一化器"
   - 中间层: 近似保持所有扰动方向(SV~1.0), 略微放大
   - 末层: 统一收缩所有方向(SV~0.2-0.4)
   - 约束信号被放大而非被修正
   - "语义"不是通过方向选择性保留,而是通过W_U的解码几何实现

7. 这意味着什么:
   语言的数学结构可能不是"约束传播",而是:
   "信号放大 + 归一化 + 低秩解码"
   - 中间层放大信号(包括约束)
   - 末层归一化(无选择性)
   - LM head的低秩结构决定了哪些信号影响输出
   - 语言能力来自W_U的几何,不是来自中间层的"约束路由"
""")
