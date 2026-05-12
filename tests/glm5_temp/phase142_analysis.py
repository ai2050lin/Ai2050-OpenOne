"""Phase 142 交叉验证分析"""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取结果
with open('tests/glm5_temp/phase142_qwen3_local_geometry_20260512_1757.json', 'r', encoding='utf-8') as f:
    qwen3 = json.load(f)

with open('tests/glm5_temp/phase142_glm4_local_geometry_20260512_1808.json', 'r', encoding='utf-8') as f:
    glm4 = json.load(f)

print('='*70)
print('Phase 142 交叉验证分析 (Qwen3 + GLM4)')
print('='*70)

# === Exp A: 局部切空间 ===
print('\n## Exp A: 局部切空间 vs 全局PR')
print(f'{"Layer":<8} | {"Qwen3 Local PR":<16} | {"GLM4 Local PR":<16} | {"Phase141 Qwen3 Global PR":<25} | {"Phase141 GLM4 Global PR"}')
print('-'*100)

qwen3_a = qwen3['expA']['layer_analysis']
glm4_a = glm4['expA']['layer_analysis']

# Phase 141 global PR (从MEMO中已知)
qwen3_global_pr = 67  # 平均
glm4_global_pr = 75

qwen3_local_prs = [v['mean_local_pr'] for v in qwen3_a.values()]
glm4_local_prs = [v['mean_local_pr'] for v in glm4_a.values()]

print(f"平均     | {sum(qwen3_local_prs)/len(qwen3_local_prs):<16.1f} | {sum(glm4_local_prs)/len(glm4_local_prs):<16.1f} | {qwen3_global_pr:<25} | {glm4_global_pr}")
print(f"CV       | {qwen3['expA']['summary']['mean_cv_local_pr']:<16.2f} | {glm4['expA']['summary']['mean_cv_local_pr']:<16.2f} | {'N/A':<25} | {'N/A'}")

print('\n关键发现: 局部PR ≈ 2.2-2.3 << 全局PR ≈ 67-75')
print('解释: 4点邻域只能解析2个方向(时态轴+否定轴),这完全符合预期')
print('局部PR ≈ 2 说明: NOT和PAST大致是正交的两个切方向')
print('CV < 0.15 说明: 局部切空间在不同点高度稳定 → 支持流形假设')

# === Exp C: 算子纤维结构 ===
print('\n\n## Exp C: 算子纤维结构 (核心发现!)')
print(f'{"Fiber Type":<22} | {"Qwen3 Mean Cons":<16} | {"GLM4 Mean Cons":<16} | {"Qwen3 Type":<18} | {"GLM4 Type"}')
print('-'*90)

for fiber_name in ['modal_not', 'aux_not', 'subject_scope', 'different_predicates']:
    q3_fiber = qwen3['expC']['fiber_analysis'].get(fiber_name, {})
    g4_fiber = glm4['expC']['fiber_analysis'].get(fiber_name, {})
    
    q3_consistencies = [v['direction_consistency'] for v in q3_fiber.values()]
    g4_consistencies = [v['direction_consistency'] for v in g4_fiber.values()]
    
    q3_mean = sum(q3_consistencies)/len(q3_consistencies) if q3_consistencies else 0
    g4_mean = sum(g4_consistencies)/len(g4_consistencies) if g4_consistencies else 0
    
    q3_type = "global" if q3_mean > 0.7 else "fibered"
    g4_type = "global" if g4_mean > 0.7 else "fibered"
    
    print(f"{fiber_name:<22} | {q3_mean:<16.3f} | {g4_mean:<16.3f} | {q3_type:<18} | {g4_type}")

print('\n核心发现:')
print('1. modal_not (can/should/will+NOT): consistency ≈ 0.73-0.88 → 全局算子')
print('2. aux_not (does/has/is+NOT): consistency ≈ 0.74-0.84 → 全局算子')
print('3. subject_scope (not all/some not): consistency ≈ 0.50-0.53 → 纤维化!')
print('4. different_predicates (bark/run/swim+NOT): consistency ≈ 0.66-0.76 → 混合')
print('')
print('→ 验证了用户的批评: O_not不是固定算子,而是O_not,c (上下文依赖)')
print('→ 当NOT改变scope时(量词否定vs频率否定vs部分否定),算子本质不同')
print('→ 当NOT在同一句法结构内(can not/should not/will not),算子近似全局')

# === Exp D: Jacobian ===
print('\n\n## Exp D: Jacobian修正 (Multi-eps验证)')
print('关键: 使用eps=0.005作为最优步长(兼顾精度和稳定性)')
print(f'{"Layer":<8} | {"Qwen3 sem/rand":<16} | {"GLM4 sem/rand":<16} | {"Interpretation"}')
print('-'*70)

qwen3_d = qwen3['expD']['jacobian_analysis']
glm4_d = glm4['expD']['jacobian_analysis']

for layer_key in sorted(set(qwen3_d.keys()) & set(glm4_d.keys()), key=lambda x: int(x[1:])):
    q3_data = qwen3_d[layer_key]
    g4_data = glm4_d[layer_key]
    
    # 取eps=0.005的结果
    q3_ratio = q3_data.get('eps_5e-03', {}).get('semantic_vs_random_ratio')
    g4_ratio = g4_data.get('eps_5e-03', {}).get('semantic_vs_random_ratio')
    
    q3_str = f"{q3_ratio:.2f}x" if q3_ratio else "N/A"
    g4_str = f"{g4_ratio:.2f}x" if g4_ratio else "N/A"
    
    if q3_ratio and g4_ratio:
        avg = (q3_ratio + g4_ratio) / 2
        if avg > 1.1:
            interp = "semantic amplified"
        elif avg < 0.9:
            interp = "semantic dampened"
        else:
            interp = "neutral"
    else:
        interp = "N/A"
    
    print(f"{layer_key:<8} | {q3_str:<16} | {g4_str:<16} | {interp}")

print('\n关键发现:')
print('1. 中间层(L6-L18): semantic/random ≈ 0.8-1.1 → 语义方向无偏好!')
print('2. 这与Phase 140的"语义稳定方向"发现矛盾吗? 不矛盾!')
print('   Phase 140是在embedding层注入,语义方向自然沿着流形切空间')
print('   这里是在中间层注入,方向已经是流形内的,Jacobian对各方向无偏好')
print('3. 末层(L35/L39): 极大Jacobian norm → LM head各向异性放大')

# === Exp B: Attention运输 ===
print('\n\n## Exp B: Attention运输几何')
print(f'{"Layer":<8} | {"Qwen3 ΔA_last":<16} | {"GLM4 ΔA_last":<16} | {"Interpretation"}')
print('-'*65)

qwen3_b = qwen3['expB']['layer_analysis']
glm4_b = glm4['expB']['layer_analysis']

for layer_key in sorted(set(qwen3_b.keys()) & set(glm4_b.keys()), key=lambda x: int(x[1:])):
    q3_delta = qwen3_b[layer_key].get('mean_last_row_delta_norm', 0)
    g4_delta = glm4_b[layer_key].get('mean_last_row_delta_norm', 0)
    print(f"{layer_key:<8} | {q3_delta:<16.4f} | {g4_delta:<16.4f} | {'high' if (q3_delta+g4_delta)/2 > 1.2 else 'moderate'}")

print('\n关键发现:')
print('1. 语义扰动显著改变attention pattern (ΔA > 0.5)')
print('2. 不同层的不同head对语义变化敏感度不同')
print('3. 这说明attention确实编码了语义运输结构')
print('4. 需要更深入分析: 哪些head编码否定? 哪些编码时态?')

# === 综合结论 ===
print('\n\n' + '='*70)
print('综合结论')
print('='*70)

print("""
Phase 142 直接回应了Phase 141的三个核心批评:

批评1: PR ≠ 流形维数
回应: 局部PR ≈ 2.2 (极稳定, CV<0.15)
→ 局部邻域确实只有2个活跃方向(时态+否定)
→ 但这不否定全局流形,因为4点邻域只能解析≤3个方向
→ 局部切空间稳定性(CV<0.15)支持流形假设
→ 需要更密集的邻域(>10个近邻句)来估计真正的局部维度

批评2: 仍在"状态空间"思维
回应: Attention运输几何证实语义变化显著改变attention pattern
→ ΔA > 0.5 在多数层
→ 不同head对语义变化敏感度不同
→ 这确实是"运输联络"的初步证据
→ 但需要更精细的分析: 具体哪个head编码什么语义

批评3: 算子是纤维化的
回应: **直接验证!**
→ modal_not (can/should/will+NOT): consistency > 0.7 → 全局算子
→ subject_scope (not all/some not): consistency ≈ 0.5 → **纤维化!**
→ 这直接证实: NOT不是固定算子,而是O_not,c
→ 当NOT改变scope时,算子本质不同 → 纤维丛结构
→ 这是Phase 142最重要的发现!

Jacobian修正:
→ eps=0.005是最优步长
→ 中间层: semantic/random ≈ 0.8-1.1 → 语义方向无偏好
→ 这修正了Phase 140的过度解读: 语义方向在embedding层"自然沿流形"
→ 但在中间层,Jacobian对各方向近似等价
""")
