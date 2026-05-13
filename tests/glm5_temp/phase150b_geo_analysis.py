"""Phase 150b: 几何基线校正 — 对比训练模型 vs 随机旋转基线"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

# 读取Phase 150的结果
with open('tests/glm5_temp/phase150_qwen3_20260513_1043.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

wu = data['W_U_analysis']
d_model = wu['shape'][1]
rank_95 = wu['rank_95']
null_dim = d_model - rank_95

print('='*60)
print('几何基线计算 (无需模型加载)')
print('='*60)
print(f'd_model={d_model}, rank_95={rank_95}, null_dim={null_dim}')

# 数学推导: 在随机旋转下
geo_P_rn = 1.0 - rank_95 / d_model
geo_P_nr = rank_95 / d_model
geo_asym = geo_P_rn - geo_P_nr

print(f'几何预测:')
print(f'  P(r->n) = {geo_P_rn:.4f}')
print(f'  P(n->r) = {geo_P_nr:.4f}')
print(f'  Asymmetry = {geo_asym:+.4f}')

# Monte Carlo验证
print(f'\nMonte Carlo验证 (5000次随机旋转)...')
n_trials = 5000
row_to_null = []
null_to_row = []

for trial in range(n_trials):
    v_row_proj = np.zeros(d_model)
    v_row_proj[:rank_95] = np.random.randn(rank_95)
    v_row_proj = v_row_proj / np.linalg.norm(v_row_proj)
    
    Q, _ = np.linalg.qr(np.random.randn(d_model, d_model))
    v_rotated = Q @ v_row_proj
    row_e = np.sum(v_rotated[:rank_95]**2) / np.sum(v_rotated**2)
    row_to_null.append(1 - row_e)
    
    v_null_proj = np.zeros(d_model)
    v_null_proj[rank_95:] = np.random.randn(null_dim)
    v_null_proj = v_null_proj / np.linalg.norm(v_null_proj)
    
    v_rotated_null = Q @ v_null_proj
    row_e_null = np.sum(v_rotated_null[:rank_95]**2) / np.sum(v_rotated_null**2)
    null_to_row.append(row_e_null)

mc_P_rn = np.mean(row_to_null)
mc_P_nr = np.mean(null_to_row)
mc_asym = mc_P_rn - mc_P_nr

print(f'  Monte Carlo P(r->n) = {mc_P_rn:.4f} (理论: {geo_P_rn:.4f})')
print(f'  Monte Carlo P(n->r) = {mc_P_nr:.4f} (理论: {geo_P_nr:.4f})')
print(f'  Monte Carlo Asymmetry = {mc_asym:+.4f} (理论: {geo_asym:+.4f})')

# 对比Phase 150实验数据
transfer = data['exp1_conditional_transfer']

print(f'\n' + '='*60)
print(f'对比: 训练模型 vs 几何基线')
print(f'='*60)
print(f'{"Layer":>6} {"P(r->r)":>8} {"P(r->n)":>8} {"P(n->r)":>8} {"Asym":>10} {"Geo_Asym":>10} {"Diff":>10} {"Interp":>15}')

for li_str, tm in sorted(transfer.items(), key=lambda x: int(x[0].replace('L',''))):
    li = int(li_str.replace('L',''))
    trained_asym = tm['asymmetry']
    diff = trained_asym - geo_asym
    if diff < -0.05:
        interp = 'Row PRESERVED'
    elif diff > 0.05:
        interp = 'Row SUPPRESSED'
    else:
        interp = 'Neutral'
    print(f'L{li:>4d} {tm["P_rr"]:>8.4f} {tm["P_rn"]:>8.4f} {tm["P_nr"]:>8.4f} {trained_asym:>+10.4f} {geo_asym:>+10.4f} {diff:>+10.4f} {interp:>15}')

# 关键判据
mid_layers = [li for li in [9,12,15,18,21,24,27] if f'L{li}' in transfer]
trained_asym_mid = np.mean([transfer[f'L{li}']['asymmetry'] for li in mid_layers])

print(f'\n=== FINAL JUDGMENT ===')
print(f'Trained model mid-layer Asymmetry: {trained_asym_mid:+.4f}')
print(f'Geometric baseline Asymmetry:      {geo_asym:+.4f}')
print(f'Difference (trained - geo):        {trained_asym_mid - geo_asym:+.4f}')

if trained_asym_mid < geo_asym - 0.02:
    print(f'\n✅ 实际Asymmetry 显著低于几何基线!')
    print(f'→ 系统比随机旋转更好地保持row-space!')
    print(f'→ 这是"主动保护row-space"的证据, 而非"主动推到null-space"!')
    print(f'→ Phase 148/150的P(r→n)>P(n→r)是高维几何的必然结果, 不是主动路由!')
elif trained_asym_mid > geo_asym + 0.02:
    print(f'\n❌ 实际Asymmetry 显著高于几何基线')
    print(f'→ 可能存在主动null-space路由')
else:
    print(f'\n↔ 差异不显著, 纯被动mixing')

# 早期层分析
print(f'\n=== Early Layer Analysis ===')
for li in [3, 6]:
    key = f'L{li}'
    if key in transfer:
        P_rr = transfer[key]['P_rr']
        geo_P_rr = rank_95 / d_model
        print(f'L{li}: P(r→r) = {P_rr:.4f} vs geo = {geo_P_rr:.4f} → {P_rr/geo_P_rr:.1f}x above baseline')

# Row-space保持分析
print(f'\n=== Row-Space Retention Analysis ===')
print(f'关键: P(r→r)在各层的演化 vs 几何基线')
for li_str, tm in sorted(transfer.items(), key=lambda x: int(x[0].replace('L',''))):
    li = int(li_str.replace('L',''))
    if li < 3:
        continue
    P_rr = tm['P_rr']
    geo_P_rr = rank_95 / d_model
    excess = P_rr - geo_P_rr
    print(f'  L{li:>2d}: P(r→r)={P_rr:.4f}, geo={geo_P_rr:.4f}, excess={excess:+.4f}')

# 总结
print(f'\n=== 综合总结 ===')
print(f'1. W_U有效rank(95%)= {rank_95}, 远高于Phase 148假设的200')
print(f'2. 几何基线Asymmetry = {geo_asym:+.4f}')
print(f'3. 实际mid-layer Asymmetry = {trained_asym_mid:+.4f}')
print(f'4. 差异 = {trained_asym_mid - geo_asym:+.4f}')
if trained_asym_mid < geo_asym:
    print(f'5. 结论: 训练模型的row-space保持 > 随机旋转基线')
    print(f'   → 训练使系统主动保持输出相关方向(row-space)')
    print(f'   → P(r→n)>P(n→r)是几何必然性, 不是主动路由')
    print(f'   → 真正的"主动机制"是保持row-space, 不是推到null-space!')
