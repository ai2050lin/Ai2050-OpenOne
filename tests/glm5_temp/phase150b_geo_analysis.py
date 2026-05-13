"""Phase 150b: 几何基线校正 — 对比训练模型 vs 随机旋转基线"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

with open('tests/glm5_temp/phase150_qwen3_20260513_1043.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

wu = data['W_U_analysis']
d_model = wu['shape'][1]
rank_95 = wu['rank_95']
null_dim = d_model - rank_95

print('='*60)
print('Phase 150b: Geometric Baseline Correction')
print('='*60)
print(f'd_model={d_model}, rank_95={rank_95}, null_dim={null_dim}')

# 几何基线
geo_P_rn = 1.0 - rank_95 / d_model
geo_P_nr = rank_95 / d_model
geo_asym = geo_P_rn - geo_P_nr
geo_P_rr = rank_95 / d_model

print(f'\nGeometric prediction (random rotation baseline):')
print(f'  P(r->r) = {geo_P_rr:.4f}')
print(f'  P(r->n) = {geo_P_rn:.4f}')
print(f'  P(n->r) = {geo_P_nr:.4f}')
print(f'  Asymmetry = P(r->n) - P(n->r) = {geo_asym:+.4f}')

# Monte Carlo验证
print(f'\nMonte Carlo (1000 trials)...')
n_trials = 1000
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

mc_asym = np.mean(row_to_null) - np.mean(null_to_row)
print(f'  MC P(r->n)={np.mean(row_to_null):.4f}, P(n->r)={np.mean(null_to_row):.4f}, Asym={mc_asym:+.4f}')

# 读取实验数据 (key是纯数字字符串如'0', '3', '6'...)
transfer = data['exp1_conditional_transfer']

print(f'\n' + '='*60)
print(f'Trained Model vs Geometric Baseline')
print(f'='*60)
print(f'{"Layer":>6} {"P(r->r)":>8} {"P(r->n)":>8} {"P(n->r)":>8} {"Asym":>10} {"GeoAsym":>10} {"Diff":>10} {"Interp":>15}')

for li_str in sorted(transfer.keys(), key=int):
    li = int(li_str)
    tm = transfer[li_str]
    trained_asym = tm['asymmetry']
    diff = trained_asym - geo_asym
    if diff < -0.05:
        interp = 'Row PRESERVED'
    elif diff > 0.05:
        interp = 'Row SUPPRESSED'
    else:
        interp = '~Neutral'
    print(f'L{li:>4d} {tm["P_rr"]:>8.4f} {tm["P_rn"]:>8.4f} {tm["P_nr"]:>8.4f} {trained_asym:>+10.4f} {geo_asym:>+10.4f} {diff:>+10.4f} {interp:>15}')

# 关键判据
all_layers = [int(k) for k in transfer.keys() if int(k) >= 6]
trained_asym_mid = np.mean([transfer[str(li)]['asymmetry'] for li in all_layers])

print(f'\n=== FINAL JUDGMENT ===')
print(f'Trained model Asymmetry (L6+): {trained_asym_mid:+.4f}')
print(f'Geometric baseline Asymmetry:  {geo_asym:+.4f}')
print(f'Difference (trained - geo):    {trained_asym_mid - geo_asym:+.4f}')

if trained_asym_mid < geo_asym - 0.02:
    print(f'\n✅ Trained Asymmetry < Geometric Baseline!')
    print(f'→ System preserves row-space BETTER than random rotation')
    print(f'→ "Active row-space preservation", NOT "active null-space routing"')
    print(f'→ P(r→n) > P(n→r) is geometric inevitability in high-dim spaces')
elif trained_asym_mid > geo_asym + 0.02:
    print(f'\n❌ Trained Asymmetry > Geometric Baseline')
    print(f'→ Possible active null-space routing')
else:
    print(f'\n↔ Close to geometric baseline')

# Early layer P(r→r) analysis
print(f'\n=== Row-Space Retention Excess ===')
for li_str in sorted(transfer.keys(), key=int):
    li = int(li_str)
    if li < 3:
        continue
    P_rr = transfer[li_str]['P_rr']
    excess = P_rr - geo_P_rr
    print(f'  L{li:>2d}: P(r→r)={P_rr:.4f}, geo={geo_P_rr:.4f}, excess={excess:+.4f} ({excess/geo_P_rr*100:+.1f}%)')
