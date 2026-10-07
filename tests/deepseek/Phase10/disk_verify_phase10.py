# -*- coding: utf-8 -*-
"""Phase 10 独立磁盘复核（disk verify）——
从冻结的 result_phase10.json 原始数组确定性重算管线量，对照 closeout 写入的派生字段。
不引用 closeout 的中间值；容差 1e-9（float64 同路径）/ 1e-6（重实现）。
"""
import os, json, hashlib, math
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T   = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
S   = os.path.join(ROOT, 'tests', 'deepseek', 'Phase10')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')

FAILS = []
NCHK = [0]
def sec(name): 
    print('\n== %s ==' % name)
def chk(cond, label, detail=''):
    NCHK[0] += 1
    ok = bool(cond)
    if not ok:
        FAILS.append('%s | %s' % (label, detail))
    print('  [%s] %s %s' % ('OK' if ok else 'FAIL', label, detail))
    return ok
def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def sha8_bytes(b):
    return hashlib.sha256(b).hexdigest()[:8]

# ---------- 载入 ----------
R = json.load(open(os.path.join(T, 'result_phase10.json'), encoding='utf-8'))
J = json.load(open(os.path.join(T, 'judgement_phase10.json'), encoding='utf-8'))
E = json.load(open(os.path.join(T, 'execution_phase10.json'), encoding='utf-8'))

FULL = R['full_L6']
PROFILE = list(R['sites']['profile'])      # 18 depth sites (6..34)
R_SITE = R['sites']['readout']             # 'R'

# ---------- A0 文件/哈希 ----------
sec('A0 文件与哈希')
chk(os.path.isfile(os.path.join(T, 'result_phase10.json')), 'result 存在')
chk(os.path.isfile(os.path.join(T, 'n2h1a3_report_qwen3-4b.txt')), 'report 存在')
chk(os.path.isfile(os.path.join(T, 'N2h1a3_design_seal.json')), 'seal 存在')
chk(os.path.isfile(os.path.join(T, 'N2h1a3_design_seal_amend1.json')), 'amend1 存在')
r_sha = sha8(os.path.join(T, 'result_phase10.json'))
chk(r_sha == J['meta']['result_sha8'], 'judgement.meta.result_sha8 == result 实际 sha8',
    '%s vs %s' % (r_sha, J['meta']['result_sha8']))
chk(sha8(os.path.join(T, 'n2h1a3_report_qwen3-4b.txt')) == J['meta']['report_sha8'],
    'judgement.meta.report_sha8 == report 实际 sha8')
chk(R['seal_sha8'] == '70bb8b92' and J['meta']['seal_sha8'] == '70bb8b92', 'seal sha8 = 70bb8b92')
chk(R['exec_sha8'] == J['meta']['exec_sha8'], 'exec sha8 一致', R['exec_sha8'])
_sd = J['meta']['smoke_dir']
chk('smoke_dir' in J['meta'] and os.path.isdir(os.path.join(T, 'smoke')),
    '冒烟目录隔离存在', str(_sd))
chk(os.path.isfile(os.path.join(T, 'smoke', 'result_phase10.json')), 'smoke/result 独立存在')

# ---------- A1 比特级锚点 ----------
sec('A1 比特级锚点（跨 Phase 复现）')
e0 = R['E0']
chk(close(e0['dDonor'], FULL, 0), 'E0(ℓ=6,α=1).dDonor ≡ full_L6', repr(e0['dDonor']))
chk(close(FULL, 10.574739583333335, 0), 'full_L6 ≡ Phase 9 full', repr(FULL))
chk(close(FULL, R['full_ref_phase9'], 0), 'bit_replication: full_L6 == full_ref_phase9')
chk(R['bit_replication'] is True, 'bit_replication 标志 True')
chk(close(e0['dDonor'], R['E1']['6'][5]['dDonor'], 0), 'E0 == E1[6][α=1] 逐位')
chk(close(R['E0b']['dDonor'], 0.3334635416666665, 0), 'E0b ≡ Phase 9 D1a 锚点', repr(R['E0b']['dDonor']))
chk(close(R['dose_coord']['mean_n6'], R['dose_coord']['mean_n6_ref_phase9'], 0),
    'mean‖P_U6(diff6)‖ 与 Phase 9 相同', repr(R['dose_coord']['mean_n6']))
chk(R['dose_coord']['n6_drift'] is False, 'n6_drift False')
chk(close(R['dose_coord']['mean_n6'], 17.06125152401808, 0), 'mean_n6 = 17.06125152401808')
chk(all(v == 0.0 for v in R['floors']['F3_dev'].values()), 'F3 四处 dev 全 0', str(R['floors']['F3_dev']))
chk(R['floors']['F6_ok'] is True and R['floors']['F6b_anchor_drift'] is False, 'F6/F6b 锚点无漂移')

# ---------- A2 J(ℓ) 从原始 dDonor 重算 ----------
sec('A2 J(ℓ) 重算（J = max(段斜率)/其余中位；仅左端点 x≥0.01 的段）')
def slopes_J(x, y):
    xs = [xx for xx in x if xx >= 0.01]
    ymap = dict(zip(x, y))
    ys = [ymap[xx] for xx in xs]
    sl = [(ys[i+1]-ys[i])/(xs[i+1]-xs[i]) for i in range(len(xs)-1)]
    am = max(range(len(sl)), key=lambda i: sl[i])
    others = [sl[i] for i in range(len(sl)) if i != am]
    med = sorted(others)[len(others)//2] if len(others) % 2 == 1 else \
          0.5*(sorted(others)[len(others)//2-1] + sorted(others)[len(others)//2])
    return sl, sl[am]/med if med != 0 else float('inf')

jabs = {}
for site in PROFILE:
    rows = R['E1'][str(site)]
    x = [r['alpha'] for r in rows]
    y = [r['dDonor']/FULL for r in rows]
    sl, Jv = slopes_J(x, y)
    jabs[site] = Jv
    pr = R['profile_abs'][str(site)]
    chk(close(Jv, pr['jump_ratio'], 1e-9), 'J_abs(ℓ=%d) 重算一致' % site, '%.10f vs %.10f' % (Jv, pr['jump_ratio']))
    chk(all(close(a, b, 1e-9) for a, b in zip(sl, pr['slopes'])), 'slopes_abs(ℓ=%d) 一致' % site)
    chk(close(y[-1], pr['y_sat'], 1e-12), 'y_sat(ℓ=%d) == y[-1]' % site)

jrel = {}
for site in PROFILE:
    rows = R['E1b'][str(site)]
    x = [r['alpha_rel'] for r in rows]
    y = [r['dDonor']/FULL for r in rows]
    sl, Jv = slopes_J(x, y)
    jrel[site] = Jv
    pr = R['profile_rel'][str(site)]
    chk(close(Jv, pr['jump_ratio'], 1e-9), 'J_rel(ℓ=%d) 重算一致' % site, '%.10f vs %.10f' % (Jv, pr['jump_ratio']))

# 与记忆中的端点值对齐
chk(close(jabs[6], 5.413883312039905, 1e-9), 'J_abs(L6) = 5.4139', '%.6f' % jabs[6])
chk(close(jabs[34], 1.1471610660486662, 1e-9), 'J_abs(L34) = 1.1472', '%.6f' % jabs[34])
chk(close(jrel[6], 14.264488433102784, 1e-9), 'J_rel(L6) = 14.2645', '%.6f' % jrel[6])
chk(close(jrel[34], 1.027639751552797, 1e-9), 'J_rel(L34) = 1.0276', '%.6f' % jrel[34])
chk(jabs[6] > jabs[34] and jrel[6] > jrel[34], 'J 随深度下降（两坐标）')

# 无断崖：最大相邻跌幅 < 2×
seq = [jabs[s] for s in PROFILE if s >= 7]
drops = [seq[i]/seq[i+1] for i in range(len(seq)-1)]
chk(max(drops) < 2.0, '最大相邻跌幅 < 2× (Q3 未触发)', 'max drop=%.4f @idx=%d' % (max(drops), drops.index(max(drops))))
chk(close(max(drops), 4.831878732200274/2.8450675188124936, 1e-6), '最大跌幅 = J(8)/J(9)',
    '%.6f' % max(drops))

# ---------- A2b 逻辑斯蒂拟合独立重实现 ----------
sec('A2b 逻辑斯蒂拟合重实现（y=A·sigmoid(k(x-x*))，A=max y，网格 k∈[1,60] step1 / x* step0.005）')
def logistic_fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = float(y.max())
    kk = np.arange(1.0, 61.0, 1.0)
    x0 = np.arange(float(x.min()), float(x.max())+1e-9, 0.005)
    P = A/(1.0 + np.exp(-(kk[:, None, None]*(x[None, None, :] - x0[None, :, None]))))
    SSE = ((P - y[None, None, :])**2).sum(axis=2)
    ij = np.unravel_index(int(np.argmin(SSE)), SSE.shape)
    ss_res = float(SSE[ij]); ss_tot = float(((y - y.mean())**2).sum())
    return kk[ij[0]], x0[ij[1]], 1.0 - ss_res/ss_tot, A

nlog = 0
for site in PROFILE:
    rows = R['E1'][str(site)]
    x = [r['alpha'] for r in rows]; y = [r['dDonor']/FULL for r in rows]
    k, xs, r2, A = logistic_fit(x, y)
    pr = R['profile_abs'][str(site)]
    ok_k = close(k, pr['k_log'], 1e-9); ok_x = close(xs, pr['x_star'], 1e-9); ok_r2 = close(r2, pr['R2_log'], 1e-6)
    nlog += 1
    chk(ok_k and ok_x and ok_r2, 'logistic_abs(ℓ=%d) k/x*/R2 重算一致' % site,
        'k=%.0f vs %.0f | x*=%.3f vs %.3f | R2=%.7f vs %.7f' % (k, pr['k_log'], xs, pr['x_star'], r2, pr['R2_log']))
# R 位点扩展网格
pr = R['profile_R_ext']
k, xs, r2, A = logistic_fit(pr['x'], pr['y'])
chk(close(k, pr['k_log'], 1e-9) and close(r2, pr['R2_log'], 1e-6), 'logistic(R,ext) k/R2 重算一致',
    'k=%.0f vs %.0f | R2=%.7f vs %.7f' % (k, pr['k_log'], r2, pr['R2_log']))

# ---------- A3 Spearman ----------
sec('A3 Spearman(J, ℓ) 重算')
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v)
        i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j)/2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return num/den

# 相对剂量下第 6 位点也与 E1 一起（绝对坐标用 alpha=1 时按 y 的 S 形），剖面 18 位点
rho_all = spearman(PROFILE, [jabs[s] for s in PROFILE])
chk(close(rho_all, J['verdict'].get('P2_spearman', R['decisions']['Q_family']['spearman']), 1e-9) or
    close(rho_all, R['decisions']['Q_family']['spearman'], 1e-9),
    'Q 族 Spearman(18 位点) = -0.8720330237358102', '%.12f' % rho_all)
chk(close(rho_all, -0.8720330237358102, 1e-9), 'rho_all ≡ -0.8720330237358102')

unreach = [s for s in PROFILE if R['profile_abs'][str(s)]['cls'] == 'UNREACH']
nonun = [s for s in PROFILE if s not in unreach]
rho_non = spearman(nonun, [jabs[s] for s in nonun])
chk(len(unreach) == 6, 'UNREACH 位点数 = 6', str(unreach))
chk(len(nonun) == 12, '非 UNREACH 位点数 = 12')
chk(close(rho_non, -0.6923076923076923, 1e-9), 'P2 Spearman(12 位点) = -0.6923076923', '%.12f' % rho_non)
chk(close(rho_non, R['decisions']['P2_spearman'], 1e-12), 'P2_spearman 字段一致')

# ---------- A4 Q 族判定重算 ----------
sec('A4 Q 族（方向修正）判定重算')
jmin = jabs[6]
jmax_non = max(jabs[s] for s in nonun)
jmax_all = max(jabs[s] for s in PROFILE)
Q1 = (max(jabs.values())/min(jabs.values()) <= 1.5) and (R['profile_R_ext']['cls'] in ('S_STRONG', 'S_WEAK'))
Q2 = (rho_all <= -0.6) and (jabs[6] >= 1.5*jabs[34])   # J(min L) >= 1.5 * J(max L)
Q3 = False
for i in range(len(PROFILE)-1):
    l0, l1 = PROFILE[i], PROFILE[i+1]
    if jabs[l0] >= 2*jabs[l1] and jabs[l0] >= 3.0:
        sub = [jabs[s] for s in PROFILE[:i+1]]
        if max(sub)/min(sub) <= 1.5:
            Q3 = True
qfam = R['decisions']['Q_family']
chk(Q1 == qfam['Q1'], 'Q1_readout_origin 重算一致', '%s' % Q1)
chk(Q2 == qfam['Q2'], 'Q2_accumulate 重算一致', '%s' % Q2)
chk(Q3 == qfam['Q3'], 'Q3_single_layer 重算一致', '%s' % Q3)
chk(Q2 is True and Q1 is False and Q3 is False, '判决族 Q_abs = Q2_accumulate')
chk(R['decisions']['Q_abs'] == 'Q2_accumulate', 'decisions.Q_abs 字段 = Q2_accumulate')
chk(R['verdict']['Q_rel'] == 'Q2_accumulate' and R['verdict']['Q_agreement'] == 'Q_ROBUST',
    'Q_rel = Q2_accumulate 且 Q_agreement = Q_ROBUST')

# ---------- A5 读数位点否证探针 ----------
sec('A5 读数位点 R（否证探针）')
pr = R['profile_R_ext']
chk(pr['cls'] == 'LINEAR', 'R 位点类标签 = LINEAR')
xr, yr = pr['x'], pr['y']
slr, Jr = slopes_J(xr, yr)
chk(close(Jr, pr['jump_ratio'], 1e-9), 'J(R) 重算一致', '%.6f' % Jr)
chk(Jr < 1.1, 'J(R) ≈ 1.00（无拐点）', '%.6f' % Jr)
# 线性拟合 y = gamma*x 的 R2
mx, my = sum(xr)/len(xr), sum(yr)/len(yr)
sxy = sum((a-mx)*(b-my) for a, b in zip(xr, yr)); sxx = sum((a-mx)**2 for a in xr)
b1 = sxy/sxx; b0 = my - b1*mx
ss_res = sum((b0+b1*a-b)**2 for a, b in zip(xr, yr)); ss_tot = sum((b-my)**2 for b in yr)
R2 = 1-ss_res/ss_tot
chk(R2 >= 0.9999, 'R 位点线性 R² ≥ 0.9999', '%.9f' % R2)
chk(close(R2, pr['R2_lin'], 1e-6), 'R2_lin 重算一致', '%.9f vs %.9f' % (R2, pr['R2_lin']))
_lx = [math.log(a) for a, b in zip(xr, yr) if a >= 0.01 and b > 0]
_ly = [math.log(b) for a, b in zip(xr, yr) if a >= 0.01 and b > 0]
_g = (len(_lx)*sum(a*b for a, b in zip(_lx, _ly)) - sum(_lx)*sum(_ly)) / (len(_lx)*sum(a*a for a in _lx) - sum(_lx)**2)
chk(close(pr['gamma'], _g, 1e-6), 'gamma 重算一致（幂律 log-log 斜率）', '%.6f vs %.6f' % (pr['gamma'], _g))
# dDonor/alpha 常数
rat = [yr[i]/xr[i] for i in range(1, len(xr))]
chk(max(rat)/min(rat) < 1.02, 'dDonor/α 近常数（线性）', '%.6f..%.6f' % (min(rat), max(rat)))
chk(R['floors']['F1_ok'] is True, 'F1_ok True')
chk(close(R['floors']['E4_max'], 0.12265624999999976, 1e-12), 'E4_max 一致', repr(R['floors']['E4_max']))
chk(close(R['floors']['maxabs_all'], 12.443900553385419, 1e-12), 'maxabs_all 一致')
chk(R['floors']['E4_max']/R['floors']['maxabs_all'] < 0.10, 'F1 比值 < 0.10',
    '%.6f' % (R['floors']['E4_max']/R['floors']['maxabs_all']))
mx_all = max(abs(r['dDonor']) for s in R['E1'] for r in R['E1'][s]) if False else None
e4sites = [e['site'] for e in R['E4']]
chk(sorted(e4sites) == [7, 20, 34], 'E4 三个深度位点 = {7,20,34}', str(e4sites))
chk(all(e['n'] == 48 for e in R['E4']), 'E4 n=48')

# ---------- A6 自基臂（基旋转限界） ----------
sec('A6 自基臂 E3（overlap 必报）')
ov = {int(k): R['E3'][k]['overlap'] for k in R['E3']}
chk(ov[7] > ov[12] > ov[20] > ov[34], 'overlap 随深度单调下降', str(ov))
chk(close(ov[7], 0.6892349123954773, 1e-9), 'overlap(L7) = 0.68923')
chk(close(ov[34], 0.02979608252644539, 1e-9), 'overlap(L34) = 0.02980')
for s in [7, 12, 20, 34]:
    a1 = [r for r in R['E3'][str(s)]['rows'] if close(r['alpha'], 1.0, 1e-12)][0]['dDonor']
    chk(abs(a1-FULL)/FULL < 0.03, '自基 α=1 在 L%d ≈ full（±3%%）' % s, '%.4f vs %.4f' % (a1, FULL))
pc = [R['E3'][str(s)]['principal_cos'][0] for s in [7, 12, 20, 34]]
chk(all(pc[i] > pc[i+1] for i in range(3)), '主角 cos 随深度下降', str([round(v, 3) for v in pc]))

# ---------- A7 双剂量坐标换算 ----------
sec('A7 双剂量坐标：x_star_rel = x_star_abs · r_ℓ')
rbar = R['dose_coord']['rbar_ell']
bad = []
for site in PROFILE:
    lhs = R['x_star_rel'][str(site)]
    rhs = R['profile_abs'][str(site)]['x_star'] * rbar[str(site)]
    if not close(lhs, rhs, 1e-9):
        bad.append((site, lhs, rhs))
chk(not bad, '全部 18 个深度位点满足 x*_rel = x*_abs · r_ℓ', str(bad[:3]))
chk(close(rbar['6'], 0.4663894842836559, 1e-9), 'r_ℓ(L6) = 0.46639')
chk(close(rbar['34'], 0.026753605620020566, 1e-9), 'r_ℓ(L34) = 0.02675')
chk(close(R['r_R'], 0.15132150288146348, 1e-9), 'r_R = 0.15132（≈深度位点 1/3）')
chk(0.0267 < rbar['34'] < 0.4665, 'r_ℓ 跨度 0.0268–0.4664（≠ rbar 0.2368）')

# ---------- A8 确认集 ----------
sec('A8 确认集 E5（n=17）')
e5 = R['E5']
chk(e5['same_cls_frac'] == 0.75, 'same_cls_frac = 0.75', str(e5['same_cls_frac']))
chk(e5['same_cls_all'] is False, 'same_cls_all False')
sc = [e5['sites'][str(s)]['same_cls'] for s in [7, 11, 20, 34]]
chk(sum(1 for v in sc if v) == 3, '4 位点中 3 个同类', str(sc))
chk(close(e5['sites']['11']['J'], 1.991710090004738, 1e-12), 'E5 L11 J = 1.9917（压阈值）')
chk(all(len(e5['sites'][str(s)]['rows']) == 7 for s in [7, 11, 20, 34]), 'E5 四站点各 7 点')

# ---------- A9 参考流程（ext）线性 ----------
sec('A9 附加一致性')
chk(R['subspace']['n_classes'] == 6 and R['subspace']['shape'] == [5, 2560], 'U6 形状 [5,2560]、6 类')
chk(R['layers']['L'] == 36 and R['layers']['primary'] == 6 and R['layers']['pre'] == 5, '层位 36 / primary 6 / pre 5')
chk(R['panel']['discovery'] == 24 and R['panel']['confirmation'] == 17, '面板 24 / 17')
chk(R['base']['ok'] is True and R['base']['donor_rank1_frac'] == 0.0, 'base 自洽')
chk(R['amend1'].startswith('N2h1a3-amend1'), 'amend1 标记存在')
chk(J['verdict']['V_readout'] == 'LINEAR' and J['verdict']['V_readout_note'] == 'READOUT_LINEAR_OR_GRADUAL',
    'V_readout = LINEAR')
_va = J['headline'].get('V_ownbasis_agree') or R['decisions'].get('V_ownbasis_agree')
chk(J['verdict']['V_ownbasis'] == 'BASIS_SENSITIVE' and _va == '2/4',
    'V_ownbasis = BASIS_SENSITIVE (2/4)', str(_va))
chk(len(J['amend_disclosure']) == 6, 'amend1 披露 6 条')
chk(len(J['honesty']) >= 6, '诚实边界条目 >= 6', str(len(J['honesty'])))

# ---------- B 继承与执行档 ----------
sec('B 面板继承（execution_phase10）')
chk(len(E.get('secondary_decisions_amend1', {})) >= 1, '执行档含 amend1 决策块')
chk(E.get('inherits_panel_sha256', '') != '' and E.get('inherits_numbers_sha256', '') != '',
    '执行档含继承哈希')

# ---------- C Ledger ----------
sec('C Ledger 补登')
LED = json.load(open(LEDGER, encoding='utf-8'))
meas = LED['measurements']
chk(len(meas) == 293, 'measurements n = 293', str(len(meas)))
tail = meas[-1]
chk(str(tail.get('phase')) == '10' or 'N2h1' in json.dumps(tail, ensure_ascii=False) or
    'depth_profile' in json.dumps(tail, ensure_ascii=False),
    'Ledger 末条 = Phase 10', json.dumps(tail, ensure_ascii=False)[:160])

# ---------- D MEMO / WLOG / MEMORY ----------
sec('D 记录落点')
mb = open(MEMO, 'rb').read()
chk(mb[:3] == b'\xef\xbb\xbf', 'MEMO 有 UTF-8 BOM')
chk(mb.count(b'\r\n') > 0 and (mb.count(b'\n') == mb.count(b'\r\n')), 'MEMO 全 CRLF（无裸 LF）')
chk(b'## Phase 10' in mb, 'MEMO 含 Phase 10 节')
chk(os.path.isfile(os.path.join(S, 'verify_append_phase10.txt')), 'append 复核文件存在（verify_append_phase10.txt）')

wl = open(WLOG, 'rb').read()
chk(b'Phase 10' in wl, '当日 wlog 含 Phase 10')

wm = open(WMEM, 'rb').read().decode('utf-8')
chk('N2h1-α-3（Phase 10' in wm, 'MEMORY 含 N2h1-α-3 条目')
chk('11 臂 + 36 条坑' in wm, 'MEMORY 技能行 = 11 臂 + 36 条坑')
chk('290→291→292→293' in wm, 'MEMORY Ledger 计数行已更新')
chk('装置铁律（Phase 10 新增）' in wm, 'MEMORY 含 Phase 10 铁律 (j)-(n)')

# ---------- 汇总 ----------
print('\n' + '='*60)
print('TOTAL CHECKS: %d' % NCHK[0])
print('TOTAL FAILS : %d' % len(FAILS))
for f in FAILS:
    print('  FAIL> ' + f)
print('=%s' % ('='*59))
out = os.path.join(T, 'disk_verify_phase10.txt')
open(out, 'w', encoding='utf-8').write(
    'Phase 10 独立磁盘复核\nTOTAL CHECKS: %d\nTOTAL FAILS: %d\n' % (NCHK[0], len(FAILS)) +
    ('\n'.join(FAILS) if FAILS else '(no failures)') + '\n')
print('written -> %s' % out)
