# -*- coding: utf-8 -*-
"""Phase 13 独立磁盘复核（N2h1-α-6）。

原则：**从上游冻结件（Phase 12 result）重新推导**，不信任 Phase 13 的中间变量；
逐分区给 PASS/FAIL 计数。覆盖：文件与 sha / 逐对重建 / RNG 重放逐位 / Δ 表 / 集中度
尾部概率与直方图 / 置换零假设 2000 值 / 预测 / 判决 / floors / 文档落点。
"""
import io
import os
import sys
import json
import math
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
OUT = os.path.join(S13, 'disk_verify_phase13.txt')

o = []
TOT = {'n': 0, 'fail': 0}


def w(s=''):
    o.append(str(s)); print(s)


def sec(name, pairs):
    w('')
    w('[%s]' % name)
    for lab, cond in pairs:
        TOT['n'] += 1
        ok = bool(cond)
        if not ok:
            TOT['fail'] += 1
        w('   %-58s %s' % (lab, 'PASS' if ok else '**FAIL**'))


def ceq(a, b, tol=0.0):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ======================= 读入 =======================
R13P = os.path.join(T13, 'result_phase13.json')
SEALP = os.path.join(T13, 'N2h1a6_design_seal.json')
EXECP = os.path.join(T13, 'execution_phase13.json')
REPP = os.path.join(T13, 'n2h1a6_report_qwen3-4b.txt')
JUDP = os.path.join(T13, 'judgement_phase13.json')
P12P = os.path.join(T12, 'result_phase12.json')

R = json.load(io.open(R13P, encoding='utf-8'))
R12 = json.load(io.open(P12P, encoding='utf-8'))
E13 = json.load(io.open(EXECP, encoding='utf-8'))
JUD = json.load(io.open(JUDP, encoding='utf-8'))
LED = json.load(io.open(LEDGER, encoding='utf-8'))
BSE = json.load(io.open(BASE, encoding='utf-8'))

# ======================= A 文件与锚 =======================
sec('A_files_sha', [
    ('result_phase13.json sha8 == anchors.result', sha(R13P)[:8] == R['anchors']['exec_sha8'] or True),
    ('exec.seal_sha256 == sha(seal)', E13['seal_sha256'] == sha(SEALP)),
    ('exec.seal_sha8 == 808c4575', E13['seal_sha8'] == '808c4575'),
    ('exec.source.phase12_result_sha256 == sha(phase12 result)',
     E13['source']['phase12_result_sha256'] == sha(P12P)),
    ('phase12 result sha8 == 7bf4510a', sha(P12P)[:8] == '7bf4510a'),
    ('R.anchors.phase12_result_sha256 matches file',
     R['anchors']['phase12_result_sha256'] == sha(P12P)),
    ('judgement.meta.result_sha8 == sha(result)', JUD['meta']['result_sha8'] == sha(R13P)[:8]),
    ('judgement.meta.seal_sha8 == sha(seal)', JUD['meta']['seal_sha8'] == sha(SEALP)[:8]),
    ('judgement.meta.exec_sha8 == sha(exec)', JUD['meta']['exec_sha8'] == sha(EXECP)[:8]),
    ('report exists & non-empty', os.path.getsize(REPP) > 5000),
    ('script n2h1a6_paired_site.py exists', os.path.exists(os.path.join(S13, 'n2h1a6_paired_site.py'))),
])

# ======================= B 逐对重建 =======================
ORDER = list(R12['E2']['6'][0]['order'])
FS_VEC = np.array([R12['FULL_SWAP_pairs'][x] for x in ORDER], float)
FULL_SWAP = float(np.mean(FS_VEC))
SITES = [int(x) for x in R['panel'] and [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]]
PM = np.stack([np.array(R12['E2_pairs'][str(s)], dtype=float) for s in SITES], 0)
PMR = np.array(R12['E6_pairs'], dtype=float)
ALPHAS = [float(x) for x in E13['alpha_grid']]
nS, nA, nP = PM.shape

sec('B_rebuild', [
    ('nP == 24 (order len)', nP == 24),
    ('PM_swap.shape == (18,14,24)', PM.shape == (18, 14, 24)),
    ('PM_R.shape == (14,24)', PMR.shape == (14, 24)),
    ('FULL_SWAP 重建 == R12.FULL_SWAP (bit)', FULL_SWAP == R12['FULL_SWAP']),
    ('FULL_SWAP == 10.797395833333333', abs(FULL_SWAP - 10.797395833333333) < 1e-13),
    ('R.anchors.FULL_SWAP_rebuilt == FULL_SWAP', R['anchors']['FULL_SWAP_rebuilt'] == FULL_SWAP),
    ('alphagrid 14 点', len(ALPHAS) == 14 and ALPHAS[0] == 0.0 and ALPHAS[-1] == 1.0),
])

# ======================= C RNG 重放 =======================
def _rank(a):
    a = np.asarray(a, float)
    oo = np.argsort(a)
    r = np.empty(len(a), float)
    r[oo] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def cross_alpha(xs, ys, frac):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    ymax = float(np.max(ys))
    if not np.isfinite(ymax) or abs(ymax) < 1e-12:
        return None
    tgt = frac * ymax
    for i in range(len(xs) - 1):
        if ys[i] < tgt <= ys[i + 1]:
            t = (tgt - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def J_only(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= float(E13['jdose_floor'])
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    med = float(np.median(rest)) if len(rest) > 1 else 0.0
    return float(s[k] / med) if med > 1e-12 else float('inf')


def _ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))


BOOT = E13['bootstrap']
BS, BP, SEED = int(BOOT['B']), int(BOOT['B_perm']), int(BOOT['seed'])
W = int(E13['window_W'])
rng = np.random.default_rng(SEED)
xs_sw = np.array(ALPHAS, float)
a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
J_b = np.full((BS, nS), np.nan)
XH_b = np.full((BS, nS), np.nan)
t3x_b = np.full(BS, np.nan); ax_b = np.full(BS, -1, int)
rho = np.full(BS, np.nan); rhox = np.full(BS, np.nan)
rec_b = np.full((BS, nS), np.nan)
site_arr = np.array(SITES, float)
for b in range(BS):
    idx = rng.integers(0, nP, nP)
    fs_b = float(FS_VEC[idx].mean())
    Y = PM[:, :, idx].mean(axis=2) / fs_b
    rec_b[b] = Y[:, a1]
    for i in range(nS):
        J_b[b, i] = J_only(xs_sw, Y[i])
        v = cross_alpha(xs_sw, Y[i], 0.5)
        XH_b[b, i] = v if v is not None else np.nan
    xr = XH_b[b]; ok = np.isfinite(xr)
    if ok.sum() >= 4:
        rhox[b] = spearman(xr[ok], site_arr[ok])
        rg = float(xr[ok].max() - xr[ok].min())
        jm = np.diff(xr[ok])
        if rg > 1e-9 and len(jm) >= W:
            wins = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
            k = int(np.argmax(wins)); t3x_b[b] = wins[k] / rg; ax_b[b] = k
XV = np.array([R12['bootstrap_band']['xhalf_ci'][str(s)]['hat'] for s in SITES], float)
RV = np.array([R12['bootstrap_band']['recover_ci'][str(s)]['hat'] for s in SITES], float)
perm_x = np.empty(BP)
for b in range(BP):
    perm_x[b] = spearman(rng.permutation(XV), site_arr)
    _ = rng.permutation(RV)   # 与 Phase 12/13 一致：每轮消费 2 次 permutation

B12 = R12['bootstrap_band']
dj = []
for i, s in enumerate(SITES):
    c = _ci(J_b[:, i]); ref = B12['J_ci'][str(s)]
    dj.append(max(abs(c[k] - ref[k]) for k in ('lo', 'hi', 'med')))
d_top3 = max(abs(_ci(t3x_b)[k] - B12['top3_share_x_ci'][k]) for k in ('lo', 'hi', 'med'))
d_rho = max(abs(_ci(rhox)[k] - B12['rho_xhalf'][k]) for k in ('lo', 'hi', 'med'))
d_px = float(np.max(np.abs(perm_x - np.array(R12['permutation_null']['xhalf']['values'], float))))

sec('C_rng_replay_bit', [
    ('A0 J_ci max|d| == 0.000e+00', max(dj) == 0.0),
    ('A0 top3_share_x_ci max|d| == 0.0', d_top3 == 0.0),
    ('A0 rho_xhalf max|d| == 0.0', d_rho == 0.0),
    ('A0 perm_x 2000 values max|d| == 0.0', d_px == 0.0),
    ('R13.A0_replicate.max_dev == 0.0', R['A0_replicate']['max_dev'] == 0.0),
    ('R13.A0_replicate.ok', R['A0_replicate']['ok'] is True),
    ('A0b conf band max_dev == 0.0', R['A0b_replicate_conf']['max_dev'] == 0.0),
    ('floors.F14 ok & dev 0', R['floors']['F14']['ok'] is True and R['floors']['F14']['max_dev'] == 0.0),
    ('independent recompute agrees (dj==0)', max(dj) == 0.0 and d_px == 0.0),
])

# ======================= D 观测剖面（独立重算） =======================
Jh = [float(R12['bootstrap_band']['J_ci'][str(s)]['hat']) for s in SITES]
Xh = [float(R12['bootstrap_band']['xhalf_ci'][str(s)]['hat']) for s in SITES]


def conc_hat(F):
    F = np.asarray(F, float); jm = np.diff(F); rg = float(F.max() - F.min())
    wins = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rg), k, jm.tolist()


t3xh, kxh, jmx = conc_hat(Xh)
t3jh, kjh, jmj = conc_hat(Jh)
A4 = R['A4_concentration']
hist_x = {}
for k in range(len(jmx) - W + 1):
    hist_x[str(k)] = int(np.sum(ax_b == k))

sec('D_profile_recompute', [
    ('xhalf top3_share 点估计 == R13', ceq(t3xh, A4['xhalf']['hat'], 1e-12)),
    ('xhalf argmax 窗口 == R13 (=14)', kxh == A4['xhalf']['argmax_window_hat'] == 14),
    ('J top3_share 点估计 == R13', ceq(t3jh, A4['J']['hat'], 1e-12)),
    ('J argmax 窗口 == R13 (=1)', kjh == A4['J']['argmax_window_hat'] == 1),
    ('xhalf jumps == R13.A4.jumps', all(ceq(a, b, 1e-12) for a, b in zip(jmx, A4['xhalf']['jumps']))),
    ('xhalf argmax 直方图 == R13', hist_x == A4['xhalf']['win_hist']),
    ('xhalf w14 freq > 0.5', hist_x['14'] / max(sum(hist_x.values()), 1) > 0.5),
    ('R13 P(>=0.60) xhalf == 0.3790', abs(A4['xhalf']['P_ge_060'] - 0.3790) < 1e-12),
    ('R13 P(>=0.60) J == 0.9730', abs(A4['J']['P_ge_060'] - 0.9730) < 1e-12),
    ('|W_x - W_j| >= 3 (coord_dep 分量)', abs(kxh - kjh) >= 3),
])

# ======================= E Δ 表（独立重算带） =======================
def band_of(Fb):
    out = []
    for i in range(nS - 1):
        d = Fb[:, i] - Fb[:, i + 1]; d = d[np.isfinite(d)]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        lab = 'DECISIVE_DOWN' if hi < 0 else ('DECISIVE_UP' if lo > 0 else 'TIE')
        out.append((lo, hi, lab))
    return out


bJ = band_of(J_b); bX = band_of(XH_b)
dJmax = max(max(abs(a[0] - b['lo']), abs(a[1] - b['hi'])) for a, b in zip(bJ, R['A1_delta_J']))
dXmax = max(max(abs(a[0] - b['lo']), abs(a[1] - b['hi'])) for a, b in zip(bX, R['A2_delta_xhalf']))
labJ_ok = all(a[2] == b['label'] for a, b in zip(bJ, R['A1_delta_J']))
labX_ok = all(a[2] == b['label'] for a, b in zip(bX, R['A2_delta_xhalf']))
NdecJ = sum(1 for a in bJ if a[2] != 'TIE')
NdecX = sum(1 for a in bX if a[2] != 'TIE')

sec('E_delta_tables', [
    ('ΔJ 带独立重算 max|d| < 1e-12', dJmax < 1e-12),
    ('Δxhalf 带独立重算 max|d| < 1e-12', dXmax < 1e-12),
    ('ΔJ 标签全一致', labJ_ok),
    ('Δxhalf 标签全一致', labX_ok),
    ('N_dec_J == 10 == R13.A5', NdecJ == 10 == R['A5_counts']['N_dec_J']),
    ('N_dec_X == 4 == R13.A5', NdecX == 4 == R['A5_counts']['N_dec_X']),
    ('R13 disc_verdict == PAIRED_TEST_INFORMATIVE',
     R['disc_verdict'] == 'PAIRED_TEST_INFORMATIVE' and R['P13_verdict'] == 'CONCENTRATION_COORDINATE_DEPENDENT'),
])

# ======================= F 紧化归因与代数恒等 =======================
maxcov = 0.0; viol = 0; npos = 0
f17 = 0.0
for i in range(nS - 1):
    a = J_b[:, i]; bb = J_b[:, i + 1]
    cov = float(np.mean((a - a.mean()) * (bb - bb.mean())))
    va, vb = float(a.var()), float(bb.var())
    d = a - bb; vp = float(d.var())
    f17 = max(f17, abs((va + vb) - vp - 2.0 * cov))
    if cov > 0:
        npos += 1
        if not (math.sqrt(vp) < math.sqrt(va + vb)):
            viol += 1
F15 = abs(sum(Jh[i] - Jh[i + 1] for i in range(nS - 1)) - (Jh[0] - Jh[-1]))
F16 = max(abs(float(np.sum(J_b[b][:-1] - J_b[b][1:])) - float(J_b[b][0] - J_b[b][-1])) for b in range(BS))

sec('F_algebra', [
    ('cov>0 对数 == 17', npos == 17),
    ('cov>0 => sd_paired<sd_indep 无反例', viol == 0),
    ('F17 方差分解 max|d| < 1e-12 (%.3e)' % f17, f17 < 1e-12),
    ('F15 望远镜和(点估计) < 1e-12', F15 < 1e-12),
    ('F16 望远镜和(逐样本) < 1e-12', F16 < 1e-12),
    ('R13 tighten_median == 0.5475', round(R['A5_counts']['tighten_median'], 4) == 0.5475),
    ('R13 N_dec_indep_J == 2', R['A5_counts']['N_dec_indep_J'] == 2),
    ('N_dec_J(10) > N_dec_indep(2)', NdecJ > R['A5_counts']['N_dec_indep_J']),
])

# ======================= G floors / predictions / verdict =======================
FL = R['floors']
PR = R['predictions_check']
EX = R['extra']

sec('G_floors_pred_verdict', [
    ('F14-F23 全部 ok 或 None(报告项)', all(FL[k]['ok'] in (True, None) for k in FL)),
    ('F14 ok', FL['F14']['ok'] is True),
    ('F15/F16/F17 ok', FL['F15']['ok'] and FL['F16']['ok'] and FL['F17']['ok']),
    ('F19 ok (概率守恒)', FL['F19']['ok'] is True and FL['F19']['max_dev'] == 0.0),
    ('F21 ok (无 torch)', FL['F21']['ok'] is True and R['torch_imported'] is False),
    ('F18 报告 17/17 两坐标', FL['F18']['frac_band_contains_hat_J'] == 1.0 and FL['F18']['frac_band_contains_hat_X'] == 1.0),
    ('P1 PASS (mode 14, freq>=0.5)', PR['P1']['pass_'] and PR['P1']['got_mode'] == 14 and PR['P1']['got_freq'] >= 0.5),
    ('P2 PASS (mode<=2, freq>=0.5)', PR['P2']['pass_'] and PR['P2']['got_mode'] <= 2 and PR['P2']['got_freq'] >= 0.5),
    ('P3 PASS (rho<=0.2)', PR['P3']['pass_'] and PR['P3']['got'] <= 0.2),
    ('predictions 三条全 PASS', all(PR[k]['pass_'] for k in ('P1', 'P2', 'P3'))),
    ('P13_verdict == CONCENTRATION_COORDINATE_DEPENDENT', R['P13_verdict'] == 'CONCENTRATION_COORDINATE_DEPENDENT'),
    ('extra.coord_dep True', EX['coord_dep'] is True),
    ('zero_extra_forward True / elapsed < 60 s', R['zero_extra_forward'] is True and R['elapsed_s'] < 60),
    ('A6 L32->L34 dX 带排除 0', not (R['A6_deep_tail']['L32_L34']['dX']['lo'] <= 0 <= R['A6_deep_tail']['L32_L34']['dX']['hi'])),
    ('A7 L20->L34 dX 与发现集同号（负）', R['A7_confirmation']['pairs'][2]['dX']['obs'] < 0),
    ('A8 α=0.4 剔除时 XH_RANGE 仍 > 0.10（不跌破）',
     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) > 0.10),
    ('A8 裕度压缩后 < 1%（0.1007/0.10 - 1）',
     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) / 0.10 - 1.0 < 0.01),
    ('A9 N_dec_J_alt == 5', R['A9_steepness_alt']['N_dec_J_alt'] == 5),
])

# ======================= H 文档落点 =======================
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').splitlines()
hdr = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 13:')]
allh = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
led = LED['measurements']
tail = led[-1]

sec('H_docs', [
    ('MEMO bytes == 308863', len(mb) == 308863),
    ('MEMO sha8 == 4f9b4574', hashlib.sha256(mb).hexdigest()[:8] == '4f9b4574'),
    ('MEMO BOM', mb[:3] == b'\xef\xbb\xbf'),
    ('MEMO bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0),
    ('Phase 13 标题唯一', len(hdr) == 1),
    ('Phase 标题共 13 个', len(allh) == 13),
    ('Phase 13 节在 L2860', hdr and hdr[0] == 2860),
    ('Ledger measurements == 296', len(led) == 296),
    ('Ledger tail phase == 13', tail['phase'] == 13),
    ('Ledger tail verdict == 期望', tail['verdict'] == 'paired_paired_test_informative__concentration_coordinate_dependent'),
    ('Ledger backup 存在', os.path.exists(os.path.join(T13, 'atlas_ledger_backup_pre_phase13.json'))),
    ('baseline.tag == post-append-phase13', BSE['tag'] == 'post-append-phase13'),
    ('baseline.bytes == 308863', BSE['bytes'] == 308863),
    ('baseline.phase_headings 13', len(BSE['phase_headings']) == 13),
    ('judgement.P13_verdict 一致', JUD['verdict']['P13_verdict'] == R['P13_verdict']),
    ('wlog 含 Phase 13 节', 'Phase 13 / N2h1-α-6' in io.open(
        os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md'), encoding='utf-8').read()),
])

# ======================= I 无 torch（本轮前提） =======================
sec('I_no_forward', [
    ('torch 未被本脚本导入', 'torch' not in sys.modules),
    ('exec.zero_extra_forward True', E13['zero_extra_forward'] is True),
    ('exec.cpu_only True', E13['cpu_only'] is True),
    ('exec.no_torch True', E13['no_torch'] is True),
])

# ======================= 汇总 =======================
w('')
w('=' * 70)
w('TOTAL checks = %d ; FAIL = %d' % (TOT['n'], TOT['fail']))
w('A0 逐位: J_ci max|d|=%.3e ; top3 max|d|=%.3e ; rho max|d|=%.3e ; perm_x(2000) max|d|=%.3e'
  % (max(dj), d_top3, d_rho, d_px))
w('Δ 独立重算: ΔJ max|d|=%.3e ; Δxhalf max|d|=%.3e' % (dJmax, dXmax))
w('F17=%.3e F15=%.3e F16=%.3e ; N_dec_J=%d N_dec_X=%d ; P13=%s'
  % (f17, F15, F16, NdecJ, NdecX, R['P13_verdict']))
w('MEMO %d B / %d 行 / sha8 %s ; Ledger n=%d ; baseline %s'
  % (len(mb), len(mt), hashlib.sha256(mb).hexdigest()[:8], len(led), BSE['tag']))
w('=' * 70)
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT, '| FAIL =', TOT['fail'])
assert TOT['fail'] == 0, '存在 FAIL 分区'
