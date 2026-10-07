# -*- coding: utf-8 -*-
"""Phase 13 可行性探针（**只**重建 Phase 12 已发表量；不产生任何新统计量）。

目的：验证「零额外前向」前提成立 —— 即利用 result_phase12.json 里落盘的逐对矩阵
      (E2_pairs / E6_pairs) + FULL_SWAP_pairs + 冻结 seed，能否**逐位**复现
      Phase 12 的 bootstrap 带（J_ci / top3_share_x_ci / rho_xhalf / 置换零假设 2000 值）。

铁律 (o)：写入走脚本 + 回读；本探针只读不写（除自写 txt 报告）。
"""
import io
import os
import json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R12P = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'result_phase12.json')
E12P = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'execution_phase12.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13', '_feas_probe.txt')
L = []
def w(s=''):
    L.append(s)

R = json.load(io.open(R12P, encoding='utf-8'))
E = json.load(io.open(E12P, encoding='utf-8'))

w('=== Phase 13 可行性探针（仅重建 Phase 12 已发表量）===')
w()

# ---------- 1. 重建 FS_VEC ----------
ORDER = list(R['E2']['6'][0]['order'])
FSD = R['FULL_SWAP_pairs']
FS_VEC = np.array([FSD[x] for x in ORDER], float)
FULL_SWAP = float(np.mean(FS_VEC))
w('[1] FS_VEC 重建：n=%d  mean=%.15f  R12 FULL_SWAP=%.15f  eq=%s' %
  (len(FS_VEC), FULL_SWAP, R['FULL_SWAP'], FULL_SWAP == R['FULL_SWAP']))
w('    FS_VEC min=%.6f max=%.6f' % (FS_VEC.min(), FS_VEC.max()))

# ---------- 2. 重建逐对矩阵 ----------
SWAP_SITES = [int(x) for x in R['sites']['profile']]
PM_swap = np.stack([np.array(R['E2_pairs'][str(s)], dtype=float) for s in SWAP_SITES], 0)
PM_R = np.array(R['E6_pairs'], dtype=float)
xs_sw = np.array([row['alpha'] for row in R['E2'][str(SWAP_SITES[0])]], float)
nS, nA, nP = PM_swap.shape
w('[2] PM_swap.shape=%s  PM_R.shape=%s  xs=%s' % (PM_swap.shape, PM_R.shape, xs_sw.tolist()))
assert PM_R.shape == (nA, nP)

# ---------- 3. 逐字复制 Phase 12 的统计函数 ----------
def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a)
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    return r

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0

def cross_alpha(xs, ys, frac):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
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
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    s_med = float(np.median(rest)) if len(rest) > 1 else 0.0
    return float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')

def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))

# ---------- 4. 重放 BRNG 流 ----------
BOOT = E['bootstrap']
BS = int(BOOT['B']); BP = int(BOOT['B_perm']); SEED = int(BOOT['seed'])
W = int(E['g_family']['G2_concentration']['window'])
BRNG = np.random.default_rng(SEED)
a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
rec_b = np.full((BS, nS), np.nan)
J_b = np.full((BS, nS), np.nan)
XH_b = np.full((BS, nS), np.nan)
recR_b = np.full(BS, np.nan); xhR_b = np.full(BS, np.nan)
rho_b = np.full(BS, np.nan); rhox_b = np.full(BS, np.nan)
t3x_b = np.full(BS, np.nan); t3r_b = np.full(BS, np.nan)
site_arr = np.array(SWAP_SITES, float)
for b in range(BS):
    idx = BRNG.integers(0, nP, nP)
    fs_b = float(FS_VEC[idx].mean())
    if abs(fs_b) < 1e-9:
        continue
    Y = PM_swap[:, :, idx].mean(axis=2) / fs_b
    rec_b[b] = Y[:, a1]
    for i in range(nS):
        J_b[b, i] = J_only(xs_sw, Y[i])
        xv = cross_alpha(xs_sw, Y[i], 0.5)
        XH_b[b, i] = xv if xv is not None else np.nan
    rr = rec_b[b]
    if len(rr) >= 4 and np.all(np.isfinite(rr)):
        rho_b[b] = spearman(rr, site_arr)
        inc = np.diff(rr); tot = rr[-1] - rr[0]
        if tot > 1e-9 and len(inc) >= W:
            wins = [abs(sum(inc[i:i + W])) for i in range(len(inc) - W + 1)]
            t3r_b[b] = max(wins) / tot
    xr = XH_b[b]; okx = np.isfinite(xr)
    if okx.sum() >= 4:
        rhox_b[b] = spearman(xr[okx], site_arr[okx])
        rngx = float(xr[okx].max() - xr[okx].min())
        jm = np.diff(xr[okx])
        if rngx > 1e-9 and len(jm) >= W:
            wins = [abs(sum(jm[j:j + W])) for j in range(len(jm) - W + 1)]
            t3x_b[b] = max(wins) / rngx
    YR = PM_R[:, idx].mean(axis=1) / fs_b
    recR_b[b] = YR[a1]
    xrv = cross_alpha(xs_sw, YR, 0.5)
    xhR_b[b] = xrv if xrv is not None else np.nan

# ---------- 5. 置换零假设重放（消耗同一 BRNG） ----------
B12 = R['bootstrap_band']
XH_HAT = {s: B12['xhalf_ci'][s]['hat'] for s in B12['xhalf_ci']}
XH_SITES = [s for s in SWAP_SITES if XH_HAT[str(s)] is not None]
XV = np.array([XH_HAT[str(s)] for s in XH_SITES], float)
RV = np.array([B12['recover_ci'][str(s)]['hat'] for s in SWAP_SITES], float)
XS_ARR = np.array(XH_SITES, float)
RS_ARR = np.array(SWAP_SITES, float)
perm_x = np.empty(BP); perm_rec = np.empty(BP)
for b in range(BP):
    perm_x[b] = spearman(BRNG.permutation(XV), XS_ARR)
    perm_rec[b] = spearman(BRNG.permutation(RV), RS_ARR)

# ---------- 6. 比对 ----------
w()
w('[3] 逐位比对 Phase 12 已发表量')
worst = 0.0
def cmpv(name, got, want):
    global worst
    d = abs(float(got) - float(want))
    worst = max(worst, d)
    w('    %-34s got=%.15f want=%.15f d=%.3e %s' %
      (name, got, want, d, 'BIT-EXACT' if d == 0.0 else ('ok' if d < 1e-12 else 'MISMATCH')))
    return d

maxd_J = 0.0
for i, s in enumerate(SWAP_SITES):
    c = _ci(J_b[:, i]); ref = B12['J_ci'][str(s)]
    for k in ('lo', 'hi', 'med'):
        maxd_J = max(maxd_J, cmpv('J_ci[L%d].%s' % (s, k), c[k], ref[k]))
    assert c['n_ok'] == ref['n_ok'], 'n_ok mismatch L%d' % s
w('    -> J_ci(18 sites) max|d| = %.3e' % maxd_J)

c = _ci(t3x_b); ref = B12['top3_share_x_ci']
w('    top3_share_x lo/hi/med d = %.3e / %.3e / %.3e' % (
    cmpv('top3_share_x.lo', c['lo'], ref['lo']),
    cmpv('top3_share_x.hi', c['hi'], ref['hi']),
    cmpv('top3_share_x.med', c['med'], ref['med'])))
w('    top3_share_recover lo/hi/med d = %.3e / %.3e / %.3e' % (
    cmpv('top3_share_rec.lo', _ci(t3r_b)['lo'], B12['top3_share_recover_ci']['lo']),
    cmpv('top3_share_rec.hi', _ci(t3r_b)['hi'], B12['top3_share_recover_ci']['hi']),
    cmpv('top3_share_rec.med', _ci(t3r_b)['med'], B12['top3_share_recover_ci']['med'])))
w('    rho_recover lo/hi/med d = %.3e / %.3e / %.3e' % (
    cmpv('rho_recover.lo', _ci(rho_b)['lo'], B12['rho_recover']['lo']),
    cmpv('rho_recover.hi', _ci(rho_b)['hi'], B12['rho_recover']['hi']),
    cmpv('rho_recover.med', _ci(rho_b)['med'], B12['rho_recover']['med'])))
w('    rho_xhalf lo/hi/med d = %.3e / %.3e / %.3e' % (
    cmpv('rho_xhalf.lo', _ci(rhox_b)['lo'], B12['rho_xhalf']['lo']),
    cmpv('rho_xhalf.hi', _ci(rhox_b)['hi'], B12['rho_xhalf']['hi']),
    cmpv('rho_xhalf.med', _ci(rhox_b)['med'], B12['rho_xhalf']['med'])))
w('    R_ci.recover lo/hi d = %.3e / %.3e' % (
    cmpv('R_ci.recover.lo', _ci(recR_b)['lo'], B12['R_ci']['recover']['lo']),
    cmpv('R_ci.recover.hi', _ci(recR_b)['hi'], B12['R_ci']['recover']['hi'])))
w('    R_ci.xhalf lo/hi d = %.3e / %.3e' % (
    cmpv('R_ci.xhalf.lo', _ci(xhR_b)['lo'], B12['R_ci']['xhalf']['lo']),
    cmpv('R_ci.xhalf.hi', _ci(xhR_b)['hi'], B12['R_ci']['xhalf']['hi'])))

# ---------- 7. 置换零假设 2000 值逐位 ----------
ref_px = np.array(R['permutation_null']['xhalf']['values'], float)
d_px = float(np.max(np.abs(perm_x - ref_px)))
ref_pr = np.array(R['permutation_null']['recover']['values'], float)
d_pr = float(np.max(np.abs(perm_rec - ref_pr)))
w('    perm_x (2000 values) max|d| = %.3e  -> %s' % (d_px, 'BIT-EXACT' if d_px == 0.0 else 'MISMATCH'))
w('    perm_rec(2000 values) max|d| = %.3e  -> %s' % (d_pr, 'BIT-EXACT' if d_pr == 0.0 else 'MISMATCH'))

# ---------- 8. 结论 ----------
ok = (d_px == 0.0) and (d_pr == 0.0) and (maxd_J < 1e-12) and (abs(FULL_SWAP - R['FULL_SWAP']) == 0.0)
w()
w('=== 可行性判据 ===')
w('  FULL_SWAP 重建逐位一致      : %s' % (FULL_SWAP == R['FULL_SWAP']))
w('  J_ci(18 位点)  max|d| <= 1e-12: %s (%.3e)' % (maxd_J < 1e-12, maxd_J))
w('  置换零假设 2000 值逐位一致   : %s' % (d_px == 0.0 and d_pr == 0.0))
w('  总最坏偏差                  : %.3e' % worst)
w('  ==> 零额外前向前提 %s' % ('成立（可进入 Phase 13 正式设计）' if ok else '不成立（需改用近似分母）'))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L[-14:]))
print('WROTE', OUT)
