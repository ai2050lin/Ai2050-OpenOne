# -*- coding: utf-8 -*-
"""Phase 13 / N2h1-alpha-6 —— 位点间配对 bootstrap（零额外前向的纯再分析）。

输入：Phase 12 落盘的逐对矩阵（E2_pairs / E6_pairs / E5_pairs）+ FULL_SWAP_pairs + 冻结 seed。
输出：result_phase13.json + n2h1a6_report_qwen3-4b.txt。

装置铁律：
  (o) 关键修改走 Python 补丁 + 回读复核（本文件由 gen 脚本族维护）。
  (q) 应由构造决定的量写成硬断言 —— F14(逐位复现) / F15,F16(望远镜和) / F17(方差分解代数恒等)。
  (r) 端点量若由构造决定饱和须降级 —— 本 Phase 不引入端点量。
  (s) 归一化方向须与物理方向一致 —— 本 Phase 不用 first_reach 归一化。
"""
import io
import os
import sys
import json
import time
import hashlib

import numpy as np

SMOKE = ('--smoke' in sys.argv)
ROOT = r'D:\AI2050\Ai2050-OpenOne'
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13')

EXEC_P = os.path.join(T13, 'execution_phase13.json')
SEAL_P = os.path.join(T13, 'N2h1a6_design_seal.json')
R12_P = os.path.join(T12, 'result_phase12.json')
E12_P = os.path.join(T12, 'execution_phase12.json')

# SMOKE 输出走 smoke 子目录，避免覆盖正式产物
OUTDIR = os.path.join(T13, 'smoke') if SMOKE else T13
os.makedirs(OUTDIR, exist_ok=True)
RESULT = os.path.join(OUTDIR, 'result_phase13.json')
REPORT = os.path.join(OUTDIR, 'n2h1a6_report_qwen3-4b.txt')

lines = []
def w(s=''):
    lines.append(s)
    print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


t0 = time.time()

# ======================= 0. 冻结件读取与前置断言 =======================
E = json.load(io.open(EXEC_P, encoding='utf-8'))
SEAL = json.load(io.open(SEAL_P, encoding='utf-8'))
R12 = json.load(io.open(R12_P, encoding='utf-8'))
E12 = json.load(io.open(E12_P, encoding='utf-8'))

_REQ = ['phase', 'name', 'seed', 'source', 'seal_sha256', 'seal_sha8', 'sites', 'alpha_grid',
        'conf_alpha_grid', 'window_W', 'jdose_floor', 'xh_frac', 'bootstrap', 'arms',
        'decision', 'floors', 'result_keys']
_missing = [k for k in _REQ if k not in E]
assert not _missing, 'exec 缺少必需字段: %s' % _missing
assert E['phase'] == 13
assert E['zero_extra_forward'] is True
assert sha(SEAL_P) == E['seal_sha256'], 'seal sha 不一致'
assert sha(R12_P) == E['source']['phase12_result_sha256'], 'F0: phase12 result sha 不一致'
assert 'torch' not in sys.modules, 'F21: 本轮不得导入 torch'

SITES = [int(x) for x in E['sites']['profile']]
CSITES = [int(x) for x in E['sites']['confirmation']]
ALPHAS = [float(x) for x in E['alpha_grid']]
CALPHAS = [float(x) for x in E['conf_alpha_grid']]
W = int(E['window_W'])
JFL = float(E['jdose_floor'])
XHF = float(E['xh_frac'])
BS = int(E['bootstrap']['B']) if not SMOKE else 200
BP = int(E['bootstrap']['B_perm']) if not SMOKE else 200
SEED = int(E['bootstrap']['seed'])
nS = len(SITES)
assert nS == 18 and len(CSITES) == 4

w('=' * 78)
w('Phase 13 / N2h1-alpha-6  位点间配对 bootstrap（零额外前向）')
w('SMOKE=%s   B=%d  B_perm=%d  seed=%d  window=%d' % (SMOKE, BS, BP, SEED, W))
w('seal sha8=%s   exec sha8=%s   phase12 result sha8=%s' % (
    E['seal_sha8'], sha(EXEC_P)[:8], E['source']['phase12_result_sha256'][:8]))
w('=' * 78)

# ======================= 1. 重建逐对矩阵 =======================
ORDER = list(R12['E2']['6'][0]['order'])
FS_VEC = np.array([R12['FULL_SWAP_pairs'][x] for x in ORDER], float)
FULL_SWAP = float(np.mean(FS_VEC))
assert FULL_SWAP == R12['FULL_SWAP'], 'F0b: FULL_SWAP 重建失败'
nP = len(ORDER)
PM_swap = np.stack([np.array(R12['E2_pairs'][str(s)], dtype=float) for s in SITES], 0)
PM_R = np.array(R12['E6_pairs'], dtype=float)
PM_conf = np.stack([np.array(R12['E5_pairs'][str(s)], dtype=float) for s in CSITES], 0)
xs_sw = np.array(ALPHAS, float)
xs_cf = np.array(CALPHAS, float)
assert PM_swap.shape == (nS, len(ALPHAS), nP), PM_swap.shape
assert PM_R.shape == (len(ALPHAS), nP), PM_R.shape
assert PM_conf.shape == (len(CSITES), len(CALPHAS), len(R12['E5_pairs'][str(CSITES[0])][0])), PM_conf.shape
w('[1] 逐对矩阵重建 OK  FS_VEC n=%d mean=%.12f  PM_swap=%s  PM_R=%s  PM_conf=%s' % (
    nP, FULL_SWAP, PM_swap.shape, PM_R.shape, PM_conf.shape))

# ======================= 2. 统计函数（逐字复制 Phase 12） =======================
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
    m = xs >= JFL
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    s_med = float(np.median(rest)) if len(rest) > 1 else 0.0
    return float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')


def J_iqr(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= JFL
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    if len(rest) < 3:
        return np.nan
    q1, q3 = np.percentile(rest, [25, 75])
    den = float(q3 - q1)
    return float(s[k_i] / den) if den > 1e-12 else float('inf')


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))


def conc_hat(Fhat):
    """点估计：max over windows of |sum of W adjacent jumps| / range。"""
    F = np.asarray(Fhat, float)
    jm = np.diff(F)
    rng = float(F.max() - F.min())
    if rng <= 1e-12 or len(jm) < W:
        return None, None, jm.tolist()
    wins = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rng), k, jm.tolist()


# ======================= 3. 重放 BRNG 流（A0/A0b） =======================
BRNG = np.random.default_rng(SEED)
a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
rec_b = np.full((BS, nS), np.nan)
J_b = np.full((BS, nS), np.nan)
JA_b = np.full((BS, nS), np.nan)
XH_b = np.full((BS, nS), np.nan)
recR_b = np.full(BS, np.nan); xhR_b = np.full(BS, np.nan)
rho_b = np.full(BS, np.nan); rhox_b = np.full(BS, np.nan)
t3x_b = np.full(BS, np.nan); t3r_b = np.full(BS, np.nan)
ax_b = np.full(BS, -1, int)     # xhalf 的 argmax 窗口起始索引
site_arr = np.array(SITES, float)
for b in range(BS):
    idx = BRNG.integers(0, nP, nP)
    fs_b = float(FS_VEC[idx].mean())
    if abs(fs_b) < 1e-9:
        continue
    Y = PM_swap[:, :, idx].mean(axis=2) / fs_b
    rec_b[b] = Y[:, a1]
    for i in range(nS):
        J_b[b, i] = J_only(xs_sw, Y[i])
        JA_b[b, i] = J_iqr(xs_sw, Y[i])
        xv = cross_alpha(xs_sw, Y[i], XHF)
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
            k = int(np.argmax(wins))
            t3x_b[b] = max(wins) / rngx
            ax_b[b] = k
    YR = PM_R[:, idx].mean(axis=1) / fs_b
    recR_b[b] = YR[a1]
    xrv = cross_alpha(xs_sw, YR, XHF)
    xhR_b[b] = xrv if xrv is not None else np.nan

# ---- 置换零假设（继续消费同一 BRNG） ----
B12 = R12['bootstrap_band']
XH_HAT = {s: B12['xhalf_ci'][s]['hat'] for s in B12['xhalf_ci']}
JHAT = {s: B12['J_ci'][s]['hat'] for s in B12['J_ci']}
XV = np.array([XH_HAT[str(s)] for s in SITES], float)
RV = np.array([B12['recover_ci'][str(s)]['hat'] for s in SITES], float)
perm_x = np.empty(BP); perm_rec = np.empty(BP)
for b in range(BP):
    perm_x[b] = spearman(BRNG.permutation(XV), site_arr)
    perm_rec[b] = spearman(BRNG.permutation(RV), site_arr)

# ---- 确认集 bootstrap（继续消费同一 BRNG） ----
Cf = dict(rho_recover=None, rho_xhalf=None, delta_x=[], delta_j=[])
if not SMOKE and len(CSITES) >= 4:
    a1c = int(np.argmin(np.abs(xs_cf - 1.0)))
    nPc = PM_conf.shape[2]
    rb = np.full(BS, np.nan); xb = np.full(BS, np.nan)
    Jc_b = np.full((BS, len(CSITES)), np.nan)
    XHc_b = np.full((BS, len(CSITES)), np.nan)
    for b in range(BS):
        idx = BRNG.integers(0, nPc, nPc)
        Yc = PM_conf[:, :, idx].mean(axis=2) / FULL_SWAP
        rb[b] = spearman(Yc[:, a1c], np.array(CSITES, float))
        xr = [cross_alpha(xs_cf, Yc[k], XHF) for k in range(len(CSITES))]
        xr = np.array([np.nan if v is None else v for v in xr], float)
        for k in range(len(CSITES)):
            Jc_b[b, k] = J_only(xs_cf, Yc[k])
        XHc_b[b, :] = xr
        ok = np.isfinite(xr)
        if ok.sum() >= 4:
            xb[b] = spearman(xr[ok], np.array(CSITES, float)[ok])
    Cf['rho_recover'] = _ci(rb)
    Cf['rho_xhalf'] = _ci(xb)
    Cf['_Jb'] = Jc_b
    Cf['_XHb'] = XHc_b

w('[2] BRNG 重放完成 (BS=%d, BP=%d, conf_BS=%d)' % (BS, BP, BS if not SMOKE else 0))

# ======================= 4. A0 / A0b 装置锚 =======================
A0 = dict(skipped=bool(SMOKE), dev={}, max_dev=None, ok=None)
if not SMOKE:
    dev = {}
    dev['J_ci'] = 0.0
    for i, s in enumerate(SITES):
        c = _ci(J_b[:, i]); ref = B12['J_ci'][str(s)]
        for k in ('lo', 'hi', 'med'):
            dev['J_ci'] = max(dev['J_ci'], abs(c[k] - ref[k]))
        assert c['n_ok'] == ref['n_ok'], 'J_ci n_ok 不一致 L%d' % s
    c = _ci(t3x_b); ref = B12['top3_share_x_ci']
    dev['top3_share_x'] = max(abs(c[k] - ref[k]) for k in ('lo', 'hi', 'med'))
    c = _ci(t3r_b); ref = B12['top3_share_recover_ci']
    dev['top3_share_recover'] = max(abs(c[k] - ref[k]) for k in ('lo', 'hi', 'med'))
    c = _ci(rho_b); ref = B12['rho_recover']
    dev['rho_recover'] = max(abs(c[k] - ref[k]) for k in ('lo', 'hi', 'med'))
    c = _ci(rhox_b); ref = B12['rho_xhalf']
    dev['rho_xhalf'] = max(abs(c[k] - ref[k]) for k in ('lo', 'hi', 'med'))
    dev['R_ci_recover'] = max(abs(_ci(recR_b)[k] - B12['R_ci']['recover'][k]) for k in ('lo', 'hi'))
    dev['R_ci_xhalf'] = max(abs(_ci(xhR_b)[k] - B12['R_ci']['xhalf'][k]) for k in ('lo', 'hi'))
    dpx = float(np.max(np.abs(perm_x - np.array(R12['permutation_null']['xhalf']['values'], float))))
    dpr = float(np.max(np.abs(perm_rec - np.array(R12['permutation_null']['recover']['values'], float))))
    dev['perm_x_2000'] = dpx
    dev['perm_rec_2000'] = dpr
    A0['dev'] = dev
    A0['max_dev'] = float(max(dev.values()))
    A0['ok'] = bool(A0['max_dev'] == 0.0)
    # 点估计复现（top3_share_x / XH_RANGE / jumps）
    t3_hat, k_hat, jm_hat = conc_hat([XH_HAT[str(s)] for s in SITES])
    A0['top3_share_x_hat_repro'] = abs(t3_hat - R12['xhalf']['top3_share_x'])
    A0['argmax_window_hat'] = k_hat
    A0['jumps_x_hat_repro'] = float(np.max(np.abs(np.array(jm_hat) - np.array(R12['xhalf']['jumps'], float))))

A0b = dict(skipped=bool(SMOKE), dev={}, max_dev=None, ok=None)
if not SMOKE:
    ref = R12['E5']['rho_boot_xhalf']
    got = Cf['rho_xhalf']
    d = {k: abs(got[k] - ref[k]) for k in ('lo', 'hi', 'med')}
    A0b['dev'] = d
    A0b['max_dev'] = float(max(d.values()))
    A0b['ok'] = bool(A0b['max_dev'] < 1e-12)

w('')
w('--- A0 装置锚（逐位复现 Phase 12）---')
if SMOKE:
    w('  (SMOKE: 跳过 —— B=%d != 2000)' % BS)
else:
    for k, v in sorted(A0['dev'].items()):
        w('  %-22s max|d| = %.3e  %s' % (k, v, 'BIT-EXACT' if v == 0.0 else ('ok' if v < 1e-12 else 'MISMATCH')))
    w('  top3_share_x 点估计复现 max|d| = %.3e ; argmax 窗口 = %d' % (A0['top3_share_x_hat_repro'], A0['argmax_window_hat']))
    w('  xhalf jumps 复现 max|d| = %.3e' % A0['jumps_x_hat_repro'])
    w('  ==> A0 %s (max|d| = %.3e)' % ('BIT-EXACT' if A0['ok'] else 'FAILED', A0['max_dev']))
w('  A0b（确认集带）%s' % ('(SMOKE 跳过)' if SMOKE else ('BIT-EXACT max|d|=%.3e' % A0b['max_dev'] if A0b['ok'] else 'FAILED max|d|=%.3e' % A0b['max_dev'])))

# ======================= 5. A1 / A2 配对判别 =======================
def deltas_of(Fb, Fhat, tag):
    rows = []
    for i in range(nS - 1):
        d = Fb[:, i] - Fb[:, i + 1]
        d = d[np.isfinite(d)]
        obs = float(Fhat[i] - Fhat[i + 1])
        if len(d) >= 10:
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            med = float(np.median(d))
        else:
            lo = hi = med = None
        if lo is None:
            lab = 'NA'
        elif hi < 0:
            lab = 'DECISIVE_DOWN'
        elif lo > 0:
            lab = 'DECISIVE_UP'
        else:
            lab = 'TIE'
        rows.append(dict(a=int(SITES[i]), b=int(SITES[i + 1]), obs=obs, lo=lo, hi=hi, med=med,
                         n_finite=int(len(d)), label=lab, contains_hat=bool(lo is not None and lo <= obs <= hi),
                         bias=(None if med is None else float(med - obs))))
    return rows


Fhat_J = [JHAT[str(s)] for s in SITES]
Fhat_X = [XH_HAT[str(s)] for s in SITES]
A1 = deltas_of(J_b, Fhat_J, 'J')
A2 = deltas_of(XH_b, Fhat_X, 'xhalf')
N_dec_J = sum(1 for r in A1 if r['label'] in ('DECISIVE_DOWN', 'DECISIVE_UP'))
N_dec_X = sum(1 for r in A2 if r['label'] in ('DECISIVE_DOWN', 'DECISIVE_UP'))

w('')
w('--- A1 相邻位点配对差 Delta_J（17 对）---')
w('%-9s %10s %10s %10s %-15s %7s' % ('pair', 'Dobs', 'lo', 'hi', 'label', 'bias'))
for r in A1:
    w('L%-3d->L%-3d %10.4f %10.4f %10.4f %-15s %+7.3f' % (
        r['a'], r['b'], r['obs'], r['lo'], r['hi'], r['label'], r['bias']))
w('  N_dec_J = %d / 17' % N_dec_J)

w('')
w('--- A2 相邻位点配对差 Delta_xhalf（17 对）---')
w('%-9s %10s %10s %10s %-15s %7s' % ('pair', 'Dobs', 'lo', 'hi', 'label', 'bias'))
for r in A2:
    w('L%-3d->L%-3d %10.4f %10.4f %10.4f %-15s %+7.3f' % (
        r['a'], r['b'], r['obs'], r['lo'], r['hi'], r['label'], r['bias']))
w('  N_dec_X = %d / 17' % N_dec_X)
w('  带含点估计的对数: J %d/17 ; xhalf %d/17' % (
    sum(1 for r in A1 if r['contains_hat']), sum(1 for r in A2 if r['contains_hat'])))

# ======================= 6. A3 紧化归因 =======================
A3 = []
n_cov_pos = 0
f17_max = 0.0
a3_viol = []
for i in range(nS - 1):
    a = J_b[:, i]; bb = J_b[:, i + 1]
    m = np.isfinite(a) & np.isfinite(bb)
    a, bb = a[m], bb[m]
    cov = float(np.mean((a - a.mean()) * (bb - bb.mean())))
    va, vb = float(a.var()), float(bb.var())
    d = a - bb
    vp = float(d.var())
    sdp = float(np.sqrt(vp)); sdi = float(np.sqrt(va + vb))
    rho = float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else None
    f17_max = max(f17_max, abs((va + vb) - vp - 2.0 * cov))
    if cov > 0:
        n_cov_pos += 1
        if not (sdp < sdi):
            a3_viol.append([int(SITES[i]), int(SITES[i + 1])])
    A3.append(dict(a=int(SITES[i]), b=int(SITES[i + 1]), cov=cov, var_a=va, var_b=vb, var_d=vp,
                   sd_paired=sdp, sd_indep=sdi, tighten=float(sdp / sdi) if sdi > 0 else None,
                   rho_pair=rho))
tight_med = float(np.median([r['tighten'] for r in A3]))
w('')
w('--- A3 紧化归因（配对口径 vs 独立口径）---')
w('%-9s %10s %10s %10s %9s %9s' % ('pair', 'cov', 'sd_pair', 'sd_indep', 'rho_pair', 'tighten'))
for r in A3:
    w('L%-3d->L%-3d %10.4f %10.4f %10.4f %9.4f %9.4f' % (
        r['a'], r['b'], r['cov'], r['sd_paired'], r['sd_indep'], r['rho_pair'], r['tighten']))
w('  cov>0 的对数 = %d / 17 ; 反例 = %s' % (n_cov_pos, a3_viol if a3_viol else 'NONE'))
w('  收紧比中位数 = %.4f' % tight_med)
w('  F17 方差分解恒等 max|d| = %.3e' % f17_max)

# ======================= 7. A5 独立 vs 配对计数 =======================
N_dec_indep = 0
indep_pairs = []
for i in range(nS - 1):
    ra = B12['J_ci'][str(SITES[i])]; rb2 = B12['J_ci'][str(SITES[i + 1])]
    if ra['lo'] is None or rb2['lo'] is None:
        continue
    disjoint = (ra['hi'] < rb2['lo']) or (rb2['hi'] < ra['lo'])
    if disjoint:
        N_dec_indep += 1
        indep_pairs.append([int(SITES[i]), int(SITES[i + 1])])
A5 = dict(N_dec_J=N_dec_J, N_dec_X=N_dec_X, N_dec_indep_J=N_dec_indep,
          indep_resolvable_pairs=indep_pairs, tighten_median=tight_med)
w('')
w('--- A5 可分辨对数：独立区间口径 vs 配对口径 ---')
w('  独立区间口径（Phase 12 J_ci 带不相交）: %d / 17' % N_dec_indep)
w('  配对口径 J_swap                     : %d / 17' % N_dec_J)
w('  配对口径 xhalf                      : %d / 17' % N_dec_X)
w('  收紧比中位数 = %.4f' % tight_med)

# ======================= 8. A4 集中度尾部概率 =======================
t3x_hat, kx_hat, jmx = conc_hat(Fhat_X)
t3j_hat, kj_hat, jmj = conc_hat(Fhat_J)
sb_x = t3x_b[np.isfinite(t3x_b)]
jm2 = np.diff(np.array(Fhat_J, float)); rng2 = float(np.array(Fhat_J, float).max() - np.array(Fhat_J, float).min())
wins_j = [abs(float(np.sum(jm2[j:j + W]))) for j in range(len(jm2) - W + 1)]
sb_j = np.array(wins_j, float) / rng2      # 点估计用的窗口份额（非 bootstrap）

# J 坐标的 bootstrap 份额（需在 b 循环外用同一 idx 重算；BRNG 已耗尽 ⇒ 用派生流）
# 说明：为避免二次消耗 BRNG，这里用**已存的 J_b** 直接算（J_b 就是同一 idx_b 下的剖面）。
t3j_b = np.full(BS, np.nan); aj_b = np.full(BS, -1, int)
for b in range(BS):
    f = J_b[b]; ok = np.isfinite(f)
    if ok.sum() < W + 2:
        continue
    ff = f[ok]
    jj = np.diff(ff)
    rg = float(ff.max() - ff.min())
    if rg <= 1e-9 or len(jj) < W:
        continue
    wins = [abs(float(np.sum(jj[j:j + W]))) for j in range(len(jj) - W + 1)]
    k = int(np.argmax(wins))
    t3j_b[b] = wins[k] / rg
    aj_b[b] = k

def tail_prob(v, thr, side):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return None
    if side == 'ge':
        return float(np.sum(v >= thr) / n)
    return float(np.sum(v <= thr) / n)

P_few_x = tail_prob(t3x_b, 0.60, 'ge'); P_acc_x = tail_prob(t3x_b, 0.40, 'le')
P_few_j = tail_prob(t3j_b, 0.60, 'ge'); P_acc_j = tail_prob(t3j_b, 0.40, 'le')
P_mid_x = float(1.0 - P_few_x - P_acc_x); P_mid_j = float(1.0 - P_few_j - P_acc_j)

def hist_of(idx_arr, nwin):
    h = {}
    v = np.asarray(idx_arr, int)
    v = v[v >= 0]
    for k in range(nwin):
        h[str(k)] = int(np.sum(v == k))
    return h

hw = len(jmx) - W + 1
A4 = dict(
    xhalf=dict(hat=t3x_hat, argmax_window_hat=kx_hat, jumps=jmx, range=float(np.max(Fhat_X) - np.min(Fhat_X)),
               P_ge_060=P_few_x, P_le_040=P_acc_x, P_mid=P_mid_x,
               ci=_ci(t3x_b), win_hist=hist_of(ax_b, hw), n_win=hw),
    J=dict(hat=t3j_hat, argmax_window_hat=kj_hat, jumps=jmj, range=rng2,
           P_ge_060=P_few_j, P_le_040=P_acc_j, P_mid=P_mid_j,
           ci=_ci(t3j_b), win_hist=hist_of(aj_b, hw), n_win=hw),
)
w('')
w('--- A4 集中度：尾部概率 + 窗口定位（窗口 W=%d，共 %d 个窗口）---' % (W, hw))
for tag, d in (('xhalf', A4['xhalf']), ('J_swap', A4['J'])):
    w('  [%s] top%d_share = %.4f (band [%s, %s])  argmax 窗口 = %d' % (
        tag, W, d['hat'],
        ('%.4f' % d['ci']['lo']) if d['ci']['lo'] is not None else 'n/a',
        ('%.4f' % d['ci']['hi']) if d['ci']['hi'] is not None else 'n/a',
        d['argmax_window_hat']))
    w('        P(share>=0.60) = %.4f ; P(share<=0.40) = %.4f ; P(mid) = %.4f' % (
        d['P_ge_060'], d['P_le_040'], d['P_mid']))
    w('        跳变(17): %s' % ' '.join('%+.4f' % x for x in d['jumps']))
    w('        argmax 窗口频次: %s' % ' '.join('w%d:%d' % (k, d['win_hist'][str(k)]) for k in range(hw)))

# ======================= 9. A6 深尾反转 =======================
A6 = {}
for (aa, bb) in ([28, 30], [30, 32], [32, 34]):
    if aa not in SITES or bb not in SITES:
        continue
    i = SITES.index(aa); j = SITES.index(bb)
    dj = J_b[:, i] - J_b[:, j]; dx = XH_b[:, i] - XH_b[:, j]
    A6['L%d_L%d' % (aa, bb)] = dict(
        dJ=dict(obs=float(Fhat_J[i] - Fhat_J[j]), **_ci(dj)),
        dX=dict(obs=float(Fhat_X[i] - Fhat_X[j]), **_ci(dx)),
    )
w('')
w('--- A6 深尾反转（L30 谷 / L34 反弹）---')
for k, v in A6.items():
    w('  %s : dJ obs=%+.4f band=[%+.4f,%+.4f] ; dX obs=%+.6f band=[%+.6f,%+.6f]' % (
        k, v['dJ']['obs'], v['dJ']['lo'], v['dJ']['hi'], v['dX']['obs'], v['dX']['lo'], v['dX']['hi']))

# ======================= 10. A7 确认集配对 =======================
A7 = dict(skipped=bool(SMOKE))
if not SMOKE:
    cJ = Cf['_Jb']; cX = Cf['_XHb']
    cJh = [R12['E5'][str(s)]['J'] for s in CSITES]
    cXh = [R12['E5'][str(s)]['xhalf'] for s in CSITES]
    dl = []
    for i in range(len(CSITES) - 1):
        dj = cJ[:, i] - cJ[:, i + 1]; dx = cX[:, i] - cX[:, i + 1]
        dl.append(dict(a=int(CSITES[i]), b=int(CSITES[i + 1]),
                       dJ=dict(obs=float(cJh[i] - cJh[i + 1]), **_ci(dj)),
                       dX=dict(obs=float(cXh[i] - cXh[i + 1]), **_ci(dx))))
    A7 = dict(skipped=False, pairs=dl, rho_boot_recover=Cf['rho_recover'], rho_boot_xhalf=Cf['rho_xhalf'])
    w('')
    w('--- A7 确认集配对（%d 位点 -> %d 对, n=17）---' % (len(CSITES), len(CSITES) - 1))
    for d in dl:
        w('  L%-3d->L%-3d dJ obs=%+.4f band=[%+.4f,%+.4f] ; dX obs=%+.6f band=[%+.6f,%+.6f]' % (
            d['a'], d['b'], d['dJ']['obs'], d['dJ']['lo'], d['dJ']['hi'],
            d['dX']['obs'], d['dX']['lo'], d['dX']['hi']))

# ======================= 11. A8 alpha 网格留一 =======================
Ymat = {s: [r['dDonor'] / FULL_SWAP for r in R12['E2'][str(s)]] for s in SITES}
A8 = dict(variants=[], skipped=bool(SMOKE))
if not SMOKE:
    keep_idx = [k for k, a in enumerate(ALPHAS) if a not in (0.0, 1.0)]
    for k in keep_idx:
        xs2 = [a for j, a in enumerate(ALPHAS) if j != k]
        xh2 = []
        for s in SITES:
            ys2 = [y for j, y in enumerate(Ymat[s]) if j != k]
            xh2.append(cross_alpha(xs2, ys2, XHF))
        if any(v is None for v in xh2):
            A8['variants'].append(dict(drop_alpha=float(ALPHAS[k]), xh_range=None, top3=None, ok=False))
            continue
        rg = float(max(xh2) - min(xh2))
        t3, kk2, _ = conc_hat(xh2)
        A8['variants'].append(dict(drop_alpha=float(ALPHAS[k]), xh_range=rg, top3=t3,
                                   argmax_window=kk2, ok=True))
    rgs = [v['xh_range'] for v in A8['variants'] if v['ok']]
    t3s = [v['top3'] for v in A8['variants'] if v['ok']]
    A8['range_min'] = min(rgs); A8['range_max'] = max(rgs)
    A8['top3_min'] = min(t3s); A8['top3_max'] = max(t3s)
    A8['range_span'] = A8['range_max'] - A8['range_min']
    A8['top3_span'] = A8['top3_max'] - A8['top3_min']
    w('')
    w('--- A8 alpha 网格留一稳健性（%d 个变体）---' % len(A8['variants']))
    for v in A8['variants']:
        w('  drop a=%-5.3f XH_RANGE=%s top3=%s' % (
            v['drop_alpha'],
            ('%.4f' % v['xh_range']) if v['xh_range'] is not None else 'n/a',
            ('%.4f' % v['top3']) if v['top3'] is not None else 'n/a'))
    w('  XH_RANGE in [%.4f, %.4f] span=%.4f ; top3 in [%.4f, %.4f] span=%.4f' % (
        A8['range_min'], A8['range_max'], A8['range_span'],
        A8['top3_min'], A8['top3_max'], A8['top3_span']))

# ======================= 12. A9 陡度统计量替代 =======================
JAlt_hat = [float(J_iqr(xs_sw, np.array([r['dDonor'] for r in R12['E2'][str(s)]], float) / FULL_SWAP)) for s in SITES]
rho_alt = spearman(JAlt_hat, Fhat_J)
DJA = [float(JAlt_hat[i] - JAlt_hat[i + 1]) for i in range(nS - 1)]
A9rows = []
for i in range(nS - 1):
    d = JA_b[:, i] - JA_b[:, i + 1]
    d = d[np.isfinite(d)]
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) >= 10 else (None, None)
    lab = 'NA' if lo is None else ('DECISIVE_DOWN' if hi < 0 else ('DECISIVE_UP' if lo > 0 else 'TIE'))
    A9rows.append(dict(a=int(SITES[i]), b=int(SITES[i + 1]), obs=float(DJA[i]), lo=lo, hi=hi, label=lab))
N_dec_J_alt = sum(1 for r in A9rows if r['label'] in ('DECISIVE_DOWN', 'DECISIVE_UP'))
A9 = dict(rho_J_vs_Jalt=rho_alt, rows=A9rows, N_dec_J_alt=N_dec_J_alt)
w('')
w('--- A9 陡度统计量替代（分母改 IQR）---')
w('  spearman(J_max/median, J_max/IQR) = %.4f' % (rho_alt if rho_alt is not None else float('nan')))
w('  N_dec_J_alt = %d / 17' % N_dec_J_alt)

# ======================= 13. 预注册预测核验 =======================
px = A4['xhalf']['win_hist']; pj = A4['J']['win_hist']
nx = max(sum(px.values()), 1); nj = max(sum(pj.values()), 1)
mode_x = int(max(range(hw), key=lambda k: px[str(k)]))
mode_j = int(max(range(hw), key=lambda k: pj[str(k)]))
XH_J = np.array(A4['xhalf']['jumps'], float); JS_J = np.array(A4['J']['jumps'], float)
rho_jumps = spearman(np.abs(XH_J), np.abs(JS_J))
PR = dict(
    P1=dict(desc='xhalf argmax 窗口众数 = 最深窗口 (idx %d)，频次 >= 0.50' % (hw - 1),
            got_mode=mode_x, got_freq=float(px[str(mode_x)] / nx),
            pass_=bool(mode_x == hw - 1 and px[str(mode_x)] / nx >= 0.50)),
    P2=dict(desc='J_swap argmax 窗口众数 in {0,1,2}，频次 >= 0.50',
            got_mode=mode_j, got_freq=float(pj[str(mode_j)] / nj),
            pass_=bool(mode_j <= 2 and pj[str(mode_j)] / nj >= 0.50)),
    P3=dict(desc='spearman(|jumps_x|, |jumps_J|) <= 0.2',
            got=rho_jumps, pass_=bool(rho_jumps is not None and rho_jumps <= 0.2)),
)
w('')
w('--- 预注册预测核验 ---')
for k in ('P1', 'P2', 'P3'):
    w('  %s %s' % (k, 'PASS' if PR[k]['pass_'] else 'FAIL'))
    w('     %s' % PR[k]['desc'])
    w('     got: %s' % {kk: vv for kk, vv in PR[k].items() if kk not in ('desc', 'pass_')})

# ======================= 14. 判决 =======================
share_x_hat = A4['xhalf']['hat']; share_j_hat = A4['J']['hat']
coord_dep = bool(abs(share_x_hat) >= 0.40 and abs(share_j_hat) >= 0.40 and abs(mode_x - mode_j) >= 3)
G0p = bool((A0['ok'] is True or SMOKE) and (A0b['ok'] is True or SMOKE))
if not G0p:
    P13 = 'DEVICE_ANCHOR_FAILED'
elif N_dec_J == 0 and N_dec_X == 0:
    P13 = 'PAIRED_TEST_UNINFORMATIVE'
elif coord_dep:
    P13 = 'CONCENTRATION_COORDINATE_DEPENDENT'
elif P_few_x >= 0.95 and P_few_j >= 0.95:
    P13 = 'CONCENTRATION_FEW_LAYER_ROBUST'
elif P_acc_x >= 0.95 and P_acc_j >= 0.95:
    P13 = 'CONCENTRATION_ACCUMULATE_ROBUST'
else:
    P13 = 'CONCENTRATION_UNDECIDED'
disc_verdict = 'PAIRED_TEST_UNINFORMATIVE' if (N_dec_J == 0 and N_dec_X == 0) else 'PAIRED_TEST_INFORMATIVE'

w('')
w('=== 判决 ===')
w('  G0p (A0 & A0b 锚) = %s' % G0p)
w('  coord_dep = %s  (|share_x|=%.4f |share_j|=%.4f  |W_x-W_j|=%d)' % (
    coord_dep, abs(share_x_hat), abs(share_j_hat), abs(mode_x - mode_j)))
w('  N_dec_J=%d  N_dec_X=%d  -> disc_verdict=%s' % (N_dec_J, N_dec_X, disc_verdict))
w('  ==> P13_verdict = %s' % P13)

# ======================= 15. floors =======================
el = time.time() - t0
F15 = 0.0
for i in range(nS - 1):
    F15 = max(F15, abs((Fhat_J[i] - Fhat_J[i + 1]) - (Fhat_J[i] - Fhat_J[i + 1])))
F15 = abs(sum(Fhat_J[i] - Fhat_J[i + 1] for i in range(nS - 1)) - (Fhat_J[0] - Fhat_J[-1]))
F16 = 0.0
for b in range(BS):
    f = J_b[b]
    if not np.all(np.isfinite(f)):
        continue
    F16 = max(F16, abs(float(np.sum(f[:-1] - f[1:])) - float(f[0] - f[-1])))
F19 = max(abs(P_few_x + P_acc_x + P_mid_x - 1.0), abs(P_few_j + P_acc_j + P_mid_j - 1.0))
F20 = bool(SMOKE or (len(A7.get('pairs', [])) == 3 and all(
    d['dJ']['lo'] is not None and d['dX']['lo'] is not None for d in A7['pairs'])))
F21 = bool('torch' not in sys.modules)
F23 = bool(SMOKE or (len([v for v in A8['variants'] if v['ok']]) == len(A8['variants'])))
floors = dict(
    F14=dict(ok=A0['ok'], max_dev=A0['max_dev'], skipped=bool(SMOKE)),
    F15=dict(ok=bool(F15 < 1e-12), max_dev=F15),
    F16=dict(ok=bool(F16 < 1e-12), max_dev=F16),
    F17=dict(ok=bool(f17_max < 1e-12), max_dev=f17_max),
    F18=dict(ok=None, frac_band_contains_hat_J=sum(1 for r in A1 if r['contains_hat']) / 17.0,
             frac_band_contains_hat_X=sum(1 for r in A2 if r['contains_hat']) / 17.0,
             note='不设断言（percentile 带不必含点估计）'),
    F19=dict(ok=bool(F19 < 1e-12), max_dev=F19),
    F20=dict(ok=F20),
    F21=dict(ok=F21),
    F22=dict(ok=A0b['ok'], max_dev=A0b['max_dev'], skipped=bool(SMOKE)),
    F23=dict(ok=F23),
)
w('')
w('=== 装置自检（F14-F23）===')
for k in sorted(floors):
    v = floors[k]
    w('  %-5s ok=%-5s %s' % (k, v['ok'], {kk: vv for kk, vv in v.items() if kk not in ('ok',)}))
w('  total %.2fs (SMOKE=%s, 零前向, 纯 CPU)' % (el, SMOKE))

# ======================= 16. 落盘 =======================
res = dict(
    phase=13, name=E['name'], smoke=bool(SMOKE), elapsed_s=round(el, 2),
    zero_extra_forward=True, torch_imported=bool('torch' in sys.modules),
    panel=dict(profile_sites=nS, adjacent_pairs=nS - 1, conf_sites=len(CSITES),
               discovery_pairs=nP, conf_pairs=int(PM_conf.shape[2]), BS=BS, BP=BP, window=W),
    anchors=dict(
        seal_sha8=E['seal_sha8'], exec_sha8=sha(EXEC_P)[:8],
        phase12_result_sha8=E['source']['phase12_result_sha256'][:8],
        phase12_result_sha256=E['source']['phase12_result_sha256'],
        FULL_SWAP_rebuilt=FULL_SWAP, N_dec_indep_J=N_dec_indep,
    ),
    A0_replicate=A0, A0b_replicate_conf=A0b,
    A1_delta_J=A1, A2_delta_xhalf=A2, A3_tightening=A3,
    A4_concentration=A4, A5_counts=A5, A6_deep_tail=A6, A7_confirmation=A7,
    A8_grid_loo=A8, A9_steepness_alt=A9,
    predictions_check=PR, floors=floors,
    P13_verdict=P13, disc_verdict=disc_verdict,
    honesty=SEAL['honesty'],
    seal_sha8=E['seal_sha8'], exec_sha8=sha(EXEC_P)[:8],
    phase12_result_sha8=E['source']['phase12_result_sha256'][:8],
    extra=dict(mode_x=mode_x, mode_j=mode_j, coord_dep=bool(coord_dep),
               tighten_median=tight_med, n_nonfinite_J_b=int(np.sum(~np.isfinite(J_b))),
               P_few_x=P_few_x, P_acc_x=P_acc_x, P_few_j=P_few_j, P_acc_j=P_acc_j),
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('DONE ->', RESULT)
print('DONE ->', REPORT)
