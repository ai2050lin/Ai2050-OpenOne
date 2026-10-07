# -*- coding: utf-8 -*-
"""Phase 12 独立磁盘复核：从冻结 result_phase12.json / execution_phase12.json 确定性重算，
   含同 seed 逐位复现 bootstrap 带与置换零假设数组、G 族布尔、F11/F12/F13/F10。末尾 TOTAL FAILS。"""
import os, io, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase12', 'disk_verify_phase12.txt')

o = []
fails = []


def w(s=''):
    o.append(str(s)); print(s)


def sec(name, conds):
    bad = [k for k, c in conds if not c]
    w('[%-22s] %2d/%2d  %s' % (name, len(conds) - len(bad), len(conds),
                               'OK' if not bad else ('FAIL -> ' + ' | '.join(bad))))
    for k in bad:
        fails.append('%s :: %s' % (name, k))
    return not bad


def ceq(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------- 0. 载入 ----------------
R = json.load(io.open(os.path.join(T, 'result_phase12.json'), encoding='utf-8'))
E = json.load(io.open(os.path.join(T, 'execution_phase12.json'), encoding='utf-8'))
SEAL = os.path.join(T, 'N2h1a5_design_seal.json')
AM1 = os.path.join(T, 'N2h1a5_design_seal_amend1.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
SK1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SK2 = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

w('=== Phase 12 / N2h1-alpha-5 disk verify (independent recompute) ===')
w('clock %s' % __import__('time').strftime('%Y-%m-%d %H:%M:%S'))
w('')

sec('A0_files', [
    ('result exists', os.path.exists(os.path.join(T, 'result_phase12.json'))),
    ('exec exists', os.path.exists(os.path.join(T, 'execution_phase12.json'))),
    ('seal exists', os.path.exists(SEAL)),
    ('amend1 exists', os.path.exists(AM1)),
    ('report exists', os.path.exists(os.path.join(T, 'n2h1a5_report_qwen3-4b.txt'))),
    ('judgement exists', os.path.exists(os.path.join(T, 'judgement_phase12.json'))),
    ('ledger backup exists', os.path.exists(os.path.join(T, 'atlas_ledger_backup_pre_phase12.json'))),
    ('preappend baseline exists', os.path.exists(os.path.join(T, 'memo_baseline_preappend_phase12.json'))),
    ('verify_append exists', os.path.exists(os.path.join(ROOT, 'tests', 'deepseek', 'Phase12', 'verify_append_phase12.txt'))),
])

sec('A1_sha8', [
    ('result sha8 7bf4510a', sha(os.path.join(T, 'result_phase12.json'))[:8] == '7bf4510a'),
    ('exec sha8 67f38c53', sha(os.path.join(T, 'execution_phase12.json'))[:8] == '67f38c53'),
    ('seal sha8 4280b23c', sha(SEAL)[:8] == '4280b23c'),
    ('amend1 sha8 f13ea993', sha(AM1)[:8] == 'f13ea993'),
    ('result.seal_sha8', R['seal_sha8'] == '4280b23c'),
    ('result.amend1_sha8', R['amend1_sha8'] == 'f13ea993'),
    ('result.exec_sha8', R['exec_sha8'] == '67f38c53'),
    ('result.inherits_panel_sha8', R['inherits_panel_sha8'] == '4573a8bd'),
    ('result.phase11_result_sha8', R['phase11_result_sha8'] == '6fb3ef82'),
    ('exec.amend1.sha256 == file', sha(AM1) == E['amend1']['sha256']),
])

sec('A2_constants', [
    ('phase == 12', R['phase'] == 12),
    ('smoke == False', R['smoke'] is False),
    ('model qwen3-4b', R['model'] == 'qwen3-4b'),
    ('template', E['template'] == '%s是一种'),
    ('seed 20261001', E['seed'] == 20261001),
    ('primary_layer 6', E['primary_layer'] == 6),
    ('sites.profile 18', len(R['sites']['profile']) == 18),
    ('sites.swap == profile', R['sites']['swap'] == R['sites']['profile']),
    ('swap_rel_sites', E['swap_rel_sites'] == [7, 12, 20, 34]),
    ('conf_sites', E['conf_sites'] == [7, 11, 20, 34]),
    ('panel discovery 24', R['panel']['discovery'] == 24),
    ('panel confirmation 17', R['panel']['confirmation'] == 17),
    ('panel usable 41', R['panel']['usable_pairs'] == 41),
    ('B 2000', E['bootstrap']['B'] == 2000),
    ('B_perm 2000', E['bootstrap']['B_perm'] == 2000),
    ('bootstrap seed', E['bootstrap']['seed'] == 20261001),
    ('no scipy', 'scipy' not in io.open(os.path.join(ROOT, 'tests', 'deepseek', 'Phase12', 'n2h1a5_swap_alloc.py'), encoding='utf-8').read()),
])

# --- 时钟 / 元数据事件（INFO 登记，非判据；冻结件不改动）---
import time as _time
_mt = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(os.path.getmtime(os.path.join(T, 'execution_phase12.json'))))
sec('A3_clock', [
    ('exec.frozen_at 是字符串', isinstance(E['frozen_at'], str)),
    ('result.elapsed_s == 264.2', ceq(R['elapsed_s'], 264.2, 1e-9)),
    ('result 落在 2026-10-02', _time.strftime('%Y-%m-%d', _time.localtime(os.path.getmtime(os.path.join(T, 'result_phase12.json')))) == '2026-10-02'),
])
w('       [时钟] exec.frozen_at = %s ; exec mtime = %s ⇒ frozen_at 晚于 mtime 约 35 min'
  % (E['frozen_at'], _mt))
w('               （元数据笔误；exec 已被 result.exec_sha8=67f38c53 锚定，**不改动冻结件**，仅登记在案 —— 与 Phase 8 节标题时钟异常同类）')

# ---------------- 1. 复刻主脚本的确定性算子 ----------------
CL = E['classifier']
ALPHAS = [r['alpha'] for r in R['E2'][str(R['sites']['swap'][0])]]
SWAP_SITES = list(R['sites']['swap'])
W = int(E['g_family']['G2_concentration']['window'])
SEED = E['bootstrap']['seed']
BS = int(E['bootstrap']['B'])
BP = int(E['bootstrap']['B_perm'])
FULL_SWAP = float(R['FULL_SWAP'])
FS_PAIRS = R['FULL_SWAP_pairs']
FS_ORDER = list(R['E2'][str(SWAP_SITES[0])][0]['order'])


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


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


def first_reach(sites, vals, target):
    for i in range(len(vals)):
        if vals[i] >= target:
            if i == 0:
                return float(sites[0])
            v0, v1 = vals[i - 1], vals[i]
            if v1 == v0:
                return float(sites[i])
            t = (target - v0) / (v1 - v0)
            return float(sites[i - 1] + t * (sites[i] - sites[i - 1]))
    return None


def curve(y, tag=''):
    xs = np.asarray(ALPHAS, float); ys = np.asarray(y, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    s_med = float(np.median(np.delete(s, k_i)))
    J = float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')
    A = float(np.max(ys))
    kk = np.arange(CL['logistic_k_min'], CL['logistic_k_max'] + 1e-9, CL['logistic_k_step'])
    x0 = np.arange(xs.min(), xs.max() + 1e-9, CL['logistic_x0_step'])
    P = A / (1.0 + np.exp(-(kk[:, None, None] * (xs[None, None, :] - x0[None, :, None]))))
    SSE = ((P - ys[None, None, :]) ** 2).sum(axis=2)
    ij = np.unravel_index(int(np.argmin(SSE)), SSE.shape)
    return dict(J=J, x_star=float(x0[ij[1]]), k_log=float(kk[ij[0]]), A=A)


# ---------------- 2. 逐对自洽 + 曲线重算 ----------------
maxd_pp = 0.0
xh = {}; rec = {}; Jsw = {}; xs_star = {}
for s in SWAP_SITES:
    rows = R['E2'][str(s)]
    for r in rows:
        pp = np.asarray(r['per_pair'], float)
        assert len(pp) == 24, 'per_pair len %d at site %s' % (len(pp), s)
        maxd_pp = max(maxd_pp, abs(float(pp.mean()) - float(r['dDonor'])))
    y = np.array([r['dDonor'] / FULL_SWAP for r in rows], float)
    xv = cross_alpha(ALPHAS, y, 0.5)
    xh[s] = xv
    a1 = int(np.argmin(np.abs(np.asarray(ALPHAS) - 1.0)))
    rec[s] = float(y[a1])
    c = curve(y)
    Jsw[s] = c['J']; xs_star[s] = c['x_star']

sec('B_per_pair_selfcheck', [('max|mean(per_pair) - dDonor| <= 1e-12', maxd_pp <= 1e-12)])
w('       max|d| = %.3e  (F10 analogue over all 18 sites x 14 alphas)' % maxd_pp)

dxh = max(abs(xh[s] - R['xhalf']['curve'][str(s)]) for s in SWAP_SITES if xh[s] is not None)
drec = max(abs(rec[s] - R['recover']['curve'][str(s)]) for s in SWAP_SITES)
dJ = max(abs(Jsw[s] - R['profile_swap'][str(s)]['jump_ratio']) for s in SWAP_SITES)
dxs = max(abs(xs_star[s] - R['profile_swap'][str(s)]['x_star']) for s in SWAP_SITES)
sec('C_curve_recompute', [
    ('xhalf 18/18 (max|d|<=1e-12)', dxh <= 1e-12),
    ('recover 18/18 (max|d|<=1e-12)', drec <= 1e-12),
    ('J_swap 18/18 (max|d|<=1e-12)', dJ <= 1e-12),
    ('x_star logistic 18/18 (<=1e-9)', dxs <= 1e-9),
    ('XH_RANGE', ceq(max(v for v in xh.values() if v is not None) - min(v for v in xh.values() if v is not None),
                     R['xhalf']['range'], 1e-12)),
    # 口径注意：冻结的 span 定义 = recover(末位点) − recover(首位点)（端点差），不是 max−min（极差）。
    ('recover span (endpoint diff L34-L6)', ceq(rec[SWAP_SITES[-1]] - rec[SWAP_SITES[0]], R['recover']['span'], 1e-12)),
    ('min recover', ceq(min(rec.values()), R['recover']['min'], 1e-12)),
])
w('       max|d|  xhalf %.3e  recover %.3e  J %.3e  x_star %.3e' % (dxh, drec, dJ, dxs))

# ---------------- 3. G 族重算 ----------------
XH_SITES = [s for s in SWAP_SITES if xh[s] is not None]
XV = [xh[s] for s in XH_SITES]
XR = max(XV) - min(XV)
JS_OK = [s for s in SWAP_SITES if np.isfinite(Jsw[s]) and Jsw[s] > 0 and abs(R['E2'][str(s)][-1]['dDonor'] / FULL_SWAP) >= CL['UNREACH_y']]
G0 = (len(JS_OK) / len(SWAP_SITES) >= E['g_family']['G0_precondition']['curve_ok_frac_min']) and (XR >= E['g_family']['G0_precondition']['xh_range_min'])
JUMPS = [XV[i + 1] - XV[i] for i in range(len(XV) - 1)]
WINS = [abs(sum(JUMPS[i:i + W])) for i in range(len(JUMPS) - W + 1)]
TOP3 = max(WINS) / XR
MAXS = max(abs(j) for j in JUMPS) / XR
G2 = 'G2a' if TOP3 >= 0.6 else ('G2b' if MAXS <= 0.4 else 'G2_mid')
RHO_X = spearman(XV, XH_SITES)
RHO_REC = spearman([rec[s] for s in SWAP_SITES], SWAP_SITES)
P11 = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', 'result_phase11.json'), encoding='utf-8'))
JINJ = {}
for _k, _v in P11['profile_abs'].items():
    try:
        JINJ[int(_k)] = _v['jump_ratio']
    except (TypeError, ValueError):
        pass
PAIRS = [(s, Jsw[s], JINJ.get(s)) for s in SWAP_SITES if np.isfinite(Jsw[s]) and JINJ.get(s) is not None]
RHO_JG = spearman([p[1] for p in PAIRS], [p[2] for p in PAIRS])
CF = E['conf_sites']
conf_xh = [R['E5'][str(s)]['xhalf'] for s in CF]
RHO_CF = spearman(conf_xh, CF)
G4 = (RHO_CF > 0) == (RHO_X > 0) and abs(RHO_CF) >= 0.5
lo = min(XV)
XN = [(xh[s] - lo) / XR for s in XH_SITES]
X_HALF = first_reach(XH_SITES, XN, 0.5)
D10 = first_reach(XH_SITES, XN, 0.1); D90 = first_reach(XH_SITES, XN, 0.9)
SPAN_90 = (D90 - D10) if (D10 is not None and D90 is not None) else None
G1 = ('G1a_crystallized' if (X_HALF is not None and SPAN_90 is not None and X_HALF <= 20 and SPAN_90 <= 14)
      else ('G1b_distributed' if (SPAN_90 is not None and SPAN_90 >= 24) else 'G1_mid'))
MIN_REC = min(rec.values())
G5 = 'LAST_POS_STATE_SUFFICIENT' if MIN_REC >= 0.9 else 'LAST_POS_STATE_NOT_SUFFICIENT'

GF = R['G_family']
sec('D_G_family', [
    ('G0 ok', bool(G0) == GF['G0']['ok']),
    ('G0 curve_ok_frac', ceq(len(JS_OK) / len(SWAP_SITES), GF['G0']['curve_ok_frac'], 1e-12)),
    ('G0 XH_RANGE', ceq(XR, GF['G0']['xh_range'], 1e-12)),
    ('G1 label (artifact)', G1 == GF['G1']['label']),
    ('G1 x_half', ceq(X_HALF, GF['G1']['x_half'], 1e-12)),
    ('G2 top3_share_x', ceq(TOP3, GF['G2']['top3_share_x'], 1e-12)),
    ('G2 max_share_x', ceq(MAXS, GF['G2']['max_share_x'], 1e-12)),
    ('G2 label', G2 == GF['G2']['label']),
    ('G3 rho_JG', ceq(RHO_JG, GF['G3']['rho_JG'], 1e-12)),
    ('G3 label', (GF['G3']['label'] == 'G3_same_gradient') == (RHO_JG >= 0.6)),
    ('G4 rho_conf_xhalf', ceq(RHO_CF, GF['G4']['rho_conf_xhalf'], 1e-12)),
    ('G4 label', (GF['G4']['label'] == 'G4_fail') == (not G4)),
    ('G5 min_recover', ceq(MIN_REC, GF['G5']['min_recover'], 1e-12)),
    ('G5 label', G5 == GF['G5']['label']),
    ('rho(xhalf,depth)', ceq(RHO_X, R['xhalf']['rho'], 1e-12)),
    ('rho(recover,depth)', ceq(RHO_REC, R['recover']['rho'], 1e-12)),
    ('G_verdict', R['G_verdict'] == 'ALLOCATION_AMBIGUOUS'),
])
w('       XH_RANGE %.6f | TOP3 %.6f | MAXS %.6f | RHO_JG %.6f | RHO_CF %+.4f | minrec %.6f'
  % (XR, TOP3, MAXS, RHO_JG, RHO_CF, MIN_REC))

# ---------------- 4. bootstrap 同 seed 复现 ----------------
E2P = {str(s): R['E2_pairs'][str(s)] for s in SWAP_SITES}
E6P = [x['per_pair'] for x in R['E6']]
PM = np.stack([np.array(E2P[str(s)], dtype=float) for s in SWAP_SITES], 0)
PM_R = np.array(E6P, dtype=float)
xs_sw = np.asarray(ALPHAS, float)
nS, nA, nP = PM.shape
a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
FS_VEC = np.array([FS_PAIRS[rw] for rw in FS_ORDER], float)
BRNG = np.random.default_rng(SEED)


def _ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    l, h = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(l), hi=float(h), med=float(np.median(v)), n_ok=int(len(v)))


rec_b = np.full((BS, nS), np.nan); J_b = np.full((BS, nS), np.nan); XH_b = np.full((BS, nS), np.nan)
recR_b = np.full(BS, np.nan); xhR_b = np.full(BS, np.nan)
rho_b = np.full(BS, np.nan); rhox_b = np.full(BS, np.nan)
t3x_b = np.full(BS, np.nan); t3r_b = np.full(BS, np.nan)
site_arr = np.array(SWAP_SITES, float)
for b in range(BS):
    idx = BRNG.integers(0, nP, nP)
    fs_b = float(FS_VEC[idx].mean())
    if abs(fs_b) < 1e-9:
        continue
    Y = PM[:, :, idx].mean(axis=2) / fs_b
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

BB = R['bootstrap_band']
ci_x = _ci(rhox_b); ci_r = _ci(rho_b); ci_t3 = _ci(t3x_b)
sec('E_bootstrap', [
    ('B', BB['B'] == BS),
    ('n_ok', BB['n_ok'] == int(np.isfinite(rhox_b).sum())),
    ('rho_xhalf hat', ceq(BB['rho_xhalf']['hat'], RHO_X, 1e-12)),
    ('rho_xhalf lo', ceq(ci_x['lo'], BB['rho_xhalf']['lo'], 1e-9)),
    ('rho_xhalf hi', ceq(ci_x['hi'], BB['rho_xhalf']['hi'], 1e-9)),
    ('rho_recover lo', ceq(ci_r['lo'], BB['rho_recover']['lo'], 1e-9)),
    ('rho_recover hi', ceq(ci_r['hi'], BB['rho_recover']['hi'], 1e-9)),
    ('top3_share_x lo', ceq(ci_t3['lo'], BB['top3_share_x_ci']['lo'], 1e-9)),
    ('top3_share_x hi', ceq(ci_t3['hi'], BB['top3_share_x_ci']['hi'], 1e-9)),
    ('top3 band crosses 0.60', BB['top3_share_x_ci']['lo'] < 0.60 < BB['top3_share_x_ci']['hi']),
    ('R_ci.recover hat', ceq(BB['R_ci']['recover']['hat'], rec[SWAP_SITES[-1]] if False else R['recover']['R'], 1e-12)),
])
for s in SWAP_SITES:
    k = str(s)
    sec('E_xhalf_ci[L%s]' % s, [
        ('hat', ceq(BB['xhalf_ci'][k]['hat'], xh[s], 1e-12)),
        ('lo', ceq(_ci(XH_b[:, SWAP_SITES.index(s)])['lo'], BB['xhalf_ci'][k]['lo'], 1e-9)),
        ('hi', ceq(_ci(XH_b[:, SWAP_SITES.index(s)])['hi'], BB['xhalf_ci'][k]['hi'], 1e-9)),
    ])
    sec('E_J_ci[L%s]' % s, [
        ('hat', ceq(BB['J_ci'][k]['hat'], Jsw[s], 1e-12)),
        ('lo', ceq(_ci(J_b[:, SWAP_SITES.index(s)])['lo'], BB['J_ci'][k]['lo'], 1e-9)),
        ('hi', ceq(_ci(J_b[:, SWAP_SITES.index(s)])['hi'], BB['J_ci'][k]['hi'], 1e-9)),
    ])

# ---------------- 5. 置换零假设（RNG 流续接）----------------
xv_arr = np.array(XV, float)
rv_arr = np.array([rec[s] for s in SWAP_SITES], float)
xs_arr = np.array(XH_SITES, float)
rs_arr = np.array(SWAP_SITES, float)
perm_x = np.empty(BP); perm_rec = np.empty(BP)
for b in range(BP):
    perm_x[b] = spearman(BRNG.permutation(xv_arr), xs_arr)
    perm_rec[b] = spearman(BRNG.permutation(rv_arr), rs_arr)
st_x = R['permutation_null']['xhalf']['values']
dpx = max(abs(perm_x[i] - st_x[i]) for i in range(min(len(perm_x), len(st_x))))
PLO = float(np.percentile(perm_x, 2.5)); PHI = float(np.percentile(perm_x, 97.5))
sec('F_permutation', [
    ('len(values) == 2000', len(st_x) == BP),
    ('逐位复现 max|d| <= 1e-12', dpx <= 1e-12),
    ('null lo', ceq(PLO, R['permutation_null']['xhalf']['lo'], 1e-12)),
    ('null hi', ceq(PHI, R['permutation_null']['xhalf']['hi'], 1e-12)),
    ('|界| < 0.6 (F7 prime)', max(abs(PLO), abs(PHI)) < 0.6),
])
w('       perm 逐位 max|d| = %.3e ; null 带 [%+.4f, %+.4f]' % (dpx, PLO, PHI))

# ---------------- 6. floors / 比特锚 ----------------
FL = R['floors']
sec('G_floors', [
    ('F1 ok', FL['F1_ok'] is True),
    ('F1 ratio', ceq(FL['E4_max'] / FL['maxabs_all'], 0.0190, 0.0005)),
    ('F6 bit_replication', R['bit_replication'] is True),
    ('F6 E0 == phase9 ref', abs(R['full_L6'] - R['full_ref_phase9']) == 0.0),
    ('E0 string bit-equal', repr(R['full_L6']) == repr(10.574739583333335)),
    ('F11 R swap == FULL_SWAP', abs(R['E6'][-1]['dDonor'] - FULL_SWAP) <= 1e-12),
    ('F10 per-pair max|d| <= 1e-11', FL.get('F10_maxd', maxd_pp) <= 1e-11),
    ('F3 dev all zero', all(v == 0.0 for v in FL['F3_dev'].values()) if isinstance(FL['F3_dev'], dict) else FL['F3_dev'] == 0.0),
    ('FULL_SWAP pairs == 41', len(FS_PAIRS) == 41),
    ('FULL_SWAP == mean(FS over order)', ceq(FULL_SWAP, float(FS_VEC.mean()), 1e-12)),
    ('n6 anchor 17.0613', ceq(R['dose_coord']['mean_n6'], 17.06125152401808, 1e-9)),
    ('n6 no drift', R['dose_coord']['n6_drift'] is False),
])
w('       FULL_SWAP = %.12f (repr %s) ; F1 = %.4f' % (FULL_SWAP, repr(FULL_SWAP), FL['E4_max'] / FL['maxabs_all']))

# ---------------- 7. 离流形 ----------------
off = [float(a) for a in R['off_manifold_alphas']]
sec('H_off_manifold', [
    ('alpha >= 0.6 起点', off[0] == 0.6),
    ('集合 == [0.6,0.7,0.8,0.9,0.95,1.0]', off == [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
    ('pert_rel 阈值 0.50', E['off_manifold_pert_rel'] == 0.5),
])

# ---------------- 8. 文档落点 ----------------
MB = open(MEMO, 'rb').read()
MT = MB.decode('utf-8-sig')
ML = MT.splitlines()
BASEJ = json.load(io.open(BASE, encoding='utf-8'))
LED = json.load(io.open(LEDGER, encoding='utf-8'))
meas = LED.get('measurements', [])
sec('I_docs', [
    ('MEMO bom', MB[:3] == b'\xef\xbb\xbf'),
    ('MEMO bare_lf 0', MB.count(b'\n') - MB.count(b'\r\n') == 0),
    ('MEMO Phase 12 heading x1', sum(1 for l in ML if l.startswith('## Phase 12:')) == 1),
    ('MEMO Phase heading 12', sum(1 for l in ML if l.startswith('## Phase ')) == 12),
    ('MEMO sha8 e5e1f8dd', hashlib.sha256(MB).hexdigest()[:8] == 'e5e1f8dd'),
    ('MEMO bytes 273579', len(MB) == 273579),
    ('MEMO lines 2858', len(ML) == 2858),
    ('baseline tag post-append-phase12', BASEJ['tag'] == 'post-append-phase12'),
    ('baseline agrees with disk', BASEJ['sha256'] == hashlib.sha256(MB).hexdigest() and BASEJ['bytes'] == len(MB)),
    ('baseline history 2', len(BASEJ['history']) == 2),
    ('ledger n == 295', len(meas) == 295),
    ('ledger tail phase 12', str(meas[-1]).find('12') >= 0),
    ('ledger tail verdict', 'swap_allocation_ambiguous__g5_last_pos_state_sufficient' in json.dumps(meas[-1], ensure_ascii=False)),
    ('preappend baseline bytes 236092', json.load(io.open(os.path.join(T, 'memo_baseline_preappend_phase12.json'), encoding='utf-8'))['bytes'] == 236092),
    ('wlog 2026-10-02 exists', os.path.exists(WLOG)),
    ('wlog has Phase 12', 'Phase 12 / N2h1-α-5' in io.open(WLOG, encoding='utf-8').read()),
    ('MEMORY n=295', 'n=**295**' in io.open(MEM, encoding='utf-8').read()),
    ('MEMORY has (r)(s)', '(r)' in io.open(MEM, encoding='utf-8').read() and '(s)' in io.open(MEM, encoding='utf-8').read()),
    ('SK1 13 arms/44 pits', ('十三个臂' in io.open(SK1, encoding='utf-8').read()) and ('44 条' in io.open(SK1, encoding='utf-8').read())),
    ('SK2 12 lessons', '12 条收尾教训' in io.open(SK2, encoding='utf-8').read()),
])

# ---------------- 9. 判决一致性 ----------------
JUD = json.load(io.open(os.path.join(T, 'judgement_phase12.json'), encoding='utf-8'))
sec('J_judgement', [
    ('judgement G_verdict', JUD['verdict']['G_verdict'] == 'ALLOCATION_AMBIGUOUS'),
    ('judgement G2 label', JUD['verdict']['G2']['label'] == 'G2_mid'),
    ('judgement G5 label', JUD['verdict']['G5']['label'] == 'LAST_POS_STATE_SUFFICIENT'),
    ('judgement G4 label', JUD['verdict']['G4']['label'] == 'G4_fail'),
    ('judgement F7pp', JUD['verdict']['F7pp'] is True),
    ('posthoc marked POST-HOC', 'POST-HOC' in JUD['posthoc_descriptive']['note']),
    ('posthoc G1 artifact declared', '仪器伪影' in JUD['posthoc_descriptive']['G1_is_an_instrument_artifact']),
])
w('       [口径] recover: 端点差 span = %+.6f ; 极差 max-min = %+.6f ; min = %.4f' %
  (rec[SWAP_SITES[-1]] - rec[SWAP_SITES[0]], max(rec.values()) - min(rec.values()), min(rec.values())))

w('')
w('================ TOTAL: %d checks, FAILS: %d ================' % (sum(1 for l in o if l.startswith('[')), len(fails)))
if fails:
    for f in fails:
        w('  FAIL %s' % f)
else:
    w('TOTAL FAILS: 0')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
