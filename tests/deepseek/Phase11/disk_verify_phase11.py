# -*- coding: utf-8 -*-
"""
Phase 11 / N2h1-alpha-4 独立磁盘复核（disk verify）
=================================================
只读冻结产物，确定性重算：J(ell) 双坐标 + 自基 -> Spearman -> 配对 bootstrap 带
（同 seed 逐位复现）-> 置换零假设带 -> B 族布尔 -> V_ownbasis 分类 -> overlap
单调 -> F8/F9/F10 -> 比特锚 E0 -> 台账/MEMO/基线/wlog 锚点。

不重跑任何前向；不写任何被复核产物；输出盘 `disk_verify_phase11.txt`。
"""
import json, os, hashlib, sys
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T11 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
T10 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
S11 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase11')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

FAILS = []
LOG = []


def sec(name, conds):
    bad = [lab for lab, c in conds if not c]
    if bad:
        for lab in bad:
            FAILS.append('%s :: %s' % (name, lab))
    LOG.append('[%s] %d 项, 失败 %d %s' % (name, len(conds), len(bad), ('<- ' + '; '.join(bad)) if bad else ''))


def rd(p):
    return json.load(open(p, encoding='utf-8'))


# ---------------- 载入 ----------------
R = rd(os.path.join(T11, 'result_phase11.json'))
E = rd(os.path.join(T11, 'execution_phase11.json'))
R10 = rd(os.path.join(T10, 'result_phase10.json'))
J11 = rd(os.path.join(T11, 'judgement_phase11.json'))
CL = E['classifier']
FULL = float(R['full_L6'])
SEED = int(E['bootstrap']['seed'])
BS = int(E['bootstrap']['B'])
BP = int(E['bootstrap']['B_perm'])
PROFILE = list(E['profile_sites'])
OB_SITES = list(E['own_basis_sites'])
R_SITE = E['readout_site']
CF_SITES = list(E.get('conf_sites') or sorted(R['E5']['sites'].keys(), key=int))
F3_SITES = list(E.get('f3_sites') or R['floors']['F3_dev'].keys())
PRIMARY = int(R['layers']['primary'])

# ---------------- 原语（与主脚本逐字一致）----------------


def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a)
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
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
    s_med = float(np.median(rest)) if len(s) > 1 else 0.0
    return float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')


def J_batch(xs, Y):
    xs = np.asarray(xs, float)
    m = xs >= 0.01
    x2 = xs[m]; Y2 = np.asarray(Y, float)[:, m]
    ns = Y2.shape[0]
    if len(x2) < 3:
        return np.full(ns, np.nan)
    S = np.diff(Y2, axis=1) / np.diff(x2)
    ai = np.argmax(S, axis=1)
    out = np.empty(ns)
    for i in range(ns):
        rest = np.delete(S[i], ai[i])
        med = float(np.median(rest)) if len(S[i]) > 1 else 0.0
        out[i] = S[i, ai[i]] / med if med > 1e-12 else np.inf
    return out


def curve_stats(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    det = dict(x=xs.tolist(), y=ys.tolist(), n=len(xs))
    if len(xs2) < 3:
        det.update(jump_ratio=None, cls='UNCLASSIFIED'); return det
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s)); s_med = float(np.median(np.delete(s, k_i))) if len(s) > 1 else 0.0
    J = float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')
    det['jump_ratio'] = J
    mf = xs >= 0.10
    xf, yf = xs[mf], ys[mf]
    R2l = None
    if len(xf) >= 3:
        b1, b0 = np.polyfit(xf, yf, 1)
        R2l = float(1.0 - np.sum((yf - (b0 + b1 * xf)) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12))
    R2g, k_log = None, None
    A = float(np.max(ys))
    if A > 1e-9:
        kk = np.arange(CL['logistic_k_min'], CL['logistic_k_max'] + 1e-9, CL['logistic_k_step'])
        x0 = np.arange(xs.min(), xs.max() + 1e-9, CL['logistic_x0_step'])
        with np.errstate(over='ignore', invalid='ignore'):
            P = A / (1.0 + np.exp(-(kk[:, None, None] * (xs[None, None, :] - x0[None, :, None]))))
        SSE = ((P - ys[None, None, :]) ** 2).sum(axis=2)
        ij = np.unravel_index(int(np.argmin(SSE)), SSE.shape)
        SSt = float(np.sum((ys - ys.mean()) ** 2))
        R2g = float(1.0 - float(SSE[ij]) / max(SSt, 1e-12)); k_log = float(kk[ij[0]])
    ysat = float(np.max(np.abs(ys)))
    if ysat < CL['UNREACH_y']:
        cls = 'UNREACH'
    elif R2g is not None and R2g >= CL['S_STRONG']['R2_log'] and J >= CL['S_STRONG']['J'] and k_log >= CL['S_STRONG']['k_log']:
        cls = 'S_STRONG'
    elif R2g is not None and R2g >= CL['S_WEAK']['R2_log'] and J >= CL['S_WEAK']['J']:
        cls = 'S_WEAK'
    elif R2g is not None and R2g >= CL['GRADUAL']['R2_log']:
        cls = 'GRADUAL'
    elif R2l is not None and R2l >= CL['LINEAR']['R2_lin'] and (R2g is None or R2g < CL['LINEAR']['R2_log_max']):
        cls = 'LINEAR'
    else:
        cls = 'UNCLASSIFIED'
    det['cls'] = cls
    det['R2_log'] = R2g; det['R2_lin'] = R2l
    return det


def boot_spearman(pair_mats, xs, sites, B, rng):
    n_sites, n_alpha, n_pairs = pair_mats.shape
    site_idx = np.array(sites, dtype=float)
    sp = np.empty(B); Jmat = np.empty((B, n_sites)); nan_ct = 0
    for b in range(B):
        idx = rng.integers(0, n_pairs, n_pairs)
        Y = pair_mats[:, :, idx].mean(axis=2) / FULL
        Jv = J_batch(xs, Y)
        Jmat[b] = Jv
        ok = np.isfinite(Jv)
        if ok.sum() < 4:
            sp[b] = np.nan; nan_ct += 1; continue
        sp[b] = spearman(Jv[ok], site_idx[ok])
    return sp, Jmat, nan_ct


def boot_stats(sp):
    sp2 = sp[np.isfinite(sp)]
    if len(sp2) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(sp2)))
    lo, hi = np.percentile(sp2, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(sp2)), n_ok=int(len(sp2)))


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# ================= A0 载入/形状 =================
sec('A0_load', [
    ('phase==11', R['phase'] == 11),
    ('model qwen3-4b', R['model'] == 'qwen3-4b'),
    ('smoke False', R['smoke'] is False),
    ('panel discovery 24', R['panel']['discovery'] == 24),
    ('panel confirmation 17', R['panel']['confirmation'] == 17),
    ('profile 18 sites', len(PROFILE) == 18 and PROFILE[0] == 6),
    ('own_basis 18 == profile', list(OB_SITES) == list(PROFILE)),
    ('E1_pairs 19 sites', len(R['E1_pairs']) == 19),
    ('E3_pairs 18 sites', len(R['E3_pairs']) == 18),
    ('E1_pairs alpha-major (7,24)', np.array(R['E1_pairs']['6'], float).shape == (7, 24)),
    ('bootstrap B/seed', BS == 2000 and BP == 2000 and SEED == 20261001),
    ('subspace G-1=5 & n_classes 6', len(R['subspace']['sing_U6']) == 5 and R['subspace']['n_classes'] == 6),
])

# ================= A1 比特锚（跨 Phase 逐位）=================
sec('A1_bit_anchors', [
    ('E0 dDonor == 10.574739583333335 (逐位)', R['E0']['dDonor'] == 10.574739583333335),
    ('E0 alpha==1', R['E0']['alpha'] == 1.0),
    ('full_L6 == E0', R['full_L6'] == R['E0']['dDonor']),
    ('full_L6 == full_ref_phase9', R['full_L6'] == R['full_ref_phase9'] == 10.574739583333335),
    ('full_L6 == Phase10 full', R['full_L6'] == R10['full_L6']),
    ('E0b dDonor ~ 0.3334635', close(R['E0b']['dDonor'], 0.3334635416666665)),
    ('mean_n6 == 17.06125152401808', close(R['dose_coord']['mean_n6'], 17.06125152401808, 1e-12)),
    ('n6 == Phase10 n6', close(R['dose_coord']['mean_n6'], R10['dose_coord']['mean_n6'], 1e-12)),
    ('r_R == 0.151321502', close(R['r_R'], 0.151321502, 1e-9)),
    ('bit_replication True', R['bit_replication'] is True),
    ('n6_drift False', R['dose_coord']['n6_drift'] is False),
])

# ================= B1 从 E1 rows 重算 abs J =================
prof_abs = {}
for s in R['E1']:
    rows = R['E1'][s]
    prof_abs[s] = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows])
c1 = []
for s in R['E1']:
    c1.append(('abs J site %s' % s, close(prof_abs[s]['jump_ratio'], R['profile_abs'][s]['jump_ratio'])))
    c1.append(('abs cls site %s' % s, prof_abs[s]['cls'] == R['profile_abs'][s]['cls']))
sec('B1_J_abs_recompute', c1)

# ================= B2 重算 rel J =================
prof_rel = {}
for s in R['E1b']:
    rows = R['E1b'][s]
    prof_rel[s] = curve_stats([r['alpha_rel'] for r in rows], [r['dDonor'] / FULL for r in rows])
c2 = []
for s in R['E1b']:
    c2.append(('rel J site %s' % s, close(prof_rel[s]['jump_ratio'], R['profile_rel'][s]['jump_ratio'])))
    c2.append(('rel cls site %s' % s, prof_rel[s]['cls'] == R['profile_rel'][s]['cls']))
sec('B2_J_rel_recompute', c2)

# ================= B3 重算自基 J =================
J_own = {}
for s in OB_SITES:
    rows = R['E3'][str(s)]['rows']
    J_own[str(s)] = J_only([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows])
c3 = [('own J site %s' % s, close(J_own[str(s)], R['E3_verdict']['J_own'][str(s)])) for s in OB_SITES]
sec('B3_J_own_recompute', c3)

# ================= C Spearman =================
Lq = [s for s in PROFILE if prof_abs[str(s)]['jump_ratio'] is not None and np.isfinite(prof_abs[str(s)]['jump_ratio'])]
Lr = [s for s in PROFILE if prof_rel[str(s)]['jump_ratio'] is not None and np.isfinite(prof_rel[str(s)]['jump_ratio'])]
Lo = [s for s in OB_SITES if np.isfinite(J_own[str(s)])]
J_abs_hat = np.array([prof_abs[str(s)]['jump_ratio'] for s in Lq])
J_rel_hat = np.array([prof_rel[str(s)]['jump_ratio'] for s in Lr])
J_own_hat = np.array([J_own[str(s)] for s in Lo])
rho_abs = spearman(J_abs_hat, np.array(Lq, float))
rho_rel = spearman(J_rel_hat, np.array(Lr, float))
rho_own = spearman(J_own_hat, np.array(Lo, float))
spread_own = float(J_own_hat.max() / max(J_own_hat.min(), 1e-9))
sec('C_spearman', [
    ('Lq == 18', len(Lq) == 18),
    ('Lr == 18', len(Lr) == 18),
    ('Lo == 18', len(Lo) == 18),
    ('rho_abs -0.8720330237358102', close(rho_abs, -0.8720330237358102)),
    ('rho_rel -0.977296181630547', close(rho_rel, -0.977296181630547)),
    ('rho_own -0.9896800825593395', close(rho_own, -0.9896800825593395)),
    ('rho_hat dict match', close(rho_abs, R['bootstrap_band']['rho_hat']['abs']) and
     close(rho_rel, R['bootstrap_band']['rho_hat']['rel']) and close(rho_own, R['bootstrap_band']['rho_hat']['own'])),
    ('E3_verdict rho_hat match', close(rho_own, R['E3_verdict']['rho_hat'])),
    ('spread_own 5.586615338374904', close(spread_own, 5.586615338374904)),
])

# ================= D 配对 bootstrap（同 seed 逐位复现）=================
BRNG = np.random.default_rng(SEED)
PM_abs = np.stack([np.array(R['E1_pairs'][str(s)], float) for s in Lq], 0)
PM_rel = np.stack([np.array(R['E1b_pairs'][str(s)], float) for s in Lr], 0)
xs_abs = np.array([r['alpha'] for r in R['E1'][str(Lq[0])]], float)
xs_rel = np.array([r['alpha_rel'] for r in R['E1b'][str(Lr[0])]], float)
sp_abs, Jm_abs, nc_a = boot_spearman(PM_abs, xs_abs, Lq, BS, BRNG)
sp_rel, Jm_rel, nc_r = boot_spearman(PM_rel, xs_rel, Lr, BS, BRNG)
ST_abs, ST_rel = boot_stats(sp_abs), boot_stats(sp_rel)
Jci = {}
for i, s in enumerate(Lq):
    col = Jm_abs[:, i]; col = col[np.isfinite(col)]
    if len(col) >= 10:
        q25, q975 = np.percentile(col, [2.5, 97.5])
        Jci[str(s)] = dict(lo=float(q25), hi=float(q975), hat=float(J_abs_hat[i]), half=float((q975 - q25) / 2.0))
    else:
        Jci[str(s)] = dict(lo=None, hi=None, hat=float(J_abs_hat[i]), half=None)
# 置换（消耗同一 RNG 流，位置必须与主脚本一致）
perm = np.empty(BP)
order_idx = np.array(Lq, float)
for b in range(BP):
    Js = BRNG.permutation(J_abs_hat)
    perm[b] = spearman(Js, order_idx)
pr = perm[np.isfinite(perm)]
PLO, PHI = float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))
# 自基（在 perm 之后，流位置必须一致）
PM_own = np.stack([np.array(R['E3_pairs'][str(s)], float) for s in Lo], 0)
xs_own = np.array([r['alpha'] for r in R['E3'][str(Lo[0])]['rows']], float)
sp_own, Jm_own, _nco = boot_spearman(PM_own, xs_own, Lo, BS, BRNG)
ST_own = boot_stats(sp_own)
spread_ci = None
sp_ci = []
for b in range(Jm_own.shape[0]):
    row = Jm_own[b][np.isfinite(Jm_own[b])]
    if len(row) >= 4:
        sp_ci.append(row.max() / max(row.min(), 1e-9))
if len(sp_ci) >= 10:
    _a, _b = np.percentile(sp_ci, [2.5, 97.5]); spread_ci = dict(lo=float(_a), hi=float(_b))
# 确认集（最后一段）
Lc = [s for s in CF_SITES if R['E5']['sites'][str(s)]['J'] is not None and np.isfinite(R['E5']['sites'][str(s)]['J'])]
PM_c = np.stack([np.array([x['per_pair'] for x in R['E5']['sites'][str(s)]['rows']], float) for s in Lc], 0)
xs_c = np.array([r['alpha'] for r in R['E5']['sites'][str(Lc[0])]['rows']], float)
sp_c, _Jc, _nc = boot_spearman(PM_c, xs_c, Lc, BS, BRNG)
ST_c = dict(hat=spearman(np.array([R['E5']['sites'][str(s)]['J'] for s in Lc]), np.array(Lc, float)), **boot_stats(sp_c))

BB = R['bootstrap_band']
sec('D_bootstrap_band', [
    ('abs n_ok 2000', ST_abs['n_ok'] == 2000),
    ('rel n_ok 2000', ST_rel['n_ok'] == 2000),
    ('own n_ok 2000', ST_own['n_ok'] == 2000),
    ('abs lo == -0.9339525283797729', close(ST_abs['lo'], BB['abs']['lo'])),
    ('abs hi == -0.7997936016511867', close(ST_abs['hi'], BB['abs']['hi'])),
    ('rel lo == -0.9876160990712074', close(ST_rel['lo'], BB['rel']['lo'])),
    ('rel hi == -0.9091847265221878', close(ST_rel['hi'], BB['rel']['hi'])),
    ('own lo == -0.9979360165118679', close(ST_own['lo'], BB['own']['lo'])),
    ('own hi == -0.9525283797729618', close(ST_own['hi'], BB['own']['hi'])),
    ('abs med', close(ST_abs['med'], BB['abs']['med'])),
    ('rel med', close(ST_rel['med'], BB['rel']['med'])),
    ('own med', close(ST_own['med'], BB['own']['med'])),
    ('spread_ci lo', close(spread_ci['lo'], R['E3_verdict']['spread_ci']['lo'])),
    ('spread_ci hi', close(spread_ci['hi'], R['E3_verdict']['spread_ci']['hi'])),
    ('conf spearman hat', close(ST_c['hat'], R['E5']['spearman_boot']['hat'])),
    ('conf spearman lo', close(ST_c['lo'], R['E5']['spearman_boot']['lo'])),
    ('conf spearman hi', close(ST_c['hi'], R['E5']['spearman_boot']['hi'])),
])
cJci = [('J_site_ci %s lo/hi' % s, close(Jci[s]['lo'], BB['J_site_ci'][s]['lo']) and close(Jci[s]['hi'], BB['J_site_ci'][s]['hi'])) for s in Jci]
sec('D2_J_site_ci', cJci)

# ================= E 置换零假设 =================
sec('E_permutation_null', [
    ('B_perm 2000', R['permutation_null']['B'] == 2000),
    ('lo == -0.46547987616099074', close(PLO, R['permutation_null']['lo'])),
    ('hi == 0.4696078431372546', close(PHI, R['permutation_null']['hi'])),
    ('|界| < 0.6', max(abs(PLO), abs(PHI)) < 0.6),
])

# ================= F 逐对落盘自洽（F10）+ E1_pairs 一致性 =================
f10max = 0.0
pr_match = True
for key, blob in (('E1', R['E1_pairs']), ('E1b', R['E1b_pairs'])):
    for s in blob:
        got = np.array(blob[s], float)
        exp = np.array([x.get('per_pair', []) for x in R[key][s]], float)
        if got.shape != exp.shape or not np.allclose(got, exp, atol=0, rtol=0):
            pr_match = False
        for row, per in zip(R[key][s], blob[s]):
            if len(per) > 0:
                f10max = max(f10max, abs(float(np.mean(per)) - row['dDonor']))
for s in R['E3_pairs']:
    for row, per in zip(R['E3'][s]['rows'], R['E3_pairs'][s]):
        if len(per) > 0:
            f10max = max(f10max, abs(float(np.mean(per)) - row['dDonor']))
sec('F_pairs_selfconsistent', [
    ('E1/E1b_pairs == rows.per_pair (逐位)', pr_match is True),
    ('F10 max(mean(per)-dDonor) == 3.5527e-15', close(f10max, R['floors']['F10_max'], 1e-15)),
    ('F10_ok True', R['floors']['F10_ok'] is True),
    ('F10 < 1e-12', f10max < 1e-12),
])

# ================= G F8 恒等：E3(L6) ≡ E1(L6) =================
f8 = 0.0
for a_, row in zip(R['E3'][str(PRIMARY)]['rows'], R['E1'][str(PRIMARY)]):
    f8 = max(f8, abs(row['dDonor'] - a_['dDonor']))
sec('G_F8_identity_L6', [
    ('F8 max|E3(L6)-E1(L6)| == 0', f8 == 0.0),
    ('floors.F8_max == 0', R['floors']['F8_max'] == 0.0),
    ('floors.F8_ok True', R['floors']['F8_ok'] is True),
])

# ================= H F9 跨 Phase 逐单元格 =================
f9 = 0.0; bad = []
for s in R10['E1']:
    if s not in R['E1']:
        bad.append(('missing', s)); continue
    for r_new, r_old in zip(R['E1'][s], R10['E1'][s]):
        if not close(r_new['alpha'], r_old['alpha'], 1e-12):
            bad.append(('alpha', s, r_new['alpha'], r_old['alpha'])); continue
        d = abs(r_new['dDonor'] - r_old['dDonor'])
        f9 = max(f9, d)
        if d > 0:
            bad.append(('dDonor', s, r_new['alpha'], d))
sec('H_F9_cross_phase', [
    ('F9 max|d| == 0 (逐单元格)', f9 == 0.0),
    ('F9 bad == []', len(bad) == 0),
    ('floors.F9_max == 0', R['floors']['F9_max'] == 0.0),
    ('floors.F9_ok True', R['floors']['F9_ok'] is True),
    ('Phase10 result sha8 anchored', R['exec_sha8'] == '4573a8bd'),
])

# ================= I V_ownbasis 分类 =================
ob_cls = {}
agree = 0
for s in OB_SITES:
    rows = R['E3'][str(s)]['rows']
    d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows])
    ob_cls[str(s)] = d['cls']
    if d['cls'] == prof_abs[str(s)]['cls']:
        agree += 1
thr = max(3, int(np.ceil(0.75 * len(OB_SITES))))
V_own = 'BASIS_ROBUST' if agree >= thr else 'BASIS_SENSITIVE'
sec('I_V_ownbasis', [
    ('ob detail match', all(ob_cls[k] == R['decisions']['V_ownbasis_detail'][k] for k in ob_cls)),
    ('agree == 6', agree == 6),
    ('V_ownbasis == BASIS_SENSITIVE', V_own == 'BASIS_SENSITIVE'),
    ('verdict.V_ownbasis 一致', R['verdict']['V_ownbasis'] == V_own),
    ('V_ownbasis_agree 6/18', R['verdict']['V_ownbasis_agree'] == '6/18'),
])

# ================= J B 族布尔与判决 =================
B1a = bool(ST_abs['hi'] is not None and ST_abs['hi'] < -0.6)
B1b = bool(ST_rel['hi'] is not None and ST_rel['hi'] < -0.6)
B1 = bool(B1a and B1b)
B2 = bool(rho_own is not None and rho_own <= -0.6 and spread_own is not None and spread_own >= 3.0)
F7p = bool(PLO is not None and PHI is not None and max(abs(PLO), abs(PHI)) < 0.6)
B4 = bool(ST_c.get('hi') is not None and ST_c['hi'] < 0.0)
n_ind = 0; pairs_ind = []
for i in range(len(Lq) - 1):
    a_ = Jci.get(str(Lq[i])); b_ = Jci.get(str(Lq[i + 1]))
    if a_ and b_ and a_['lo'] is not None and b_['lo'] is not None:
        if not (a_['hi'] < b_['lo'] or b_['hi'] < a_['lo']):
            n_ind += 1; pairs_ind.append((Lq[i], Lq[i + 1]))
B_verdict = ('GATE_POWER_INSUFFICIENT' if not F7p else
             ('Q2_ESTABLISHED_WITH_BAND' if (B1 and B2) else
              ('Q2_COORD_ROBUST_BASIS_AMBIGUOUS' if B1 else 'Q2_BAND_AMBIGUOUS')))
V = R['verdict']
sec('J_B_family', [
    ('B1a True', B1a and V['B1a'] is True),
    ('B1b True', B1b and V['B1b'] is True),
    ('B1 True', B1 and V['B1'] is True),
    ('B2 True', B2 and V['B2'] is True),
    ('F7prime True', F7p and V['F7prime'] is True),
    ('B4 True (确认集 hi<0)', B4 and V['B4'] is True),
    ('B_verdict == Q2_ESTABLISHED_WITH_BAND', B_verdict == 'Q2_ESTABLISHED_WITH_BAND' and V['B_verdict'] == B_verdict),
    ('V_abs P0_no_verdict', V['V_abs'] == 'P0_no_verdict'),
    ('Q_abs == Q_rel == Q2_accumulate', V['Q_abs'] == 'Q2_accumulate' and V['Q_rel'] == 'Q2_accumulate'),
    ('Q_agreement Q_ROBUST', V['Q_agreement'] == 'Q_ROBUST'),
    ('V_readout LINEAR', V['V_readout'] == 'LINEAR'),
    ('B3 n_indistinguishable == 16', n_ind == 16 and BB['n_indistinguishable'] == 16),
    ('B3 pairs 一致', [tuple(x) for x in pairs_ind] == [tuple(x) for x in BB['indistinguishable_pairs']]),
    ('B3 (22,24) 是唯一可分辨对', (22, 24) not in [tuple(x) for x in pairs_ind]),
    ('judgement verdict 与 result 一致', ((not isinstance(J11, dict)) or (not isinstance(J11.get('verdict'), dict)) or (J11['verdict'].get('B_verdict') == B_verdict))),
])

# ================= K overlap 单调 =================
ov = {s: float(R['E3_verdict']['overlap'][str(s)]) for s in OB_SITES}
ov_seq = [ov[s] for s in OB_SITES]
mono = all(ov_seq[i] >= ov_seq[i + 1] - 1e-9 for i in range(len(ov_seq) - 1))
sec('K_overlap', [
    ('overlap(L6) ~ 1.0', close(ov[PRIMARY], 1.0, 1e-6)),
    ('overlap(L7) ~ 0.6892', close(ov[7], 0.6892, 5e-3)),
    ('overlap(L34) ~ 0.0298', close(ov[34], 0.0298, 5e-3)),
    ('overlap 逐位点单调不增', mono),
    ('overlap 18 位点', len(ov) == 18),
])

# ================= L 地板/漂移 =================
FL = R['floors']
sec('L_floors', [
    ('maxabs_all 12.443900553385419', close(FL['maxabs_all'], 12.443900553385419)),
    ('E4_max 0.12265624999999976', close(FL['E4_max'], 0.12265624999999976)),
    ('F1_ok True 且比 <0.10', FL['F1_ok'] is True and FL['E4_max'] < 0.10 * FL['maxabs_all']),
    ('F3_ok True', FL['F3_ok'] is True),
    ('F6_ok True', FL['F6_ok'] is True),
    ('E4 三站点', len(R['E4']) == 3),
    ('off_manifold [1.25]', R['off_manifold_alphas'] == [1.25]),
    ('drift_flags 无漂移', R['drift_flags']['n6_drift'] is False and R['drift_flags']['anchor_drift'] is False),
])

# ================= M 收尾链锚点（磁盘）=================
def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


mb = rd(BASE)
led = rd(LEDGER)
memo_b = open(MEMO, 'rb').read()
memo_t = memo_b.decode('utf-8-sig')
import re
heads = [int(m.group(1)) for m in re.finditer(r'^## Phase (\d+)[：:]', memo_t, re.M)]
sec('M_closeout_anchors', [
    ('MEMO phase 11 标题唯一', len([h for h in heads if h == 11]) == 1),
    ('MEMO Phase 标题 11 个', len(heads) == 11),
    ('MEMO BOM', memo_b[:3] == b'\xef\xbb\xbf'),
    ('MEMO bare_lf == 0', memo_b.replace(b'\r\n', b'').count(b'\n') == 0),
    ('baseline bytes == MEMO bytes', mb.get('bytes') == len(memo_b)),
    ('baseline lines ~ MEMO lines', abs(int(mb.get('lines', -1)) - memo_b.count(b'\n')) <= 1),
    ('baseline sha8 == MEMO sha8', mb.get('bytes') == len(memo_b) and mb.get('sha256', '')[:8] == sha8_file(MEMO)),
    ('baseline phase_headings == 11', len(mb.get('phase_headings', [])) == 11),
    ('baseline phase_headings 为 11 个行号', isinstance(mb.get('phase_headings'), list) and mb['phase_headings'][-1] == 2305),
    ('Ledger n == 294', len(led.get('measurements', [])) == 294),
    ('Ledger 末条 phase 11', led['measurements'][-1].get('phase') == 11),
    ('Ledger 备份存在 (Phase11 目录, v2 约定)', os.path.exists(os.path.join(T11, 'atlas_ledger_backup_pre_phase11.json'))),
    ('Ledger 备份为 pre-append 293 条', len(rd(os.path.join(T11, 'atlas_ledger_backup_pre_phase11.json'))['measurements']) == 293),
    ('pre-append 基线快照存在', os.path.exists(os.path.join(T11, 'memo_baseline_preappend_phase11.json'))),
    ('MEMORY 有 n=294 锚', 'n=**294**' in open(MEMORY, encoding='utf-8').read()),
    ('MEMORY 有 Q2_ESTABLISHED_WITH_BAND', 'Q2_ESTABLISHED_WITH_BAND' in open(MEMORY, encoding='utf-8').read()),
])

# ---------------- 汇总 ----------------
LOG.append('')
LOG.append('TOTAL FAILS: %d' % len(FAILS))
for f in FAILS:
    LOG.append('  FAIL %s' % f)
LOG.append('')
LOG.append('reproduced: rho_abs=%.6f rho_rel=%.6f rho_own=%.6f spread=%.4f' % (rho_abs, rho_rel, rho_own, spread_own))
LOG.append('bands: abs[%.4f,%.4f] rel[%.4f,%.4f] own[%.4f,%.4f]' % (
    ST_abs['lo'], ST_abs['hi'], ST_rel['lo'], ST_rel['hi'], ST_own['lo'], ST_own['hi']))
LOG.append('perm null [%.4f,%.4f]  B_verdict=%s' % (PLO, PHI, B_verdict))

OUT = os.path.join(T11, 'disk_verify_phase11.txt')
open(OUT, 'w', encoding='utf-8').write('\n'.join(LOG))
print('\n'.join(LOG))
