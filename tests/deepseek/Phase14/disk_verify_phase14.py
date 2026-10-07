# -*- coding: utf-8 -*-
"""Phase 14 独立磁盘复核（N2h1-α-7）。

原则：**从非 Phase 14 的上游冻结件（Phase 12 / Phase 13 result）与 Phase 14 落盘的
原始逐对矩阵出发独立重算**，不信任 Phase 14 的中间统计量；逐分区给 PASS/FAIL 计数。

覆盖：
  A 文件与 sha（seal / amend1 / amend2 / exec / result / report / judgement）
  B FULL_SWAP 重建（Phase 12 上游）逐位
  C 点估计：从落盘 curve 独立重算 xhalf / J
  D 集中度：从独立重算的剖面重算 top3 / argmax
  E bootstrap / 置换零假设 / 配对 Δ 的**完整独立重放**
  F α=1 逐对恒等式（元素级集合相等，与子集无关）
  G 位置通道端点与判决
  H 预测 / 判决 / floors / G0p
  I 文档落点（MEMO 前缀锚 / Ledger / baseline / wlog）

注（Phase 13 教训 #13）：本脚本只硬编码**当前**基线值 —— 追加后基线未知，故对 MEMO 使用
「pre-append 前缀锚」（bytes=308863 / sha8=4f9b4574）+ 结构断言，不用追加后的 bytes。
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
T14 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
S14 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase14')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(S14, 'disk_verify_phase14.txt')

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
        w('   %-62s %s' % (lab, 'PASS' if ok else '**FAIL**'))


def ceq(a, b, tol=0.0):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ======================= 读入 =======================
R14P = os.path.join(T14, 'result_phase14.json')
SEALP = os.path.join(T14, 'N2h1a7_design_seal.json')
AM1P = os.path.join(T14, 'N2h1a7_design_seal_amend1.json')
AM2P = os.path.join(T14, 'N2h1a7_design_seal_amend2.json')
EXECP = os.path.join(T14, 'execution_phase14.json')
REPP = os.path.join(T14, 'n2h1a7_report_qwen3-4b.txt')
JUDP = os.path.join(T14, 'judgement_phase14.json')
P12P = os.path.join(T12, 'result_phase12.json')
P13P = os.path.join(T13, 'result_phase13.json')

R = json.load(io.open(R14P, encoding='utf-8'))
R12 = json.load(io.open(P12P, encoding='utf-8'))
R13 = json.load(io.open(P13P, encoding='utf-8'))
E14 = json.load(io.open(EXECP, encoding='utf-8'))
JUD = json.load(io.open(JUDP, encoding='utf-8'))
LED = json.load(io.open(LEDGER, encoding='utf-8'))
BSE = json.load(io.open(BASE, encoding='utf-8'))
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
AM2 = json.load(io.open(AM2P, encoding='utf-8'))

sec('A_files_sha', [
    ('exec.seal_sha256 == sha(seal)', E14['seal_sha256'] == sha(SEALP)),
    ('exec.amend1.sha256 == sha(amend1)', E14['amend1']['sha256'] == sha(AM1P)),
    ('exec.amend2.sha256 == sha(amend2)', E14['amend2']['sha256'] == sha(AM2P)),
    ('R.seal_sha256 == sha(seal)', R['seal_sha256'] == sha(SEALP)),
    ('R.amend1.sha256 == sha(amend1)', R['amend1']['sha256'] == sha(AM1P)),
    ('R.amend2.sha256 == sha(amend2)', R['amend2']['sha256'] == sha(AM2P)),
    ('R.seal_sha8 == exec.seal_sha8', R['seal_sha8'] == E14['seal_sha8']),
    ('judgement.meta.result_sha8 == sha(result)', JUD['meta']['result_sha8'] == sha(R14P)[:8]),
    ('judgement.meta.seal_sha8 == sha(seal)', JUD['meta']['seal_sha8'] == sha(SEALP)[:8]),
    ('judgement.meta.amend1_sha8 == sha(amend1)', JUD['meta']['amend1_sha8'] == sha(AM1P)[:8]),
    ('judgement.meta.amend2_sha8 == sha(amend2)', JUD['meta']['amend2_sha8'] == sha(AM2P)[:8]),
    ('judgement.meta.exec_sha8 == sha(exec)', JUD['meta']['exec_sha8'] == sha(EXECP)[:8]),
    ('report exists & non-empty', os.path.getsize(REPP) > 5000),
    ('script n2h1a7_prefix_swap.py exists', os.path.exists(os.path.join(S14, 'n2h1a7_prefix_swap.py'))),
    ('amend1 kind is schema_amend', 'schema_amend' in AM1['kind'] if 'kind' in AM1 else False),
    ('amend2 adds A8 arm', 'A8_cumulative_layer_substitution' in json.dumps(AM2, ensure_ascii=False)),
])

# ======================= B FULL_SWAP 重建 =======================
ORDER = list(R12['E2']['6'][0]['order'])
FS_VEC = np.array([R12['FULL_SWAP_pairs'][x] for x in ORDER], float)
FULL_SWAP = float(np.mean(FS_VEC))
nP = len(ORDER)
sec('B_rebuild', [
    ('ORDER len == 24', nP == 24),
    ('FULL_SWAP 重建 == R12.FULL_SWAP (bit)', FULL_SWAP == R12['FULL_SWAP']),
    ('FULL_SWAP == R14.A0a.rebuilt (bit)', FULL_SWAP == R['A0a_full_swap']['rebuilt']),
    ('FULL_SWAP == R14.dose_coord.full_swap (bit)', FULL_SWAP == R['dose_coord']['full_swap']),
    ('R14.inherits.FULL_SWAP == FULL_SWAP (bit)', R['inherits']['FULL_SWAP'] == FULL_SWAP),
    ('mean over ALL 41 keys != FULL_SWAP (口径分离)',
     abs(float(np.mean(list(R12['FULL_SWAP_pairs'].values()))) - FULL_SWAP) > 1e-6),
    ('R12 sha == inherits 记录', sha(P12P)[:8] == R['inherits']['phase12_result_sha8']),
    ('R13 sha == inherits 记录', sha(P13P)[:8] == R['inherits']['phase13_result_sha8']),
])

# ======================= 统计量（逐字复刻） =======================
W = int(E14['stats']['window_W'])
JFL = float(E14['stats']['jdose_floor'])
XHF = float(E14['stats']['xhalf_frac'])
SEED = int(E14['bootstrap']['seed'])
BS = int(E14['bootstrap']['BS'])
BP = int(E14['bootstrap']['BP'])


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
    m = xs >= JFL
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    med = float(np.median(rest)) if len(rest) > 1 else 0.0
    return float(s[k] / med) if med > 1e-12 else float('inf')


def conc_hat(F):
    F = np.asarray(F, float)
    if not np.isfinite(F).all():
        return None, None, []
    jm = np.diff(F); rg = float(F.max() - F.min())
    if rg <= 1e-12:
        return None, None, []
    wins = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rg), k, jm.tolist()


def _ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))


def _hist(v):
    v = np.asarray(v)[np.asarray(v) >= 0]
    if len(v) == 0:
        return {}
    u, c = np.unique(v, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(u, c)}


# ======================= C 点估计独立重算 =======================
SITES = [int(x) for x in R['sites']['profile']]
xh_ind, J_ind = {}, {}
for s in SITES:
    c = R['A1_curves'][str(s)]
    xh_ind[s] = cross_alpha(c['x'], c['y'], XHF)
    J_ind[s] = J_only(c['x'], c['y'])
d_xh = max(abs(xh_ind[s] - R['A1_xhalf'][str(s)]) for s in SITES
           if xh_ind[s] is not None and R['A1_xhalf'][str(s)] is not None)
d_J = max(abs(J_ind[s] - R['A1_J'][str(s)]) for s in SITES
          if np.isfinite(J_ind[s]) and np.isfinite(R['A1_J'][str(s)]))
A8CUR = R['A8_curves']
a8_ind_x, a8_ind_J = {}, {}
for k in A8CUR:
    a8_ind_x[k] = cross_alpha(A8CUR[k]['x'], A8CUR[k]['y'], XHF)
    a8_ind_J[k] = J_only(A8CUR[k]['x'], A8CUR[k]['y'])
d8_x = max(abs(a8_ind_x[k] - R['A8_xhalf'][k]) for k in A8CUR
           if a8_ind_x[k] is not None and R['A8_xhalf'][k] is not None)
d8_J = max(abs(a8_ind_J[k] - R['A8_J'][k]) for k in A8CUR
           if np.isfinite(a8_ind_J[k]) and np.isfinite(R['A8_J'][k]))

sec('C_point_estimates', [
    ('A1 xhalf 独立重算 max|d| == 0 (%.1e)' % d_xh, d_xh == 0.0),
    ('A1 J 独立重算 max|d| == 0 (%.1e)' % d_J, d_J == 0.0),
    ('A8 xhalf 独立重算 max|d| == 0 (%.1e)' % d8_x, d8_x == 0.0),
    ('A8 J 独立重算 max|d| == 0 (%.1e)' % d8_J, d8_J == 0.0),
    ('18 位点 A1 xhalf 全可达', all(xh_ind[s] is not None for s in SITES)),
    ('A8 位点数 == 18', len(A8CUR) == 18),
])

# ======================= D 集中度点估计 =======================
C_A1 = R['A6_concentration']['A1']; C_A8 = R['A6_concentration']['A8']
tx1, kx1, jx1 = conc_hat([xh_ind[s] for s in SITES])
tj1, kj1, jj1 = conc_hat([J_ind[s] for s in SITES])
_a8i = sorted(C_A8['sites'], key=lambda z: int(z))
tx8, kx8, jx8 = conc_hat([a8_ind_x[str(i)] for i in _a8i])
tj8, kj8, jj8 = conc_hat([a8_ind_J[str(i)] for i in _a8i])

sec('D_concentration_point', [
    ('A1 top3_x 独立重算 == R14 (=%s)' % tx1, ceq(tx1, C_A1['top3_x'], 1e-12)),
    ('A1 argmax_w_x 独立重算 == R14 (=%s)' % kx1, kx1 == C_A1['argmax_w_x']),
    ('A1 top3_j 独立重算 == R14 (=%s)' % tj1, ceq(tj1, C_A1['top3_j'], 1e-12)),
    ('A1 argmax_w_j 独立重算 == R14 (=%s)' % kj1, kj1 == C_A1['argmax_w_j']),
    ('A8 top3_x 独立重算 == R14 (=%s)' % tx8, ceq(tx8, C_A8['top3_x'], 1e-12)),
    ('A8 argmax_w_x 独立重算 == R14 (=%s)' % kx8, kx8 == C_A8['argmax_w_x']),
    ('A8 top3_j 独立重算 == R14 (=%s)' % tj8, ceq(tj8, C_A8['top3_j'], 1e-12)),
    ('A8 argmax_w_j 独立重算 == R14 (=%s)' % kj8, kj8 == C_A8['argmax_w_j']),
    ('A1 窗口语义 w 覆盖 sites[w..w+3]', C_A1['win_sem_x']['b'] == SITES[C_A1['argmax_w_x'] + 3]),
    ('A8 窗口语义 w 覆盖 i[w..w+3]', C_A8['win_sem_x']['b'] == _a8i[C_A8['argmax_w_x'] + 3]),
])

# ======================= E bootstrap / null / paired 完整重放 =======================
def replay(per_lists, alpha_grid, sites, expect_boot, expect_null, expect_paired):
    PM = np.stack([np.array(p, float) for p in per_lists], 0)
    nS = PM.shape[0]
    xs = np.array(alpha_grid, float)
    BRNG = np.random.default_rng(SEED)
    xh_b = np.full((BS, nS), np.nan); J_b = np.full((BS, nS), np.nan)
    t3x_b = np.full(BS, np.nan); t3j_b = np.full(BS, np.nan)
    ax_b = np.full(BS, -1, int); aj_b = np.full(BS, -1, int)
    dJ_b = np.full((BS, nS - 1), np.nan); dX_b = np.full((BS, nS - 1), np.nan)
    for b in range(BS):
        idx = BRNG.integers(0, nP, nP)
        fs_b = float(FS_VEC[idx].mean())
        if abs(fs_b) < 1e-9:
            continue
        Yb = PM[:, :, idx].mean(axis=2) / fs_b
        for i in range(nS):
            v = cross_alpha(xs, Yb[i], XHF)
            xh_b[b, i] = v if v is not None else np.nan
            J_b[b, i] = J_only(xs, Yb[i])
        dX_b[b] = xh_b[b, :-1] - xh_b[b, 1:]
        dJ_b[b] = J_b[b, :-1] - J_b[b, 1:]
        xr = xh_b[b]
        if np.all(np.isfinite(xr)):
            rr = float(xr.max() - xr.min()); jm = np.diff(xr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3x_b[b] = max(ws) / rr; ax_b[b] = int(np.argmax(ws))
        jr = J_b[b]
        if np.all(np.isfinite(jr)):
            rr = float(jr.max() - jr.min()); jm = np.diff(jr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3j_b[b] = max(ws) / rr; aj_b[b] = int(np.argmax(ws))
    fnx = t3x_b[np.isfinite(t3x_b)]; fnj = t3j_b[np.isfinite(t3j_b)]
    hx = _hist(ax_b); hj = _hist(aj_b)
    mx = int(max(hx, key=lambda k: hx[k])) if hx else -1
    mj = int(max(hj, key=lambda k: hj[k])) if hj else -1
    out = {}
    out['P_ge_060_x'] = float(np.mean(fnx >= 0.60)) if len(fnx) else None
    out['P_le_040_x'] = float(np.mean(fnx <= 0.40)) if len(fnx) else None
    out['P_ge_060_j'] = float(np.mean(fnj >= 0.60)) if len(fnj) else None
    out['P_le_040_j'] = float(np.mean(fnj <= 0.40)) if len(fnj) else None
    out['mode_x'] = mx; out['mode_j'] = mj
    out['freq_x'] = (hx[str(mx)] / max(sum(hx.values()), 1)) if hx else None
    out['freq_j'] = (hj[str(mj)] / max(sum(hj.values()), 1)) if hj else None
    out['hist_x'] = hx; out['hist_j'] = hj
    out['ci_top3_x'] = _ci(t3x_b); out['ci_top3_j'] = _ci(t3j_b)
    out['n_ok_x'] = int(len(fnx)); out['n_ok_j'] = int(len(fnj))
    NBRNG = np.random.default_rng(SEED + 13)
    t3n_x = np.full(BP, np.nan); t3n_j = np.full(BP, np.nan)
    jxv = np.array(jj1 if False else [], float)
    return out, (xh_b, J_b, dX_b, dJ_b), (t3x_b, t3j_b, ax_b, aj_b)


def replay_null(jumps_x, jumps_j, range_x, range_j):
    NBRNG = np.random.default_rng(SEED + 13)
    t3n_x = np.full(BP, np.nan); t3n_j = np.full(BP, np.nan)
    jxv = np.array(jumps_x, float); jjv = np.array(jumps_j, float)
    for b in range(BP):
        px = jxv[NBRNG.permutation(len(jxv))]
        pj = jjv[NBRNG.permutation(len(jjv))]
        wx = [abs(float(np.sum(px[j:j + W]))) for j in range(len(px) - W + 1)]
        wj = [abs(float(np.sum(pj[j:j + W]))) for j in range(len(pj) - W + 1)]
        t3n_x[b] = max(wx) / range_x
        t3n_j[b] = max(wj) / range_j
    return float(np.percentile(t3n_x, 95)), float(np.percentile(t3n_j, 95)), t3n_x, t3n_j


def replay_paired(dX_b, dJ_b, sites):
    rows = []; ndJ = 0; ndX = 0
    for i in range(len(sites) - 1):
        rj = _ci(dJ_b[:, i]); rx = _ci(dX_b[:, i])
        lj = ('DECISIVE_DOWN' if (rj['lo'] is not None and rj['lo'] > 0)
              else 'DECISIVE_UP' if (rj['hi'] is not None and rj['hi'] < 0) else 'TIE')
        lx = ('DECISIVE_DOWN' if (rx['lo'] is not None and rx['lo'] > 0)
              else 'DECISIVE_UP' if (rx['hi'] is not None and rx['hi'] < 0) else 'TIE')
        ndJ += int(lj != 'TIE'); ndX += int(lx != 'TIE')
        rows.append((sites[i], sites[i + 1], lj, lx))
    return rows, ndJ, ndX


# --- A1 ---
re1, st1, tmp1 = replay([R['A1_perpair'][str(s)] for s in SITES],
                        C_A1['alpha_grid'], SITES, C_A1['bootstrap'], C_A1['null'], C_A1['paired'])
BO1 = C_A1['bootstrap']
dP = max(abs(re1[k] - BO1[k]) for k in ('P_ge_060_x', 'P_le_040_x', 'P_ge_060_j', 'P_le_040_j'))
mode_ok = (re1['mode_x'] == BO1['mode_x'] and re1['mode_j'] == BO1['mode_j'])
freq_ok = all(ceq(re1['freq_' + c], BO1['freq_' + c], 1e-15) for c in ('x', 'j'))
hist_ok = (re1['hist_x'] == BO1['hist_x'] and re1['hist_j'] == BO1['hist_j'])
ci_ok = all(ceq(re1['ci_top3_' + c][k], BO1['ci_top3_' + c][k], 1e-15)
            for c in ('x', 'j') for k in ('lo', 'hi', 'med'))
# null 重放必须走主脚本的剖面构造路径（PM.mean(axis=2)/FULL_SWAP，见主脚本 L846-L850）。
# 若改用 A1_curves 的 y（curve_from_rows 产出），求和顺序不同 ⇒ 8/18 位点差 1 ULP
# （jumps max|d| = 3.331e-16）⇒ null 95 分位无法逐位复现。
PM1 = np.stack([np.array(R['A1_perpair'][str(s)], float) for s in SITES], 0)
Y1 = PM1.mean(axis=2) / FULL_SWAP
_xh_pm1 = [cross_alpha(C_A1['alpha_grid'], Y1[i], XHF) for i in range(len(SITES))]
_Jv_pm1 = [J_only(C_A1['alpha_grid'], Y1[i]) for i in range(len(SITES))]
jx1_pm = np.diff(np.array(_xh_pm1, float)).tolist()
jj1_pm = np.diff(np.array(_Jv_pm1, float)).tolist()
d_jpm1 = max(abs(a - b) for a, b in zip(jx1_pm, C_A1['jumps_x']))
d_jjm1 = max(abs(a - b) for a, b in zip(jj1_pm, C_A1['jumps_j']))
nx95, nj95, _t3nx, _t3nj = replay_null(jx1_pm, jj1_pm, C_A1['range_x'], C_A1['range_j'])
nul1 = C_A1['null']
dnull = max(abs(nx95 - nul1['null_x_95']), abs(nj95 - nul1['null_j_95']))
prows, pndJ, pndX = replay_paired(st1[2], st1[3], SITES)
plab_ok = all((prows[i][2] == C_A1['paired']['rows'][i]['label_j'] and
               prows[i][3] == C_A1['paired']['rows'][i]['label_x']) for i in range(len(prows)))

# --- A8 ---
_a8_sites = _a8i
re8, st8, tmp8 = replay([R['A8_perpair'][str(i)] for i in _a8_sites],
                        C_A8['alpha_grid'], _a8_sites, C_A8['bootstrap'], C_A8['null'], C_A8['paired'])
BO8 = C_A8['bootstrap']
dP8 = max(abs(re8[k] - BO8[k]) for k in ('P_ge_060_x', 'P_le_040_x', 'P_ge_060_j', 'P_le_040_j'))
mode8_ok = (re8['mode_x'] == BO8['mode_x'] and re8['mode_j'] == BO8['mode_j'])
freq8_ok = all(ceq(re8['freq_' + c], BO8['freq_' + c], 1e-15) for c in ('x', 'j'))
hist8_ok = (re8['hist_x'] == BO8['hist_x'] and re8['hist_j'] == BO8['hist_j'])
ci8_ok = all(ceq(re8['ci_top3_' + c][k], BO8['ci_top3_' + c][k], 1e-15)
             for c in ('x', 'j') for k in ('lo', 'hi', 'med'))
# A8 的 null 同样走 PM 路径（PM.mean(axis=2)/FULL_SWAP），与主脚本逐位一致
PM8 = np.stack([np.array(R['A8_perpair'][str(i)], float) for i in _a8_sites], 0)
Y8 = PM8.mean(axis=2) / FULL_SWAP
_xh_pm8 = [cross_alpha(C_A8['alpha_grid'], Y8[i], XHF) for i in range(len(_a8_sites))]
_Jv_pm8 = [J_only(C_A8['alpha_grid'], Y8[i]) for i in range(len(_a8_sites))]
_jx8 = np.diff(np.array(_xh_pm8, float)).tolist()
_jj8 = np.diff(np.array(_Jv_pm8, float)).tolist()
d_jpm8 = max(abs(a - b) for a, b in zip(_jx8, C_A8['jumps_x']))
d_jjm8 = max(abs(a - b) for a, b in zip(_jj8, C_A8['jumps_j']))
x95_8, j95_8, _t8x, _t8j = replay_null(_jx8, _jj8, C_A8['range_x'], C_A8['range_j'])
nul8 = C_A8['null']
dnull8 = max(abs(x95_8 - nul8['null_x_95']), abs(j95_8 - nul8['null_j_95']))

sec('E_bootstrap_replay', [
    ('A1 4 个尾部概率 max|d| == 0 (%.1e)' % dP, dP == 0.0),
    ('A1 众数窗口一致', mode_ok),
    ('A1 众数频次 max|d| <= 1e-15', freq_ok),
    ('A1 窗口直方图逐键一致', hist_ok),
    ('A1 top3 CI (lo/hi/med x2 坐标) 一致', ci_ok),
    ('A1 jumps 重算（PM 路径）max|d| == 0 (x %.1e / J %.1e)' % (d_jpm1, d_jjm1),
     d_jpm1 == 0.0 and d_jjm1 == 0.0),
    ('A1 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull, dnull == 0.0),
    ('A1 配对 Δ 标签全一致', plab_ok),
    ('A1 N_dec_J 独立 == R14', pndJ == C_A1['paired']['N_dec_J']),
    ('A1 N_dec_X 独立 == R14', pndX == C_A1['paired']['N_dec_X']),
    ('A8 4 个尾部概率 max|d| == 0 (%.1e)' % dP8, dP8 == 0.0),
    ('A8 众数窗口一致', mode8_ok),
    ('A8 众数频次 max|d| <= 1e-15', freq8_ok),
    ('A8 窗口直方图逐键一致', hist8_ok),
    ('A8 top3 CI 一致', ci8_ok),
    ('A8 jumps 重算（PM 路径）max|d| == 0 (x %.1e / J %.1e)' % (d_jpm8, d_jjm8),
     d_jpm8 == 0.0 and d_jjm8 == 0.0),
    ('A8 置换 null 95 分位 max|d| == 0 (%.1e)' % dnull8, dnull8 == 0.0),
    ('A8 site 数组为 0..17', _a8_sites == list(range(18))),
])

# ======================= F α=1 逐对恒等式（元素级） =======================
FS_SET = sorted(FS_VEC.tolist())
f30 = {}
for s in SITES:
    a1_last = sorted(np.array(R['A1_perpair'][str(s)], float)[-1].tolist())
    f30['A1_L%d' % s] = (len(a1_last) == nP and
                         all(abs(a - b) <= 0 for a, b in zip(a1_last, FS_SET)))
# A8 的 alpha=1 **不是**满替换：逐对恒等式只覆盖 A1（主脚本 L685 `for s in _A1_SITES`）。
# 独立核验「口径分离」：A8 每个支撑的 alpha=1 逐对向量都**不**等于 FULL_SWAP_pairs，
# 且端点 y(i,1) 逐位等于 Phase 12 `recover(site_i)`（18/18）。
f30a8_neq = {}
a8_y1 = {}
for i in _a8_sites:
    a8_last = sorted(np.array(R['A8_perpair'][str(i)], float)[-1].tolist())
    f30a8_neq['A8_i%d' % i] = (len(a8_last) == nP and a8_last != FS_SET)
    a8_y1[i] = float(R['A8_curves'][str(i)]['y'][-1])
mean_ident = {}
for s in SITES:
    mean_ident['A1_L%d' % s] = ceq(float(np.mean(np.array(R['A1_perpair'][str(s)], float)[-1])), FULL_SWAP, 1e-12)
REC12 = {int(k): float(v) for k, v in E14['inherits']['recover_12_by_site'].items()}
d_a8rec = max(abs(a8_y1[i] - REC12[SITES[i]]) for i in _a8_sites)

sec('F_alpha1_perpair_identity', [
    ('A1 全 18 位点 alpha=1 逐对 == FULL_SWAP_pairs（元素级）', all(f30.values())),
    ('A8 全 18 支撑 alpha=1 **不**等于 FULL_SWAP_pairs（口径分离）', all(f30a8_neq.values())),
    ('A1 逐对均值 == FULL_SWAP (bit)', all(mean_ident.values())),
    ('A8 端点 y(i,1) == Phase12 recover(site_i) 18/18 (max|d| %.1e)' % d_a8rec, d_a8rec == 0.0),
    ('R14.A8_verdict.F30a_dev == 0.0', R['A8_verdict']['F30a_dev'] == 0.0),
    ('R14.floors.F30.F30a.n_pairs == 432 (18 位点 x 24 对)', R['floors']['F30']['F30a']['n_pairs'] == 432),
])

# ======================= G 位置通道 =======================
PS = R['A3_position_summary']
y0 = np.array([PS['y0'][k] for k in sorted(PS['y0'], key=int)], float)
y1 = np.array([PS['y1'][k] for k in sorted(PS['y1'], key=int)], float)
y01 = np.array([PS['y01'][k] for k in sorted(PS['y01'], key=int)], float)
rat = y0 / y1
S_ = y0 + y1 - y01
pos_verdict = ('POS_LAST_POSITION_DOMINANT' if float(np.median(rat)) <= 0.30
               else 'POS_TWO_POSITION_COMPARABLE' if float(np.median(rat)) <= 0.70
               else 'POS_FIRST_POSITION_DOMINANT')
add_verdict = 'SUB_ADDITIVE' if float(np.mean(S_ > 0)) >= 0.5 else 'SUPER_ADDITIVE'
sec('G_position', [
    ('median(y0/y1) 独立重算一致', ceq(float(np.median(rat)), PS['median_ratio'], 1e-12)),
    ('位置判决独立重算一致', pos_verdict == PS['verdict_position']),
    ('可加性判决独立重算一致', add_verdict == PS['verdict_additivity']),
    ('n(y0>y1) 一致', int(np.sum(y0 > y1)) == PS['n_y0_gt_y1']),
    ('n(y0>0) 一致', int(np.sum(y0 > 0)) == PS['n_y0_positive']),
    ('n(S>0) 一致', int(np.sum(S_ > 0)) == PS['n_S_positive']),
    ('18 位点位置端点齐全', len(y0) == 18 and len(y1) == 18 and len(y01) == 18),
    ('y01 面板级常数（range < 1e-9）', float(y01.max() - y01.min()) < 1e-9),
    ('F30b 面板级 dev == 0.0', R['floors']['F30']['F30b']['dev'] == 0.0),
    ('F29 面板级 dev == 0.0', R['floors']['F29']['dev'] == 0.0),
])

# ======================= H 预测 / 判决 / floors =======================
PC = R['predictions_check']; FL = R['floors']; V = R['verdict']; EX = R['extra']
sec('H_predictions_verdict_floors', [
    ('floors 键 == F24..F35 + G0p', set(FL) == {'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30',
                                                'F31', 'F32', 'F33', 'F35', 'G0p'}),
    ('F24 ok (FULL_SWAP bit-equal)', FL['F24']['ok'] is True and FL['F24']['rebuilt'] == FULL_SWAP),
    ('F25 ok (n6 dev <= tol)', FL['F25']['ok'] is True and FL['F25']['dev'] <= FL['F25']['tol']),
    ('F26 ok (U6 dev)', FL['F26']['ok'] is True and FL['F26']['dev'] <= 1e-9),
    ('F27 ok (T=2)', FL['F27']['ok'] is True and FL['F27']['distinct_T'] == [2]),
    ('F28 ok (alpha=0 no-op)', FL['F28']['ok'] is True and FL['F28']['dev'] == 0.0),
    ('F29 面板级 dev == 0.0', FL['F29']['dev'] == 0.0),
    ('F30a 逐对恒等式 dev == 0.0', FL['F30']['F30a']['dev'] == 0.0 and FL['F30']['F30a']['ok'] is True),
    ('F30b 面板级常数 dev == 0.0', FL['F30']['F30b']['dev'] == 0.0 and FL['F30']['F30b']['ok'] is True),
    ('F31 ok (base 无退化)', FL['F31']['ok'] is True),
    ('F32 ok (18/18 xhalf 可达)', FL['F32']['ok'] is True),
    ('G0p == True', FL['G0p']['ok'] is True and V['G0p'] is True),
    ('预测 P1..P7 全部登记', set(PC) == {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'}),
    ('P1..P7 == 预注册事实（P1/P2/P3/P7 真，P4/P5/P6 假）',
     all(PC[k]['pass_'] is True for k in ('P1', 'P2', 'P3', 'P7')) and
     all(PC[k]['pass_'] is False for k in ('P4', 'P5', 'P6'))),
    ('R14.verdict.A8 第三口径 = A8_cumulative_layer', V['primary_third_family'] == 'A8_cumulative_layer'),
    ('R14.verdict.A1 == A8_verdict.V_A1（一致）', V['V_A1'] == R['A8_verdict']['V_A1']),
    ('端点在 A1 面板级退化 -> 非 FAIL（有声明）', V['endpoint_dev'] is not None),
    ('A7 dense 网格 ratio 记录存在', R['A7_range_grid']['dense']['range'] is not None),
    ('extra.FULL_PANEL == True', EX['FULL_PANEL'] is True),
    ('smoke == False', R['smoke'] is False),
    ('三层继承锚齐备', all(k in R['inherits'] for k in ('FULL_SWAP', 'MODE_X_13', 'MODE_J_13', 'XH_RANGE_12'))),
])

# ======================= I 文档落点 =======================
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').splitlines()
hdr14 = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 14')]
allh = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
PRE_BYTES, PRE_SHA8 = 308863, '4f9b4574'
prefix = mb[:PRE_BYTES]
led = LED['measurements']
tail = led[-1]
wtxt = io.open(WLOG, encoding='utf-8').read()
sec('I_docs', [
    ('MEMO 前缀锚：前 %d B sha8 == %s' % (PRE_BYTES, PRE_SHA8),
     len(mb) > PRE_BYTES and hashlib.sha256(prefix).hexdigest()[:8] == PRE_SHA8),
    ('MEMO 前缀逐字节未变（追加非回改：字节数只增）', len(mb) > PRE_BYTES),
    ('MEMO BOM', mb[:3] == b'\xef\xbb\xbf'),
    ('MEMO bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0),
    ('Phase 14 标题唯一', len(hdr14) == 1),
    ('Phase 标题共 14 个', len(allh) == 14),
    ('Phase 14 节位于文件尾部段', hdr14 and hdr14[0] > allh[-2]),
    ('Ledger measurements == 297', len(led) == 297),
    ('Ledger tail phase == 14', tail['phase'] == 14),
    ('Ledger tail prereg_id == N2h1a7', tail['prereg_id'] == 'N2h1a7'),
    ('Ledger tail result_sha8 == sha(result)', tail['result_sha8'] == sha(R14P)[:8]),
    ('Ledger tail seal_sha8 == sha(seal)', tail['seal_sha8'] == sha(SEALP)[:8]),
    ('Ledger backup 存在', os.path.exists(os.path.join(T14, 'atlas_ledger_backup_pre_phase14.json'))),
    ('Ledger 前 296 条未被改动', len(led) >= 297 and led[295]['phase'] == 13),
    ('baseline.tag == post-append-phase14', BSE['tag'] == 'post-append-phase14'),
    ('baseline.bytes == 实盘 MEMO bytes', BSE['bytes'] == len(mb)),
    ('baseline.sha8 == 实盘 MEMO sha8', BSE['sha8'] == hashlib.sha256(mb).hexdigest()[:8]),
    ('baseline.phase_headings 14 个', len(BSE['phase_headings']) == 14),
    ('judgement.verdict 与 result 一致', JUD['verdict']['verdict_same_coordinate_A8'] == V['verdict_same_coordinate']),
    ('wlog 含 Phase 14 节', 'Phase 14 / N2h1-α-7' in wtxt),
])

# ======================= 汇总 =======================
w('')
w('=' * 74)
w('TOTAL checks = %d ; FAIL = %d' % (TOT['n'], TOT['fail']))
w('FULL_SWAP 独立重建 = %.15f (bit-equal %s)' % (FULL_SWAP, FULL_SWAP == R['A0a_full_swap']['rebuilt']))
w('点估计重算 max|d|: A1 xhalf %.1e / J %.1e ; A8 xhalf %.1e / J %.1e' % (d_xh, d_J, d8_x, d8_J))
w('bootstrap 重放 max|d|: A1 %.1e ; A8 %.1e ; null A1 %.1e / A8 %.1e' % (dP, dP8, dnull, dnull8))
w('alpha=1 逐对恒等式：A1 %d/18 ; A8 非恒等 %d/18 ; A8 端点==recover max|d| %.1e'
  % (sum(f30.values()), sum(f30a8_neq.values()), d_a8rec))
w('null 重放 jumps 自检（PM 路径）：A1 x %.1e / J %.1e ; A8 x %.1e / J %.1e'
  % (d_jpm1, d_jjm1, d_jpm8, d_jjm8))
w('判决：同坐标 %s | 跨族 %s | 位置 %s | 可加性 %s'
  % (V['verdict_same_coordinate'], V['verdict_cross_family'], V['verdict_position'], V['verdict_additivity']))
w('MEMO %d B / %d 行 / sha8 %s ; Ledger n=%d ; baseline %s'
  % (len(mb), len(mt), hashlib.sha256(mb).hexdigest()[:8], len(led), BSE['tag']))
w('=' * 74)
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT, '| FAIL =', TOT['fail'])
assert TOT['fail'] == 0, '存在 FAIL 分区'
