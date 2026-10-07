# -*- coding: utf-8 -*-
"""Phase 20 **独立磁盘复核**（独立实现，不 import 主脚本）。

原则（承 P8–P19 纪律）：
  - 重实现关键统计量：区间求和质心 / 相邻差 com_layer / **平均秩** spearman / 置换零假设 / cross_alpha / J_only；
  - 所有标签一律「**重算 → 按主脚本文义导出标签 → 与落盘比对**」，**不得写死「预期成功」**；
  - 覆盖：跨口径配对、锚逐位、零假设可复现性、Ledger / MEMO / 预告一致、前向会计。
产物：tests/deepseek_temp/Phase20/disk_verify_phase20.txt（要求 0 FAIL）
"""
import os
import io
import json
import math
import time
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

o = []
N = dict(PASS=0, FAIL=0, WARN=0)


def w(s=''):
    o.append(str(s)); print(s)


def chk(gid, name, ok, got=None, exp=None, tol=None):
    st = 'PASS' if ok else 'FAIL'
    N[st] += 1
    extra = ''
    if got is not None or exp is not None:
        extra = '  got=%s exp=%s' % (got, exp)
    if tol is not None:
        extra += ' tol=%s' % tol
    w('  [%s] %-4s %-52s%s' % (st, gid, name, extra))
    return bool(ok)


# ================================================================ 独立统计实现
def com_interval(mass_by_site, sites):
    """区间求和质心：W_j = sum_{l in [s_j, s_{j+1})} mass[l]；mid_j = (s_j+s_{j+1})/2。"""
    vals = []
    for j in range(len(sites) - 1):
        acc = 0.0
        for l in range(int(sites[j]), int(sites[j + 1])):
            acc += float(mass_by_site.get(int(l), 0.0))
        vals.append(acc)
    vals = np.array(vals, float)
    s = np.array(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, vals
    return float((vals * mid).sum() / den), vals


def com_layer_of(jumps, sites):
    j = np.array(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None
    s = np.array(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if den <= 1e-12:
        return None
    return float((a * mid).sum() / den)


def rankdata_avg(x):
    """平均秩（独立实现；与主脚本的 ordinal argsort 秩在无并列时相同）。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman_avg(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    ra = rankdata_avg(a); rb = rankdata_avg(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def spearman_ord(a, b):
    """与主脚本 `spearman()` 逐字同义：序数秩（argsort(argsort)）+ std 守卫。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def cross_alpha(xs, ys, frac):
    xs = np.array(xs, float); ys = np.array(ys, float)
    if len(xs) < 2:
        return None
    ym = float(np.max(ys))
    if not np.isfinite(ym) or abs(ym) < 1e-12:
        return None
    tgt = frac * ym
    for i in range(len(xs) - 1):
        if ys[i] < tgt <= ys[i + 1]:
            t = (tgt - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def J_only(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return float('nan')
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    smed = float(np.median(rest)) if len(rest) else 0.0
    return float(s[k] / smed) if smed > 1e-12 else float('inf')


def perm_null_com(mass_by_site, sites, seed, n_bp):
    obs, vals = com_interval(mass_by_site, sites)
    if vals is None:
        return dict(obs_com=None, n_ok=0)
    s = np.array(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    rg = np.random.default_rng(int(seed))
    out = np.full(n_bp, np.nan)
    n = len(vals)
    for b in range(n_bp):
        pp = vals[rg.permutation(n)]
        den = float(pp.sum())
        out[b] = float((pp * mid).sum() / den) if den > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if (obs is not None and obs <= p5) else
            'high' if (obs is not None and obs >= p95) else 'none')
    return dict(obs_com=obs, n_ok=int(len(fin)), com_p5=p5, com_p95=p95, com_tail=tail)


def perm_null_share(bmlp, battn, sites, nb, seed, n_bp):
    mi = {int(s): i for i, s in enumerate(sites)}
    idx = [mi[l] for l in nb if l in mi]
    m = np.array([abs(bmlp.get(int(l), 0.0)) for l in sites], float)
    a = np.array([abs(battn.get(int(l), 0.0)) for l in sites], float)
    obs = float(m[idx].sum() / (m[idx].sum() + a[idx].sum()))
    rg = np.random.default_rng(int(seed))
    out = np.full(n_bp, np.nan)
    n = len(sites)
    for b in range(n_bp):
        mp = m[rg.permutation(n)]
        d = mp[idx].sum() + a[idx].sum()
        out[b] = float(mp[idx].sum() / d) if d > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if obs <= p5 else 'high' if obs >= p95 else 'none')
    return dict(obs_share=obs, n_ok=int(len(fin)), share_p5=p5, share_p95=p95, share_tail=tail)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


# ================================================================ 装载
w('=== Phase 20 独立磁盘复核  clock=%s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
RESP = os.path.join(P20T, 'result_phase20.json')
SEALP = os.path.join(P20T, 'N2h1a13_design_seal.json')
EXECP = os.path.join(P20T, 'execution_phase20.json')
R = load(RESP)
EX = load(EXECP)
SEAL = load(SEALP)
R18 = load(os.path.join(P18T, 'result_phase18.json'))
R16 = load(os.path.join(P16T, 'result_phase16.json'))
R17 = load(os.path.join(P17T, 'result_phase17.json'))
LG = load(LEDGER)
ARMS = list(EX['arm_order'])
V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']; FL = R['floors']
QPM = {(p['arm_nf4'] + '|' + p['arm_bf16']): p for p in R['quant_pairs']}
AKM = {'A0_nf4': 'A0_calib_qwen3-4b-nf4', 'A1_nf4': 'A1_glm4-9b-nf4'}
PID, PID2 = 'A0_nf4|A0_bf16', 'A1_nf4|A1_bf16'
BP = int(R['bootstrap']['BP'])
SEEDS = R['bootstrap']['seeds']
w('result sha8=%s | exec sha8=%s | seal sha8=%s | BP=%d' % (sha8b(open(RESP, 'rb').read()),
                                                           sha8b(open(EXECP, 'rb').read()),
                                                           sha8b(open(SEALP, 'rb').read()), BP))
w('arms = %s ; quant_pairs = %s' % (ARMS, sorted(QPM)))

# ================================================================ G1 指纹链
w('')
w('=== G1 指纹链（seal / exec / three anchors） ===')
_SB8 = sha8b(open(SEALP, 'rb').read())
_EB8 = sha8b(open(EXECP, 'rb').read())
chk('G1a', 'seal sha8 == exec.seal_sha256[:8]', _SB8 == EX['seal_sha256'][:8], _SB8, EX['seal_sha256'][:8])
chk('G1b', 'exec sha8 == result.exec_sha256[:8]', _EB8 == R['exec_sha256'][:8], _EB8, R['exec_sha256'][:8])
chk('G1c', 'seal sha8 == result.seal_sha256[:8]', _SB8 == R['seal_sha256'][:8], _SB8, R['seal_sha256'][:8])
_p18 = open(os.path.join(P18T, 'result_phase18.json'), 'rb').read()
_p16 = open(os.path.join(P16T, 'result_phase16.json'), 'rb').read()
_p17 = open(os.path.join(P17T, 'result_phase17.json'), 'rb').read()
chk('G1d', 'P18 result sha 一致', hashlib.sha256(_p18).hexdigest() == R['anchor_result_p18_sha256'])
chk('G1e', 'P16 result sha 一致', hashlib.sha256(_p16).hexdigest() == R['anchor_result_p16_sha256'])
chk('G1f', 'P17 result sha 一致', hashlib.sha256(_p17).hexdigest() == R['anchor_result_p17_sha256'])
chk('G1g', 'seal 是 design_seal/phase20', SEAL['phase'] == 20 and SEAL['kind'] == 'design_seal')
chk('G1h', 'result 非 smoke/probe', (not R.get('smoke')) and (not R.get('probe')))

# ================================================================ G2 逐臂重算内部量
w('')
w('=== G2 逐臂：由落盘谱重算 com_B / comlayer_B / com_V / spearman ===')
for a in ARMS:
    S = R['arms'][a]['E10_summary']
    sites = [int(s) for s in S['sites_all']]
    reach = [int(s) for s in S['reach']]
    idx = {l: i for i, l in enumerate(sites)}
    ba = {l: S['b_all'][idx[l]] for l in sites}
    bm = {l: S['b_mlp'][idx[l]] for l in sites}
    bt = {l: S['b_attn'][idx[l]] for l in sites}
    bc = {l: S['b_cum'][idx[l]] for l in sites}
    b1 = {l: S['b_top1'][idx[l]] for l in sites}
    wa = {l: S['w_own_all'][idx[l]] for l in sites}
    # com_B 族
    for key, tbl in (('INC_ALL', ba), ('INC_MLP', bm), ('INC_ATTN', bt), ('INC_TOP1', b1), ('CUM_ALL', bc)):
        r_, _ = com_interval({l: abs(tbl[l]) for l in sites}, reach)
        chk('G2c', '%s com_B(%s)' % (a, key), r_ is not None and abs(r_ - S['com_B'][key]) <= 1e-9,
            None if r_ is None else round(r_, 9), round(S['com_B'][key], 9), 1e-9)
    for key, tbl in (('INC_ALL', ba), ('INC_MLP', bm), ('CUM_ALL', bc)):
        r_, _ = com_interval({l: abs(tbl[l]) for l in sites}, sites)
        chk('G2cf', '%s com_B_full(%s)' % (a, key), abs(r_ - S['com_B_full'][key]) <= 1e-9,
            round(r_, 9), round(S['com_B_full'][key], 9), 1e-9)
    # com_layer_B_*
    for key, tbl, kk in (('INC_ALL', ba, 'comlayer_B_all'), ('INC_MLP', bm, 'comlayer_B_mlp'),
                         ('INC_ATTN', bt, 'comlayer_B_attn')):
        arr = np.array([tbl[l] for l in reach], float)
        r_ = com_layer_of(np.diff(arr), reach)
        chk('G2cl', '%s %s' % (a, kk), r_ is not None and abs(r_ - S[kk]) <= 1e-9,
            None if r_ is None else round(r_, 9), round(S[kk], 9), 1e-9)
    # com_V（本臂自谱）
    r_, _ = com_interval({l: wa[l] for l in sites}, reach)
    chk('G2cv', '%s com_V_own_spectrum' % a, abs(r_ - S['com_V_own_spectrum']) <= 1e-9,
        round(r_, 9), round(S['com_V_own_spectrum'], 9), 1e-9)
    # spearman(w_own_all, |b_all|)
    _xo = [wa[l] for l in reach]; _yo = [abs(ba[l]) for l in reach]
    sp = spearman_ord(_xo, _yo); spa = spearman_avg(_xo, _yo)
    chk('G2sp', '%s spearman(w_own,b_all)（主脚本文义：序数秩）' % a,
        sp is not None and abs(sp - S['spearman_wall_ball_own']) <= 1e-9,
        None if sp is None else round(sp, 9), round(S['spearman_wall_ball_own'], 9), 1e-9)
    if sp is not None and spa is not None and abs(sp - spa) > 1e-9:
        w('  [WARN] %s b 谱存在并列 ⇒ 序数秩 %s vs 平均秩 %s' % (a, round(sp, 9), round(spa, 9)))
        N['WARN'] += 1

# ================================================================ G3 校准臂 ← 冻结锚
w('')
w('=== G3 校准臂逐位复现 P18 / P16 / P17 冻结锚 ===')
for a, ak in AKM.items():
    S = R['arms'][a]['E10_summary']
    A18 = R18['arms'][ak]['E7_summary']
    A16 = R16['verdict'][ak]
    A17 = R17['arms'][ak]['E5_com_V']
    chk('G3a', '%s com_B(all) == P18' % a, abs(S['com_B']['INC_ALL'] - A18['com_B']['INC_ALL']) <= 1e-4,
        round(S['com_B']['INC_ALL'], 9), round(A18['com_B']['INC_ALL'], 9))
    chk('G3b', '%s com_B(mlp/attn/cum) == P18' % a,
        all(abs(S['com_B'][k] - A18['com_B'][k]) <= 1e-4 for k in ('INC_MLP', 'INC_ATTN', 'CUM_ALL')))
    chk('G3c', '%s comlayer_B(all/mlp/attn) == P18' % a,
        all(abs(S[k] - A18[k]) <= 1e-4 for k in ('comlayer_B_all', 'comlayer_B_mlp', 'comlayer_B_attn')))
    chk('G3d', '%s share_mlp_beh_nb == P18' % a, abs(S['share_mlp_beh_nb'] - A18['share_mlp_beh_nb']) <= 1e-4,
        round(S['share_mlp_beh_nb'], 9), round(A18['share_mlp_beh_nb'], 9))
    chk('G3e', '%s FULL_SWAP == P16' % a,
        abs(S['full_swap'] - float(R16['arms'][ak]['E2_full_swap']['FULL_SWAP'])) <= 1e-4,
        round(S['full_swap'], 6), round(float(R16['arms'][ak]['E2_full_swap']['FULL_SWAP']), 6))
    chk('G3f', '%s com_layer(x) == P16' % a, abs(S['com_layer_x'] - A16['Q4_com_x']) <= 1e-4,
        round(S['com_layer_x'], 9), round(A16['Q4_com_x'], 9))
    chk('G3g', '%s com_layer(J) == P16' % a, abs(S['com_layer_j'] - A16['Q4_com_j']) <= 1e-4,
        round(S['com_layer_j'], 9), round(A16['Q4_com_j'], 9))
    chk('G3h', '%s com_V(本臂自谱) == P17 锚' % a, abs(S['com_V_own_spectrum'] - A17['com_V']) <= 1e-4,
        round(S['com_V_own_spectrum'], 9), round(A17['com_V'], 9))
    chk('G3i', '%s nb == P18' % a, [int(x) for x in S['nb']] == [int(x) for x in A18['nb']], S['nb'], A18['nb'])

# ================================================================ G4 跨口径配对重算
w('')
w('=== G4 跨口径配对：由两臂落盘谱独立重算 Δ 与 ρ ===')
for pk in (PID, PID2):
    p = QPM.get(pk)
    if not p:
        chk('G4x', '%s 存在' % pk, False)
        continue
    an, ab = pk.split('|')
    Sa, Sb = R['arms'][an]['E10_summary'], R['arms'][ab]['E10_summary']
    ra, rb = [int(s) for s in Sa['reach']], [int(s) for s in Sb['reach']]
    ia = {l: i for i, l in enumerate([int(s) for s in Sa['sites_all']])}
    ib = {l: i for i, l in enumerate([int(s) for s in Sb['sites_all']])}
    for key, kk in (('com_B_all', 'com_B_all'), ('com_V', 'com_V')):
        d = (V[ab][kk] - V[an][kk])
        chk('G4d', '%s Δ%s 重算' % (pk, kk), abs(d - p[kk]['delta']) <= 1e-9,
            round(d, 9), round(p[kk]['delta'], 9), 1e-9)
    # [E-comv] (a) 冻结谱 com_V 两臂按构造恒等（记录事实，防止被误读为跨精度证据）
    chk('G4df', '%s com_V（P17 冻结谱）两臂按构造恒等' % pk, abs(V[ab]['com_V'] - V[an]['com_V']) <= 1e-12,
        round(V[ab]['com_V'] - V[an]['com_V'], 12), 0.0)
    # [E-comv] (b) 自谱质心的跨口径位移才是真检验（描述性，须在容差内）
    _dvo = (Sa['com_V_own_spectrum'] - Sb['com_V_own_spectrum'])
    chk('G4dw', '%s Δcom_V(自谱) 描述性 ≤ 容差' % pk, abs(_dvo) <= FL['QUANT_TOL_COMV'],
        round(_dvo, 6), FL['QUANT_TOL_COMV'])
    for kk in ('comlayer_B_all', 'com_layer_x', 'com_layer_j'):
        d = (V[ab][kk] - V[an][kk])
        chk('G4d', '%s Δ%s 重算' % (pk, kk), abs(d - p[kk]['delta']) <= 1e-9,
            round(d, 9), round(p[kk]['delta'], 9), 1e-9)
    # 行为谱秩相关（REACH 公共位点）
    com = [l for l in ra if l in set(rb)]
    _xa = [Sa['b_all'][ia[l]] for l in com]; _yb = [Sb['b_all'][ib[l]] for l in com]
    rho = spearman_ord(_xa, _yb); rhoa = spearman_avg(_xa, _yb)
    chk('G4r', '%s ρ(b_all)（主脚本文义：序数秩）' % pk,
        rho is not None and abs(rho - p['rho_b_all']['rho']) <= 1e-9,
        None if rho is None else round(rho, 9), round(p['rho_b_all']['rho'], 9), 1e-9)
    if rho is not None and rhoa is not None and abs(rho - rhoa) > 1e-9:
        w('  [WARN] %s b_all 存在并列 ⇒ 序数秩 %s vs 平均秩 %s' % (pk, round(rho, 9), round(rhoa, 9)))
        N['WARN'] += 1
    _xa2 = [Sa['b_mlp'][ia[l]] for l in com]; _yb2 = [Sb['b_mlp'][ib[l]] for l in com]
    rho2 = spearman_ord(_xa2, _yb2)
    chk('G4r2', '%s ρ(b_mlp)（主脚本文义：序数秩）' % pk,
        rho2 is not None and abs(rho2 - p['rho_b_mlp']['rho']) <= 1e-9,
        None if rho2 is None else round(rho2, 9), round(p['rho_b_mlp']['rho'], 9), 1e-9)
    # 份额同侧 + xhalf 位移
    ss = bool((Sa['share_mlp_beh_nb'] > FL['MLP_DOM_MIN']) == (Sb['share_mlp_beh_nb'] > FL['MLP_DOM_MIN']))
    chk('G4s', '%s share 同侧重算' % pk, ss == p['share_mlp_beh_nb']['same_side'], ss,
        p['share_mlp_beh_nb']['same_side'])
    sxa = {s: Sa['xhalf'][i] for i, s in enumerate(Sa['profile_sites'])}
    sxb = {s: Sb['xhalf'][i] for i, s in enumerate(Sb['profile_sites'])}
    cm = [s for s in sxa if s in sxb and sxa[s] is not None and sxb[s] is not None]
    md = max([abs(sxa[s] - sxb[s]) for s in cm]) if cm else None
    chk('G4x', '%s max|Δxhalf| 重算' % pk, md is not None and abs(md - p['xhalf']['max_abs_dxh']) <= 1e-12,
        md, p['xhalf']['max_abs_dxh'], 1e-12)

# ================================================================ G5 com_layer 由 xhalf/J 重算
w('')
w('=== G5 com_layer(x)/(J) 由落盘 xhalf/J 独立重算 ===')
for a in ARMS:
    S = R['arms'][a]['E10_summary']
    dom = [int(s) for s in S['com_layer_domain']]
    xs = {int(s): S['xhalf'][i] for i, s in enumerate(S['profile_sites'])}
    js = {int(s): S['J'][i] for i, s in enumerate(S['profile_sites'])}
    xv = [xs[l] for l in dom]
    jv = [js[l] for l in dom]
    if all(v is not None for v in xv) and all(v is not None for v in jv):
        cx = com_layer_of(np.diff(np.array(xv, float)), dom)
        cj = com_layer_of(np.diff(np.array(jv, float)), dom)
        chk('G5x', '%s com_layer(x) 重算' % a, abs(cx - S['com_layer_x']) <= 1e-9,
            round(cx, 9), round(S['com_layer_x'], 9), 1e-9)
        chk('G5j', '%s com_layer(J) 重算' % a, abs(cj - S['com_layer_j']) <= 1e-9,
            round(cj, 9), round(S['com_layer_j'], 9), 1e-9)
    else:
        chk('G5x', '%s com_layer 域内 xhalf/J 无空值' % a, False)

# ================================================================ G6 零假设可复现
w('')
w('=== G6 置换零假设可复现（同种子同 BP 独立重跑） ===')
for a in ARMS:
    S = R['arms'][a]['E10_summary']
    sites = [int(s) for s in S['sites_all']]
    reach = [int(s) for s in S['reach']]
    idx = {l: i for i, l in enumerate(sites)}
    ba = {l: abs(S['b_all'][idx[l]]) for l in sites}
    bm = {l: abs(S['b_mlp'][idx[l]]) for l in sites}
    bt = {l: abs(S['b_attn'][idx[l]]) for l in sites}
    r1 = perm_null_com(ba, reach, SEEDS['comB_inc'], BP)
    chk('G6a', '%s null com_B(all) obs/p5/p95 复现' % a,
        abs(r1['obs_com'] - S['null_comB_inc']['obs_com']) <= 1e-9
        and abs(r1['com_p5'] - S['null_comB_inc']['com_p5']) <= 1e-9
        and abs(r1['com_p95'] - S['null_comB_inc']['com_p95']) <= 1e-9
        and r1['com_tail'] == S['null_comB_inc']['com_tail'],
        r1['com_tail'], S['null_comB_inc']['com_tail'])
    r2 = perm_null_com(bm, reach, SEEDS['comB_mlp'], BP)
    chk('G6b', '%s null com_B(mlp) 复现' % a,
        abs(r2['obs_com'] - S['null_comB_mlp']['obs_com']) <= 1e-9
        and r2['com_tail'] == S['null_comB_mlp']['com_tail'], r2['com_tail'],
        S['null_comB_mlp']['com_tail'])
    r3 = perm_null_share({l: S['b_mlp'][idx[l]] for l in sites},
                         {l: S['b_attn'][idx[l]] for l in sites}, reach, [int(x) for x in S['nb']],
                         SEEDS['share_mlp'], BP)
    chk('G6c', '%s null share 复现' % a,
        abs(r3['obs_share'] - S['null_share']['obs_share']) <= 1e-9
        and r3['share_tail'] == S['null_share']['share_tail'], r3['share_tail'],
        S['null_share']['share_tail'])
    dom = [int(s) for s in S['com_layer_domain']]
    xs = {int(s): S['xhalf'][i] for i, s in enumerate(S['profile_sites'])}
    if S['null_comlayer_x'] is not None and all(xs.get(l) is not None for l in dom):
        r4 = perm_null_com({l: abs(xs[l]) for l in dom}, dom, SEEDS['comlayer'], BP)
        chk('G6d', '%s null com_layer(x) 复现' % a, r4['com_tail'] == S['null_comlayer_x']['com_tail'],
            r4['com_tail'], S['null_comlayer_x']['com_tail'])

# ================================================================ G7 判决标签一致性（重算 → 导出标签 → 比对）
w('')
w('=== G7 判决标签：由落盘量按 seal 语义**重算标签**再比对 ===')
lab = {}
lab['Q4_com_B_stable'] = all(abs(QPM[k]['com_B_all']['delta']) <= FL['QUANT_TOL_COMB'] for k in (PID, PID2))
lab['Q7_share_stable'] = all(QPM[k]['share_mlp_beh_nb']['same_side']
                             and abs(QPM[k]['share_mlp_beh_nb']['delta']) <= FL['QUANT_TOL_SHARE']
                             for k in (PID, PID2))
lab['Q11_profile_stable'] = all(abs(QPM[k]['com_layer_x']['delta']) <= FL['QUANT_TOL_COMLAYER']
                                and abs(QPM[k]['com_layer_j']['delta']) <= FL['QUANT_TOL_COMLAYER']
                                for k in (PID, PID2))
lab['Q12_xhalf_stable'] = all(QPM[k]['xhalf']['max_abs_dxh'] <= FL['QUANT_TOL_XHALF'] for k in (PID, PID2))
lab['Q8_mlp_dom_retained'] = all(V[a]['share_mlp_beh_nb'] > FL['MLP_DOM_MIN'] for a in ARMS)
lab['Q9_shallow_retained'] = all(QPM[k]['gap_sign_same'] for k in (PID, PID2))
lab['Q10_coupled_retained'] = all(QPM[k]['coupled_same'] for k in (PID, PID2))
for k in sorted(lab):
    chk('G7', 'JV["%s"] 与重算一致（主脚本文义）' % k, bool(JV[k]) == bool(lab[k]), lab[k], JV[k])
_strict = {'Q9': all(V[a]['gap'] >= FL['SHALLOWER_MIN'] for a in ARMS),
           'Q10': all(V[a]['spearman_wall_ball'] > 0 for a in ARMS)}
for _nm in ('Q9', 'Q10'):
    _key = 'Q9_shallow_retained' if _nm == 'Q9' else 'Q10_coupled_retained'
    if bool(_strict[_nm]) != bool(lab[_key]):
        w('  [WARN] %s seal 严格形（四臂皆满足）=%s 与主脚本文义（逐配对同侧）=%s 不一致'
          % (_nm, _strict[_nm], lab[_key]))
        N['WARN'] += 1
# 预测标签
pre_expect = {
    'P1': bool(JV['Q0_apparatus_all'] and JV['Q0_device_all'] and JV['Q1_joint'] == 'FID_ALL_PASS'),
    'P2': bool(JV['Q2_joint'] == 'ANCHOR_ALL_OK'),
    'P3': bool(lab['Q4_com_B_stable']),
    'P4': bool(all(QPM[k]['rho_b_all']['rho'] >= FL['RHO_B_MIN'] for k in (PID, PID2))),
    'P5': bool(lab['Q7_share_stable'] and lab['Q8_mlp_dom_retained']),
    'P6': bool(lab['Q9_shallow_retained']),
    'P7': bool(lab['Q10_coupled_retained']),
    'P8': bool(lab['Q11_profile_stable']),
    'P9': bool(lab['Q12_xhalf_stable']),
    'P10': None,
}
for k in sorted(pre_expect):
    chk('G7p', 'PC["%s"].pass_ 与重算一致' % k,
        (PC[k]['pass_'] is None and pre_expect[k] is None) or bool(PC[k]['pass_']) == bool(pre_expect[k]),
        PC[k]['pass_'], pre_expect[k])

# ================================================================ G8 前向会计
w('')
w('=== G8 前向会计（公式 vs 落盘） ===')
for a in ARMS:
    rec = R['arms'][a]
    S = rec['E10_summary']
    nsite = len(S['sites_all'])
    nd, nc = len(EX['discovery']), len(EX['confirmation'])
    nprof = len(S['profile_sites'])
    nalph = len(S['alphas'])
    est = (2 + 1 + len(EX['instances_all']) + len(EX['components']) * nsite * nd
           + len(EX['components_confirmation']) * nsite * nc + nprof * nalph * nd)
    chk('G8', '%s n_forwards 会计一致' % a, abs(rec['n_forwards'] - est) <= 3, rec['n_forwards'], est, 3)

# ================================================================ G9 Ledger
w('')
w('=== G9 Ledger 条目 ===')
ent = [m for m in LG['measurements'] if m.get('phase') == 20]
chk('G9a', 'Ledger 恰 1 条 phase=20', len(ent) == 1, len(ent), 1)
if ent:
    e = ent[-1]
    chk('G9b', 'Ledger result_sha8 == 落盘 result', e.get('result_sha8') == sha8b(open(RESP, 'rb').read()),
        e.get('result_sha8'), sha8b(open(RESP, 'rb').read()))
    chk('G9c', 'Ledger seal/exec sha8 一致',
        e.get('seal_sha8') == sha8b(open(SEALP, 'rb').read())
        and e.get('exec_sha8') == sha8b(open(EXECP, 'rb').read()))
    chk('G9d', 'Ledger n_rows == Σ n_forwards',
        e.get('n_rows') == sum(int(R['arms'][a]['n_forwards']) for a in ARMS),
        e.get('n_rows'), sum(int(R['arms'][a]['n_forwards']) for a in ARMS))
    chk('G9e', 'Ledger model_scope 含两模型且排除 A2',
        ('qwen3-4b' in e.get('model_scope', '')) and ('glm4-9b' in e.get('model_scope', ''))
        and ('14B' in e.get('model_scope', '')) and ('excluded' in e.get('model_scope', '')))
    # verdict 串由重算标签合成
    vk = ('%s__%s__%s__%s__%s__%s__%s' % (
        str(JV['Q1_joint']).lower(), str(JV['Q2_joint']).lower(),
        'cB' if lab['Q4_com_B_stable'] else 'ncB',
        'rho' if all(QPM[k]['rho_b_all']['rho'] >= FL['RHO_B_MIN'] for k in (PID, PID2)) else 'nrho',
        'share' if lab['Q7_share_stable'] else 'nshare',
        'cl' if lab['Q11_profile_stable'] else 'ncl',
        'xh' if lab['Q12_xhalf_stable'] else 'nxh',
    ))
    chk('G9f', 'Ledger verdict 与重算标签一致', e.get('verdict') == vk, e.get('verdict'), vk)
    # rev_note 里的关键数字必须来自 result
    rn = e.get('rev_note', '')
    for tok in [('%.4f' % QPM[PID]['com_B_all']['delta']), ('%.4f' % QPM[PID2]['com_B_all']['delta']),
                ('%.4f' % QPM[PID]['rho_b_all']['rho']), ('%.4f' % QPM[PID2]['rho_b_all']['rho'])]:
        chk('G9g', 'rev_note 含现场渲染数字 %s' % tok, tok in rn)

# ================================================================ G10 MEMO
w('')
w('=== G10 MEMO 落盘 ===')
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig')
lines = mt.split('\r\n')
BASE = load(os.path.join(P20T, 'memo_baseline_preappend_phase20.json'))
chk('G10a', 'BOM 存在', mb[:3] == b'\xef\xbb\xbf')
chk('G10b', 'bare_lf == 0', (mb.count(b'\n') - mb.count(b'\r\n')) == 0)
chk('G10c', 'Phase 20 标题唯一', len([1 for l in lines if l.startswith('## Phase 20')]) == 1)
nh = len([1 for l in lines if l.startswith('## Phase ')])
chk('G10d', 'Phase 标题数 = 基线+1', nh == int(BASE['phase_headings']) + 1, nh, int(BASE['phase_headings']) + 1)
pre = mb[:int(BASE['bytes'])]
chk('G10e', '前缀锚：sha8 与基线一致', sha8b(pre) == BASE['sha8'], sha8b(pre), BASE['sha8'])
for tok in ('N2h1-α-13', 'com_B', 'com_layer', 'xhalf', 'A0_bf16', 'A1_bf16', 'Phase 21'):
    chk('G10f', 'MEMO 含锚点 %s' % tok, tok in mt)
chk('G10g', 'MEMO 长度 > 基线', len(mb) > int(BASE['bytes']), len(mb), int(BASE['bytes']))
# ---- 完整性事件：P19 基线快照之后 MEMO 被就地规范化 ----
DR = load(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra',
                       'memo_drift_phase19_postbaseline.json'))
chk('G10h', '漂移审计：预测字节增 == 实测字节增',
    DR['predicted_delta_bytes'] == DR['observed_delta_bytes'],
    DR['predicted_delta_bytes'], DR['observed_delta_bytes'])
chk('G10i', '漂移审计：行数不变且残差为 0',
    DR['observed_delta_lines'] == 0 and DR['residual_bytes'] == 0,
    (DR['observed_delta_lines'], DR['residual_bytes']), (0, 0))
chk('G10j', '漂移审计：规范化标题数 == 10', len(DR['normalized_phases']) == 10,
    len(DR['normalized_phases']), 10)
chk('G10k', '漂移审计：判别性证据全一致',
    all(d['short_explains_key'] for d in DR['discriminating_evidence']),
    sum(1 for d in DR['discriminating_evidence'] if d['short_explains_key']),
    len(DR['discriminating_evidence']))
_ib = load(os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json'))
chk('G10l', '新基线 sections 键口径 == 完整标题行',
    _ib.get('sections_key_rule') == 'full-heading-line',
    _ib.get('sections_key_rule'), 'full-heading-line')
chk('G10m', '新基线 history 中 P19 条目标注 stale',
    any(h.get('tag') == DR['prev_baseline']['tag'] and h.get('stale') is True
        for h in _ib.get('history', [])))
_n_hdr2 = sum(1 for l in lines if l.startswith('## '))
chk('G10n', '新基线 sections 键数 == 标题行数（无碰撞）',
    len(_ib.get('sections', {})) == _n_hdr2, len(_ib.get('sections', {})), _n_hdr2)
chk('G10o', '新基线 drift_events 登记 1 条', len(_ib.get('drift_events', [])) == 1,
    len(_ib.get('drift_events', [])), 1)

# ================================================================ G11 归一化/自检
w('')
w('=== G11 其它自检 ===')
for a in ARMS:
    S = R['arms'][a]['E10_summary']
    ns = len(S['sites_all'])
    chk('G11a', '%s 谱长一致（b/w）' % a,
        all(len(S[k]) == ns for k in ('b_all', 'b_mlp', 'b_attn', 'b_top1', 'b_cum',
                                      'w_own_all', 'w_own_mlp', 'w_own_attn', 'pert_rel_inc')))
    chk('G11b', '%s 保真度门内' % a,
        R['arms'][a]['E2_fidelity']['arch_max'] <= FL['P20_FID_ARCH']
        and R['arms'][a]['E2_fidelity']['blk_max'] <= FL['P20_FID_BLK'],
        round(R['arms'][a]['E2_fidelity']['arch_max'], 6), FL['P20_FID_ARCH'])
    chk('G11c', '%s 装置自检（determinism/hook）' % a,
        R['arms'][a]['E0_selfcheck']['determinism_maxdiff'] <= 1e-6
        and R['arms'][a]['E0_selfcheck']['hook_effect_maxdiff'] > 1e-6)
    chk('G11d', '%s U 秩 == n_classes-1' % a,
        R['arms'][a]['E3_U']['rank'] == R['arms'][a]['E3_U']['n_classes'] - 1,
        R['arms'][a]['E3_U']['rank'], R['arms'][a]['E3_U']['n_classes'] - 1)
    chk('G11e', '%s reach 与冻结域一致（长度=18/21 依模型）' % a,
        len(S['reach']) in (18, 21), len(S['reach']))
for nm in ('_probe20_A0_nf4.json', '_probe20_A0_bf16.json'):
    pth = os.path.join(P20T, nm)
    ok = os.path.exists(pth)
    if ok:
        try:
            load(pth)
        except Exception:
            ok = False
    chk('G11f', '探针落盘可读 %s' % nm, ok)

# ================================================================ G12 P9 域分解（独立重算）
w('')
w('=== G12 P9 xhalf 域分解（独立重算；as-coded 判决不改） ===')
PH = load(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'))
PHD = {r['pair']: r for r in PH['pairs']}
R16 = load(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16', 'result_phase16.json'))
_XH12 = sorted(int(k) for k in R16['inheritance_used']['XH_12_by_site'].keys())
_p16max = None
for _a, _rec in R16['arms'].items():
    _e6 = _rec.get('E6_calibration') or {}
    if _e6.get('max_abs_dxh') is not None:
        _p16max = float(_e6['max_abs_dxh'])
        break
chk('G12a', 'P16 标定域 == 冻结 REACH(ell>=6), n=18',
    _XH12 == [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34], _XH12)
chk('G12b', 'posthoc 记录的 P16 标定域与实盘一致',
    sorted(int(s) for s in PH['p16_calib_domain']) == _XH12)
chk('G12c', 'posthoc 记录的 P16 标定最大值与 P16 result 一致',
    abs(PH['p16_calib_max_abs_dxh'] - _p16max) <= 1e-12,
    PH['p16_calib_max_abs_dxh'], round(_p16max, 12))
_TO = float(FL['QUANT_TOL_XHALF'])
for _p in R['quant_pairs']:
    _k = _p['arm_nf4'] + '|' + _p['arm_bf16']
    _x = _p['xhalf']
    _sv = {int(s): abs(float(a) - float(b)) for s, a, b in zip(_x['sites'], _x['nf4'], _x['bf16'])}
    _full = max(_sv.values())
    _dom = [s for s in _XH12 if s in _sv]
    _dommax = max(_sv[s] for s in _dom) if _dom else None
    _shy = [s for s in _sv if s not in _XH12]
    _shymax = max(_sv[s] for s in _shy) if _shy else None
    chk('G12d', '%s as-coded max|dxhalf| 独立重算' % _k,
        abs(_full - _x['max_abs_dxh']) <= 1e-12, round(_full, 12), round(_x['max_abs_dxh'], 12))
    chk('G12e', '%s 冻结 REACH 域 max|dxhalf| == posthoc' % _k,
        _dommax is not None and abs(_dommax - PHD[_k]['reach_domain_max_abs_dxh']) <= 1e-12,
        round(_dommax, 12) if _dommax else None, PHD[_k]['reach_domain_max_abs_dxh'])
    chk('G12f', '%s 浅端 max|dxhalf| == posthoc' % _k,
        _shymax is not None and abs(_shymax - PHD[_k]['shallow_max_abs_dxh']) <= 1e-12,
        round(_shymax, 12) if _shymax else None, PHD[_k]['shallow_max_abs_dxh'])
    chk('G12g', '%s 冻结域内 max <= 容差（两对皆 PASS）' % _k, _dommax is not None and _dommax <= _TO,
        round(_dommax, 12) if _dommax else None, _TO)
    chk('G12h', '%s as-coded 判决与 result 一致' % _k,
        (_x['max_abs_dxh'] <= _TO) == PHD[_k]['as_coded_pass'],
        _x['max_abs_dxh'] <= _TO, PHD[_k]['as_coded_pass'])
_A0K = 'A0_nf4|A0_bf16'
chk('G12i', 'A0 as-coded FAIL（冻结判决不改）', PHD[_A0K]['as_coded_pass'] is False)
chk('G12j', 'A0 超差 100%% 由浅端承担（as-coded == shallow）',
    abs(PHD[_A0K]['shallow_max_abs_dxh'] - PHD[_A0K]['as_coded_max_abs_dxh']) <= 1e-12)
chk('G12k', 'result 的 P9 判决 == FAIL（未改判）',
    R['predictions_check']['P9']['pass_'] is False,
    R['predictions_check']['P9']['pass_'])
chk('G12l', 'A0 冻结域值复现 P16 标定值（<=1e-12 相对）',
    abs(PHD[_A0K]['reach_domain_max_abs_dxh'] - _p16max) <= 1e-12,
    round(PHD[_A0K]['reach_domain_max_abs_dxh'], 15), round(_p16max, 15))

# ================================================================ 汇总
w('')
w('=== 汇总：PASS=%d FAIL=%d WARN=%d ===' % (N['PASS'], N['FAIL'], N['WARN']))
w('VERDICT = %s' % ('ALL_PASS' if N['FAIL'] == 0 else 'HAS_FAIL'))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(os.path.join(P20T, 'disk_verify_phase20.txt'), 'w', encoding='utf-8',
        newline='\n').write('\n'.join(o) + '\n')
print('DISK VERIFY DONE  PASS=%d FAIL=%d  -> disk_verify_phase20.txt' % (N['PASS'], N['FAIL']))
