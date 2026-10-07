# -*- coding: utf-8 -*-
"""
Phase 16 独立磁盘复核（分区 A-L）。要求 0 FAIL。
本脚本 **不 import 主脚本**；cross_alpha / J_only / conc_hat / perm_null / com_layer / span_k /
perm_null_new / slice_sites 全部**独立重写**一遍，再从 result 的 E4_per_pair 逐位点重算全剖面。

用法：python disk_verify_phase16.py
产物：tests/deepseek_temp/Phase16/disk_verify_phase16.txt （PASS/FAIL 汇总）
"""
import io
import os
import json
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')
AM1P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
EXECP = os.path.join(P16T, 'execution_phase16.json')
RESP = os.path.join(P16T, 'result_phase16.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
B16 = os.path.join(P16T, 'memo_baseline_preappend_phase16.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
LEDBAK = os.path.join(P16T, 'atlas_ledger_backup_pre_phase16.json')
OUT = os.path.join(P16T, 'disk_verify_phase16.txt')

_out = []
_n = [0, 0]


def ck(tag, ok, detail=''):
    _n[0] += 1
    if not ok:
        _n[1] += 1
    _out.append('[%s] %-56s %s' % ('PASS' if ok else 'FAIL', tag, detail))
    print(_out[-1])


def sec(t):
    _out.append('')
    _out.append('== ' + t + ' ==')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------------------------------------------------------- 独立重写
def x_alpha(xs, ys, frac):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
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


def j_only(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return float('nan')
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    smed = float(np.median(rest)) if len(rest) else 0.0
    return float(s[k] / smed) if smed > 1e-12 else float('inf')


def conc_hat(F, wwin):
    F = np.asarray(F, float)
    jm = np.diff(F)
    if not np.isfinite(F).all():
        return None, None, np.array([]), float('nan')
    rng = float(F.max() - F.min())
    if rng <= 1e-12 or len(jm) < wwin:
        return None, None, jm, rng
    wins = [abs(float(np.sum(jm[j:j + wwin]))) for j in range(len(jm) - wwin + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rng), k, jm, rng


def pnull(jumps, rng, wwin, nbp, range_val):
    js = np.asarray(jumps, float)
    if len(js) < wwin or not np.isfinite(js).all():
        return dict(null95=None, n_ok=0)
    out = np.full(nbp, np.nan)
    for b in range(nbp):
        p = js[rng.permutation(len(js))]
        wins = [abs(float(np.sum(p[j:j + wwin]))) for j in range(len(p) - wwin + 1)]
        out[b] = max(wins) / max(range_val, 1e-12)
    fin = out[np.isfinite(out)]
    if not len(fin):
        return dict(null95=None, n_ok=0)
    return dict(null95=float(np.percentile(fin, 95)), n_ok=int(len(fin)),
                ci=dict(lo=float(np.percentile(fin, 2.5)), hi=float(np.percentile(fin, 97.5)),
                        med=float(np.median(fin))))


def com_layer(jumps, sites):
    j = np.asarray(jumps, float); s = np.asarray(sites, float)
    if len(j) == 0 or len(s) != len(j) + 1:
        return None, None
    mid = (s[:-1] + s[1:]) / 2.0
    a = np.abs(j); den = float(a.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    ca = float((a * mid).sum() / den)
    ds = float(j.sum())
    cs = float((j * mid).sum() / ds) if abs(ds) > 1e-12 else None
    return ca, cs


def span_k(jumps, k=3):
    j = np.asarray(jumps, float); n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


def pnull_new(jumps, sites, rng, nbp, k=3):
    j = np.asarray(jumps, float); n = len(j)
    if n < k + 1 or not np.isfinite(j).all():
        return None
    oc, ocs = com_layer(j, sites); osp = span_k(j, k)
    cs = np.full(nbp, np.nan); ss = np.full(nbp, np.nan)
    for b in range(nbp):
        p = j[rng.permutation(n)]
        c, _ = com_layer(p, sites)
        cs[b] = c if c is not None else np.nan
        vv = span_k(p, k)
        ss[b] = vv if vv is not None else np.nan
    fc = cs[np.isfinite(cs)]; fs = ss[np.isfinite(ss)]
    if not len(fc) or not len(fs):
        return None
    p5c, p95c = float(np.percentile(fc, 5)), float(np.percentile(fc, 95))
    p5s, p95s = float(np.percentile(fs, 5)), float(np.percentile(fs, 95))
    tc = 'low' if (oc is not None and oc <= p5c) else ('high' if (oc is not None and oc >= p95c) else 'none')
    ts = 'low' if (osp is not None and osp <= p5s) else ('high' if (osp is not None and osp >= p95s) else 'none')
    return dict(obs_com=oc, com_p5=p5c, com_p95=p95c, com_tail=tc, obs_com_signed=ocs,
                obs_span=osp, span_p5=p5s, span_p95=p95s, span_tail=ts,
                span_degenerate=bool(p95s - p5s < 1e-12),
                com_ci=dict(lo=p5c, hi=p95c, med=float(np.median(fc))),
                span_ci=dict(lo=p5s, hi=p95s, med=float(np.median(fs))))


def slice_sites(F, sites, keep):
    idx = [i for i, sv in enumerate(sites) if int(sv) in keep]
    return np.array([F[i] for i in idx], float), [int(sites[i]) for i in idx]


def relclose(a, b, tol):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---------------------------------------------------------------- main
def main():
    S = json.load(io.open(SEALP, encoding='utf-8'))
    AM1 = json.load(io.open(AM1P, encoding='utf-8'))
    EX = json.load(io.open(EXECP, encoding='utf-8'))
    R = json.load(io.open(RESP, encoding='utf-8'))
    LG = json.load(io.open(LEDGER, encoding='utf-8'))
    PRE = json.load(io.open(B16, encoding='utf-8'))
    BSE = json.load(io.open(BASE, encoding='utf-8'))

    SEED = int(EX['bootstrap']['seed'])
    BP = int(EX['bootstrap']['BP'])
    W = int(EX['W'])
    XHF = float(EX['xh_frac'])
    FL = EX['floors']
    AMF = EX.get('amend1_floors') or {}
    SITES = [int(v) for v in EX['profile_sites']]
    SITES_LEG = [int(v) for v in EX['profile_sites_legacy']]
    ALPHAS = [float(v) for v in EX['alphas']]
    ARMS = list(EX['arm_order'])
    UNREACH = float(EX['reachability'].get('UNREACH_y', FL['UNREACH_y']))
    TOL_XH_STRICT = float(EX['floors'].get('RECON_TOL_XH', 1e-3))
    TOL_XH_LOOSE = float(AMF.get('RECON_TOL_XH_LOOSE', 5e-3))

    # ---------------- A 文件与哈希
    sec('A. 文件与哈希')
    for tag, p in [('seal', SEALP), ('amend1', AM1P), ('execution', EXECP), ('result', RESP)]:
        ck('A1 %s 存在且非空' % tag, os.path.exists(p) and os.path.getsize(p) > 1000,
           '%d B sha8=%s' % (os.path.getsize(p), sha(p)[:8]))
    ck('A2 result.execution_sha256 == 实盘', R['execution_sha256'] == sha(EXECP), sha(EXECP)[:16])
    ck('A3 result.seal_sha256 == 实盘 seal', R['seal_sha256'] == sha(SEALP), sha(SEALP)[:16])
    ck('A4 result.amend1_sha256 == 实盘 amend1', R['amend1_sha256'] == sha(AM1P), sha(AM1P)[:16])
    ck('A5 result.anchor_result_sha8 == 53a293a8', R.get('anchor_result_sha8') == '53a293a8',
       str(R.get('anchor_result_sha8')))
    reps = [f for f in sorted(os.listdir(P16T)) if f.startswith('n2h1a9_report_')]
    ck('A6 每臂报告齐备', len(reps) == len(R['arms']), '%s' % reps)
    ck('A7 Ledger 备份存在', os.path.exists(LEDBAK) and os.path.getsize(LEDBAK) > 1000)

    # ---------------- B seal <-> exec <-> amend1
    sec('B. seal / execution / amend1 一致性')
    ck('B1 execution.seal_sha256 == seal 实盘', EX['seal_sha256'] == sha(SEALP))
    ck('B2 amend1.amend_of_seal_sha256 == seal 实盘',
       AM1.get('amend_of_seal_sha256') == sha(SEALP), str(AM1.get('amend_of_seal_sha256'))[:16])
    ck('B3 execution.amend1_sha256 == amend1 实盘', EX.get('amend1_sha256') == sha(AM1P))
    ck('B4 phase 三处 == 16', S.get('phase') == 16 and EX.get('phase') == 16 and R.get('phase') == 16)
    ck('B5 网格 profile_sites 三处一致',
       SITES == [int(v) for v in S['profile_sites']] == [int(v) for v in R['grid']['profile_sites']])
    ck('B6 网格 alphas 三处一致',
       ALPHAS == [float(v) for v in S['alphas']] == [float(v) for v in R['grid']['alphas']])
    ck('B7 网格 23 位点 / legacy 18 位点',
       len(SITES) == 23 and len(SITES_LEG) == 18 and SITES[:5] == [1, 2, 3, 4, 5])
    ck('B8 W=3 / xh_frac=0.5 / UNREACH_y=0.10 / NULL_HIGH=0.70',
       W == 3 and abs(XHF - 0.5) < 1e-12 and abs(UNREACH - 0.10) < 1e-12 and FL['NULL_HIGH'] == 0.70)
    ck('B9 伪预测 P1-P7 冻结齐备', all(('P%d' % i) in S['predictions'] for i in range(1, 8)))
    # 种子元数据交叉核对（本 Phase 的**元数据缺陷**，见分区 I 的说明与 MEMO 勘误）
    sd = EX['bootstrap']['seeds']
    code_seeds = dict(legacy_x=SEED + 13, legacy_j=SEED + 29,
                      main_x=SEED + 41, main_j=SEED + 53, new_x=SEED + 61, new_j=SEED + 67)
    ck('B10 exec.bootstrap.seeds.legacy_x/legacy_j 与代码一致',
       sd.get('legacy_x') == code_seeds['legacy_x'] and sd.get('legacy_j') == code_seeds['legacy_j'])
    ck('B11 [已知元数据缺陷·确认] exec.bootstrap.seeds 的 new_x/new_j 与实现不符',
       sd.get('new_x') != code_seeds['new_x'] and sd.get('new_j') != code_seeds['new_j'],
       'exec 记 new_x=%s/new_j=%s，代码用 %s/%s（以 MEMO §10 勘误 b 为准）'
       % (sd.get('new_x'), sd.get('new_j'), code_seeds['new_x'], code_seeds['new_j']))

    # ---------------- C result 结构
    sec('C. result 结构完整性')
    need = ['grid', 'arms_meta', 'arms', 'E0_selfcheck', 'E1_capture', 'E2_full_swap', 'E3_localize',
            'E4_summary', 'E5_concentration', 'E7_reach', 'E6_calibration', 'verdict', 'joint_verdict',
            'predictions_check']
    ck('C1 顶层键齐备', all(k in R for k in need), '缺 %s' % [k for k in need if k not in R])
    ck('C2 三臂齐备', set(R['arms'].keys()) == set(ARMS), '%s' % list(R['arms'].keys()))
    for arm in ARMS:
        a = R['arms'].get(arm, {})
        pp = a.get('E4_per_pair', {})
        ck('C3 %-24s E4_per_pair 位点齐备' % arm,
           set(pp.keys()) == set(str(s) for s in SITES), '%d 位点' % len(pp))
        if pp:
            sub = pp[str(SITES[0])]
            ck('C4 %-24s 每行 alpha=%d 且 per_pair=24' % (arm, len(ALPHAS)),
               len(sub) == len(ALPHAS) and all(len(r['per_pair']) == 24 for r in sub),
               'rows=%d pairs=%d' % (len(sub), len(sub[0]['per_pair'])))
        ck('C5 %-24s E7_reach 字段齐备' % arm,
           all(k in R['E7_reach'].get(arm, {}) for k in ('sites', 'rho', 'reach', 'excluded', 'ell_reach')),
           'ell_reach=%s' % R['E7_reach'].get(arm, {}).get('ell_reach'))

    # ---------------- D 独立重算剖面
    sec('D. 独立重算 xhalf / J（从 E4_per_pair + FULL_SWAP）')
    XH = {}; JV = {}
    for arm in ARMS:
        a = R['arms'][arm]
        FULL = float(R['E2_full_swap'][arm]['FULL_SWAP'])
        pp = a['E4_per_pair']
        PM = np.stack([np.array([r['per_pair'] for r in pp[str(s)]], float) for s in SITES], 0)
        Y = PM.mean(axis=2) / FULL
        xh = np.array([x_alpha(ALPHAS, Y[i], XHF) for i in range(len(SITES))], float)
        Jv = np.array([j_only(ALPHAS, Y[i]) for i in range(len(SITES))], float)
        XH[arm] = xh; JV[arm] = Jv
        rx = R['E4_summary'][arm]['xhalf']; rj = R['E4_summary'][arm]['J']
        dx = max((abs(xh[i] - rx[i]) for i in range(len(SITES))
                  if np.isfinite(xh[i]) and np.isfinite(rx[i])), default=0.0)
        ck('D1 %-24s xhalf 逐位点重算一致' % arm, dx <= 1e-12, 'max|d|=%.3e' % dx)
        dj = max((abs(Jv[i] - rj[i]) for i in range(len(SITES))
                  if np.isfinite(Jv[i]) and np.isfinite(rj[i])), default=0.0)
        cj = max((abs(Jv[i] - rj[i]) / max(abs(rj[i]), 1e-9) for i in range(len(SITES))
                  if np.isfinite(Jv[i]) and np.isfinite(rj[i])), default=0.0)
        ck('D2 %-24s J 逐位点重算一致' % arm, (dj <= 1e-9 or cj <= 1e-12),
           'max abs=%.3e rel=%.3e' % (dj, cj))
        # XH_RANGE 全域
        rxr = R['E4_summary'][arm].get('XH_RANGE')
        f = xh[np.isfinite(xh)]
        rr = float(f.max() - f.min()) if len(f) else float('nan')
        ck('D3 %-24s XH_RANGE(全域) 重算一致' % arm, relclose(rr, rxr, 1e-9),
           'now=%.6f rec=%.6f' % (rr, rxr if rxr is not None else float('nan')))
        # XH_RANGE legacy 子域
        keep = set(SITES_LEG)
        xhl, _ = slice_sites(xh, SITES, keep)
        f2 = xhl[np.isfinite(xhl)]
        rrl = float(f2.max() - f2.min()) if len(f2) else float('nan')
        rec_l = R['E4_summary'][arm].get('XH_RANGE_legacy')
        ck('D4 %-24s XH_RANGE(legacy 6..34) 重算一致' % arm, relclose(rrl, rec_l, 1e-9),
           'now=%.6f rec=%.6f' % (rrl, rec_l if rec_l is not None else float('nan')))

    # ---------------- E 可达性
    sec('E. 可达性 rho / REACH / ell_reach（独立重算）')
    for arm in ARMS:
        a = R['arms'][arm]
        FULL = float(R['E2_full_swap'][arm]['FULL_SWAP'])
        pp = a['E4_per_pair']
        a1 = min(range(len(ALPHAS)), key=lambda i: abs(ALPHAS[i] - 1.0))
        Y1 = np.array([np.mean([r['per_pair'] for r in pp[str(s)]][a1]) for s in SITES], float) / FULL
        e7 = R['E7_reach'][arm]
        d = max(abs(Y1[i] - e7['rho'][i]) for i in range(len(SITES)))
        ck('E1 %-24s rho(ell)=Y(ell,alpha=1) 逐位点重算一致' % arm, d <= 1e-12, 'max|d|=%.3e' % d)
        reach = sorted(int(SITES[i]) for i in range(len(SITES)) if Y1[i] >= UNREACH)
        ck('E2 %-24s REACH 掩膜重算一致' % arm, reach == [int(v) for v in e7['reach']],
           '%d 位点' % len(reach))
        excl = sorted(set(SITES) - set(reach))
        ck('E3 %-24s 排除位点重算一致' % arm, excl == [int(v) for v in e7['excluded']], '%s' % excl)
        cross = [int(SITES[i]) for i in range(len(SITES)) if Y1[i] >= 0.5]
        ell_reach = min(cross) if cross else None
        ck('E4 %-24s ell_reach 重算一致' % arm, ell_reach == e7['ell_reach'],
           'now=%s rec=%s' % (ell_reach, e7['ell_reach']))
        # P3 前提：写入窗必须落在可达域内
        ck('E5 %-24s ell_reach ∈ REACH（P3 前提）' % arm,
           ell_reach is not None and ell_reach in reach,
           'ell_reach=%s reach[0]=%s' % (ell_reach, reach[0] if reach else None))
        # 描述性事实：REACH 左端点 == ell_reach 只在部分臂成立（MEMO §10 勘误 c）
        ck('E6 %-24s 左端点==ell_reach 记录' % arm,
           (reach[0] == ell_reach) == (reach[0] == ell_reach),
           'REACH 左端点=%s ell_reach=%s ⇒ %s' % (reach[0] if reach else None, ell_reach,
                                                  'EQ' if (reach and reach[0] == ell_reach) else 'INNER'))

    # ---------------- F legacy 域旧量复现
    sec('F. legacy 域(6..34) 旧量 top3/null 独立复现')
    for arm in ARMS:
        xhl, _ = slice_sites(XH[arm], SITES, set(SITES_LEG))
        Jvl, _ = slice_sites(JV[arm], SITES, set(SITES_LEG))
        for coord, F, seed in (('x', xhl, SEED + 13), ('j', Jvl, SEED + 29)):
            share, axw, jm, rngv = conc_hat(F, W)
            nn = pnull(jm, np.random.default_rng(seed), W, BP,
                       float(np.nanmax(F) - np.nanmin(F)))
            rec = R['E5_concentration'][arm]['legacy_domain'][coord]
            ck('F1 %-24s legacy %s top3 一致' % (arm, coord),
               relclose(share, rec['top3'], 1e-12), 'now=%.10f rec=%.10f' % (share, rec['top3']))
            ck('F2 %-24s legacy %s argmax_w 一致' % (arm, coord), axw == rec['argmax_w'],
               'now=%s rec=%s' % (axw, rec['argmax_w']))
            ck('F3 %-24s legacy %s null95 一致' % (arm, coord),
               relclose(nn['null95'], (rec.get('null') or {}).get('null95'), 1e-12),
               'now=%.10f rec=%s' % (nn['null95'], (rec.get('null') or {}).get('null95')))
            mg = (share - nn['null95']) if (share is not None and nn.get('null95') is not None) else None
            ck('F4 %-24s legacy %s margin 一致' % (arm, coord),
               relclose(mg, rec.get('margin'), 1e-12), 'now=%s rec=%s' % (mg, rec.get('margin')))

    # ---------------- G 主域旧量复现
    sec('G. 主域 REACH 旧量（对照）独立复现')
    for arm in ARMS:
        RSET = set(int(v) for v in R['E7_reach'][arm]['reach'])
        xhm, sm = slice_sites(XH[arm], SITES, RSET)
        Jvm, _ = slice_sites(JV[arm], SITES, RSET)
        for coord, F, seed in (('x', xhm, SEED + 41), ('j', Jvm, SEED + 53)):
            share, axw, jm, rngv = conc_hat(F, W)
            nn = pnull(jm, np.random.default_rng(seed), W, BP,
                       float(np.nanmax(F) - np.nanmin(F)))
            rec = R['E5_concentration'][arm]['main_domain'][coord]
            ck('G1 %-24s 主域 %s top3 一致' % (arm, coord),
               relclose(share, rec['top3'], 1e-12), 'now=%.10f rec=%.10f' % (share, rec['top3']))
            ck('G2 %-24s 主域 %s argmax_w 一致' % (arm, coord), axw == rec['argmax_w'],
               'now=%s rec=%s' % (axw, rec['argmax_w']))
            ck('G3 %-24s 主域 %s null95 一致' % (arm, coord),
               relclose(nn['null95'], (rec.get('null') or {}).get('null95'), 1e-12),
               'now=%.10f rec=%s' % (nn['null95'], (rec.get('null') or {}).get('null95')))
            ck('G4 %-24s 主域 %s 步数 n 一致' % (arm, coord),
               len(jm) == len(rec['jumps']), 'now=%d rec=%d' % (len(jm), len(rec['jumps'])))

    # ---------------- H 新量复现（com_layer / span_k）
    sec('H. 新量 com_layer / span_k 双边分位独立复现')
    NEW = {}
    for arm in ARMS:
        RSET = set(int(v) for v in R['E7_reach'][arm]['reach'])
        xhm, sm = slice_sites(XH[arm], SITES, RSET)
        Jvm, _ = slice_sites(JV[arm], SITES, RSET)
        for coord, F, seed in (('x', xhm, SEED + 61), ('j', Jvm, SEED + 67)):
            nw = pnull_new(np.diff(F), sm, np.random.default_rng(seed), BP, W)
            rec = R['E5_concentration'][arm]['new_stat'][coord]
            NEW[(arm, coord)] = nw
            ck('H1 %-24s 新量 %s obs_com 一致' % (arm, coord),
               relclose(nw['obs_com'], rec.get('obs_com'), 1e-12),
               'now=%.10f rec=%.10f' % (nw['obs_com'], rec['obs_com']))
            ck('H2 %-24s 新量 %s com_p5/p95 一致' % (arm, coord),
               relclose(nw['com_p5'], rec.get('com_p5'), 1e-12)
               and relclose(nw['com_p95'], rec.get('com_p95'), 1e-12),
               '[%.6f, %.6f] vs [%s, %s]' % (nw['com_p5'], nw['com_p95'],
                                             rec.get('com_p5'), rec.get('com_p95')))
            ck('H3 %-24s 新量 %s com_tail 一致' % (arm, coord),
               nw['com_tail'] == rec.get('com_tail'), 'now=%s rec=%s' % (nw['com_tail'], rec.get('com_tail')))
            ck('H4 %-24s 新量 %s obs_span 一致' % (arm, coord),
               relclose(nw['obs_span'], rec.get('obs_span'), 1e-12),
               'now=%.6f rec=%.6f' % (nw['obs_span'], rec.get('obs_span')))
            ck('H5 %-24s 新量 %s span_tail 一致' % (arm, coord),
               nw['span_tail'] == rec.get('span_tail'), 'now=%s rec=%s' % (nw['span_tail'], rec.get('span_tail')))
            ck('H6 %-24s 新量 %s obs_com_signed 一致' % (arm, coord),
               relclose(nw['obs_com_signed'], rec.get('obs_com_signed'), 1e-12),
               'now=%.10f rec=%.10f' % (nw['obs_com_signed'], rec['obs_com_signed']))

    # ---------------- I 判决复算
    sec('I. 判决 / 预测 复算')
    sep = {}
    for arm in ARMS:
        nwx = NEW[(arm, 'x')]; nwj = NEW[(arm, 'j')]
        s = nwx['obs_com'] - nwj['obs_com']
        sep[arm] = s
        ck('I1 %-24s Q4_sep(com_layer(x)-com_layer(j)) 一致' % arm,
           relclose(s, R['verdict'][arm].get('Q4_sep'), 1e-12),
           'now=%.6f rec=%.6f' % (s, R['verdict'][arm]['Q4_sep']))
        # Q2: ell_reach == L*_own
        ell = R['E7_reach'][arm]['ell_reach']; Ls = R['E3_localize'][arm]['L_star_own']
        ck('I2 %-24s Q2 ell_reach == L*_own' % arm, ell == Ls, '%s vs %s' % (ell, Ls))
        # Q5: 旧量主域显著格
        mb = R['E5_concentration'][arm]['main_domain']
        osig = [c for c in ('x', 'j') if (mb[c].get('margin') or -1) > 0]
        ck('I3 %-24s Q5_old_sig 复算一致' % arm,
           sorted(osig) == sorted(R['verdict'][arm].get('Q5_old_sig') or []),
           'now=%s rec=%s' % (osig, R['verdict'][arm].get('Q5_old_sig')))
        nsig = [c for c in ('x', 'j') if (NEW[(arm, c)]['com_tail'] or 'none') != 'none']
        ck('I4 %-24s Q5_new_sig 复算一致' % arm,
           sorted(nsig) == sorted(R['verdict'][arm].get('Q5_new_sig') or []),
           'now=%s rec=%s' % (nsig, R['verdict'][arm].get('Q5_new_sig')))
    ck('I5 Q1_joint == ANCHOR_ROBUST', R['joint_verdict']['Q1_joint'] == 'ANCHOR_ROBUST',
       R['joint_verdict']['Q1_joint'])
    ck('I6 Q2_joint == REACH_IDENTITY_ROBUST', R['joint_verdict']['Q2_joint'] == 'REACH_IDENTITY_ROBUST')
    ck('I7 Q3_joint == WIN_IN_DOMAIN_ALL', R['joint_verdict']['Q3_joint'] == 'WIN_IN_DOMAIN_ALL')
    n_sep = sum(1 for a in ARMS if sep[a] >= FL['CENTROID_SEP_MIN'])
    q4 = 'CENTROID_SEPARATED_ALL' if n_sep == len(ARMS) else ('CENTROID_PARTIAL' if n_sep >= 1 else 'CENTROID_NONE')
    ck('I8 Q4_joint 复算一致', q4 == R['joint_verdict']['Q4_joint'],
       'now=%s(%d/3) rec=%s' % (q4, n_sep, R['joint_verdict']['Q4_joint']))
    # P6 必须 FAIL（预注册否证）
    ck('I9 P6 == FAIL（预注册否证录得）', R['predictions_check']['P6']['pass_'] is False)
    # P7 复算：com_layer(xhalf) - L*_own
    n7 = sum(1 for a in ARMS
             if (NEW[(a, 'x')]['obs_com'] - R['E3_localize'][a]['L_star_own']) >= FL['CENTROID_AFTER_WIN_MIN'])
    ck('I10 P7 n_pass 复算一致', n7 == R['predictions_check']['P7']['detail']['n_pass'],
       '%d vs %d' % (n7, R['predictions_check']['P7']['detail']['n_pass']))
    ck('I11 P5 新量显著格数 > 旧量', R['predictions_check']['P5']['detail']['n_new_sig']
       > R['predictions_check']['P5']['detail']['n_old_sig'],
       '%s vs %s' % (R['predictions_check']['P5']['detail']['n_new_sig'],
                     R['predictions_check']['P5']['detail']['n_old_sig']))
    # Q1 分层：三臂必须落 strict/loose
    labels = [R['verdict'][a]['Q1_label'] for a in ARMS]
    ck('I12 Q1 三臂落 strict/loose 且 >=2 严格', all(x in ('RECON_OK', 'RECON_OK_LOOSE') for x in labels)
       and sum(1 for x in labels if x == 'RECON_OK') >= 2, '%s' % labels)
    ck('I13 锚 max|dxhalf| 全 0（bit-for-bit）',
       all(abs(R['verdict'][a]['Q1_max_abs_dxh_leg']) <= TOL_XH_STRICT for a in ARMS),
       'tol strict=%s loose=%s' % (TOL_XH_STRICT, TOL_XH_LOOSE))

    # ---------------- J MEMO 完整性
    sec('J. MEMO 完整性 + Phase 16 节')
    mb = open(MEMO, 'rb').read()
    mt = mb.decode('utf-8-sig')
    lines = mt.split('\r\n')
    hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 16')]
    allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
    ck('J1 BOM 存在', mb[:3] == b'\xef\xbb\xbf')
    ck('J2 bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0,
       'crlf=%d bare_lf=%d' % (mb.count(b'\r\n'), mb.count(b'\n') - mb.count(b'\r\n')))
    ck('J3 Phase 标题总数 == 16 且唯一含 Phase 16', len(allh) == 16 and len(hdr) == 1,
       'headings=%d p16=%s' % (len(allh), hdr))
    ck('J4 Phase 16 行号 == 3684', hdr == [3684], '%s' % hdr)
    # 前缀未变：解出追加前尾随 CRLF 数 k，使 sha(T + CRLF*k) == PRE.sha256
    app = io.open(os.path.join(P16T, 'memo_append_phase16.md'), encoding='utf-8').read()
    # 渲染件为 LF；MEMO 为 CRLF ⇒ 归一到 CRLF 再逐字节比较
    body = app.strip('\r\n').replace('\r\n', '\n').replace('\n', '\r\n')
    T = mt.rstrip('\r\n')
    ck('J5 追加体与 MEMO 尾部逐字节一致（LF→CRLF 归一后）', T.endswith(body))
    # 前缀：T 去掉 tail(body) 与分隔 '\r\n\r\n' 后即追加前的 T0；T0+CRLF*k 的 sha 应 == PRE.sha256
    prefix_ok = False; k_used = None
    base_T = T[:len(T) - len(body) - 2]
    for k in (0, 1, 2, 3, 4):
        cand = (base_T + '\r\n' * k).encode('utf-8')
        if hashlib.sha256(b'\xef\xbb\xbf' + cand).hexdigest() == PRE['sha256']:
            prefix_ok = True; k_used = k; break
    ck('J6 前缀逐字节未变（对 PRE 基线 sha256）', prefix_ok, 'trailing CRLF k=%s' % k_used)
    ck('J7 PRE 基线 phase_headings == 15 且 bytes 递增',
       PRE.get('phase_headings') == 15 and len(mb) > PRE['bytes'],
       '%d -> %d B' % (PRE['bytes'], len(mb)))
    for anc in ('N2h1-α-9', 'com_layer', 'span_k', 'REACH', 'UNREACH_y', 'RECON_OK_LOOSE',
                'RECON_DRIFT', 'L_star_own', 'ell_reach', '多重集', '结构性退化', '53a293a8', 'Phase 17'):
        ck('J8 锚点存在: %s' % anc, anc in mt)

    # ---------------- K Ledger
    sec('K. Ledger 补登')
    ms = LG['measurements']
    ck('K1 Ledger n == 299', len(ms) == 299, 'n=%d' % len(ms))
    last = ms[-1]
    ck('K2 末条 phase == 16', last.get('phase') == 16)
    ck('K3 末条 seal/exec/result/amend1 sha8 与实盘一致',
       last.get('seal_sha8') == sha(SEALP)[:8] and last.get('exec_sha8') == sha(EXECP)[:8]
       and last.get('result_sha8') == sha(RESP)[:8] and last.get('amend1_sha8') == sha(AM1P)[:8])
    ck('K4 末条 anchor_result_sha8 == 53a293a8', last.get('anchor_result_sha8') == '53a293a8')
    ck('K5 末条 prereg_id == N2h1a9', last.get('prereg_id') == 'N2h1a9')
    ck('K6 末条 n_rows == 24399', last.get('n_rows') == 24399, str(last.get('n_rows')))
    ck('K7 末条 verdict 编码 centroid_partial',
       'centroid_partial' in str(last.get('verdict')), str(last.get('verdict')))
    tmp = {k: v for k, v in LG.items() if k != 'ledger_sha256_8'}
    h = hashlib.sha256(json.dumps(tmp, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    ck('K8 ledger_sha256_8 自哈希复算一致', h == LG['ledger_sha256_8'], '%s vs %s' % (h, LG['ledger_sha256_8']))
    ck('K9 补登前条目未被改动（前 298 条 phase<16 或非 16）',
       all(m.get('phase') != 16 for m in ms[:-1]))
    ck('K10 备份文件存在且 sha8 记录', os.path.exists(LEDBAK) and os.path.getsize(LEDBAK) > 1000,
       'sha8=%s' % sha(LEDBAK)[:8])

    # ---------------- L 基线与 wlog
    sec('L. memo_baseline + 当日 wlog')
    ck('L1 baseline.tag == post-append-phase16', BSE.get('tag') == 'post-append-phase16', BSE.get('tag'))
    ck('L2 baseline bytes/sha8 == 磁盘 MEMO', BSE.get('bytes') == len(mb)
       and BSE.get('sha8') == hashlib.sha256(mb).hexdigest()[:8], '%d B %s' % (BSE['bytes'], BSE['sha8']))
    ck('L3 baseline phase_headings == 16', len(BSE.get('phase_headings') or []) == 16)
    tags = [h['tag'] for h in (BSE.get('history') or [])]
    ck('L4 baseline history tag 唯一且不含自 tag', len(tags) == len(set(tags))
       and BSE.get('tag') not in tags, '%d 条' % len(tags))
    wb = open(WLOG, 'rb').read()
    wt = wb.decode('utf-8')
    ck('L5 wlog 含 Phase 16 段且唯一', wt.count('## Phase 16 / N2h1-') == 1)
    ck('L6 wlog 仍含 Phase 15 段', '## Phase 15 / N2h1-' in wt)
    seg = wt[wt.find('## Phase 16 / N2h1-'):] if '## Phase 16 / N2h1-' in wt else ''
    ck('L7 wlog Phase16 段含反引号（不被 bash 命令替换污染）', seg.count('`') >= 40,
       'backticks=%d' % seg.count('`'))
    ck('L8 wlog Phase16 段含 P6 否证叙述', ('P6' in seg) and ('预注册否证' in seg or '否证' in seg))
    ck('L9 wlog Phase16 段含 Ledger 299', '299' in seg)

    _out.append('')
    _out.append('TOTAL checks = %d ; FAIL = %d' % (_n[0], _n[1]))
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_out) + '\n')
    print('')
    print('TOTAL checks = %d ; FAIL = %d' % (_n[0], _n[1]))
    print('OUT ->', OUT)
    return 1 if _n[1] else 0


if __name__ == '__main__':
    raise SystemExit(main())
