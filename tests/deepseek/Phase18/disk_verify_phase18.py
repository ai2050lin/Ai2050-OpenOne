# -*- coding: utf-8 -*-
"""
Phase 18 独立磁盘复核（分区 A-N）。要求 0 FAIL。
本脚本 **不 import 主脚本**；com_of_mass（区间求和）/ stat_com_layer / spearman /
perm_null_com / perm_null_share 全部**独立重写**，再从 _armrec18_*.json 的 b 谱与 P17 的 w 谱、
P16 的 J/xhalf 序列逐位点重算。

用法：python disk_verify_phase18.py
产物：tests/deepseek_temp/Phase18/disk_verify_phase18.txt （PASS/FAIL 汇总）
"""
import io
import os
import json
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
SEALP = os.path.join(P18T, 'N2h1a11_design_seal.json')
EXECP = os.path.join(P18T, 'execution_phase18.json')
RESP = os.path.join(P18T, 'result_phase18.json')
PROBEP = os.path.join(P18T, '_probe_feasibility_A0.json')
PROBEAP = os.path.join(P18T, '_probe_analysis_A0.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
B18 = os.path.join(P18T, 'memo_baseline_preappend_phase18.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
LEDBAK = os.path.join(P18T, 'atlas_ledger_backup_pre_phase18.json')
OUT = os.path.join(P18T, 'disk_verify_phase18.txt')

_out = []
_n = [0, 0]


def ck(tag, ok, detail=''):
    _n[0] += 1
    if not ok:
        _n[1] += 1
    _out.append('[%s] %-62s %s' % ('PASS' if ok else 'FAIL', tag, detail))
    print(_out[-1])


def sec(t):
    _out.append('')
    _out.append('== ' + t + ' ==')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def jload(p):
    return json.loads(open(p, 'rb').read().decode('utf-8'))


# ---------------------------------------------------------------- 独立重写
def _com_interval(mass_by_site, sites):
    """区间求和口径：W_j = sum_{l in [s_j,s_{j+1})} mass[l] ; com = sum W_j mid_j / sum W_j。"""
    s = np.asarray(sites, float)
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    return float((vals * mid).sum() / den), vals


def _com_layer(jumps, sites):
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None
    s = np.asarray(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if den <= 1e-12:
        return None
    return float((a * mid).sum() / den)


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def _perm_null_com(vals, sites, seed, n_bp):
    s = np.asarray(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    n = len(vals)
    rng = np.random.default_rng(int(seed))
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = vals[rng.permutation(n)]
        den = float(p.sum())
        out[b] = float((p * mid).sum() / den) if den > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    if len(fin) == 0:
        return None, None
    return float(np.percentile(fin, 5)), float(np.percentile(fin, 95))


def _perm_null_share(b_mlp, b_attn, sites_reach, nb, seed, n_bp):
    mi = {int(s): i for i, s in enumerate(sites_reach)}
    idx = [mi[l] for l in nb if l in mi]
    m = np.array([abs(b_mlp.get(int(l), 0.0)) for l in sites_reach], float)
    a = np.array([abs(b_attn.get(int(l), 0.0)) for l in sites_reach], float)
    n = len(sites_reach)
    rng = np.random.default_rng(int(seed))
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        mp = m[rng.permutation(n)]
        d = mp[idx].sum() + a[idx].sum()
        out[b] = float(mp[idx].sum() / d) if d > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    if len(fin) == 0:
        return None, None
    return float(np.percentile(fin, 5)), float(np.percentile(fin, 95))


SEAL = jload(SEALP)
EX = jload(EXECP)
R = jload(RESP)
PROBE = jload(PROBEP)
PROBEA = jload(PROBEAP)
LG = jload(LEDGER)
PRE = jload(B18)
INFRA = jload(BASE)
P16P = os.path.join(ROOT, EX['anchor_result_p16_path'])
P17P = os.path.join(ROOT, EX['anchor_result_p17_path'])
A16 = jload(P16P)
A17 = jload(P17P)

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
FL = R['floors']
AO = EX['arm_order']
BP = int(EX['bootstrap']['BP'])
SEEDS = EX['bootstrap']['seeds']
NBW = int(EX['neighbourhood_width'])
PROFILE = [int(x) for x in EX['profile_sites']]


def loadrec(a):
    return jload(os.path.join(P18T, '_armrec18_%s.json' % a))


REC = {a: loadrec(a) for a in AO}
S = {a: REC[a]['E7_summary'] for a in AO}

# ================================================================ A
sec('A. 文件哈希 / 锚 / Ledger 对齐')
ck('seal sha8 == result.seal_sha256[:8]',
   sha(SEALP)[:8] == str(R['seal_sha256'])[:8], sha(SEALP)[:8])
ck('exec sha8 == result.exec_sha256[:8]',
   sha(EXECP)[:8] == str(R['exec_sha256'])[:8], sha(EXECP)[:8])
ck('anchor(P16) 文件 sha256 == exec.anchor_result_p16_sha256',
   sha(P16P) == EX['anchor_result_p16_sha256'], sha(P16P)[:16])
ck('anchor(P17) 文件 sha256 == exec.anchor_result_p17_sha256',
   sha(P17P) == EX['anchor_result_p17_sha256'], sha(P17P)[:16])
ck('result.anchor_p16_sha256[:8] == anchor sha8',
   sha(P16P)[:8] == str(R['anchor_result_p16_sha256'])[:8], str(R['anchor_result_p16_sha256'])[:8])
ck('result.anchor_p17_sha256[:8] == anchor sha8',
   sha(P17P)[:8] == str(R['anchor_result_p17_sha256'])[:8], str(R['anchor_result_p17_sha256'])[:8])
ent = [m for m in LG['measurements'] if m.get('phase') == 18]
ck('Ledger n == 301', len(LG['measurements']) == 301, str(len(LG['measurements'])))
ck('Ledger phase18 条目唯一', len(ent) == 1, str(len(ent)))
e = ent[-1]
res_sha8 = sha(RESP)[:8]
ck('Ledger.result_sha8 == 实际 result sha8', e.get('result_sha8') == res_sha8,
   '%s vs %s' % (e.get('result_sha8'), res_sha8))
ck('Ledger.seal/exec/probe sha8 一致',
   e.get('seal_sha8') == sha(SEALP)[:8] and e.get('exec_sha8') == sha(EXECP)[:8]
   and e.get('probe_sha8') == sha(PROBEP)[:8])
nrows = sum(int(R['arms'][a]['n_forwards']) for a in AO)
ck('Ledger.n_rows == Σ n_forwards', int(e.get('n_rows')) == nrows, '%s vs %d' % (e.get('n_rows'), nrows))
ck('Ledger.n_forwards_per_arm 一致',
   e.get('n_forwards_per_arm') == {a: int(R['arms'][a]['n_forwards']) for a in AO})
ck('Ledger 备份存在', os.path.exists(LEDBAK))
ck('rev_note 由 result 现场渲染（含三臂 com_B 串）',
   '/'.join('%.3f' % V[a]['Q6_com_B'] for a in AO) in (e.get('rev_note') or ''),
   '/'.join('%.3f' % V[a]['Q6_com_B'] for a in AO))
ck('rev_note 含三臂 share_mlp_beh 串',
   '/'.join('%.3f' % V[a]['Q4_share_mlp_beh_nb'] for a in AO) in (e.get('rev_note') or ''))

# ================================================================ B
sec('B. 装置门 / 保真度门')
for a in AO:
    x = REC[a]['E2_fidelity']
    ck('%-22s arch_max<=%s' % (a, FL['P18_FID_ARCH']), x['arch_max'] <= FL['P18_FID_ARCH'], '%.4e' % x['arch_max'])
    ck('%-22s blk_max<=%s' % (a, FL['P18_FID_BLK']), x['blk_max'] <= FL['P18_FID_BLK'], '%.4e' % x['blk_max'])
    ck('%-22s determinism==0' % a, float(REC[a]['E0_selfcheck']['determinism_maxdiff']) == 0.0)
    ck('%-22s Q0_device==cuda' % a, R['arms'][a]['Q0_device'] == 'cuda')
    ck('%-22s T=2_only' % a, bool(R['arms'][a]['T2_only']))
    ck('%-22s Q1_label==FID_PASS' % a, V[a]['Q1_label'] == 'FID_PASS')
    ck('%-22s 输出维度 ok' % a, bool(REC[a]['F4_dims_ok']) and bool(REC[a]['F5_o_proj_ok']))
ck('joint Q1 == FID_ALL_PASS', JV['Q1_joint'] == 'FID_ALL_PASS')

# ================================================================ C
sec('C. com_B 家族：独立区间求和重算（REACH 域 + 全域）')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]
    RE = [int(x) for x in s['reach']]
    ball = {int(l): float(v) for l, v in zip(AS, s['b_all'])}
    bmlp = {int(l): float(v) for l, v in zip(AS, s['b_mlp'])}
    batt = {int(l): float(v) for l, v in zip(AS, s['b_attn'])}
    btop = {int(l): float(v) for l, v in zip(AS, s['b_top1'])}
    bcum = {int(l): float(v) for l, v in zip(AS, s['b_cum'])}
    bconf = {int(l): float(v) for l, v in zip(AS, s['b_all_conf'])}
    W = {k: {int(l): float(v) for l, v in enumerate(A17['arms'][a]['E5_com_V'][k])}
         for k in ('w_all', 'w_mlp', 'w_attn')}
    def cend(tab, sites):
        return _com_interval({l: abs(tab[l]) for l in AS}, sites)[0]
    cB = cend(ball, RE); cBf = cend(ball, AS); cBc = cend(bconf, RE)
    cBm = cend(bmlp, RE); cBa = cend(batt, RE); cBt = cend(btop, RE); cBcu = cend(bcum, RE)
    cVr = _com_interval({l: W['w_all'][l] for l in AS}, RE)[0]
    ck('%-22s com_B(all) 重算==stored' % a, abs(cB - s['com_B']['INC_ALL']) <= 1e-9, '%.6f' % cB)
    ck('%-22s com_B(all)==verdict Q6_com_B' % a, abs(cB - V[a]['Q6_com_B']) <= 1e-9)
    ck('%-22s com_B_full(all) 重算一致' % a, abs(cBf - s['com_B_full']['INC_ALL']) <= 1e-9, '%.6f' % cBf)
    ck('%-22s com_B_conf(all) 重算一致' % a, abs(cBc - s['com_B_conf']['INC_ALL']) <= 1e-9, '%.6f' % cBc)
    ck('%-22s com_B(mlp) 重算一致' % a, abs(cBm - s['com_B']['INC_MLP']) <= 1e-9, '%.6f' % cBm)
    ck('%-22s com_B(attn) 重算一致' % a, abs(cBa - s['com_B']['INC_ATTN']) <= 1e-9, '%.6f' % cBa)
    ck('%-22s com_B(top1) 重算一致' % a, abs(cBt - s['com_B']['INC_TOP1']) <= 1e-9, '%.6f' % cBt)
    ck('%-22s com_B(cum) 重算一致' % a, abs(cBcu - s['com_B']['CUM_ALL']) <= 1e-9, '%.6f' % cBcu)
    ck('%-22s com_V(P17) 同域重算一致' % a, abs(cVr - s['com_V_recomputed']) <= 1e-9, '%.6f' % cVr)
    ck('%-22s com_V(P17)==P17 冻结锚' % a, abs(cVr - float(EX['p17_anchors'][a]['com_V'])) <= 1e-4)
    ck('%-22s gap==com_V-com_B(all)' % a,
       abs((cVr - cB) - V[a]['Q6_gap']) <= 1e-9, '%.6f' % (cVr - cB))
    ck('%-22s Q6 label==SHALLOWER(gap>=%s)' % (a, FL['SHALLOWER_MIN']),
       (cVr - cB) >= FL['SHALLOWER_MIN'] and V[a]['Q6_label'] == 'SHALLOWER')

# ================================================================ D
sec('D. 行为份额（邻域 nb 与 REACH）独立重算')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]; RE = [int(x) for x in s['reach']]; nb = [int(x) for x in s['nb']]
    ball = {int(l): float(v) for l, v in zip(AS, s['b_all'])}
    bmlp = {int(l): float(v) for l, v in zip(AS, s['b_mlp'])}
    batt = {int(l): float(v) for l, v in zip(AS, s['b_attn'])}
    btop = {int(l): float(v) for l, v in zip(AS, s['b_top1'])}
    W = {k: {int(l): float(v) for l, v in enumerate(A17['arms'][a]['E5_com_V'][k])}
         for k in ('w_all', 'w_mlp', 'w_attn')}
    ck('%-22s nb == P17 冻结邻域' % a, nb == [int(x) for x in EX['p17_anchors'][a]['neighbourhood']], str(nb))
    def shv(cn, cd, sites):
        num = float(sum(abs((bmlp if cn == 'INC_MLP' else batt if cn == 'INC_ATTN' else btop)[l]) for l in sites))
        den = float(sum(abs(ball[l]) for l in sites))
        return num / den
    sh_mlp = shv('INC_MLP', 'INC_ALL', nb); sh_att = shv('INC_ATTN', 'INC_ALL', nb)
    sh_t1 = shv('INC_TOP1', 'INC_ALL', nb)
    sh_mlp_re = shv('INC_MLP', 'INC_ALL', RE)
    v_m = float(sum(W['w_mlp'][l] for l in nb)); v_a = float(sum(W['w_attn'][l] for l in nb))
    v_all = float(sum(W['w_all'][l] for l in nb))
    sh_vec = v_m / v_all
    ck('%-22s share_mlp_beh_nb 重算一致' % a, abs(sh_mlp - s['share_mlp_beh_nb']) <= 1e-9, '%.6f' % sh_mlp)
    ck('%-22s share_attn_beh_nb 重算一致' % a, abs(sh_att - s['share_attn_beh_nb']) <= 1e-9, '%.6f' % sh_att)
    ck('%-22s share_top1_beh_nb 重算一致' % a, abs(sh_t1 - s['share_top1_beh_nb']) <= 1e-9, '%.6f' % sh_t1)
    ck('%-22s share_mlp_beh_reach 重算一致' % a, abs(sh_mlp_re - s['share_mlp_beh_reach']) <= 1e-9, '%.6f' % sh_mlp_re)
    ck('%-22s share_mlp_vec_nb 重算一致' % a, abs(sh_vec - s['share_mlp_vec_nb']) <= 1e-9, '%.6f' % sh_vec)
    ck('%-22s share_mlp_vec_nb==P17 冻结锚' % a,
       abs(sh_vec - float(EX['p17_anchors'][a]['share_mlp_nb'])) <= 1e-4, '%.6f' % sh_vec)
    ck('%-22s Q4 label==MLP_DOMINANT_BEH' % a,
       sh_mlp >= FL['MLP_DOM_MIN'] and V[a]['Q4_label'] == 'MLP_DOMINANT_BEH')

# ================================================================ E
sec('E. 行为剖面的 com_layer（对 b 取相邻差）')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]; RE = [int(x) for x in s['reach']]
    for c, key in (('INC_ALL', 'comlayer_B_all'), ('INC_MLP', 'comlayer_B_mlp'), ('INC_ATTN', 'comlayer_B_attn')):
        bm = {int(l): float(v) for l, v in zip(AS, s['b_all' if c == 'INC_ALL' else 'b_mlp' if c == 'INC_MLP' else 'b_attn'])}
        arr = np.array([bm[l] for l in RE], float)
        cl = _com_layer(np.diff(arr), RE)
        ck('%-22s %s 重算一致' % (a, key), abs(cl - s[key]) <= 1e-9, '%.6f' % cl)

# ================================================================ F
sec('F. spearman（同对象 w vs |b|，以及 P17 口径 w vs J）')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]; RE = [int(x) for x in s['reach']]
    ball = {int(l): float(v) for l, v in zip(AS, s['b_all'])}
    bmlp = {int(l): float(v) for l, v in zip(AS, s['b_mlp'])}
    batt = {int(l): float(v) for l, v in zip(AS, s['b_attn'])}
    W = {k: {int(l): float(v) for l, v in enumerate(A17['arms'][a]['E5_com_V'][k])}
         for k in ('w_all', 'w_mlp', 'w_attn')}
    sp1 = _spearman([W['w_all'][l] for l in RE], [abs(ball[l]) for l in RE])
    sp2 = _spearman([W['w_mlp'][l] for l in RE], [abs(bmlp[l]) for l in RE])
    sp3 = _spearman([W['w_attn'][l] for l in RE], [abs(batt[l]) for l in RE])
    ck('%-22s spearman(w_all,|b_all|) 重算一致' % a, abs(sp1 - s['spearman_wall_ball']) <= 1e-12, '%.6f' % sp1)
    ck('%-22s spearman(w_mlp,|b_mlp|) 重算一致' % a, abs(sp2 - s['spearman_wmlp_bmlp']) <= 1e-12, '%.6f' % sp2)
    ck('%-22s spearman(w_attn,|b_attn|) 重算一致' % a, abs(sp3 - s['spearman_wattn_battn']) <= 1e-12, '%.6f' % sp3)
    # P17 口径：w_all vs P16 J（REACH 交集）
    J16 = {int(k): float(v) for k, v in zip(A16['E4_summary'][a]['sites'], A16['E4_summary'][a]['J'])}
    xl = [l for l in RE if l in J16]
    spJ = _spearman([W['w_all'][l] for l in xl], [J16[l] for l in xl])
    ck('%-22s spearman(w_all,J) 重算一致（P17 口径）' % a, abs(spJ - s['spearman_wall_J']) <= 1e-12, '%.6f' % spJ)
    ck('%-22s Q7 label==EFFICACY_COUPLED' % a,
       sp1 > 0 and V[a]['Q7_label'] == 'EFFICACY_COUPLED')

# ================================================================ G
sec('G. r_lin（层内增益诊断）独立重算')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]; RE = [int(x) for x in s['reach']]; nb = [int(x) for x in s['nb']]
    L_star = int(EX['p16_anchors'][a]['L_star_own'])
    ball = {int(l): float(v) for l, v in zip(AS, s['b_all'])}
    bmlp = {int(l): float(v) for l, v in zip(AS, s['b_mlp'])}
    batt = {int(l): float(v) for l, v in zip(AS, s['b_attn'])}
    rl = {l: (abs(ball[l] - (bmlp[l] + batt[l])) / max(abs(ball[l]), 1e-12)) for l in AS}
    ck('%-22s rlin_by_site 重算一致（逐位点）' % a,
       all(abs(rl[int(l)] - float(v)) <= 1e-12 for l, v in s['rlin_by_site'].items()))
    arr = np.array([rl[l] for l in AS], float)
    order = np.argsort(-arr)
    amax = int(AS[int(order[0])]); peak = float(arr[order[0]]); second = float(arr[order[1]])
    ck('%-22s rlin_argmax 重算一致' % a, amax == int(s['rlin_argmax']))
    ck('%-22s rlin_peak_ratio 重算一致' % a, abs(peak / second - s['rlin_peak_ratio']) <= 1e-12)
    ck('%-22s rlin_nb 重算一致' % a,
       abs(float(np.mean([rl[l] for l in nb])) - s['rlin_nb']) <= 1e-12, '%.4f' % s['rlin_nb'])
    ck('%-22s rlin_reach 重算一致' % a,
       abs(float(np.mean([rl[l] for l in RE])) - s['rlin_reach']) <= 1e-12, '%.4f' % s['rlin_reach'])
    # REACH 域峰值比：这是 E-rlin 勘误的核心事实 —— seal rationale 的「0.186@L26 / 4.05」是 A0 的
    # REACH 域读数（本脚本独立重算必须复现），而判据域 ALL_SITES 上的比值骤降；A1/A2 在任何域都 < 3。
    _re_sorted = sorted(((rl[l], l) for l in RE), reverse=True)
    _rr = (_re_sorted[0][0] / _re_sorted[1][0]) if _re_sorted[1][0] > 1e-12 else None
    if a.startswith('A0'):
        ck('%-22s REACH 域峰值比 复现 seal rationale 4.05' % a,
           _rr is not None and abs(_rr - 4.05) <= 0.01, '%.3f' % _rr)
        ck('%-22s REACH 域次大 == seal rationale 的 L26(0.186)' % a,
           _re_sorted[1][1] == 26 and abs(_re_sorted[1][0] - 0.186) <= 1e-3,
           'L%d = %.3f' % (_re_sorted[1][1], _re_sorted[1][0]))
    else:
        ck('%-22s REACH 域峰值比 < 3（P6 在任何域都 FAIL）' % a,
           _rr is not None and _rr < FL['RLIN_PEAK_RATIO_MIN'], '%.3f' % _rr)
    # Q8 label 由重算量按主脚本文义导出（位置 + 分离度），再比对报告值——不预设「应为通过」
    _lab8 = ('SUPERADD_AT_WINDOW'
             if (amax == L_star and (peak / second) >= FL['RLIN_PEAK_RATIO_MIN'])
             else 'NO_WINDOW_CONTRAST')
    ck('%-22s Q8 label 由重算量导出' % a, V[a]['Q8_label'] == _lab8,
       '%s (位置 %s, 比值 %.3f)' % (_lab8, 'OK' if amax == L_star else 'NO', peak / second))

# ================================================================ H
sec('H. 桥接门：CUM_ALL@L* vs P16 FULL_SWAP')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]
    L_star = int(EX['p16_anchors'][a]['L_star_own'])
    bcum = {int(l): float(v) for l, v in zip(AS, s['b_cum'])}
    cb = bcum[L_star]
    fs = float(A16['arms'][a]['E2_full_swap']['FULL_SWAP'])
    rel = abs(cb - fs) / max(abs(fs), 1e-12)
    ck('%-22s bridge_site == L*_own' % a, int(s['bridge_site']) == L_star, str(s['bridge_site']))
    ck('%-22s cum_bridge 重算一致' % a, abs(cb - s['cum_bridge']) <= 1e-9, '%.6f' % cb)
    ck('%-22s full_swap 重算一致（P16 E2_full_swap）' % a, abs(fs - float(s['full_swap'])) <= 1e-9, '%.6f' % fs)
    ck('%-22s bridge_rel 重算一致' % a, abs(rel - s['bridge_rel']) <= 1e-12, '%.4f' % rel)
    _lab3 = 'BRIDGE_OK' if rel <= FL['BRIDGE_TOL_CUM'] else 'BRIDGE_DRIFT'
    ck('%-22s Q3 label 由重算 rel 导出' % a, V[a]['Q3_label'] == _lab3,
       '%s (rel %.4f / tol %s)' % (_lab3, rel, FL['BRIDGE_TOL_CUM']))

# ================================================================ I
sec('I. 置换零假设（com_B 与份额）独立重跑')
for a in AO:
    s = S[a]
    AS = [int(x) for x in s['sites_all']]; RE = [int(x) for x in s['reach']]; nb = [int(x) for x in s['nb']]
    ball = {int(l): float(v) for l, v in zip(AS, s['b_all'])}
    bmlp = {int(l): float(v) for l, v in zip(AS, s['b_mlp'])}
    batt = {int(l): float(v) for l, v in zip(AS, s['b_attn'])}
    _, va = _com_interval({l: abs(ball[l]) for l in AS}, RE)
    _, vm = _com_interval({l: abs(bmlp[l]) for l in AS}, RE)
    p5a, p95a = _perm_null_com(va, RE, SEEDS['comB_inc'], BP)
    p5m, p95m = _perm_null_com(vm, RE, SEEDS['comB_mlp'], BP)
    p5s, p95s = _perm_null_share(bmlp, batt, RE, nb, SEEDS['share_mlp'], BP)
    na = s['null_comB_inc']; nm = s['null_comB_mlp']; ns_ = s['null_share']
    oa = _com_interval({l: abs(ball[l]) for l in AS}, RE)[0]
    ck('%-22s null(all) obs==com_B' % a, abs(oa - na['obs_com']) <= 1e-9)
    ck('%-22s null(all) p5/p95 重算一致' % a,
       abs(p5a - na['com_p5']) <= 1e-9 and abs(p95a - na['com_p95']) <= 1e-9, 'p95=%.4f' % p95a)
    ck('%-22s null(all) tail 重算一致' % a,
       na['com_tail'] == ('high' if oa >= p95a else 'low' if oa <= p5a else 'none'))
    ck('%-22s null(mlp) p5/p95 重算一致' % a,
       abs(p5m - nm['com_p5']) <= 1e-9 and abs(p95m - nm['com_p95']) <= 1e-9, 'p95=%.4f' % p95m)
    ck('%-22s null(mlp) tail 重算一致' % a,
       nm['com_tail'] == ('high' if nm['obs_com'] >= p95m else 'low' if nm['obs_com'] <= p5m else 'none'))
    ck('%-22s null(share) p5/p95 重算一致' % a,
       abs(p5s - ns_['share_p5']) <= 1e-9 and abs(p95s - ns_['share_p95']) <= 1e-9, 'p95=%.4f' % p95s)
    ck('%-22s null(share) tail 重算一致' % a,
       ns_['share_tail'] == ('high' if ns_['obs_share'] >= p95s else 'low' if ns_['obs_share'] <= p5s else 'none'))

# ================================================================ J
sec('J. 判决 / 预测一致性')
_n3 = sum(1 for a in AO if V[a]['Q3_label'] == 'BRIDGE_OK')
_q3j = 'BRIDGE_ALL_OK' if _n3 == len(AO) else ('BRIDGE_PARTIAL' if _n3 > 0 else 'BRIDGE_DRIFT_ALL')
ck('Q3 计数/联合标签 由重算导出', JV['Q3_joint'] == _q3j, '%s (n_ok=%d)' % (_q3j, _n3))
ck('Q4 计数与逐臂标签一致',
   JV['Q4_counts']['MLP_DOMINANT_BEH'] == sum(1 for a in AO if V[a]['Q4_label'] == 'MLP_DOMINANT_BEH'),
   str(JV['Q4_counts']))
ck('Q6 计数与逐臂标签一致',
   JV['Q6_counts']['SHALLOWER'] == sum(1 for a in AO if V[a]['Q6_label'] == 'SHALLOWER'),
   str(JV['Q6_counts']))
ck('Q7 计数与逐臂标签一致',
   JV['Q7_counts']['COUPLED'] == sum(1 for a in AO if V[a]['Q7_label'] == 'EFFICACY_COUPLED'),
   str(JV['Q7_counts']))
_n8 = sum(1 for a in AO if V[a]['Q8_label'] == 'SUPERADD_AT_WINDOW')
_q8j = ('SUPERADD_AT_WINDOW_ALL' if _n8 == len(AO)
        else ('SUPERADD_AT_WINDOW_PARTIAL' if _n8 > 0 else 'NO_WINDOW_CONTRAST_ALL'))
ck('Q8 计数/联合标签 由重算导出',
   JV['Q8_counts']['SUPERADD_AT_WINDOW'] == _n8 and JV['Q8_joint'] == _q8j, '%s (n=%d)' % (_q8j, _n8))
ck('P1..P6 pass 与 JOINT 一致（镜像主脚本文义）',
   PC['P1']['pass_'] == bool(JV['Q0_apparatus_all'] and JV['Q0_device_all'] and JV['Q1_joint'] == 'FID_ALL_PASS')
   and PC['P2']['pass_'] == bool(JV['Q2_joint'] == 'ANCHOR_ALL_OK' and JV['Q3_joint'] == 'BRIDGE_ALL_OK')
   and PC['P3']['pass_'] == bool(JV['Q4_counts']['MLP_DOMINANT_BEH'] >= 2 and JV['Q5_counts']['CONSISTENT'] >= 2)
   and PC['P4']['pass_'] == bool(JV['Q6_counts']['SHALLOWER'] >= len(AO) - 1)
   and PC['P5']['pass_'] == bool(JV['Q7_counts']['COUPLED'] >= 2)
   and PC['P6']['pass_'] == bool(JV['Q8_counts']['SUPERADD_AT_WINDOW'] >= 2),
   str({k: PC[k]['pass_'] for k in sorted(PC)}))
ck('P7 描述性 N/A', PC['P7']['pass_'] is None)
_anc_bits = {}
for a in AO:
    d = REC[a]['E8_anchor']['detail']
    bits = [bool(REC[a]['E8_anchor'].get('ok'))]
    for k, vv in d.items():
        if isinstance(vv, dict) and 'ok' in vv:
            bits.append(bool(vv['ok']))
            if isinstance(vv.get('got'), (int, float)) and isinstance(vv.get('expected'), (int, float)):
                bits.append(abs(float(vv['got']) - float(vv['expected'])) <= 1e-6)
    _anc_bits[a] = all(bits)
ck('P2 anchor detail 全 ok + 逐位<=1e-6（三臂）', all(_anc_bits.values()), str(_anc_bits))
ck('确认集 Δ(INC_ALL) <= CONF_TOL_COMB',
   max(V[a]['Q9_conf']['d_com']['INC_ALL'] for a in AO) <= FL['CONF_TOL_COMB'],
   '%.3f' % max(V[a]['Q9_conf']['d_com']['INC_ALL'] for a in AO))

# ================================================================ K
sec('K. 探针跨实现交叉验证（A0，独立区间求和）')
_pb = {int(k): float(v) for k, v in PROBEA['b']['INC_ALL'].items()}
_pl = {int(k): float(v) for k, v in PROBEA['b']['INC_MLP'].items()}
_pa = {int(k): float(v) for k, v in PROBEA['b']['INC_ATTN'].items()}
_pc, _ = _com_interval({l: abs(_pb[l]) for l in _pb}, [int(x) for x in PROBEA['reach']])
_pm, _ = _com_interval({l: abs(_pl[l]) for l in _pl}, [int(x) for x in PROBEA['reach']])
ck('探针 b 谱重算 com_B(all) == 探针自报', abs(_pc - PROBEA['com_B']['INC_ALL']) <= 1e-9,
   '%.9f vs %.9f' % (_pc, PROBEA['com_B']['INC_ALL']))
ck('探针 b 谱重算 com_B(mlp) == 探针自报', abs(_pm - PROBEA['com_B']['INC_MLP']) <= 1e-9)
ck('探针 com_B(all) == 生产 A0 com_B（跨实现逐位一致）',
   abs(PROBEA['com_B']['INC_ALL'] - V['A0_calib_qwen3-4b-nf4']['Q6_com_B']) <= 1e-9,
   '%.9f vs %.9f' % (PROBEA['com_B']['INC_ALL'], V['A0_calib_qwen3-4b-nf4']['Q6_com_B']))
_shp = (sum(abs(_pl[l]) for l in PROBEA['nb']) / sum(abs(_pb[l]) for l in PROBEA['nb']))
ck('探针 share_mlp_beh(nb) == 生产 A0（跨实现逐位一致）',
   abs(_shp - V['A0_calib_qwen3-4b-nf4']['Q4_share_mlp_beh_nb']) <= 1e-9,
   '%.9f vs %.9f' % (_shp, V['A0_calib_qwen3-4b-nf4']['Q4_share_mlp_beh_nb']))
_pw = {int(l): float(v) for l, v in enumerate(A17['arms']['A0_calib_qwen3-4b-nf4']['E5_com_V']['w_all'])}
_sp = _spearman([_pw[l] for l in PROBEA['reach']], [abs(_pb[l]) for l in PROBEA['reach']])
ck('探针 spearman(w_all,|b_all|) == 生产 A0', abs(_sp - V['A0_calib_qwen3-4b-nf4']['Q7_spearman_wall_ball']) <= 1e-9,
   '%.6f vs %.6f' % (_sp, V['A0_calib_qwen3-4b-nf4']['Q7_spearman_wall_ball']))

# ================================================================ L
sec('L. MEMO 落盘（结构 / 前缀锚 / 编码）')
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig')
lines = mt.split('\r\n')
h18 = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 18')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
ck('BOM 存在', mb[:3] == b'\xef\xbb\xbf')
ck('bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0)
ck('Phase 18 标题唯一', len(h18) == 1, str(h18))
ck('Phase 标题总数 == 18', len(allh) == 18, str(len(allh)))
ck('MEMO 前缀 == 追加前基线（逐字节）',
   mb[:int(PRE['bytes'])] == open(MEMO, 'rb').read()[:int(PRE['bytes'])]
   and hashlib.sha256(mb[:int(PRE['bytes'])]).hexdigest()[:8] == str(PRE['sha8']), str(PRE['sha8']))
ck('MEMO 追加前基线 phase_headings == 17', int(PRE['phase_headings']) == 17)
for tok in ('MLP_DOMINANT_BEH_ALL', 'BEHAVIOR_SHALLOWER', JV['Q3_joint'], 'com_B', 'share_mlp_beh',
            'N2h1-α-11', 'Phase 19', 'ee27627a', '0c1949cb'):
    ck('MEMO 含锚点 %s' % tok, tok in mt)

# ================================================================ M
sec('M. wlog 落盘')
wb = open(WLOG, 'rb').read().decode('utf-8')
ck('wlog 含 Phase 18 段', '## Phase 18 / N2h1-' in wb)
ck('wlog 含 Phase 17 段（未被破坏）', '## Phase 17 / N2h1-' in wb)
ck('wlog 含 com_B / share_mlp_beh', ('com_B' in wb) and ('share_mlp_beh' in wb))

# ================================================================ N
sec('N. _infra 基线刷新')
ck('infra tag == post-append-phase18', INFRA.get('tag') == 'post-append-phase18', str(INFRA.get('tag')))
ck('infra bytes/lines/sha8 == 实际 MEMO',
   int(INFRA['bytes']) == len(mb) and int(INFRA['lines']) == len(lines)
   and INFRA['sha8'] == hashlib.sha256(mb).hexdigest()[:8],
   '%d/%d/%s' % (INFRA['bytes'], INFRA['lines'], INFRA['sha8']))
htags = [h.get('tag') for h in (INFRA.get('history') or [])]
ck('infra history 含 pre-append-phase18', 'pre-append-phase18' in htags, str(htags))
ck('infra history 不含自条目 post-append-phase18', 'post-append-phase18' not in htags)
ck('infra phase_headings == 18',
   len([x for x in (INFRA.get('phase_headings') or []) if isinstance(x, int)]) == 18)

# ================================================================
sec('汇总')
tot = _n[0]; fail = _n[1]
_out.append('')
_out.append('TOTAL %d checks, %d FAIL' % (tot, fail))
_out.append('clock done')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_out) + '\n')
print('\nTOTAL %d checks, %d FAIL -> %s' % (tot, fail, OUT))
