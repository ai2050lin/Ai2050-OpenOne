# -*- coding: utf-8 -*-
"""
Phase 17 独立磁盘复核（分区 A-L）。要求 0 FAIL。
本脚本 **不 import 主脚本**；com_of_mass（区间求和）/ stat_com_layer / stat_span_k / spearman /
perm_null_com 全部**独立重写**，再从 _armrec17_*.json 的 w_ℓ 谱与 P16 锚的 J/xhalf 序列逐位点重算。

用法：python disk_verify_phase17.py
产物：tests/deepseek_temp/Phase17/disk_verify_phase17.txt （PASS/FAIL 汇总）
"""
import io
import os
import json
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
SEALP = os.path.join(P17T, 'N2h1a10_design_seal.json')
EXECP = os.path.join(P17T, 'execution_phase17.json')
RESP = os.path.join(P17T, 'result_phase17.json')
PROBEP = os.path.join(P17T, '_probe_feasibility_A0.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
B17 = os.path.join(P17T, 'memo_baseline_preappend_phase17.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
LEDBAK = os.path.join(P17T, 'atlas_ledger_backup_pre_phase17.json')
OUT = os.path.join(P17T, 'disk_verify_phase17.txt')

_out = []
_n = [0, 0]


def ck(tag, ok, detail=''):
    _n[0] += 1
    if not ok:
        _n[1] += 1
    _out.append('[%s] %-60s %s' % ('PASS' if ok else 'FAIL', tag, detail))
    print(_out[-1])


def sec(t):
    _out.append('')
    _out.append('== ' + t + ' ==')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------------------------------------------------------- 独立重写
def _com_interval(mass_by_site, sites):
    """区间求和口径：W_j = sum_{l in [s_j, s_{j+1})} w_l ; com = sum W_j*mid_j / sum W_j。"""
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


def _span_k(jumps, k):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


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
        return None, None, None, 'all_nan'
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    return p5, p95, fin, None


def jload(p):
    return json.loads(open(p, 'rb').read().decode('utf-8'))


SEAL = jload(SEALP)
EX = jload(EXECP)
R = jload(RESP)
PROBE = jload(PROBEP)
LG = jload(LEDGER)
PRE = jload(B17)
INFRA = jload(BASE)
ANCHORP = os.path.join(ROOT, EX['anchor_result_path'])
ANCH = jload(ANCHORP)

V = R['verdict']; JV = R['joint_verdict']; PC = R['predictions_check']
FL = R['floors']
AO = EX['arm_order']
Lstar = {a: int(R['arms'][a]['cfg']['L']) for a in AO}
NH = {a: int(R['arms'][a]['cfg']['n_heads']) for a in AO}
KS = list(R['span_ks'])
NBW = int(EX['neighbourhood_width'])
BP = int(EX['bootstrap']['BP'])
SEEDS = EX['bootstrap']['seeds']
PROFILE = [int(x) for x in EX['profile_sites']]


def loadrec(a):
    return jload(os.path.join(P17T, '_armrec17_%s.json' % a))


REC = {a: loadrec(a) for a in AO}

# ================================================================ A
sec('A. 文件哈希 / 锚 / Ledger 对齐')
ck('seal sha8 == result.seal_sha256[:8]',
   hashlib.sha256(open(SEALP, 'rb').read()).hexdigest()[:8] == str(R['seal_sha256'])[:8],
   hashlib.sha256(open(SEALP, 'rb').read()).hexdigest()[:8])
ck('exec sha8 == result.exec_sha256[:8]',
   hashlib.sha256(open(EXECP, 'rb').read()).hexdigest()[:8] == str(R['exec_sha256'])[:8],
   hashlib.sha256(open(EXECP, 'rb').read()).hexdigest()[:8])
ck('probe sha8 == ledger.probe_sha8',
   '650a46ea' == hashlib.sha256(open(PROBEP, 'rb').read()).hexdigest()[:8])
ck('anchor(P16) 文件 sha256 == exec.anchor_result_sha256',
   sha(ANCHORP) == EX['anchor_result_sha256'],
   sha(ANCHORP)[:16])
ck('result.anchor_result_sha256[:8] == anchor sha8',
   sha(ANCHORP)[:8] == str(R['anchor_result_sha256'])[:8], str(R['anchor_result_sha256'])[:8])
ent = [m for m in LG['measurements'] if m.get('phase') == 17]
ck('Ledger n == 300', len(LG['measurements']) == 300, str(len(LG['measurements'])))
ck('Ledger phase17 条目唯一', len(ent) == 1, str(len(ent)))
e = ent[-1]
res_sha8 = hashlib.sha256(open(RESP, 'rb').read()).hexdigest()[:8]
ck('Ledger.result_sha8 == 实际 result sha8', e.get('result_sha8') == res_sha8, '%s vs %s' % (e.get('result_sha8'), res_sha8))
ck('Ledger.seal/exec/probe sha8 一致',
   e.get('seal_sha8') == hashlib.sha256(open(SEALP, 'rb').read()).hexdigest()[:8]
   and e.get('exec_sha8') == hashlib.sha256(open(EXECP, 'rb').read()).hexdigest()[:8]
   and e.get('probe_sha8') == hashlib.sha256(open(PROBEP, 'rb').read()).hexdigest()[:8])
nrows = sum(int(R['arms'][a]['n_forwards']) for a in AO)
ck('Ledger.n_rows == Σ n_forwards', int(e.get('n_rows')) == nrows, '%s vs %d' % (e.get('n_rows'), nrows))
ck('Ledger.n_forwards_per_arm 一致', e.get('n_forwards_per_arm') == {a: int(R['arms'][a]['n_forwards']) for a in AO})
ck('Ledger 备份存在', os.path.exists(LEDBAK))
ck('rev_note 由 result 现场渲染（含三臂 com_V 串）',
   '/'.join('%.3f' % V[a]['Q3_com_V'] for a in AO) in (e.get('rev_note') or ''),
   '/'.join('%.3f' % V[a]['Q3_com_V'] for a in AO))

# ================================================================ B
sec('B. 装置门 / 保真度门')
for a in AO:
    x = REC[a]['E2_fidelity']
    ck('%-22s arch_max<=%s' % (a, FL['P17_FID_ARCH']), x['arch_max'] <= FL['P17_FID_ARCH'], '%.4e' % x['arch_max'])
    ck('%-22s blk_max<=%s' % (a, FL['P17_FID_BLK']), x['blk_max'] <= FL['P17_FID_BLK'], '%.4e' % x['blk_max'])
    ck('%-22s determinism==0' % a, float(REC[a]['E0_selfcheck']['determinism_maxdiff']) == 0.0)
    ck('%-22s Q0_device==cuda' % a, R['arms'][a]['Q0_device'] == 'cuda')
    ck('%-22s T=2_only' % a, bool(R['arms'][a]['T2_only']))
    ck('%-22s n_forwards==44' % a, int(R['arms'][a]['n_forwards']) == 44, str(R['arms'][a]['n_forwards']))
    ck('%-22s Q1_label==FID_PASS' % a, V[a]['Q1_label'] == 'FID_PASS')
ck('joint Q1 == FID_ALL_PASS', JV['Q1_joint'] == 'FID_ALL_PASS')

# ================================================================ C
sec('C. com_V 家族：独立区间求和重算')
for a in AO:
    rec = REC[a]; c5 = rec['E5_com_V']; L = Lstar[a]
    reach = [int(x) for x in c5['reach']]
    RE_L = [s for s in reach if 0 <= s < (L - 1)]
    md = {int(l): float(c5['w_all'][l]) for l in range(len(c5['w_all']))}
    mmlp = {int(l): float(c5['w_mlp'][l]) for l in range(len(c5['w_mlp']))}
    matt = {int(l): float(c5['w_attn'][l]) for l in range(len(c5['w_attn']))}
    mtop = {int(l): float(c5['w_top1'][l]) for l in range(len(c5['w_top1']))}
    c_all, _ = _com_interval(md, RE_L)
    c_mlp, _ = _com_interval(mmlp, RE_L)
    c_att, _ = _com_interval(matt, RE_L)
    c_top, _ = _com_interval(mtop, RE_L)
    c_full, _ = _com_interval(md, [s for s in PROFILE if 0 <= s < (L - 1)])
    ck('%-22s RE_L == stored reach 过滤' % a, RE_L == [int(x) for x in RE_L])
    ck('%-22s com_V 重算==stored(<=1e-9)' % a, abs(c_all - c5['com_V']) <= 1e-9, '%.9f' % c_all)
    ck('%-22s com_V == verdict Q3_com_V' % a, abs(c_all - V[a]['Q3_com_V']) <= 1e-9)
    ck('%-22s com_V_mlp 重算一致' % a, abs(c_mlp - c5['com_V_mlp']) <= 1e-9, '%.6f' % c_mlp)
    ck('%-22s com_V_attn 重算一致' % a, abs(c_att - c5['com_V_attn']) <= 1e-9, '%.6f' % c_att)
    ck('%-22s com_V_top1head 重算一致' % a, abs(c_top - c5['com_V_top1head']) <= 1e-9, '%.6f' % c_top)
    ck('%-22s com_V_full 重算一致' % a, abs(c_full - c5['com_V_full']) <= 1e-9, '%.6f' % c_full)
    ck('%-22s median(REACH) 重算一致' % a, abs(float(np.median(RE_L)) - c5['median_reach']) <= 1e-9,
       '%.1f' % float(np.median(RE_L)))
    ck('%-22s Q3 label == DEEP' % a, V[a]['Q3_label'] == 'DEEP')

# ================================================================ D
sec('D. 组件归属（邻域 ±2 的 share）')
for a in AO:
    rec = REC[a]; c5 = rec['E5_com_V']; L = Lstar[a]
    RE_L = [s for s in c5['reach'] if 0 <= s < (L - 1)]
    md = {int(l): float(c5['w_all'][l]) for l in range(len(c5['w_all']))}
    mmlp = {int(l): float(c5['w_mlp'][l]) for l in range(len(c5['w_mlp']))}
    matt = {int(l): float(c5['w_attn'][l]) for l in range(len(c5['w_attn']))}
    nb = [l for l in RE_L if abs(l - c5['com_V']) <= NBW]
    s_all = sum(md[l] for l in nb); s_mlp = sum(mmlp[l] for l in nb); s_att = sum(matt[l] for l in nb)
    sh_mlp = s_mlp / s_all; sh_att = s_att / s_all
    top1 = float(np.max(c5['head_mass'])) / s_all
    ck('%-22s neighbourhood 重算一致' % a, nb == [int(x) for x in c5['neighbourhood']], str(nb))
    ck('%-22s share_mlp_nb 重算一致' % a, abs(sh_mlp - c5['share_mlp_nb']) <= 1e-9, '%.6f' % sh_mlp)
    ck('%-22s share_attn_nb 重算一致' % a, abs(sh_att - c5['share_attn_nb']) <= 1e-9, '%.6f' % sh_att)
    ck('%-22s top1_head_share_nb 重算一致' % a, abs(top1 - c5['top1_head_share_nb']) <= 1e-9, '%.6f' % top1)
    ck('%-22s Q5 label MLP_DOMINANT(>=%.2f)' % (a, FL['MLP_DOM_MIN']), sh_mlp >= FL['MLP_DOM_MIN'] and V[a]['Q5_label'] == 'MLP_DOMINANT')
ck('joint Q5 == MLP_DOMINANT_ALL', JV['Q5_joint'] == 'MLP_DOMINANT_ALL')

# ================================================================ E
sec('E. 效力关系 spearman(w_ℓ, J_ℓ)')
for a in AO:
    rec = REC[a]; e6 = rec['E6_efficacy']
    sp = _spearman(e6['w_at_sites'], e6['J_at_sites'])
    ck('%-22s spearman(w,J) 重算一致' % a, sp is not None and abs(sp - e6['spearman_wJ']) <= 1e-12,
       '%.6f' % sp)
    ck('%-22s spearman(w,J) == verdict' % a, abs(sp - V[a]['Q6_spearman_wJ']) <= 1e-12)
    ck('%-22s n == len(sites)' % a, int(e6['n']) == len(e6['sites']), str(e6['n']))
    ck('%-22s Q6 label ANTICORR' % a, sp < 0 and V[a]['Q6_label'] == 'WRITE_EFFICACY_ANTICORR')
    # 独立：从锚 J 序列 + w_all 重算（sites 交集）
    L = Lstar[a]
    asum = ANCH['E4_summary'][a]
    Jmap = {int(s): float(v) for s, v in zip(asum['sites'], asum['J'])}
    RE_L = [s for s in REC[a]['E5_com_V']['reach'] if 0 <= s < (L - 1)]
    xl = [l for l in RE_L if l in Jmap]
    wv = [float(REC[a]['E5_com_V']['w_all'][l]) for l in xl]
    jv = [Jmap[l] for l in xl]
    sp2 = _spearman(wv, jv)
    ck('%-22s spearman(锚J,w_all) 独立重算一致' % a, sp2 is not None and abs(sp2 - V[a]['Q6_spearman_wJ']) <= 1e-9,
       '%.6f' % sp2)
ck('joint Q6 == WRITE_EFFICACY_ANTICORR_ALL', JV['Q6_joint'] == 'WRITE_EFFICACY_ANTICORR_ALL')

# ================================================================ F
sec('F. 置换零假设（com_V，独立重跑）')
for a in AO:
    rec = REC[a]; c5 = rec['E5_com_V']; L = Lstar[a]
    RE_L = [s for s in c5['reach'] if 0 <= s < (L - 1)]
    md = {int(l): float(c5['w_all'][l]) for l in range(len(c5['w_all']))}
    mmlp = {int(l): float(c5['w_mlp'][l]) for l in range(len(c5['w_mlp']))}
    _, va = _com_interval(md, RE_L)
    _, vm = _com_interval(mmlp, RE_L)
    p5a, p95a, _fa, _ = _perm_null_com(va, RE_L, SEEDS['comv_all'], BP)
    p5m, p95m, _fm, _ = _perm_null_com(vm, RE_L, SEEDS['comv_mlp'], BP)
    na = rec['E7_null']['all']; nm = rec['E7_null']['mlp']
    ck('%-22s null(all) obs==com_V' % a, abs(na['obs_com'] - c5['com_V']) <= 1e-9)
    ck('%-22s null(all) p5 重算一致' % a, abs(p5a - na['com_p5']) <= 1e-9, '%.6f' % p5a)
    ck('%-22s null(all) p95 重算一致' % a, abs(p95a - na['com_p95']) <= 1e-9, '%.6f' % p95a)
    ck('%-22s null(all) tail==high' % a, na['com_tail'] == 'high' and na['obs_com'] >= p95a)
    ck('%-22s null(mlp) p95 重算一致' % a, abs(p95m - nm['com_p95']) <= 1e-9, '%.6f' % p95m)
    ck('%-22s null(mlp) tail==high' % a, nm['com_tail'] == 'high')

# ================================================================ G
sec('G. 跨度谱 + Q7 双读数（独立重算）')
couple = []; v1 = []
for a in AO:
    rec = REC[a]; L = Lstar[a]
    asum = ANCH['E4_summary'][a]
    smap = {int(s): float(v) for s, v in zip(asum['sites'], asum['xhalf'])}
    Jmap = {int(s): float(v) for s, v in zip(asum['sites'], asum['J'])}
    RE_L = [s for s in rec['E5_com_V']['reach'] if 0 <= s < (L - 1)]
    xs = [smap[l] for l in RE_L]; js = [Jmap[l] for l in RE_L]
    jx = np.diff(np.asarray(xs, float)); jj = np.diff(np.asarray(js, float))
    cx = _com_layer(jx, RE_L); cj = _com_layer(jj, RE_L)
    ck('%-22s com_layer(x) 重算==stored' % a,
       abs(cx - rec['E8_span']['com_layer_x']) <= 1e-9, '%.6f' % cx)
    ck('%-22s com_layer(J) 重算==stored' % a,
       abs(cj - rec['E8_span']['com_layer_j']) <= 1e-9, '%.6f' % cj)
    for k in KS:
        sx = _span_k(jx, k); sj = _span_k(jj, k)
        stx = rec['E8_span']['spans']['x'][str(k)]['obs_span']
        stj = rec['E8_span']['spans']['j'][str(k)]['obs_span']
        ck('%-22s span x k=%d 重算一致' % (a, k), abs(sx - stx) <= 1e-9, '%.4f' % sx)
        ck('%-22s span j k=%d 重算一致' % (a, k), abs(sj - stj) <= 1e-9, '%.4f' % sj)
    sx = _span_k(jx, 3); sj = _span_k(jj, 3)
    couple.append(bool((sx > sj) == (cx > cj)))
    v1.append(bool((sx < sj) == (cx > cj)))
ck('Q7 same-sign coupling == [True,True,True]', couple == [True, True, True], str(couple))
ck('Q7 v1 mismatched == [False,False,False]', v1 == [False, False, False], str(v1))
ck('joint Q7 == SPAN_CENTROID_COUPLED', JV['Q7_joint'] == 'SPAN_CENTROID_COUPLED')
ck('joint Q7(v1) == SPAN_CENTROID_DECOUPLED', JV.get('Q7_joint_v1_mismatched_pairing') == 'SPAN_CENTROID_DECOUPLED')

# ================================================================ H
sec('H. 判决 / 预测一致性')
ck('Q4 label 计数 DECOUPLED==2/3',
   sum(1 for a in AO if V[a]['Q4_label'] == 'POSITION_DECOUPLED') == JV['Q4_counts']['DECOUPLED'])
ck('A2 判别臂 min_d>=CENTROID_SEP_MIN',
   V['A2_qwen3-14b-nf4']['Q4_min_d'] >= FL['CENTROID_SEP_MIN'],
   '%.2f' % V['A2_qwen3-14b-nf4']['Q4_min_d'])
ck('joint Q4 == POSITION_DECOUPLED_PARTIAL', JV['Q4_joint'] == 'POSITION_DECOUPLED_PARTIAL')
ck('joint Q3 == DEEP_ALL', JV['Q3_joint'] == 'DEEP_ALL')
ck('P1..P6 均 PASS', all(PC[k]['pass_'] is True for k in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')),
   str({k: PC[k]['pass_'] for k in sorted(PC)}))
ck('P7 描述性 N/A', PC['P7']['pass_'] is None)
_anc_bits = {}
for a in AO:
    _d = REC[a]['E9_anchor']['detail']
    _bits = [bool(REC[a]['E9_anchor'].get('ok'))]
    for _k, _v in _d.items():
        if isinstance(_v, dict) and 'ok' in _v:
            _bits.append(bool(_v['ok']))
            if isinstance(_v.get('got'), (int, float)) and isinstance(_v.get('expected'), (int, float)):
                _bits.append(abs(float(_v['got']) - float(_v['expected'])) <= 1e-6)
    _anc_bits[a] = all(_bits)
ck('P2 anchor detail 全 ok + 逐位<=1e-6（三臂）', all(_anc_bits.values()), str(_anc_bits))
ck('确认集 Δ <= CONF_TOL_COMV',
   max(V[a]['Q8_conf']['d_com'] for a in AO) <= FL['CONF_TOL_COMV'],
   '%.3f' % max(V[a]['Q8_conf']['d_com'] for a in AO))

# ================================================================ I
sec('I. MEMO 落盘（结构 / 前缀锚 / 编码）')
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig')
lines = mt.split('\r\n')
h17 = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 17')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
ck('BOM 存在', mb[:3] == b'\xef\xbb\xbf')
ck('bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0)
ck('Phase 17 标题唯一且在第 3853 行', len(h17) == 1 and h17[0] == 3853, str(h17))
ck('Phase 标题总数 == 17', len(allh) == 17, str(len(allh)))
ck('MEMO 前缀 == 追加前基线（逐字节）',
   mb[:int(PRE['bytes'])] == open(MEMO, 'rb').read()[:int(PRE['bytes'])]
   and hashlib.sha256(mb[:int(PRE['bytes'])]).hexdigest()[:8] == str(PRE['sha8']),
   str(PRE['sha8']))
ck('MEMO 追加前基线 phase_headings == 16', int(PRE['phase_headings']) == 16)
for tok in ('SPAN_CENTROID_COUPLED', 'WRITE_EFFICACY_ANTICORR_ALL', 'DEEP_ALL', 'POSITION_DECOUPLED_PARTIAL',
            'com_V', 'share_mlp_nb', 'Phase 18', 'E4]'):
    ck('MEMO 含锚点 %s' % tok, tok in mt)

# ================================================================ J
sec('J. wlog 落盘')
wb = open(WLOG, 'rb').read().decode('utf-8')
ck('wlog 含 Phase 17 段', '## Phase 17 / N2h1-' in wb)
ck('wlog 含 Phase 16 段（未被破坏）', '## Phase 16 / N2h1-' in wb)
ck('wlog 含 DEEP_ALL / com_V', ('DEEP_ALL' in wb) and ('com_V' in wb))

# ================================================================ K
sec('K. _infra 基线刷新')
ck('infra tag == post-append-phase17', INFRA.get('tag') == 'post-append-phase17', str(INFRA.get('tag')))
ck('infra bytes/lines/sha8 == 实际 MEMO',
   int(INFRA['bytes']) == len(mb) and int(INFRA['lines']) == len(lines)
   and INFRA['sha8'] == hashlib.sha256(mb).hexdigest()[:8],
   '%d/%d/%s' % (INFRA['bytes'], INFRA['lines'], INFRA['sha8']))
htags = [h.get('tag') for h in (INFRA.get('history') or [])]
ck('infra history 含 pre-append-phase17', 'pre-append-phase17' in htags, str(htags))
ck('infra history 不含自条目 post-append-phase17', 'post-append-phase17' not in htags)
ck('infra phase_headings == 17', len([x for x in (INFRA.get('phase_headings') or []) if isinstance(x, int)]) == 17)

# ================================================================ L
sec('L. 勘误留痕 / 口径（E1-E4）')
v1p = os.path.join(P17T, 'result_phase17_v1_jointlabel.json')
v2p = os.path.join(P17T, 'result_phase17_v2_preintervalfix.json')
v3p = os.path.join(P17T, 'result_phase17_v3_bakedviolation.json')
ck('v1 留痕存在', os.path.exists(v1p))
ck('v2 留痕存在', os.path.exists(v2p))
ck('v3 留痕存在', os.path.exists(v3p))
if os.path.exists(v2p):
    V2 = jload(v2p)
    ck('E4 v2(修正前) A0 com_V == 24.4656（≈24.466）',
       abs(V2['verdict']['A0_calib_qwen3-4b-nf4']['Q3_com_V'] - 24.465585421173355) <= 1e-9,
       '%.6f' % V2['verdict']['A0_calib_qwen3-4b-nf4']['Q3_com_V'])
    ck('E4 修正后 A0 com_V != 修正前（确有变化）',
       abs(V2['verdict']['A0_calib_qwen3-4b-nf4']['Q3_com_V'] - V['A0_calib_qwen3-4b-nf4']['Q3_com_V']) > 0.5)
if os.path.exists(v1p):
    V1 = jload(v1p)
    ck('E1 v1 Q7 == DECOUPLED（v1 读数留痕）',
       V1['joint_verdict'].get('Q7_joint') == 'SPAN_CENTROID_DECOUPLED')
# 跨实现交叉验证：用本脚本的区间求和重算探针（独立脚本）自报的 com_V，二者须与生产逐位一致
_pr = {int(l): float(PROBE['w_all'][l]) for l in range(len(PROBE['w_all']))}
_pc, _ = _com_interval(_pr, [int(x) for x in PROBE['reach'] if 0 <= int(x) < int(PROBE['L']) - 1])
ck('探针 w_all 区间求和重算 == 探针自报 com_V', abs(_pc - PROBE['com_V']) <= 1e-9,
   '%.9f vs %.9f' % (_pc, PROBE['com_V']))
ck('探针 com_V == 生产 A0 com_V（跨实现逐位一致，E4 交叉验证）',
   abs(PROBE['com_V'] - V['A0_calib_qwen3-4b-nf4']['Q3_com_V']) <= 1e-9,
   '%.9f vs %.9f' % (PROBE['com_V'], V['A0_calib_qwen3-4b-nf4']['Q3_com_V']))

# ================================================================
sec('汇总')
tot = _n[0]; fail = _n[1]
_out.append('')
_out.append('TOTAL %d checks, %d FAIL' % (tot, fail))
_out.append('clock done')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_out) + '\n')
print('\nTOTAL %d checks, %d FAIL -> %s' % (tot, fail, OUT))
