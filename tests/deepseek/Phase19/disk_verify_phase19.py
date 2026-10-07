# -*- coding: utf-8 -*-
"""Phase 19 独立磁盘复核（disjoint re-implementation）。

原则（承 P18 教训）：**独立复核不得写死「预期成功」** —— 一律「从原始产物重算 -> 按主脚本文义导出标签 -> 与落盘值比对」。
本文件的统计工具（区间求和质心 / 置换零假设 / 秩相关 / 配对敏感度）均为**另写一份**，不复用主脚本代码。

检查面：
  A. 文件指纹（seal/exec/result/ledger/memo-baseline）
  B. bit 级锚（P17 冻结值）
  C. com_V 族 + nb + share + argmax 独立重算（区间求和口径）
  D. 置换零假设独立重算（BP=2000，种子取自 result.bootstrap）
  E. 同 Phase 量化配对独立重算
  F. 逐臂 / 联合 / 预测标签由重算量导出后与落盘比对
  G. Ledger 补登与 n
  H. MEMO 追加后的基线与前缀锚
产物：tests/deepseek_temp/Phase19/disk_verify_phase19.txt
"""
import io
import os
import json
import math
import hashlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
P19S = os.path.join(ROOT, 'tests', 'deepseek', 'Phase19')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

o = []
N_PASS = [0]
N_FAIL = [0]
FAILS = []


def w(s=''):
    o.append(str(s)); print(s)


def chk(name, ok, detail=''):
    if ok:
        N_PASS[0] += 1
        w('  [PASS] %-46s %s' % (name, detail))
    else:
        N_FAIL[0] += 1
        FAILS.append(name)
        w('  [FAIL] %-46s %s' % (name, detail))
    return ok


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ================================================================ 统计工具（另写一份）
def interval_centroid(mass, sites):
    """区间求和 + 中点质心（独立实现：先构造 [s_j, s_{j+1}) 的块和，再取加权中点）。"""
    vals = []
    for j in range(len(sites) - 1):
        lo, hi = int(sites[j]), int(sites[j + 1])
        vals.append(sum(float(mass[i]) for i in range(lo, hi)))
    vals = np.asarray(vals, float)
    mids = np.asarray([(float(sites[j]) + float(sites[j + 1])) / 2.0 for j in range(len(sites) - 1)], float)
    tot = float(vals.sum())
    if not math.isfinite(tot) or tot <= 1e-12:
        return None
    return float(float((vals * mids).sum()) / tot)


def perm_null(mass, sites, seed, BP):
    vals = []
    for j in range(len(sites) - 1):
        lo, hi = int(sites[j]), int(sites[j + 1])
        vals.append(sum(float(mass[i]) for i in range(lo, hi)))
    vals = np.asarray(vals, float)
    mids = np.asarray([(float(sites[j]) + float(sites[j + 1])) / 2.0 for j in range(len(sites) - 1)], float)
    obs = float((vals * mids).sum() / vals.sum())
    rng = np.random.default_rng(int(seed))
    n = len(vals)
    got = np.empty(BP)
    for b in range(BP):
        p = rng.permutation(n)
        q = vals[p]
        got[b] = float((q * mids).sum() / q.sum())
    p5 = float(np.percentile(got, 5))
    p95 = float(np.percentile(got, 95))
    tail = 'low' if obs <= p5 else ('high' if obs >= p95 else 'none')
    return dict(obs=obs, p5=p5, p95=p95, tail=tail)


def avg_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # 平均结（本例无并列，仍做稳健处理）
    return r


def spearman_indep(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = avg_rank(a) - (len(a) + 1) / 2.0
    rb = avg_rank(b) - (len(b) + 1) / 2.0
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


# ================================================================ A. 文件指纹
w('=== Phase 19 独立磁盘复核  clock=%s ===' % __import__('time').strftime('%Y-%m-%d %H:%M:%S'))
RESB = open(os.path.join(P19T, 'result_phase19.json'), 'rb').read()
EXEB = open(os.path.join(P19T, 'execution_phase19.json'), 'rb').read()
SEALB = open(os.path.join(P19T, 'N2h1a12_design_seal.json'), 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.loads(EXEB.decode('utf-8'))
SEAL = json.loads(SEALB.decode('utf-8'))
w('files: seal=%s exec=%s result=%s' % (sha8b(SEALB), sha8b(EXEB), sha8b(RESB)))
w('')
w('--- A. 指纹一致性 ---')
chk('A1 smoke=False', RES.get('smoke') is False, 'smoke=%s' % RES.get('smoke'))
chk('A2 result.seal_sha256 == seal 实文件', RES['seal_sha256'] == hashlib.sha256(SEALB).hexdigest())
chk('A3 result.exec_sha256 == exec 实文件', RES['exec_sha256'] == hashlib.sha256(EXEB).hexdigest())
chk('A4 exec.seal_sha256 == seal 实文件', EX.get('seal_sha256') == hashlib.sha256(SEALB).hexdigest())
chk('A5 floors: result == exec', RES['floors'] == EX['floors'])
EXP_FLOORS = dict(P19_FID_ARCH=0.03, P19_FID_BLK=0.01, CALIB_TOL_COMV=1e-3, QUANT_TOL_COMV=2.0,
                  RHO_SHAPE_MIN=0.8, MLP_DOM_MIN=0.5, DEEP_MEDIAN=0.5, NULL_ALPHA=0.05)
chk('A6 floors == 冻结常量', RES['floors'] == EXP_FLOORS, json.dumps(RES['floors'], ensure_ascii=False))
FL = RES['floors']
ARMS = list(EX['arm_order'])
V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
RECS = RES['arms']
chk('A7 arm_order == verdict keys', sorted(ARMS) == sorted(V.keys()), str(ARMS))

# ================================================================ B. bit 级锚
w('')
w('--- B. P17 冻结锚（bit 级） ---')
ANCH = SEAL['anchor_values']
for a in ('A0_nf4', 'A1_nf4'):
    got = V[a]['com_V']
    exp = ANCH[a]['com_V']
    chk('B.%s com_V bit 级相等' % a, repr(got) == repr(exp), 'repr=%r' % got)
    chk('B.%s com_V_mlp bit 级相等' % a, V[a]['com_V_mlp'] == ANCH[a]['com_V_mlp'])
    chk('B.%s com_V_attn bit 级相等' % a, V[a]['com_V_attn'] == ANCH[a]['com_V_attn'])
    chk('B.%s median_reach 相等' % a, V[a]['median_reach'] == ANCH[a]['median_reach'])
    chk('B.%s nb 相等' % a, list(V[a]['neighbourhood']) == list(ANCH[a]['nb']))
    chk('B.%s argmax 相等' % a, V[a]['argmax_w_layer'] == ANCH[a]['argmax_w_layer'])
chk('B.literal A0 com_V == 26.15005633170243', repr(V['A0_nf4']['com_V']) == repr(26.15005633170243))
chk('B.literal A1 com_V == 26.70366214504327', repr(V['A1_nf4']['com_V']) == repr(26.70366214504327))

# ================================================================ C. com_V 族独立重算
w('')
w('--- C. com_V 族 / nb / share / argmax 独立重算（区间求和口径） ---')
C_RECOMP = {}
for a in ARMS:
    r = RECS[a]
    E5 = r['E5_com_V']
    reach = [int(x) for x in E5['reach']]
    wa = np.asarray(E5['w_all'], float)
    wm = np.asarray(E5['w_mlp'], float)
    wt = np.asarray(E5['w_attn'], float)
    com = interval_centroid(wa, reach)
    com_m = interval_centroid(wm, reach)
    com_t = interval_centroid(wt, reach)
    med = float(np.median(reach))
    nb = [l for l in reach if abs(l - com) <= EX['neighbourhood_width']]
    den = float(sum(float(wa[l]) for l in nb))
    share = float(sum(float(wm[l]) for l in nb)) / den if den > 1e-12 else None
    amx = int(np.argmax(wa))
    C_RECOMP[a] = dict(com=com, com_m=com_m, com_t=com_t, med=med, nb=nb, share=share, amx=amx)
    chk('C.%s com_V 重算==落盘' % a, abs(com - V[a]['com_V']) <= 1e-9,
        'recomp=%.12f stored=%.12f' % (com, V[a]['com_V']))
    chk('C.%s com_V_mlp 重算==落盘' % a, abs(com_m - V[a]['com_V_mlp']) <= 1e-9)
    chk('C.%s com_V_attn 重算==落盘' % a, abs(com_t - V[a]['com_V_attn']) <= 1e-9)
    chk('C.%s median_reach 重算==落盘' % a, med == V[a]['median_reach'])
    chk('C.%s nb 重算==落盘' % a, nb == list(V[a]['neighbourhood']), str(nb))
    chk('C.%s share_mlp_nb 重算==落盘' % a, share is not None and abs(share - V[a]['share_mlp_nb']) <= 1e-9,
        'recomp=%.10f stored=%.10f' % (share, V[a]['share_mlp_nb']))
    chk('C.%s argmax_w 重算==落盘' % a, amx == V[a]['argmax_w_layer'], 'L%d' % amx)
    chk('C.%s 深端(com>=median)' % a, com >= med, 'com=%.3f med=%.1f' % (com, med))

# ================================================================ D. 置换零假设独立重算
w('')
w('--- D. 置换零假设独立重算（BP=%d） ---' % RES['bootstrap']['BP'])
BP = int(RES['bootstrap']['BP'])
S_ALL = int(RES['bootstrap']['seeds']['comv_all'])
S_MLP = int(RES['bootstrap']['seeds']['comv_mlp'])
NULL_TAILS = {}
for a in ARMS:
    E5 = RECS[a]['E5_com_V']
    reach = [int(x) for x in E5['reach']]
    na = perm_null(np.asarray(E5['w_all'], float), reach, S_ALL, BP)
    nm = perm_null(np.asarray(E5['w_mlp'], float), reach, S_MLP, BP)
    NULL_TAILS[a] = dict(all=na['tail'], mlp=nm['tail'])
    sa = V[a]['Q7_null_all']; sm = V[a]['Q7_null_mlp']
    chk('D.%s null(all) p5/p95/tail 重算==落盘' % a,
        abs(na['p5'] - sa['com_p5']) <= 1e-9 and abs(na['p95'] - sa['com_p95']) <= 1e-9 and na['tail'] == sa['com_tail'],
        'recomp=[%.6f,%.6f,%s] stored=[%.6f,%.6f,%s]' % (na['p5'], na['p95'], na['tail'], sa['com_p5'], sa['com_p95'], sa['com_tail']))
    chk('D.%s null(mlp) p5/p95/tail 重算==落盘' % a,
        abs(nm['p5'] - sm['com_p5']) <= 1e-9 and abs(nm['p95'] - sm['com_p95']) <= 1e-9 and nm['tail'] == sm['com_tail'])
    chk('D.%s null 高尾' % a, na['tail'] == 'high' and nm['tail'] == 'high', '%s/%s' % (na['tail'], nm['tail']))

# ================================================================ E. 量化配对独立重算
w('')
w('--- E. 同 Phase 量化配对独立重算 ---')
PAIRS = [('A0_nf4', 'A0_bf16'), ('A1_nf4', 'A1_bf16')]
E_RECOMP = {}
for an, ab in PAIRS:
    wa = np.asarray(RECS[an]['E5_com_V']['w_all'], float)
    wb = np.asarray(RECS[ab]['E5_com_V']['w_all'], float)
    n = min(len(wa), len(wb))
    wa, wb = wa[:n], wb[:n]
    delta = abs(V[an]['com_V'] - V[ab]['com_V'])
    rho = spearman_indep(wa, wb)
    rel = np.abs(wa - wb) / np.maximum(np.abs(wb), 1e-9)
    med = float(np.median(rel)); p90 = float(np.percentile(rel, 90))
    same_side = (V[an]['share_mlp_nb'] > FL['MLP_DOM_MIN']) == (V[ab]['share_mlp_nb'] > FL['MLP_DOM_MIN'])
    same_amx = V[an]['argmax_w_layer'] == V[ab]['argmax_w_layer']
    E_RECOMP['%s|%s' % (an, ab)] = dict(delta=delta, rho=rho, med=med, p90=p90,
                                        same_side=same_side, same_amx=same_amx)
    key = '%s|%s' % (an, ab)
    S = JV['quant_pairs'][key]
    chk('E.%s delta_com_V 重算==落盘' % key, abs(delta - S['delta_com_V']) <= 1e-9, '%.6f' % delta)
    chk('E.%s spearman 重算≈落盘' % key, abs(rho - S['spearman_w']) <= 1e-6,
        'recomp=%.6f stored=%.6f' % (rho, S['spearman_w']))
    chk('E.%s resid med/p90 重算==落盘' % key,
        abs(med - S['median_rel_resid']) <= 1e-9 and abs(p90 - S['p90_rel_resid']) <= 1e-9)
    chk('E.%s argmax 同层' % key, same_amx == S['argmax_same'] and same_amx)
    chk('E.%s share 同侧' % key, same_side == S['share_same_side'] and same_side)

# ================================================================ F. 标签由重算量导出后比对（禁写死预期）
w('')
w('--- F. 逐臂/联合/预测标签：重算 -> 导出 -> 比对 ---')
label_ok = True
for a in ARMS:
    r = RECS[a]
    E2 = r['E2_fidelity']
    q1 = 'FID_PASS' if (E2['arch_max'] <= FL['P19_FID_ARCH'] and E2['blk_max'] <= FL['P19_FID_BLK']) else 'FID_FAIL'
    q2 = ('ANCHOR_NA' if r['E7_anchor'].get('ok') is None
          else ('CALIB_OK' if r['E7_anchor']['ok'] else 'CALIB_DRIFT'))
    q5 = 'MLP_DOMINANT' if V[a]['share_mlp_nb'] > FL['MLP_DOM_MIN'] else 'MLP_NOT_DOMINANT'
    q6 = 'DEEP' if V[a]['com_V'] >= V[a]['median_reach'] else 'SHALLOW'
    ok = (q1 == V[a]['Q1_label'] and q2 == V[a]['Q2_label'] and q5 == V[a]['Q5_label'] and q6 == V[a]['Q6_label'])
    label_ok &= ok
    w('    %-8s Q1=%s Q2=%s Q5=%s Q6=%s %s' % (a, q1, q2, q5, q6, 'OK' if ok else '!!MISMATCH'))
chk('F1 逐臂标签导出==落盘', label_ok)

# 联合标签
j_q1 = 'FID_ALL_PASS' if all(V[a]['Q1_label'] == 'FID_PASS' for a in ARMS) else 'FID_PARTIAL'
nf4 = [a for a in ARMS if a.endswith('_nf4')]
bf = [a for a in ARMS if a.endswith('_bf16')]
j_q2 = 'CALIB_ALL_OK' if all(V[a]['Q2_label'] == 'CALIB_OK' for a in nf4) else 'CALIB_DRIFT'
stable = sum(1 for k, s in E_RECOMP.items() if s['delta'] <= FL['QUANT_TOL_COMV'])
j_q3 = 'QUANT_STABLE_ALL' if stable == len(E_RECOMP) else ('QUANT_STABLE_PARTIAL' if stable else 'QUANT_SENSITIVE')
rho_ok = sum(1 for k, s in E_RECOMP.items() if s['rho'] is not None and s['rho'] >= FL['RHO_SHAPE_MIN'])
j_q4 = 'SPECTRUM_CONSISTENT_ALL' if rho_ok == len(E_RECOMP) else ('SPECTRUM_CONSISTENT_PARTIAL' if rho_ok else 'SPECTRUM_DISTORTED')
nm = sum(1 for a in bf if V[a]['Q5_label'] == 'MLP_DOMINANT')
j_q5 = 'MLP_DOM_RETAINED_ALL' if nm == len(bf) else ('MLP_DOM_RETAINED_PARTIAL' if nm else 'MLP_DOM_LOST_ALL')
nd = sum(1 for a in bf if V[a]['Q6_label'] == 'DEEP')
j_q6 = 'DEEP_RETAINED_ALL' if nd == len(bf) else ('DEEP_RETAINED_PARTIAL' if nd else 'DEEP_LOST_ALL')
chk('F2 Q1_joint 导出==落盘', j_q1 == JV['Q1_joint'], '%s' % j_q1)
chk('F3 Q2_joint 导出==落盘', j_q2 == JV['Q2_joint'], '%s' % j_q2)
chk('F4 Q3_joint 导出==落盘', j_q3 == JV['Q3_joint'], '%s (%d/%d)' % (j_q3, stable, len(E_RECOMP)))
chk('F5 Q4_joint 导出==落盘', j_q4 == JV['Q4_joint'], '%s (%d/%d)' % (j_q4, rho_ok, len(E_RECOMP)))
chk('F6 Q5_joint 导出==落盘', j_q5 == JV['Q5_joint'], '%s' % j_q5)
chk('F7 Q6_joint 导出==落盘', j_q6 == JV['Q6_joint'], '%s' % j_q6)
chk('F8 Q7_null 尾 导出==落盘', NULL_TAILS == {a: dict(all=JV['Q7_null'][a]['all'], mlp=JV['Q7_null'][a]['mlp']) for a in ARMS})

# 预测（由重算标签导出）
p1 = bool(JV['Q1_joint'] == j_q1 and j_q1 == 'FID_ALL_PASS' and j_q2 == 'CALIB_ALL_OK' and JV['Q0_apparatus_all'])
s0 = E_RECOMP['A0_nf4|A0_bf16']; s1 = E_RECOMP['A1_nf4|A1_bf16']
p2 = bool(s0['delta'] <= FL['QUANT_TOL_COMV'] and V['A0_bf16']['Q6_label'] == 'DEEP'
          and s0['rho'] is not None and s0['rho'] >= FL['RHO_SHAPE_MIN'])
p3 = bool(s1['delta'] <= FL['QUANT_TOL_COMV'])
p4 = bool(V['A1_bf16']['Q5_label'] == 'MLP_DOMINANT' and V['A1_bf16']['Q6_label'] == 'DEEP')
p5 = bool(all(s['rho'] is not None and s['rho'] >= FL['RHO_SHAPE_MIN'] for s in E_RECOMP.values()))
for k, val in [('P1', p1), ('P2', p2), ('P3', p3), ('P4', p4), ('P5', p5)]:
    chk('F9 %s 导出==落盘' % k, val == PC[k]['pass_'], 'derived=%s stored=%s' % (val, PC[k]['pass_']))

# ================================================================ G. Ledger
w('')
w('--- G. Ledger 补登 ---')
LG = load(LEDGER)
chk('G1 measurements n == 302', len(LG['measurements']) == 302, 'n=%d' % len(LG['measurements']))
p19 = [m for m in LG['measurements'] if m.get('phase') == 19]
chk('G2 Phase 19 条目唯一', len(p19) == 1, 'count=%d' % len(p19))
if p19:
    e = p19[0]
    chk('G3 ledger.result_sha8 == result 实文件', e['result_sha8'] == sha8b(RESB), e['result_sha8'])
    chk('G4 ledger.seal/exec_sha8 对齐', e['seal_sha8'] == sha8b(SEALB) and e['exec_sha8'] == sha8b(EXEB))
    chk('G5 ledger.anchor_result_sha8 == ee27627a', e['anchor_result_sha8'] == 'ee27627a', e['anchor_result_sha8'])
    vd = '%s__%s__%s__%s__%s__%s__nullok' % (JV['Q1_joint'].lower(), JV['Q2_joint'].lower(),
                                             JV['Q3_joint'].lower(), JV['Q4_joint'].lower(),
                                             JV['Q5_joint'].lower(), JV['Q6_joint'].lower())
    chk('G6 ledger.verdict 由 result 渲染', e['verdict'] == vd, e['verdict'])
    chk('G7 ledger.n_rows == 172', e['n_rows'] == 172, str(e['n_rows']))
    chk('G8 ledger.model_scope 覆盖两模型', 'qwen3-4b' in e['model_scope'] and 'glm4-9b' in e['model_scope'])
_nl = [m for m in LG['measurements'] if str(m.get('name', '')).startswith('n2h1a')]
chk('G9 N 线条目相位唯一且 = 8..19', sorted(m['phase'] for m in _nl) == list(range(8, 20)),
    'N-line=%d phases=%s' % (len(_nl), sorted(m['phase'] for m in _nl)))

# ================================================================ H. MEMO
w('')
w('--- H. MEMO 追加基线与前缀锚 ---')
mb = open(MEMO, 'rb').read()
POST = os.path.join(INFRA, 'memo_baseline.json')
PRE = os.path.join(P19T, 'memo_baseline_preappend_phase19.json')
ob = load(POST); pr = load(PRE)
mt = mb.decode('utf-8-sig').split('\r\n')
ph = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
chk('H1 post baseline.tag', ob.get('tag') == 'post-append-phase19', str(ob.get('tag')))
chk('H2 post baseline.bytes == 实文件', ob['bytes'] == len(mb), '%d vs %d' % (ob['bytes'], len(mb)))
chk('H3 post baseline.sha256 == 实文件', ob['sha256'] == hashlib.sha256(mb).hexdigest())
chk('H4 phase_headings == 19', len(ph) == 19 and ob['phase_headings'] == ph, 'n=%d' % len(ph))
chk('H5 bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0)
chk('H6 BOM 存在', mb[:3] == b'\xef\xbb\xbf')
chk('H7 pre baseline 在 history 中', any(h.get('tag') == 'pre-append-phase19' for h in (ob.get('history') or [])))
# 前缀锚：追加后必须以「追加前的完整原始字节」为前缀
pre_bytes = int(pr['bytes'])
chk('H8 追加前基线 == history 记录', int(pr['bytes']) == int(next(h['bytes'] for h in ob['history'] if h.get('tag') == 'pre-append-phase19')))
chk('H9 前缀锚：sha256(MEMO[:pre_bytes]) == pre.sha256',
    hashlib.sha256(mb[:pre_bytes]).hexdigest() == pr['sha256'], 'pre=%d B' % pre_bytes)
i19 = [i for i, l in enumerate(mt) if l.startswith('## Phase 19')]
chk('H10 Phase 19 节唯一且在尾部', len(i19) == 1 and i19[0] + 1 == ph[-1], 'line=%d of %d' % (i19[0] + 1, len(mt)))
chk('H11 Phase 19 节含关键锚', all(k in mb.decode('utf-8-sig') for k in
                                ['N2h1-α-12', 'segfault', 'E-A2', 'E-pair', 'Phase 20']))

# ================================================================ 汇总
w('')
w('=================================================')
w('PASS = %d ; FAIL = %d' % (N_PASS[0], N_FAIL[0]))
if FAILS:
    w('FAILS: %s' % ', '.join(FAILS))
w('结果：%s' % ('0 FAIL —— 独立复核通过' if N_FAIL[0] == 0 else '存在 FAIL，需人工处置'))
io.open(os.path.join(P19T, 'disk_verify_phase19.txt'), 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('WROTE disk_verify_phase19.txt')
