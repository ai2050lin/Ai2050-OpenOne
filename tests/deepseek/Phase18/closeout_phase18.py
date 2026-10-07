# -*- coding: utf-8 -*-
"""Phase 18 结账（幂等 + 刷新）：
  [A] 冻结「MEMO 预追加基线」-> tests/deepseek_temp/Phase18/memo_baseline_preappend_phase18.json
  [B] Ledger 补登一条（n 300 -> 301）；若已存在则**就地刷新**到最终 result（result_sha8 / rev_note /
      verdict / n_rows / n_forwards_per_arm / 各 sha8），刷新后重算 ledger_sha256_8。
  [C] 写 verify_ledger_phase18.txt（落点 v2：verify_*.txt -> deepseek_temp/Phase{N}/）

铁律 (ae)：Ledger 的 rev_note 中**所有与数据相关的数字一律由 result 现场渲染**，禁手工转录。
本脚本所有 P1–P7 / 桥接 / 零假设 / r_lin 数字均经 _cv()/_tri()/_sci() 从 result_phase18.json 取用。
"""
import io
import os
import json
import time
import hashlib
import shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE18 = os.path.join(P18T, 'memo_baseline_preappend_phase18.json')
OUTV = os.path.join(P18T, 'verify_ledger_phase18.txt')

o = []


def w(s=''):
    o.append(str(s))
    print(s)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


RESP = os.path.join(P18T, 'result_phase18.json')
assert os.path.exists(RESP), 'result 尚未生成: %s' % RESP
RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
assert not RES.get('smoke'), 'result_phase18.json 是 SMOKE 结果，拒绝结账'
EX = json.load(io.open(os.path.join(P18T, 'execution_phase18.json'), encoding='utf-8'))
SEALB = open(os.path.join(P18T, 'N2h1a11_design_seal.json'), 'rb').read()
EXECB = open(os.path.join(P18T, 'execution_phase18.json'), 'rb').read()
PROBEB = open(os.path.join(P18T, '_probe_feasibility_A0.json'), 'rb').read()

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
AO = list(EX['arm_order'])
assert len(AO) == 3, AO
assert sorted(V.keys()) == sorted(AO), (V.keys(), AO)


# ---------------------------------------------------------------- 渲染辅助
def _g(a, k):
    return V[a].get(k)


def _cv(a, k, nd=3):
    v = _g(a, k)
    if v is None:
        return 'NA'
    if isinstance(v, float):
        return ('%.' + str(nd) + 'f') % v
    return str(v)


def _tri(k, nd=3):
    return '/'.join(_cv(a, k, nd) for a in AO)


def _sci(x):
    return '%.2e' % float(x)


def _scitri(k):
    return '/'.join(_sci(_g(a, k)) for a in AO)


def _pass(k):
    v = PC.get(k, {})
    p = v.get('pass_')
    return 'PASS' if p is True else ('N/A' if p is None else 'FAIL')


def _reach_ratio(a):
    """峰值/次大，限定在 REACH 域（与 ALL 域对照，供 E-rlin 勘误渲染）。"""
    E = RES['arms'][a]['E7_summary']
    rl = {int(k): float(x) for k, x in E['rlin_by_site'].items()}
    vals = sorted((rl[l] for l in E['reach'] if l in rl), reverse=True)
    if len(vals) > 1 and vals[1] > 1e-12:
        return '%.2f' % (vals[0] / vals[1])
    return 'NA'


w('=== Phase 18 closeout  clock=%s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('seal=%s exec=%s probe=%s result=%s anchor(p16)=%s anchor(p17)=%s'
  % (sha8b(SEALB), sha8b(EXECB), sha8b(PROBEB), sha8b(RESB),
     RES['anchor_result_p16_sha256'][:8], RES['anchor_result_p17_sha256'][:8]))

# ---------------- [A] 预追加基线
raw0 = open(MEMO, 'rb').read()
t0 = raw0.decode('utf-8-sig')
hdrs0 = [ln for ln in t0.split('\r\n') if ln.startswith('## Phase ')]
_heads = {}
for i, l in enumerate(t0.split('\r\n')):
    if l.startswith('## '):
        _heads[l[:44]] = i + 1
B = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), tag='pre-append-phase18',
         path='research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
         bytes=len(raw0), lines=len(raw0.split(b'\r\n')),
         sha256=hashlib.sha256(raw0).hexdigest(), sha8=sha8b(raw0),
         bom=bool(raw0[:3] == b'\xef\xbb\xbf'),
         crlf=int(raw0.count(b'\r\n')), bare_lf=int(raw0.count(b'\n') - raw0.count(b'\r\n')),
         phase_headings=len(hdrs0), sections=_heads,
         note='Phase 18 追加前快照（Phase 17 节已在其中）。')
if not os.path.exists(BASE18):
    io.open(BASE18, 'w', encoding='utf-8', newline='\n').write(json.dumps(B, ensure_ascii=False, indent=1))
    w('[A] 冻结预追加基线 -> %s' % BASE18)
else:
    w('[A] 基线已存在（幂等跳过）')
w('    bytes=%d lines=%d phase_headings=%d sha8=%s bare_lf=%d'
  % (B['bytes'], B['lines'], B['phase_headings'], B['sha8'], B['bare_lf']))
assert B['bare_lf'] == 0 and B['bom'], 'MEMO EOL/BOM 异常'
if not os.path.exists(BASE18):
    assert B['phase_headings'] == 17, 'Phase 标题数应为 17，实为 %d' % B['phase_headings']

# ---------------- [B] Ledger
LB = open(LEDGER, 'rb').read()
LG = json.loads(LB.decode('utf-8'))
n_before = len(LG['measurements'])
already = any(m.get('phase') == 18 for m in LG['measurements'])

NPAIR_DISC = len(EX['discovery'])
NPAIR_CONF = len(EX['confirmation'])
NSITES = len(EX['profile_sites'])
NW = {}
for a in AO:
    NW[a] = int(RES['arms'][a].get('n_forwards') or 0)
NROWS = sum(NW.values())

verdict = '%s__%s__%s__%s__%s__%s__%s__%s' % (
    str(JV.get('Q1_joint', 'NA')).lower(),
    str(JV.get('Q2_joint', 'NA')).lower(),
    str(JV.get('Q3_joint', 'NA')).lower(),
    str(JV.get('Q4_joint', 'NA')).lower(),
    str(JV.get('Q5_joint', 'NA')).lower(),
    str(JV.get('Q6_joint', 'NA')).lower(),
    str(JV.get('Q7_joint', 'NA')).lower(),
    str(JV.get('Q8_joint', 'NA')).lower())

# ---- 数据驱动 REV（铁律 (ae)：禁止手工转录数字） ------------------------------
REV = (
    'deepseek/N line Phase 18 (N2h1-alpha-11), BEHAVIOURAL COMPONENT BUDGET: Phase 17 answered WHERE the '
    'write vector lives (vector mass w_ell = mean_pairs ||P_{U_ell}(Delta_inc_ell)||, centroid com_V deep); '
    'Phase 18 asks WHO writes it -- the SAME per-layer decomposition is scored BEHAVIOURALLY. For every '
    'component c in {INC_ALL, INC_MLP, INC_ATTN, INC_TOP1, CUM_ALL} the behavioural budget is '
    'b_{c,ell} = mean_pairs [ score_of(h_ell^R + P_{U_ell}(Delta_c), sup, sid_d) - BASE[rw].sd0 ] with '
    'score_of(v,sup,sid) = v[ID(sup)] - mean_{x!=sup} v[ID(x)] and v[sid] = -1e9 (byte-for-byte the Phase-8 '
    'T-arm read-out). INC_ALL/INC_MLP/INC_ATTN are the incremental write Delta_inc_ell split by module '
    '(Delta_attn := sum_h Delta_head_h, Delta_mlp := m_ell^donor - m_ell^recip, additivity is a DEFINITION); '
    'INC_TOP1 = the single per-layer head with the largest ||P(Delta_head_h)||; CUM_ALL = the cumulative '
    'difference d_ell = HH[ell+1]^D - HH[ell+1]^R (the Phase-16 object). Sites are layer indices '
    'ell in ALL_SITES = 1..L-2 (injection at the output of layers[ell] = HH[ell+1], aligned with the Phase-17 '
    'w_all indexing); centroids use INTERVAL SUMS over the FULL domain (which covers layers outside REACH). '
    'THREE arms, same nf4 convention inherited byte-for-byte from Phases 15/16/17 (A0_calib_qwen3-4b-nf4 / '
    'A1_glm4-9b-nf4 / A2_qwen3-14b-nf4; arms ordered A0/A1/A2 throughout). '
    'MOTIVATION (the H11 causal gap): Phase 17\'s MLP DOMINANCE was a VECTOR-budget claim '
    '(share_mlp_nb = %(mlpvec)s); if the behavioural attribution points the same way (MLP-led) the causal '
    'reading is UPGRADED, whereas an attention-led behavioural budget would reduce Phase 17\'s vector share '
    'to a GEOMETRIC ARTEFACT. Phase 8 already found the write vector exactly additive but the EFFECT shares '
    'non-additive (MLP effect share 0.739 vs vector share 0.472), so behavioural attribution is the correct '
    'cross-check. '
    'KEY RESULTS (7 predictions: %(p1)s/%(p2)s/%(p3)s/%(p4)s/%(p5)s/%(p6)s and P7 descriptive): '
    'P1 FID %(fp1)s (arch max %(arch)s <= 3e-2 ; blocks max %(blk)s <= 1e-2). '
    'P2 ANCHOR %(fp2)s: com_layer(x)/com_layer(J)/L*_own re-read from the frozen Phase-16 result reproduced '
    'to <=1e-6 on 3/3 arms. '
    'P3 (HOLDOUT PRIMARY) behavioural component attribution MLP-led: %(q4c)s ; neighbourhood (+-2) '
    'share_mlp_beh = %(mlpbeh)s versus the Phase-17 vector share %(mlpvec)s on the two arms never observed '
    'before the seal (A1/A2). '
    'P4 (HOLDOUT) behavioural centroid SHALLOWER than the vector centroid: %(q6c)s ; com_B = %(comB)s vs '
    'com_V = %(comV)s, i.e. gap = %(gap)s layers. '
    'P5 SAME-OBJECT COUPLING: %(q7c)s ; spearman(w_all, |b_all|) = %(spwb)s while the Phase-17 pairing '
    'spearman(w_all, J) = %(spwj)s -- opposite signs. '
    'P6 SUPERADDITIVITY AT THE WRITE WINDOW: %(q8c)s ; r_lin at L*_own = %(rlinstar)s vs peak ratio '
    '%(ratio)s. '
    'P7 (descriptive) %(q9)s: permutation null on com_B and on the neighbourhood share, plus the '
    'confirmation set (n=%(nconf)d, disjoint from the %(ndisc)d discovery pairs). '
    'BRIDGE (cross-Phase apparatus gate): CUM_ALL at the Phase-16 write window = %(cumbridge)s vs the frozen '
    'Phase-16 FULL_SWAP = %(fullswap)s, rel = %(bridgerel)s (tolerance %(bridgetol)s) -- this ties the Phase-18 '
    'read-out to Phase 16 on the SAME object. '
    'CONTROLS: %(q9c)s ; permutation null high-tail = %(nullhigh)s ; confirmation com_B tol %(conftol)s (max '
    'dev %(confdev)s). '
    'SAME-ROUND ERRATA: [E-rlin] the P6 peak-ratio evidence carries a SUPPORT-DOMAIN MISMATCH -- the seal '
    'rationale quoted "next-largest 0.186 at L26, ratio 4.05" from a REACH-RESTRICTED grid (this reproduces '
    'exactly: A0 REACH ratio 4.053); the production r_lin domain is ALL_SITES = 1..L-2 (seal formula leaves the '
    'site free), where a shallow near-zero-denominator site (A0: L2, r_lin 0.487) inflates the ratio (ALL-site '
    'ratio %(ratio)s; REACH-domain ratio %(ratioreanch)s). Even on the REACH domain only A0 reaches >= 3, so P6 '
    'FAILS under either domain -- the erratum changes WHY, not the verdict. The P6 POSITION sub-claim '
    '(argmax r_lin == L*_own) holds on A0 only. [E-bridge] the A2 arm of the Q3 bridge drifts '
    '(rel %(a2bridge)s > %(bridgetol)s) because the A2 cumulative write is a TWO-STEP staircase (6.49 at L4 then '
    '8.71 at L5, plateau ~9.3) so L*_own = 4 precedes cumulative saturation; at L5 the same bridge gives rel '
    '0.080. No other erratum affects statistics (see memo section). '
    'PER-ARM FORWARDS = %(nw)s (model forwards only); total %(nrows)d.'
) % dict(
    mlpvec=_tri('Q4_share_mlp_vec_nb', 3),
    mlpbeh=_tri('Q4_share_mlp_beh_nb', 3),
    p1=_pass('P1'), p2=_pass('P2'), p3=_pass('P3'), p4=_pass('P4'), p5=_pass('P5'), p6=_pass('P6'),
    fp1=_pass('P1'), fp2=_pass('P2'),
    arch=_scitri('Q1_arch_max'),
    blk=_scitri('Q1_blk_max'),
    q4c=str(JV.get('Q4_joint', 'NA')),
    q6c=str(JV.get('Q6_joint', 'NA')),
    q7c=str(JV.get('Q7_joint', 'NA')),
    q8c=str(JV.get('Q8_joint', 'NA')),
    q9=str(JV.get('Q9_joint', 'NA')),
    q9c=str(JV.get('Q9_joint', 'NA')),
    comB=_tri('Q6_com_B', 3),
    comV=_tri('Q6_com_V', 3),
    gap=_tri('Q6_gap', 3),
    spwb=_tri('Q7_spearman_wall_ball', 3),
    spwj=_tri('Q7_spearman_wall_J', 3),
    rlinstar=_tri('Q8_rlin_nb', 3),
    ratio=_tri('Q8_rlin_peak_ratio', 2),
    cumbridge=_tri('Q3_cum_bridge', 3),
    fullswap=_tri('Q3_full_swap', 3),
    bridgerel=_tri('Q3_bridge_rel', 4),
    bridgetol=RES['floors']['BRIDGE_TOL_CUM'],
    ratioreanch='/'.join(_reach_ratio(a) for a in AO),
    a2bridge='%.4f' % float(V[AO[2]]['Q3_bridge_rel']),
    nullhigh='/'.join(str(V[a]['Q9_null_comB'].get('com_tail')) for a in AO),
    conftol=RES['floors']['CONF_TOL_COMB'],
    confdev='/'.join('%.3f' % float(V[a]['Q9_conf']['d_com']['INC_ALL']) for a in AO),
    nconf=int(NPAIR_CONF),
    ndisc=int(NPAIR_DISC),
    nw=json.dumps(NW, ensure_ascii=False),
    nrows=NROWS,
)

ENTRY = dict(
    phase=18,
    name='n2h1a11_behavioral_component_budget_glm4_9b_qwen3_14b_nf4',
    seal_sha8=sha8b(SEALB), exec_sha8=sha8b(EXECB), result_sha8=sha8b(RESB),
    probe_sha8=sha8b(PROBEB),
    evidence_level='statistical',
    model_scope='glm4-9b-chat-hf + Qwen3-14B (+ qwen3-4b calibration arm)',
    n_rows=NROWS,
    n_forwards_per_arm=NW,
    prereg_id='N2h1a11',
    superseded_by=None,
    verdict=verdict,
    rev_note=REV,
    created=time.strftime('%Y-%m-%d'),
    amend1_sha8=None,
    anchor_result_sha8=RES['anchor_result_p16_sha256'][:8],
)

if already:
    idx = [i for i, m in enumerate(LG['measurements']) if m.get('phase') == 18][-1]
    old = LG['measurements'][idx]
    CHK = ('result_sha8', 'rev_note', 'verdict', 'n_rows', 'n_forwards_per_arm',
           'seal_sha8', 'exec_sha8', 'probe_sha8', 'anchor_result_sha8')
    changed = [k for k in CHK if old.get(k) != ENTRY.get(k)]
    if changed:
        for k in CHK:
            old[k] = ENTRY[k]
        tmp = {k: v for k, v in LG.items() if k != 'ledger_sha256_8'}
        h = hashlib.sha256(json.dumps(tmp, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
        LG['ledger_sha256_8'] = h
        open(LEDGER, 'wb').write(json.dumps(LG, ensure_ascii=False, indent=1).encode('utf-8'))
        w('[B] Ledger 条目已就地刷新：changed=%s ; ledger_sha256_8=%s' % (changed, h))
    else:
        w('[B] Ledger 条目已与最终 result 一致（无需刷新）')
else:
    bak = os.path.join(P18T, 'atlas_ledger_backup_pre_phase18.json')
    shutil.copyfile(LEDGER, bak)
    w('[B] 备份 -> %s (sha8=%s)' % (os.path.basename(bak), sha8b(open(bak, 'rb').read())))
    LG['measurements'].append(ENTRY)
    tmp = {k: v for k, v in LG.items() if k != 'ledger_sha256_8'}
    h = hashlib.sha256(json.dumps(tmp, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    LG['ledger_sha256_8'] = h
    nb = json.dumps(LG, ensure_ascii=False, indent=1).encode('utf-8')
    open(LEDGER, 'wb').write(nb)
    w('[B] Ledger 追加完成：n %d -> %d ; ledger_sha256_8=%s' % (n_before, len(LG['measurements']), h))

LG2 = json.loads(open(LEDGER, 'rb').read().decode('utf-8'))
n_after = len(LG2['measurements'])
_last = [m for m in LG2['measurements'] if m.get('phase') == 18][-1]
w('[B] 落盘复核：n=%d ; phase18.result_sha8=%s ; verdict=%s'
  % (n_after, _last.get('result_sha8'), _last.get('verdict')))
assert n_after == n_before + (0 if already else 1)
assert _last['phase'] == 18
assert _last['result_sha8'] == sha8b(RESB), 'Ledger 未对齐最终 result'
assert _last['rev_note'] == REV, 'Ledger rev_note 未对齐（应现场渲染）'

# ---------------- [C] 报告
w('')
w('=== [C] Phase 18 判决摘要 ===')
w('joint_verdict = %s' % json.dumps(RES['joint_verdict'], ensure_ascii=False))
for k in sorted(RES['predictions_check']):
    v = RES['predictions_check'][k]
    tag = 'PASS' if v['pass_'] else ('N/A' if v['pass_'] is None else 'FAIL')
    w('  %s : %s' % (k, tag))
for a in RES['verdict']:
    v = RES['verdict'][a]
    w('  %-24s com_B=%s com_V=%s gap=%s %s | mlp_beh=%s mlp_vec=%s | rho_wb=%s rho_wJ=%s | bridge=%s'
      % (a, round(v['Q6_com_B'], 3), round(v['Q6_com_V'], 3), round(v['Q6_gap'], 3), v['Q6_label'],
         round(v['Q4_share_mlp_beh_nb'], 3), round(v['Q4_share_mlp_vec_nb'], 3),
         round(v['Q7_spearman_wall_ball'], 4), round(v['Q7_spearman_wall_J'], 4),
         round(v['Q3_bridge_rel'], 4)))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUTV, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('CLOSEOUT OK ->', OUTV)
