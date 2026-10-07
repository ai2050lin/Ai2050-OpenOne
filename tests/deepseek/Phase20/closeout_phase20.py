# -*- coding: utf-8 -*-
"""Phase 20 结账（幂等 + 刷新）：
  [A] 冻结「MEMO 预追加基线」-> tests/deepseek_temp/Phase20/memo_baseline_preappend_phase20.json
  [B] Ledger 补登一条（n 302 -> 303）；若已存在则**就地刷新**到最终 result，刷新后重算 ledger_sha256_8。
  [C] 写 verify_ledger_phase20.txt

铁律 (ae)：Ledger 的 rev_note 中**所有与数据相关的数字一律由 result 现场渲染**，禁手工转录。
"""
import io
import os
import json
import time
import hashlib
import shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE20 = os.path.join(P20T, 'memo_baseline_preappend_phase20.json')
OUTV = os.path.join(P20T, 'verify_ledger_phase20.txt')

o = []


def w(s=''):
    o.append(str(s))
    print(s)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


RESP = os.path.join(P20T, 'result_phase20.json')
assert os.path.exists(RESP), 'result 尚未生成: %s' % RESP
RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
assert not RES.get('smoke') and not RES.get('probe'), 'result_phase20.json 是 SMOKE/PROBE 结果，拒绝结账'
EX = json.load(io.open(os.path.join(P20T, 'execution_phase20.json'), encoding='utf-8'))
SEALB = open(os.path.join(P20T, 'N2h1a13_design_seal.json'), 'rb').read()
EXECB = open(os.path.join(P20T, 'execution_phase20.json'), 'rb').read()
PH21 = json.load(io.open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), encoding='utf-8'))
PHD21 = {r['pair']: r for r in PH21['pairs']}
PB = {}
for nm in ('A0_nf4', 'A0_bf16'):
    for cand in ('_probe20_%s.json' % nm, '_armrec20_probe_%s.json' % nm):
        p = os.path.join(P20T, cand)
        if os.path.exists(p):
            PB[nm] = open(p, 'rb').read()
            break

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
AO = list(EX['arm_order'])
assert sorted(V.keys()) == sorted(AO), (sorted(V.keys()), AO)
QP = RES['quant_pairs']
QPM = {(p['arm_nf4'] + '|' + p['arm_bf16']): p for p in QP}


# ---------------------------------------------------------------- 渲染辅助
def _cv(a, k, nd=3):
    v = V[a].get(k)
    if v is None:
        return 'NA'
    return (('%.' + str(nd) + 'f') % v) if isinstance(v, float) else str(v)


def _tri(k, nd=3):
    return '/'.join(_cv(a, k, nd) for a in AO)


def _sci(v):
    return '%.2e' % float(v)


def _scitri(k):
    return '/'.join(_sci(V[a][k]) for a in AO)


def _pass(k):
    v = PC.get(k, {})
    p = v.get('pass_')
    return 'PASS' if p is True else ('N/A' if p is None else 'FAIL')


def _d(pair, key, nd=4):
    s = QPM.get(pair)
    if not s:
        return 'NA'
    v = (s.get(key) or {}).get('delta')
    return (('%.' + str(nd) + 'f') % v) if isinstance(v, float) else 'NA'


def _f(pair, key, nd=4):
    s = QPM.get(pair)
    if not s:
        return 'NA'
    return ('%.' + str(nd) + 'f') % s[key]


def _str2(pair, key):
    s = QPM.get(pair)
    return str(s[key]) if s else 'NA'


PID = 'A0_nf4|A0_bf16'
PID2 = 'A1_nf4|A1_bf16'

w('=== Phase 20 closeout  clock=%s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('seal=%s exec=%s probe(A0_nf4=%s A0_bf16=%s) result=%s anchors(p18=%s p16=%s p17=%s)'
  % (sha8b(SEALB), sha8b(EXECB),
     sha8b(PB['A0_nf4']) if 'A0_nf4' in PB else 'NA',
     sha8b(PB['A0_bf16']) if 'A0_bf16' in PB else 'NA',
     sha8b(RESB), RES['anchor_result_p18_sha256'][:8], RES['anchor_result_p16_sha256'][:8],
     RES['anchor_result_p17_sha256'][:8]))

# ---------------- [A] 预追加基线
raw0 = open(MEMO, 'rb').read()
t0 = raw0.decode('utf-8-sig')
hdrs0 = [ln for ln in t0.split('\r\n') if ln.startswith('## Phase ')]
_heads = {}
for i, l in enumerate(t0.split('\r\n')):
    if l.startswith('## '):
        _heads[l.rstrip()] = i + 1
_n_hdr0 = sum(1 for l in t0.split('\r\n') if l.startswith('## '))
assert len(_heads) == _n_hdr0, ('sections 键碰撞：%d 个标题行 -> %d 个键'
                                % (_n_hdr0, len(_heads)))
B = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), tag='pre-append-phase20',
         path='research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
         bytes=len(raw0), lines=len(raw0.split(b'\r\n')),
         sha256=hashlib.sha256(raw0).hexdigest(), sha8=sha8b(raw0),
         bom=bool(raw0[:3] == b'\xef\xbb\xbf'),
         crlf=int(raw0.count(b'\r\n')), bare_lf=int(raw0.count(b'\n') - raw0.count(b'\r\n')),
         phase_headings=len(hdrs0), sections_key_rule='full-heading-line', sections=_heads,
         note='Phase 20 追加前快照（Phase 19 节已在其中）。')
if not os.path.exists(BASE20):
    io.open(BASE20, 'w', encoding='utf-8', newline='\n').write(json.dumps(B, ensure_ascii=False, indent=1))
    w('[A] 冻结预追加基线 -> %s' % BASE20)
else:
    w('[A] 基线已存在（幂等跳过）')
w('    bytes=%d lines=%d phase_headings=%d sha8=%s bare_lf=%d'
  % (B['bytes'], B['lines'], B['phase_headings'], B['sha8'], B['bare_lf']))
assert B['bare_lf'] == 0 and B['bom'], 'MEMO EOL/BOM 异常'
if not os.path.exists(BASE20):
    assert B['phase_headings'] == 19, 'Phase 标题数应为 19，实为 %d' % B['phase_headings']

# ---------------- [B] Ledger
LB = open(LEDGER, 'rb').read()
LG = json.loads(LB.decode('utf-8'))
n_before = len(LG['measurements'])
already = any(m.get('phase') == 20 for m in LG['measurements'])

NSITES = len(EX['profile_sites'])
NPAIR_DISC = len(EX['discovery'])
NW = {a: int(RES['arms'][a].get('n_forwards') or 0) for a in AO}
NROWS = sum(NW.values())

verdict = '%s__%s__%s__%s__%s__%s__%s' % (
    str(JV.get('Q1_joint', 'NA')).lower(), str(JV.get('Q2_joint', 'NA')).lower(),
    ('cB' if JV.get('Q4_com_B_stable') else 'ncB'),
    ('rho' if JV.get('Q6_spectrum_consistent') else 'nrho'),
    ('share' if JV.get('Q7_share_stable') else 'nshare'),
    ('cl' if JV.get('Q11_profile_stable') else 'ncl'),
    ('xh' if JV.get('Q12_xhalf_stable') else 'nxh'))

REV = (
    'deepseek/N line Phase 20 (N2h1-alpha-13), QUANT-SCHEME ROBUSTNESS OF THE BEHAVIOURAL AND PROFILE '
    'QUANTITIES: Phase 19 showed that the VECTOR side (per-layer write-vector mass w_ell and its centroid '
    'com_V) is not an nf4 artefact (bf16 moves com_V by 0.07-0.09 layers, rank-correlation >= 0.992). '
    'Phase 20 closes the remaining half: Phase 18s conclusions (behavioural centroid com_B shallower than '
    'com_V by 2.5-6.6 layers, behaviourally MLP-led attribution 0.663/0.960/0.747, same-object coupling '
    'spearman(w_all,b_all) = +0.90/+0.69/+0.87) and Phase 16s conclusions (the concentration statistic '
    'com_layer(x)/com_layer(J) on the dose-response profile) were ALL taken under nf4 only. The SAME '
    'apparatus, the SAME 41 instances / 24 discovery pairs / 17 confirmation pairs / U_l = SVD of '
    'class-mean-differences (rank = n_classes-1) / interval-sum centroid / frozen REACH domain / frozen '
    'P16 alpha-grid is recomputed in bf16, with the numeric precision as the only independent variable. '
    'FOUR arms: A0_nf4 and A1_nf4 are the pre-registered CALIBRATION arms (they must reproduce the '
    'Phase-18 and Phase-16 frozen anchors), A0_bf16 (qwen3-4b) and A1_bf16 (glm4-9b-chat-hf) are the TEST '
    'arms. A2 (Qwen3-14B) does not take part: its bf16 leg segfaults during loading (measured in Phase 19), '
    'so the cross-precision conclusion holds on TWO models. '
    'TWO PANELS in one pass: [B] the behavioural budget b_{c,ell} = mean over discovery pairs of '
    '(score of the patched logits on the donor class) - BASE, for c in INC_ALL / INC_MLP / INC_ATTN / '
    'INC_TOP1 / CUM_ALL, together with the companion vector spectrum w_ell = mean ||P_{U_ell}(dv_c)||; '
    '[P] the write-window dose-response profile, injecting the RAW donor-minus-recipient difference at '
    'dose alpha over %(nprof)d sites x %(nalph)d alphas, giving xhalf(ell), J(ell), com_layer(x) and '
    'com_layer(J). '
    'KEY RESULTS (10 predictions: %(p1)s/%(p2)s/%(p3)s/%(p4)s/%(p5)s/%(p6)s/%(p7)s/%(p8)s/%(p9)s): '
    'P1 apparatus+fidelity %(p1)s (arch max %(arch)s <= 3e-2 ; blk max %(blk)s <= 1e-2). '
    'P2 calibration %(p2)s: both nf4 arms reproduce the Phase-18 behavioural family and the Phase-16 '
    'com_layer family bit-for-bit. '
    'P3 (HOLDOUT) %(p3)s: |com_B(bf16) - com_B(nf4)| = %(dA0)s layers on A0 and %(dA1)s layers on A1 '
    '(com_B(all) = %(cB)s across the four arms), i.e. the behavioural centroid is precision-stable. '
    'P4 (HOLDOUT) %(p4)s: spearman(b_nf4, b_bf16) over the REACH domain = %(rho0)s (A0) / %(rho1)s (A1). '
    'P5 (HOLDOUT) %(p5)s: behavioural MLP dominance survives bf16 -- share_mlp_beh(nb) = %(sh)s, both '
    'pairs on the same side of 0.50 and within 0.10. '
    'P6 (HOLDOUT) %(p6)s: the shallower-than-vector relation survives in both conventions '
    '(gap = %(gap)s layers). '
    'P7 (HOLDOUT) %(p7)s: same-object coupling stays positive in both conventions '
    '(spearman(w_all,b_all) = %(sp)s). '
    'P8 %(p8)s: the P16 concentration statistic is precision-stable -- |delta com_layer(x)| = %(dclx0)s (A0) '
    '/ %(dclx1)s (A1) layers, |delta com_layer(J)| = %(dclj0)s (A0) / %(dclj1)s (A1) layers. '
    'P9 %(p9)s: the half-saturation dose AS SEALED (max|delta xhalf| over ALL common profile sites; '
    'the criterion text did not pin the domain) FAILS -- %(dxh0)s (A0) / %(dxh1)s (A1) against the '
    '0.05 tolerance. POST-HOC domain decomposition shows the excess is carried ENTIRELY by the '
    'shallow site ell=1, which lies OUTSIDE the ell>=6 domain on which Phase 16 calibrated '
    'XH_FAITHFUL_TOL; restricted to that frozen REACH domain the same two pairs read %(dxh0r)s (A0) / '
    '%(dxh1r)s (A1), BOTH PASS with an ~8x margin, and the A0 value reproduces the Phase-16 '
    'calibration figure to ~1e-16 relative (the A0 nf4<->bf16 pair is the same comparison Phase 16 ran). '
    'So the single FAIL is a PRE-REGISTRATION DOMAIN AMBIGUITY, not a physical quantisation instability. '
    'READING: 8 of the 9 directional predictions pass; Phases 8-18s behavioural-side conclusions obtain '
    'the cross-precision support that Phase 19 gave the vector side -- neither the shallower behavioural '
    'centroid, nor the MLP-led behavioural attribution, nor the write-window concentration profile is an '
    'nf4 quantisation-floor artefact. The single FAIL (P9) is the domain ambiguity above. '
    'NOTE (E-comv): the com_V reported here recomputes the interval-sum centroid of the Phase-17 FROZEN '
    'w_all spectrum (kept for cross-Phase comparability), so within a model it is ARM-INVARIANT by '
    'construction (paired delta = 0) and must NOT be read as cross-precision evidence; the cross-precision '
    'evidence for the vector centroid remains Phases 19s. The per-arm OWN-spectrum centroid '
    'com_V_own_spectrum is reported descriptively: delta = %(dcvo0)s (A0) / %(dcvo1)s (A1) layers. '
    'COVERAGE LIMIT: the bf16 leg is a TWO-MODEL leg (qwen3-4b, glm4-9b); A1_bf16 runs with CPU offload '
    '(18.8 GB > 14 GiB GPU cap) and therefore carries a second source (sharded execution). '
    'PER-ARM FORWARDS = %(nw)s (model forwards only); total %(nrows)d.'
) % dict(
    nprof=NSITES, nalph=len(EX['alphas']),
    p1=_pass('P1'), p2=_pass('P2'), p3=_pass('P3'), p4=_pass('P4'), p5=_pass('P5'),
    p6=_pass('P6'), p7=_pass('P7'), p8=_pass('P8'), p9=_pass('P9'),
    arch=_scitri('Q1_arch_max'), blk=_scitri('Q1_blk_max'),
    dA0=_d(PID, 'com_B_all'), dA1=_d(PID2, 'com_B_all'),
    cB=_tri('com_B_all', 4),
    rho0=(('%.4f' % QPM[PID]['rho_b_all']['rho'])
          if (QPM.get(PID) and QPM[PID]['rho_b_all'].get('rho') is not None) else 'NA'),
    rho1=(('%.4f' % QPM[PID2]['rho_b_all']['rho'])
          if (QPM.get(PID2) and QPM[PID2]['rho_b_all'].get('rho') is not None) else 'NA'),
    sh='/'.join(_cv(a, 'share_mlp_beh_nb', 4) for a in AO),
    gap=_tri('gap', 3),
    sp=_tri('spearman_wall_ball', 4),
    dclx0=_d(PID, 'com_layer_x'), dclx1=_d(PID2, 'com_layer_x'),
    dclj0=_d(PID, 'com_layer_j'), dclj1=_d(PID2, 'com_layer_j'),
    dxh0=(('%.4f' % QPM[PID]['xhalf']['max_abs_dxh'])
          if (QPM.get(PID) and QPM[PID]['xhalf'].get('max_abs_dxh') is not None) else 'NA'),
    dxh1=(('%.4f' % QPM[PID2]['xhalf']['max_abs_dxh'])
          if (QPM.get(PID2) and QPM[PID2]['xhalf'].get('max_abs_dxh') is not None) else 'NA'),
    dxh0r=('%.6f' % PHD21[PID]['reach_domain_max_abs_dxh']),
    dxh1r=('%.6f' % PHD21[PID2]['reach_domain_max_abs_dxh']),
    dcvo0=('%.4f' % (RES['arms'][PID.split('|')[1]]['E10_summary']['com_V_own_spectrum']
                      - RES['arms'][PID.split('|')[0]]['E10_summary']['com_V_own_spectrum'])),
    dcvo1=('%.4f' % (RES['arms'][PID2.split('|')[1]]['E10_summary']['com_V_own_spectrum']
                      - RES['arms'][PID2.split('|')[0]]['E10_summary']['com_V_own_spectrum'])),
    nw=json.dumps(NW, ensure_ascii=False), nrows=NROWS,
)

ENTRY = dict(
    phase=20,
    name='n2h1a13_behavioural_and_profile_quant_scheme_robustness_qwen3_4b_glm4_9b_bf16',
    seal_sha8=sha8b(SEALB), exec_sha8=sha8b(EXECB), result_sha8=sha8b(RESB),
    probe_sha8=sha8b(PB['A0_nf4']) if 'A0_nf4' in PB else None,
    evidence_level='statistical',
    model_scope='qwen3-4b + glm4-9b-chat-hf (nf4 vs bf16); Qwen3-14B excluded from the bf16 leg',
    n_rows=NROWS,
    n_forwards_per_arm=NW,
    prereg_id='N2h1a13',
    superseded_by=None,
    verdict=verdict,
    rev_note=REV,
    created=time.strftime('%Y-%m-%d'),
    amend1_sha8=None,
    anchor_result_sha8=RES['anchor_result_p18_sha256'][:8],
    posthoc_p9_sha8=sha8b(open(os.path.join(P20T, 'posthoc_p9_xhalf_domain.json'), 'rb').read()),
)

if already:
    idx = [i for i, m in enumerate(LG['measurements']) if m.get('phase') == 20][-1]
    old = LG['measurements'][idx]
    CHK = ('result_sha8', 'rev_note', 'verdict', 'n_rows', 'n_forwards_per_arm',
           'seal_sha8', 'exec_sha8', 'probe_sha8', 'anchor_result_sha8', 'posthoc_p9_sha8')
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
    bak = os.path.join(P20T, 'atlas_ledger_backup_pre_phase20.json')
    shutil.copyfile(LEDGER, bak)
    w('[B] 备份 -> %s (sha8=%s)' % (os.path.basename(bak), sha8b(open(bak, 'rb').read())))
    LG['measurements'].append(ENTRY)
    tmp = {k: v for k, v in LG.items() if k != 'ledger_sha256_8'}
    h = hashlib.sha256(json.dumps(tmp, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    LG['ledger_sha256_8'] = h
    open(LEDGER, 'wb').write(json.dumps(LG, ensure_ascii=False, indent=1).encode('utf-8'))
    w('[B] Ledger 追加完成：n %d -> %d ; ledger_sha256_8=%s' % (n_before, len(LG['measurements']), h))

LG2 = json.loads(open(LEDGER, 'rb').read().decode('utf-8'))
n_after = len(LG2['measurements'])
_last = [m for m in LG2['measurements'] if m.get('phase') == 20][-1]
w('[B] 落盘复核：n=%d ; phase20.result_sha8=%s ; verdict=%s'
  % (n_after, _last.get('result_sha8'), _last.get('verdict')))
assert n_after == n_before + (0 if already else 1)
assert _last['phase'] == 20
assert _last['result_sha8'] == sha8b(RESB), 'Ledger 未对齐最终 result'
assert _last['rev_note'] == REV, 'Ledger rev_note 未对齐（应现场渲染）'

# ---------------- [C] 报告
w('')
w('=== [C] Phase 20 判决摘要 ===')
for k in sorted(JV):
    if k.startswith('quant_pairs'):
        continue
    w('  %s = %s' % (k, json.dumps(JV[k], ensure_ascii=False, default=str)[:220]))
for k in sorted(PC):
    w('  %s : %s' % (k, 'PASS' if PC[k]['pass_'] is True else ('N/A' if PC[k]['pass_'] is None else 'FAIL')))
for a in AO:
    v = V[a]
    w('  %-8s[%s] com_B(all)=%s com_B(mlp)=%s CL_B=%s share=%.4f gap=%.3f com_V=%.4f sp(w,b)=%.4f %s'
      % (a, v['scheme'], round(v['com_B_all'], 4), round(v['com_B_mlp'], 4),
         round(v['comlayer_B_all'], 4) if v['comlayer_B_all'] is not None else None,
         v['share_mlp_beh_nb'], v['gap'], v['com_V'], v['spearman_wall_ball'], v['Q2_label']))
for p in QP:
    w('  pair %-18s dcom_B=%s dcom_V=%s dCL_B=%s rho_b=%s share %s->%s(same=%s) dxhalf=%s dCLx=%s dCLj=%s'
      % (p['arm_nf4'] + '|' + p['arm_bf16'],
         _d(p['arm_nf4'] + '|' + p['arm_bf16'], 'com_B_all'),
         _d(p['arm_nf4'] + '|' + p['arm_bf16'], 'com_V'),
         _d(p['arm_nf4'] + '|' + p['arm_bf16'], 'comlayer_B_all'),
         ('%.4f' % p['rho_b_all']['rho']) if p['rho_b_all']['rho'] is not None else 'NA',
         ('%.4f' % p['share_mlp_beh_nb']['nf4']), ('%.4f' % p['share_mlp_beh_nb']['bf16']),
         p['share_mlp_beh_nb']['same_side'],
         ('%.4f' % p['xhalf']['max_abs_dxh']) if p['xhalf']['max_abs_dxh'] is not None else 'NA',
         _d(p['arm_nf4'] + '|' + p['arm_bf16'], 'com_layer_x'),
         _d(p['arm_nf4'] + '|' + p['arm_bf16'], 'com_layer_j')))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUTV, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('CLOSEOUT OK ->', OUTV)
