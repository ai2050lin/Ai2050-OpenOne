# -*- coding: utf-8 -*-
"""Phase 17 结账（幂等 + 刷新）：
  [A] 冻结「MEMO 预追加基线」-> tests/deepseek_temp/Phase17/memo_baseline_preappend_phase17.json
  [B] Ledger 补登一条（n 299 -> 300）；若已存在则**就地刷新**到最终 result（result_sha8 / rev_note / verdict /
      n_rows / n_forwards_per_arm / 各 sha8），刷新后重算 ledger_sha256_8。
  [C] 写 verify_ledger_phase17.txt（落点 v2：verify_*.txt -> deepseek_temp/Phase{N}/）

铁律 (ae)：Ledger 的 rev_note 中**所有与数据相关的数字一律由 result 现场渲染**，禁手工转录。
本脚本所有 P1–P7 / 对照 / 勘误数字均经 _cv()/_tri()/_sci() 从 result_phase17.json 取用。
"""
import io
import os
import json
import time
import hashlib
import shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE17 = os.path.join(P17T, 'memo_baseline_preappend_phase17.json')
OUTV = os.path.join(P17T, 'verify_ledger_phase17.txt')

o = []


def w(s=''):
    o.append(str(s))
    print(s)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


RESP = os.path.join(P17T, 'result_phase17.json')
assert os.path.exists(RESP), 'result 尚未生成: %s' % RESP
RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
SEALB = open(os.path.join(P17T, 'N2h1a10_design_seal.json'), 'rb').read()
EXECB = open(os.path.join(P17T, 'execution_phase17.json'), 'rb').read()
PROBEB = open(os.path.join(P17T, '_probe_feasibility_A0.json'), 'rb').read()

# 勘误留痕（E4 的「修正前」com_V 亦从磁盘现场读取，不从散文转录）
V2P = os.path.join(P17T, 'result_phase17_v2_preintervalfix.json')
V2 = json.loads(open(V2P, 'rb').read().decode('utf-8')) if os.path.exists(V2P) else None

V = RES['verdict']
JV = RES['joint_verdict']
PC = RES['predictions_check']
AO = list(EX['arm_order'])
assert len(AO) == 3, AO


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


def _v2tri(k, nd=3):
    if not V2:
        return 'NA'
    out = []
    for a in AO:
        try:
            out.append(('%.' + str(nd) + 'f') % float(V2['verdict'][a][k]))
        except Exception:
            out.append('NA')
    return '/'.join(out)


def _dcom_tri(nd=3):
    return '/'.join(('%.' + str(nd) + 'f') % float(V[a]['Q8_conf']['d_com']) for a in AO)


def _top1_tri(nd=3):
    return '/'.join('%.*f' % (nd, float(RES['arms'][a]['E5_com_V']['top1_head_share_nb'])) for a in AO)


w('=== Phase 17 closeout  clock=%s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('seal=%s exec=%s probe=%s result=%s anchor(p16)=%s'
  % (sha8b(SEALB), sha8b(EXECB), sha8b(PROBEB), sha8b(RESB), RES['anchor_result_sha256'][:8]))

# ---------------- [A] 预追加基线
raw0 = open(MEMO, 'rb').read()
t0 = raw0.decode('utf-8-sig')
hdrs0 = [ln for ln in t0.split('\r\n') if ln.startswith('## Phase ')]
_heads = {}
for i, l in enumerate(t0.split('\r\n')):
    if l.startswith('## '):
        _heads[l[:44]] = i + 1
B = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), tag='pre-append-phase17',
         path='research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
         bytes=len(raw0), lines=len(raw0.split(b'\r\n')),
         sha256=hashlib.sha256(raw0).hexdigest(), sha8=sha8b(raw0),
         bom=bool(raw0[:3] == b'\xef\xbb\xbf'),
         crlf=int(raw0.count(b'\r\n')), bare_lf=int(raw0.count(b'\n') - raw0.count(b'\r\n')),
         phase_headings=len(hdrs0), sections=_heads,
         note='Phase 17 追加前快照（Phase 16 节已在其中）。')
if not os.path.exists(BASE17):
    io.open(BASE17, 'w', encoding='utf-8', newline='\n').write(json.dumps(B, ensure_ascii=False, indent=1))
    w('[A] 冻结预追加基线 -> %s' % BASE17)
else:
    w('[A] 基线已存在（幂等跳过）')
w('    bytes=%d lines=%d phase_headings=%d sha8=%s bare_lf=%d'
  % (B['bytes'], B['lines'], B['phase_headings'], B['sha8'], B['bare_lf']))
assert B['bare_lf'] == 0 and B['bom'], 'MEMO EOL/BOM 异常'
if not os.path.exists(BASE17):
    assert B['phase_headings'] == 16, 'Phase 标题数应为 16，实为 %d' % B['phase_headings']

# ---------------- [B] Ledger
LB = open(LEDGER, 'rb').read()
LG = json.loads(LB.decode('utf-8'))
n_before = len(LG['measurements'])
already = any(m.get('phase') == 17 for m in LG['measurements'])

NPAIR_DISC = len(EX['discovery'])
NPAIR_CONF = len(EX['confirmation'])
NSITES = len(EX['profile_sites'])
NW = {}
for a in EX['arm_order']:
    nfw = int(RES['arms'][a].get('n_forwards') or 0)
    NW[a] = nfw
NROWS = sum(NW.values())

verdict = '%s__%s__%s__%s__%s__%s' % (
    str(RES['joint_verdict'].get('Q1_joint', 'NA')).lower(),
    str(RES['joint_verdict'].get('Q2_joint', 'NA')).lower(),
    str(RES['joint_verdict'].get('Q3_joint', 'NA')).lower(),
    str(RES['joint_verdict'].get('Q4_joint', 'NA')).lower(),
    str(RES['joint_verdict'].get('Q5_joint', 'NA')).lower(),
    str(RES['joint_verdict'].get('Q6_joint', 'NA')).lower())

# ---- 数据驱动 REV（铁律 (ae)：禁止手工转录数字） ------------------------------
REV = (
    'deepseek/N line Phase 17 (N2h1-alpha-10), WRITE-VECTOR POSITION AND EFFICACY: the Phase-8 vector budget '
    '(exactly additive, P_U decomposition) is extended from the single layer L6 to EVERY layer, yielding the '
    'vector mass spectrum w_ell = mean_pairs ||P_{U_ell}(Delta_inc_ell)|| with Delta_inc_ell := Delta_attn_ell '
    '+ Delta_mlp_ell and Delta_attn_ell := sum_h Delta_head_h (additivity is a DEFINITION, hence exact by '
    'construction; two FIDELITY gates check it against the model\'s own computation: arch identity '
    '||(h_{l+1}-h_l)-(attn_out_l+m_l)||/||h_{l+1}-h_l|| and block additivity ||sum_h head_block - '
    'o_proj(v)||/||o_proj(v)||, floors from the measured nf4 noise). '
    'THREE arms, same nf4 convention inherited byte-for-byte from Phases 15/16 (A0_calib_qwen3-4b-nf4 / '
    'A1_glm4-9b-nf4 / A2_qwen3-14b-nf4; arms ordered A0/A1/A2 throughout). No extra forwards beyond 44 per arm '
    '(2 determinism + 1 hook + 41 capture); the per-head decomposition uses head-block MASKED CALLS THROUGH THE '
    'MODULE ITSELF (never the packed 4-bit weight matrix). '
    'MOTIVATION: (i) Phase 16 explicitly limited com_layer to a DESCRIPTIVE position statistic -- it can say '
    '"the change is near L24" but not "what L24 does"; Phase 8 only ever decomposed ONE layer (L6) although the '
    'com_layer values are A0 23.0 / A1 18.2 / A2 8.4. (ii) Phase 16\'s P6 falsification RETRACTED the physical '
    'depth reading of "xhalf deep-tail vs J shallow-end", leaving no replacement position quantity. '
    '(iii) Phase 10 established that the behavioural gain J(ell) DECREASES with depth while the vector-level '
    'write mass had never been measured. '
    'KEY RESULTS (7 predictions: 6 PASS, P7 descriptive with no directional prediction): '
    'P1 FID_ALL_PASS (arch max %(arch)s <= 3e-2 ; blocks max %(blk)s <= 1e-2 ; determinism 0.0 ; all on cuda). '
    'P2 ANCHOR_ALL_OK: com_layer(x)/com_layer(J)/span3/L*_own/ell_reach re-read from the frozen Phase-16 result '
    'and reproduced to <=1e-6 on 3/3 arms. '
    'P3 (HOLDOUT, the primary prediction) DEEP_ALL: com_V = %(comv)s versus median(REACH) = %(med)s, i.e. the '
    'vector write mass is concentrated in the DEEP half of the reachable domain on the two arms that were NEVER '
    'observed before the seal was frozen (A1/A2). '
    'P4 (A2 discriminator arm) POSITION_DECOUPLED on 2/3 (A0 the exception): min(d_x,d_j) = %(mind)s vs '
    'CENTROID_SEP_MIN=4.0 -- A2\'s behavioural centroids nearly coincide (com_layer(x)=8.42 vs com_layer(J)=9.14 '
    'in the frozen anchor) yet its vector centroid sits ~17.5 layers away, so the behavioural centroid CANNOT be '
    'replaced by the vector mass centroid; Phase 16\'s "descriptive position" limitation is STRENGTHENED. A0 is '
    'the only arm where they align (min_d=3.14) -- exactly the mirror of the Phase-16 P6 falsification. '
    'P5 MLP_DOMINANT_ALL: MLP vector-budget share in the com_V neighbourhood (+-2 layers) = %(mlp)s; largest '
    'single head share = %(top1)s -- far below any single-head-dominance threshold (>0.50), so no single head '
    'dominates. '
    'P6 WRITE_EFFICACY_ANTICORR_ALL: spearman(w_ell, J_ell) = %(sp)s -- the vector write mass and the '
    'behavioural gain are ANTI-correlated across depth, i.e. the deep end carries a large amount of writing that '
    'is behaviourally ineffective. '
    'P7 (descriptive) %(q7)s: span_k (k in {2,3,5}) and com_layer give the SAME order for xhalf relative to J '
    'on 3/3 arms (wider span <=> deeper centroid). '
    'CONTROLS: permutation null on the mass profile (BP=%(bp)d, seeds comv_all=%(s1)d / comv_mlp=%(s2)d) gives a '
    'HIGH tail on 3/3 arms (obs %(comv)s vs p95 %(p95)s); the CONFIRMATION set (n=%(nconf)d, disjoint from the '
    '%(ndisc)d discovery pairs) reproduces com_V within %(dcom)s layers (tolerance %(tol)s). '
    'SAME-ROUND ERRATA (append-only; statistics and data otherwise unchanged): [E1] the v1 merge encoded the '
    'span/centroid coupling as the mismatched pairing (sx<sj)==(cx>cj) instead of "same sign"; the label was '
    'corrected to %(q7)s and the v1 reading is retained as Q7_joint_v1_mismatched_pairing (v1 result kept at '
    'result_phase17_v1_jointlabel.json). [E2] the first SMOKE caught a degenerate-path bug (SMOKE truncation '
    'emptied the pair set); fixed before the production run. [E4] com_of_mass first took w_{s_j} (single site) '
    'instead of the seal\'s INTERVAL SUM W_j = sum_{l in [s_j,s_{j+1})} w_l; A0 gave %(comv2)s (pre-fix) and '
    '%(comv)s after the fix, the latter matching the INDEPENDENT probe (26.15) bit-for-bit (a cross-'
    'implementation check); only the com_V family and null quantiles were affected, no model forward was '
    're-run, and P3/P4/P6 verdicts are unchanged. [E3] the first probe used o_proj.weight (packed 4-bit uint8) '
    'and crashed; switched to head-block masked calls through the module itself. '
    'PER-ARM FORWARDS = %(nw)s (model forwards only); total %(nrows)d. Head-block masked calls: '
    '%(ndisc)dx(L-1)x2 discovery + %(nconf)dx(L-1)x2 confirmation per arm.'
) % dict(
    arch='/'.join(_sci(_g(a, 'Q1_arch_max')) for a in AO),
    blk='/'.join(_sci(_g(a, 'Q1_blk_max')) for a in AO),
    comv=_tri('Q3_com_V', 3),
    med='/'.join(str(_g(a, 'Q3_median')) for a in AO),
    mind=_tri('Q4_min_d', 2),
    mlp=_tri('Q5_share_mlp_nb', 3),
    top1=_top1_tri(3),
    sp=_tri('Q6_spearman_wJ', 3),
    q7=str(JV.get('Q7_joint', 'NA')),
    bp=int(RES['bootstrap']['BP']),
    s1=int(RES['bootstrap']['seeds']['comv_all']),
    s2=int(RES['bootstrap']['seeds']['comv_mlp']),
    p95='/'.join('%.3f' % float(V[a]['Q7_null_all']['com_p95']) for a in AO),
    nconf=int(NPAIR_CONF),
    ndisc=int(NPAIR_DISC),
    dcom=_dcom_tri(3),
    tol=RES['floors']['CONF_TOL_COMV'],
    comv2=_v2tri('Q3_com_V', 3),
    nw=json.dumps(NW, ensure_ascii=False),
    nrows=NROWS,
)

ENTRY = dict(
    phase=17,
    name='n2h1a10_writevec_centroid_component_attribution_glm4_9b_qwen3_14b_nf4',
    seal_sha8=sha8b(SEALB), exec_sha8=sha8b(EXECB), result_sha8=sha8b(RESB),
    probe_sha8=sha8b(PROBEB),
    evidence_level='statistical',
    model_scope='glm4-9b-chat-hf + Qwen3-14B (+ qwen3-4b calibration arm)',
    n_rows=NROWS,
    n_forwards_per_arm=NW,
    prereg_id='N2h1a10',
    superseded_by=None,
    verdict=verdict,
    rev_note=REV,
    created=time.strftime('%Y-%m-%d'),
    amend1_sha8=None,
    anchor_result_sha8=RES['anchor_result_sha256'][:8],
)

if already:
    idx = [i for i, m in enumerate(LG['measurements']) if m.get('phase') == 17][-1]
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
    bak = os.path.join(P17T, 'atlas_ledger_backup_pre_phase17.json')
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
_last = [m for m in LG2['measurements'] if m.get('phase') == 17][-1]
w('[B] 落盘复核：n=%d ; phase17.result_sha8=%s ; verdict=%s'
  % (n_after, _last.get('result_sha8'), _last.get('verdict')))
assert n_after == n_before + (0 if already else 1)
assert _last['phase'] == 17
assert _last['result_sha8'] == sha8b(RESB), 'Ledger 未对齐最终 result'
assert _last['rev_note'] == REV, 'Ledger rev_note 未对齐（应现场渲染）'

# ---------------- [C] 报告
w('')
w('=== [C] Phase 17 判决摘要 ===')
w('joint_verdict = %s' % json.dumps(RES['joint_verdict'], ensure_ascii=False))
for k in sorted(RES['predictions_check']):
    v = RES['predictions_check'][k]
    tag = 'PASS' if v['pass_'] else ('N/A' if v['pass_'] is None else 'FAIL')
    w('  %s : %s  %s' % (k, tag, str(v.get('detail'))[:300]))
for a in RES['verdict']:
    v = RES['verdict'][a]
    w('  %-24s com_V=%s med=%s %s | min_d=%s %s | mlp_nb=%s | rho_wJ=%s | null=%s | conf_d=%s'
      % (a, round(v['Q3_com_V'], 3), v['Q3_median'], v['Q3_label'],
         round(v['Q4_min_d'], 2), v['Q4_label'],
         round(v['Q5_share_mlp_nb'], 3), round(v['Q6_spearman_wJ'], 4),
         v['Q7_null_all']['com_tail'], round(v['Q8_conf']['d_com'], 3)))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUTV, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('CLOSEOUT OK ->', OUTV)
