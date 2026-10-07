# -*- coding: utf-8 -*-
"""Phase 16 结账（幂等）：
  [A] 冻结「MEMO 预追加基线」-> tests/deepseek_temp/Phase16/memo_baseline_preappend_phase16.json
  [B] Ledger 补登一条（n 298 -> 299），并重算 ledger_sha256_8
  [C] 写 verify_ledger_phase16.txt
幂等：按 phase 成员检查（已含 16 则跳过追加）。
"""
import io
import os
import json
import time
import hashlib
import shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
BASE16 = os.path.join(P16T, 'memo_baseline_preappend_phase16.json')
OUTV = os.path.join(P16T, 'verify_ledger_phase16.txt')   # v2 约定：校验 verify_*.txt -> deepseek_temp/Phase{N}/

o = []


def w(s=''):
    o.append(str(s))
    print(s)


def sha8b(b):
    return hashlib.sha256(b).hexdigest()[:8]


RESP = os.path.join(P16T, 'result_phase16.json')
assert os.path.exists(RESP), 'result 尚未生成: %s' % RESP
RESB = open(RESP, 'rb').read()
RES = json.loads(RESB.decode('utf-8'))
EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
SEALB = open(os.path.join(P16T, 'N2h1a9_design_seal.json'), 'rb').read()
AM1B = open(os.path.join(P16T, 'N2h1a9_design_seal_amend1.json'), 'rb').read()
EXECB = open(os.path.join(P16T, 'execution_phase16.json'), 'rb').read()

w('=== Phase 16 closeout  clock=%s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('seal=%s exec=%s amend1=%s result=%s anchor(p15)=%s'
  % (sha8b(SEALB), sha8b(EXECB), sha8b(AM1B), sha8b(RESB), RES['anchor_result_sha8']))

# ---------------- [A] 预追加基线
raw0 = open(MEMO, 'rb').read()
t0 = raw0.decode('utf-8-sig')
hdrs0 = [ln for ln in t0.split('\r\n') if ln.startswith('## Phase ')]
_heads = {}
for i, l in enumerate(t0.split('\r\n')):
    if l.startswith('## '):
        _heads[l[:44]] = i + 1
B = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), tag='pre-append-phase16',
         path='research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
         bytes=len(raw0), lines=len(raw0.split(b'\r\n')),
         sha256=hashlib.sha256(raw0).hexdigest(), sha8=sha8b(raw0),
         bom=bool(raw0[:3] == b'\xef\xbb\xbf'),
         crlf=int(raw0.count(b'\r\n')), bare_lf=int(raw0.count(b'\n') - raw0.count(b'\r\n')),
         phase_headings=len(hdrs0), sections=_heads,
         note='Phase 16 追加前快照（Phase 15 节已在其中）。')
if not os.path.exists(BASE16):
    io.open(BASE16, 'w', encoding='utf-8', newline='\n').write(json.dumps(B, ensure_ascii=False, indent=1))
    w('[A] 冻结预追加基线 -> %s' % BASE16)
else:
    w('[A] 基线已存在（幂等跳过）')
w('    bytes=%d lines=%d phase_headings=%d sha8=%s bare_lf=%d'
  % (B['bytes'], B['lines'], B['phase_headings'], B['sha8'], B['bare_lf']))
if not os.path.exists(BASE16):
    # 断言只在首次冻结时生效（重跑时 MEMO 已含 Phase 16 节，标题数必然为 16）
    assert B['bare_lf'] == 0 and B['bom'], 'MEMO EOL/BOM 异常'
    assert B['phase_headings'] == 15, 'Phase 标题数应为 15，实为 %d' % B['phase_headings']
else:
    assert B['bare_lf'] == 0 and B['bom'], 'MEMO EOL/BOM 异常'

# ---------------- [B] Ledger
LB = open(LEDGER, 'rb').read()
LG = json.loads(LB.decode('utf-8'))
n_before = len(LG['measurements'])
already = any(m.get('phase') == 16 for m in LG['measurements'])

NPAIRS = len(EX['discovery'])   # 24 discovery pairs（profile 用的配对集；pairs_all=41 是 5 元组实例表，非配对数）
NSITES = len(EX['profile_sites'])
NALPHA = len(EX['alphas'])
CONST = 12   # E0(2 determ + 1 hook) + F3(3) + BASE 重算(3) + 杂项，与 Phase 15 的 6437=6048+336+41+12 一致
NW = {}
for a in EX['arm_order']:
    nc = len(RES['arms'][a].get('cands_used') or EX['localize']['cands'])
    NW[a] = NSITES * NALPHA * NPAIRS + nc * NPAIRS + 41 + CONST
NROWS = sum(NW.values())

verdict = '%s__%s__%s__%s' % (str(RES['joint_verdict'].get('Q1_joint', 'NA')).lower(),
                              str(RES['joint_verdict'].get('Q2_joint', 'NA')).lower(),
                              str(RES['joint_verdict'].get('Q4_joint', 'NA')).lower(),
                              str(RES['joint_verdict'].get('Q5_joint', 'NA')).lower())

REV = (
    'deepseek/N line Phase 16 (N2h1-alpha-9), WRITE-WINDOW-ORIGIN profile + concentration-statistic redesign, '
    'THREE arms under the single nf4 convention inherited byte-for-byte from Phase 15 '
    '(A0_calib_qwen3-4b-nf4 / A1_glm4-9b-nf4 / A2_qwen3-14b-nf4). '
    'MOTIVATION: Phase 15 independently localized the write window by B_cat (own-subspace adjacent max increment) '
    'at L*_own = 6 (A0) / 3 (A1) / 4 (A2), but the depth profile only covered [6..34] -- so for A1/A2 the write '
    'window lay OUTSIDE the profiled domain and the Phase-15 death-line question "write window vs concentration '
    'window" was structurally unanswerable. '
    'CHANGE 1 (grid): profile_sites extended downward to [1,2,3,4,5] + legacy [6..34] = 23 sites. '
    'CHANGE 2 (domain): primary domain = reachability mask REACH = {ell : rho(ell) >= UNREACH_y=0.10} with '
    'rho(ell) = Y(ell, alpha=1) = dDonor/FULL_SWAP, so that the write window becomes the LEFT ENDPOINT of the '
    'domain; masked-out sites are reported explicitly, never dropped silently. '
    'CHANGE 3 (statistic redesign): the legacy concentration statistic top3_share = max_w|sum(jm[j:j+W])|/range is '
    'an EXTREME-VALUE 3-window share whose window position is a deterministic function of the curve shape; and '
    'because the permutation null preserves the jump MULTISET, any statistic depending only on the multiset '
    '(spectral entropy, max/mean, participation ratio) is identically equal to the observation under the null and '
    'therefore structurally untestable (an explicitly excluded family). Replaced by two ORDER-SENSITIVE, '
    'SCALE-FREE quantities defined on the PHYSICAL depth axis: com_layer = sum_j |dj|*mid_j / sum_j |dj| (centroid '
    'in layer units; invariant to grid refinement -- a property the Phase-15 jump-index normalization lacked), and '
    'span_k = (max_idx-min_idx of the k largest |dj|)/(n-1) for k=3 (clustering of the dominant changes). Both are '
    'tested TWO-SIDED under the same permutation protocol (BP=2000, dedicated seeds SEED+41/+53 for the legacy '
    'control and SEED+61/+67 for the new statistics). '
    'ROW COUNT / BUDGET: 23 sites x 14 alphas x 24 discovery pairs per arm = 7728 forwards per arm, plus a '
    '14-15-candidate independent write-window localization and a 41-instance capture; per-arm forwards = %s ; '
    'total %d real GPU forwards. Model cost measured in the feasibility probe at 0.0358-0.0428 s/forward. '
    'ANCHORING (P2): the legacy [6..34] sub-range is recomputed in the same run and compared bit-for-bit against '
    'the FROZEN Phase-15 result (sha8 53a293a8): max|xhalf_new - xhalf_15|, max relative |J_new - J_15|, the legacy '
    'argmax_w integers and the legacy top3 shares. '
    'AMEND1 (criterion tiering, no hypothesis change; sha8 %s): SMOKE showed that xhalf is an INTERPOLATED '
    'quantity (cross_alpha on the alpha grid), so a single hard 1e-3 gate on it violates the soft-gate-first rule; '
    'the anchor criterion was tiered into RECON_OK (<=1e-3) / RECON_OK_LOOSE (<=5e-3, requires a per-site dxh '
    'table in the report) / RECON_DRIFT, with P2 requiring all three arms in the first two tiers and >=2/3 strict. '
    'GRID/MATERIAL/DEFINITIONS, P3-P7, UNREACH_y, CENTROID_SEP_MIN and CENTROID_AFTER_WIN_MIN were NOT changed.'
) % (json.dumps(NW, ensure_ascii=False), NROWS, sha8b(AM1B))

ENTRY = dict(
    phase=16,
    name='n2h1a9_writewin_origin_profile_centroid_span_glm4_9b_qwen3_14b_nf4',
    seal_sha8=sha8b(SEALB), exec_sha8=sha8b(EXECB), result_sha8=sha8b(RESB),
    evidence_level='statistical',
    model_scope='glm4-9b-chat-hf + Qwen3-14B (+ qwen3-4b calibration arm)',
    n_rows=NROWS,
    n_forwards_per_arm=NW,
    prereg_id='N2h1a9',
    superseded_by=None,
    verdict=verdict,
    rev_note=REV,
    created=time.strftime('%Y-%m-%d'),
    amend1_sha8=sha8b(AM1B),
    anchor_result_sha8=RES['anchor_result_sha8'],
)

if already:
    w('[B] Ledger 已含 phase 16（幂等跳过）')
else:
    bak = os.path.join(P16T, 'atlas_ledger_backup_pre_phase16.json')
    shutil.copyfile(LEDGER, bak)
    w('[B] 备份 -> %s (sha8=%s)' % (os.path.basename(bak), sha8b(open(bak, 'rb').read())))
    LG['measurements'].append(ENTRY)
    # 重算自哈希
    tmp = {k: v for k, v in LG.items() if k != 'ledger_sha256_8'}
    h = hashlib.sha256(json.dumps(tmp, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    LG['ledger_sha256_8'] = h
    nb = json.dumps(LG, ensure_ascii=False, indent=1).encode('utf-8')
    open(LEDGER, 'wb').write(nb)
    w('[B] Ledger 追加完成：n %d -> %d ; ledger_sha256_8=%s' % (n_before, len(LG['measurements']), h))

LG2 = json.loads(open(LEDGER, 'rb').read().decode('utf-8'))
n_after = len(LG2['measurements'])
w('[B] 落盘复核：n=%d ; last.phase=%s ; last.verdict=%s'
  % (n_after, LG2['measurements'][-1].get('phase'), LG2['measurements'][-1].get('verdict')))
assert n_after == n_before + (0 if already else 1)
assert LG2['measurements'][-1]['phase'] == 16

# ---------------- [C] 报告
w('')
w('=== [C] Phase 16 判决摘要 ===')
w('joint_verdict = %s' % json.dumps(RES['joint_verdict'], ensure_ascii=False))
for k in sorted(RES['predictions_check']):
    v = RES['predictions_check'][k]
    w('  %s : %s  %s' % (k, 'PASS' if v['pass_'] else 'FAIL', str(v.get('detail'))[:260]))
for a in RES['verdict']:
    v = RES['verdict'][a]
    w('  %-24s Q0=%s Q1=%s ell_reach=%s L*=%s Q4sep=%s Q5=%s'
      % (a, v.get('Q0_device'), v.get('Q1_label'), v.get('Q2_ell_reach'),
         v.get('Q2_L_star_own'), v.get('Q4_sep'), v.get('Q5_label')))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUTV, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('CLOSEOUT OK ->', OUTV)
