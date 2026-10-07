# -*- coding: utf-8 -*-
"""
Phase 15 收尾：judgement_phase15.json（三臂判决 + 对照表，数字全部由 result 渲染）
+ Ledger 补登（297 -> 298，含备份与 migration_history round 8）
+ 备忘录 pre-append 基线快照。
幂等：若 Ledger 已存在 phase 15 条目则跳过补登；基线快照可重复生成。
用法：python closeout_phase15.py
"""
import io
import os
import json
import time
import shutil
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
RESP = os.path.join(P15T, 'result_phase15.json')
SEALP = os.path.join(P15T, 'N2h1a8_design_seal.json')
AM1P = os.path.join(P15T, 'N2h1a8_design_seal_amend1.json')
EXECP = os.path.join(P15T, 'execution_phase15.json')
OUT = os.path.join(P15T, 'closeout_phase15.txt')

_o = []


def w(s=''):
    _o.append(str(s))
    print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha(p)[:8]


R = json.load(io.open(RESP, encoding='utf-8'))
S = json.load(io.open(SEALP, encoding='utf-8'))
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
EX = json.load(io.open(EXECP, encoding='utf-8'))
res_sha = sha(RESP)
seal_sha = sha(SEALP)
am1_sha = sha(AM1P)
exec_sha = sha(EXECP)
assert AM1['amend_of_seal_sha256'] == seal_sha, 'amend1 指向的 seal 漂移'
assert R['amend1_sha256'] == am1_sha, 'result 记录的 amend1 sha 与磁盘不符'

ARMS = EX['arm_order']
V = R['verdict']
J = R['joint_verdict']
PC = R['predictions_check']
E4 = R['E4_summary']
E5 = R['E5_concentration']
E2 = R['E2_full_swap']
E3 = R['E3_localize']
E6 = R['E6_calibration']
FL = R['floors']

# ---------------------------------------------------------------- 1. judgement
w('Phase 15 closeout ; clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('result sha8 %s ; seal sha8 %s ; exec sha8 %s' % (res_sha[:8], seal_sha[:8], exec_sha[:8]))

rows = []
for a in ARMS:
    if a not in E5:
        rows.append(dict(arm=a, status='ERROR'))
        continue
    e5, e2, e3, e4 = E5[a], E2[a], E3[a], E4[a]
    rows.append(dict(
        arm=a, status='OK',
        model=EX['arms'][a]['model'], role=EX['arms'][a]['role'],
        L=R['arms'][a]['cfg']['L'], hid=R['arms'][a]['cfg']['hid'],
        tie=R['arms'][a]['cfg']['tie'], ckpt_gb=EX['arms'][a]['ckpt_gb'],
        FULL_SWAP=e2['FULL_SWAP'],
        T2_only=R['arms'][a]['T2_only'], determinism=R['arms'][a]['E0_selfcheck']['determinism_maxdiff'],
        L_star_own=e3['L_star_own'], L_star_increment=e3['L_star_increment'],
        xhalf=e4['xhalf'], J=e4['J'], XH_RANGE=e4['XH_RANGE'],
        share_x=e5['top3_x'], argmax_w_x=e5['argmax_w_x'], win_sem_x=e5['win_sem_x'],
        share_j=e5['top3_j'], argmax_w_j=e5['argmax_w_j'], win_sem_j=e5['win_sem_j'],
        null95_x=(e5['null_x'] or {}).get('null95'),
    ))
    rows[-1]['null95_j'] = (e5['null_j'] or {}).get('null95')
    rows[-1]['margin_x'] = e5['margin_x']
    rows[-1]['margin_j'] = e5['margin_j']
    rows[-1]['d_argmax_window'] = e5['d_argmax_window']
    rows[-1]['spearman_xh_depth'] = e5['spearman_xh_depth']
    rows[-1]['spearman_J_depth'] = e5['spearman_J_depth']
    rows[-1]['sup_id_arm'] = R['arms'][a].get('sup_id_arm')
    rows[-1]['sup_id_matches_ref'] = R['arms'][a].get('sup_id_matches_ref')
    rows[-1]['F2_base_bad_n'] = len(R['arms'][a].get('F2_base_bad') or [])
    rows[-1]['verdict'] = V[a]

INHP = S['inheritance_anchors']['inherited_published']
judgement = dict(
    phase=15, name=S['name'],
    seal_sha8=seal_sha[:8], exec_sha8=exec_sha[:8], result_sha8=res_sha[:8],
    amend1_sha8=am1_sha[:8], amend1_kind=AM1['kind'],
    amend1_incident=AM1['evidence_from_device_gate'],
    sup_id_per_arm=R.get('sup_id_per_arm'),
    sup_id_ref=R.get('sup_id_ref'),
    rows=rows,
    cross_model=dict(arms_used=J['arms_used_for_cross_model'],
                     Q2_joint=J['Q2_joint'], Q3_joint=J['Q3_joint']),
    calibration=dict(
        arm='A0_calib_qwen3-4b-nf4',
        max_abs_dxh=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('max_abs_dxh'),
        tol=FL['XH_FAITHFUL_TOL'],
        argmax_w_x_nf4=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('argmax_w_x_nf4'),
        argmax_w_x_bf16=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('argmax_w_x_bf16'),
        argmax_same=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('argmax_same'),
        max_abs_drecover=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('max_abs_drecover'),
        share_x_nf4=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('share_x_nf4'),
        share_x_bf16=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('share_x_bf16'),
        XH_RANGE_nf4=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('XH_RANGE_nf4'),
        XH_RANGE_bf16=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('XH_RANGE_bf16'),
        J_ratio_min=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('J_ratio_min'),
        J_ratio_max=(E6.get('A0_calib_qwen3-4b-nf4') or {}).get('J_ratio_max'),
        label=V.get('A0_calib_qwen3-4b-nf4', {}).get('Q1_label')),
    reference_4B=dict(d_argmax=abs(int(INHP['MODE_X_13']) - int(INHP['MODE_J_13'])),
                      MODE_X_13=INHP['MODE_X_13'], MODE_J_13=INHP['MODE_J_13'],
                      SHARE_X_13=INHP['SHARE_X_13'], SHARE_J_13=INHP['SHARE_J_13'],
                      XH_RANGE_12=INHP['XH_RANGE_12'], FULL_SWAP_12=INHP['FULL_SWAP_12']),
    predictions=PC,
    verdict_labels={a: {k: V[a].get(k) for k in sorted(V[a])} for a in V},
    honesty=S['honesty'],
    quant=EX['quant'],
    created=time.strftime('%Y-%m-%d %H:%M:%S'),
)
jp = os.path.join(P15T, 'judgement_phase15.json')
io.open(jp, 'w', encoding='utf-8', newline='\n').write(json.dumps(judgement, ensure_ascii=False, indent=1))
w('judgement_phase15.json written (%d bytes)' % os.path.getsize(jp))

# ---------------------------------------------------------------- 2. Ledger 补登
bk = os.path.join(P15T, 'atlas_ledger_backup_pre_phase15.json')
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
already = any(int(m.get('phase', -1)) == 15 for m in LG['measurements'] if isinstance(m, dict))
w('ledger measurements before = %d ; already_has_phase15 = %s' % (n0, already))
if not already:
    shutil.copy2(LEDGER, bk)
    b_sha = sha(LEDGER)


    def fnum(x, n=6):
        return 'n/a' if x is None else ('%.*f' % (n, x))

    SITES_N = len(R['grid']['profile_sites'])
    ALPHA_N = len(R['grid']['alphas'])

    def fwd_count(a):
        """每臂真实前向数：capture + 独立定位 + 剖面 + 自检（determinism 2 / hook 1 / F3 3x3）"""
        n_cap = int(R['arms'][a]['E1_capture']['n'])
        n_loc = len(R['arms'][a].get('cands_used') or []) * 24
        n_prof = SITES_N * ALPHA_N * 24
        return n_cap + n_loc + n_prof + 12

    NFW_ARM = {a: fwd_count(a) for a in ARMS if a in R['arms']}
    TOTAL_FW = sum(NFW_ARM.values())

    rep = J['arms_used_for_cross_model']
    rev = ('deepseek/N line Phase 15 (N2h1-alpha-8), cross-model recomputation of the unified profile on THREE arms under a '
           'single nf4 (4-bit weights / bf16 compute) numerical convention. Arms: '
           + '; '.join('%s=%s(L=%d,hid=%d,tie=%s)' % (a, EX['arms'][a]['model'], R['arms'][a]['cfg']['L'],
                                                     R['arms'][a]['cfg']['hid'], R['arms'][a]['cfg']['tie'])
                       for a in ARMS) + '. ')
    rev += ('Row-count / budget: %d profile sites x %d alphas x 24 discovery pairs per arm = %d forwards per arm, plus a '
            '15-candidate independent write-window localization (B_cat, adjacent-max-increment) and a 41-instance capture; '
            'per-arm forwards = %s ; total %d real GPU forwards. ' % (
                len(R['grid']['profile_sites']), len(R['grid']['alphas']),
                len(R['grid']['profile_sites']) * len(R['grid']['alphas']) * 24,
                NFW_ARM, TOTAL_FW))
    rev += ('Why nf4: bf16 qwen3-14b (29.5GB) needs >15GB of CPU-side residency while this host has ~17GB free RAM; the bf16 '
            'auto device_map route offloaded 10 modules to DISK (layer placement L16..L39 = meta) at 7.32 s/forward '
            '(316 min/arm, infeasible), and the bf16 max_memory route was killed during load with no Python traceback. '
            'glm4-9b bf16 with max_memory was feasible (0.745 s/forward, L30..L39 meta) but would have broken the '
            'cross-arm convention. unified nf4 gives 0.036-0.041 s/forward on all three models. ')
    rev += ('Device anchors: 41/41 instances T=2 on every arm (hist %s); determinism max|dlogits| = 0.000e+00 on every arm; '
            'alpha=0 restoration max|dScore| = 0.000e+00 at the first/middle/last profile site of every arm; '
            'o_proj.in_features == n_heads * head_dim on every arm; independent localization reproduces the Phase-8 geometry. ' % (
                R['arms'][ARMS[0]]['token_len_hist'],))
    rev += ('AMEND1 (apparatus fix, no hypothesis change; sha8 %s): the FIRST formal run was intercepted by the device gate '
            'F2_base_ok on the A1 arm - the frozen panel.sup_id (fruit=104618 etc.) is the QWEN vocabulary id and was used '
            'globally, but glm4-9b-chat-hf has a DIFFERENT vocabulary (151329 vs 151643), so A1 was reading WRONG category '
            'tokens (bad=%d/41, receptor-class score mean %+.3f, FULL_SWAP %+.3f vs 4B %+.3f, flat dose-response with '
            'negative low-alpha segment). Fix: sup_id is now resolved PER ARM from that arm tokenizer with a hard assertion '
            'F1b (each of the 6 category words is exactly 1 token and decode(id) == word). A0/A2 resolve bit-identically to '
            'the frozen reference, so the fix has zero effect on the valid arms. The first run was discarded and re-run; its '
            'stdout is kept as _formal_stdout_run1_INVALID_supid.log. Without this gate the failure would have been '
            'published as the mechanism finding "the is-a relation does not hold in GLM4". ' % (
                am1_sha[:8], AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
                AM1['evidence_from_device_gate']['A1_F2_receptor_class_score_mean'],
                AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
                AM1['evidence_from_device_gate']['A0_FULL_SWAP']))
    rev += ('Q1 calibration (A0 nf4 vs Phase-12 bf16 published): max|dxhalf| = %s (tolerance %s) ; argmax_w_x nf4 = %s vs bf16 = %s ; '
            'max|drecover| = %s ; share_x nf4 = %s vs bf16 = %s ; XH_RANGE nf4 = %s vs bf16 = %s ; J ratio band [%s, %s] => %s. ' % (
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('max_abs_dxh'), 6), FL['XH_FAITHFUL_TOL'],
                (E6.get('A0_calib_qwen3-4b-nf4') or {}).get('argmax_w_x_nf4'),
                (E6.get('A0_calib_qwen3-4b-nf4') or {}).get('argmax_w_x_bf16'),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('max_abs_drecover'), 6),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('share_x_nf4'), 6),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('share_x_bf16'), 6),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('XH_RANGE_nf4'), 6),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('XH_RANGE_bf16'), 6),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('J_ratio_min'), 3),
                fnum((E6.get('A0_calib_qwen3-4b-nf4') or {}).get('J_ratio_max'), 3),
                V.get('A0_calib_qwen3-4b-nf4', {}).get('Q1_label')))
    rev += ('Q2 the two-coordinate argmax separation (Phase 13 gave 13 window units on qwen3-4b): '
            + ' ; '.join('%s d_argmax=%s (%s ; x-window w=%s [%s..%s], j-window w=%s [%s..%s])' % (
                a, V[a].get('Q2_d_argmax'), V[a].get('Q2_label'),
                (E5[a]['win_sem_x'] or {}).get('w'), (E5[a]['win_sem_x'] or {}).get('a'), (E5[a]['win_sem_x'] or {}).get('b'),
                (E5[a]['win_sem_j'] or {}).get('w'), (E5[a]['win_sem_j'] or {}).get('a'), (E5[a]['win_sem_j'] or {}).get('b'))
                for a in rep) + '. Joint verdict Q2 = %s. ' % J['Q2_joint'])
    rev += ('Q3 the permutation-null calibration (the real increment of Phase 14; Phase 14 found null95_x = 0.6998 > observed '
            'share_x = 0.5745 on qwen3-4b, i.e. the xhalf-coordinate concentration criterion had no discriminative power there): '
            + ' ; '.join('%s null95_x=%s share_x=%s margin_x=%s ; null95_j=%s share_j=%s margin_j=%s' % (
                a, fnum((E5[a]['null_x'] or {}).get('null95'), 6), fnum(E5[a]['top3_x'], 6), fnum(E5[a]['margin_x'], 6),
                fnum((E5[a]['null_j'] or {}).get('null95'), 6), fnum(E5[a]['top3_j'], 6), fnum(E5[a]['margin_j'], 6))
                for a in rep) + '. Joint verdict Q3 = %s. ' % J['Q3_joint'])
    rev += ('Q5 profile shape: '
            + ' ; '.join('%s XH_RANGE=%s spearman(xhalf,depth)=%s spearman(J,depth)=%s L*_own=%s' % (
                a, fnum(E4[a]['XH_RANGE'], 6), fnum(E5[a]['spearman_xh_depth'], 4),
                fnum(E5[a]['spearman_J_depth'], 4), E3[a]['L_star_own']) for a in rep) + '. ')
    rev += ('Pre-registered predictions: ' + ' '.join('%s=%s' % (k, PC[k]['pass_']) for k in sorted(PC)) + '. ')
    rev += ('honesty: (i) the numerical convention is nf4, NOT the bf16 of Phases 12-14; A0 is the only quantization-fidelity '
            'evidence and, if it had failed, every A1/A2 profile statement would be demoted to a descriptive nf4 statement; '
            '(ii) all three arms were intercepted at the activation level, no weight-level verification; '
            '(iii) A1 and A2 differ in BOTH family and scale, so family and scale cannot be separated; '
            '(iv) the localization arm uses a per-layer independently rebuilt U_ell and is descriptive only; '
            '(v) the concentration statistic is extremal, so no share value may be quoted without its null95 and margin; '
            '(vi) profile arms use the bare residual difference d_ell (no U projection), so the "do not inherit L6/U6" '
            'instruction is satisfied structurally.')
    entry = {
        'phase': 15,
        'name': 'n2h1a8_cross_model_unified_profile_glm4_9b_qwen3_14b_nf4',
        'seal_sha8': seal_sha[:8],
        'exec_sha8': exec_sha[:8],
        'result_sha8': res_sha[:8],
        'evidence_level': 'statistical',
        'model_scope': 'glm4-9b-chat-hf + Qwen3-14B (+ qwen3-4b calibration arm)',
        'n_rows': int(TOTAL_FW),
        'n_forwards_per_arm': NFW_ARM,
        'prereg_id': 'N2h1a8',
        'superseded_by': None,
        'verdict': '%s__%s' % (J['Q2_joint'].lower(), J['Q3_joint'].lower()),
        'rev_note': rev,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    LG['measurements'].append(entry)
    LG.setdefault('migration_history', []).append({
        'phase': 15,
        'from_version': LG.get('version'), 'to_version': LG.get('version'),
        'backup': 'tests/deepseek_temp/Phase15/atlas_ledger_backup_pre_phase15.json',
        'backup_sha256_8': sha8(bk),
        'note': ('deepseek/N line backfill round 8 (continues Phase 8-14 backfills): appended Phase 15 (N2h1-alpha-8), '
                 'the first three-arm cross-model phase of this line, executed under a unified nf4 convention because the '
                 'host cannot hold bf16 qwen3-14b. ledger_sha256_8 remains NOT recomputed (recipe unknown; stale since '
                 'Phase 8) => use per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
    })
    io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(LG, ensure_ascii=False, indent=1))
    LG2 = json.load(io.open(LEDGER, encoding='utf-8'))
    w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
      (n0, len(LG2['measurements']), b_sha[:8], sha(LEDGER)[:8], sha8(bk)))
    w('ledger tail verdict: %s' % LG2['measurements'][-1]['verdict'])
else:
    w('ledger already has phase 15 -> skip append (idempotent)')

# ---------------------------------------------------------------- 3. MEMO 基线快照
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
mlines = T.splitlines()
heads = {}
for i, l in enumerate(mlines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'pre-append',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mlines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': hashlib.sha256(mb).hexdigest()[:8],
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': sum(1 for k in heads if k.startswith('## Phase ')),
        'sections': heads,
        'note': 'Phase 15 追加前快照（Phase 14 节已在其中）。'}
io.open(os.path.join(P15T, 'memo_baseline_preappend_phase15.json'), 'w', encoding='utf-8',
        newline='\n').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings %d' %
  (base['bytes'], base['lines'], base['sha8'], base['bare_lf'], base['phase_headings']))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_o) + '\n')
print('DONE ->', OUT)
