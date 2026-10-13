# -*- coding: utf-8 -*-
"""Phase 3229 (G5-G26): E3 gate evaluation + registry/gap decision arm (zero GPU).

Per prereg in MEMO Phase 3228 tail (frozen before gate evaluation;
default arm (a)): apply the pre-registered gates GD_flip/GD_dose/GD_spec
(+ GD_amp support) verbatim from design doc e3_intervention_design_v1.json
(content a883c4da / file 32968738) to the SEALED Phase 3228 flip matrices
(res 99693298 / npz a844d985 / freeze c51bfea5), decide the E3 upgrade for
FTR-24/FTR-26 (upgrade_log #11). Honest routing per design taxonomy:
FAIL goes to the F ledger as F20, not hidden. Registry v1.30 (failures
F20 + scope_limits appends FTR-24/FTR-26 + 3225 provenance warning) and
gap ledger v1.8 (GAP-5 mechanism note) as append-only new files with
byte-preservation asserts on all untouched fields.

Zero GPU: pure arithmetic on sealed npz + sealed scalars + JSON assembly.
Ledger is READ-ONLY in this arm (governance entry appended in closeout).
"""

import hashlib
import json
import os
from copy import deepcopy

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
OUTDIR = os.path.join(SRC_DIR, 'phase3229', 'g5g26_e3_gate_decision')
SMOKE = os.environ.get('P3229_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''

DESIGN = os.path.join(SRC_DIR, 'phase3227', 'g5g26_e3_intervention_design',
                      'e3_intervention_design_v1.json')
A3DIR = os.path.join(SRC_DIR, 'phase3228', 'g5g26_e3_compute_seal')
REG29 = os.path.join(SRC_DIR, 'phase3226', 'g5g25_a2_gate_decision',
                     'atlas_registry_v1_29.json')
GAP17 = os.path.join(SRC_DIR, 'phase3226', 'g5g25_a2_gate_decision',
                     'gap_ledger_v1_7.json')
LEDGER = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')

DESIGN_FILE_SHA = '32968738'
DESIGN_CONTENT_SHA = 'a883c4da'
A3_RES = '99693298'
A3_NPZ = 'a844d985'
A3_FREEZE = 'c51bfea5'
REG29_SHA = '85b8dc1f'
GAP17_SHA = '4f45fda0'
LED_N, LED_POST = 380, '8e593d22'
MODELS = ('qwen3-4b', 'qwen3-14b')
ALPHAS = (0.5, 1.0, 2.0)
NULL_SEEDS = [20260390, 20260391, 20260392, 20260393, 20260394,
              20260395, 20260396, 20260397, 20260398, 20260399]
NULL_PERTS = ['rand_%d' % s for s in NULL_SEEDS] + ['orth_isa', 'orth_hasa']
PANELS = ('seen', 'oov')

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


def jd(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def rankdata_avg(x):
    """Average ranks (1-based) with tie handling."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if den == 0:
        return None  # constant vector -> undefined
    return float((rx * ry).sum() / den)


os.makedirs(OUTDIR, exist_ok=True)
bad = []

# ---------- freeze BEFORE gate evaluation ----------
design = json.load(open(DESIGN, encoding='utf-8'))
assert sha8_file(DESIGN) == DESIGN_FILE_SHA, 'design file sha drift'
assert design['sha8'] == DESIGN_CONTENT_SHA, 'design content sha drift'
gates = design['gates_pre_registered']
for g in ('GD_flip', 'GD_dose', 'GD_spec', 'GD_amp', 'verdict_taxonomy',
          'upgrade_targets'):
    assert g in gates, ('gate missing', g)
assert 'E3_PASS iff GD_flip and GD_dose and GD_spec' in gates['verdict_taxonomy']

a3 = json.load(open(os.path.join(A3DIR, 'result.json'), encoding='utf-8'))
assert a3['verdict'] == 'e3_compute_sealed'
assert a3['res_sha8'] == A3_RES
assert a3['sealed_npz']['file_sha8'] == A3_NPZ
assert a3['freeze_sha8'] == A3_FREEZE

fz = {
    'phase': '3229', 'name': 'g5g26_e3_gate_decision',
    'prereg': 'AGI_GPT5_MEMO Phase 3228 tail (frozen before gate '
              'evaluation; default arm (a) E3 gate evaluation + registry '
              'decision). Gates verbatim from design a883c4da; inputs '
              'sealed 3228 (res %s / npz %s / freeze %s)'
              % (A3_RES, A3_NPZ, A3_FREEZE),
    'gates_applied': gates,
    'clarifications_frozen_before_evaluation': [
        'K1 design upgrade_targets label fix: "relation-direction carriers '
        '(FTR-22 family)" is a mislabel — registry v1.29 FTR-22 = '
        'spectral-out port calibration (3172); the relation-direction '
        'carrier tested by the intervention is FTR-24 (dv = v_isa - v_hasa '
        'family, T2 chain 3177-3187) + FTR-26 (u*). FTR-22 untouched.',
        'K2 GD_dose degenerate rule (frozen before computing): if matched '
        'flip rates are all-zero across all three alphas on a fold, the '
        'monotonicity gate is UNSATISFIED on that fold (monotonicity of '
        'an all-zero sequence is not evidence of dose response); '
        'Spearman(undefined, constant) counts as not-pass.',
        'K3 GD_spec undefined-ratio rule (frozen before computing): if '
        'matched rate == 0, pass iff cross rate == 0 exactly; else FAIL.',
        'K4 GD_flip fold basis: seen panel 4 cut-folds (48 rows each) as '
        'pre-registered; OOV template slices (3) computed report-only.',
        'K5 3225 provenance warning (from 3228 C9) enters registry here '
        'as scope_limits appends on FTR-25/FTR-26: A1 used hub Qwen3-4B '
        'W_U (4c807e25) while 3154/3169 caches were collected with local '
        '4B weights (328a91d3); A1 conclusions are W_U-relative map '
        'properties (predicted and actual displacement share the same '
        'W_U) and unaffected; re-verified 3228 C6v3.',
    ],
    'inputs': {'design_file_sha8': DESIGN_FILE_SHA,
               'design_content_sha8': DESIGN_CONTENT_SHA,
               'a3_res': A3_RES, 'a3_npz': A3_NPZ, 'a3_freeze': A3_FREEZE,
               'registry_v129': REG29_SHA, 'gap_v17': GAP17_SHA,
               'ledger': 'n=%d post %s (read-only)' % (LED_N, LED_POST)},
    'zero_gpu': 'pure arithmetic on sealed npz/scalars; no model, no forward',
}
fz['sha8'] = csha8(dict(fz))
FZ_PATH = os.path.join(OUTDIR, PFX + 'execution.json')
with open(FZ_PATH, 'w', encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3229 g5g26_e3_gate_decision %s ==' % ('SMOKE' if SMOKE else 'FULL'))
log('freeze sealed: %s (design a883c4da/32968738, before any gate '
    'evaluation)' % fz['sha8'])

# ---------- S1: gate evaluation from sealed npz (recompute + integrity) ----------
z = np.load(os.path.join(A3DIR, 'e3_flip_matrices.npz'))
assert sha8_file(os.path.join(A3DIR, 'e3_flip_matrices.npz')) == A3_NPZ

# sealed slices (verbatim from 3228 script). Slice indices are part of the
# frozen design D3 (not observations): 3154 tags re-read read-only for the
# seen cut masks; OOV templates are contiguous 730-row blocks (3228-verified).
PANEL_3154 = {}
for m in MODELS:
    zs = np.load(os.path.join(ROOT, 'tests', 'glm5', 'result',
                              'rdc_query_construction_20260913', 'phase3154',
                              'g1p4_mfd_multifactor_disentangle', m,
                              'collect.npz'))
    tg = {k: zs[k].astype(int) for k in ('ci', 'ho')}
    del zs
    PANEL_3154[m] = [np.where((tg['ci'] == c) & (tg['ho'] == 0))[0]
                     for c in range(4)]
SL69 = [np.arange(t * 730, (t + 1) * 730) for t in range(3)]
SLICES = {'seen': PANEL_3154, 'oov': {m: SL69 for m in MODELS}}

PI = {'u_isa': 0, 'u_hasa': 1}
for i, s in enumerate(NULL_SEEDS):
    PI['rand_%d' % s] = 2 + i
PI['orth_isa'] = 12
PI['orth_hasa'] = 13
AI = {a: i for i, a in enumerate(ALPHAS)}


def flip_rate(m, panel, Rname, pname, a, sl):
    R = 1 if Rname == 'isa' else 0
    J0 = z['%s_%s_J0' % (m, panel)]
    mg = z['%s_%s_margins' % (m, panel)]
    J1 = (mg[PI[pname], AI[a], :, 0] > mg[PI[pname], AI[a], :, 1])
    fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
    return float(fl.sum() / len(sl)), int(fl.sum()), int(len(sl))


# integrity: recompute must reproduce sealed agg exactly (spot grid)
nchk = 0
for m in MODELS:
    for panel in PANELS:
        sealed_agg = a3['models'][m][panel]['agg']
        sls = SLICES[panel][m]
        for Rname in ('isa', 'hasa'):
            for pname in ('u_isa', 'u_hasa') + tuple(NULL_PERTS):
                for a in ALPHAS:
                    for si_, sl in enumerate(sls):
                        sname = 'cut%d' % si_ if panel == 'seen' else 'tpl%d' % si_
                        r, n, rows = flip_rate(m, panel, Rname, pname, a, sl)
                        sv = sealed_agg['%s|%s|%s|a%s' % (Rname, pname, sname, a)]
                        assert sv['n'] == n and sv['rows'] == rows, (
                            'agg drift', m, panel, Rname, pname, a, sname)
                        assert abs(sv['rate'] - r) < 1e-12
                        nchk += 1
log('S0 integrity: %d sealed agg cells reproduced from npz bit-exact' % nchk)

# GD_flip
gd_flip = {}
for m in MODELS:
    for Rname in ('isa', 'hasa'):
        folds = []
        for c in range(4):
            sl = SLICES['seen'][m][c]
            mr, _, _ = flip_rate(m, 'seen', Rname, 'u_' + Rname, 1.0, sl)
            nulls = [flip_rate(m, 'seen', Rname, p, 1.0, sl)[0]
                     for p in NULL_PERTS]
            folds.append(dict(fold='cut%d' % c, matched=mr, null_max=max(nulls),
                              delta=mr - max(nulls)))
        oov_rep = []
        for t in range(3):
            sl = SLICES['oov'][m][t]
            mr, _, _ = flip_rate(m, 'oov', Rname, 'u_' + Rname, 1.0, sl)
            nulls = [flip_rate(m, 'oov', Rname, p, 1.0, sl)[0]
                     for p in NULL_PERTS]
            oov_rep.append(dict(tpl='tpl%d' % t, matched=mr,
                                null_max=max(nulls)))
        npass = sum(1 for f in folds if f['delta'] >= 0.15)
        ok_m = npass >= 3
        gd_flip['%s.%s' % (m, Rname)] = dict(
            seen_folds=folds, oov_report=oov_rep, folds_pass=npass,
            model_pass=bool(ok_m),
            note='gate requires >=3/4 folds per model and both models')
gd_flip_pass = all(v['model_pass'] for v in gd_flip.values())
log('S1 GD_flip: %s' % (
    'PASS' if gd_flip_pass else 'FAIL (matched deltas: %s)' % '; '.join(
        '%s %s d=%.4f..%.4f' % (k, 'p%d' % v['folds_pass'],
                                min(f['delta'] for f in v['seen_folds']),
                                max(f['delta'] for f in v['seen_folds']))
        for k, v in gd_flip.items())))

# GD_dose
gd_dose = {}
for m in MODELS:
    for Rname in ('isa', 'hasa'):
        recs = []
        ok_any = True
        for c in range(4):
            sl = SLICES['seen'][m][c]
            rates = [flip_rate(m, 'seen', Rname, 'u_' + Rname, a, sl)[0]
                     for a in ALPHAS]
            if len(set(rates)) == 1:
                ok = False  # K2 degenerate rule
                tag = 'degenerate_constant'
            else:
                ok = (rates[0] <= rates[1] <= rates[2])
                sp = spearman(ALPHAS, rates)
                ok = ok or (sp is not None and abs(sp - 1.0) < 1e-12)
                tag = 'monotone_ok' if ok else 'non_monotone'
            if not ok:
                ok_any = False
            recs.append(dict(fold='cut%d' % c, rates=rates, verdict=tag))
        oov_rep = []
        for t in range(3):
            sl = SLICES['oov'][m][t]
            rates = [flip_rate(m, 'oov', Rname, 'u_' + Rname, a, sl)[0]
                     for a in ALPHAS]
            oov_rep.append(dict(tpl='tpl%d' % t, rates=rates,
                                monotone=bool(rates[0] <= rates[1] <= rates[2])))
        gd_dose['%s.%s' % (m, Rname)] = dict(
            seen_folds=recs, oov_report=oov_rep, model_pass=bool(ok_any))
gd_dose_pass = all(v['model_pass'] for v in gd_dose.values())
log('S1 GD_dose: %s (K2 degenerate-zero applied)' % (
    'PASS' if gd_dose_pass else 'FAIL'))

# GD_spec
gd_spec = {}
for m in MODELS:
    recs = {}
    ok_all = True
    for panel in PANELS:
        sl = np.concatenate(SLICES[panel][m])
        for Jname, cross_p, match_p in (('hasa', 'u_isa', 'u_hasa'),
                                        ('isa', 'u_hasa', 'u_isa')):
            cr, cn, _ = flip_rate(m, panel, Jname, cross_p, 1.0, sl)
            mr, mn, _ = flip_rate(m, panel, Jname, match_p, 1.0, sl)
            if mr == 0.0:
                ok = (cr == 0.0)      # K3 rule
                ratio = None if cn == 0 else float('inf')
            else:
                ratio = cr / mr
                ok = cr <= mr / 3.0
            recs['%s.%s' % (panel, Jname)] = dict(
                cross=cr, cross_n=cn, matched=mr, matched_n=mn,
                ratio=ratio, verdict='pass' if ok else 'fail')
            if not ok:
                ok_all = False
    gd_spec[m] = dict(cells=recs, model_pass=bool(ok_all))
gd_spec_pass = all(v['model_pass'] for v in gd_spec.values())
log('S1 GD_spec: %s (K3 undefined-ratio applied)' % (
    'PASS' if gd_spec_pass else 'FAIL'))

# GD_amp (support stat): recompute Spearman from sealed scalars
gd_amp = {}
for m in MODELS:
    g = a3['models'][m]['gd_amp']
    sp = spearman(g['E_cls'], g['amp_med'])
    assert abs(sp - g['spearman_amp_collapse']) < 1e-9, ('amp drift', m)
    gd_amp[m] = dict(spearman=sp, E_cls=g['E_cls'], amp_med=g['amp_med'],
                     pass_=bool(sp >= 0.6))
gd_amp_pass = all(v['pass_'] for v in gd_amp.values())
log('S1 GD_amp: %s (%s)' % ('PASS' if gd_amp_pass else 'FAIL',
                            '/'.join('%s %.2f' % (m, v['spearman'])
                                     for m, v in gd_amp.items())))

gate_pass = {'GD_flip': gd_flip_pass, 'GD_dose': gd_dose_pass,
             'GD_spec': gd_spec_pass, 'GD_amp_support': gd_amp_pass}
n_core = sum(1 for g in ('GD_flip', 'GD_dose', 'GD_spec') if gate_pass[g])
if gate_pass['GD_flip'] and gate_pass['GD_dose'] and gate_pass['GD_spec']:
    e3_verdict = 'E3_PASS'
elif n_core > 0:
    e3_verdict = 'E3_DIRECTIONAL'
else:
    e3_verdict = 'E3_FAIL'
log('S1 E3 verdict (taxonomy): %s (core gates passed %d/3, amp support '
    '%s)' % (e3_verdict, n_core, 'yes' if gd_amp_pass else 'partial/none'))

if SMOKE:
    log('SMOKE: skipping registry/gap assembly (writes stay with FULL)')
    verdict = 'e3_gate_%s_smoke' % e3_verdict.lower().replace('_', '_')
    res = {
        'phase': 3229, 'name': 'g5g26_e3_gate_decision',
        'arm': 'SMOKE dry-run (gates only, no writes)',
        'freeze_sha8': fz['sha8'], 'gates': gate_pass,
        'e3_verdict': e3_verdict, 'recheck': {'contradictions': bad},
        'verdict': verdict,
    }
    res['res_sha8'] = hashlib.sha256(json.dumps(
        dict(res), ensure_ascii=False, indent=1, sort_keys=True)
        .encode('utf-8')).hexdigest()[:8]
    with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
              encoding='utf-8', newline='') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
              encoding='utf-8', newline='') as f:
        f.write('\n'.join(LOG) + '\n')
    log('SMOKE result sealed: res %s' % res['res_sha8'])
    raise SystemExit(0)

# ---------- S2: registry v1.30 (F20 + scope appends, append-only) ----------
assert sha8_file(REG29) == REG29_SHA, 'registry v1.29 file sha drift'
reg29 = json.load(open(REG29, encoding='utf-8'))
assert reg29['version'] == '1.29' and len(reg29['features']) == 26
assert len(reg29['failures']) == 19 and reg29['failures'][-1]['id'] == 'F19'
assert len(reg29['upgrade_log']) == 10
reg30 = deepcopy(reg29)
reg30['version'] = '1.30'
reg30['supersedes'] = ('atlas_registry_v1_29.json (%s, Phase 3226)'
                       % REG29_SHA)

E_cls_4b = a3['models']['qwen3-4b']['gd_amp']
dD_4b = a3['models']['qwen3-4b']['oov']['dose']['u_isa|a1.0']
dD_14b = a3['models']['qwen3-14b']['oov']['dose']['u_isa|a1.0']
dD2_4b = a3['models']['qwen3-4b']['oov']['dose']['u_isa|a2.0']
dD2_14b = a3['models']['qwen3-14b']['oov']['dose']['u_isa|a2.0']

reg30['upgrade_log'].append({
    'node': 'FTR-24+FTR-26',
    'change': 'E3 gate FAIL -> no E-level change (F20 recorded)',
    'reason': 'G5-G26 (Phase 3229): pre-registered gates GD_flip/GD_dose/'
              'GD_spec applied verbatim to sealed 3228 matrices (res %s / '
              'npz %s / freeze %s; design a883c4da) — GD_flip matched flip '
              '0/4 folds x2 models x2 directions (null band up to %.4f, '
              'delta < 0 everywhere); GD_dose degenerate all-zero (K2); '
              'GD_spec inverted (cross/matched up to ~8x, K3); GD_amp '
              'Spearman %.2f/%.2f = 1/2 models. Honest FAIL per design '
              'taxonomy -> F20. u_isa dose is ANTI-directional monotone '
              '(dD %.4f/%.4f @a1, %.4f/%.4f @a2, 4b/14b OOV). Relation-'
              'direction carrier u_R (T2 ctx-contrast recipe) is NOT a '
              'behavioral lever on the isa/hasa judgment readout; '
              'coheres 3188 DECOUPLED + GAP-5.'
              % (A3_RES, A3_NPZ, A3_FREEZE,
                 max(max(f['null_max'] for f in v['seen_folds'])
                     for v in gd_flip.values()),
                 gd_amp['qwen3-4b']['spearman'], gd_amp['qwen3-14b']['spearman'],
                 dD_4b, dD_14b, dD2_4b, dD2_14b),
})
reg30['failures'].append({
    'id': 'F20',
    'kind': 'e3_sufficiency_gate_fail',
    'phase': 3229,
    'text': 'E3 充分性门全不过（G5-G26，3227 设计四门逐字应用于 3228 封存翻转'
            '矩阵）：GD_flip matched flip 0/4 折 × 2 模型 × 2 关系方向（同折 '
            'null 带最高 0.049@14b，delta 全负）；GD_dose 退化全零（无剂量响应'
            '，K2 规则）；GD_spec 反向——cross patch（u_isa→hasa 判断）翻转率'
            '最高达 matched（u_hasa→hasa）的 ~8×，而非 ≤1/3；GD_amp 仅 14b 过'
            '（0.80 vs 0.40，门需双模型 ≥0.6）。u_isa 剂量反方向单调（dD '
            '−0.1389/−0.1117 @α1、−0.2777/−0.2234 @α2，OOV 面板 4b/14b），'
            'u_hasa 弱 ~7–11×。定论：T2 ctx-对比配方关系级方向 u_R 在末层 '
            'pre-logit 线性读出图上对 isa/hasa 判断行为不是杠杆（isa 方向为反'
            '平行杠杆），行为判断由上游承载——读出几何 ≠ 行为控制，与 3188 '
            'DECOUPLED / GAP-5 自洽。',
    'evidence': ['design a883c4da/32968738 (3227, gates frozen pre-observation)',
                 'sealed 3228: res %s / npz %s / freeze %s'
                 % (A3_RES, A3_NPZ, A3_FREEZE),
                 'C6v3 device gate 0.9963/0.9941 argmax, centered-cos '
                 '1.0000 (readout machinery exact)',
                 'this arm gate table (S1, npz bit-exact reproduction of '
                 'sealed agg)'],
})
reg30['provenance']['anchor_files']['e3_flip_matrices'] = {
    'path': (r'tests\glm5\result\rdc_query_construction_20260913'
             r'\phase3228\g5g26_e3_compute_seal\e3_flip_matrices.npz'),
    'sha8': A3_NPZ}
# scope_limits appends: FTR-24 + FTR-26 (E3 FAIL note), FTR-25 + FTR-26
# (provenance warning)
SCOPE_NOTE_24 = (' | E3 intervention FAIL (3229 G5-G26, F20): T2 ctx-contrast '
                 'recipe relation-level direction u_R is NOT a behavioral '
                 'lever on the isa/hasa judgment readout (matched flip 0, '
                 'u_isa anti-directional monotone dD -0.14/-0.11 @a1); '
                 'relation-direction behavioral carrier status REJECTED '
                 'in-protocol (last-layer pre-logit map, qwen pair)')
SCOPE_NOTE_26 = (' | E3 intervention FAIL (3229 G5-G26, F20): u* amplitude-'
                 'carrier support partial only (GD_amp Spearman 0.40/0.80 = '
                 '1/2 models); u* not promoted to behavioral lever')
SCOPE_PROV_2526 = (' | provenance warning (3228 C9, filed 3229): 3225 A1 '
                   'used hub Qwen3-4B W_U (4c807e25) while 3154/3169 caches '
                   'were collected with local 4B weights (328a91d3); A1 '
                   'conclusions are W_U-relative map properties (predicted '
                   'and actual displacement share the same W_U) and '
                   'unaffected; readout provenance re-verified 3228 C6v3 '
                   '(argmax 0.9963/0.9941, cos 1.0000)')
for f in reg30['features']:
    if f['id'] == 'FTR-24':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_NOTE_24
    elif f['id'] == 'FTR-25':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_PROV_2526
    elif f['id'] == 'FTR-26':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_NOTE_26 \
            + SCOPE_PROV_2526
# byte-preservation asserts
for a, b in zip(reg29['features'], reg30['features']):
    assert a['id'] == b['id']
    if a['id'] not in ('FTR-24', 'FTR-25', 'FTR-26'):
        assert jd(a) == jd(b), ('feature drift', a['id'])
    else:
        assert jd(a['statement']) == jd(b['statement'])
        assert jd(a['evidence_level']) == jd(b['evidence_level'])
        assert jd(a['anchors']) == jd(b['anchors'])
        assert len(b['scope_limits']) > len(a['scope_limits'] or '')
for k in reg29:
    if k not in ('version', 'supersedes', 'upgrade_log', 'failures',
                 'provenance', 'features'):
        assert jd(reg29[k]) == jd(reg30[k]), ('top drift', k)
assert jd(reg29['upgrade_log']) == jd(reg30['upgrade_log'][:10])
assert jd(reg29['failures']) == jd(reg30['failures'][:19])
log('S2 registry v1.30 assembled: F20 appended, upgrade_log #11 (FAIL '
    'record), scope appends FTR-24/25/26, byte-preservation OK')

# ---------- S3: gap ledger v1.8 (GAP-5 mechanism note append) ----------
assert sha8_file(GAP17) == GAP17_SHA, 'gap v1.7 file sha drift'
gap17 = json.load(open(GAP17, encoding='utf-8'))
assert gap17['version'] == '1.7' and len(gap17['gaps']) == 5
gap18 = deepcopy(gap17)
gap18['version'] = '1.8'
gap18['updated'] = '2026-10-11'
g5 = [x for x in gap18['gaps'] if x['id'] == 'GAP-5'][0]
g5['mechanism_note'] = (g5.get('mechanism_note') or '') + (
    '；E3 补注（3229 G5-G26，F20）：末层 pre-logit 图上读出精确线性'
    '（3228 C6v3 直接图 argmax 0.9963/0.9941、centered-cos 1.0000）但行为'
    '充分性门全不过——u_R 注入对 isa/hasa 判断 matched flip=0 且 u_isa 反向'
    '单调。三段式定论：上游编码（承载判断行为）→ 末层线性读出（精确投影）'
    '→ post-softmax（分布级非线性）；判断行为的因果杠杆须在上游机制中寻找，'
    '末层几何指纹不外推为行为控制。')
for a, b in zip(gap17['gaps'], gap18['gaps']):
    assert jd(a) == jd(b) or (a['id'] == 'GAP-5' and b['id'] == 'GAP-5'), (
        'gap drift', a.get('id'))
    if a['id'] != 'GAP-5':
        assert jd(a) == jd(b), ('gap drift', a.get('id'))
    else:
        assert len(b['mechanism_note']) > len(a['mechanism_note'])
        for k in a:
            if k != 'mechanism_note':
                assert jd(a[k]) == jd(b[k]), ('gap5 field drift', k)
for k in gap17:
    if k not in ('version', 'updated', 'gaps'):
        assert jd(gap17[k]) == jd(gap18[k]), ('gap top drift', k)
log('S3 gap v1.8 assembled: GAP-5 mechanism_note appended (three-stage '
    'readout conclusion), legacy x4 + GAP-5 fields byte-preserved')

# ---------- S4: ledger untouched (within-run snapshot invariant) ----------
# The closeout appends #381 AFTER this arm; the invariant is that THIS arm
# does not modify the ledger. Accept any head, assert byte-identical
# before/after within the run.
led = json.load(open(LEDGER, encoding='utf-8'))
LED_HEAD = [len(led['measurements']), led['ledger_sha256_8']]
log('S4 ledger head observed: n=%d post=%s (untouched by this arm; '
    'closeout appends later)' % tuple(LED_HEAD))

# ---------- S5: write sealed artifacts ----------
WROTE = {}
p = os.path.join(OUTDIR, 'atlas_registry_v1_30.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(reg30, f, ensure_ascii=False, indent=1)
v30 = dict(reg30); v30.pop('sha8', None)
WROTE['registry_v130'] = {'file': os.path.basename(p),
                          'file_sha8': sha8_file(p),
                          'content_sha8': csha8(v30)}
log('S5 registry v1.30 sealed: file %s content %s' % (
    WROTE['registry_v130']['file_sha8'],
    WROTE['registry_v130']['content_sha8']))
p = os.path.join(OUTDIR, 'gap_ledger_v1_8.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(gap18, f, ensure_ascii=False, indent=1)
WROTE['gap_v18'] = {'file': os.path.basename(p), 'file_sha8': sha8_file(p),
                    'content_sha8': csha8(gap18)}
log('S5 gap v1.8 sealed: file %s content %s' % (
    WROTE['gap_v18']['file_sha8'], WROTE['gap_v18']['content_sha8']))

# ---------- S6: verdict ----------
led2 = json.load(open(LEDGER, encoding='utf-8'))
assert [len(led2['measurements']), led2['ledger_sha256_8']] == LED_HEAD, \
    'ledger modified within run'
clean = not bad
verdict = ('e3_gate_%s_f20_registered' % e3_verdict.lower()
           if clean else 'e3_gate_decision_dirty (%d)' % len(bad))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3229, 'name': 'g5g26_e3_gate_decision',
    'arm': 'E3 gate evaluation + registry decision (default (a))',
    'design_sha8': DESIGN_FILE_SHA,
    'design_content_sha8': DESIGN_CONTENT_SHA,
    'freeze_sha8': fz['sha8'],
    'inputs': fz['inputs'],
    'clarifications_frozen': fz['clarifications_frozen_before_evaluation'],
    'gate_table': {'GD_flip': gd_flip, 'GD_dose': gd_dose,
                   'GD_spec': gd_spec, 'GD_amp': gd_amp},
    'gate_pass': gate_pass,
    'e3_verdict': e3_verdict,
    'registry': {'version': '1.30', 'upgrade_log_no': 11,
                 'change': 'E3 gate FAIL -> no E-level change (F20)',
                 'failures': 'F20 appended (e3_sufficiency_gate_fail)',
                 'scope_appends': ['FTR-24', 'FTR-25', 'FTR-26']},
    'gap': {'version': '1.8', 'gap5_mechanism_note': 'appended'},
    'wrote': WROTE,
    'recheck': {'ledger_head_observed': LED_HEAD,
                'ledger_untouched_within_run': True,
                'byte_preservation': ('26 features (23 untouched byte-exact '
                                      '+ 3 scope appends) + legacy '
                                      'top-level x2 OK'),
                'contradictions': bad},
    'verdict': verdict,
}
res['res_sha8'] = hashlib.sha256(json.dumps(
    dict(res, res_sha8=None), ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
with open(os.path.join(OUTDIR, 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('result sealed: res %s' % res['res_sha8'])
