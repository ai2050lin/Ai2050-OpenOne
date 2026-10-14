# -*- coding: utf-8 -*-
"""Phase 3231 (G5-G28): registry decision arm (zero GPU).

Per prereg in MEMO Phase 3230 tail (default arm (a)): route the sealed 3230
verdict dR_lever_fail into the registries —
  (1) F20 refinement append (recipe-generalization verdict: the state-
      contrast direction dR is also NOT a behavioral lever + transfer-
      orthogonality mechanism cos(dR,dw)=+0.0997/+0.0625 + decoupling
      diagnostics cos(dR,v_isa)=-0.2463/-0.0547);
  (2) GAP-5 mechanism_note append (second-recipe confirmation of the
      three-stage picture);
  (3) FTR-24/25/26 scope_limits notes.
Registry v1.30 -> v1.31, gap ledger v1.8 -> v1.9, append-only new files
with byte-preservation asserts on all untouched fields.

Zero GPU: pure arithmetic on the sealed 3230 npz + JSON assembly.
Ledger is READ-ONLY in this arm (governance entry appended in closeout).
"""

import hashlib
import json
import os
from copy import deepcopy

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
OUTDIR = os.path.join(SRC_DIR, 'phase3231', 'g5g28_registry_decision')
SMOKE = os.environ.get('P3231_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''

A4DIR = os.path.join(SRC_DIR, 'phase3230', 'g5g27_relcontrast_efficacy')
REG30 = os.path.join(SRC_DIR, 'phase3229', 'g5g26_e3_gate_decision',
                     'atlas_registry_v1_30.json')
GAP18 = os.path.join(SRC_DIR, 'phase3229', 'g5g26_e3_gate_decision',
                     'gap_ledger_v1_8.json')
LEDGER = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')

A4_RES = '1b63b063'
A4_NPZ = 'd430d29e'
A4_FREEZE = '1d02dd97'
REG30_SHA = 'c11d13bd'
GAP18_SHA = '7065ec4e'
LED_N, LED_POST = 382, '3cba5317'
MODELS = ('qwen3-4b', 'qwen3-14b')
ALPHAS = (0.5, 1.0, 2.0)

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


os.makedirs(OUTDIR, exist_ok=True)
bad = []

# ---------- freeze BEFORE any registry assembly ----------
a4 = json.load(open(os.path.join(A4DIR, 'result.json'), encoding='utf-8'))
assert a4['verdict'] == 'dR_lever_fail'
assert a4['res_sha8'] == A4_RES
assert a4['sealed_npz']['file_sha8'] == A4_NPZ
assert a4['freeze_sha8'] == A4_FREEZE
assert a4['gates_global']['GA_flip'] is False
assert a4['gates_global']['GB_dir'] is True

fz = {
    'phase': '3231', 'name': 'g5g28_registry_decision',
    'prereg': 'AGI_GPT5_MEMO Phase 3230 tail (default arm (a) G5-G28 '
              'registry decision): F20 refinement append + GAP-5 '
              'mechanism_note append + FTR-24/25/26 scope notes; registry '
              'v1.31, gap v1.9. Inputs sealed 3230 (res %s / npz %s / '
              'freeze %s)' % (A4_RES, A4_NPZ, A4_FREEZE),
    'clarifications_frozen_before_assembly': [
        'K1 pure registration arm: every scientific verdict of 3230 '
        '(dR_lever_fail, GA_flip/GB_dir gates, diagnostic cosines) was '
        'frozen pre-observation in 3230 freeze 1d02dd97; this arm makes '
        'NO new observation, re-opens NO gate, and only executes the '
        'registry/gap appends routed by the 3230 prereg.',
        'K2 refinement landing: the F20 failure entry gains a NEW key '
        '"refinements" (append-only list); F20 pre-existing keys '
        '(id/kind/phase/text/evidence) stay byte-exact; upgrade_log '
        'untouched (the FAIL is already recorded as #11, E-levels '
        'unchanged); NO new failure entry is created.',
        'K3 scope-note semantics: the dR second-recipe FAIL is further '
        'negative evidence on BEHAVIORAL-LEVER status only; it does not '
        'change the geometric-band E-levels (E2_held_out etc.) or the '
        'existing anchors of FTR-24/25/26.',
        'K4 GA/GB reading (inherited from 3230 freeze): GB_dir PASS only '
        'shows linear-map self-consistency (dR pushes the margin in its '
        'own direction); the behavioral-lever gate is GA_flip.',
        'K5 ledger read-only in arm (n=382 post 3cba5317); governance '
        'entry #383 appended in closeout; glm4 deferred (2/3 model '
        'scope).',
    ],
    'inputs': {'a4_res': A4_RES, 'a4_npz': A4_NPZ, 'a4_freeze': A4_FREEZE,
               'registry_v130': REG30_SHA, 'gap_v18': GAP18_SHA,
               'ledger': 'n=%d post %s (read-only)' % (LED_N, LED_POST)},
    'zero_gpu': 'pure arithmetic on sealed npz/scalars + JSON assembly; '
                'no model, no forward',
}
fz['sha8'] = csha8(dict(fz))
FZ_PATH = os.path.join(OUTDIR, PFX + 'execution.json')
with open(FZ_PATH, 'w', encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3231 g5g28_registry_decision %s ==' % ('SMOKE' if SMOKE else 'FULL'))
log('freeze sealed: %s (before any registry assembly)' % fz['sha8'])

# ---------- S1: integrity recompute from sealed 3230 npz ----------
NPZP = os.path.join(A4DIR, 'e3_relcontrast_flip.npz')
assert sha8_file(NPZP) == A4_NPZ
z = np.load(NPZP)

INTEG = {}
for m in MODELS:
    mg = z[m + '_seen_margins']
    m0 = z[m + '_seen_m0']
    d0 = m0[:, 0] - m0[:, 1]
    rec = {'d0_mean': float(d0.mean())}
    assert abs(rec['d0_mean'] - a4['models'][m]['seen']['d0_mean']) < 1e-9
    # GB_dir dose recompute (pert index 0 = dR)
    for ai, a in enumerate(ALPHAS):
        dd = float(((mg[0, ai, :, 0] - mg[0, ai, :, 1]) - d0).mean())
        sv = a4['models'][m]['seen']['dose']['dR|a%s' % a]
        assert abs(dd - sv) < 1e-9, ('dD drift', m, a, dd, sv)
        rec['dD_a%s' % a] = dd
    rec['GB_monotone'] = bool(rec['dD_a0.5'] < rec['dD_a1.0'] < rec['dD_a2.0']
                              and rec['dD_a0.5'] > 0)
    # GA_flip recompute at a=1.0 on the 4 seen cut folds (3154 masks)
    zs = np.load(os.path.join(ROOT, 'tests', 'glm5', 'result',
                              'rdc_query_construction_20260913', 'phase3154',
                              'g1p4_mfd_multifactor_disentangle', m,
                              'collect.npz'))
    tg = {k: zs[k].astype(int) for k in ('ci', 'ho')}
    del zs
    J0 = z[m + '_seen_J0']
    jud = (mg[0, 1, :, 0] > mg[0, 1, :, 1]).astype(np.int8)
    flips = []
    for c in range(4):
        sl = np.where((tg['ci'] == c) & (tg['ho'] == 0))[0]
        cnt = 0
        for Rv in (0, 1):
            fl = (J0[sl] == (1 - Rv)) & (jud[sl] == Rv)
            cnt += int(fl.sum())
        flips.append(cnt)
    rec['GA_flip_counts_a1'] = flips
    assert flips == [0, 0, 0, 0], ('GA flip recompute nonzero', m, flips)
    # cos(dR, dR_raw) recompute
    rec['cos_dR_dRraw'] = float(z[m + '_dR_unit'] @ z[m + '_dR_raw_unit'])
    assert abs(rec['cos_dR_dRraw']
               - a4['models'][m]['dR']['diag']['cos_dR_dRraw']) < 1e-6
    INTEG[m] = rec
log('S1 integrity: d0/dD(x3)/GA_flip(4 folds)/cos(dR,dR_raw) recomputed '
    'bit-exact from sealed npz x2 models; GB monotone both True; GA all 0')

if SMOKE:
    res = {
        'phase': 3231, 'name': 'g5g28_registry_decision',
        'arm': 'SMOKE dry-run (integrity only, no writes)',
        'freeze_sha8': fz['sha8'], 'integrity': INTEG,
        'recheck': {'contradictions': bad},
        'verdict': 'f20_refinement_smoke',
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

# ---------- S2: registry v1.31 (F20 refinement + scope notes) ----------
assert sha8_file(REG30) == REG30_SHA, 'registry v1.30 file sha drift'
reg30 = json.load(open(REG30, encoding='utf-8'))
assert reg30['version'] == '1.30' and len(reg30['features']) == 26
assert len(reg30['failures']) == 20 and reg30['failures'][-1]['id'] == 'F20'
assert len(reg30['upgrade_log']) == 11
reg31 = deepcopy(reg30)
reg31['version'] = '1.31'
reg31['supersedes'] = ('atlas_registry_v1_30.json (%s, Phase 3229)'
                       % REG30_SHA)

GB_4b = a4['models']['qwen3-4b']['seen']['dose']
GB_14b = a4['models']['qwen3-14b']['seen']['dose']
DG_4b = a4['models']['qwen3-4b']['dR']['diag']
DG_14b = a4['models']['qwen3-14b']['dR']['diag']
D0 = {m: a4['models'][m]['seen']['d0_mean'] for m in MODELS}
NORM = {m: a4['models'][m]['dR']['norm_post'] for m in MODELS}

f20 = reg31['failures'][-1]
assert 'refinements' not in f20
f20['refinements'] = [{
    'phase': 3231, 'source_phase': 3230, 'arm': 'G5-G27 -> G5-G28',
    'kind': 'recipe_generalization_second_recipe',
    'text': ('配方普适化定论（3230 G5-G27 实测 → 3231 G5-G28 入册）：状态对比'
             '方向 dR = mean(h[isa 64 行]) − mean(h[hasa 64 行])（3157 面板，'
             'NH−1 post-norm pre-logit 槽，|dR|=%.2f/%.2f ≈ 行态 RMS 的 '
             '7–9×）在末层 pre-logit 精确线性读出图上同样不是行为杠杆——'
             'GA_flip 0/4 折 × 2 模型 × 全 α（null 带亦 0），尽管 GB_dir '
             'PASS（dR 按自身方向推 margin，dD 严格单调 '
             '%+.4f/%+.4f/%+.4f（4b）、%+.4f/%+.4f/%+.4f（14b），dR_neg '
             '完美镜像=线性图自洽）。机制归因=传递正交：cos(dR, '
             'unit(W_isa−W_hasa))=%+.4f/%+.4f ≈ 0，α=2 的 Δm 0.71/0.65 ≪ '
             '|d0| %.2f/%.2f。解耦诊断：cos(dR,v_isa)=%+.4f/%+.4f、'
             'cos(dR,v_hasa)=%+.4f/%+.4f（|cos|<0.5）——3228 的反方向发现'
             '不能归因 ctx-对比配方的反平行；两个方向族以不同方式非杠杆'
             '（u_R 反方向；dR 同方向但读出近正交）。「读出几何≠行为控制」'
             '获第二配方独立确证。'
             % (NORM['qwen3-4b'], NORM['qwen3-14b'],
                GB_4b['dR|a0.5'], GB_4b['dR|a1.0'], GB_4b['dR|a2.0'],
                GB_14b['dR|a0.5'], GB_14b['dR|a1.0'], GB_14b['dR|a2.0'],
                DG_4b['cos_dR_wdw'], DG_14b['cos_dR_wdw'],
                abs(D0['qwen3-4b']), abs(D0['qwen3-14b']),
                DG_4b['cos_dR_visa'], DG_14b['cos_dR_visa'],
                DG_4b['cos_dR_vhasa'], DG_14b['cos_dR_vhasa'])),
    'evidence': ['sealed 3230: res %s / npz %s / freeze %s'
                 % (A4_RES, A4_NPZ, A4_FREEZE),
                 'this arm S1 integrity: d0/dD/GA_flip/cos recomputed '
                 'bit-exact from sealed npz x2 models',
                 'cos(dR,dw) independently recomputed from W_U slices in '
                 'the 3231 verify (sha-pinned weights)'],
}]
reg31['provenance']['anchor_files']['e3_relcontrast_flip'] = {
    'path': (r'tests\glm5\result\rdc_query_construction_20260913'
             r'\phase3230\g5g27_relcontrast_efficacy\e3_relcontrast_flip.npz'),
    'sha8': A4_NPZ}

SCOPE_NOTE_24 = (' | dR second-recipe FAIL (3230 G5-G27 -> 3231 G5-G28 '
                 'refinement): state-contrast direction dR = mean(h[isa]) - '
                 'mean(h[hasa]) (3157 panels, NH-1 slot, |dR| ~7-9x row RMS) '
                 'is also NOT a behavioral lever — GA_flip 0/4 folds x2 '
                 'models at all alphas (null band 0) though GB_dir PASS '
                 '(own-direction monotone dD +0.3545/+0.3263 @a1); transfer-'
                 'orthogonality mechanism cos(dR, unit(W_isa-W_hasa)) = '
                 '+0.0997/+0.0625; recipes decoupled cos(dR,v_isa) = '
                 '-0.2463/-0.0547 (|cos|<0.5). Both direction families fail '
                 'as levers for different reasons (u_R anti-directional; dR '
                 'same-direction but readout-orthogonal) — relation-'
                 'direction behavioral-carrier status REJECTED under a '
                 'second, independent recipe (F20 recipe-generalized)')
SCOPE_NOTE_2526 = (' | dR second-recipe note (3231 G5-G28): the state-'
                   'contrast recipe dR (NH-1 slot) also fails as a '
                   'behavioral lever (GA_flip 0, transfer-orthogonal '
                   'cos(dR,dw)~+0.10/+0.06) — reinforces that geometric '
                   'readout (this feature E2_held_out band) does NOT '
                   'entail behavioral control; behavioral-lever status '
                   'remains unestablished for the relation-direction/u* '
                   'family (F20 second recipe, GAP-5 second confirmation)')
for f in reg31['features']:
    if f['id'] == 'FTR-24':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_NOTE_24
    elif f['id'] == 'FTR-25':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_NOTE_2526
    elif f['id'] == 'FTR-26':
        f['scope_limits'] = (f.get('scope_limits') or '') + SCOPE_NOTE_2526
# byte-preservation asserts
for a, b in zip(reg30['features'], reg31['features']):
    assert a['id'] == b['id']
    if a['id'] not in ('FTR-24', 'FTR-25', 'FTR-26'):
        assert jd(a) == jd(b), ('feature drift', a['id'])
    else:
        assert jd(a['statement']) == jd(b['statement'])
        assert jd(a['evidence_level']) == jd(b['evidence_level'])
        assert jd(a['anchors']) == jd(b['anchors'])
        assert len(b['scope_limits']) > len(a['scope_limits'] or '')
for k in reg30:
    if k not in ('version', 'supersedes', 'upgrade_log', 'failures',
                 'provenance', 'features'):
        assert jd(reg30[k]) == jd(reg31[k]), ('top drift', k)
assert jd(reg30['upgrade_log']) == jd(reg31['upgrade_log'])
assert jd(reg30['failures'][:19]) == jd(reg31['failures'][:19])
for k in reg30['failures'][-1]:
    assert jd(reg30['failures'][-1][k]) == jd(f20[k]), ('F20 field drift', k)
log('S2 registry v1.31 assembled: F20 refinements appended (pre-existing '
    'fields byte-exact), provenance anchor e3_relcontrast_flip, scope '
    'notes FTR-24/25/26, upgrade_log untouched (11), byte-preservation OK')

# ---------- S3: gap ledger v1.9 (GAP-5 mechanism note append) ----------
assert sha8_file(GAP18) == GAP18_SHA, 'gap v1.8 file sha drift'
gap18 = json.load(open(GAP18, encoding='utf-8'))
assert gap18['version'] == '1.8' and len(gap18['gaps']) == 5
gap19 = deepcopy(gap18)
gap19['version'] = '1.9'
gap19['updated'] = '2026-10-13'
g5 = [x for x in gap19['gaps'] if x['id'] == 'GAP-5'][0]
g5['mechanism_note'] = (g5.get('mechanism_note') or '') + (
    '；第二配方确证（3230 G5-G27 → 3231 G5-G28，F20 refinement）：状态对比'
    '方向 dR（3157 面板均值差，NH−1 槽，|dR|≈7–9× 行态 RMS）在末层精确线性'
    '图上亦 flip=0——GB_dir 过（dR 按自身方向单调推 margin，dR_neg 完美镜像'
    '=图自洽）而 GA_flip 不过（传递正交 cos(dR,Δw)≈+0.10/+0.06，Δm@α2 '
    '0.71/0.65 ≪ |d0| 3.97/4.69）；解耦诊断 cos(dR,v_isa)=−0.2463/−0.0547 '
    '排除配方反平行解释。关系信号在状态空间巨大而在判断读出上近乎不可见：'
    '三段式（上游编码承载行为 → 末层精确线性投影 → post-softmax 分布级'
    '非线性）在第二独立配方下成立，行为杠杆须在上游机制中寻找。')
for a, b in zip(gap18['gaps'], gap19['gaps']):
    if a['id'] != 'GAP-5':
        assert jd(a) == jd(b), ('gap drift', a.get('id'))
    else:
        assert len(b['mechanism_note']) > len(a['mechanism_note'])
        for k in a:
            if k != 'mechanism_note':
                assert jd(a[k]) == jd(b[k]), ('gap5 field drift', k)
for k in gap18:
    if k not in ('version', 'updated', 'gaps'):
        assert jd(gap18[k]) == jd(gap19[k]), ('gap top drift', k)
log('S3 gap v1.9 assembled: GAP-5 mechanism_note appended (second-recipe '
    'confirmation), legacy x4 + GAP-5 fields byte-preserved')

# ---------- S4: ledger untouched (within-run snapshot invariant) ----------
led = json.load(open(LEDGER, encoding='utf-8'))
LED_HEAD = [len(led['measurements']), led['ledger_sha256_8']]
assert LED_HEAD == [LED_N, LED_POST], ('ledger head drift', LED_HEAD)
log('S4 ledger head observed: n=%d post=%s (untouched by this arm; '
    'closeout appends #383 later)' % tuple(LED_HEAD))

# ---------- S5: write sealed artifacts ----------
WROTE = {}
p = os.path.join(OUTDIR, 'atlas_registry_v1_31.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(reg31, f, ensure_ascii=False, indent=1)
v31 = dict(reg31)
v31.pop('sha8', None)
WROTE['registry_v131'] = {'file': os.path.basename(p),
                          'file_sha8': sha8_file(p),
                          'content_sha8': csha8(v31)}
log('S5 registry v1.31 sealed: file %s content %s' % (
    WROTE['registry_v131']['file_sha8'],
    WROTE['registry_v131']['content_sha8']))
p = os.path.join(OUTDIR, 'gap_ledger_v1_9.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(gap19, f, ensure_ascii=False, indent=1)
WROTE['gap_v19'] = {'file': os.path.basename(p), 'file_sha8': sha8_file(p),
                    'content_sha8': csha8(gap19)}
log('S5 gap v1.9 sealed: file %s content %s' % (
    WROTE['gap_v19']['file_sha8'], WROTE['gap_v19']['content_sha8']))

# ---------- S6: verdict ----------
led2 = json.load(open(LEDGER, encoding='utf-8'))
assert [len(led2['measurements']), led2['ledger_sha256_8']] == LED_HEAD, \
    'ledger modified within run'
clean = not bad
verdict = ('f20_refinement_registered' if clean
           else 'registry_decision_dirty (%d)' % len(bad))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3231, 'name': 'g5g28_registry_decision',
    'arm': 'F20 refinement + GAP-5 note + scope notes (default (a))',
    'freeze_sha8': fz['sha8'],
    'inputs': fz['inputs'],
    'clarifications_frozen': fz['clarifications_frozen_before_assembly'],
    'integrity': INTEG,
    'source_verdict': 'dR_lever_fail (3230, sealed %s)' % A4_RES,
    'registry': {'version': '1.31',
                 'change': 'F20 refinements appended (recipe-generalized); '
                           'upgrade_log untouched (11); E-levels unchanged',
                 'scope_appends': ['FTR-24', 'FTR-25', 'FTR-26'],
                 'provenance_anchor': 'e3_relcontrast_flip %s' % A4_NPZ},
    'gap': {'version': '1.9', 'gap5_mechanism_note': 'appended '
            '(second-recipe confirmation)'},
    'wrote': WROTE,
    'recheck': {'ledger_head_observed': LED_HEAD,
                'ledger_untouched_within_run': True,
                'byte_preservation': ('26 features (23 untouched byte-exact '
                                      '+ 3 scope appends) + failures[:19] '
                                      '+ F20 pre-existing fields + '
                                      'upgrade_log x11 + legacy top-level '
                                      'OK; gap legacy x4 + GAP-5 fields OK'),
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
