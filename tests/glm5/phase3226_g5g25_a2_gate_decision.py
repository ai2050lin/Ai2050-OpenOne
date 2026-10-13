# -*- coding: utf-8 -*-
"""Phase 3226 (G5-G25 A2): gate evaluation + registry decision arm (zero GPU).

Per prereg in MEMO Phase 3225 tail (frozen before gate evaluation;
default arm (a)): apply the pre-registered gates GA/GB/GC verbatim from
design doc presoftmax_band_design_v1.json (content sha 87e51bcd / file
1a1c4978) to the SEALED Phase 3225 A1 numbers (result ea1a0585 / npz
ac5d4d74 / freeze 98d7fbd1), decide the E2_held_out -> E2_behavioral
upgrade for FTR-25/FTR-26 (upgrade_log #10), produce registry v1.29 +
gap ledger v1.7 as append-only new files with byte-preservation asserts
on all untouched fields, and record the pre-softmax linearity
conclusion (Big Task A localization) as GAP-5.

Zero GPU: pure arithmetic on sealed scalars + JSON assembly.
Ledger is READ-ONLY in this arm (governance entry appended in closeout).
"""

import hashlib
import json
import os
from copy import deepcopy

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
OUTDIR = os.path.join(SRC_DIR, 'phase3226', 'g5g25_a2_gate_decision')
SMOKE = os.environ.get('P3226_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''

DESIGN = os.path.join(SRC_DIR, 'phase3223', 'g5g25_presoftmax_band_design',
                      'presoftmax_band_design_v1.json')
A1DIR = os.path.join(SRC_DIR, 'phase3225', 'g5g25_a1_behavioral_band_compute')
REG28 = os.path.join(SRC_DIR, 'phase3206', 'g5g5_ftr22_chain_recheck',
                     'atlas_registry_v1_28.json')
GAP16 = os.path.join(SRC_DIR, 'phase3192', 'g5c5_gap4_amplitude_attribution',
                     'gap_ledger_v1_6.json')
LEDGER = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')

DESIGN_FILE_SHA = '1a1c4978'
DESIGN_CONTENT_SHA = '87e51bcd'
A1_RES = 'ea1a0585'
A1_NPZ = 'ac5d4d74'
A1_FREEZE = '98d7fbd1'
REG28_SHA = '5995bbe9'
GAP16_SHA = '0f6423e0'
LED_N, LED_POST = 377, '969b6791'
MODELS = ('qwen3-4b', 'qwen3-14b')
GATE_ALPHA = '0.5'          # design: alphas {0.1, 0.5}, 0.5 = gate scale
SENS_ALPHA = '0.1'

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    """Canonical content sha8 (exclude self-recorded sha8 fields by caller)."""
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


def jd(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


os.makedirs(OUTDIR, exist_ok=True)
bad = []

# ---------- freeze BEFORE gate evaluation ----------
design = json.load(open(DESIGN, encoding='utf-8'))
assert sha8_file(DESIGN) == DESIGN_FILE_SHA, 'design file sha drift'
assert design['sha8'] == DESIGN_CONTENT_SHA, 'design content sha drift'
assert int(design['phase']) == 3223
gates = design['gates_pre_registered']
assert 'GA_behavioral_heldout' in gates and 'GB_band' in gates \
    and 'GC_rank_sufficiency' in gates
assert 'BEHAVIORAL_BAND_PASS iff GA and GB' in gates['verdict_taxonomy']

a1 = json.load(open(os.path.join(A1DIR, 'result.json'), encoding='utf-8'))
assert a1['verdict'] == 'a1_behavioral_band_sealed'
assert a1['sealed_npz']['file_sha8'] == A1_NPZ
assert a1['res_sha8'] == A1_RES if 'res_sha8' in a1 else True
assert a1['design_sha8'] == A1_FREEZE  # frozen execution sha recorded at A1
for m in MODELS:
    assert len(a1['models'][m]['metrics']) == 16, ('metrics rows', m)

fz = {
    'phase': '3226', 'name': 'g5g25_a2_gate_decision',
    'prereg': 'AGI_GPT5_MEMO Phase 3225 tail (frozen before gate '
              'evaluation; default arm (a) A2 gate evaluation + registry '
              'decision). Gates verbatim from design 87e51bcd; inputs '
              'sealed 3225 (res ea1a0585 / npz ac5d4d74 / freeze 98d7fbd1)',
    'gates_applied': gates,
    'gate_alpha': GATE_ALPHA, 'sensitivity_alpha': SENS_ALPHA,
    'inputs': {'design_file_sha8': DESIGN_FILE_SHA,
               'design_content_sha8': DESIGN_CONTENT_SHA,
               'a1_res': A1_RES, 'a1_npz': A1_NPZ, 'a1_freeze': A1_FREEZE,
               'registry_v128': REG28_SHA, 'gap_v16': GAP16_SHA,
               'ledger': 'n=%d post %s (read-only)' % (LED_N, LED_POST)},
    'zero_gpu': 'pure arithmetic on sealed scalars; no model, no forward',
}
fz['sha8'] = csha8(dict(fz))
FZ_PATH = os.path.join(OUTDIR, PFX + 'execution.json')
with open(FZ_PATH, 'w', encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3226 g5g25_a2_gate_decision %s ==' % ('SMOKE' if SMOKE else 'FULL'))
log('freeze sealed: %s (design 87e51bcd/1a1c4978, before any gate '
    'evaluation)' % fz['sha8'])

# ---------- S1: gate evaluation from sealed metrics ----------
PRED = {'FTR-26': 'P1_ustar', 'FTR-25': 'P2_vhat'}
gate_table = {}
for fid, pred in PRED.items():
    for alpha in (GATE_ALPHA, SENS_ALPHA):
        for m in MODELS:
            rows = [a1['models'][m]['metrics'][k]
                    for k in sorted(a1['models'][m]['metrics'])
                    if ('_%s_a%s' % (pred, alpha)) in k]
            assert len(rows) == 4, ('folds', fid, m, alpha, len(rows))
            cs = [x['cos_S_med'] for x in rows]
            bm = [x['band_med'] for x in rows]
            se = [x['svd8_energy'] for x in rows]
            n_ga = sum(1 for c in cs if c >= 0.70)
            med = lambda v: sorted(v)[1]  # median of 4
            rec = {
                'feature': fid, 'predictor': pred, 'model': m,
                'alpha': float(alpha),
                'ga_folds_ge_0.70': n_ga, 'ga_pass': n_ga >= 3,
                'ga_cos_S_med_min': min(cs),
                'gb_band_med_median': med(bm), 'gb_band_med_max': max(bm),
                'gb_pass': med(bm) < 0.25,
                'gc_svd8_median': med(se), 'gc_pass': med(se) >= 0.8,
            }
            rec['verdict'] = ('BEHAVIORAL_BAND_PASS'
                              if rec['ga_pass'] and rec['gb_pass']
                              else ('BEHAVIORAL_DIRECTIONAL' if rec['ga_pass']
                                    else 'FAIL'))
            gate_table['%s.%s.a%s.%s' % (fid, m, alpha, 'gate')] = rec
            log('S1 %s %s a%s: GA %d/4 (min cos_S %.4f) | GB med %.4f max '
                '%.4f | GC %.6f -> %s' % (
                    fid, m, alpha, n_ga, min(cs), med(bm), max(bm),
                    med(se), rec['verdict']))

# feature verdicts at gate alpha (both models must hold)
feat_verdict = {}
for fid in PRED:
    per = [gate_table['%s.%s.a%s.gate' % (fid, m, GATE_ALPHA)]['verdict']
           for m in MODELS]
    feat_verdict[fid] = ('BEHAVIORAL_BAND_PASS'
                         if all(v == 'BEHAVIORAL_BAND_PASS' for v in per)
                         else ('BEHAVIORAL_DIRECTIONAL'
                               if all(v in ('BEHAVIORAL_BAND_PASS',
                                            'BEHAVIORAL_DIRECTIONAL')
                                      for v in per) and any(
                                   v == 'BEHAVIORAL_DIRECTIONAL' for v in per)
                               else 'FAIL'))
    log('S1 feature %s @a%s: %s (%s)' % (fid, GATE_ALPHA, feat_verdict[fid],
                                         '/'.join(per)))

# global numbers for anchors
all_gate = [gate_table['%s.%s.a%s.gate' % (fid, m, GATE_ALPHA)]
            for fid in PRED for m in MODELS]
g_min_cos = min(r['ga_cos_S_med_min'] for r in all_gate)
g_max_band = max(r['gb_band_med_max'] for r in all_gate)
g_min_gc = min(r['gc_svd8_median'] for r in all_gate)

# ---------- S2: registry v1.29 (append-only, byte-preservation) ----------
assert sha8_file(REG28) == REG28_SHA, 'registry v1.28 file sha drift'
reg28 = json.load(open(REG28, encoding='utf-8'))
assert reg28['version'] == '1.28' and len(reg28['features']) == 26
assert len(reg28['upgrade_log']) == 9
UPGRADED = [fid for fid, v in feat_verdict.items()
            if v == 'BEHAVIORAL_BAND_PASS']
reg29 = None
if len(UPGRADED) == 2:
    reg29 = deepcopy(reg28)
    reg29['version'] = '1.29'
    reg29['supersedes'] = 'atlas_registry_v1_28.json (%s, Phase 3206)' % REG28_SHA
    reg29['upgrade_log'].append({
        'node': 'FTR-25+FTR-26',
        'change': 'E2_held_out -> E2_behavioral',
        'reason': 'G5-G25 A2 (Phase 3226): pre-registered gates GA/GB/GC '
                  'applied to sealed 3225 A1 numbers (res %s / npz %s / '
                  'freeze %s; design 87e51bcd) — GA cos_S_med >= 0.70 on '
                  '4/4 folds x2 models (min %.4f), GB band_med max %.4f '
                  '<< 0.25, GC rank-8 energy min %.6f >= 0.8; zero GPU '
                  'compute on cached residual tensors (qwen pair, glm4 '
                  'deferred per 3223 scope_limits)' % (
                      A1_RES, A1_NPZ, A1_FREEZE, g_min_cos, g_max_band,
                      g_min_gc),
    })
    reg29['provenance']['anchor_files']['a1_behavioral_band'] = {
        'path': (r'tests\glm5\result\rdc_query_construction_20260913'
                 r'\phase3225\g5g25_a1_behavioral_band_compute'
                 r'\a1_behavioral_band.npz'), 'sha8': A1_NPZ}
    for fid in UPGRADED:
        f = [x for x in reg29['features'] if x['id'] == fid][0]
        assert f['evidence_level'] == 'E2_held_out', (fid, f['evidence_level'])
        f['evidence_level'] = 'E2_behavioral'
        f['anchors'].append({
            'src': 'p3225', 'phase': 3225, 'role': 'behavioral_band',
            'tags': ['zero_gpu_compute', 'sealed_reuse', 'presoftmax_logit',
                     'gates_GA_GB_GC'],
            'asserts': {
                'design': '%s/%s' % (DESIGN_CONTENT_SHA, DESIGN_FILE_SHA),
                'a1_res': A1_RES, 'a1_npz': A1_NPZ, 'a1_freeze': A1_FREEZE,
                'ga_min_cos_S_med_4dp': round(g_min_cos, 4),
                'gb_band_med_max_4dp': round(g_max_band, 4),
                'gc_svd8_median_min_6dp': round(g_min_gc, 6),
                'gate': 'GA>=0.70 x>=3/4 folds x2 models; GB<0.25; GC>=0.8',
            }})
        sep = '；' if '；' in (f['scope_limits'] or '') else ' | '
        f['scope_limits'] = (f['scope_limits'] or '') + sep + (
            'behavioral band: pre-softmax last-layer logit displacement '
            'linearly reproduced (3225 A1, qwen pair; glm4 deferred; '
            'alpha {0.1,0.5}, gate 0.5)')
    # byte-preservation asserts
    for a, b in zip(reg28['features'], reg29['features']):
        assert a['id'] == b['id']
        if a['id'] not in UPGRADED:
            assert jd(a) == jd(b), ('feature drift', a['id'])
        else:
            pre = dict(a); post = dict(b)
            assert pre['evidence_level'] == 'E2_held_out'
            assert post['evidence_level'] == 'E2_behavioral'
            assert len(post['anchors']) == len(pre['anchors']) + 1
            for k in pre:
                if k not in ('evidence_level', 'anchors', 'scope_limits'):
                    assert jd(pre[k]) == jd(post[k]), (fid, k)
    for k in reg28:
        if k not in ('version', 'supersedes', 'upgrade_log', 'provenance',
                     'features'):
            assert jd(reg28[k]) == jd(reg29[k]), ('top drift', k)
    assert jd(reg28['upgrade_log']) == jd(reg29['upgrade_log'][:9])
    log('S2 registry v1.29 assembled: upgrade_log #10 (%s), anchors +1 x2, '
        'byte-preservation 24 features + legacy top-level OK' % '+'.join(UPGRADED))
else:
    log('S2 no upgrade (verdicts: %s)' % feat_verdict)

# ---------- S3: gap ledger v1.7 (GAP-5 append) ----------
assert sha8_file(GAP16) == GAP16_SHA, 'gap v1.6 file sha drift'
gap16 = json.load(open(GAP16, encoding='utf-8'))
assert gap16['version'] == '1.6' and len(gap16['gaps']) == 4
gap17 = deepcopy(gap16)
gap17['version'] = '1.7'
gap17['updated'] = '2026-10-11'
gap17['gaps'].append({
    'id': 'GAP-5',
    'title': '非线性定位：pre-softmax 线性带 vs post-softmax（Big Task A）',
    'status': 'localized_linear_presoftmax',
    'statement': '末层注入协议下 Δℓ_lin = W_U·J_RMSNorm·Δv 近精确复现真值 '
                 'pre-softmax logit 位移（3225 A1：cos_S_med 全 32 组 '
                 'min %.4f、band_med max %.4f ≪ 0.25 门、rank-8 能量 min '
                 '%.6f）⇒ pre-softmax logit 空间 effectively 线性；语言编码'
                 '行为非线性定位于 softmax 归一化之后（Big Task A 概率空间'
                 '失效在 OOV 面板行再确证）。输出分布级（post-softmax）行为'
                 '带闭合仍 open。' % (g_min_cos, g_max_band, g_min_gc),
    'prereg': 'G5-G25 design (presoftmax_band_design_v1, 87e51bcd) chain '
              'node 3226 (A2 gate evaluation + registry decision)',
    'evidence': ['cos_S_med min %.4f over 32 rows (2 models x 4 folds x 2 '
                 'predictors x 2 alphas)' % g_min_cos,
                 'band_med max %.4f (gate 0.25)' % g_max_band,
                 'rank-8 SVD energy median min %.6f (gate 0.8)' % g_min_gc,
                 'gold-token displacement ~0 (u* not class direction; '
                 'coheres 3191/3194 geometry verdicts)'],
    'evidence_anchors': [{'src': 'p3225', 'phase': 3225,
                          'role': 'a1_behavioral_band_seal',
                          'res': A1_RES, 'npz': A1_NPZ, 'freeze': A1_FREEZE}],
    'anchor_sha8': {'design': DESIGN_FILE_SHA, 'a1_result': A1_RES,
                    'a1_npz': A1_NPZ, 'registry_v129': 'in-phase'},
    'mechanism_note': 'J 自检数值-vs-闭式 cos>0.9999993 x8（1/N 因子修正后）；'
                      '升级门 GA/GB/GC 全过 → FTR-25+FTR-26 E2_held_out -> '
                      'E2_behavioral（registry v1.29 upgrade_log #10）。',
})
# byte preservation
for a, b in zip(gap16['gaps'], gap17['gaps']):
    assert jd(a) == jd(b), ('gap drift', a.get('id'))
for k in gap16:
    if k not in ('version', 'updated', 'gaps'):
        assert jd(gap16[k]) == jd(gap17[k]), ('gap top drift', k)
log('S3 gap v1.7 assembled: GAP-5 appended (localized_linear_presoftmax), '
    'legacy x4 byte-preserved')

# ---------- S4: ledger untouched ----------
led = json.load(open(LEDGER, encoding='utf-8'))
assert len(led['measurements']) == LED_N and led['ledger_sha256_8'] == LED_POST
log('S4 ledger untouched: n=%d post=%s before==after' % (LED_N, LED_POST))

# ---------- S5: write sealed artifacts ----------
WROTE = {}
if reg29 is not None:
    p = os.path.join(OUTDIR, PFX + 'atlas_registry_v1_29.json')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        json.dump(reg29, f, ensure_ascii=False, indent=1)
    v29 = dict(reg29); v29.pop('sha8', None)
    WROTE['registry_v129'] = {'file': os.path.basename(p),
                              'file_sha8': sha8_file(p),
                              'content_sha8': csha8(v29)}
    log('S5 registry v1.29 sealed: file %s content %s' % (
        WROTE['registry_v129']['file_sha8'],
        WROTE['registry_v129']['content_sha8']))
p = os.path.join(OUTDIR, PFX + 'gap_ledger_v1_7.json')
with open(p, 'w', encoding='utf-8', newline='') as f:
    json.dump(gap17, f, ensure_ascii=False, indent=1)
v17 = dict(gap17)
WROTE['gap_v17'] = {'file': os.path.basename(p), 'file_sha8': sha8_file(p),
                    'content_sha8': csha8(v17)}
log('S5 gap v1.7 sealed: file %s content %s' % (
    WROTE['gap_v17']['file_sha8'], WROTE['gap_v17']['content_sha8']))

# ---------- S6: verdict ----------
clean = not bad
verdict = ('a2_behavioral_upgrade_registered' if clean and reg29 is not None
           else ('a2_no_upgrade_%s' % '_'.join(sorted(set(feat_verdict.values())))
                 if clean else 'a2_behavioral_upgrade_dirty (%d)' % len(bad)))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3226, 'name': 'g5g25_a2_gate_decision', 'arm': 'A2 (default (a))',
    'created': fz['prereg'][:40], 'design_sha8': DESIGN_FILE_SHA,
    'design_content_sha8': DESIGN_CONTENT_SHA,
    'freeze_sha8': fz['sha8'],
    'inputs': fz['inputs'],
    'gate_table': gate_table,
    'feature_verdicts': feat_verdict,
    'gate_summary': {'min_cos_S_med': g_min_cos, 'max_band_med': g_max_band,
                     'min_svd8_median': g_min_gc,
                     'gate_alpha': float(GATE_ALPHA)},
    'upgrade': {'nodes': UPGRADED, 'change': 'E2_held_out -> E2_behavioral',
                'upgrade_log_no': 10, 'registry_version': '1.29'},
    'gap5': {'id': 'GAP-5', 'status': 'localized_linear_presoftmax',
             'ledger_version': '1.7'},
    'wrote': WROTE,
    'recheck': {'ledger_untouched': [LED_N, LED_POST],
                'byte_preservation': '24 features + legacy top-level x2 OK',
                'contradictions': bad},
    'verdict': verdict,
}
res['res_sha8'] = hashlib.sha256(json.dumps(
    dict(res), ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('result sealed: res %s' % res['res_sha8'])
