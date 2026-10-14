# -*- coding: utf-8 -*-
"""Phase 3233 (G5-G30): GAP-5 upstream-lever scan DESIGN arm.

Per prereg in MEMO Phase 3232 tail (frozen before any observation;
default arm (a)): DESIGN-ONLY, zero GPU, no observation. GAP-5 three
stage picture (3229) says behavior is carried UPSTREAM while the final
layer readout is an exact linear projection (3228 C6v3 cos 1.0000) and
F20 (3229/3230) proved the relation direction is NOT a lever AT the
final layer (GA_flip 0 x2 models x all alpha; transfer-orthogonal
cos(dR,dW)=+0.0997/+0.0625). This arm freezes the UPSTREAM lever scan
protocol: patch the residual stream at layer L in {0..NL-1}, FULLY
recompute downstream layers (stronger in-protocol causality than the
3227 last-layer injection which bypassed upstream computation), and
measure isa/hasa judgment flips. Layer-position prediction UL_loc
(report-only support): the lever band should overlap the K1 mechanism
layer k*=3 (depth 0.075-0.083, three models, phase 3152), NOT the
readout layer.

Chain: 3234 = compute+seal arm (GPU layer scan), 3235 = gate
evaluation + registry/gap decision arm.
"""

import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
OUTDIR = os.path.join(SRC_DIR, 'phase3233', 'g5g30_gap5_upstream_lever_design')
SMOKE = os.environ.get('P3233_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''

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


os.makedirs(OUTDIR, exist_ok=True)
bad = []

# ---------- freeze BEFORE any observation ----------
fz = {
    'phase': '3233', 'name': 'g5g30_gap5_upstream_lever_design',
    'prereg': 'AGI_GPT5_MEMO Phase 3232 tail (frozen before any '
              'observation; default arm (a) GAP-5 upstream-lever scan '
              'design, design-only zero GPU)',
    'data_basis_pinned': {
        'caches': {'p3154_4b': 'bb99438f', 'p3154_14b': '6d142b9d',
                   'p3154_glm4': 'cf2659f1',
                   'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
                   'p3169_glm4': '9cc3f8b4'},
        'u_star_npz': 'ac5d4d74',
        'a1_result': 'ea1a0585',
        'direction_pipeline': '3194 module import (verbatim Qn pipeline)',
        'k1_anchor': 'phase3152 summary result_summary.json '
                     '(k*=3 all three models; NL=36/40/40; depth '
                     '0.0833/0.075/0.075)',
    },
    'zero_gpu': 'design-only; no observation, no model forward, no GPU',
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3233 g5g30_gap5_upstream_lever_design %s =='
    % ('SMOKE' if SMOKE else 'FULL'))
log('freeze sealed: %s (before any observation)' % fz['sha8'])

# ---------- D1: data basis anchors re-verified on disk ----------
D1 = {}
for tag, rel in [
    ('p3154_4b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz'),
    ('p3154_14b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz'),
    ('p3154_glm4', r'phase3154\g1p4_mfd_multifactor_disentangle\glm4\collect.npz'),
    ('p3169_4b', r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    ('p3169_14b', r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    ('p3169_glm4', r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
]:
    p = os.path.join(SRC_DIR, rel)
    got = sha8_file(p)
    assert got == fz['data_basis_pinned']['caches'][tag], (tag, got)
    D1[tag] = got
log('D1 caches x6 re-verified: %s' % D1)

# ---------- D2: u* + k* + chain-state anchors ----------
P225 = os.path.join(SRC_DIR, 'phase3225', 'g5g25_a1_behavioral_band_compute')
assert sha8_file(os.path.join(P225, 'a1_behavioral_band.npz')) == 'ac5d4d74'
a1 = json.load(open(os.path.join(P225, 'result.json'), encoding='utf-8'))
assert a1['verdict'] == 'a1_behavioral_band_sealed'
k1 = json.load(open(os.path.join(
    SRC_DIR, r'phase3152\g1p2_tri_model_k1\summary\result_summary.json'),
    encoding='utf-8'))
K1_ANCH = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    d = k1['k1_3model'][m]
    K1_ANCH[m] = {'NL': d['NL'], 'kstar': d['kstar'],
                  'kstar_depth_frac': round(
                      k1['k1_depth_note'][m]['kstar_depth_frac'], 4)}
assert K1_ANCH['qwen3-4b']['kstar'] == 3
assert K1_ANCH['qwen3-14b']['kstar'] == 3
assert K1_ANCH['glm4-9b']['kstar'] == 3
led = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                     encoding='utf-8'))
REG31 = sha8_file(os.path.join(SRC_DIR, r'phase3231\g5g28_registry_decision'
                               r'\atlas_registry_v1_31.json'))
GAP10 = sha8_file(os.path.join(SRC_DIR, r'phase3232\g5g29_gd_amp_class_robustness'
                               r'\gap_ledger_v1_10.json'))
assert (len(led['measurements']), led['ledger_sha256_8']) == (384, '0e47bfd9')
assert REG31 == 'ef5189e7' and GAP10 == '378de1cd'
D2 = {'u_star_npz': 'ac5d4d74', 'a1_verdict': a1['verdict'],
      'k1_anchor': K1_ANCH,
      'chain_state': {'ledger_n': 384, 'ledger_post': '0e47bfd9',
                      'registry_v1_31_file': REG31,
                      'gap_v1_10_file': GAP10}}
log('D2 u* + k1 anchor + chain state verified: %s' % K1_ANCH)

# ---------- D3: candidate lever list (pre-registered) ----------
CAND = {
    'C1_u_R': 'relation-level direction u_R (R in {isa, hasa}), T2 chain '
              'recipe verbatim (3177-3187; module import of 3194 Qn '
              'pipeline); NOT a lever at final layer (3229/3230 F20) - '
              'upstream test is the GAP-5 direct prediction',
    'C2_u_star': 'spectral-out collapse shared direction u* (3189-3232 '
                 'chain; sealed 3225 npz ac5d4d74); class-level amplitude '
                 'carrier proven (3232), lever status unknown',
    'C3_dR': 'relation contrast direction dR (3230 recipe); final-layer '
             'non-lever with transfer-orthogonal readout coupling '
             '(cos(dR,dW)=+0.0997/+0.0625)',
    'C4_dv': 'unembedding token-pair difference dW = W_U[isa] - '
             'W_U[hasa] (token ids 101158/18830; local hub shard)',
    'C5_kstar_proj': 'k*-band class-subspace projection: u_R decomposed '
                     'into P_kstar(u_R) (in-plane) and orthogonal rest '
                     '(3152 rank-5 band recipe); tests whether the lever '
                     'lives in the composition subspace or elsewhere',
    'C6_head_redist': 'head-level redistribution (3160: attention '
                      'reallocation is the main executor; 3161 no single '
                      'point head) - second intervention modality, '
                      'report-only design reservation (deferred to a '
                      'later arm if C1-C5 saturate)',
}

# ---------- D4: pre-registered gates (dimensionality-checked) ----------
D4 = {
    'UL_flip': 'max over L in {0..NL-1} flip_rate(+u_R, alpha=1) - '
               'max(flip_rate over nulls) >= 0.15 on >= 3/4 folds x 2 '
               'models (per relation direction). DIMENSION CHECK: '
               'flip_rate is a ratio in [0,1], same family as GD_flip '
               '(0.15); falsifiability proven in-protocol (3229/3230 '
               'GA_flip = 0 shows the gate CAN fail)',
    'UL_dose': 'at the argmax layer: flip_rate monotone non-decreasing '
               'in alpha {0.5, 1, 2} (all pairwise increases or Spearman '
               '== 1.0) x 2 models',
    'UL_spec': 'relation specificity at the argmax layer: cross-patch '
               'flip rate (patch u_isa, measure hasa judgment, and vice '
               'versa) <= 1/3 of matched rate x 2 models',
    'UL_amp': 'u* channel (C2) at its argmax layer: OOV per-class margin '
              'shift rank-correlates with 3169 per-class collapse energy '
              '>= 0.6 across OOV classes x 2 models (n=4 OOV classes - '
              'small-n scope limit; inherits GD_amp family)',
    'UL_loc': 'REPORT-ONLY position prediction (not a gate): the UL_flip '
              'argmax layer band should overlap [kstar-2, kstar+2] = '
              '[1, 5] (3152: k*=3 three models); overlap recorded as '
              'mechanism-layer support, non-overlap recorded verbatim - '
              'neither changes the verdict',
    'verdict_taxonomy': 'UPSTREAM_LEVER_FOUND iff UL_flip and UL_dose '
                        'and UL_spec (per direction family, >= 2 models); '
                        'UPSTREAM_PARTIAL if UL_flip passes but '
                        'UL_dose/UL_spec fail (directional record + F '
                        'ledger note); UPSTREAM_ABSENT if UL_flip fails '
                        'across the full layer grid x 2 models '
                        '(=> distributed-carrier confirmation, F ledger '
                        'F20 family extension, NOT hidden)',
    'dimensionality_note': 'all gates are ratios or rank correlations '
                           'with proven in-protocol failure modes; no '
                           'raw-logit thresholds borrowed across '
                           'dimensions (lesson: Q04 E_ar gate)',
}

# ---------- D5: protocol operationalization ----------
D3 = {
    'intervention': 'residual-stream patch h_L <- h_L + alpha*u at layer '
                    'L, then FULL downstream recomputation (all layers '
                    '> L + final norm + W_U) - stronger in-protocol '
                    'causality than 3227 last-layer injection (which '
                    'bypassed upstream computation entirely)',
    'layer_grid': 'coarse scan L in {0..NL-1} step 1, alpha=1, seen '
                  'panel per fold (3154 caches); fine scan at argmax '
                  'band [L*-1, L*+1] with alpha {0.5, 1, 2} on OOV '
                  'panel seed-0 rows (3169 caches)',
    'judgment': 'J = sign of (mean-logit isa-token-set - mean-logit '
                'hasa-token-set) on exact recomputed logits; token sets '
                'pinned from sealed 3169 D2 first-token distinctness',
    'nulls': ['random norm-matched (10 pinned seeds, per layer)',
              'orthogonalized (u projected out of row state, per layer)'],
    'budget': {'coarse_scan': 'NL x 2 dirs x 208 rows x 2 models '
                              '~ 3.2e4 forwards (qwen pair only)',
               'fine_scan': '3 layers x 3 alphas x 2 dirs x 730 rows x '
                            '2 models ~ 2.6e4 forwards',
               'nulls': 'x11 multiplier on coarse argmax layer only',
               'total_estimate': '~1.5-2 h GPU serial (4b ~0.02-0.04 '
                                 's/fwd, 14b ~0.1 s/fwd)'},
    'falsifiability': 'UL_flip gate is the same family that FAILED '
                      'in-protocol at 3229/3230 (GA_flip=0); the scan '
                      'can genuinely return UPSTREAM_ABSENT',
}

# ---------- D6: seal design doc ----------
doc = {
    'schema': 'g5g30_upstream_lever_design_v1',
    'phase': '3233', 'name': 'g5g30_gap5_upstream_lever_design',
    'arm': 'design-only (default (a), zero GPU, no observation)',
    'objective': 'Freeze the upstream-lever scan protocol that tests '
                 'GAP-5 stage-1 (upstream encoding carries behavior): '
                 'layer-wise residual patch with full downstream '
                 'recomputation; locate whether ANY single direction is '
                 'a behavioral lever above the readout layer, or whether '
                 'the carrier is distributed (F20 extension).',
    'motivation': 'GAP-5 three-stage picture (3229): upstream encoding '
                  'carries behavior -> final-layer exact linear readout '
                  '(3228 C6v3 cos 1.0000) -> post-softmax distribution '
                  'nonlinearity. F20 (3229/3230) proved the readout '
                  'geometry is NOT the control geometry; both direction '
                  'recipes (u_R, dR) are non-levers at the last layer. '
                  'The upstream lever question is the last open stage of '
                  'the GAP-5 chain (gap v1.10 mechanism_note).',
    'candidate_levers_pre_registered': CAND,
    'protocol_operationalization': D3,
    'gates_pre_registered': D4,
    'chain': {
        '3234': 'G5-G30 compute+seal arm (GPU): coarse layer scan + '
                'fine argmax-band dose scan + nulls per model x fold, '
                'sealed npz; freeze-before-observation with seed '
                'pinning; qwen pair (glm4 deferred, W_U absent from '
                'local hub)',
        '3235': 'G5-G30 gate evaluation + registry/gap decision arm: '
                'UL_* gates applied; UPSTREAM_LEVER_FOUND -> candidate '
                'upgrade via upgrade_log; UPSTREAM_ABSENT -> F ledger '
                'F20-family extension + GAP-5 mechanism_note append '
                '(distributed-carrier confirmation)',
    },
    'scope_limits': [
        'qwen pair only (glm4 W_U absent from local hub); 3-model '
        'generality deferred',
        'in-protocol causality: downstream full recomputation, not '
        'full-regeneration sampling (E3 taxonomy boundary, recorded '
        'verbatim)',
        'UL_amp n=4 OOV classes: rank-correlation floor, small-n scope',
        'C6 head-redistribution modality is design-reserved only '
        '(report-only; not in the 3234 compute arm)',
        'coarse scan is alpha=1 single-dose; dose curve only at the '
        'argmax band (budget constraint, pre-registered)',
    ],
    'honesty_points': [
        'the scan can genuinely FAIL everywhere (UPSTREAM_ABSENT): the '
        '0.15 flip gate is the same family that produced GA_flip=0 at '
        '3229/3230',
        'UL_loc is a prediction, not a gate: it can be recorded against '
        'the design without changing the verdict (prevents '
        'post-hoc gate bending)',
        'null controls are per-layer norm-matched, so a positive at '
        'layer L cannot be explained by mere norm inflation',
    ],
    'status': 'DESIGN_SEALED_NOT_OBSERVED',
}
doc['sha8'] = csha8(dict(doc))
DOC_P = os.path.join(OUTDIR, PFX + 'g5g30_upstream_lever_design_v1.json')
with open(DOC_P, 'w', encoding='utf-8', newline='') as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
log('D6 design doc sealed: content %s file %s' % (
    doc['sha8'], sha8_file(DOC_P)))

# ---------- D7: ledger untouched (already asserted in D2) ----------
log('D7 ledger untouched: n=384 post=0e47bfd9 (asserted in D2)')

clean = not bad
verdict = ('upstream_lever_design_registered' if clean
           else 'upstream_lever_design_dirty (%d)' % len(bad))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3233, 'name': 'g5g30_gap5_upstream_lever_design',
    'arm': 'design-only (default (a))',
    'freeze_sha8': fz['sha8'],
    'design_sha8': doc['sha8'],
    'design_file_sha8': sha8_file(DOC_P),
    'candidates': sorted(CAND.keys()),
    'gates': sorted(D4.keys()),
    'k1_anchor': K1_ANCH,
    'chain': {'3234': 'compute+seal (GPU layer scan)',
              '3235': 'gate+registry/gap decision'},
    'recheck': {'ledger_untouched': [384, '0e47bfd9'],
                'registry_v1_31_file': REG31,
                'gap_v1_10_file': GAP10,
                'contradictions': bad},
    'verdict': verdict,
    'res_sha8': None,
}
res['res_sha8'] = csha8(dict(res))
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('result sealed: res %s' % res['res_sha8'])
