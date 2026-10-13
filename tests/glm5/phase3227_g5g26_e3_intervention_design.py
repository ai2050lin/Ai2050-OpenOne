# -*- coding: utf-8 -*-
"""Phase 3227 (G5-G26): criterion-2 (sufficiency) E3 intervention design arm.

Per prereg in MEMO Phase 3226 tail (frozen before any observation;
default arm (a)): DESIGN-ONLY, zero GPU, no observation. Freeze the
minimal-intervention protocol that would license the E2 -> E3 upgrade
(3223 completion criterion_2): patch the relation direction at the
final layer of cached residuals, evaluate EXACT post-patch logits on
CPU (l' = W_U @ ln_f(h + a*u) — licensed by the 3225 A1 linearity
result at the gate scale), and measure judgment flips against pinned
null controls. Seals the design doc with pre-registered gates
GD_flip / GD_dose / GD_spec / GD_amp; the compute+seal arm is 3228,
the gate+registry-decision arm is 3229 (upgrade_log #11).

Key honesty points recorded in the design:
- in-protocol causality only (E3 taxonomy): residual-stream injection
  at the LAST layer bypasses upstream computation — that is exactly
  what the G-line E3 grade licenses, no stronger claim;
- exactness at any alpha: ln_f is evaluated directly, so the patched
  logits are ground truth; the A1 license only concerns the LINEAR
  interpretation, alpha=2 doubles as the nonlinearity-onset probe;
- GD_amp has n=4 OOV classes (rank correlation floor, small-n scope).
"""

import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = r'tests\glm5\result\rdc_query_construction_20260913'
OUTDIR = os.path.join(SRC_DIR, 'phase3227', 'g5g26_e3_intervention_design')
SMOKE = os.environ.get('P3227_SMOKE') == '1'
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
    'phase': '3227', 'name': 'g5g26_e3_intervention_design',
    'prereg': 'AGI_GPT5_MEMO Phase 3226 tail (frozen before any '
              'observation; default arm (a) criterion-2 sufficiency E3 '
              'intervention design, design-only zero GPU)',
    'data_basis_pinned': {
        'caches': {'p3154_4b': 'bb99438f', 'p3154_14b': '6d142b9d',
                   'p3154_glm4': 'cf2659f1',
                   'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
                   'p3169_glm4': '9cc3f8b4'},
        'w_u_shards': {'qwen3-4b': ['4c807e25', 'c7b1aa8f'],
                       'qwen3-14b': ['0d6b9229']},
        'u_star_npz': 'ac5d4d74', 'a1_result': 'ea1a0585',
        'direction_recipes': ['3194 module import (verbatim Qn pipeline, '
                              'device anchors drift==0 re-proven in 3225)',
                              '3158/3191/3191 sealed held-out protocols'],
    },
    'zero_gpu': 'design-only; no observation, no model forward, no GPU',
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3227 g5g26_e3_intervention_design %s =='
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

# ---------- D2: W_U + u* + A1 license anchors ----------
P225 = os.path.join(SRC_DIR, 'phase3225', 'g5g25_a1_behavioral_band_compute')
assert sha8_file(os.path.join(P225, 'a1_behavioral_band.npz')) == 'ac5d4d74'
a1 = json.load(open(os.path.join(P225, 'result.json'), encoding='utf-8'))
assert a1['verdict'] == 'a1_behavioral_band_sealed'
D2 = {'w_u_available_qwen_pair': True, 'glm4_w_u': 'deferred (absent hub)',
      'a1_linearity_license': 'cos_S>=0.99999 band<=0.0052 at alpha<=0.5 '
      '(exact CPU logit evaluation valid at ANY alpha; linear '
      'INTERPRETATION licensed only at gate scale)'}
log('D2 W_U (qwen pair) + u* npz + A1 license anchored')

# ---------- D3: judgment operationalization (on-disk anchors) ----------
# seen panel: 3154 columns g(group mcon/mcot) l/s(order) ti x4 ci ho;
# OOV panel: 3169 marg (3 seeds, 730 rows, 10) = relation readout margins;
# first-token distinctness (D2 gate) inherited from 3169.
D3 = {'seen_panel': '3154 H (208, NH, D) + column schema g/l/s/ti/ci/ho/d',
      'oov_panel': '3169 H (3, 730, NH, D) + marg (3, 730, 10) — '
                   'relation-readout margin cache, 3 seeds',
      'judgment': 'J(h) = sign of (mean-logit isa-token-set - mean-logit '
                  'hasa-token-set) on exact post-patch logits; token sets '
                  'pinned from sealed prompt ids (3169 D2 first-token '
                  'distinctness)',
      'directions': 'u_R (R in {isa, hasa}) = relation-level direction, '
                    'T2 chain recipe verbatim (3177-3187; module import of '
                    '3194 Qn pipeline); u* from sealed 3225 npz (ac5d4d74)',
      'intervention_site': 'final layer (NH-1) residual injection; patched '
                           'logits l\' = W_U @ ln_f(h + a*u) EXACT on CPU',
      'alphas': [0.5, 1.0, 2.0],
      'nulls': ['random norm-matched (10 pinned seeds)',
                'orthogonalized (u projected out of row)']}
log('D3 judgment/intervention operationalization pinned (3154 columns + '
    '3169 marg readout)')

# ---------- D4: pre-registered gates ----------
D4 = {
    'GD_flip': 'flip_rate(+u_R, alpha=1) - max(flip_rate over nulls) >= '
               '0.15 on >= 3/4 folds x 2 models (per relation direction)',
    'GD_dose': 'flip_rate monotone non-decreasing in alpha {0.5, 1, 2} '
               '(all pairwise increases or Spearman == 1.0) x 2 models',
    'GD_spec': 'relation specificity: cross-patch flip rate (patch u_isa, '
               'measure hasa judgment, and vice versa) <= 1/3 of matched '
               'rate x 2 models',
    'GD_amp': 'u* amplitude (row projection onto sealed u*) rank-correlates '
              'with measured per-class spectral-out collapse (3169 recipe '
              'on cached panel) >= 0.6 across OOV classes x 2 models '
              '(n=4 classes — small-n scope limit)',
    'verdict_taxonomy': 'E3_PASS iff GD_flip and GD_dose and GD_spec; '
                        'GD_amp reported as amplitude-carrier support; '
                        'partial => E3_DIRECTIONAL; else FAIL (FAIL goes '
                        'to F ledger, not hidden)',
    'upgrade_targets': 'relation-direction carriers (FTR-22 family) and '
                       'FTR-26 (u*) -> E3 via upgrade_log #11 (registry '
                       'v1.30) upon E3_PASS in 3229',
    'e3_taxonomy_note': 'in-protocol causality (G-line E3 grade): '
                        'intervention on cached residual stream at the '
                        'last layer bypasses upstream computation — no '
                        'full-regeneration claim',
}
log('D4 gates pre-registered: GD_flip/GD_dose/GD_spec/GD_amp + taxonomy')

# ---------- D5: seal design doc ----------
doc = {
    'schema': 'g5g26_e3_intervention_design_v1',
    'phase': '3227', 'name': 'g5g26_e3_intervention_design',
    'arm': 'design-only (default (a), zero GPU, no observation)',
    'objective': 'Freeze the minimal-intervention protocol licensing '
                 'E2 -> E3 (completion criterion_2 of presoftmax band '
                 'design 87e51bcd): direction patch -> in-scope judgment '
                 'flip + u* amplitude prediction of spectral-out collapse.',
    'math': {
        'patched_logits_exact': "l'(h, u, a) = W_U @ ln_f(h + a*u) — "
                                'ground truth for any a (ln_f evaluated '
                                'directly, RMSNorm nonlinearity inside)',
        'flip': "flip(h, R, a) = [J(l'(h+a*u_R)) != J(l(h)) and flips "
                'toward R]',
        'flip_rate': 'mean over panel rows (seen panel per fold; OOV '
                     'panel per seed)',
    },
    'data_basis': {'caches': D1, 'w_u_and_license': D2,
                   'protocol_operationalization': D3},
    'gates_pre_registered': D4,
    'chain': {
        '3228': 'G5-G26 compute+seal arm (zero GPU compute): exact flip '
                'matrices + null controls + amplitude correlations per '
                'model x fold, sealed npz; freeze-before-observation with '
                'seed pinning',
        '3229': 'G5-G26 gate evaluation + registry decision arm: gates '
                'GD_* applied, E2 -> E3 upgrade (upgrade_log #11) upon '
                'E3_PASS; FAIL routes to F ledger',
    },
    'scope_limits': [
        'qwen pair only (glm4 W_U absent from local hub); 3-model '
        'generality deferred',
        'in-protocol causality only — last-layer residual injection, no '
        'upstream recomputation (E3 taxonomy boundary, recorded verbatim)',
        'GD_amp n=4 OOV classes: rank-correlation floor, small-n scope',
        'alpha=2 rows double as nonlinearity-onset probe (A1 linear '
        'license only at gate scale 0.5)',
    ],
    'status': 'DESIGN_SEALED_NOT_OBSERVED',
}
doc['sha8'] = csha8(dict(doc))
DOC_P = os.path.join(OUTDIR, PFX + 'e3_intervention_design_v1.json')
with open(DOC_P, 'w', encoding='utf-8', newline='') as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
log('D5 design doc sealed: content %s file %s' % (
    doc['sha8'], sha8_file(DOC_P)))

# ---------- D6: ledger untouched ----------
led = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                     encoding='utf-8'))
assert len(led['measurements']) == 378 and led['ledger_sha256_8'] == '62b2d7d6'
log('D6 ledger untouched: n=378 post=62b2d7d6 before==after')

clean = not bad
verdict = ('e3_intervention_design_registered' if clean
           else 'e3_intervention_design_dirty (%d)' % len(bad))
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s' % verdict)

with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')

res = {
    'phase': 3227, 'name': 'g5g26_e3_intervention_design',
    'arm': 'design-only (default (a))',
    'freeze_sha8': fz['sha8'],
    'design_sha8': doc['sha8'],
    'design_file_sha8': sha8_file(DOC_P),
    'gates': sorted(D4.keys()),
    'chain': {'3228': 'compute+seal', '3229': 'gate+registry decision '
              '(upgrade_log #11)'},
    'recheck': {'ledger_untouched': [378, '62b2d7d6'],
                'contradictions': bad},
    'verdict': verdict,
    'res_sha8': None,
}
res['res_sha8'] = csha8(dict(res))
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('result sealed: res %s' % res['res_sha8'])
