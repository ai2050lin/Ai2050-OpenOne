# -*- coding: utf-8 -*-
"""Ledger update: append M2936 to atlas_ledger.json."""
import hashlib
import json
import os

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913\phase2936'
        r'\anchoring_law')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2936_ledger_update_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


verdict_text = (
    'anchoring_not_established - ZERO forward, run1 (2 s) '
    'ANCHORS 5/5: a1 2935 func ci_rel vs 2933 max abs diff '
    '0.00e+00 (bit-exact); a4 2934 pos1/func raw vs 2935 func '
    'raw 0.00e+00 (bit-exact); a2 grids identical; a3 SHA '
    'chain ok; a5 masks 469/382/295. QUESTION: additive '
    'anchor (supp = a + b*ci, a>0) vs multiplicative model '
    'for semantic-context suppression. P1 ADDITIVE ANCHOR '
    'REJECTED: per null set Spearman(supp_raw, ci_raw_func) '
    'is NEGATIVE (-0.3232/-0.4168/-0.3517/-0.3552, median '
    '-0.3535), R2_linear 0.14-0.20 (battery bar 0.8), '
    'intercept a>0 significant (p_a 1e-4) but model R2 far '
    'too low; supp_raw<=0 in 750-805/1120 cells. P2 '
    'k-vs-ci_func median rho -0.2430. P3 same-condition '
    'battery also fails (rho -0.2668, R2lin 0.0477). P4 '
    'within-layer rho(supp, ci) median -0.5524. SEAL SCALE-'
    'AUDIT (the real finding): null/func scale ratio 0.41-'
    '0.54 (baseline readout magnitude COLLAPSES under null '
    'context: 92.29 -> 37-49); RAW-caliber amp null/func med '
    '0.86-0.90 < 1 (absolute ablation disturbance SHRINKS '
    'under null context, LOAD shrinks more: 0.835 vs DEEP '
    '0.919) while rel-caliber amp 1.65-2.13 > 1 - the '
    '2934/2935 null amplification is ENTIRELY denominator-'
    'driven (rel caliber), not numerator-driven; rel-'
    'caliber supp vs ci_func rho +0.30..+0.53 (sign opposite '
    'to raw caliber). Survivor core splits in RAW caliber: '
    '(5,6) 1.61 / (8,2) 1.42 / (21,6) 1.41 / (7,19) 1.29 / '
    '(1,6) 1.14 raw-increased vs (14,9) 0.50 / (20,8) 0.58 '
    'raw-halved. 2934 same/func raw amp 0.9302. REGISTERED: '
    '(1) all 2934/2935 CI-verdicts are rel-caliber '
    'statements - magnitude statements require caliber tag; '
    '(2) lin_r-CI law itself unchanged (both calibers '
    'preserve band structure direction), but amplification-'
    'gradient-inversion narrative is a rel-caliber artifact '
    'of scale collapse; (3) additive anchor and '
    'multiplicative models both rejected - supp_raw is '
    'negatively correlated with ci_func (larger-CI cells '
    'shrink LESS in absolute terms under null context). '
    'Lesson: ratio claims must register denominator caliber '
    '(extends discipline 16).')

d = json.load(open(P, encoding='utf-8'))
exec36 = json.load(open(os.path.join(OUTD, 'execution.json'),
                        encoding='utf-8'))
src_str = ' / '.join('%s %s' % (k, v)
                     for k, v in exec36['sources'].items())
d['measurements'].append({
    'meas_id': 'M2936_anchoring_law',
    'type': 'anchoring_law',
    'verdict': verdict_text,
    'source': ('tests/glm5/phase2936_anchoring_law.py '
               'sha256_8=%s; outputs execution %s result %s '
               'npz %s; sources %s'
               % (exec36['script_sha256_8'],
                  sha8(os.path.join(OUTD, 'execution.json')),
                  sha8(os.path.join(OUTD, 'result.json')),
                  sha8(os.path.join(OUTD, 'anchoring_law.npz')),
                  src_str))})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
L['connects'].append('M2936_anchoring_law')
L['phase_updated'] = 2936
json.dump(d, open(P, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)

d2 = json.load(open(P, encoding='utf-8'))
rep = ['n_meas=%d' % len(d2['measurements']),
       'last id=%s' % d2['measurements'][-1]['meas_id'],
       'connects_len=%d tail=%s'
       % (len(d2['linkage'][-1]['connects']),
          d2['linkage'][-1]['connects'][-2:]),
       'ledger sha8=%s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('OK ledger 2936', flush=True)
