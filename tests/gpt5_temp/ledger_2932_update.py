# -*- coding: utf-8 -*-
"""Phase 2932 ledger entry: append M2932, L14.connects, sha."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2932_ledger_update_report.txt')

d = json.load(open(P, encoding='utf-8'))
meas = d['measurements']
link = d['linkage']
assert len(meas) == 70, 'meas len %d' % len(meas)
assert meas[-1]['meas_id'] == 'M2931_skeleton_overlap_null'
L14 = link[-1]
assert L14['link_id'] == 'L14_readout_spectrum_cross_model'
assert L14['connects'][-1] == 'M2931_skeleton_overlap_null'

VERDICT = (
    'skeleton_functionally_load_bearing - one run (52 s), REAL '
    'causal ablation: o_proj-input head slice zeroed at pos 1 '
    'per cell, full batched forward over 57 func-condition '
    'prompts; readout final residual projected on dirs_word[35] '
    '(rebuilt this run). ANCHORS 5/5: a1 dirs_word rebuild diff '
    '2.17e-08 (9th consecutive forward anchoring); a2 batched '
    'baseline determinism rel 0.0; a3 all-heads L18 hook '
    'efficacy 1.4631 > 0.01*scale; a4 masks 469/382/295; a5 '
    'baseline separation 185.70 (scale 92.29). CELLS: shared '
    'gated skeleton 295 + 353 unique matched background '
    '(2 per skeleton cell, same-layer, rng 2907) = 648 '
    'ablated cells. P1 MAIN: D_s = CI_rel(skel) - mean(2 '
    'matched bg) median 0.000568, one-sided sign-permutation '
    'p=1.0e-4 (min attainable 1/10001); skeleton CI_rel median '
    '0.005771 vs bg 0.005135 (ratio 1.124) - the convention-'
    'invariant skeleton is FUNCTIONALLY LOAD-BEARING but the '
    'margin is modest (~12%). P2 lin_r STRATIFICATION PREDICTS '
    'FUNCTION: QL layers (lin_r<0.9) 0.005994 (n=196) vs DEEP '
    '(lin_r>1.4) 0.004555 (n=50), diff 0.001439 p=7.0e-4 - '
    'convention-invariance of response structure tracks causal '
    'impact. P3 Spearman(CI_rel, rho29)=0.2916 (p=2.0e-4), '
    '(CI_rel, rho_mirror)=0.3543 (p=2.0e-4). SEAL STRUCTURE: '
    'per-layer top-half hits band-structured - L6-L12 strong '
    '(L9 19/22, L10 18/24, L8 15/21), L28/L29/L34 REVERSED '
    '(0/3, 0/2, 0/1) - skeleton functional load concentrates '
    'in the mid quasi-linear band; survivor 7 CI median 0.00615 '
    'vs skeleton-peer 0.00576 (2/7 above peers, not uniform); '
    'top functional cell overall (12,22) CI 0.02698 is a '
    'skeleton cell; CI_logit direction-consistent (skel 0.04996 '
    'vs bg 0.04921). CHAIN: 2929 skeleton exists -> 2930 rho '
    'convention-invariant -> 2931 overlap 1.9x independence -> '
    '2932 skeleton causally load-bearing with lin_r-predicted '
    'functional impact - the response-structure skeleton is a '
    'real functional circuit, not an epiphenomenon.')

meas.append({
    'meas_id': 'M2932_skeleton_functional_ablation',
    'type': 'skeleton_functional_ablation',
    'verdict': VERDICT,
    'source': {
        'path': ('phase2932/skeleton_functional_ablation/'
                 'skeleton_functional_ablation.npz'),
        'sha256_8': 'b8a74702',
        'phase': 2932},
})
L14['connects'].append('M2932_skeleton_functional_ablation')
assert len(meas) == 71
assert len(L14['connects']) == 39
json.dump(d, open(P, 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)

h = hashlib.sha256()
with open(P, 'rb') as f:
    for ch in iter(lambda: f.read(1 << 20), b''):
        h.update(ch)
rep = ['measurements: %d' % len(meas),
       'L14.connects: %d (tail3 %s)'
       % (len(L14['connects']), L14['connects'][-3:]),
       'ledger sha8: %s' % h.hexdigest()[:8],
       'ts: %s' % time.strftime('%Y-%m-%dT%H:%M:%S')]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('ledger ok')
