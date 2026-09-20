# -*- coding: utf-8 -*-
"""Phase 2933 ledger entry: append M2933, L14.connects, sha."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2933_ledger_update_report.txt')

d = json.load(open(P, encoding='utf-8'))
meas = d['measurements']
link = d['linkage']
assert len(meas) == 71, 'meas len %d' % len(meas)
assert meas[-1]['meas_id'] == 'M2932_skeleton_functional_ablation'
L14 = link[-1]
assert L14['link_id'] == 'L14_readout_spectrum_cross_model'
assert L14['connects'][-1] == 'M2932_skeleton_functional_ablation'

VERDICT = (
    'full_atlas_unstructured - one run (82 s), REAL causal '
    'ablation FULL GRID: all 1120 gated non-degenerate cells '
    '(L0 excluded), 2932 protocol verbatim. ANCHORS 6/6: a1 '
    'dirs_word rebuild 2.17e-08 (10th consecutive forward '
    'anchoring); a2 determinism 0.0; a3 hook efficacy ok; a4 '
    'masks 469/382/295; a5 sep 185.70; a6 CROSS-PHASE CI_rel '
    'reproduction on the 2932 648-cell subset max abs diff '
    '0.00e+00 (bit-exact). P2 BAND STRUCTURE DECISIVELY '
    'CONFIRMED: LOAD band (L6-L12) median CI 0.005605 vs DEEP '
    'band (L28-L35) 0.002955 (x1.90), exact layer permutation '
    'p2b=0.000000 (0/6435); per-layer median CI anti-correlates '
    'with lin_r at Spearman -0.6762 (p=1.0e-4) - quasi-linear '
    'layers are causally load-bearing, deep nonlinear layers '
    'functionally quiet. P3 GRID COUPLING WEAK/MARGINAL: '
    'Spearman(CI, rho86)=0.0989 p=1.4e-3 (exceeds prereg '
    'threshold 1e-3 by 4e-4 -> frozen verdict map lands on '
    'unstructured despite the band branch being met); '
    'Spearman(CI, rho_mirror)=0.2326 p=2.0e-4 - mirror-probe '
    'response structure predicts function better than the '
    'original caliber. SEAL: deep band is quiet on median but '
    'has rare strong outliers ((15,34) CI 0.0404 grid max, '
    'deep p90 0.0062 vs load p90 0.0092); within-layer CI-rho86 '
    'coupling peaks at L9 0.654 / L24 0.501 / L11 0.476; '
    'skeleton-vs-rest full coverage 0.005771 vs 0.004877 '
    '(x1.18, consistent with 2932 x1.124); survivor layer CI '
    'ranks 3..26 ((14,9) rank 3, (20,8) rank 7, (1,6) rank 26) '
    '- survivors are structure cores, not uniform function '
    'peaks. REGISTERED: verdict is frozen-map faithful; the '
    'substantive finding is the lin_r-CI law (-0.676) plus '
    'band confirmation - the grid-level rho-CI coupling is '
    'real but weak and caliber-dependent.')

meas.append({
    'meas_id': 'M2933_full_atlas_ci',
    'type': 'full_atlas_ci',
    'verdict': VERDICT,
    'source': {
        'path': 'phase2933/full_atlas_ci/full_atlas_ci.npz',
        'sha256_8': '1ff6df21',
        'phase': 2933},
})
L14['connects'].append('M2933_full_atlas_ci')
assert len(meas) == 72
assert len(L14['connects']) == 40
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
