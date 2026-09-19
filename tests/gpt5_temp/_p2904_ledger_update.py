# -*- coding: utf-8 -*-
"""Phase 2904 ledger update: audit-gate measurement (all-void),
N11 negative (covariance anisotropy invisible to margin family),
L14 linkage refinement."""
import io
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\ledger_update_2904.txt'
log_lines = []

led = AtlasLedger.load(verify_sha=True)
rep = []
stale = led.verify(rep)
log_lines.append('pre-update stale: %s'
                 % (stale if stale else 'none'))

mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2904_b_row_space_structure_audit_gate' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2904_b_row_space_structure_audit_gate',
        'type': 'structure_audit_gate',
        'verdict': 'detector_insensitive_all_void - anchor 4/4 '
                   '(margin/acc recomputed from float32 npz match '
                   '2902/2903 stored: glm4 mlp 0.01979/0.67949, '
                   'glm4 attn 0.12126/0.73077, qwen mlp '
                   '0.17991/0.84211, qwen attn -0.01946/0.61404); '
                   'audit_1st_order PASS (synthetic class-mean '
                   'shift: margin_full 0.176 > p95 0.031, '
                   'margin_within -0.034 <= p95 -0.031); '
                   'audit_negative PASS (iid 3% FP); '
                   'audit_2nd_order FAIL (covariance-anisotropy '
                   'construction: margin_within -0.032 not > p95 '
                   '-0.026) => main reading (P={glm4 attn, qwen '
                   'mlp} margin_within decomposition) never '
                   'executed; construction diagnosed as '
                   'first-moment-blind, see N11',
        'source': {'path': 'phase2904/b_row_space_structure/'
                           'result.json',
                   'sha256_8': '3a782d18', 'phase': 2904}})

nids = [n['neg_id'] for n in led.doc['negatives']]
if 'N11_covariance_anisotropy_invisible_to_margin' not in nids:
    led.doc['negatives'].append({
        'neg_id': 'N11_covariance_anisotropy_invisible_to_margin',
        'kind': 'detector_domain_boundary',
        'claim': 'The 2896-family margin metrics (row-space '
                 'same-label vs diff-label cosine advantage, incl. '
                 'the within-class-centered variant) are blind to '
                 'class-covariance anisotropy by first-moment '
                 'identity: for mean-zero rows with class-dependent '
                 'covariance, E[cos|same] - E[cos|diff] = 0, so no '
                 'amount of sample size moves the margin',
        'evidence': '2904b synthetic diagnostic: cov-4x-sigma '
                    'construction pairwise gap -0.001 (n=400k '
                    'pairs), n=57 pipeline detection 6/100 (= FP '
                    'baseline); skew construction (90% mass at '
                    '+2w, 10% at -18w, mean exactly 0) gap +0.092, '
                    'detection 74/100; iid null 3/100. margin '
                    'metrics detect class-mean shifts (1st order) '
                    'and within-class skew/asymmetric-subcluster '
                    'structure (3rd moment), NOT covariance '
                    'anisotropy (2nd moment)',
        'sources': [{'path': 'phase2904/b_row_space_structure/'
                             'result.json',
                     'sha256_8': '3a782d18', 'phase': 2904}],
        'status': 'confirmed',
        'reopen_condition': 'a margin variant with covariance '
                            'sensitivity (e.g. second-moment '
                            'statistics of pair similarities) '
                            'would have to be preregistered with '
                            'its own algebraic audit'})

for l in led.doc['linkage']:
    if l['link_id'] == 'L14_margin_hierarchy_spectrum':
        ev = l['evidence']
        add = (' / 2904: audit gate - scalar margin variants are '
               '1st-order (class-mean) and skewness-sensitive only; '
               'covariance anisotropy invisible by first-moment '
               'identity (N11); higher-order reading requires '
               'skew-audited detector (2905)')
        if '2904: audit gate' not in ev:
            l['evidence'] = ev + add
            l['phase_updated'] = 2904

led.save()
log_lines.append('ledger saved')

rep2 = []
led2 = AtlasLedger.load(verify_sha=False)
stale2 = led2.verify(rep2)
log_lines.append('post-update stale: %s'
                 % (stale2 if stale2 else 'none'))
log_lines.append('measurements=%d errata=%d negatives=%d '
                 'growth=%d linkage=%d'
                 % (len(led2.doc['measurements']),
                    len(led2.doc['errata_ledger']),
                    len(led2.doc['negatives']),
                    len(led2.doc['growth_curve']),
                    len(led2.doc['linkage'])))
log_lines.append('ledger_sha=%s'
                 % led2.doc.get('self_sha256_8', 'n/a'))

with io.open(OUT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(log_lines) + '\n')
print('written')
