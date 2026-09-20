# -*- coding: utf-8 -*-
"""Ledger M2935 entry + linkage update + report."""
import hashlib
import json

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2935_ledger_update_report.txt')

verdict = (
    'null_amp_mixed - run1 crashed at P2 serialization '
    '(tuple-indexed 1-D array, no computation loss prereg '
    'impact), fixed pos_of map, run2 (267 s) ANCHORS 7/7: a1 '
    'dirs_word rebuild 2.17e-08 (12th consecutive forward '
    'anchoring); a2 determinism 0.0; a4 masks 469/382/295; '
    'a5 func sep 185.70; a6 func CI_rel vs 2933 FULL 1120 '
    'cells max abs diff 0.00e+00 (bit-exact); a7 func sep > '
    'all null-set seps (185.7 vs 77.3/106.1/106.1/81.5). '
    'DESIGN: REAL ablation pos1 x 5 conditions (func + 4 '
    'independently resampled null-tid sets, seeds '
    '2896/2914/2915/2916, 2927 sampling rule verbatim), 1120 '
    'cells. P1 LAW REPLICATION: lin_r-CI law stable in ALL 4 '
    'null sets (rho -0.6574/-0.6608/-0.6619/-0.6420, all '
    'p_band<=6e-4) - the law is not a property of any '
    'particular null token draw. P2 AMPLIFICATION-RATIO '
    'STABILITY: pairwise Spearman median 0.8044 (6 pairs '
    '0.796-0.818) - stable but below the 0.9 context-'
    'generality bar. P3 RAW-CI CONSISTENCY: pairwise median '
    '0.9420 - CI profiles across null resamples are highly '
    'consistent. FROZEN VERDICT null_amp_mixed: amplification '
    'is predominantly a context-statistics effect (H2), with '
    'a minority token-identity component (~13% per-cell '
    'cross-set std). SEAL: (1) AMPLIFICATION GRADIENT '
    'REVERSED - the most amplified layers are the load band/ '
    'early layers (L6 x3.15, L3 x2.82, L1 x2.51) while the '
    'deep/mid layers amplify least (L11 x1.59): func semantic '
    'context SUPPRESSES ablation sensitivity most where CI is '
    'largest - amplification ratio and absolute CI are '
    'anti-correlated across layers; (2) survivor core splits '
    'in two: L6-type survivors highly amplified (amp 2.56-3.18) '
    'vs L8/L9-type survivors barely amplified (amp 1.20-1.26, '
    'func CI already high); (3) skeleton cells slightly LESS '
    'amplifiable (1.77 vs rest 1.91); (4) extreme token-id '
    'outlier cells exist ((14,6) amp std 1.38 around mean '
    '7.14). REGISTERED: null amplification mechanism = mostly '
    'context statistics + layer-dependent suppression by '
    'semantic context; token-identity effects confined to '
    'sparse outlier cells.')
src = {'path': 'phase2935/null_amp_anatomy/null_amp_anatomy.npz',
       'sha256_8': '3b947b5d', 'phase': 2935}

d = json.load(open(P, encoding='utf-8'))
rep = []
assert all(m['meas_id'] != 'M2935_null_amp_anatomy'
           for m in d['measurements'])
d['measurements'].append({'meas_id': 'M2935_null_amp_anatomy',
                          'type': 'null_amp_anatomy',
                          'verdict': verdict,
                          'source': src})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
if 'M2935_null_amp_anatomy' not in L['connects']:
    L['connects'].append('M2935_null_amp_anatomy')
h = hashlib.sha256()
h.update(json.dumps(d, sort_keys=True,
                    ensure_ascii=False).encode('utf-8'))
led_sha = h.hexdigest()[:8]
json.dump(d, open(P, 'w', encoding='utf-8'), indent=1,
          ensure_ascii=False)
d2 = json.load(open(P, encoding='utf-8'))
rep.append('n_meas=%d last=%s' % (len(d2['measurements']),
                                  d2['measurements'][-1]['meas_id']))
rep.append('connects_len=%d tail=%s'
           % (len(d2['linkage'][-1]['connects']),
              d2['linkage'][-1]['connects'][-2:]))
rep.append('ledger_sha256_8=%s' % led_sha)
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('ledger ok')
