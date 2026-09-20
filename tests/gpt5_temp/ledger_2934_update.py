# -*- coding: utf-8 -*-
"""Ledger M2934 entry + linkage update + report."""
import hashlib
import json

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2934_ledger_update_report.txt')

verdict = (
    'linr_ci_law_general - run1 (batch171 concat) anchor-fail '
    'a6 3.69e-03 > 1e-4 (bf16 cross-batch-composition noise; '
    'a2 same-batch determinism 0.00), corrected to per-condition '
    'batch57 matching the 2933 composition; run2 (511 s) ANCHORS '
    '7/7: a1 dirs_word rebuild 2.17e-08 (11th consecutive '
    'forward anchoring); a2 determinism 0.0; a3 pos01 hook ok; '
    'a4 masks 469/382/295; a5 func sep 185.70; a6 pos1/func '
    'CI_rel vs 2933 FULL 1120 cells max abs diff 0.00e+00 '
    '(bit-exact); a7 func sep > null sep. DESIGN: REAL ablation '
    '3x3 = configs {pos0, pos1, pos01} x readout conds '
    '{same, func, null}, all 1120 gated cells. P1 POSITION '
    'DIMENSION: lin_r-CI law stable under ALL THREE configs - '
    'pos0 LOAD 0.004594 vs DEEP 0.001459 (x3.15) rho(med_l, '
    'lin_r)=-0.8443; pos1 -0.6762; pos01 -0.6793 (all '
    'p_band<=6e-4, p_linr<=1e-4). P2 CONDITION DIMENSION: '
    'stable under ALL THREE conds at pos1 - same -0.6854, '
    'func -0.6762, null -0.6574 (p<=2e-4). The 2933 law is '
    'position- and condition-GENERAL, not a pos1/func artifact. '
    'P3 SEAL: (1) STRONG SUBADDITIVITY pos01/pos1 grand-median '
    'ratio 1.016 - ctx-slot (pos0) contribution is already '
    'covered by word-slot ablation; pos0 top layers L1-L5 (early '
    'layers dominate the ctx slot); (2) NULL-CONDITION '
    'AMPLIFICATION: null-context CI exceeds func CI in both '
    'bands (LOAD x1.83, DEEP x2.00) - random-token context '
    'makes the readout MORE ablation-sensitive everywhere; '
    'sep same 116.7 < func 185.7 < null-context CI pattern; '
    '(3) survivor 7 all shared-skeleton members, CI consistent '
    'with 2933; skeleton-vs-rest x1.183; top-5 pos1/func cells '
    'bit-identical to 2933 ((15,34) 0.040383 grid max). '
    'REGISTERED: lin_r-CI law upgraded to general; '
    'null-amplification and pos0-early-layer profile are new '
    'descriptive findings for follow-up.')
src = {'path': 'phase2934/loadband_anatomy/loadband_anatomy.npz',
       'sha256_8': '62da7e11', 'phase': 2934}

d = json.load(open(P, encoding='utf-8'))
rep = []
assert all(m['meas_id'] != 'M2934_loadband_anatomy'
           for m in d['measurements'])
d['measurements'].append({'meas_id': 'M2934_loadband_anatomy',
                          'type': 'loadband_anatomy',
                          'verdict': verdict,
                          'source': src})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
if 'M2934_loadband_anatomy' not in L['connects']:
    L['connects'].append('M2934_loadband_anatomy')
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
