# -*- coding: utf-8 -*-
"""Phase 2905 ledger update: M2905 measurement + L14 refinement."""
import io
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\ledger_update_2905.txt'
log_lines = []

led = AtlasLedger.load(verify_sha=True)
rep = []
stale = led.verify(rep)
log_lines.append('pre-update stale: %s'
                 % (stale if stale else 'none'))

mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2905_margin_structure_skew_audited' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2905_margin_structure_skew_audited',
        'type': 'structure_decomposition',
        'verdict': 'margin_carried_by_class_mean_shift - audits '
                   'all pass (anchor 4/4; audit_1st PASS; '
                   'audit_2nd_skew 20/20 detection vs >=12/20 '
                   'gate, median margin_within +0.085; audit_'
                   'negative PASS; from_mean identity 4/4). '
                   'Positive set P={glm4 attn, qwen mlp}: both '
                   'margin_within (-0.0229 / -0.0332) <= perm '
                   'p95_within (-0.0149 / -0.0195) => 1st-order '
                   'class-mean shift carries the margin. Fisher '
                   'F / Hotelling T2 sign pattern matches margin '
                   'positivity 4/4 (qwen mlp F 5.12>p95 2.29, T2 '
                   '51.1>23.1; glm4 attn F 6.68>2.70, T2 54.9>'
                   '27.1; both negative groups below). delta '
                   'layer profile qwen mlp concentrated L27/L30/'
                   'L34 (-0.173/-0.118/-0.107); margin_from_mean '
                   '>> margin_full (1.47 vs 0.18, 0.39 vs 0.12) '
                   '- class-mean structure diluted by within-'
                   'class scatter',
        'source': {'path': 'phase2905/margin_structure_skew_'
                           'audited/result.json',
                   'sha256_8': '05a42b4d', 'phase': 2905}})

for l in led.doc['linkage']:
    if l['link_id'] == 'L14_readout_spectrum_cross_model':
        add = (' | 2905 main reading (skew-audited): margin is '
               'carried by 1st-order class-mean shift in the B '
               'row space; F/T2 significance pattern matches '
               'margin positivity 4/4; margin level = class-mean '
               'shift diluted by within-class scatter')
        if '2905 main reading' not in l.get('notes', ''):
            l['notes'] = l.get('notes', '') + add
            l['phase_updated'] = 2905
            if 'M2905_margin_structure_skew_audited' \
                    not in l.get('connects', []):
                l['connects'].append(
                    'M2905_margin_structure_skew_audited')

led.save()
log_lines.append('ledger saved')

rep2 = []
led2 = AtlasLedger.load(verify_sha=False)
stale2 = led2.verify(rep2)
e14 = [l for l in led2.doc['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model'][0]
log_lines.append('post-update stale: %s'
                 % (stale2 if stale2 else 'none'))
log_lines.append('measurements=%d negatives=%d L14_has_2905=%s '
                 'connects_n=%d'
                 % (len(led2.doc['measurements']),
                    len(led2.doc['negatives']),
                    '2905 main reading' in e14['notes'],
                    len(e14['connects'])))

import hashlib
log_lines.append('ledger_file_sha8=%s'
                 % hashlib.sha256(open(
                     r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
                     r'\atlas\atlas_ledger.json',
                     'rb').read()).hexdigest()[:8])

with io.open(OUT, 'w', encoding='utf-8') as g:
    g.write('\n'.join(log_lines) + '\n')
print('written')
