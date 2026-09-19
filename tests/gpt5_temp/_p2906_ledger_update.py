# -*- coding: utf-8 -*-
"""Phase 2906 ledger update: M2906 measurement + L14 refinement."""
import io
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from rdc_atlas_ledger import AtlasLedger

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\ledger_update_2906.txt'
log_lines = []

led = AtlasLedger.load(verify_sha=True)
rep = []
stale = led.verify(rep)
log_lines.append('pre-update stale: %s'
                 % (stale if stale else 'none'))

mid = [m['meas_id'] for m in led.doc['measurements']]
if 'M2906_amplitude_law_neuron_attribution' not in mid:
    led.doc['measurements'].append({
        'meas_id': 'M2906_amplitude_law_neuron_attribution',
        'type': 'amplitude_law_test',
        'verdict': 'amplitude_law_not_established (2/4) with '
                   'channel-patterned failure: anchors 4/4 '
                   '(incl. Delta_B vs 2905 round4, maxabs ~5e-5); '
                   'coverage audit 40/40; M1 isotropic summary '
                   '(class means + per-class scalar variance) '
                   'RECONSTRUCTS both mlp channels (glm4 mlp '
                   '0.0198 in [-0.0005,0.0828]; qwen mlp 0.1799 '
                   'in [0.1128,0.2984]) but BOTH attn channels '
                   'fall BELOW their M1 intervals (glm4 attn '
                   '0.1213 < lo 0.1649; qwen attn -0.0195 < lo '
                   '-0.0116) - attn within-class shape beyond '
                   'isotropy actively suppresses the margin '
                   '(N11 second-moment blind spot visible in '
                   'amplitude). Neuron readout attribution '
                   '(descriptive): top-64/9728-13696 down_proj '
                   'column energy share of Delta_r exceeds '
                   'label-permutation p95 for both mlp channels '
                   '(glm4 0.195 vs 0.170, 10/12 layers, L34 '
                   '0.310; qwen 0.199 vs 0.193, L29-31/34) while '
                   'both attn channels sit at permutation '
                   'baseline (glm4 0.149 vs 0.162; qwen 0.243 '
                   'vs 0.288) - readout concentration matches '
                   'margin positivity 4/4',
        'source': {'path': 'phase2906/amplitude_law_neuron_'
                           'attribution/result.json',
                   'sha256_8': '277bff9e', 'phase': 2906}})

for l in led.doc['linkage']:
    if l['link_id'] == 'L14_readout_spectrum_cross_model':
        add = (' | 2906: amplitude law channel-split - M1 '
               'isotropic summary suffices for mlp margins, attn '
               'margins suppressed by beyond-isotropy within-'
               'class shape; neuron-level readout of the class-'
               'mean shift is down_proj-column concentrated in '
               'mlp, permutation-baseline in attn (4/4 match '
               'with margin positivity)')
        if '2906: amplitude law channel-split' not in \
                l.get('notes', ''):
            l['notes'] = l.get('notes', '') + add
            l['phase_updated'] = 2906
            if 'M2906_amplitude_law_neuron_attribution' \
                    not in l.get('connects', []):
                l['connects'].append(
                    'M2906_amplitude_law_neuron_attribution')

led.save()
log_lines.append('ledger saved')

rep2 = []
led2 = AtlasLedger.load(verify_sha=False)
stale2 = led2.verify(rep2)
e14 = [l for l in led2.doc['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model'][0]
log_lines.append('post-update stale: %s'
                 % (stale2 if stale2 else 'none'))
log_lines.append('measurements=%d L14_has_2906=%s connects_n=%d'
                 % (len(led2.doc['measurements']),
                    '2906: amplitude law channel-split'
                    in e14['notes'],
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
