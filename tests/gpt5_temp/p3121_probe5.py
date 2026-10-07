# -*- coding: utf-8 -*-
"""Probe5: p119_readout.npz key shapes + span-length
recomputation feasibility + fam_r feasibility."""
import io
import json

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3121'
        r'\omega_p119_repl_causality_erase_'
        'polarity_recon')
o = []
z = np.load(OUTD + r'\p119_readout.npz',
            allow_pickle=True)
for k in sorted(z.files):
    o.append('%s : %s %s' % (k, z[k].shape, z[k].dtype))

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
o.append('')
o.append('eff P keys: %r'
         % sorted(res['part_a']['effects']['P'].keys()))
o.append('first_token keys: %r'
         % sorted(res['part_b']['first_token']
                  ['clean'].keys()))

# span recomputation feasibility: need ann from 3120
P118 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3120'
        r'\omega_p118_content_attr_amplifier_'
        'behavior_opshape')
import os
o.append('')
o.append('p118 dir: %r' % sorted(
    os.listdir(P118)))
z18 = np.load(P118 + r'\p118_readout.npz',
              allow_pickle=True)
o.append('p118 npz keys (ann*): %r'
         % [k for k in sorted(z18.files)
            if 'ann' in k])
o.append('p118 npz keys (gt*): %r'
         % [k for k in sorted(z18.files)
            if k.startswith('gt')])

io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3121_probe5_out.txt', 'w',
        encoding='utf-8').write('\n'.join(o) + '\n')
print('probe5 ok')
