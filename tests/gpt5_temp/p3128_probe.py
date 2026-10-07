# -*- coding: utf-8 -*-
"""Probe: dump p3128 result.json structure + npz keys to txt."""
import io
import json
import numpy as np

OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3128'
        r'\omega_p126_joint_swap_coord_'
        r'inject_interaction_s0match')

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
z = np.load(OUTD + r'\p126_readout.npz')

L = []
L.append('== result.json top keys ==')
for k in sorted(r.keys()):
    L.append('  ' + k)

L.append('')
L.append('== result.json full dump (json) ==')
L.append(json.dumps(r, ensure_ascii=False,
                    indent=1, sort_keys=True,
                    default=str)[:30000])

L.append('')
L.append('== npz keys ==')
for k in sorted(z.files):
    a = z[k]
    L.append('  %s shape=%s dtype=%s' %
             (k, a.shape, a.dtype))
    if a.size <= 12:
        L.append('    vals=%s' % (a.tolist(),))

with io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
             r'\gpt5_temp\p3128_probe.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(L))
print('PROBE_OK %d lines' % len(L))
