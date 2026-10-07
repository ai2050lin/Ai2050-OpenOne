# -*- coding: utf-8 -*-
"""Probe: enumerate phase3079/3080/3081/3082
result dirs and npz key inventory for the
3086 continuum test. Report -> p3086_probe.txt"""
import io
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
        r'\result'
        r'\rdc_query_construction_20260913')
o = []
for ph in ('phase3079', 'phase3080',
           'phase3081', 'phase3082'):
    d = os.path.join(BASE, ph)
    o.append('== %s ==' % ph)
    if not os.path.isdir(d):
        o.append('  MISSING DIR')
        continue
    for arm in sorted(os.listdir(d)):
        ad = os.path.join(d, arm)
        if not os.path.isdir(ad):
            continue
        o.append('  arm %s:' % arm)
        for f in sorted(os.listdir(ad)):
            o.append('    %s %d'
                     % (f, os.path.getsize(
                         os.path.join(ad, f))))
        for f in sorted(os.listdir(ad)):
            if f.endswith('.npz'):
                z = np.load(
                    os.path.join(ad, f),
                    allow_pickle=False)
                keys = sorted(z.files)
                o.append('    NPZ %s keys(%d):'
                         % (f, len(keys)))
                for i in range(0, len(keys), 6):
                    o.append('      ' + ' '.join(
                        keys[i:i + 6]))
with io.open(r'D:\AI2050\Ai2050-OpenOne'
             r'\tests\gpt5_temp'
             r'\p3086_probe.txt', 'w',
             encoding='utf-8') as fh:
    fh.write('\n'.join(o) + '\n')
print('PROBE_DONE')
