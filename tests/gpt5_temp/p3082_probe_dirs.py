# -*- coding: utf-8 -*-
"""Probe: list phase3076/3079/3080/3081 result
dirs + npz keys (Glob unreliable on this disk).
Writes report to file."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = (ROOT + r'\tests\gpt5_temp'
       r'\p3082_probe_dirs.txt')
o = []

for ph in ('phase3076', 'phase3079', 'phase3080',
           'phase3081'):
    pd = os.path.join(BASE, ph)
    if not os.path.isdir(pd):
        o.append(ph + ': MISSING DIR')
        continue
    o.append('== ' + ph + ' ==')
    for dirpath, dirnames, filenames in \
            os.walk(pd):
        rel = os.path.relpath(dirpath, pd)
        for fn in sorted(filenames):
            if fn.endswith('.npz'):
                full = os.path.join(dirpath, fn)
                sz = os.path.getsize(full)
                o.append('  %s/%s (%.1f MB)'
                         % (rel, fn, sz / 1e6))
            elif fn in ('result.json',
                        'seal.json',
                        'execution.json'):
                o.append('  %s/%s'
                         % (rel, fn))

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
