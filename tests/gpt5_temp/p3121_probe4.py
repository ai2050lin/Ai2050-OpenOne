# -*- coding: utf-8 -*-
"""Probe: exact key structure of 3121 result.json /
design_seal.json + WLOG_C existence."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3121'
        r'\omega_p119_repl_causality_erase_'
        'polarity_recon')
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy\memory')
o = []


def walk(d, prefix, depth, maxdepth):
    if depth > maxdepth:
        return
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                o.append(prefix + k + ' (dict)')
                walk(v, prefix + k + '.', depth + 1,
                     maxdepth)
            elif isinstance(v, list):
                o.append(prefix + k + ' list[%d]'
                         % len(v))
            else:
                o.append(prefix + k + ' = %r' % (v,))


res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
o.append('== result.json keys ==')
walk(res, '', 0, 2)

seal = json.load(io.open(OUTD + r'\design_seal.json',
                         encoding='utf-8'))
o.append('')
o.append('== design_seal.json keys ==')
walk(seal, '', 0, 2)

o.append('')
o.append('== WLOG_C ==')
o.append('dir exists: %r' % os.path.isdir(WLOG_C))
if os.path.isdir(WLOG_C):
    o.append('files: %r' % sorted(os.listdir(WLOG_C)))

o.append('')
o.append('== OUTD files ==')
o.append('%r' % sorted(os.listdir(OUTD)))

io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3121_probe4_out.txt', 'w',
        encoding='utf-8').write('\n'.join(o) + '\n')
print('probe4 ok')
