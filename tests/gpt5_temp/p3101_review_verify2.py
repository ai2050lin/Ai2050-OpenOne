# -*- coding: utf-8 -*-
"""Verify R53-magnitude / R51 against sealed result.json + source code.

R53: review says 3100 "6% high-order residual" is actually 26.55-41.18%
     amplitude error. Check RESD / residual stats in 3100 result.json.
R51: review says 3099 Jaccard gate 1.6941 > 1 (over upper bound).
     Check 3099 result.json for JAC value and source for gate formula.
"""
import io
import json
import os
import re

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P98 = os.path.join(BASE, 'phase3100', 'omega_p98_upstream_predict')
P99 = os.path.join(BASE, 'phase3099', 'omega_p97_mlp_neuron_anatomy')
SRC99 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
         r'\phase3099_omega_p97_mlp_neuron_anatomy.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3101_review_verify2.txt')

lines = []


def dump_json_block(tag, path, needles):
    try:
        with io.open(path, encoding='utf-8') as f:
            txt = f.read()
    except Exception as e:
        lines.append('%s: LOAD ERR %r' % (tag, e))
        return
    lines.append('=== %s (%s) size=%d' % (tag, os.path.basename(path),
                                          len(txt)))
    # print any line containing a needle (case-insens)
    low = txt.lower()
    for nd in needles:
        idx = 0
        cnt = 0
        while True:
            i = low.find(nd.lower(), idx)
            if i < 0:
                break
            s = max(0, i - 80)
            e = min(len(txt), i + 160)
            lines.append('[%s] ...%s...' % (nd, txt[s:e].replace('\n', ' ')))
            idx = i + 1
            cnt += 1
            if cnt >= 6:
                lines.append('[%s] ... (more truncated)' % nd)
                break
        if cnt == 0:
            lines.append('[%s] NOT FOUND' % nd)


dump_json_block('P98 result.json', os.path.join(P98, 'result.json'),
                ['resd', 'residual', 'high_order', 'higher', 'amplitude',
                 '26.55', '41.18', 'pct', 'share'])
dump_json_block('P98 run_log tail', os.path.join(P98, 'run_log.txt'),
                ['resd', 'residual', 'high_order', 'amplitude', 'verdict',
                 'h_f1', 'h_f2', 'h_f3'])

dump_json_block('P99 result.json', os.path.join(P99, 'result.json'),
                ['jac', '1.6941', 'gate', 'verdict', 'share_top256',
                 'top256', 'k256'])
dump_json_block('P99 run_log tail', os.path.join(P99, 'run_log.txt'),
                ['jac', '1.6941', 'gate', 'verdict'])

# R51 source: find gate formula in 3099 source
try:
    with io.open(SRC99, encoding='utf-8') as f:
        src = f.read()
    lines.append('=== 3099 source size=%d' % len(src))
    for i, ln in enumerate(src.splitlines(), 1):
        low = ln.lower()
        if ('jaccard' in low or ('jac' in low and 'gate' in low)
                or 'min(' in low and 'gate' in low):
            lines.append('L%d: %s' % (i, ln.strip()))
    # also search for the literal threshold use
    for i, ln in enumerate(src.splitlines(), 1):
        if '1.0' in ln and ('gate' in ln.lower()
                            or 'jac' in ln.lower()):
            lines.append('L%d: %s' % (i, ln.strip()))
except Exception as e:
    lines.append('3099 src err %r' % e)

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
