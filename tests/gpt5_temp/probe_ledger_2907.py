# -*- coding: utf-8 -*-
"""Probe atlas_ledger.json structure for M2907 registration."""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\probe_ledger_2907.txt')

led = json.load(open(P, encoding='utf-8'))
lines = []
lines.append('top keys: %s' % sorted(led.keys()))
for k in ('ledger_sha', 'measurements', 'errata', 'negatives',
          'growth', 'linkage'):
    v = led.get(k)
    if isinstance(v, list):
        lines.append('%s: list n=%d' % (k, len(v)))
    else:
        lines.append('%s: %s' % (k, v))

meas = led.get('measurements', [])
lines.append('---- last measurement full ----')
lines.append(json.dumps(meas[-1], indent=1, ensure_ascii=False))
lines.append('---- measurement ids tail 6 ----')
for e in meas[-6:]:
    lines.append('  %s | %s' % (e.get('id'), e.get('title', '')))

err = led.get('errata', [])
lines.append('---- errata all ids/titles ----')
for e in err:
    lines.append('  %s | %s' % (e.get('id'), e.get('title', '')))
lines.append('---- last errata full ----')
if err:
    lines.append(json.dumps(err[-1], indent=1, ensure_ascii=False))

link = led.get('linkage', [])
lines.append('---- linkage ids ----')
for e in link:
    lines.append('  %s | keys=%s' % (e.get('id'),
                                     sorted(e.keys())))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
