# -*- coding: utf-8 -*-
"""Probe atlas_ledger.json v2: errata_ledger + last linkages."""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\probe_ledger_2907b.txt')

led = json.load(open(P, encoding='utf-8'))
lines = []
err = led.get('errata_ledger', [])
lines.append('errata_ledger n=%d' % len(err))
for e in err:
    lines.append('  %s | keys=%s' % (e.get('errata_id',
                                             e.get('id')),
                                     sorted(e.keys())))
lines.append('---- last errata full ----')
if err:
    lines.append(json.dumps(err[-1], indent=1,
                            ensure_ascii=False))
link = led.get('linkage', [])
lines.append('---- linkage tail 2 full ----')
for e in link[-2:]:
    lines.append(json.dumps(e, indent=1, ensure_ascii=False))
lines.append('---- measurement meas_id tail 6 ----')
for e in led.get('measurements', [])[-6:]:
    lines.append('  %s' % e.get('meas_id'))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
