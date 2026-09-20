# -*- coding: utf-8 -*-
"""Inspect atlas ledger tail + L14 linkage structure."""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    led = json.load(f)
print('top keys:', list(led.keys()))
ms = led.get('measurements', [])
print('n_measurements:', len(ms))
print('last 2:', json.dumps(ms[-2:], ensure_ascii=False,
                            indent=1))
lks = led.get('linkages', led.get('links', []))
print('n_linkages:', len(lks))
for lk in lks:
    if str(lk.get('id', '')) == 'L14' \
            or str(lk.get('link_id', '')) == 'L14':
        print('L14:', json.dumps(lk, ensure_ascii=False,
                                 indent=1))
