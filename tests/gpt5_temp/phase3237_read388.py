# -*- coding: utf-8 -*-
"""Read ledger entry #388 as closeout template."""
import json

lp = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
d = json.load(open(lp, encoding='utf-8'))
ms = d['measurements']
print('n =', len(ms))
print(json.dumps(ms[-1], ensure_ascii=False, indent=1))
