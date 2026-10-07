# -*- coding: utf-8 -*-
import json, io
led = json.load(io.open(r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
out = ['type=%s' % type(led).__name__]
if isinstance(led, list):
    out.append('len=%d' % len(led))
    out.append(json.dumps(led[-1], ensure_ascii=False, indent=1)[:1500])
else:
    out.append('keys=%s' % sorted(led.keys()))
    for k in sorted(led.keys()):
        v = led[k]
        if isinstance(v, list) and v:
            out.append('list %s len=%d' % (k, len(v)))
            out.append(json.dumps(v[-1], ensure_ascii=False, indent=1)[:1500])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3136_ledpeek.txt','w',encoding='utf-8').write('\n'.join(out))
print('OK')
