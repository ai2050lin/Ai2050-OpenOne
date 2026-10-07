# -*- coding: utf-8 -*-
import io
import json
import traceback

LP = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
      r'\atlas_ledger.json')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\ledger_probe_result.txt')
L = []
try:
    with io.open(LP, encoding='utf-8') as f:
        led = json.load(f)
    L.append('top keys=%s' % list(led.keys()))
    L.append('n_measurements=%d n_linkage=%d'
             % (len(led['measurements']),
                len(led['linkage'])))
    last_m = led['measurements'][-1]
    L.append('last measurement keys=%s'
             % list(last_m.keys()))
    L.append('last measurement=%s'
             % json.dumps(last_m, ensure_ascii=False))
    last_l = led['linkage'][-1]
    L.append('last linkage keys=%s'
             % list(last_l.keys()))
    s = json.dumps(last_l, ensure_ascii=False)
    L.append('last linkage (first 1500 chars)=%s'
             % s[:1500])
    prev_l = led['linkage'][-2]
    L.append('prev linkage keys=%s'
             % list(prev_l.keys()))
    msg = 'PROBE_OK'
except Exception:
    msg = 'PROBE_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg + '\n' + '\n'.join(L))
print(msg)
