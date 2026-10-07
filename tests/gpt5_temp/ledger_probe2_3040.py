# -*- coding: utf-8 -*-
import hashlib
import io
import json
import traceback

LP = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
      r'\atlas_ledger.json')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\ledger_probe2_result.txt')
L = []
try:
    with io.open(LP, encoding='utf-8') as f:
        raw = f.read()
    led = json.loads(raw)
    L.append('sha in file=%s' % led['ledger_sha256_8'])
    d2 = dict(led)
    d2.pop('ledger_sha256_8')
    for name, s in (
            ('default', json.dumps(d2,
                                   sort_keys=True)),
            ('compact', json.dumps(d2,
                                   sort_keys=True,
                                   separators=(',',
                                               ':'))),
            ('indent1', json.dumps(d2,
                                   sort_keys=True,
                                   indent=1))):
        h = hashlib.sha256(s.encode('utf-8')) \
            .hexdigest()[:8]
        L.append('%s -> %s %s'
                 % (name, h,
                    'MATCH' if h ==
                    led['ledger_sha256_8'] else ''))
    l14 = led['linkage'][-1]
    L.append('L14 link_id=%s phase=%s '
             'phase_updated=%s n_connects=%d'
             % (l14['link_id'], l14['phase'],
                l14['phase_updated'],
                len(l14['connects'])))
    L.append('last 5 connects=%s'
             % l14['connects'][-5:])
    L.append('notes (last 600)=%s'
             % str(l14['notes'])[-600:])
    L.append('relation key present=%s'
             % ('relation' in l14))
    msg = 'PROBE_OK'
except Exception:
    msg = 'PROBE_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg + '\n' + '\n'.join(L))
print(msg)
