# -*- coding: utf-8 -*-
"""Phase 3237 pre-flight: measure current anchors (no memory, disk only)."""
import json, os, hashlib

ATLAS = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'

def file_sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:8]

# 1) ledger
lp = os.path.join(ATLAS, 'atlas_ledger.json')
d = json.load(open(lp, encoding='utf-8'))
print('ledger top keys:', sorted(d.keys()))
if isinstance(d, dict):
    for k in ('n', 'count', 'length'):
        if k in d:
            print('  ledger', k, '=', d[k])
    ents = d.get('entries')
    if ents is None and isinstance(d, list):
        ents = d
if ents is not None:
    print('  len(entries) =', len(ents))
    last = ents[-1]
    print('  last entry keys:', sorted(last.keys()))
    for k in ('phase_id', 'id', 'post', 'post_sha8', 'pre', 'pre_sha8'):
        if k in last:
            print('   ', k, '=', last[k])
print('  ledger file sha8 =', file_sha8(lp))

# 2) registry & gap
for name in sorted(os.listdir(ATLAS)):
    if name.endswith('.json'):
        p2 = os.path.join(ATLAS, name)
        try:
            dd = json.load(open(p2, encoding='utf-8'))
        except Exception as e:
            print(name, 'ERR', e)
            continue
        inner = {k: dd[k] for k in ('sha8', 'content_sha8', 'version', 'v') if k in dd}
        print(name, '| file_sha8 =', file_sha8(p2), '| inner:', inner)
