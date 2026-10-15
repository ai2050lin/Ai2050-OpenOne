# -*- coding: utf-8 -*-
"""Phase 3237 preflight #2: locate registry/gap files + ledger tail."""
import json, os, hashlib

ATLAS = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
BASE = r'D:\AI2050\Ai2050-OpenOne\research\gpt5'

def file_sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:8]

# find registry/gap anywhere under research/gpt5
hits = []
for root, dirs, files in os.walk(BASE):
    for fn in files:
        if 'registry_v1_3' in fn or 'gap_ledger' in fn:
            p = os.path.join(root, fn)
            hits.append((p, file_sha8(p), os.path.getmtime(p)))
for p, s, m in sorted(hits):
    print(round(m), s, p)

print('---ledger tail---')
lp = os.path.join(ATLAS, 'atlas_ledger.json')
d = json.load(open(lp, encoding='utf-8'))
ms = d.get('measurements')
print('measurements type/len:', type(ms).__name__, len(ms) if hasattr(ms, '__len__') else None)
if isinstance(ms, list) and ms:
    last = ms[-1]
    print('last keys:', sorted(last.keys()))
    for k in ('phase_id', 'id', 'post_sha8', 'post', 'pre_sha8', 'pre'):
        if k in last:
            print(' ', k, '=', last[k])
bl = d.get('blocks')
print('blocks type/len:', type(bl).__name__, len(bl) if hasattr(bl, '__len__') else None)
if isinstance(bl, list) and bl:
    lb = bl[-1]
    print('last block keys:', sorted(lb.keys()) if isinstance(lb, dict) else lb)
    if isinstance(lb, dict):
        for k in ('block_id', 'phase_id', 'post_sha8', 'post', 'n_measurements'):
            if k in lb:
                print(' ', k, '=', lb[k])
print('ledger_sha256_8 =', d.get('ledger_sha256_8'))
