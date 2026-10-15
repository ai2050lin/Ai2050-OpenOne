# -*- coding: utf-8 -*-
"""Phase 3237 closeout S1: verify ledger_sha256_8 recipe, then append #389.

Recipe candidates for ledger_sha256_8 (currently 05a31fc4, n=388):
  A) csha8(whole dict minus 'ledger_sha256_8')
  B) csha8(measurements list)
Self-check first on the current head; only append if the recipe reproduces.
"""
import hashlib
import json
import os

LP = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


d = json.load(open(LP, encoding='utf-8'))
cur = d['ledger_sha256_8']
print('current head:', len(d['measurements']), cur, '| file', sha8_file(LP))

cA = csha8({k: v for k, v in d.items() if k != 'ledger_sha256_8'})
cB = csha8(d['measurements'])
cC = csha8({k: v for k, v in d.items() if k != 'ledger_sha256_8'} | {})
print('recipe A (dict minus key):', cA, 'MATCH' if cA == cur else 'no')
print('recipe B (measurements):', cB, 'MATCH' if cB == cur else 'no')

if cA == cur:
    RECIPE = 'A'
elif cB == cur:
    RECIPE = 'B'
else:
    # try compact json variants
    for sep, ind in ((None, None), (',', None)):
        pass
    h = hashlib.sha256(json.dumps(
        {k: v for k, v in d.items() if k != 'ledger_sha256_8'},
        ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:8]
    print('recipe C (compact, no indent):', h,
          'MATCH' if h == cur else 'no')
    h2 = hashlib.sha256(json.dumps(
        d['measurements'], ensure_ascii=False, sort_keys=True)
        .encode('utf-8')).hexdigest()[:8]
    print('recipe D (measurements compact):', h2,
          'MATCH' if h2 == cur else 'no')
    raise SystemExit(1)
print('RECIPE =', RECIPE)
