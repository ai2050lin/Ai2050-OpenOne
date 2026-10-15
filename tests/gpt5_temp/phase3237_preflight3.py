# -*- coding: utf-8 -*-
"""Phase 3237 preflight #3: measure input anchors from disk (no memory)."""
import json, os, hashlib

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]

# 1) registry v1.31
for cand in [os.path.join(BASE, 'phase3231', 'g5g28_registry_decision',
                          'atlas_registry_v1_31.json')]:
    print('REG31 exists:', os.path.exists(cand))
    if os.path.exists(cand):
        r = json.load(open(cand, encoding='utf-8'))
        print('  version:', r.get('version'), '| features:', len(r.get('features', [])),
              '| failures:', len(r.get('failures', [])),
              '| upgrade_log:', len(r.get('upgrade_log', [])))
        print('  file_sha8:', sha8(cand))
        v = dict(r); v.pop('sha8', None)
        print('  content_sha8:', csha8(v))
        print('  last failure id:', r['failures'][-1]['id'],
              '| has refinements:', 'refinements' in r['failures'][-1])

# 2) gap ledger: find latest
for root, dirs, files in os.walk(BASE):
    for fn in files:
        if fn.startswith('gap_ledger_v1_1') and fn.endswith('.json'):
            p = os.path.join(root, fn)
            g = json.load(open(p, encoding='utf-8'))
            print('GAP:', p.replace(BASE, ''), '| version:', g.get('version'),
                  '| gaps:', len(g.get('gaps', [])), '| file_sha8:', sha8(p))

# 3) 3235 result
p35 = os.path.join(BASE, 'phase3235', 'g5g31_gap5_gate_decision', 'result.json')
r35 = json.load(open(p35, encoding='utf-8'))
print('---3235---')
print('verdict:', r35.get('verdict'), '| res_sha8:', r35.get('res_sha8'),
      '| freeze:', r35.get('freeze_sha8'))
print('top keys:', sorted(r35.keys()))
for k in ('models', 'gates', 'gates_global', 'summary', 'decision'):
    if k in r35:
        s = json.dumps(r35[k], ensure_ascii=False)
        print(' ', k, '=', s[:500])
