# -*- coding: utf-8 -*-
"""Phase 3237 preflight #4: 3235 evidence details + GAP-5 note tail."""
import json, os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'

r35 = json.load(open(os.path.join(BASE, 'phase3235', 'g5g31_gap5_gate_decision',
                                   'result.json'), encoding='utf-8'))
print('--- 3235 anchors ---')
print(json.dumps(r35['anchors'], ensure_ascii=False, indent=1)[:800])
print('--- 3235 design_basis ---')
print(json.dumps(r35['design_basis'], ensure_ascii=False)[:600])
print('--- 3235 models detail (trimmed) ---')
for m, mv in r35['models'].items():
    keep = {}
    for k, v in mv.items():
        s = json.dumps(v)
        if k in ('ul_flip', 'ul_amp', 'ul_dose', 'ul_spec', 'gate_eval', 'nulls'):
            keep[k] = s[:400]
        else:
            keep[k] = ('<%s>' % type(v).__name__)
    print(m, json.dumps(keep, ensure_ascii=False)[:900])

g10 = json.load(open(os.path.join(BASE, 'phase3232', 'g5g29_gd_amp_class_robustness',
                                  'gap_ledger_v1_10.json'), encoding='utf-8'))
g5 = [x for x in g10['gaps'] if x['id'] == 'GAP-5'][0]
print('--- GAP-5 fields ---')
print(sorted(g5.keys()))
note = g5.get('mechanism_note') or ''
print('note len:', len(note))
print('note tail 300:', note[-300:])
print('other keys sample:', json.dumps({k: v for k, v in g5.items()
      if k != 'mechanism_note'}, ensure_ascii=False)[:600])
print('--- gap top keys ---')
print(sorted(g10.keys()), '| version', g10['version'], '| updated', g10.get('updated'))
