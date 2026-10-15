# -*- coding: utf-8 -*-
"""Phase 3237 preflight #5: exact UL_amp / UL_dose gate numbers from 3235."""
import json, os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
r35 = json.load(open(os.path.join(BASE, 'phase3235', 'g5g31_gap5_gate_decision',
                                   'result.json'), encoding='utf-8'))
for m, mv in r35['models'].items():
    print('===', m, '===')
    print('Lstar:', json.dumps(mv['Lstar']))
    ua = mv.get('ul_amp', {})
    print('ul_amp:', json.dumps(ua, ensure_ascii=False)[:700])
    ud = mv.get('ul_dose', {})
    print('ul_dose monotone:', {k: v.get('monotone') for k, v in ud.items()})
    print('gates:', r35['gates'][m])
    print('contradictions:', r35.get('contradictions'))
