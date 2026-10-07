# -*- coding: utf-8 -*-
"""Phase 16 设计探针 2：把 Phase 15 result 里的剖面曲线与定位量全部摊开。"""
import io
import json
import os

R = r'D:\AI2050\Ai2050-OpenOne'
OUT = R + r'\tests\deepseek_temp\Phase16\_probe_p15curves.txt'
L = []


def p(*a):
    L.append(' '.join(str(x) for x in a))


res = json.load(io.open(R + r'\tests\deepseek_temp\Phase15\result_phase15.json', encoding='utf-8'))

p('=== [A] 顶层 ===')
p('phase', res.get('phase'), '| grid keys', list(res.get('grid', {}).keys()))
p('grid:', json.dumps(res.get('grid'), ensure_ascii=False)[:600])
p('floors:', json.dumps(res.get('floors'), ensure_ascii=False))
p('arms_meta:', json.dumps(res.get('arms_meta'), ensure_ascii=False)[:600])
p('inheritance_used:', json.dumps(res.get('inheritance_used'), ensure_ascii=False)[:800])
p('extra:', json.dumps(res.get('extra'), ensure_ascii=False)[:900])


def walk(name, obj, depth=0):
    pad = '  ' * depth
    if isinstance(obj, dict):
        ks = list(obj.keys())
        p('%s%s{ %s }' % (pad, name, ', '.join(str(k) for k in ks[:14]) + (' ...' if len(ks) > 14 else '')))
        if depth < 2:
            for k in ks[:6]:
                walk(str(k), obj[k], depth + 1)
    elif isinstance(obj, list):
        p('%s%s[ n=%d ] head=%s' % (pad, name, len(obj), json.dumps(obj[:4], ensure_ascii=False)[:200]))
    else:
        p('%s%s = %s' % (pad, name, json.dumps(obj, ensure_ascii=False)[:200]))


for ac in ('E0_selfcheck', 'E1_capture', 'E2_full_swap', 'E3_localize', 'E4_summary', 'E5_concentration', 'E6_calibration',
           'predictions_check', 'verdict', 'arms'):
    p('')
    p('=== [%s] ===' % ac)
    walk(ac, res.get(ac))

p('')
p('=== [Z] 全文 E3_localize ===')
p(json.dumps(res.get('E3_localize'), ensure_ascii=False, indent=1)[:6000])

p('')
p('=== [Z] 全文 E5_concentration ===')
p(json.dumps(res.get('E5_concentration'), ensure_ascii=False, indent=1)[:9000])

p('')
p('=== [Z] verdict / joint_verdict ===')
p(json.dumps(res.get('verdict'), ensure_ascii=False, indent=1)[:6000])

p('')
p('=== [Z] E6_calibration ===')
p(json.dumps(res.get('E6_calibration'), ensure_ascii=False, indent=1)[:5000])

open(OUT, 'w', encoding='utf-8', newline='\r\n').write('\r\n'.join(L) + '\r\n')
print('WROTE', OUT, len(L), 'lines')
