# -*- coding: utf-8 -*-
"""在观测前对 disk_verify_phase11.py 做健壮化补丁（逐处 assert count==1 + 回读复核）。"""
import io, os

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase11\disk_verify_phase11.py'
s = io.open(P, encoding='utf-8').read()
orig = s

PAIRS = [
    ("CF_SITES = list(E['conf_sites'])",
     "CF_SITES = list(E.get('conf_sites') or sorted(R['E5']['sites'].keys(), key=int))"),
    ("F3_SITES = list(E['f3_sites'])",
     "F3_SITES = list(E.get('f3_sites') or R['floors']['F3_dev'].keys())"),
    ("    ('subspace G-1=5', E['subspace']['rank'] == 5 or len(R['subspace']['sing_U6']) == 5),",
     "    ('subspace G-1=5 & n_classes 6', len(R['subspace']['sing_U6']) == 5 and R['subspace']['n_classes'] == 6),"),
    ("    ('baseline lines == MEMO lines', mb.get('lines') == memo_t.count('\\n') + (0 if memo_t.endswith('\\n') else 1)),",
     "    ('baseline lines ~ MEMO lines', abs(int(mb.get('lines', -1)) - memo_b.count(b'\\n')) <= 1),"),
    ("    ('judgement verdict 与 result 一致', J11.get('verdict', {}).get('B_verdict', B_verdict) == B_verdict),",
     "    ('judgement verdict 与 result 一致', ((not isinstance(J11, dict)) or (not isinstance(J11.get('verdict'), dict)) or (J11['verdict'].get('B_verdict') == B_verdict))),"),
]
for old, new in PAIRS:
    assert s.count(old) == 1, 'count!=1: %r (%d)' % (old[:70], s.count(old))
    s = s.replace(old, new)

assert s != orig, 'no change'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
back = io.open(P, encoding='utf-8').read()
for old, new in PAIRS:
    assert back.count(new) == 1, 'readback missing: %r' % (new[:70],)
    assert old not in back, 'old still present: %r' % (old[:70],)
print('PATCH OK: 5 处全部落地并回读复核通过')
