# -*- coding: utf-8 -*-
"""Patch 2 for phase3120: SMOKE keeps FULL material
(alignment-exact), only GPU loops are limited."""
import io

F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3120_omega_p118_content_attr_'
     r'amplifier_behavior_opshape.py')
RPT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3120_patch2_report.txt')
src = io.open(F, encoding='utf-8').read()
o = []

PA_OLD = """if SMOKE:
    D13 = os.path.join(D13, 'smoke')
    D05 = os.path.join(D05, 'smoke')
    OUT = os.path.join(OUT, 'smoke')
"""
PA_NEW = """if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
"""

PB_OLD = """NP_ = len(pks)
assert NP_ == 672 or SMOKE
log('records rebuilt: %d (%d pairs)' % (NB, NP_))
"""
PB_NEW = """NP_ = len(pks)
assert NP_ == 672
NP_B = min(NP_B, NP_)
log('records rebuilt: %d (%d pairs)' % (NB, NP_))
"""

for (nm, old, new) in (('PA_smoke_dirs',
                        PA_OLD, PA_NEW),
                       ('PB_np_assert',
                        PB_OLD, PB_NEW)):
    cnt = src.count(old)
    if cnt == 1:
        src = src.replace(old, new)
        o.append('%s: applied' % nm)
    elif cnt == 0 and new in src:
        o.append('%s: already applied' % nm)
    else:
        o.append('%s: ERROR count=%d' % (nm, cnt))

chk = [('full material in smoke',
        "assert NP_ == 672" in src),
       ('no smoke D13 switch',
        "D13 = os.path.join(D13, 'smoke')"
        not in src),
       ('np_b min guard', 'NP_B = min(NP_B, NP_)'
        in src)]
for (nm, ok) in chk:
    o.append('CHK %s: %s' % (nm, 'OK' if ok
                             else 'FAIL'))
ok_all = all(ok for (_, ok) in chk) and \
    all('ERROR' not in line for line in o)
if ok_all:
    with io.open(F, 'w', encoding='utf-8') as f:
        f.write(src)
    o.append('FILE WRITTEN')
else:
    o.append('FILE NOT WRITTEN')
io.open(RPT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('patch2 done ok_all=%s' % ok_all)
