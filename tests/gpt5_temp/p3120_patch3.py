# -*- coding: utf-8 -*-
"""Patch 3 for phase3120: class_table/subtag_table
dm must be sliced to content steps (11 cols)."""
import io

F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3120_omega_p118_content_attr_'
     r'amplifier_behavior_opshape.py')
RPT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3120_patch3_report.txt')
src = io.open(F, encoding='utf-8').read()
o = []

P7_OLD = """def class_table(mD, ann_c):
    dm = (mD[:, 1:] - mD[:, :-1]) \\
        .astype(np.float64)
    out = {}"""
P7_NEW = """def class_table(mD, ann_c):
    dm = (mD[:, 1:] - mD[:, :-1]) \\
        .astype(np.float64)[:, 1:]
    out = {}"""

P8_OLD = """def subtag_table(mD, ann_c, ann_s):
    dm = (mD[:, 1:] - mD[:, :-1]) \\
        .astype(np.float64)
    out = {}"""
P8_NEW = """def subtag_table(mD, ann_c, ann_s):
    dm = (mD[:, 1:] - mD[:, :-1]) \\
        .astype(np.float64)[:, 1:]
    out = {}"""

for (nm, old, new) in (('P7_class_table',
                        P7_OLD, P7_NEW),
                       ('P8_subtag_table',
                        P8_OLD, P8_NEW)):
    cnt = src.count(old)
    if cnt == 1:
        src = src.replace(old, new)
        o.append('%s: applied' % nm)
    elif cnt == 0 and new in src:
        o.append('%s: already applied' % nm)
    else:
        o.append('%s: ERROR count=%d' % (nm, cnt))

bad = src.count(
    "mD[:, 1:] - mD[:, :-1]) \\\n"
    "        .astype(np.float64)\n")
o.append('CHK remaining unsliced dm: %d' % bad)
ok_all = (bad == 0) and \
    all('ERROR' not in line for line in o)
if ok_all:
    with io.open(F, 'w', encoding='utf-8') as f:
        f.write(src)
    o.append('FILE WRITTEN')
else:
    o.append('FILE NOT WRITTEN')
io.open(RPT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('patch3 done ok_all=%s' % ok_all)
