# -*- coding: utf-8 -*-
import io, re
t = io.open(r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3200_g5e2_ftr23_heldout_upgrade.py', encoding='utf-8').read()
# extract old arg from patch file between rep(r""" and the terminator before r""" (P10 old)
p = io.open(r'D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/phase3201_create_from_3200.py', encoding='utf-8').read()
i = p.index('rep(r"""    # preservation asserts')
j = p.index("'failures changed'\"\"\",", i)
old = p[i+len('rep(r"""'):j] + "'failures changed'"
print('old len', len(old), 'count', t.count(old))
# find first divergence
k = t.index('    # preservation asserts')
seg = t[k:k+len(old)]
for n,(a,b) in enumerate(zip(old, seg)):
    if a != b:
        print('diverge at', n)
        print('patch:', repr(old[max(0,n-80):n+80]))
        print('src  :', repr(seg[max(0,n-80):n+80]))
        break
else:
    print('prefix identical; len seg', len(seg))
