# -*- coding: utf-8 -*-
import io
P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
old = ('2998/2999 Ω-F2 GLM4：xdir ~98% 洗消（max '
       '0.128@L4）；类分离词携带；门无量纲。')
new = '2998/2999 Ω-F2 GLM4：xdir ~98% 洗消；类分离词携带；门无量纲。'
assert old in t
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem04c.txt', 'w',
        encoding='utf-8').write(
    'len=%d ok=%s max3004=%s\n' % (
        len(t2), len(t2) <= 3000, 'max=3004' in t2))
print('ok')
