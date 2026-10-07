# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
pairs = [
    ('KV 全强度消融饱和（任意位 22-31/32 分歧，D=2.0 p=0.407）→需分级干预；',
     'KV 消融饱和（任意位 22-31/32，p=0.407）→需分级干预；'),
    ('256 tok 漂移收缩保持。', '256tok 漂移收缩保持。'),
    ('2992 字典：符号率 0.405=快照，SAE 缓。',
     '2992 字典：符号率 0.405=快照。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:16])
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem08b.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
