# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('2990 L12#2 消融 p=0.42=快照；2991 头谱反相关——'
     '头=路由=关系属性。',
     '2990 L12#2 消融=快照；2991 头谱反相关=路由'
     '关系属性。'),
    ('2993 Ω-D 逻辑签名在场且长度稳健；2994 Ω-E：',
     '2993 Ω-D 逻辑签名长度稳健；2994 Ω-E '),
    ('2997 重分级：T3_M2989→replicated，cards_v21 立 '
     'basis-hash 溯源纪律。',
     '2997 重分级 T3_M2989→replicated；cards_v21 立 '
     'basis-hash 溯源。'),
]
for old, new in pairs:
    assert old in t, 'MISS: %s' % old[:30]
    t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_len4.txt', 'w',
        encoding='utf-8').write('len=%d' % len(t2))
print('compressed')
