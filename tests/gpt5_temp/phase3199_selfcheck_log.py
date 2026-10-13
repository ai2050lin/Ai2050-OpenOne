# -*- coding: utf-8 -*-
import io, hashlib
p = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-10.md'
t = io.open(p, encoding='utf-8').read()
ADD = ('- 3199 self-check（磁盘回读）：registry v1.22 9fd87433 / atlas_v1_24 '
       '74a11492 / ledger post cffec403 / MEMO 4b9359ab 全部实测吻合；独立复核 '
       '18/18 ALL PASS（#8 首跑 FAIL 为 verify 判据误用 4dp 舍入值要求 1e-9，'
       '改为 SMOKE/FULL 载荷逐位比较后通过，只修 verify 未动封存产物）。\n')
t = t + ADD
io.open(p, 'w', encoding='utf-8', newline='').write(t)
print('self-check ->', hashlib.sha256(io.open(p, encoding='utf-8').read().encode()).hexdigest()[:8])
