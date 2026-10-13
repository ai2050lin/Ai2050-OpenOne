# -*- coding: utf-8 -*-
import io, hashlib
p = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-10.md'
t = io.open(p, encoding='utf-8').read()
ADD = ('- 3200 self-check（磁盘回读）：registry v1.23 5f5b9e73 / atlas_v1_25 '
       '7dc08f1e / ledger post 8964e20d / MEMO 989fd907 全部实测吻合；独立复核 '
       '18/18 ALL PASS（#16 首跑 FAIL 为 verify 判据用 4dp 串匹配全精度锚值，'
       '改 5e-5 容差后通过，只修 verify 未动封存产物）。\n')
t = t + ADD
io.open(p, 'w', encoding='utf-8', newline='').write(t)
print('self-check ->', hashlib.sha256(io.open(p, encoding='utf-8').read().encode()).hexdigest()[:8])
