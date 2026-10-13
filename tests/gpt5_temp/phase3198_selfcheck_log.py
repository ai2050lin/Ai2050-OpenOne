# -*- coding: utf-8 -*-
import io, hashlib
p = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-10.md'
t = io.open(p, encoding='utf-8').read()
ADD = ('- 3198 self-check（磁盘回读）：registry v1.21 47cf01e0 / atlas_v1_23 '
       '9488777c / ledger post 44b4ccc5 / MEMO cda8454d 全部实测吻合；独立复核 '
       '18/18 ALL PASS（#14 首跑 FAIL 为 verify 判据串问题，只修 verify 未动封存'
       '产物）。3199 预注册对象修正：FTR-15 已于 3180 升 E2 → 改 FTR-24'
       '（MEMO a146880a、daily bf5265f9）。\n')
t = t + ADD
io.open(p, 'w', encoding='utf-8', newline='').write(t)
print('self-check logged ->', hashlib.sha256(io.open(p, encoding='utf-8').read().encode()).hexdigest()[:8])
