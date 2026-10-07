# -*- coding: utf-8 -*-
"""Phase 3010 MEMORY micro compression round 5."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14 connects；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8 位。',
     '- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8。'),
    ('- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链或上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚优先 max|Δ| vs 上 Phase 产物。',
     '- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚 max|Δ| vs 上 Phase。'),
    ('- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；显著集重叠 null 校准；退化统计量加非退化门；maxT 选拔层与 rho 结构层分账；镜像 −dirs 对照必配；功能主张三层分账（结构/选拔/消融 CI），承重报效应量+剖面。',
     '- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；显著集重叠 null 校准；退化统计量加非退化门；maxT 选拔层与 rho 结构层分账；镜像 −dirs 对照必配；功能主张三层分账，承重报效应量+剖面。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem10e.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
