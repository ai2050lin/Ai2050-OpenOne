# -*- coding: utf-8 -*-
"""MEMORY patch for Phase 3016 (compress + append)."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
pairs = [
    ('## 机制链状态（2936-3015）',
     '## 机制链状态（2936-3016）'),
    ('3007 Ω-P2a：锁定比 0.457（11.74<25.70）；扰动 3/4 零分歧。',
     '3007 Ω-P2a：锁定比 0.457；扰动 3/4 零分歧。'),
    ('3009 Ω-P2c：KV 饱和与强度无关→token 分歧读出退役；坐标解离 logic<sham。',
     '3009 Ω-P2c：KV 饱和与强度无关→token 分歧退役；坐标 logic<sham。'),
    ('（D=0.0297 p=0.0135）',
     '（D=.0297 p=.0135）'),
    ('npz8 位级。**破坏粗粒度易行，伪造须情景 K,V。**',
     'npz8 位级。**破坏粗粒度易行，伪造须情景 K,V。**'
     ' 3016 Ω-P2j：**放大=分布式+深层收敛**——单载体复原仅 8.5%'
     '（restoration 散布 -.83~+.39，delta 能量≠效应）；l* 散布 L4-24、'
     'qh* 9 头、层内 top1 仅 .128；lens 中层 29× final（L8）但 L32/35 '
     '收敛 .76/.56；sham delta .865 无效应——L3 门控=种子，命运由分布式深层读出决定。'),
    ('- max=3015，下一个 3016（A 主选 g7/query头28 下游放大定位——o_proj 输出追踪 L4+；B L31 次峰；C 情景性检验；D 重定向终点测量）。',
     '- max=3016，下一个 3017（A 主选 深层收敛机制——L28-35 谁消化 KV 扰动（晚层响应剖面+补偿方向）；B L31 次峰；C 情景性检验；D 重定向终点）。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:24])
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem16.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3016=%s max16=%s'
    % (len(t2), miss, '3016 Ω-P2j' in t2,
       'max=3016' in t2))
print('mem patch ok')
