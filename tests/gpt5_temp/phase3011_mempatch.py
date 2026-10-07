# -*- coding: utf-8 -*-
"""MEMORY patch: append 3011, compress 3009/3010 lines
(equal-length discipline)."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    # compress 3009
    (' 3009 Ω-P2c：**KV 饱和与强度无关**（s=0.2 sham '
     '26/32）→token 分歧因果工具退役；logic D>0 3/4 尺度'
     '（受混杂）；**坐标解离 logic<content<sham**（边界'
     '触发器）。',
     ' 3009 Ω-P2c：KV 饱和与强度无关→token 分歧因果工具'
     '退役；坐标解离 logic<content<sham。'),
    # compress 3010
    (' 3010 Ω-P2d：**换读出保干预→逻辑位因果特异性确立**'
     '（JS s=0 D=0.0297 p=0.0135，四尺度 p<0.05；sham '
     '0.0017）；**读出解离**：c8 序 logic<content<sham '
     'vs JS 序 logic≫content——逻辑 token 门控分布边界'
     '非流形位置；T3 位级×3。',
     ' 3010 Ω-P2d：换读出保干预→逻辑位因果特异性确立'
     '（JS D=0.0297 p=0.0135 四尺度 p<0.05）；读出解离='
     'c8 vs JS 反序——逻辑 token 门控分布边界非流形'
     '位置。'),
    # next-step line
    ('- max=3010，下一个 3011（A 主选 层×尺度 JS 剖面定位'
     '分布门控承载层——K/V 分臂×层子集×s 网格；B steering-'
     'vector 响应读出；C Ω-A2 GLM4 家族 Base 对照）。'
     '方案 v5。',
     '- max=3011，下一个 3012（A 主选 L3 门控手术可操作'
     '性检验：L3 KV 擦除后内容均值回填 vs 零 vs 随机——'
     '信息性 vs 容量性；B L31 次峰定位；C steering-vector '
     '响应读出；D GLM4-9B-Base 家族对照）。方案 v5。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

# append 3011 to the chain line (after the 3010 text)
anchor = ('逻辑 token 门控分布边界非流形位置。')
if anchor in t:
    t = t.replace(anchor, anchor +
                  ' 3011 Ω-P2e：**门控定位于 L3 KV**'
                  '（D_l*=0.0137 p_maxT=1e-4；2×次峰 L31；'
                  'K/V 双臂承载；剂量平坦 s≤0.5）；'
                  '白盒手术靶点确立；早层写+中带调。',
                  1)
else:
    miss.append('anchor3011')

t = t.replace('## 机制链状态（2936-3010）',
              '## 机制链状态（2936-3011）', 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open((r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\tmp_mem11.txt'), 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3011=%s'
    % (len(t2), miss, '3011 Ω-P2e' in t2))
print('ok')
