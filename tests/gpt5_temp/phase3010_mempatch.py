# -*- coding: utf-8 -*-
"""Phase 3010 MEMORY patch: append 3010 + equal-length
compression of 3005-3009 chain + next-step update."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    # compress chain lines
    ('3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；qwen 中带 xdir 特异 ~150×。',
     '3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；qwen 中带 xdir 特异~150×。'),
    ('3004 Ω-P4：算子共享低秩（.715/.826）但共享向≠xdir=重定向+放大；**T3 轴序 bug bit 级**，T3_corrected 替代，TRK_DROP 退役。',
     '3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+放大；**T3 轴序 bug bit 级**，T3_corrected 替代。'),
    ('3006 Base：**中带与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784；尾 1.69×；s2 1.08→1.50；23.2×=家族差异）。',
     '3006 Base：**中带与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784；尾 1.69×；23.2×=家族差异）。'),
    ('3008 Ω-P2b：**held-out 轴生成态分离 logic>content（p 0.0002×2，n 114/1280）**；KV 消融饱和（任意位 22-31/32）→分级干预。',
     '3008 Ω-P2b：**held-out 轴生成态分离 logic>content（p 0.0002×2）**；KV 消融饱和（任意位 22-31/32）→分级干预。'),
    ('3009 Ω-P2c：**KV 饱和与强度无关**（s=0.2 sham 26/32，headroom 未达）→token 分歧因果工具退役；logic D>0 3/4 尺度（受混杂只登记）；**坐标解离 logic<content<sham**（边界触发器）；基线 T3 跨 run 位级一致。',
     '3009 Ω-P2c：**KV 饱和与强度无关**（s=0.2 sham 26/32）→token 分歧因果工具退役；logic D>0 3/4 尺度（受混杂）；**坐标解离 logic<content<sham**（边界触发器）。'),
    # header
    ('## 机制链状态（2936-3009）', '## 机制链状态（2936-3010）'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:24])

anchor = ('**坐标解离 logic<content<sham**（边界触发器）。')
if anchor in t:
    add = (' 3010 Ω-P2d：**换读出保干预→逻辑位因果特异性确立**'
           '（JS 分布距离 s=0 D=0.0297 p=0.0135，四尺度 D>0 p<0.05；'
           'sham 0.0017 headroom 现身）；**读出解离**：c8 序 '
           'logic<content<sham vs JS 序 logic≫content——逻辑 token '
           '门控分布边界非流形位置；T3 位级 49.5123×3。')
    t = t.replace(anchor, anchor + add, 1)
else:
    miss.append('anchor3010')

# next step
old_next = ('- max=3009，下一个 3010（A 主选 logit-lens 分布距离'
            '因果读出——换读出保干预；B 单层×单奇异 KV 缩放定位'
            '边界触发层/子空间；C Ω-A2 GLM4 家族 Base 对照）。'
            '方案 v5。')
new_next = ('- max=3010，下一个 3011（A 主选 层×尺度 JS 剖面'
            '定位分布门控承载层——K/V 分臂×层子集×s 网格；B '
            'steering-vector 响应读出；C Ω-A2 GLM4 家族 Base '
            '对照）。方案 v5。')
if old_next in t:
    t = t.replace(old_next, new_next, 1)
else:
    miss.append('next')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem10.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3010=%s' % (len(t2), miss,
                                   '3010 Ω-P2d' in t2))
print('ok')
