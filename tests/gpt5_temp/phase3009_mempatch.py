# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('## 机制链状态（2936-3008）',
     '## 机制链状态（2936-3009）'),
    (' 3007 Ω-P2a 生成轨迹记录器：锁定比 0.457'
     '（logic 11.74<content 25.70 类序单调）；'
     '扰动 2/4 divergent 但 3/4 零 token 分歧'
     '=argmax 阈下；恢复非 xdir 特异（生成态）。 ',
     ' 3007 Ω-P2a 生成记录器：锁定比 0.457'
     '（logic 11.74<content 25.70）；扰动 2/4 '
     'divergent 但 3/4 零 token 分歧=argmax 阈下。 '),
    (' 3008 Ω-P2b：**2993 held-out 轴生成态分离 '
     'logic>content（p_fam 0.0002×2，n 114/1280）**；'
     'KV 消融饱和（任意位 22-31/32，p=0.407）→需'
     '分级干预；256tok 漂移收缩保持。',
     ' 3008 Ω-P2b：**held-out 轴生成态分离 '
     'logic>content（p 0.0002×2，n 114/1280）**；'
     'KV 消融饱和（任意位 22-31/32）→分级干预。'),
    ('- max=3008，下一个 3009（A 主选 KV 缩放因果'
     '扫描 scale 0.2-0.8 恢复 headroom+逻辑位特异性'
     '重测；B P2b 扰动-恢复全网格；C Ω-A2 GLM4 家族'
     '对照）。方案 v5。',
     '- max=3009，下一个 3010（A 主选 logit-lens '
     '分布距离因果读出——换读出保干预；B 单层×'
     '单奇异 KV 缩放定位边界触发层/子空间；C '
     'Ω-A2 GLM4 家族 Base 对照）。方案 v5。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:30])

add = (' 3009 Ω-P2c：**KV 饱和与强度无关**（s=0.2 sham '
       '仍 26/32，headroom 未达）→token 分歧因果工具'
       '退役；logic D>0 3/4 尺度（受混杂只登记）；'
       '**坐标解离 logic<content<sham**（边界触发器）；'
       '基线 T3 跨 run 位级 49.5123。\n')
anchor = '256tok 漂移收缩保持。'
if anchor in t:
    t = t.replace(anchor, anchor + add, 1)
else:
    miss.append('anchor-3008tail')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem09.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
