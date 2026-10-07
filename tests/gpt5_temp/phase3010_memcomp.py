# -*- coding: utf-8 -*-
"""Phase 3010 MEMORY compression round 2."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('2938-2970：子空间/词盲/线性壳/阈值/头集中/组反转/重平衡/承重/路由跳变/秩1/签名/载体h15/消融/塌缩/峰锁/fr晚/延迟；',
     '2938-2970：子空间/词盲/线性壳/阈值/头集中/组反转/重平衡/承重/路由跳变/秩1/签名/载体h15/消融/塌缩/峰锁/fr晚；'),
    ('2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/剂量窗/转正分布/h12载体/交互=h12/三要素否定/正交重定向/局部化/perp=重写/坍缩≠漂移/普查/注册表/快照/头谱=路由关系属性。',
     '2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/剂量窗/转正/h12载体/交互=h12/三要素否定/正交重定向/局部化/perp=重写/坍缩≠漂移/普查/注册表/快照/头谱=路由属性。'),
    ('3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；qwen 中带 xdir 特异~150×。',
     '3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；qwen 中带 xdir 特异~150×；'),
    ('3003 Ω-G3：中带=xdir 特异剂量线性符号不对称通道（−xdir 弱 34-45%）；方案 v5 发布。',
     '3003 Ω-G3：中带=xdir 特异剂量线性符号不对称（−xdir 弱 34-45%）；方案 v5。'),
    ('3007 Ω-P2a 生成记录器：锁定比 0.457（logic 11.74<content 25.70）；扰动 2/4 divergent 但 3/4 零 token 分歧。',
     '3007 Ω-P2a 生成记录器：锁定比 0.457（logic 11.74<content 25.70）；扰动 2/4 divergent 但 3/4 零分歧。'),
    ('3010 Ω-P2d：**换读出保干预→逻辑位因果特异性确立**（JS 分布距离 s=0 D=0.0297 p=0.0135，四尺度 D>0 p<0.05；sham 0.0017 headroom 现身）；**读出解离**：c8 序 logic<content<sham vs JS 序 logic≫content——逻辑 token 门控分布边界非流形位置；T3 位级 49.5123×3。',
     '3010 Ω-P2d：**换读出保干预→逻辑位因果特异性确立**（JS s=0 D=0.0297 p=0.0135，四尺度 p<0.05；sham 0.0017）；**读出解离**：c8 序 logic<content<sham vs JS 序 logic≫content——逻辑 token 门控分布边界非流形位置；T3 位级×3。'),
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
        r'\.workbuddy\tmp_mem10b.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
