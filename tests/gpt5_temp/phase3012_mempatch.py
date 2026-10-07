# -*- coding: utf-8 -*-
"""Phase 3012 MEMORY patch (append + equal-length
compression)."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
pairs = [
    # trim 3011 tail (duplicate info)
    ('K/V 双臂承载；剂量平坦 s≤0.5）；白盒手术靶点'
     '确立；早层写+中带调',
     'K/V 双臂；早层写+中带调'),
    # compress 3007
    (' 3007 Ω-P2a 生成记录器：锁定比 0.457',
     '3007 Ω-P2a：锁定比 0.457'),
    # compress 3008
    ('KV 零化饱和（任意位 22-31/32）→分级干预',
     'KV 零化饱和→分级干预'),
    # compress 3010
    ('换读出保干预→逻辑位因果特异性确立（JS D=0.0297 '
     'p=0.0135 四尺度 p<0.05）；读出解离=c8 vs JS '
     '反序——逻辑 token 门控分布边界非流形位置',
     '换读出保干预→逻辑位因果特异性确立（D=0.0297 '
     'p=0.0135 四尺度）；读出解离=c8 vs JS 反序——'
     '逻辑 token 门控分布边界'),
    # compress core sentence
    ('核心：null 重编码全层分布式涌现，对单点操作化'
     '关闭；头级重要性=关系属性。',
     '核心：null 重编码全层分布式涌现；头级重要性='
     '关系属性。'),
    # replace next-step block
    ('- max=3011，下一个 3012（A 主选 L3 门控手术'
     '可操作性检验：L3 KV 擦除后内容均值回填 vs 零 '
     'vs 随机——信息性 vs 容量性；B L31 次峰定位；'
     'C steering-vector 响应读出；D GLM4-9B-Base '
     '家族对照）。方案 v5。',
     '- max=3012，下一个 3013（A 主选 logic 位 K,V '
     '内容分解——低秩结构定位"不可统计替代"；B L31 '
     '次峰；C 反向手术剂量律 s 网格；D GLM4-9B-Base '
     '对照）。方案 v5。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])
# append 3012
anchor = '早层写+中带调'
assert anchor in t, 'anchor missing'
add = ('。 3012 Ω-P2f：**L3 门控=混合**——mean 回填不'
       '恢复（recov −0.026 p=0.507）、噪声部分恢复'
       '（0.256<0.5）≈1/4 容量+内容特异主体；content '
       '位 KV 统计可互换（JS 0.00025）而 logic 位不可；'
       '手术=可破坏不可伪造。')
t = t.replace(anchor, anchor + add, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open((r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\tmp_mem12.txt'), 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3012=%s'
    % (len(t2), miss, '3012 Ω-P2f' in t2))
print('ok')
