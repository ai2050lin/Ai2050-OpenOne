# -*- coding: utf-8 -*-
"""MEMORY patch for Phase 3015 (compress + append)."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
pairs = [
    ('## 机制链状态（2936-3014）',
     '## 机制链状态（2936-3015）'),
    ('（−0.036）、LOO 秩 k=8 仅 0.397、噪声>两均值臂；logic 更紧（0.914 vs 0.847）但均值≠内容',
     '（−0.036）、LOO k=8 仅 0.397；logic 更紧但均值≠内容'),
    ('（−0.026）、噪声 0.256≈1/4 容量+内容特异主体；content',
     '（−0.026）；content'),
    ('（D=0.0297 p=0.0135）；读出解离=分布边界',
     '（D=0.0297 p=0.0135）'),
    ('2992 字典：符号率 0.405=快照。2993 逻辑签名稳健；2994 轴防御。',
     '2992 符号率 .405=快照；2993 签名稳健；2994 轴防御。'),
    ('中带 xdir 特异150×；3003 剂量线性不对称；方案 v5。',
     '中带 xdir 特异 150×；3003 剂量不对称；v5。'),
    ('3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+放大；T3 轴序 bug 修正。',
     '3004 Ω-P4：共享低秩（.715/.826）≠xdir=重定向+放大；轴序 bug 修正。'),
    ('npz8 跨 run 位级。**破坏粗粒度易行，伪造须情景 K,V。**',
     'npz8 跨 run 位级。**破坏粗粒度易行，伪造须情景 K,V。**'
     ' 3015 Ω-P2i：**K 消费=领先者+情景背景**——g7 领先 9/11 位'
     '（share 0.390，置换 p=1.0，有效头 3/8）；头级 K 特异'
     '（V 擦除 ~1e-4）；K 擦除→注意力重定向（熵 .836→.700）'
     '非均匀化；影响 vs 注意力 rho .45。'),
    ('- numpy 标量入 json 转 int()/float()；dict 键容器禁直接 np.array；大小写敏感常量对齐（3014 JOINT bug）。',
     '- numpy 标量入 json 转 int()/float()；dict 键容器禁 np.array；常量大小写对齐（3014）；'
     '**GQA：KV 缓存 8 头（32 query 头共享），头级干预单位=KV 头（3015）**。'),
    ('- max=3014，下一个 3015（A 主选 K 路由机制定位——L3 logic 位 K 的下游消费头/attention 分配与 softmax 熵；B L31 次峰；C 情景性检验（同词异位 K,V 相似度）；D GLM4-9B-Base 对照）。',
     '- max=3015，下一个 3016（A 主选 g7/query头28 下游放大定位——o_proj 输出追踪 L4+；B L31 次峰；C 情景性检验；D 重定向终点测量）。'),
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
        r'\.workbuddy\tmp_mem15.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3015=%s max15=%s'
    % (len(t2), miss, '3015 Ω-P2i' in t2,
       'max=3015' in t2))
print('mem patch ok')
