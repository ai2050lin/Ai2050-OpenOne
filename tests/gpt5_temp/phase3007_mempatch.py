# -*- coding: utf-8 -*-
import io
P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
pairs = [
    ('## 机制链状态（2936-3006）',
     '## 机制链状态（2936-3007）'),
    ('/单token坍缩≠漂移/普查v2/注册表/L12#2消融=快照/'
     '头谱=路由关系属性。',
     '/坍缩≠漂移/普查/注册表/L12#2快照/'
     '头谱=路由关系属性。'),
    ('2995-2997 Ω-F1 GLM4 复制+审计重分级。'
     '2998/2999 Ω-F2 GLM4：xdir ~98% 洗消；'
     '类分离词携带；门无量纲。',
     '2995-2999 Ω-F GLM4：复制+重分级；'
     'xdir ~98% 洗消；词携带；门无量纲。'),
    ('3000 Ω-F：qwen 0.829 vs GLM4 0.036（23.2×）；'
     'L3 双符号塌缩。',
     '3000 Ω-F：qwen/GLM4 23.2×；L3 塌缩。'),
    ('3001 Ω-G1：GLM4 词携带 86%+一般洗消'
     '（T3 作废见 3004）。',
     '3001 Ω-G1：GLM4 词携带 86%+一般洗消。'),
    ('（注入 91% 衰减+方向清除）；**跨模型对照闭环：'
     'qwen 构造性重写 vs GLM4 破坏性重写**。',
     '；**跨模型闭环：qwen 构造性 vs GLM4 '
     '破坏性重写**。'),
    ('3006 Ω-A1 Base：base_shared_eraser（2-D 门）——'
     '**中带机制与 chat 同构=预训练涌现非对齐雕刻**'
     '（r1 .656/medcos .784 头部 .78/.90；剂量线性 '
     '1.08@s2；T3 尾 1.69× vs chat 1.66×；sep_f '
     '192.9/185.7）；对齐仅上调（s2 1.08→1.50）；'
     'eraser 标签=band 中位边界效应（实为先衰减后'
     '放大）；23.2× 免疫差=家族差异。',
     '3006 Ω-A1 Base：**中带机制与 chat 同构=预训练'
     '涌现非对齐雕刻**（r1 .656/.784 头 .78/.90；'
     'T3 尾 1.69×；sep 192.9/185.7）；对齐仅上调'
     '（s2 1.08→1.50）；eraser=中位边界效应；'
     '23.2×=家族差异。 3007 Ω-P2a 生成轨迹记录器：'
     '锁定比 0.457（logic 11.74<content 25.70 类序'
     '单调）；扰动 2/4 divergent 但 3/4 零 token '
     '分歧=argmax 阈下；恢复非 xdir 特异（生成态）。'),
    ('- max=3006，下一个 3007（A 主选生成轨迹记录器 '
     'v5-P2a 自回归动力学；B Ω-A2 GLM4-9B-Base 下载'
     '+家族对照补齐空白一；C Base 剂量加密定饱和点）。'
     '方案 v5。',
     '- max=3007，下一个 3008（A 主选逻辑锁定升级：'
     '2993 签名机器移植生成态+长生成+逻辑位 KV 因果'
     '探测；B P2b 扰动-恢复全网格 t0×scale×方向；'
     'C Ω-A2 GLM4 家族对照）。方案 v5。'),
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
        r'\.workbuddy\tmp_mem07.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
