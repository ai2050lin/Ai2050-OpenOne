# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('2993 Ω-D 逻辑签名长度稳健；2994 Ω-E 类轴注入=轴防御。',
     '2993 Ω-D 逻辑签名长度稳健；2994 轴防御。'),
    ('2995-2999 Ω-F GLM4：复制+重分级；xdir ~98% 洗消；词携带；门无量纲。',
     '2995-2999 Ω-F GLM4：复制重分级；xdir ~98% 洗消。'),
    ('3005 算子结构：GLM4 L19 带 r1 .189=逐细胞散射；**跨模型闭环：qwen 构造性 vs GLM4 破坏性重写**。',
     '3005：GLM4 r1 .189=逐细胞散射；**闭环：qwen 构造性 vs GLM4 破坏性重写**。'),
    ('3006 Ω-A1 Base：**中带机制与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784 头 .78/.90；T3 尾 1.69×；sep 192.9/185.7）；对齐仅上调（s2 1.08→1.50）；eraser=中位边界效应；23.2×=家族差异。',
     '3006 Base：**中带与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784；尾 1.69×；s2 1.08→1.50；eraser=边界效应；23.2×=家族差异）。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

add = (' 3008 Ω-P2b：**2993 held-out 轴生成态分离 logic>content（p_fam 0.0002×2，n 114/1280）**；KV 全强度消融饱和（任意位 22-31/32 分歧，D=2.0 p=0.407）→需分级干预；256 tok 漂移收缩保持。')
anchor = '恢复非 xdir 特异（生成态）。'
assert anchor in t
t = t.replace(anchor, anchor + add, 1)

old_next = ('- max=3007，下一个 3008（A 主选逻辑锁定升级：2993 签名机器移植生成态+长生成+逻辑位 KV 因果探测；B P2b 扰动-恢复全网格 t0×scale×方向；C Ω-A2 GLM4 家族对照）。方案 v5。')
new_next = ('- max=3008，下一个 3009（A 主选 KV 缩放因果扫描 scale 0.2-0.8 恢复 headroom+逻辑位特异性重测；B P2b 扰动-恢复全网格；C Ω-A2 GLM4 家族对照）。方案 v5。')
if old_next in t:
    t = t.replace(old_next, new_next, 1)
else:
    miss.append('next')

t = t.replace('（2936-3007）', '（2936-3008）', 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem08.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3008=%s'
    % (len(t2), miss, '3008' in t2))
print('ok')
