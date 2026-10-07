# -*- coding: utf-8 -*-
"""MEMORY patch (3017): add P2k entry, fix the
self_attn pre-hook gauge note, equal-length
compression."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
miss = []

pairs = [
    # compress 3005-3006
    ('GLM4 r1 .189=逐细胞散射；**qwen 构造性 vs GLM4 破坏性**。3006 Base：**中带同构=预训练涌现**（r1 .656/.784）。3007 Ω-P2a：锁定比 0.457',
     'GLM4 r1 .189=散射；**qwen 构造性 vs GLM4 破坏性**。3006 Base：**中带同构=预训练涌现**。3007：锁定比 .457'),
    # compress 3009
    ('KV 饱和与强度无关→token 分歧退役；坐标 logic<sham。',
     'KV 饱和与强度无关→token 分歧退役。'),
    # compress 3011
    ('**门控定位于 L3 KV**（D=.0137 p_maxT 1e-4；次峰 L31）。',
     '**门控定位于 L3 KV**（D=.0137 p 1e-4；次峰 L31）。'),
    # compress 3014
    ('——retain(0.5)=1.011（半≥全）；**K/V 分岔**：KONLY 非单调 vs VONLY 单调=K 路由破坏。',
     '（半≥全擦除）；**K/V 分岔**=K 路由破坏（KONLY 非单调 vs VONLY 单调）。'),
    # compress 3016
    ('单载体复原仅 8.5%（散布 -.83~+.39，delta 能量≠效应）；l* 散布 L4-24、qh* 9 头、层内 top1 .128；lens 中层 29×（L8）但 L32/35 收敛 .76/.56；sham delta .865 无效应',
     '单载体复原 8.5%（散布 ±.8，delta 能量≠效应）；l* L4-24、qh* 9 头、层内 top1 .128；lens 中层 29× 但深层收敛 <1；sham 无效应'),
    # compress 2995-3000
    ('2995-3000 Ω-F：重分级；xdir 98% 洗消；qwen/GLM4 23.2×。',
     '2995-3000 Ω-F：重分级；xdir 98% 洗消。'),
    # compress 2992-2994
    ('2992 符号率 .405=快照；2993 签名稳健；2994 轴防御。',
     '2992-2994：符号率 .405=快照；签名稳健；轴防御。'),
    # compress 3004
    ('3004 Ω-P4：共享低秩≠xdir=重定向+放大；轴序 bug 修正。',
     '3004 Ω-P4：共享低秩≠xdir=重定向+放大。'),
    # compress 3015
    ('**K 消费=领先 g7+情景背景**（9/11 位，share .390，p=1.0；V 擦除~1e-4=K 特异）',
     '**K 消费=领先 g7+情景背景**（share .390，p=1.0；V 擦除~1e-4=K 特异）'),
    # env-defect: drop session anecdote
    ('Edit 幻影→Python 补丁（本会话三遇+补丁锚 miss 两次）；replace 未命中→先 Grep 再跑。',
     'Edit 幻影→Python 补丁；replace 未命中→先 Grep 再跑。'),
    # engineering note fix (3017 lesson)
    ('层输入挂 self_attn pre-hook（破坏 RoPE），args 空用 kwargs；单样本保 batch 维；逐词 LS 禁跨词聚合。',
     'args 空用 kwargs；单样本保 batch 维；逐词 LS 禁跨词聚合；**真残差流=decoder-layer pre-hook，self_attn pre-hook=post-LN 口径（3017）**；恒等门分母与噪声同尺度（bf16 差分放大）。',
     ),
    # next-step line
    ('- max=3016，下一个 3017（A 主选 深层收敛机制——L28-35 谁消化 KV 扰动（晚层响应剖面+补偿方向）；B L31 次峰；C 情景性检验；D 重定向终点）。',
     '- max=3017，下一个 3018（A 主选 稀释定标——范数增长层剖面 attn/mlp 预算，分割稀释 vs 抵消份额；B L31 次峰；C 情景性检验；D 重定向终点）。'),
    # chain header
    ('## 机制链状态（2936-3016）',
     '## 机制链状态（2936-3017）'),
]
for pair in pairs:
    a, b = pair[0], pair[1]
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

# append 3017 entry after the 3015 entry
a = '影响-注意力 rho .45。'
b = ('影响-注意力 rho .45。 3017 Ω-P2k：**吸收=混合**'
     '——反平行带=中带 L5-25+27（L10 −.496，attn/mlp '
     '双反平行，maxT 2e-4）非晚层；‖e‖ 沿深增 10×、'
     '相对误差降 3.7×——lens 收敛=稀释+部分抵消，'
     '无晚层补偿器。')
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('3015-tail')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem17.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s has3017=%s max17=%s lesson=%s'
    % (len(t2), miss, '3017 Ω-P2k' in t2,
       'max=3017' in t2,
       'decoder-layer pre-hook' in t2))
print('ok')
