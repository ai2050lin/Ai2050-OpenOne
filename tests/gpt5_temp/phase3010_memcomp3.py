# -*- coding: utf-8 -*-
"""Phase 3010 MEMORY micro compression round 3."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('2992 字典：符号率 0.405=快照。2993 逻辑签名长度稳健；2994 轴防御。',
     '2992 字典：符号率 0.405=快照。2993 逻辑签名长度稳健；2994 轴防御。'),
    ('2995-2999 Ω-F GLM4：重分级；xdir ~98% 洗消。3000 Ω-F：qwen/GLM4 23.2×；L3 塌缩。',
     '2995-2999 Ω-F GLM4：重分级；xdir ~98% 洗消。3000：qwen/GLM4 23.2×；L3 塌缩。'),
    ('3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+放大；**T3 轴序 bug bit 级**，T3_corrected 替代。',
     '3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+放大；**T3 轴序 bug bit 级**修正。'),
    ('3005：GLM4 r1 .189=逐细胞散射；**闭环：qwen 构造性 vs GLM4 破坏性**。',
     '3005：GLM4 r1 .189=逐细胞散射；**闭环：qwen 构造性 vs GLM4 破坏性**。'),
    ('3006 Base：**中带与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784；尾 1.69×；23.2×=家族差异）。',
     '3006 Base：**中带与 chat 同构=预训练涌现非对齐雕刻**（r1 .656/.784；尾 1.69×）。'),
    ('3008 Ω-P2b：**held-out 轴生成态分离 logic>content（p 0.0002×2）**；KV 消融饱和（任意位 22-31/32）→分级干预。',
     '3008 Ω-P2b：**held-out 轴生成态分离 logic>content（p 0.0002×2）**；KV 零化饱和（任意位 22-31/32）→分级干预。'),
    ('- "好的，继续"=AI 主导不停；结构化输出；主线第一调用须是主线工具。',
     '- "好的，继续"=AI 主导不停；结构化输出。'),
    ('MEMO append-only 唯一目标：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。',
     'MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存）。'),
]
miss = []
for a, b in pairs:
    if a in t and a != b:
        t = t.replace(a, b, 1)
    elif a != b:
        miss.append(a[:20])

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem10c.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
