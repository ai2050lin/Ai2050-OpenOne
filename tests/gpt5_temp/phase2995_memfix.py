# -*- coding: utf-8 -*-
import io

MP = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
      r'\.workbuddy\memory\MEMORY.md')
mm = io.open(MP, encoding='utf-8').read()
out = []
pairs = [
    # rejoin 2994 orphaned tail + drop duplicated fragment
    ('2995 Ω-F1 GLM4：类分离+头集中复制（p 0.0014/maxT 地板），'
     '注册表不复制（z 负）首分化卡（存活 6%，随机反放大 168%）'
     '——读出轴防御；错误吸引子未操作化。',
     '2995 Ω-F1 GLM4：类分离+头集中复制（p 0.0014/maxT 地板），'
     '注册表不复制（z 负）首分化卡。错误吸引子未操作化。'),
    ('机制链状态（2936-2994）', '机制链状态（2936-2995）'),
    ('A 主选 Ω-F 跨模型卡片复制 GLM4-9B——卡片集 v2 带适用域标签，'
     '锚结构按新模型重建；B 2989 T3 加密+k 剂量；'
     'C 逻辑签名头级因果复测；D KV-cache 时序滞后补遗',
     'A 主选 Ω-F2 续卡复制（s_c 注入机器+词盲卡）；'
     'B T3 空间口径复审；C L 词轴位定量；D 2989 T3 加密+k 剂量'),
    ('（时间取 execution created）', '（时间=created）'),
    ('2989 MLP 注册表：lang 分布式（因果 6/9 层）',
     '2989 注册表：lang 分布式（6/9 层）'),
    ('对齐超 null 但符号一致率 0.405——特征对齐=快照',
     '对齐超 null 但符号一致率 0.405=快照'),
]
for a, b in pairs:
    if a in mm:
        mm = mm.replace(a, b, 1)
    else:
        out.append('MISS ' + a[:24])
io.open(MP, 'w', encoding='utf-8').write(mm)
out.insert(0, 'chars=%d ok3000=%s max2995=%s next2996=%s'
           % (len(mm), len(mm) <= 3000, 'max=2995' in mm,
              '**2996**' in mm))
out.append('chain2995=%s orphan_clean=%s'
           % ('2995 Ω-F1' in mm,
              '（存活 6%' not in mm))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem2995b.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('done')
