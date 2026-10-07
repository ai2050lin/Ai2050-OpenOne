# -*- coding: utf-8 -*-
"""MEMORY.md patch after Phase 3004: append chain +
compress elders + new engineering rule + next step."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
orig = len(t)

pairs = [
    # 1) chain header
    ('## 机制链状态（2936-3002）',
     '## 机制链状态（2936-3004）'),
    # 2) compress early chain
    ('2938 子空间→2940 词盲→2943 线性壳→2945 阈值→2947 '
     '头集中→2949 组反转→2950 重平衡→2951 承重→2952 路由'
     '跳变 2960 秩1 2962 词类签名→2963 类效应→2964 载体 '
     'h15→2965 消融→2967 塌缩→2968 峰锁→2969 fr 晚→2970 '
     '延迟→',
     '2938-2970：子空间/词盲/线性壳/阈值/头集中/组反转/'
     '重平衡/承重/路由跳变/秩1/词类签名/类效应/载体h15/'
     '消融/塌缩/峰锁/fr晚/延迟；'),
    ('2995 Ω-F1 GLM4：类分离+头集中复制。2996 审计：2995 '
     'dirs 退化→T3 作废；双口径双模型在场。2997 重分级 '
     'T3_M2989→replicated；cards_v21 立 basis-hash 溯源。',
     '2995-2997 Ω-F1 GLM4 复制+审计重分级+basis-hash。'),
    ('3002 Ω-G2 qwen 镜像：context_entangled——词携带 63%'
     '（同词上下文砍 59%）vs GLM4 86%；中带 xdir 特异 '
     '~150×、L17 传至 L35、L4 即杀=双重反转。',
     '3002 Ω-G2 qwen：context_entangled——词携带 63% vs '
     '86%；中带 xdir 特异 ~150×。'),
    # 3) append 3004 after 3003 sentence
    ('方案 v5 发布。核心：null 重编码全层分布式涌现，对单'
     '点操作化关闭；头级重要性=关系属性。',
     '方案 v5 发布。3004 Ω-P4：算子低秩共享（r1 0.715/'
     'medcos 0.826 vs null 0.022）但共享向≠xdir'
     '（aligned 0.0075）=重定向+放大非守恒；**3001/3002 '
     'T3 轴序 bug bit 级证实**（(1,n,hid) 跨细胞算错维，'
     'a12a=0 锁定），T3_corrected 替代（L4 注入 93% 次层'
     '衰减、L17 先降 70% 再放大 1.66×），TRK_DROP 门退役'
     '，GLM4 重算=3005。核心：null 重编码全层分布式涌现'
     '，对单点操作化关闭；头级重要性=关系属性。'),
    # 4) engineering rule append (compress line 1st)
    ('- logits /√HD；self_attn 输出 tuple；model.norm '
     'pre-hook 捕 final-norm 输入；dirs_word=post-LN '
     'attnin 口径不混用。',
     '- logits /√HD；self_attn 输出 tuple；model.norm '
     'pre-hook 捕 final-norm 输入；dirs_word=post-LN '
     'attnin 口径不混用；**逐层捕获 np.stack 单元素列表→'
     '(1,n,hid)，SVD/范数/投影前必须 [0] 或按声明维 axis'
     '（3004 轴序 bug 教训）**。'),
    # 5) next step
    ('- max=3003，下一个 3004（A v5-P1 Qwen3-4B-Base 对照'
     '三件套（下载前置）；B v5-P4 Dl 低秩+能量分账（零 '
     'forward）；C v5-P2a 生成轨迹记录器；D v5-P3 跨语言'
     '同构）。方案 v5：research\\gpt5\\docs\\'
     'plan_v4_micro_macro_merge.md。',
     '- max=3004，下一个 3005（A 主选 GLM4 T3 轴序 bug '
     '重算双剖面→跨模型算子对照；B Base 已就位（8.06GB '
     '三 sha 验证）→对齐三件套 v5-P1；C 生成轨迹 v5-P2a'
     '；D 跨语言同构 v5-P3）。方案 v5：research\\gpt5\\'
     'docs\\plan_v5_dynamic_manifold_control.md。'),
]
miss = []
for old, new in pairs:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        miss.append(old[:40])
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem04.txt', 'w',
        encoding='utf-8').write(
    'miss=%s len %d->%d max04=%s chain04=%s\n' % (
        miss, orig, len(t2),
        'max=3004' in t2, '3004 Ω-P4' in t2))
print('ok')
