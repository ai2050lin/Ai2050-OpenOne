# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('## 机制链状态（2936-3001）',
     '## 机制链状态（2936-3002）'),
    ('2982 三要素否定（-0.51，h9 反平行）',
     '2982 三要素否定（-0.51）'),
    ('2996 审计：2995 dirs 全层退化（单槽 hook）→T3 作废；'
     '真 K2 z +10~31、K1 share p=0——双口径双模型在场。',
     '2996 审计：2995 dirs 退化→T3 作废；双口径双模型在场。'),
    ('2998 Ω-F2 GLM4 版负：xdir 注入 L17-19 sep 免疫'
     '（0.027 vs 0.86）~98% 洗消；类分离由词 token 携带；'
     '门须无量纲。2999 扫描：无层位偏移带（max 0.128@L4）'
     's_c>6 全局洗消；弱带 L2-4<L7-13<qwen L15-17。',
     '2998/2999 Ω-F2 GLM4：xdir 注入 ~98% 洗消'
     '（max 0.128@L4、s_c>6）；类分离词 token 携带；'
     '门无量纲。'),
    ('3000 Ω-F 收官：qwen 处处传播（band 25/28、area 0.829、'
     'a9 bit 级 0.0）vs GLM4 0.036/2——刻度差 23.2×；'
     '注记 qwen L3 双符号均塌缩=早层部分幅度性。',
     '3000 Ω-F 收官：qwen 处处传播（area 0.829）vs GLM4 '
     '0.036——23.2×；qwen L3 双符号塌缩。'),
    ('3001 Ω-G1：GLM4 鲁棒=词 token 携带（word_eff 86% vs '
     'ctx 23%）+一般性洗消（随机方向更弱=非选择性）；'
     'L19 注入次层即杀、L4 存续但 xdir 分量正交化。',
     '3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般性洗消（非选择）；'
     'L19 次层即杀、L4 存续正交化。'),
    ('L19 次层即杀、L4 存续正交化。核心：null 重编码全层'
     '分布式涌现，对单点操作化关闭；头级重要性=关系属性。',
     'L19 次层即杀、L4 存续正交化。3002 Ω-G2 qwen 镜像：'
     '判决 context_entangled——词携带仅 63%（同词上下文砍 '
     '59%）vs GLM4 86%；中带 xdir 特异放大带 ~150× over '
     '随机（sub 0.0098 vs xdir 1.50），L17 一路传至 L35、'
     'L4 即杀=与 GLM4 传播带位置/符号双重反转。核心：'
     'null 重编码全层分布式涌现，对单点操作化关闭；'
     '头级重要性=关系属性。'),
    ('- max=3001，下一个 3002（A 主选 qwen 词 token swap '
     '分解镜像；B L4 存续/L20 即杀分界；C 2989 T3 加密+k '
     '剂量；D Ω-E 错误吸引子）。方案 v4：',
     '- max=3002，下一个 3003（A 主选 qwen 放大带剂量律+'
     '镜像对称性；B 3001/3002 差异头级定位；C 2989 T3 '
     '加密+k 剂量；D Ω-E 错误吸引子操作化）。方案 v4：'),
]
for old, new in pairs:
    assert old in t, 'MISS: %s' % old[:30]
    t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
for _, new in pairs:
    assert new in t2, 'NEW MISS: %s' % new[:30]
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_len2.txt', 'w',
        encoding='utf-8').write('len=%d' % len(t2))
print('mem patched ok')
