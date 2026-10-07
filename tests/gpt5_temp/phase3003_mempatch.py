# -*- coding: utf-8 -*-
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般性洗消（非选择）；',
     '3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般洗消；'),
    ('2998/2999 Ω-F2 GLM4：xdir 注入 ~98% 洗消'
     '（max 0.128@L4、s_c>6）；类分离词 token 携带；'
     '门无量纲。',
     '2998/2999 Ω-F2 GLM4：xdir ~98% 洗消'
     '（max 0.128@L4）；类分离词携带；门无量纲。'),
    ('3002 Ω-G2 qwen 镜像：判决 context_entangled——'
     '词携带仅 63%（同词上下文砍 59%）vs GLM4 86%；'
     '中带 xdir 特异放大带 ~150× over 随机（sub 0.0098 '
     'vs xdir 1.50），L17 一路传至 L35、L4 即杀=与 GLM4 '
     '传播带位置/符号双重反转。',
     '3002 Ω-G2 qwen 镜像：context_entangled——词携带 63%'
     '（同词上下文砍 59%）vs GLM4 86%；中带 xdir 特异 '
     '~150×、L17 传至 L35、L4 即杀=双重反转。'),
    ('L17 传至 L35、L4 即杀=双重反转。核心：',
     'L17 传至 L35、L4 即杀=双重反转。3003 Ω-G3：'
     'asymmetric_amplification——中带=xdir 特异剂量线性'
     '符号不对称压缩通道（−xdir 弱 34-45%、双向压 sep），'
     '排除对称无洗消/整流；GLM4-qwen 中带谱系完成。'
     '方案 v5 发布。核心：'),
    ('- max=3002，下一个 3003（A 主选 qwen 放大带剂量律+'
     '镜像对称性；B 3001/3002 差异头级定位；C 2989 T3 '
     '加密+k 剂量；D Ω-E 错误吸引子操作化）。方案 v4：',
     '- max=3003，下一个 3004（A v5-P1 Qwen3-4B-Base '
     '对照三件套（下载前置）；B v5-P4 Dl 低秩+能量分账'
     '（零 forward）；C v5-P2a 生成轨迹记录器；D v5-P3 '
     '跨语言同构）。方案 v5：'),
]
for old, new in pairs:
    assert old in t, 'MISS: %s' % old[:30]
    t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
for _, new in pairs:
    assert new in t2, 'NEW MISS: %s' % new[:30]
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_len3.txt', 'w',
        encoding='utf-8').write('len=%d' % len(t2))
print('mem patched ok')
