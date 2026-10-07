# -*- coding: utf-8 -*-
"""Phase 3006 MEMORY compress round 2. Idempotent."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
o = []
miss = []


def rep(old, new, tag):
    global t
    if old in t:
        t = t.replace(old, new, 1)
        o.append(tag)
    else:
        miss.append(tag)


rep('2995-2997 Ω-F1 GLM4 复制+审计重分级+basis-hash。',
    '2995-2997 Ω-F1 GLM4 复制+审计重分级。', 'f1')
rep('2993 Ω-D 逻辑签名长度稳健；2994 Ω-E 类轴注入'
    '被擦除、随机反放大=轴防御。',
    '2993 Ω-D 逻辑签名长度稳健；2994 Ω-E 类轴注入='
    '轴防御。', 'de')
rep('3002 Ω-G2 qwen：context_entangled——词携带 63% '
    'vs 86%；中带 xdir 特异 ~150×。',
    '3002 Ω-G2 qwen：词携带 63% vs 86%；中带 xdir '
    '特异 ~150×。', 'g2')
rep('3000 Ω-F：qwen area 0.829 vs GLM4 0.036（23.2×）',
    '3000 Ω-F：qwen 0.829 vs GLM4 0.036（23.2×）',
    'f0')
rep('3001 Ω-G1：GLM4 鲁棒=词携带 86%+一般洗消（T3 '
    '数值作废见 3004）。',
    '3001 Ω-G1：GLM4 词携带 86%+一般洗消（T3 作废见 '
    '3004）。', 'g1')
rep('（r1 .715/medcos .826）但共享向≠xdir=重定向+'
    '放大；**3001/3002 T3 轴序 bug bit 级**（(1,n,'
    'hid) 错维，a12a=0），T3_corrected 替代，TRK_DROP '
    '退役，GLM4 重算=3005。',
    '（.715/.826）但共享向≠xdir=重定向+放大；**3001/'
    '3002 T3 轴序 bug bit 级**（(1,n,hid) 错维），'
    'T3_corrected 替代，TRK_DROP 退役。', 'p4')
rep('3006 Ω-A1 Base（全内部几何）：判决 '
    'base_shared_eraser（2-D 门）——',
    '3006 Ω-A1 Base：base_shared_eraser（2-D 门）——',
    'a1')
rep('sep_f 192.9 vs chat 185.7', 'sep_f 192.9/185.7',
    'sep')
rep('（剖面实为先衰减后放大）', '（实为先衰减后放大）',
    'lab')
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
o.append('len=%d miss=%s' % (len(t2), miss))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem06b.txt', 'w',
        encoding='utf-8').write('\n'.join(o))
print('ok')
