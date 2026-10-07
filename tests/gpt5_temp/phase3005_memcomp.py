# -*- coding: utf-8 -*-
"""Phase 3005 MEMORY compress round 2: dedupe 3004,
fix %% literals, compress. Idempotent."""
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


rep('\n- 3004 算子结构（qwen L17 带 r1 .715/medcos '
    '.826 共享重定向；a12a 揭 3001/3002 T3 轴序 bug '
    'bit 级，T3_corrected 替代：L4 注入 93%% 衰减、'
    'L17 先降后放大 1.66×）；3005 算子结构（GLM4 L19 '
    '带 r1 .189/medcos .130=逐细胞散射，L20 头部 '
    '.41/.52 单调衰减；L19 注入 91%% 衰减+方向清除；'
    'a6/a9/a12a 全 0.0 bit 级）。**跨模型对照闭环：'
    'qwen 构造性重写 vs GLM4 破坏性重写**。\n',
    '\n- 3005 算子结构：GLM4 L19 带 r1 .189/medcos '
    '.130=逐细胞散射（L20 头部 .41/.52 单调衰减）；'
    'L19 注入 91% 衰减+方向清除；a6/a9/a12a 全 0.0 '
    'bit 级。**跨模型对照闭环：qwen 构造性重写 vs '
    'GLM4 破坏性重写**。\n', 'dedupe')
rep('3003 Ω-G3：asymmetric_amplification——中带=xdir '
    '特异剂量线性符号不对称压缩通道（−xdir 弱 34-45%、'
    '双向压 sep），排除对称无洗消/整流；GLM4-qwen 中带'
    '谱系完成。方案 v5 发布。',
    '3003 Ω-G3：中带=xdir 特异剂量线性符号不对称压缩'
    '通道（−xdir 弱 34-45%），GLM4-qwen 谱系完成；'
    '方案 v5 发布。', 'c3003')
rep('3004 Ω-P4：算子低秩共享（r1 0.715/medcos 0.826 '
    'vs null 0.022）但共享向≠xdir（aligned 0.0075）'
    '=重定向+放大非守恒；**3001/3002 T3 轴序 bug '
    'bit 级证实**（(1,n,hid) 跨细胞算错维，a12a=0 '
    '锁定），T3_corrected 替代（L4 注入 93% 次层衰减、'
    'L17 先降 70% 再放大 1.66×），TRK_DROP 门退役，'
    'GLM4 重算=3005。',
    '3004 Ω-P4：qwen 算子共享低秩（r1 .715/medcos '
    '.826）但共享向≠xdir=重定向+放大；**3001/3002 '
    'T3 轴序 bug bit 级**（(1,n,hid) 错维，a12a=0），'
    'T3_corrected 替代，TRK_DROP 退役，GLM4 重算'
    '=3005。', 'c3004')
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
o.append('len=%d miss=%s' % (len(t2), miss))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem05b.txt', 'w',
        encoding='utf-8').write('\n'.join(o))
print('ok')
