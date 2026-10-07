# -*- coding: utf-8 -*-
"""Phase 3006 MEMORY patch: append 3006, update next,
equal-length compressions. Idempotent."""
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


rep('## 机制链状态（2936-3005）',
    '## 机制链状态（2936-3006）', 'hdr')
rep('- 3005 算子结构：GLM4 L19 带 r1 .189/medcos '
    '.130=逐细胞散射（L20 头部 .41/.52 单调衰减）；'
    'L19 注入 91% 衰减+方向清除；a6/a9/a12a 全 0.0 '
    'bit 级。**跨模型对照闭环：qwen 构造性重写 vs '
    'GLM4 破坏性重写**。',
    '- 3005 算子结构：GLM4 L19 带 r1 .189=逐细胞散射'
    '（注入 91% 衰减+方向清除）；**跨模型对照闭环：'
    'qwen 构造性重写 vs GLM4 破坏性重写**。3006 Ω-A1 '
    'Base（全内部几何）：判决 base_shared_eraser（2-D '
    '门）——**中带机制与 chat 同构=预训练涌现非对齐'
    '雕刻**（r1 .656/medcos .784 头部 .78/.90；剂量'
    '线性 1.08@s2；T3 尾 1.69× vs chat 1.66×；sep_f '
    '192.9 vs chat 185.7）；对齐仅上调（s2 1.08→'
    '1.50）；eraser 标签=band 中位边界效应（剖面实为'
    '先衰减后放大）；23.2× 免疫差=家族差异。',
    'next')
rep('## 下一步\n- max=3005，下一个 3006（A 主选 Base '
    '对齐三件套 v5-P1：语言轴扫描+剂量+算子结构→对齐 '
    'delta 剖面；B 生成轨迹 v5-P2a；C GLM4 L20 洗消'
    '启动时刻精扫）。方案 v5。',
    '## 下一步\n- max=3006，下一个 3007（A 主选生成'
    '轨迹记录器 v5-P2a 自回归动力学；B Ω-A2 GLM4-9B-'
    'Base 下载+家族对照补齐空白一；C Base 剂量加密定'
    '饱和点）。方案 v5。',
    'n2')
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
o.append('len=%d miss=%s' % (len(t2), miss))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem06.txt', 'w',
        encoding='utf-8').write('\n'.join(o))
print('ok')
