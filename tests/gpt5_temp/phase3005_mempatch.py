# -*- coding: utf-8 -*-
"""Phase 3005 MEMORY patch: append 3005 + update next,
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


rep('## 机制链状态（2936-3004）',
    '## 机制链状态（2936-3005）', 'hdr')
rep('## 下一步\n- max=3004，下一个 3005（A 主选 GLM4 '
    'T3 轴序 bug 重算双剖面→跨模型算子对照；B Base '
    '已就位（8.06GB 三 sha 验证）→对齐三件套 v5-P1；'
    'C 生成轨迹 v5-P2a；D 跨语言同构 v5-P3）。方案 v5：'
    'research\\gpt5\\docs\\plan_v5_dynamic_manifold_'
    'control.md。',
    '- 3004 算子结构（qwen L17 带 r1 .715/medcos .826 '
    '共享重定向；a12a 揭 3001/3002 T3 轴序 bug bit 级，'
    'T3_corrected 替代：L4 注入 93%% 衰减、L17 先降后放大 '
    '1.66×）；3005 算子结构（GLM4 L19 带 r1 .189/medcos '
    '.130=逐细胞散射，L20 头部 .41/.52 单调衰减；'
    'L19 注入 91%% 衰减+方向清除；a6/a9/a12a 全 0.0 '
    'bit 级）。**跨模型对照闭环：qwen 构造性重写 vs GLM4 '
    '破坏性重写**。\n\n## 下一步\n- max=3005，下一个 '
    '3006（A 主选 Base 对齐三件套 v5-P1：语言轴扫描+'
    '剂量+算子结构→对齐 delta 剖面；B 生成轨迹 v5-P2a；'
    'C GLM4 L20 洗消启动时刻精扫）。方案 v5。',
    'next')
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
o.append('len=%d miss=%s' % (len(t2), miss))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem05.txt', 'w',
        encoding='utf-8').write('\n'.join(o))
print('ok')
