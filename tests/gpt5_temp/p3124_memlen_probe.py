# -*- coding: utf-8 -*-
"""Probe: simulate MEMORY.md replace chain, report len + counts (no write)."""
import io

MEMO_W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
mem_old = io.open(MEMO_W, encoding='utf-8').read()

NEW_3124 = (u'- 3124（T4）：残差分解 m-结构≈0、共模 '
            u'A1 0.356/P 0.190、余项 0.65–0.81 主导；'
            u'det 重构 r −0.12、Kalman q̂→0=锚点静态、'
            u'own r 0.006 → **(m,锚点,线性算子) 族不足、'
            u'存在第三系统成分**；L35 释压=范数×语义各半'
            u'（0.50/0.47）；**GLM4 跨模型：语法/内容 '
            u'L*=20/20 双方向（Qwen L21/L20）复现写入链'
            u'上游、span 672/672、readout AUC 0.83**。')
NEW_3118_20 = (u'- 3118–3120：AUC 0.981→0.672 强振荡、'
               u'状态补偿 0.52 vs 闭环放大 1.08–1.73；'
               u'答案 token 仅 t=1、振荡由内容步驱动；'
               u'重述步上推（断言侵蚀）、标点步恢复 gap；'
               u'L30/L32 反向于 L26；Δm 线性 slope −0.68。')
NEW_3113_17 = (u'- 3113–3117：MLP 主写 L20–28（相关峰 '
               u'+2.48）、L32 负写=擦除相；head=相关影子、'
               u'三层联合 additive；TOP3=L26/L33/L31；'
               u'A1 首 token 20.2% 分叉。')
NEW_NEXT = (u'- max=3124，下一 3125：**残差第三成分定位'
            u'（自相关/谱+内容尾迹+跨步互相关）+ GLM4 反事实'
            u'生成对照 + Qwen 输入流对照（干预语义变量隔离）'
            u'+ GLM4 写入链层识别**。')

lines = mem_old.splitlines()
out = []
skip = 0
for ln in lines:
    if skip > 0:
        skip -= 1
        continue
    if ln.startswith(u'## 机制链状态'):
        out.append(u'## 机制链状态（3124）')
    elif ln.startswith(u'- 3123'):
        out.append(ln)
        out.append(NEW_3124)
    elif ln.startswith(u'- 3120'):
        out.append(NEW_3118_20)
        skip = 2
    elif ln.startswith(u'- 3114'):
        out.append(NEW_3113_17)
        skip = 1
    elif ln.startswith(u'- max=3123'):
        out.append(NEW_NEXT)
    else:
        out.append(ln)
mem_new = u'\n'.join(out) + u'\n'

rep = []
rep.append(('len_old', len(mem_old)))
rep.append(('len_new', len(mem_new)))
rep.append(('cnt_hdr3124', mem_new.count(u'## 机制链状态（3124）')))
rep.append(('cnt_3124', mem_new.count(u'- 3124（T4）')))
rep.append(('cnt_3123', mem_new.count(u'- 3123（T4）')))
rep.append(('cnt_max3124', mem_new.count(u'max=3124')))
rep.append(('cnt_3119', mem_new.count(u'- 3119：')))
rep.append(('cnt_3118old', mem_new.count(u'- 3118（T4）')))
rep.append(('cnt_3113old', mem_new.count(u'- 3113：')))
rep.append(('cnt_3118_20', mem_new.count(u'- 3118–3120：')))
rep.append(('cnt_3113_17', mem_new.count(u'- 3113–3117：')))

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3124_memlen_report.txt',
          'w', encoding='utf-8') as f:
    for k, v in rep:
        f.write('%s = %r\n' % (k, v))
    f.write('PASS_3000 = %s\n' % (len(mem_new) < 3000))
print('WROTE_OK')
