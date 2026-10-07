# -*- coding: utf-8 -*-
"""Phase 2980 closeout fix2: worklog + MEMORY only
(MEMO already appended)."""
import io

WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'\.workbuddy\memory\MEMORY.md')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2980' not in wl:
    entry = (u"\n## Phase 2980（2026-09-20）竞争瓶颈头功能身份\n"
             u"- 判决 bottleneck_heads_carry_interaction（run3 权威，锚 7/7，a5/a6 均 0.00）。\n"
             u"- 两次 correction：run1 a2/a3 锚对象错（单位向量 vs raw 向量；n17 应为 L17 层输入 2560 维范数）；run2 误用 2977 npz raw 向量致剂量放大 4.5-14 倍（根因：2977 npz 存归一化前向量）。\n"
             u"- 核心：消融 h12 使交互消失约 97 个百分点量级（0.02358→0.00056），随机头小于 2 个百分点，T3 mean|dB| 均衡（非全局破坏）——交互项头级集中、主效应分布式的头级分化。\n"
             u"- Ledger 119 条 / L14 87 / hash cc4bf46a。\n")
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')
else:
    print('wslog already present, skip')

mem = io.open(MEMFILE, encoding='utf-8').read()
changed = False
old_next = (u'max=2979，下一个 **2980**（A 主选：竞争瓶颈头 h9/h12 消融'
            u'功能身份（2965 机器）；B 2978 dose law 分层修正复测（离线）；'
            u'C Ω-C 长上下文开题；D 同词跨语言语境对照）。方案 v3 见')
new_next = (u'max=2980，下一个 **2981**（A 主选：h12 功能画像——write '
            u'identity + 头级 g/cos 机制分解；B h12 消融主效应归宿'
            u'（重平衡检验）；C Ω-C 长上下文开题；D 2978 dose law '
            u'分层修正复测）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
elif 'max=2980' not in mem:
    print('WARN: memory next-candidate line not found')
old_chain = (u'2979 权威：转正分布式无正头，h9/h12 同头持续，'
             u'g=0.71/cos=0.96 饱和挤压倾向。')
new_chain = (u'2979 权威：转正分布式无正头，h9/h12 同头持续，'
             u'g=0.71/cos=0.96 饱和挤压倾向→2980 消融定案：'
             u'h12 是交互必要载体（消融消失 97%，随机头 <2%，'
             u'无全局破坏）——交互项头级集中、主效应分布式。')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
elif '2980 消融定案' not in mem:
    print('WARN: chain line not found')
if changed:
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout-fix2 done')
