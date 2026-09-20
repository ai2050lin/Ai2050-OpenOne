# -*- coding: utf-8 -*-
"""Phase 2970 closeout fix: MEMO hash placeholders + MEMORY next."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'\.workbuddy\memory\MEMORY.md')

memo = io.open(MEMO, encoding='utf-8').read()
old = (u'**产物**：`phase2970/delay_carrier_localization/` '
       u'execution {a1} / result {b1} / delay_carrier.npz {c1} '
       u'/ script {d1}。')
new = (u'**产物**：`phase2970/delay_carrier_localization/` '
       u'execution f8bd0454 / result e6b85164 / '
       u'delay_carrier.npz d0835266 / script da58f04a。')
assert old in memo, 'memo placeholder line not found'
memo = memo.replace(old, new, 1)
assert '{a1}' not in memo[memo.find('## Phase 2970'):]
io.open(MEMO, 'w', encoding='utf-8').write(memo)

mem = io.open(MEMFILE, encoding='utf-8').read()
old7 = (u'当前 max=**2970**，下一个 **2970**（候选 A 跨语言峰位延迟载体定位'
        u'（fr-en 峰位差 13 对词全层头解剖）；B 语言×词类双因子签名矩阵'
        u'（n≥60 合并词表）；C h8/h21 峰位词属性离线检验；D 2961 卡组扩充'
        u'补 2962-2969）。')
new7 = (u'当前 max=**2970**，下一个 **2971**（候选 A 机制链收官卡片扩充'
        u'——2962-2970 九环入 2961 卡组（纯文档 Phase，含四语言表结构注记）；'
        u'B 语言×词类双因子签名矩阵（n≥60 合并词表预注册）；C 延迟头群功能'
        u'身份（top 延迟头消融，2965 机器）；D h8/h21 峰位词属性离线检验）。')
assert old7 in mem, 'memory L7 anchor not found'
mem = mem.replace(old7, new7, 1)
io.open(MEMFILE, 'w', encoding='utf-8').write(mem)

m2 = io.open(MEMO, encoding='utf-8').read()
mm = io.open(MEMFILE, encoding='utf-8').read()
res = [
    'memo hash line fixed: %s' % ('execution f8bd0454' in m2),
    'memory next2971: %s' % ('下一个 **2971**' in mm),
]
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_fix6.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('fixed')
