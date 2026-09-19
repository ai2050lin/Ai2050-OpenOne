import os

memo = 'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
sec = open('D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/p2846_memo_section.md',
           encoding='utf-8').read()
with open(memo, 'a', encoding='utf-8') as f:
    f.write(sec)

log = ('C:/Users/Admin/WorkBuddy/2026-09-17-01-30-05/.workbuddy/'
       'memory/2026-09-17.md')
entry = '''

## Phase 2846（2026-09-18，MA2 全头因果普查）
- 全头普查完成（Gen1 一次成功 5527s，1152 头 x 80 词，~9.6 万前向）：C1=true（28/36 层指数主导 → exponential_dominant_full）、C2=true（top-64 头承载 51.6% 正载荷）、C3=false（份额-必要性 rho=0.1496<0.3，对齐谱⊥因果谱全尺度确认）。
- 重大新发现：因果前缘双峰结构——早中层簇（L13 h30 独大 10.12%、L2 h31/L5 h25/26/L4 h4）+ 中晚层簇（L22 h28/L23 h29 等）；此前 2824 起全部扫描聚焦 L20-35，L13 前缘首次曝光。修正"语义调制=中晚层现象"。
- 质量注记：逐层 clamp 残差 ≤0.0586，全局统计 0.5708 与逐层记录不一致（疑聚合口径差），2847 对质。
- SHA256：script 456f63b1 / exec 88bdcbae / result da6eda5e / census_full 75a43f9b + 36 分层 checkpoint 全登记。
- 接续 2847：L13 h30 解剖（为何对齐谱隐形因果谱独大）+ 早中层簇联合钳制 + 残差对质。
'''
with open(log, 'a', encoding='utf-8') as f:
    f.write(entry)

n = len(open(memo, encoding='utf-8').readlines())
ok = 'Phase 2846' in open(memo, encoding='utf-8').read()
ok2 = 'Phase 2846' in open(log, encoding='utf-8').read()
print('memo lines %d, 2846 in memo %s, 2846 in log %s' % (n, ok, ok2))
