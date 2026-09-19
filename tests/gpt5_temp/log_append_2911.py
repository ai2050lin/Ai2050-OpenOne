# -*- coding: utf-8 -*-
"""Append Phase 2911 closure note to workspace daily log."""
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')

NOTE = """
## Phase 2911 闭环（08:36-08:42）

- **Phase 2911（交替结构形式检验）闭环完成**：execution 0ab48eaf / result 61a08678 / npz 8c791cbd / 脚本 7793c93a；6s 零前向。锚 4/4（含 a3 跨 Phase 互锚 2910 score_true 5e-7）、置换校准 pass（0.88/0.505）。
- 判决 **alternation_not_confirmed_margin_only**：P1 margin 交替真实且 qwen_attn 特有（8/9 flips p=0.0195，他组 p>=0.25）；P3 列相关无振荡（osc +0.050 正向，S 空集）；P2 delta 符号随机（5/9）。
- **载体定位（diag_2911c）**：类间符号平衡 gap_j = |pos_frac0 − pos_frac1| 锯齿 7/8，与 margin_j 强对应（gap 大→margin 正）；主要由类 0 正率波动驱动（pf0 0.32-0.86 宽幅 vs pf1 0.37-0.60）；gap→margin 定律跨组成立（glm4_attn L08/L09）。
- 附带：glm4_attn delta 全同号（0/11 flips，反向尾 p~0.0005）。
- 教训：diag_2911b 指标设计错误（|delta| 符号检验无信息；归一化定义与目标泛函不同构）→ diag_2911c 修正；诊断指标必须与目标量定义同构。
- Ledger：M2911 + L14 精化（connects 18），measurements 50，ledger SHA c8a0d016。MEMO 2911 节 @7389 行。
- **2912 候选**：A（主选）符号平衡锯齿正式化 + 词级归因（词级符号置换 null + pf0 驱动词识别，词 x 层符号矩阵，零前向）；B 头级 W_VO；C 前向 SwiGLU。
"""

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(NOTE)
tail = open(LOG, encoding='utf-8').read()
print('OK appended, contains 2911:', 'Phase 2911 闭环' in tail)
