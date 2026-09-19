# -*- coding: utf-8 -*-
"""Append Phase 2910 closure note to workspace daily log."""
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')

NOTE = """
## Phase 2910 闭环（08:12-08:17）

- **Phase 2910（qwen_attn 全层聚合效应分解）闭环完成**：execution d4c7ed55 / result 48001e65 / npz 1013f7da / 脚本 0a3066f9；85s 零前向。锚 4/4、审计 d=1 39/40 + d=5 40/40。
- 判决 **qwen_attn_partial_coherence**（below_frac 0.50，冻结中段）。
- **决定性发现 1：相邻层交替**——L00-L05 严格偏下/偏上交替（p: 0.327/0.778/0.104/0.697/0.253/0.849），L07/L09 恢复偏下。
- **决定性发现 2：累积曲线后半层主导单调入深尾**——k=1..4 震荡 0.34-0.54，k=5 0.139 -> k=8 0.010 -> k=10 0.025 稳定；**三方互证**：cum k=10=0.025 复现 2908 全层 p=0.026、cum k=5=0.139 复现 2909 S_front=0.140（独立 rng 键同值到千分位）。
- 排除两个朴素假说：uniform coherence（0.90 阈值未达）与 averaging artifact（累积单调非震荡）。
- 对照形态学：glm4_attn 早降内部型 / qwen_mlp 单层主导型（L01 score 1.0195）/ glm4_mlp 尾部逆转型 / qwen_attn 深尾聚合型——四通道四种聚合形态。
- Ledger：M2910 + L14 精化（connects 17），measurements 49，ledger SHA 2acb5381。MEMO 2910 节 @7344 行。
- **2911 候选**：A（主选）交替结构形式检验（逐层 delta 符号交替 + 相邻/隔层 B 列相关 oscillation index，零前向）；B 头级 W_VO 分解；C 前向 SwiGLU。
"""

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(NOTE)
tail = open(LOG, encoding='utf-8').read()
print('OK appended, contains 2910:', 'Phase 2910 闭环' in tail)
