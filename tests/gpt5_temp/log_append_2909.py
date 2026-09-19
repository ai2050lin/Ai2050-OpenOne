# -*- coding: utf-8 -*-
"""Append Phase 2909 closure note to workspace daily log."""
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')

NOTE = """
## Phase 2909 闭环（08:04-08:10）

- **Phase 2909（p 轴谱系稳定性检验）闭环完成，负结果**：execution 8c6bf1a3 / result 2fb42fa1 / npz 6dafc809 / 脚本 a9f4d1a4；92s 零前向。实现期 3 个 bug（n1 未定义 / d=1 std float 转换 / SCORES 查表）均清目录重跑处理。
- 判决 **p_axis_order_unstable**：锚 4/4、审计 7/7（V2 38/40、V3 39/40 其余 40/40）；28 配置网格（4 组 x [V0-V3 变体 + S_front/S_back/S_key 子集]，5 seed x 10000 draws）。
- **p 轴升格否决（N12 入账）**：排序仅 cosine-mean margin 族内不变（V1 保持；V2_colz/V3_acc/全部子集打破，qwen_attn 0.026 -> 0.097/0.735/0.13-0.61）；qwen_attn tail fragile、glm4_attn interior robust（唯一不变分量）。
- **幸存结构发现：qwen_attn 下尾是全层聚合效应**——S_front 0.140 / S_back 0.130 / S_key 0.607 各自 null 样，仅全层 cosine 平均聚合出 p=0.026；margin 负值非单层属性而是跨层一致微移的聚合。
- Ledger：M2909 + N12（negatives 12）+ L14 精化（connects 16），measurements 48，ledger SHA a9c1963f。MEMO 2909 节 @7296 行。
- **2910 候选**：A（主选）qwen_attn 全层聚合效应分解（逐层 margin/delta 符号相干性检验，零前向）；B 头级 per-head W_VO；C 前向 SwiGLU。
"""

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(NOTE)
tail = open(LOG, encoding='utf-8').read()
print('OK appended, contains 2909:', 'Phase 2909 闭环' in tail)
