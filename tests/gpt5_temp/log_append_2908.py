# -*- coding: utf-8 -*-
"""Append Phase 2908 closure note to workspace daily log."""
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')

NOTE = """
## Phase 2908 闭环（07:52-07:56）

- **Phase 2908（qwen_attn M1 边界高精度判定）闭环完成**：execution 4a3587e3 / result dff824f8 / npz 553ab55b / 脚本 9f034700；38s 零前向（fast_margin 预计算 mask 提速 ~10x）。
- 判决 **qwen_attn_at_m1_edge**：20000 draws x 5 seed + 覆盖率审计 40/40；锚 4/4（fast==full 精确 0）。
- qwen_attn p_median=0.0276，seed 范围 [0.0262, 0.0295]：**5/5 seed 名义 inside（below 被高精度否定）；3SE 保守带内记 at_m1_edge**（距 confirmed-inside 阈值 0.0007，<1 SE）。
- 其他组：qwen mlp p=0.474 / glm4 mlp 0.375 / glm4 attn 0.168——glm4 attn firmly inside，**2906 通道分裂问题正式关闭**。
- **谱系新轴**：null 内百分位 p 排序（0.474 > 0.375 > 0.168 > 0.028）独立于 margin 幅值排序；qwen attn 唯一双轴皆末通道。
- Ledger：M2908 + L14 精化（connects 15），measurements 47，ledger SHA 0f413c96。MEMO 2908 节 @7252 行（title_ok 复核）。
- **2909 候选**：A（主选）p 轴谱系稳定性预注册检验（跨 margin 家族变体/seed 域/层子集，零前向）；B 头级 per-head W_VO 分解；C 前向 SwiGLU 激活级。
"""

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(NOTE)
tail = open(LOG, encoding='utf-8').read()
print('OK appended, contains 2908:', 'Phase 2908 闭环' in tail)
