# -*- coding: utf-8 -*-
"""Append Phase 2907 closure note to workspace daily log."""
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')

NOTE = """
## Phase 2907 闭环（07:05-07:10）

- **Phase 2907（attn 形状修正阶梯）闭环完成**：产物 SHA 登记（execution bd3d5322 / result 390f27c9 / npz 9bd10e2d / 脚本 87376b53）。
- **关键发现：四组全部 level=M1**——RMS sigma 定义下各向同性 summary 还原全部四通道，形状修正不必要；qwen attn 富余仅 0.0004（MC borderline）。
- **判决映射缺口**：冻结映射未枚举 both-M1，机械输出 shape_correction_mixed 与实际不符；保留冻结标签，M2907 verdict 注明。
- **diag_2907b 2x2x2 归因网格**（sigma定义 x seed x 流结构，双锚复现 9e-7）：glm4 attn 判定 100% 由 sigma 定义驱动；qwen attn MC 噪声级 borderline。
- **2906 勘误 E9 入账**：2906 prereg 文本冻结 RMS 但实现用 mean-std（drift 自洽通过共享定义的审计）；2906 attn-below 读法不稳健， 幅值事实撤回。
- Ledger：M2907 + E9 + L14 精化（connects 14），measurements 46 / errata 9，ledger SHA a25a7122。
- MEMO 追加 Phase 2907 节（7201 行，title_ok 复核）。
- **2908 候选**：A（主选）qwen_attn M1 边界判定加密（400->10000 抽样 + 多 seed，零前向秒级）；B 头级 W_VO 分解；C 前向 SwiGLU 激活级归因。
"""

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(NOTE)
tail = open(LOG, encoding='utf-8').read()
print('OK appended, contains 2907:', 'Phase 2907 闭环' in tail)
