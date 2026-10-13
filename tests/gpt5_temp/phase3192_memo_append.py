# -*- coding: utf-8 -*-
"""Phase 3192 MEMO append (three-pass verdict + C-line close + 3193 prereg)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

BLOCK = """

## Phase 3192 — G5-C5 GAP-4 幅度级归因（C 线，零 GPU，2026-10-10）

**预注册**：MEMO 3191 尾（观测前冻结）。3191 升格 E2 后，最后一问：谱外幅度增益是否由共享方向 u* 承载。

**协议（freeze efc062be）**：残差链 = 3191 verbatim；设备锚：3189/3190/3191 封存标量 drift == 0 逐位 ×3、3169 逐 seed E 复现 0 drift——残差链跨 3189–3192 完全同一。幅度口径：**未归一化** 3-seed 均值 OOV 残差行（876）与 seen 面板逐 (row,seed) 残差在 u* 上的 |proj| 中位比（保留幅度信息，unit 化只剩方向）。门（freeze 钉定）：USTAR_AMPLITUDE_CARRIER iff amp_ratio ≥ 1.3 ×≥2/3。

**观测（主臂）**：
| 模型 | amp_ratio | med\\|proj\\| OOV vs seen | dir_ratio（unit 口径） | KS(amp) |
|---|---|---|---|---|
| qwen3-4b | 10.5984 | 8.97e1 vs 8.46e0 | 6.2337 | 0.9977 |
| qwen3-14b | 7.1100 | 2.53e2 vs 3.55e1 | 4.7547 | 0.9477 |
| glm4-9b | 8.8909 | 5.00e1 vs 5.63e0 | 5.3813 | 0.9715 |

**verdict（重复一）**：USTAR_AMPLITUDE_CARRIER 3/3——amp_ratio 7.11–10.60 ≫ 门 1.3：谱外崩塌幅度**主要承载在共享方向 u* 上**，非正交补弥散。CARRIER 分支执行：GAP-4 mechanism_note append（u*_amplitude_carrier）+ evidence append，gap ledger v1.5→v1.6（append-only），registry v1.18 不动（无特征级变化）。

**verdict（重复二）**：对账 3169 ratio_B 2.5388——amp_ratio 是 ratio_B 的 3–4 倍（超比例载体）：OOV 行方向在 u* 上的幅度集中度**高于**整体崩塌比。dir_ratio（unit 方向口径）4.75–6.23 与 KS(amp) 0.95–1.00 一致：方向与幅度双集中在 u*——「共享核 + 类内弥散」（3190/3191）中的弥散部分在 u* 正交补内是**次级**的。

**verdict（重复三）**：**C 线五连（3188–3192）闭环，GAP-4 机制链收官**：残留 1.8× / 崩塌 2.54× = 共享方向 u* 以超比例幅度承载（3192），该方向跨类共享（3189）、非单类公共核（3190）、跨模型 held-out 可预测（3191，E2_held_out）、独立于关系偏移 Δv（3188）。机制陈述：谱外 token 触发的崩塌能量主要泄入单一共享方向，该方向可由类级几何预测。

**诚实登记**：SMOKE/FULL arm payload 逐位一致（diff 仅 mode/res_sha8/seal_sha8），本轮零 crash 零补丁；GAP-4 append 断言（statement/prereg/status/anchor_sha8/evidence_anchors 字节保全 + mechanism_note/evidence 前缀）全过。

**登记**：gap ledger v1.5→**v1.6 `0f6423e0`**（GAP-4 mechanism_note + evidence append-only，其余 gaps/appendix 字节保全）；registry v1.18 `c08cbcc2` **不变**；atlas_v1_20.html `88300285`（754 字段 0/0/0/0，渲染 registry v1.18 + gap v1.6）；ledger #344（4ccecd1b→本 post）；design `efc062be`；arm res `6221fe00`/seal `58c4f183`；final res `7f35fca4`/seal `b147d3d2`。图谱现状：25 特征（E2×18/E3×3/E1×3）、failures F1–F19、升级链 7、gap v1.6、ledger n=344。

**3193 预注册（G5-D1，零 GPU，封存复用）**——C 线收官后开 D 线：**u* 的子空间成分分解**（共享崩塌方向由什么构成）：对每模型，将 u* 对 3169 已封存结构方向投影分解——(a) 3165 跨族可分方向族、(b) 3166 逻辑对比方向、(c) 3157 变换代数 T_C 方向、(d) 10 类 seen 类均值方向（B 协议残差类均值）：报告 |cos| 谱与累计份额（top-k 结构方向解释 u* 的比例）；门（freeze 前细化）：若任一已知结构方向 |cos| ≥ 0.5 ×≥2/3 ⇒ u* 可归约为已知机制成分（u*_decomposed，GAP-4 mechanism_note 二次 append / gap v1.7）；若 max |cos| < 0.3 ×≥2/3 ⇒ u* 是**新成分**（u*_novel_direction 入 FTR 候选 FTR-26 立项，registry v1.19）；中间带 ⇒ 混合，report-only。设备锚：3189–3192 封存标量 drift == 0 逐位。挂账不变：5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(BLOCK)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO appended: pre %s post %s' % (pre, post))
