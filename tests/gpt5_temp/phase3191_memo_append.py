# -*- coding: utf-8 -*-
"""Phase 3191 MEMO append (three-pass verdict + honest log + 3192 prereg)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

BLOCK = """

## Phase 3191 — G5-C4 FTR-25 held_out 升格臂（C 线，零 GPU，2026-10-10）

**预注册**：MEMO 3190 尾（观测前冻结）。3190 判定共享崩塌方向 LOO 稳健后，走 E1→E2 通道：检验「其余 3 类均值方向可预测留出类行方向」的预测性主张。

**协议（freeze c4673cc3）**：残差链 = 3190 verbatim；设备锚：3189 封存标量 drift == 0 逐位、3190 封存标量（top1/cross_med/within/loo_max_delta/loo_top1）drift == 0 逐位、3169 逐 seed E 复现 0 drift——残差链跨 3189/3190/3191 完全同一。门（freeze 钉定）：HOLDOUT_PREDICTIVE iff 留出类 219 行方向对其余 3 类均值方向的中位 |cos| ≥ 0.3 **全 4 折** ×≥2/3 模型。

**观测（主臂）**：
| 模型 | med\\|cos_held\\|（4 折） | min | 门 | KS(\\|cos\\|; 类内) |
|---|---|---|---|---|
| qwen3-4b | 0.4432/0.3906/0.4931/0.4547 | 0.3906 | PASS | 0.22/0.18/0.27/0.19 |
| qwen3-14b | 0.4405/0.4174/0.5001/0.4296 | 0.4174 | PASS | 0.17/0.17/0.18/0.19 |
| glm4-9b | 0.4233/0.3717/0.4330/0.3807 | 0.3717 | PASS | 0.47/0.33/0.30/0.43 |

**verdict（重复一）**：held_out 预测性 3/3 全过——任一 OOV 类的行方向都可由其余 3 类均值方向以中位 |cos| ≥ 0.37 预测：共享崩塌方向是**跨模型留出可预测**的几何，非训练折内伪影。

**verdict（重复二）**：E1_repeatable → **E2_held_out**（第 18 个 E2，升级链 #7）。report-only KS(|cos_held| vs 类内 pairwise cos) = 0.17–0.47：留出类行方向比类内余弦更弥散——与 top1≈0.30 共享核 + 类内弥散的 3190 图景一致（方向可预测、幅度不对齐）。

**verdict（重复三）**：C 线四连（3188–3191）完整闭环 GAP-4 残留机制链：Δv 解耦（3188）→ 跨类共享方向（3189）→ 非单类承载 + 公共核 + seen 缺席（3190）→ **跨模型 held-out 可预测（3191，E2）**。GAP-4 残留 1.8× 的承载几何已从结构线索升级为可预测证据；下一步只剩幅度级归因（为何共享方向在谱外增益 1.8× 而面板内不显）。

**诚实登记**：SMOKE-1——CONTENT_3190 锚误设为 final sha（fc8be7db/f7d8571a），p3190 result.json 内嵌实为 arm 内容锚（24461856/0c975f5b），「内容 sha ≠ 文件 sha」教训再次确认；FULL-1 crash——FTR-25 E 级字段名 legacy 口径为 `evidence_level`（非 `e_level`），崩在 final 写盘前，arm res/seal 跨次重跑逐位一致（836fe933/a66c0b3d），registry v1.18 重写 c08cbcc2，零观测泄漏。SMOKE/FULL arm payload 逐位一致（diff 仅 mode/res_sha8/seal_sha8）。

**登记**：registry v1.17→**v1.18 `c08cbcc2`**（FTR-25 evidence_level E1_repeatable→E2_held_out + anchors append role=held_out + statement append + upgrade_log #7，append-only 全断言）；atlas_v1_19.html `d5157665`（754 字段 0/0/0/0）；ledger #343（5727e791→本 post）；design `c4673cc3`；arm res `836fe933`/seal `a66c0b3d`；final res `28a852b0`/seal `6eb89727`。图谱现状：25 特征（**E2×18/E3×3/E1×3**）、failures F1–F19、升级链 7、ledger n=343。

**3192 预注册（G5-C5，零 GPU，封存复用）**——GAP-4 幅度级归因：共享崩塌方向 u* 在谱外的**增益幅度**归因（为何 OOV 行方向在 u* 上投影幅度大于 seen 面板行方向）：对每模型，计算 (a) OOV 876 行与 seen 面板逐 seed 测试行在 u* 上的投影**幅度分布**（|proj|，非 KS 的分布形状而是尺度）；(b) 幅度比 median(|proj_oov|)/median(|proj_seen|) 与 3169 ratio_B 2.5388 的对账（共享方向承载多大比例的 1.8× 残留）；(c) 门（freeze 前细化）：幅度比 ≥1.3 ×≥2/3 模型 ⇒ 共享方向是谱外崩塌幅度的主要载体（u*_amplitude_carrier 入 GAP-4 mechanism_note append）；否则幅度弥散在正交补（counter_evidence，方向对但幅度不在 u*）。设备锚：3189/3190/3191 封存标量 drift == 0 逐位。挂账不变：5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(BLOCK)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO appended: pre %s post %s' % (pre, post))
