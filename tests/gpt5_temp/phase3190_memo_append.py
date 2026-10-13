# -*- coding: utf-8 -*-
"""Phase 3190 MEMO append (three-pass verdict + honest log + 3191 prereg)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

BLOCK = """

## Phase 3190 — G5-C3 共享崩塌方向内容归因与稳健性（C 线，零 GPU，2026-10-10）

**预注册**：MEMO 3189 尾（观测前冻结）。3189 判定谱外崩塌由跨类共享方向承载（top1≈0.30、cross_med 0.44–0.57）后，检验该方向的类承载稳健性并做 report-only 内容归因。

**协议（freeze bfe79091）**：残差链 = 3189 verbatim；设备锚：3189 封存标量（top1/top2/cross_med/within/r80/r90）drift == 0 逐位 ×3、3169 逐 seed E 复现 0 drift ×3——残差链与 3189 完全同一。u* = 876 OOV 行方向矩阵的 top-1 奇异方向（特征空间）。门（freeze 钉定）：ROBUST iff LOO 4 折（剔单类 219 行重算 top1）max |Δ| < 0.05 ×≥2/3。(b) 预注册降级已钉定：embedding 差分需权重加载超零 GPU 预算 → cos(u*, v_c)（v_c = 3189 类均值方向）。

**观测（主臂）**：
| 模型 | LOO top1（4 折） | max Δ | 门 | KS(u*; OOV vs seen) | cos(u*, v_c) |
|---|---|---|---|---|---|
| qwen3-4b | 0.3246/0.3408/0.3118/0.3217 | 0.0342 | PASS | 0.9932 | -0.806/-0.762/-0.836/-0.819 |
| qwen3-14b | 0.3136/0.3147/0.2937/0.3149 | 0.0214 | PASS | 0.9761 | -0.807/-0.797/-0.852/-0.808 |
| glm4-9b | 0.3181/0.3305/0.3140/0.3340 | 0.0332 | PASS | 0.9715 | -0.778/-0.746/-0.802/-0.724 |

**verdict（重复一）**：LOO 稳健 3/3——剔除任一 OOV 类后 top1 最大波动 0.0214–0.0342，全部 <0.05 门：共享崩塌方向非单类承载，3189 的跨类共享读法稳健。ROBUST 分支执行：FTR-25 anchors append（role=loo_robustness）+ statement append，E 级维持 E1_repeatable（稳定性≠预测性，held_out 升级留专用臂）。

**verdict（重复二）**：内容归因（report-only）——(a) u* 对 seen 面板基线残差方向的投影 KS = 0.9932/0.9761/0.9715：共享崩塌方向在 seen 基线残差中**基本不存在**（OOV 特异几何），与 GAP-4 谱外崩塌 2.54× 一致；(b) |cos(u*, v_c)| = 0.72–0.85（符号对奇异向量任意）且四类均高：u* 与全部四个类均值方向强对齐——共享方向是类级崩塌几何的**公共核**，类内残差绕该核的弥散解释了 top1 仅 ≈0.30。

**verdict（重复三）**：C 线三连（3188/3189/3190）闭环 GAP-4 残留机制图景：残留 1.8× = 独立于关系偏移 Δv（3188 解耦）的跨类共享崩塌方向承载（3189），该方向非单类伪影且为四类公共核、在 seen 基线中缺席（3190）。FTR-25 降级 E2 需 held_out 专用臂（预测而非稳定）。

**诚实登记**：SMOKE-1 crash + 修正——初写 `ustar = U[:,0]`（样本空间 876 维向量），`Qn @ ustar` 维度 crash（876 vs 2560）：(a)(b) 两读数要求 u* 落在特征（残差）空间，应为 `Vt[0]`（2560 维，与 v_c 可比）。修正后 SMOKE 通过；FULL 与 SMOKE arm 科学字段逐位一致（diff 仅 mode/res_sha8/seal_sha8），零观测泄漏。观测性读数（top1/cross_med 等）在修正前后不受影响（能量谱 U/Vt 共享奇异值）。

**登记**：registry v1.16→**v1.17 `73923c83`**（FTR-25 anchors + statement append-only，25 特征字节保全）；atlas_v1_18.html `580af247`（746 字段 0/0/0/0）；ledger #342（f6584f85→5727e791）；arm res `24461856`/seal `0c975f5b`；final res `fc8be7db`/seal `f7d8571a`；design `bfe79091`。图谱现状：25 特征（E2×17/E3×3/E1×4，FTR-25 候选带 LOO 稳健锚）、failures F1–F19、upgrade_log 6、ledger n=342。

**3191 预注册（G5-C4，零 GPU，封存复用）**——FTR-25 held_out 升格臂（E1→E2 通道）：用 3 类 OOV 行方向的均值方向（train 折）预测留出类的逐行方向份额：对每模型 4 折，v̂_held = 其余 3 类类均值方向 unit 化，计算留出类 219 行方向与 v̂_held 的 cos 分布（中位 |cos|）；门（freeze 前细化）：held-out 类行方向中位 |cos(v̂_held)| ≥ 0.3 ×≥2/3 模型 ⇒ 预测性成立，FTR-25 升 E2_held_out（registry v1.17→v1.18）；未过 ⇒ E1 维持 + counter_evidence。设备锚：3189/3190 封存标量 drift == 0 逐位；report-only：cos 分布 vs 3189 类内 cos 分布的 KS。挂账提醒：5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(BLOCK)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO appended: pre %s post %s' % (pre, post))
