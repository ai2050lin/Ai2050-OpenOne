# -*- coding: utf-8 -*-
"""Phase 3195 MEMO append: G5-D3 section appended to AGI_GPT5_MEMO.md."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
assert pre == 'e6a4813f', 'MEMO pre sha mismatch: %s' % pre

SEC = """

## Phase 3195 — G5-D3 u* 内容归因深化（零 GPU，封存复用，report-only）【2026-10-10】

**预注册钉定（先于观测）**：候选张成集 C = 22 方向 {dvu, v_s×6, u_seen, v_c×4, KOUT top-10 主成分（mean-centered 原始 876 行 KOUT 矩阵）}；度量 = 残差比 ‖u*−P_C u*‖/‖u*‖（QR 正交化）；门：<0.5 ×3/3 ⇒ PARTIALLY_DECOMPOSED（归因注记入 ledger+MEMO，registry v1.20 文件字节不动）；否则 NOVEL_CORE 维持。

**结果**：门 **3/3 全过** ⇒ PARTIALLY_DECOMPOSED：
- 残差比 0.0181 / 0.0174 / 0.0181（投影范数 0.9998 ×3）——u* 几乎完全落在候选张成空间内。
- 主载体：**v_c×4（OOV 类均值方向）**，|cos| 0.724–0.852 跨模型（与 3190 report-only 逐位一致）；次载体 KOUT pc0/pc2（0.38–0.43）；dvu/v_s/u_seen 单方向贡献仍低（3193 NOVEL 读数对照锚逐位复现：gate_max 0.2130/0.3303/0.2457、|cos(u*,dvu)| 0.1064/0.0405/0.0240）。

**理论意义（D 线三连诠释闭环）**：3193 NOVEL 与 3195 可分解**并立不矛盾**——NOVEL 是"限制门集（dvu/v_s/u_seen）内无单方向 ≥0.3"；3195 加入 v_c 后子空间级近完全张成。u* 不是新基元方向，而是 **OOV 残差几何的类均值主导混合**：类级崩塌公共核（3190 内容归因）在方向代数上的体现——FTR-26 的 held-out 可预测性（3194）由类均值结构承载。图谱修正：FTR-26 定位从"未知新成分"收敛为"可归约混合体（载体谱 = v_c×4 ≫ kout_pc > 其余）"，E 级维持 E2_held_out。

**设备锚**：drift69/89/90/91/93/94 全 ==0 逐位 ×3，dv_drift==0——残差链 3189→3195 七连逐位同一。

**产物**：report-only——registry v1.20 字节不动（回读 `2aac7e5e`）、无 atlas 新版；design `c23f7ada`，res `21aee353`/seal `f9fe19e0`，final `b527537a`/`551e8b40`，ledger #347（post `11ce7697`）。

**诚实日志**：SMOKE-1 候选数断言 17→22（预注册算术 slip：1+6+1+4+10=22；DESIGN 重 freeze c23f7ada）；SMOKE/FULL 载荷逐位一致。

**3196 预注册（G5-D4，零 GPU，封存复用）**——u* 非负组成份额臂：对每模型，在 22 方向候选集上对 u* 做**非负最小二乘（NNLS，scipy.optimize.nnls）**，报告归一化份额谱（份额/Σ份额）；门（freeze 前细化）：3/3 模型 v_c×4 合计份额 ≥0.8 且 dvu+v_s×6+u_seen 合计份额 <0.1 ⇒ CLASS_MEAN_DOMINATED 确证（report-only，registry v1.20 仍不动）；否则 MIXED 登记谱。设备锚：3189–3195 封存标量 drift == 0 逐位；对照锚：3195 残差比（0.0181/0.0174/0.0181）同协议复算 drift==0。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(SEC)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO: %s -> %s (+%d chars)' % (pre, post, len(SEC)))
