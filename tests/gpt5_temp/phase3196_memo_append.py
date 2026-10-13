# -*- coding: utf-8 -*-
"""Phase 3196 MEMO append: G5-D4 section appended to AGI_GPT5_MEMO.md."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
assert pre == 'fd899a1d', 'MEMO pre sha mismatch: %s' % pre

SEC = """

## Phase 3196 — G5-D4 u* 非负组成份额臂（NNLS，零 GPU，封存复用，report-only）【2026-10-10】

**预注册钉定（先于观测）**：22 方向候选集 {dvu, v_s×6, u_seen, v_c×4, KOUT top-10} 上对 u* 做 NNLS（scipy.optimize.nnls），份额 = w/Σw；门：v_c×4 合计 ≥0.8 且 known-struct（dvu+v_s×6+u_seen）合计 <0.1 ×3/3 ⇒ CLASS_MEAN_DOMINATED（report-only，registry v1.20 字节不动）；否则 MIXED 登记谱。对照锚：3195 QR 投影残差比（0.01806072230473137/0.017358575654316224/0.01814552246882951）复算 drift==0。

**结果**：门 **0/3 ⇒ MIXED**：
- vc_share = **0.0000 ×3**（NNLS 对 v_c 零权重）；known_share 0.1999/0.3998/0.1139。
- NNLS 份额谱由 **kout_pc 主导**（0.216–0.343）+ u_seen（0.074–0.343）+ 小份额 dvu/v_s。

**关键负结果（D 线第四连）**：**nnls_resid 0.8184/0.8125/0.8236 ≫ QR 投影残差比 0.018**——u* 虽几乎完全落在候选张成空间内（3195 proj 0.9998），但候选集的**非负锥中完全不存在 u* 的表示**；cos 级 v_c 主导（0.72–0.85）只能经**负系数相消**成立。诠释：u* 是**带符号对比混合方向**——与其作为 OOV 特异残差方向的来源一致（必含「减去 seen 基线」负分量）；**无符号约束的 cos 级读数会夸大类均值构成**。登记 caveat：共线候选上 NNLS 非唯一，但 resid 0.82 非非唯一性伪影（任何非负组合都拟合不了）。

**设备锚**：drift69/89/90/91/93/94 全 ==0 逐位 ×3、dv_drift==0、3195 比值对照锚 drift==0——残差链 3189→3196 八连逐位同一。

**产物**：report-only——registry v1.20 字节不动（回读 `2aac7e5e`）、无 atlas 新版、FTR-26 维持 E2_held_out；design `6723c371`，res `e69ab1e7`/seal `c0cdd97f`，final `7313de4d`/`f57cc8b7`，ledger #348（post `86f3a444`）。

**诚实日志**：SMOKE-1 leftover return（decomposed→dominated，机械性）；SMOKE/FULL 载荷逐位一致。

**3197 预注册（G5-D5，零 GPU，封存复用，D 线收官检查臂）**——u* 带符号组成份额臂：对每模型，在 22 方向候选集上做**无约束最小二乘（岭稳定 λ=1e-12）**，报告**带符号**权重谱（含负系数）；门（freeze 前细化）：3/3 模型 LS 拟合残差 < 0.05 且 v_c×4 合计带符号权重（取绝对值归一）≥0.5 ⇒ SIGNED_CLASS_CONTRAST 确证（D 线定性收官：u* = 带符号类对比混合，registry v1.20 仍不动）；否则 SIGNED_MIXED 登记谱。对照锚：3195 残差比 + 3196 NNLS resid（0.8184/0.8125/0.8236）复算 drift==0；设备锚 3189–3196 封存标量 drift==0。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(SEC)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO: %s -> %s (+%d chars)' % (pre, post, len(SEC)))
