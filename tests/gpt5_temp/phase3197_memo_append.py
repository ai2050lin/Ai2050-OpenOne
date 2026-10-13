# -*- coding: utf-8 -*-
"""Phase 3197 MEMO append: G5-D5 section appended to AGI_GPT5_MEMO.md."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
assert pre == 'bf1f926a', 'MEMO pre sha mismatch: %s' % pre

SEC = """

## Phase 3197 — G5-D5 u* 带符号组成份额臂（D 线收官检查，零 GPU，report-only）【2026-10-10】

**预注册钉定（先于观测）**：22 方向候选集上对 u* 做无约束岭稳定 LS（λ=1e-12），报告带符号权重谱；门：LS 拟合残差 <0.05 且 |v_c×4| l1 份额 ≥0.5 ×3/3 ⇒ SIGNED_CLASS_CONTRAST（D 线定性收官，registry v1.20 字节不动）；否则 SIGNED_MIXED。对照锚：3195 QR 残差比 + 3196 nnls_resid（4dp 封存标量）复算 drift==0。

**结果**：门 **3/3 全过 ⇒ SIGNED_CLASS_CONTRAST，D 线定性收官**：
- ls_resid **0.0181 / 0.0174 / 0.0181**（≈3195 QR 残差比——带符号 LS 近完全拟合 u*）。
- **v_c×4 权重全负**：4b −0.336/−0.320/−0.303/−0.273（v_c[8]/[6]/[9]/[7]）、14b −0.355/−0.289/−0.281/−0.276、glm4 −0.361/−0.320/−0.314/−0.302；vc_l1_share **0.9159/0.8905/0.9244**；次大 |w| ≤0.024；n_neg 18/13/13（of 22）。

**定谳**：**u* ≈ −(w₈v_c[8]+w₆v_c[6]+w₉v_c[9]+w₇v_c[7])——「同时偏离全部 OOV 类均值」的反类均值对比轴（anti-class-mean contrast axis）**。与来源一致：u* = OOV 特异残差行方向的 SVD top-1，必然指向所有类均值之外。

**D 线五连闭环（3193→3197）**：
1. 3193 NOVEL——限制门集（dvu/v_s/u_seen）内单方向级不可归约；
2. 3194 E2_held_out——跨类留出可预测（第 19 个 E2，升级链 #8）；
3. 3195 PARTIALLY_DECOMPOSED——子空间级可归约（v_c 加入后 proj 0.9998）；
4. 3196 非负锥 FAIL——nnls_resid 0.82 ≫ ratio 0.018（cos 级读数需负号相消）；
5. 3197 带符号定谳——负类均值和、反类均值对比轴。
方法学教训沉淀：**共线方向集上，子空间投影（3195）、非负份额（3196）、带符号份额（3197）三层读数必须分开报告——无符号 cos 级读数会系统性夸大类均值构成**。

**设备锚**：drift69/89/90/91/93/94 全 ==0 逐位 ×3、dv_drift==0、3195/3196 对照锚 drift==0——残差链 3189→3197 九连逐位同一。

**产物**：report-only——registry v1.20 字节不动（回读 `2aac7e5e`）、FTR-26 维持 E2_held_out；design `c3a64857`，res `dd2af224`/seal `4311d796`，final `f8a39985`/`2d361f0d`，ledger #349（post `694bfcbf`）。

**诚实日志**：SMOKE 前 sha 锚占位 f086e6a3 → 实测 f3f86f47（p3196_result，实测探针先于 SMOKE，无观测污染）；SMOKE/FULL 载荷逐位一致。

**3198 预注册（G5-D6，零 GPU，D 线收官入册）**——D 线五连结论入册臂：FTR-26 statement append（SIGNED_CLASS_CONTRAST 定谳 + 3193–3197 证据链摘要 + anchors append role=signed_closing），registry v1.20→v1.21（append-only，25 legacy 特征字节保全）+ atlas_v1_23.html；门（freeze 前细化）：statement append 成功 + 771→N 字段校验 0 mismatch + legacy 字节保全断言 ⇒ REGISTERED（registry v1.21 `sha8 实测`）；设备锚：3189–3197 封存标量 drift==0；对照锚：3197 signed_weights（v_c 权重 4dp）复算 drift==0。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(SEC)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO: %s -> %s (+%d chars)' % (pre, post, len(SEC)))
