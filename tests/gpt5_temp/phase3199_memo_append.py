# -*- coding: utf-8 -*-
"""Phase 3199 MEMO append (assert prefix)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
old = io.open(P, encoding='utf-8').read()
pre_sha = hashlib.sha256(old.encode('utf-8')).hexdigest()[:8]

ADD = '''

## Phase 3199 — G5-E1 FTR-24 held_out 升格臂（3194 同协议，零 GPU，封存复用）【2026-10-10】

**预注册钉定（先于观测）**：FTR-24（Δv 关系偏移方向，E1_repeatable）held_out 升格臂——每模型 4 折留出一个 OOV 类（219 行），P1 = 封存方向 dvu = unit(v_isa−v_hasa)（3183/3157 verbatim，label-free）；门（freeze 前钉定）：median |cos(行, dvu)| ≥0.3 全 4 折 ×**3/3** 模型 ⇒ E1→E2_held_out（evidence_level 变更 + anchors append role=held_out + statement append + upgrade_log #9），registry v1.21→v1.22 + atlas_v1_24.html；否则 counter_evidence append、E 级维持。诚实修正：预注册对象初稿 FTR-15，回查索引发现其已于 3180 升 E2（16-fold LOO PASS），freeze 前改钉 FTR-24。设备锚：3169 per-seed E + 3157 tc + 3183 dv 标量 + 3189/3190/3191/3193 封存标量 drift==0 逐位；封存 3197 result 文件锁 + signed_weights/ls_resid 4dp 回读 drift==0 ×3（ANCH_3197 freeze 前钉定，承自 3198）；3195/3196 标量本轮不复算（22 方向 QR 机构不重跑，其共享 Qn 基底由 3169/3189/3190/3191 逐位锚覆盖）。

**结果**：门 **0/3 ⇒ NOT_PREDICTIVE（关键负结果，E 级维持 E1）**：
- med|cos_held, dvu| 4 折：4b 0.0404/0.0751/0.0957/0.0610（min **0.0404**）、14b 0.0947/0.0598/0.0698/0.0695（min **0.0598**）、glm4 0.0501/0.0714/0.0355/0.0588（min **0.0355**）——12/12 折远低于 0.3；全量 876 行 med|cos| 仅 0.0610/0.0734/0.0535。
- 对照：同协议 3191 v̂_held（类均值方向）min 0.3906/0.4174/0.3717——**类均值结构承载留出可预测性（FTR-25/26），关系方向 dvu 完全不承载**。

**诠释**：FTR-24 的低秩结构证据（3184 LOO-min pc ≥0.4 ×3）不受影响，但 Δv 是 **isa/hasa 面板内的关系几何**，对谱外 OOV 行方向**无跨类预测力**——与 3188 DECOUPLED（Δv×谱外崩塌幅度份额≈噪声）和 3189 类均值主导（共享崩塌方向=u* 类均值核）三方互证：**关系偏移与谱外可预测性是两个解耦的通道**。counter_evidence 第 4 条 append，upgrade_log 维持 8 条，failures F1–F19 不变，25 legacy 特征字节保全。

**设备锚**：drift69/89/90/91/93 全 ==0 逐位 ×3、dv_drift==0、tc drift <1e-9、3197 4dp 回读 drift==0 ×3。

**产物**：registry v1.21 `47cf01e0` → v1.22 `9fd87433`（append-only，counter_evidence 口径）；atlas_v1_24.html `74a11492`（**775/775 字段，0/0/0/0**）；design `cfbe183c`，res `94442e48`/seal `b908731e`，final `06644f1f`/`82009c4a`，ledger #351（post `cffec403`）。SMOKE/FULL 载荷逐位一致（零 GPU）。

**3200 预注册（G5-E2，零 GPU，封存复用）**——FTR-23 held_out 升格臂（3194 同协议）：FTR-23 = 关系特异残差方向（3177 发现/3178 归因确证，E1_repeatable，family=context）；对每模型 4-fold OOV 类留出，P1 = FTR-23 封存方向，报告留出行 |cos| 中位数；门（freeze 前细化）：median |cos_held| ≥0.3 全 4 折 ×3/3 模型 ⇒ E1→E2_held_out（registry v1.22→v1.23 + atlas_v1_25.html + upgrade_log #9）；否则 counter_evidence append、E 级不变。对照锚：FTR-23 方向恒等锚（3177/3178 封存标量复算 drift==0）+ 3191 v̂_held 逐位；设备锚：3189–3193 封存标量 drift==0 逐位 + 3197 4dp 回读。若 FTR-23 亦 NOT_PREDICTIVE ⇒ E1 残余（FTR-04 已闭/FTR-23/24 维持）转入「E1 特征的 held-out 不可预测性」作为图谱方法论定论（可证伪记录），G5-E 线收官转向 appendix 挂账清算。
'''
new = old + ADD
assert new.startswith(old)
io.open(P, 'w', encoding='utf-8', newline='').write(new)
back = io.open(P, encoding='utf-8').read()
assert back == new
post_sha = hashlib.sha256(back.encode('utf-8')).hexdigest()[:8]
print('MEMO appended: pre %s -> post %s (+%d chars)' % (pre_sha, post_sha, len(ADD)))
