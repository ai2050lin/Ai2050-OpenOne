# -*- coding: utf-8 -*-
"""Phase 3194 MEMO append: G5-D2 section appended to AGI_GPT5_MEMO.md."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
assert pre == '434a0de8', 'MEMO pre sha mismatch: %s' % pre

SEC = """

## Phase 3194 — G5-D2 FTR-26 held_out 升格臂（E1→E2 通道，零 GPU，封存复用）【2026-10-10】

**预注册钉定（先于观测）**：门 = 留出类行方向中位 |cos(row, u*)| ≥ 0.3 全 4 折 × ≥2/3 模型 ⇒ FTR-26 升 E2_held_out（registry v1.19→v1.20）；未过 ⇒ E1 维持 + counter_evidence。双预测子：P1 = u*（全 876 行 SVD top-1，label-free 非泄漏——u* 是方向汇总不是留出类标签）；P2 = v̂_held（3191 协议 verbatim 对照锚，med_abs_cos_held / min_med_abs_cos 逐位复现断言）。残差链 = 3193 verbatim（含修正版 build_dvu）。

**结果**：门 **3/3 全过** ⇒ HOLDOUT_PREDICTIVE_USTAR：
- qwen3-4b：4 折 0.5632 / 0.4910 / 0.5996 / 0.5657（min 0.4910）
- qwen3-14b：0.5264 / 0.5303 / 0.6035 / 0.5198（min 0.5198）
- glm4-9b：0.5757 / 0.5254 / 0.5875 / 0.5221（min 0.5221）

**关键读数**：全部 12 个折中位 |cos| 均**高于** FTR-25 同协议对照 min（0.3906/0.4174/0.3717，本轮逐位复现）——u* 的跨类泛化强于 FTR-25 共享方向；u*（3193 判 NOVEL：与 dvu 方向级正交、不可归约已知结构）同时是**留出可预测**的。机制新颖 + held-out 可预测双属性确证。

**设备锚**：drift69/89/90/91/93 全 ==0 逐位 ×3，dv_drift==0——残差链 3189→3194 六连逐位同一。

**产物**：FTR-26 E1_repeatable→E2_held_out（第 19 个 E2，upgrade_log #8）：evidence_level 变更 + anchors append（role=held_out）+ statement append，registry v1.19→v1.20（`2aac7e5e`，26 特征 / E2×19 / E3×3 / E1×3 / failures 19 / upgrades 8），atlas_v1_22.html（`b991709b`，771 字段 0/0/0/0）。design `a733e0b6`，res `f4836e97`/seal `be9ce95b`，final `fca2ccbb`/`90bb8a9c`，ledger #346（post `40c1cea7`）。

**诚实日志**：SMOKE-1 NameError（NPZ57_KEY 字典字面量遗漏——SRC p3157 三键均在、仅 dict 未定义），单行修复无观测污染；SMOKE/FULL per_model 载荷逐位一致。本轮零 DESIGN 漂移。

**理论意义**：D 线两连（3193+3194）确立 u* 是一个**新的承重轴**：内容上不可归约为 isa/hasa 偏移方向（dvu）、seen 类均值方向、或任何已知结构方向（NOVEL 2/3），但在跨类留出协议下可预测（3/3，min ≥ 0.49）。与 FTR-25（跨类共享方向）形成互补对：FTR-25 描述"共享什么"，FTR-26 描述"还剩什么且仍可泛化"。

**3195 预注册（G5-D3，零 GPU，封存复用）**——u* 内容归因深化臂：检验 u* 是否可由**未见于门集**的候选结构方向线性张成：对每模型，构造候选方向集 {dvu, v_s×6, u_seen, v_c×4, KOUT 主成分 top-k（自 p3169 collect 斜面）}，报告 u* 在该张成子空间上的投影残差比 ‖u*−P_C u*‖/‖u*‖；门（freeze 前细化）：3/3 模型残差比 < 0.5 ⇒ PARTIALLY_DECOMPOSED（FTR-26 statement 补充归因注记，registry 不动）；否则 NOVEL_CORE 确证维持 + 报告 top 贡献方向谱。设备锚：3189–3194 封存标量 drift == 0 逐位；对照锚：3193 门集读数（gate_max 0.2130/0.3303/0.2457、|cos(u*,dvu)| 0.1064/0.0405/0.0240）同协议复算须 drift==0。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(SEC)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO: %s -> %s (+%d chars)' % (pre, post, len(SEC)))
