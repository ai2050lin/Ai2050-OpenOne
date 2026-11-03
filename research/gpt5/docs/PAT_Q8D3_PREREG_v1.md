# PAT_Q8D3_PREREG_v1 — pos0 / 格式位置干预（Q8-D 挂账项）

冻结时间：2026-11-01 18:47。冻结后只许改代码实现错误，不许改阈值与网格。

## 0. 问题

Q8-D2 只测了**内容位置**（LAST）与全部位置（ALLPOS）的频率方向平移，得到 NULL_BOUNDED。
未测的位置是 **pos0 / 格式位置**：Q5-A 证明 pos0 `<|im_start|>` 占 next-token 分布竞争的
46%（寄存器 vs 内容的战场）。若频率旋钮在计算中真有位置，最可能在寄存器位置。

预注册的可证伪命题：**"频率方向在 pos0 上被使用"**。若 pos0 也是 NULL_BOUNDED，
则"输入侧任何位置都不使用频率方向"的结论完备，Q8-D 线彻底关闭。

## 1. 材料

与 Q8-D2 完全相同的候选模板（36）× 名词（50），但包上**chat template**
（add_generation_prompt=True），使 pos0 成为格式 token：
- Qwen3-4B：`<|im_start|>`（模板自带默认 system 行，pos0 仍为首个 `<|im_start|>`）
- GLM-4-9B：`[gMASK]`（8-bit 加载，与 D2 相同的量化口径）

**模板闸重新评估**（包装改变了上下文，D2 的闸表不可迁移）：每模板用 12 个名词的
包装版 prompt，只看基线 p_M ≥ 0.80 且 H ≥ 1.00，冻结通过集后再做干预。
取前 200 条。

## 2. 干预臂

位移向量与 D2 完全相同（w_freq 从 CHAR∪WORD 池重建，δ 单位 = zipf）：

| 臂 | 位置 | 方向 | δ |
|---|---|---|---|
| A1 | POS0 | FREQ | −2, −1, −0.5, +0.5, +1, +2 |
| A2 | POS0 | RAND ×8 | +2 |
| A3 | POS0 | EMB ×8 | +2 |
| A4 | LAST | FREQ | −2, +2（同跑内对照，包装版基线） |

统计量与 D2 相同：ΔE[zipf] 均值 + bootstrap 95% CI + MDE、top1 一致率、ΔH、KL。

## 3. 判决（预注册枚举）

**D3-1 主判（pos0 频率因果）**，口径 FREQ × POS0 × IN：
- `CAUSAL_KNOB`：ρ_dose ≥ 0.8 且 ΔE(+2) ≥ +0.10 且 CI 不含 0
- `CAUSAL_KNOB_INVERSE`：对称负向
- `NULL_BOUNDED`：MDE ≤ 0.02 且 |ΔE(+2)| < MDE
- `PENDING_POWER`：MDE > 0.10
- `WEAK_COUPLING` / `KNOB_NOT_USED`：同 D2

**D3-2 位置敏感性（KL 比）**，δ=+2：r = KL(POS0) / KL(LAST·A4)
- `REGISTER_SENSITIVE`：r ≥ 3（支持 Q5-A 寄存器假说）
- `POSITION_NEUTRAL`：0.5 ≤ r < 3
- `LAST_DOMINANT`：r < 0.5

**D3-3 方向特异性（pos0, δ=+2）**：FREQ |ΔE| 对 RAND p95 与 EMB p95，
枚举同 D2（DIRECTION_SPECIFIC / ON_MANIFOLD_ONLY / GENERIC_PERTURBATION）。

## 4. 预注册的将被证伪的表述

- 若 D3-1 = CAUSAL_*：撤回 Q8-D2 的"频率子空间不被使用"，改为"只在寄存器位置使用"。
- 若 D3-2 = LAST_DOMINANT：Q5-A 的 pos0 竞争优势对**频率扰动**不成立（只对内容竞争成立）。

## 5. 已知陷阱（沿用 D2 清单）

p_M 崩塌（闸门重评）、bf16 batch 内核差（ΔE 口径而非 logit）、GLM 8-bit 量化噪声
（容差 0.05）、lm_head meta 断言、复核独立代码路径。

## 6. 与 D2 结果不可直接比对的说明

A4（LAST × 包装版）是跑内对照；D2 的数字只作参考不作判据。
