# -*- coding: utf-8 -*-
"""memo_2912_append.py -- append Phase 2912 section to AGI_GPT5_MEMO."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')

SEC = u'''
## Phase 2912: 符号平衡锯齿正式化与词级归因（gap_zigzag_absent；2/3 反转律与离散 tie 校准勘误） [2026-09-19 09:04]

### 动机与预注册（execution.json 先冻结）
2911 定位交替载体为类间符号平衡 gap_j = |pos_frac0_j - pos_frac1_j|（qwen_attn 描述性锯齿 7/8，类 0 正率波动驱动）。本 Phase 正式化：零前向矩阵分析。探针：P1 gap 序列 zigzag 检验（k = 内点方向反转数，per-layer 独立类标签置换 null，类大小固定 22/35，N_PERM=20000，单侧 p = P(perm >= obs)）；P2 diff(gap) lag-1 自相关负向检验；P3 词级符号翻转归因（描述性：各类 top-5 翻转词占比 + max-flip 词 fair-coin 尾）。锚：a1 全层 margin/acc vs stored 2e-5；a2 Delta_B vs 2905 1e-4；a3 逐层 d=1 score vs 2910 score_true 1e-6。校准（rng [2912,0]，200 个 iid 57x10 符号矩阵 x 1000 perms）+ 冻结判决映射（S = 双显著组集合）。

### run1 校准失败与构造代数审计（diag_2912a/b）
run1 判决 audit_calib_fail_all_void：frac_in_band=0.8950（合格）但 p_median=0.6868 超 v1 带 [0.40,0.60]。按 2809 制度先做构造代数诊断再动主检验：
- diag_2912a（5000 个 iid 矩阵，四种 gap 构造）：未置换分裂 5.217 / per-column 置换 5.181 / 全局行置换 5.212 / 重复 5.219——四种构造 zigzag 分布相同，**置换 null 构造无偏移**；worked example + pooled lag-1 自相关 raw -0.491 / perm -0.502。
- diag_2912b（主实现精确复刻，1000 矩阵 x 500 perms）：k_obs 均值 5.179 vs k_perm 均值 5.207——**obs 与 null 同分布**；null k 直方图匹配 Binomial(8, 2/3)；tie 率 tau_emp 0.2207 ≈ tau_pool 0.2202（内部一致）；p_mean 0.6183 vs 理论 0.6101；p_median 0.7006。
- **根因一（诊断对照勘误）**：diag_2912a 写入的"binomial(8,0.5) mean 4.0"对照本身错误。zigzag/局部极值计数的 iid null 是 **2/3 up-down 反转律**：三 iid 值的中间点为局部极值的概率 = 1/3 + 1/3 = 2/3，E[k] = 8 x 2/3 = 5.33（tie 折损后 ~5.2，观测吻合）。
- **根因二（v1 判据口径缺陷）**：含等号单侧离散置换 p = P(perm >= obs) 的期望不是 0.5 而是 **E[p] = 0.5 + tau/2**（tau = sum pk^2 ~ 0.22），故 p 中位数自然中心 ~0.70。v1 的 p_median 带 [0.40,0.60] 只对连续 p 成立——校准失败是判据缺陷而非 null 错误。
- **v2 修订（解析、非拟合）**：保留 frac 带 [0.80,0.97]；p_median 带替换为 p_mean 对解析期望 0.5 + tau_hat/2 的 3SE 带（SE = 0.5/sqrt(200)，tau_hat 从校准自身 null k 直方图解析计算）。p 定义与其余冻结元素不动；PREREG 登记 calibration_v1_superseded + calibration_v2_note；run1 产物按重跑纪律删除。

### v2 主结果（execution d8b84a99 / result 4611db7f / npz 08dd335d；created 2026-09-19T09:04:34）
- 校准 v2 通过：frac_in_band=0.8950；p_mean=0.5994 vs e_p=0.6100（tau=0.2200），差 0.011 << 3SE=0.106，带 [0.5039,0.7160]；p_median=0.6868 与 run1 几乎同值（同 rng 流），证实 run1 失败纯系判据口径。
- 锚 4/4：margin/acc vs stored（最大偏差 ~2e-5 级）；dLB（Delta_B vs 2905）4.35-4.93e-5 < 1e-4；sLB（score vs 2910 score_true）4.54-4.98e-7 < 1e-6。
- 主检验（20000 perms/组）：

| 组 | zigzag k | p_zig | rho1 | p_rho |
|---|---|---|---|---|
| glm4_mlp | 7 | 0.539 | -0.449 | 0.567 |
| glm4_attn | 6 | 0.794 | -0.108 | 0.928 |
| qwen_mlp | 5 | 0.708 | -0.528 | 0.440 |
| qwen_attn | 7 | 0.172 | -0.694 | 0.177 |

- S（双显著集）= 空 => 判决 **gap_zigzag_absent**。
- P3 描述性：top-5 翻转词占比 c0 0.25-0.41；max-flip 词 fair-coin 尾仅 qwen_mlp 显著（8 flips，p=0.0195）；qwen_attn 7 flips p=0.090 不显著。

### 关键解读（核心发现，重复三遍）
1. **描述性"锯齿 7/8"在正确 null 下不罕见**：iid 序列期望自带 5.33/8 个反转（2/3 律），7/8 的 p~0.17。2911c 的锯齿印象是 null 基线误读——局部极值类描述统计必须对照 2/3 律基线，不能对照 1/2 直觉。
2. **交替证据载体 = margin 符号序列，非 gap 幅度锯齿**：2911 P1（margin 符号 8/9 翻转 vs null 翻转率 0.5，p=0.0195）真实且特有；2912 gap 幅度锯齿在 2/3 律 null 下无附加证据力（qwen_attn p_zig=0.172 / p_rho=0.177）。两条检验不矛盾：margin 交替的几何推论（gap 高低交替）被 null 的厚上尾吸收。2911 判决 alternation_not_confirmed_margin_only 中 "margin_only" 经 2912 进一步坐实。
3. **校准方法论入账**：含等号单侧离散置换 p 的期望 E[p] = 0.5 + tau/2（tau = sum pk^2），置换检验校准带必须做离散 tie 修正；zigzag/局部极值 null 是 2/3 反转律。两条已制度化为 N13 notes（E10 级教训，无需 errata 条目——run1 未入账，v2 在主观测前冻结）。

### 硬伤
- 校准 v1 失败曾触发 all_void，v2 修订虽为解析非拟合且在 probes 观测前冻结，但严格说是第二次冻结——修订过程与理由已在 PREREG/MEMO 全程留痕（与 2906-2907 sigma 定义勘误同型）。
- 四组 rho1 全负（-0.11 至 -0.69）方向一致但均不显著，4 重比较下不能升格为发现；qwen_attn rho1=-0.694 是全谱系最负值，若未来需要可作专项高功效检验（非本 Phase 结论）。
- P3 词级归因为描述性，max-flip 尾检验的 fair-coin 假设忽略类内翻转率 pooling（脚本内已注明）。

### 结论
1. gap_zigzag_absent：类间符号平衡 gap 的锯齿在 per-layer 置换 null 下不显著——交替结构在幅度域无证据。
2. 2/3 反转律与离散 tie 修正两条方法论常数入账（N13）。
3. 2903-2912 谱系终态：qwen_attn 交替 = margin 符号域现象（2911 P1 唯一显著），其聚合下尾（2908/2910）与符号平衡波动（2911c）为同一现象的不同泛函投影；幅度域（gap zigzag）与信号域（osc）均排除。

### 接续
- 2913 候选：A（主选）**头级 per-head W_VO 分解**（qwen_attn L07/L09 层，把 B 的类相关行空间结构归因到头级 W_VO 贡献，零前向可用缓存 W_O/W_V）；B（备选）margin 符号交替的词级符号翻转动力（pf0 波动驱动词集合的层间演化追踪）；C 前向 SwiGLU 激活级归因（eps=1.0 预算，非零前向）。

### 文件
- 脚本 tests/glm5/phase2912_sign_balance_zigzag.py（48b4a335，v2 含 calibration_v2_note；run1 脚本 87376b53 系 2907 遗留哈希条目勿混）
- 产物 phase2912/sign_balance_zigzag/：execution.json d8b84a99 / result.json 4611db7f / sign_balance_zigzag.npz 08dd335d
- 诊断 tests/gpt5_temp/diag_2912a.py（构造对比；内含 binomial(8,0.5) 错误对照已由 diag_2912b 勘误）-> diag_2912b.py（主实现精确复刻 + tie/解析核对）
- Ledger：M2912_sign_balance_zigzag + L14 再精化（2912 gap-zigzag formalization；connects 19）+ N13_gap_zigzag_absent_2_3_reversal_law（measurements 51 / errata 9 / negatives 13 / ledger 7e2501bd）
'''

with io.open(P, 'a', encoding='utf-8') as f:
    f.write(SEC)

print('OK memo 2912 appended')
