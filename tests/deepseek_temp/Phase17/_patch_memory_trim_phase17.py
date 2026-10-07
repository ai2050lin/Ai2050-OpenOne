# -*- coding: utf-8 -*-
"""把 MEMORY.md 的 §2 索引条目压到注入安全（目标 < 7100 chars）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
s = io.open(P, encoding='utf-8').read()
REPS = []

REPS.append((
    "- **P8–P11（写入端）**：写入端**分布式**（向量预算 MLP 0.4717 / 最大单头 0.0742）⇒ 停追「搬运工」；`I_nl=6.85`。`amp` 平坦（12× ⇒ +6.6%）⇒ L6 内**无阈值增益**；行为 S 形（**x\\*≈0.59–0.63**）⇒ 非线性在 L6 **之后**；末位 U 充分（95.5%）不必要（损 5.8%）。**「栈=软门」**：J(ℓ) 5.41→1.15、**无断崖 ⇒ 逐层累积**；自基 `overlap` 0.689→0.030 ⇒ 深度塌掉主要是**方向失配**。三网 `rho(depth)` ≈ −0.87/−0.98/−0.99（16/17 重叠 ⇒ 趋势可信、逐位取值不可排序）。",
    "- **P8–P11（写入端）**：写入端**分布式**（向量预算 MLP 0.4717 / 最大单头 0.0742）⇒ 停追「搬运工」。`amp` 平坦（12× ⇒ +6.6%）⇒ L6 内**无阈值增益**、非线性在 L6 **之后**（行为 S 形，`x*`≈0.6）。**「栈=软门」**：J(ℓ) 5.41→1.15、无断崖 ⇒ 逐层累积；自基 `overlap` 0.689→0.030 ⇒ 深度塌掉主要是**方向失配**。",
))

REPS.append((
    "- **P12–P14（集中度）**：`J_swap` 25.35→1.06、`rho(xhalf,depth)=−0.783` ⇒ 累积换族存活；`top3_share_x=0.5745` 跨 0.60 ⇒ `ALLOCATION_AMBIGUOUS`。P13 逐位复现 P12 全部带 + 两置换 null（0.0）；`N_dec_J=10/17` vs 边际 2/17；**L34 反弹为真** ⇒ `CONCENTRATION_COORDINATE_DEPENDENT`（argmax 相距 13）。**P14：「第三条独立口径」不存在**（位置前缀 + A8 双双退化到 P12 单点族，18/18 逐位）；真增量 = 零假设校准：`xhalf` 观测 0.5745 **<** null 0.6998 ⇒ **无区分力**；`xhalf` 是网格不变量（`XH_RANGE` 0.109745），`J` 不是。",
    "- **P12–P14（集中度）**：`rho(xhalf,depth)=−0.783`；P13 复现 P12 + 两置换 null；**L34 反弹为真** ⇒ `CONCENTRATION_COORDINATE_DEPENDENT`。**P14：「第三条独立口径」不存在**（真增量 = 零假设校准：`xhalf` 观测 0.5745 **<** null 0.6998 ⇒ **无区分力**；`xhalf` 网格不变量，`J` 不是）。",
))

REPS.append((
    "- **P16 = 写入窗原点化剖面 + 集中度重设计（N2h1-α-9）**：①网格 `[1..5]+[6..34]`（**23 位点**）；②主域 **`REACH={ℓ:ρ(ℓ)≥0.10}`**；③旧量 `top3_share` → (**`com_layer`**, **`span_k`**)（物理层号质心、网格不变量；双边）。**设计期整族排除**（置换 null 保留 jump **多重集** ⇒ 谱熵/参与比等**结构性退化**、p≡1）。锚 **bit-for-bit**（`RECON_OK`）。**主结果**：`ℓ_reach==L*_own` **3/3 严格**（两种不同可观测量在同位点重合）；新量 5 格显著（旧 2）。**⚠️ P6 预注册否证**：`com_layer(xhalf)−com_layer(J)` = +14.169/+9.947/**−0.716** ⇒ **A2 同家族 3.5× 反号** ⇒ 按 `may_falsify_the_whole_line`，**P12/13/14 物理深度表述整体撤回**（6 PASS/1 FAIL）。",
    "- **P16（N2h1-α-9）**：网格 `[1..5]+[6..34]` + 主域 **`REACH={ℓ:ρ(ℓ)≥0.10}`**；旧量 `top3_share` → (**`com_layer`**, **`span_k`**)。**整族排除**结构性退化量（置换 null 保留多重集 ⇒ p≡1）。锚 **bit-for-bit**；`ℓ_reach==L*_own` **3/3 严格**。**⚠️ P6 预注册否证**：`com_layer(xhalf)−com_layer(J)` = +14.169/+9.947/**−0.716** ⇒ **A2 同家族 3.5× 反号** ⇒ 按 `may_falsify_the_whole_line`，**P12/13/14 物理深度表述整体撤回**（6 PASS/1 FAIL）。",
))

REPS.append((
    "- **P17 = 写入向量位置与效力（N2h1-α-10）**：P8 向量预算 **L6 → 逐层** ⇒ `w_ℓ=mean_pairs‖P_{U_ℓ}(Δ_inc,ℓ)‖` + 质心 `com_V`（**区间求和** `W_j=Σ_{ℓ∈[s_j,s_{j+1})}w_ℓ`，与 `com_layer` 同 mid）；`Δ_inc:=Δ_attn+Δ_mlp`、`Δ_attn:=Σ_hΔ_head_h` ⇒ **可加性由构造成立** + 两保真度门（arch ≤3e-2 / blocks ≤1e-2）。**7 预测 6 PASS + P7 描述性**：**P3 holdout `DEEP_ALL`** —— `com_V`=**26.150/26.704/26.675** vs `median(REACH)` **17/14/14**（A1/A2 seal 前**未观测**）；**P4 `POSITION_DECOUPLED` 2/3** —— A2 行为质心近重合（8.42/9.14）而向量质心 26.675 ⇒ **行为质心不可由向量质量质心替代**（P16 限界**加强**）；**P5 `MLP_DOMINANT_ALL`** —— 邻域（**三臂恰都 [26,28]**）`share_mlp_nb` **0.740/0.975/0.824**（≥L6 的 0.4717）、最大单头 ≤0.09；**P6 `WRITE_EFFICACY_ANTICORR_ALL`** —— `spearman(w_ℓ,J_ℓ)` **−0.546/−0.792/−0.603** ⇒ **深端有大量写入但对行为无效**（补 P16 撤回后的空洞）；**P7 描述性 `SPAN_CENTROID_COUPLED`（3/3）**。对照：置换 null **3/3 high 尾**、确认集（n=17 不相交）Δ ≤0.193。**同轮勘误 E1–E4**：**[E4]（最重要）`com_of_mass` 首版取位点单层而非 seal 的区间求和**（A0 24.466→**26.150**），修正后与**独立探针逐位相同**（跨实现交叉验证）、**未重跑前向**、判决不变；E1 Q7 同号口径（v1 错位配对误报 DECOUPLED）；E2 SMOKE 退化路径；E3 探针 nf4 权重路径。",
    "- **P17 = 写入向量位置与效力（N2h1-α-10）**：P8 向量预算 **L6 → 逐层** ⇒ `w_ℓ=mean_pairs‖P_{U_ℓ}(Δ_inc,ℓ)‖` + 质心 `com_V`（**区间求和**，与 `com_layer` 同 mid）；可加性**由构造成立** + 两保真度门（arch ≤3e-2 / blocks ≤1e-2）。**7 预测 6 PASS + P7 描述性**：**P3 holdout `DEEP_ALL`** `com_V`=**26.150/26.704/26.675** vs `median(REACH)` **17/14/14**（A1/A2 seal 前**未观测**）；**P4 `POSITION_DECOUPLED` 2/3**（A2 行为质心近重合 8.42/9.14 而向量质心 26.675 ⇒ **行为质心不可由向量质量质心替代**）；**P5 `MLP_DOMINANT_ALL`** 邻域（**三臂恰都 [26,28]**）`share_mlp_nb` **0.740/0.975/0.824**；**P6 `WRITE_EFFICACY_ANTICORR_ALL`** `spearman(w,J)` **−0.546/−0.792/−0.603** ⇒ **深端有大量写入但对行为无效**；**P7 描述性 `SPAN_CENTROID_COUPLED` 3/3**。对照：置换 null **3/3 high 尾**、确认集 Δ ≤0.193。**勘误 E4（最重要）**：`com_of_mass` 首版取**位点单层**而非 seal 的**区间求和**（A0 24.466→26.150），修正后与**独立探针逐位相同**、未重跑前向、判决不变；E1 Q7 同号口径；E2 SMOKE 退化；E3 探针 nf4 路径。",
))

for i, (old, new) in enumerate(REPS):
    n = s.count(old)
    print('REP[%d] count=%d' % (i, n))
    assert n == 1, 'REP[%d] expect 1 got %d' % (i, n)
    s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
t = io.open(P, encoding='utf-8').read()
assert 'P17 = 写入向量位置与效力' in t and 'P18 最高' in t and '(ae)' in t
print('MEMORY.md now: %d B / %d chars / %d lines' % (len(t.encode('utf-8')), len(t), len(t.splitlines())))
