# -*- coding: utf-8 -*-
"""更新工作区 MEMORY.md 到 Phase 17。所有替换 assert count==1 + 回读复核。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
s = io.open(P, encoding='utf-8').read()
REPS = []

REPS.append(("（**九次 P8–P16**）", "（**十次 P8–P17**）"))

REPS.append((
    "- Ledger `research\\gpt5\\atlas\\atlas_ledger.json` n=**299**（P8–P16 各 1；**N 线 P3–P7 待补**；`ledger_sha256_8=28ef0f92`）。基线 `_infra\\memo_baseline.json`：MEMO **411,257 B / 3,852 行 / sha8 12903cd5 / 16 标题**（P16@L3684；pre-append 锚 389,885 B / `0bd7bfd0`）。",
    "- Ledger `research\\gpt5\\atlas\\atlas_ledger.json` n=**300**（P8–P17 各 1；**N 线 P3–P7 待补**；`ledger_sha256_8=4abbabec`）。基线 `_infra\\memo_baseline.json`：MEMO **435,628 B / 4,103 行 / sha8 7dd4220a / 17 标题**（P17@L3853；pre-append 锚 411,257 B / `12903cd5`）。",
))

REPS.append((
    "（3 点 α 网格 Δ=0.0105 vs 14 点 Δ=0.000）。",
    "（3 点 α 网格 Δ=0.0105 vs 14 点 Δ=0.000）。(ad) **实现必须与 seal 字面定义逐字一致** —— 口径歧义（区间求和 vs 位点取值）只能由**独立实现交叉验证**捕捉，「数值合理」不足为凭。(ae) 文档/元数据数字**一律由 result 现场渲染**、禁手工转录 —— 自查须覆盖 Ledger 与 MEMO 散文（P17 两次手打被拦：`rev_note`、§1/§5/§6/§10）。",
))

REPS.append(("## 2 N 线主线（deepseek，P4→P16）— 细节见 MEMO",
             "## 2 N 线主线（deepseek，P4→P17）— 细节见 MEMO"))

REPS.append((
    "- **P15 = 跨模型复算「统一剖面」**：三臂同一 **nf4**（A0 qwen3-4b 校准 / A1 glm4-9b / A2 Qwen3-14B；bf16 三路线实测否决）。`NF4_FAITHFUL`（`max|dxhalf|=0.0062`）；`ARGS_GAP_LAYERSTACK`（`d_argmax` 14/10/5 ⇒ 两坐标窗位置分离是层栈性质）；`CONC_JUDGE_ALIVE_X`（null95_x 0.6982/0.5384/0.4367）。**⚠️ 6 个「臂×坐标」格只 2 格过零假设**（A0·J +0.1057、A1·xhalf +0.0357）⇒ **无跨模型稳健集中度判据**（坐标×模型双重依赖）。剖面跨模型同构（`spearman(J,depth)`≈−1）；`L*_own` = A0 **L6** / A1 **L3** / A2 **L4**。**事故 amend1**：`sup_id` 是 qwen 词表 id 被全局使用 ⇒ A1 读错类别 token，被正交装置门拦下、逐臂现场解析修复。",
    "- **P15 = 跨模型复算「统一剖面」**：三臂同 **nf4**（bf16 三路线实测否决）；`NF4_FAITHFUL`（`max|dxhalf|=0.0062`）；**⚠️ 6 个「臂×坐标」格只 2 格过零假设**（A0·J / A1·xhalf）⇒ **无跨模型稳健集中度判据**（坐标×模型双依赖）；剖面同构（`spearman(J,depth)`≈−1）；`L*_own`=L6/L3/L4。**事故 amend1**：`sup_id`（qwen 词表 id）被全局误用 ⇒ A1 读错类别 token，被正交门拦下、逐臂现场解析修复。",
))

P16_OLD = "- **P16 = 写入窗原点化剖面 + 集中度重设计（N2h1-α-9）**：三改动 —— ①网格下探 `[1..5]+[6..34]`（**23 位点**）；②主域 = **可达性掩膜** `REACH={ℓ:ρ(ℓ)≥UNREACH_y=0.10}`（`ρ=Y(ℓ,α=1)=dDonor/FULL_SWAP`）；③旧量 `top3_share` → (**`com_layer`**, **`span_k`**)（`com_layer`=**物理层号**质心、单位「层」、网格不变量；`span_k`=k 个最大 |Δ| 跨度；**双边**）。**设计期整族排除**：置换零假设保留 jump **多重集** ⇒ 谱熵/max÷mean/参与比在零假设下**恒等于观测**、p≡1 ⇒ **结构性退化族**（P15「换谱熵」设想否证）。**锚 bit-for-bit**（`max|Δxhalf|=0.000e+00`、argmax 全同 ⇒ 三臂 `RECON_OK` 严格）。**主结果 1**：`ℓ_reach == L*_own` **3/3 严格**（6/3/4）⇒ 两种完全不同的可观测量在同一位点重合。**主结果 2**：旧量 6 格只 2 格显著 → 新量 **5** 格。**⚠️ 主结论·P6 = 预注册否证**：`com_layer(xhalf)−com_layer(J)` = A0 **+14.169** / A1 **+9.947** / A2 **−0.716 层** ⇒ `CENTROID_PARTIAL`；**A2（同家族 3.5×）反号** ⇒ 按 seal `may_falsify_the_whole_line` 条款，**P12/13/14「`xhalf` 深尾集中 / `J` 浅端集中」的物理深度表述整体撤回**（只保留旧坐标下的形状差异）。装置门全过 ⇒ **成功的自我否证**。7 预测 **6 PASS / 1 FAIL（P6）**。"
P16_NEW = (
    "- **P16 = 写入窗原点化剖面 + 集中度重设计（N2h1-α-9）**：①网格 `[1..5]+[6..34]`（**23 位点**）；"
    "②主域 **`REACH={ℓ:ρ(ℓ)≥0.10}`**；③旧量 `top3_share` → (**`com_layer`**, **`span_k`**)（物理层号质心、网格不变量；双边）。"
    "**设计期整族排除**（置换 null 保留 jump **多重集** ⇒ 谱熵/参与比等**结构性退化**、p≡1）。锚 **bit-for-bit**（`RECON_OK`）。"
    "**主结果**：`ℓ_reach==L*_own` **3/3 严格**（两种不同可观测量在同位点重合）；新量 5 格显著（旧 2）。"
    "**⚠️ P6 预注册否证**：`com_layer(xhalf)−com_layer(J)` = +14.169/+9.947/**−0.716** ⇒ **A2 同家族 3.5× 反号** "
    "⇒ 按 `may_falsify_the_whole_line`，**P12/13/14 物理深度表述整体撤回**（6 PASS/1 FAIL）。\n"
    "- **P17 = 写入向量位置与效力（N2h1-α-10）**：P8 向量预算 **L6 → 逐层** ⇒ `w_ℓ=mean_pairs‖P_{U_ℓ}(Δ_inc,ℓ)‖` + 质心 "
    "`com_V`（**区间求和** `W_j=Σ_{ℓ∈[s_j,s_{j+1})}w_ℓ`，与 `com_layer` 同 mid）；`Δ_inc:=Δ_attn+Δ_mlp`、`Δ_attn:=Σ_hΔ_head_h` "
    "⇒ **可加性由构造成立** + 两保真度门（arch ≤3e-2 / blocks ≤1e-2）。**7 预测 6 PASS + P7 描述性**："
    "**P3 holdout `DEEP_ALL`** —— `com_V`=**26.150/26.704/26.675** vs `median(REACH)` **17/14/14**（A1/A2 seal 前**未观测**）；"
    "**P4 `POSITION_DECOUPLED` 2/3** —— A2 行为质心近重合（8.42/9.14）而向量质心 26.675 ⇒ **行为质心不可由向量质量质心替代**（P16 限界**加强**）；"
    "**P5 `MLP_DOMINANT_ALL`** —— 邻域（**三臂恰都 [26,28]**）`share_mlp_nb` **0.740/0.975/0.824**（≥L6 的 0.4717）、最大单头 ≤0.09；"
    "**P6 `WRITE_EFFICACY_ANTICORR_ALL`** —— `spearman(w_ℓ,J_ℓ)` **−0.546/−0.792/−0.603** ⇒ **深端有大量写入但对行为无效**（补 P16 撤回后的空洞）；"
    "**P7 描述性 `SPAN_CENTROID_COUPLED`（3/3）**。对照：置换 null **3/3 high 尾**、确认集（n=17 不相交）Δ ≤0.193。"
    "**同轮勘误 E1–E4**：**[E4]（最重要）`com_of_mass` 首版取位点单层而非 seal 的区间求和**（A0 24.466→**26.150**），"
    "修正后与**独立探针逐位相同**（跨实现交叉验证）、**未重跑前向**、判决不变；E1 Q7 同号口径（v1 错位配对误报 DECOUPLED）；E2 SMOKE 退化路径；E3 探针 nf4 权重路径。"
)
REPS.append((P16_OLD, P16_NEW))

REPS.append((
    "（A2 左端点 ℓ=3 而 `ell_reach`=4）⇒ 判据写「写入窗 ∈ REACH」；exec `bootstrap.seeds` 的 `new_x/new_j` **元数据错记**（记 +41/+53，实为 +61/+67），只影响零假设分位复现。",
    "（A2 左端点 ℓ=3 而 `ell_reach`=4）⇒ 判据写「写入窗 ∈ REACH」；exec `bootstrap.seeds` 的 `new_x/new_j` **元数据错记**（记 +41/+53，实为 +61/+67），只影响零假设分位复现；⑨ **P17**：`com_V`（向量质量质心）与 `com_layer`（行为质心）是**两件事**（A2 判别臂 `min_d`=17.53）；「深端写入无效」是**相关性**陈述（因果需逐层组件**行为**预算 = 下一死线）；`w_ℓ` 绝对量级只在臂内可比（跨臂只比质心位置与份额）。",
))

REPS.append((
    "不回改冻结件，以 MEMO 勘误节为准。",
    "不回改冻结件，以 MEMO 勘误节为准。\n- **P17 新增三条**：①**seal 口径与实现必须逐字对齐** —— 「区间求和 vs 位点取值」只能靠**生产 vs 独立探针逐位比对**检出；②**手工转录数字在 Ledger 与 MEMO 散文里都会发生** ⇒ 两处都要数据驱动渲染；③**臂记录里算好的量（`com_V`）在 MERGE 阶段不会重算** ⇒ 改口径必须**重跑臂**。",
))

REPS.append((
    "- **P17 最高 = 把「位置」接到「组件」**：在每臂 `com_layer` 邻域（**±2 层**）做**逐层组件预算**，沿用 P8 的**向量预算 `share_v`**（精确可加；**禁**效应份额），回答「质心所在层由谁贡献写入向量」。\n- **并列 = `span_k` 体系化**：P16 只作对照 ⇒ 需在三臂 × 双坐标 × k∈{2,3,5} 给**跨度谱**，检验「`J` 跨度小 = 单一步主导」是否跨模型稳健。\n- **第三 = `xhalf` 可达域敏感性**：深尾抬升（A0 L32→L34 上升）是否与「末层被排除」（patch 必剔末层）有关。",
    "- **P18 最高 = 逐层组件「行为」预算**：把 P17 的「向量预算」换成**行为预算**（逐层组件对 `Δlogit(is-a)` 的贡献），补 H11 因果缺口 —— 与 P17 §7 的 MLP 主导（`share_mlp_nb` 0.740/0.975/0.824）交叉验证：同向 ⇒ 升级因果；以 attn 为主 ⇒ P17 向量份额是**几何假象**。\n- **并列 = NF4 vs BF16 的 `w_ℓ` 口径**：P17 全在 nf4 ⇒ 须在 A0 同尺度 bf16 复算 `w_ℓ` 谱，确认峰值位置与 `com_V` 不随量化口径漂移。\n- **第三 = 邻域宽度 ±2 敏感性**：三臂邻域**恰都 [26,28]** ⇒ 验 ±1/±3 是否改变 MLP 主导结论。",
))

REPS.append(("（**15 臂 + 56 坑**）", "（**15 臂 + 57 坑**）"))
REPS.append(("（**27 教训 / 九次链 P8–P16**）", "（**28 教训 / 十次链 P8–P17**）"))

for i, (old, new) in enumerate(REPS):
    n = s.count(old)
    print('REP[%d] count=%d' % (i, n))
    assert n == 1, 'REP[%d] expect 1 got %d' % (i, n)
    s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

t = io.open(P, encoding='utf-8').read()
assert 'n=**300**' in t and '(ad)' in t and '(ae)' in t and 'P17 = 写入向量位置与效力' in t
assert 'P18 最高' in t and '57 坑' in t and '28 教训' in t
assert '28ef0f92' not in t and '299' not in t.split('## 2')[0]
b = len(t.encode('utf-8'))
print('MEMORY.md updated: %d B / %d chars / %d lines' % (b, len(t), len(t.splitlines())))
