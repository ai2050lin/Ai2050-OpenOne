# -*- coding: utf-8 -*-
"""memo_2924_append.py -- append Phase 2924 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T12:43:41 -> [2026-09-19 12:43]).
Idempotent: refuses if a 2924 title already exists.
"""
import hashlib
import json
import os
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
ROADMAP = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs"
           r"\lpf_multiaxis_gating_roadmap_v1.md")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2924_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2924_depth_polarity_gradient" in m_ids, \
    "run ledger_2924_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2924_depth_polarity_gradient" in l14["connects"]
assert len(L["measurements"]) == 63 and len(l14["connects"]) == 31
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2924:" not in txt, "already appended"

SEC = """
## Phase 2924: 深度分级极性编码检验——depth_gradient_absent [2026-09-19 12:43]

### 目的（2923 接续候选 A / 纲领 afa60f18 §4 2924 行）
2923 在 13 个 sig 事件上发现"浅 LOW 驱动 / 深 HIGH 驱动"层位结构（quasi-post-hoc）。本 Phase 零前向检验该结构是否为**全格连续梯度**：全 32 头（不限 sig 格）的极间对比方向 D[h,l] = m_hi[h,l] - m_lo[h,l] 是否随深度上升。zero-forward（1.2 s）。

### 设计（冻结口径）
- 主统计量：c_ax[l] = median_h D[h,l]（36 层剖面/轴）；rho_axis = Spearman(l, c_ax)；方向判据（运行前冻结）：c_ax 随 l 上升（浅 LOW / 深 HIGH）=> rho > 0。
- null：每轴 2000 次保组大小 pole 置换（fresh default_rng(2899)），D/c/rho 全重算；p_axis = 单尾 (#{rho_perm >= rho_obs}+1)/2001；逐层 p_l 单尾。
- 判决映射：>= 2/3 属性轴 rho>0 且 p<=0.05 => depth_graded_polarity_confirmed；1 => partial；0 => absent。
- P2 lang 对照（en/L 对比剖面同机器，描述）；P3 事件级 Spearman(d_pole, 事件层 l) + rng(2900) 置换（quasi-post-hoc 标注）；P4 对比强度剖面 median_h |D| + rho(|d_pole|, l)（描述）。

### 锚（3/3，全过）
- a1：2921 sign_M 重算（2922 a1 verbatim）median ~6e-10、max 0。
- a2：13 事件 d_pole 重算 vs 2923 npz d_pole13——**bit 级一致（max diff 0.0）**。
- a3：2923 npz t5_perm p50 6.0 / max 11。

### 结果
- **判决：depth_gradient_absent（0/3 轴通过）**。
- P1：speed rho **+0.203** p=0.175（方向对但不显著）；size rho **-0.061** p=0.607（**反向**：shallow12 +1.1e-05 -> deep12 -2.4e-05）；moist rho -0.054 p=0.640（两端均 LOW 侧 -4.4e-05/-3.8e-05）。逐层显著格散点（speed 4、size 2、moist 0），无连贯梯度。c 剖面四分位均在 null 范围。
- P2 lang 对照：rho -0.079 p=0.632——对照干净（无梯度）。
- P3 事件级结构**成立**：**Spearman(d_pole, 事件层 l) = +0.654，p_perm = 0.025**（13 事件，quasi-post-hoc）。
- P4：对比强度层平坦（各轴 shallow12/deep12 中位 |D| 同为 ~1e-4 量级：speed 0.317->0.418e-3、size 0.467->0.324e-3、moist 0.369->0.402e-3）；**rho(|d_pole|, l) = -0.495**——浅层事件对比更强（(21,7) |1.84| 浅层最强、(11,23) |0.84| 深层最弱）。

### 解读
1. **核心张力 -> 结构定位**：事件级 rho +0.654（p 0.025）显著、全格中位剖面 0/3 轴显著——**2923 的层位结构是"事件选择性质"（哪些格成为 maxT-sig 事件），不是全格极性编码梯度**。sig 事件格是特殊的双极对比检测器（2923：13/13 contrast 型），其读出符号按层组织；背景格的对比方向为噪声级/非层组织。
2. **2923 判决的边界划定**（不推翻、限定范围）：layer_structured 在事件级成立；不外推为深度分级读出编码。这正是 quasi-post-hoc 标注制度的价值——2923 已把 P2 权重降为描述性，2924 的独立全格检验给出否定。
3. size 全格反向（浅 +/深 -）与 speed 正向并存——轴间不一致进一步排除统一编码梯度。
4. 对比强度深度平坦 + rho(|d_pole|, l) = -0.495：事件"选拔"偏浅层强对比格——浅层格更容易产生强对比事件（LOW 端），深层 HIGH 驱动事件对比弱。事件层分布偏深（2922：median peak 15）与"浅层更强对比"并存的张力指向：sig 判据是 margin（sign 一致性）而非对比幅度。
5. 理论对应：Cmp(o,r,v) 图景下，(头,层) 格不是预布线的极性通道阵列——事件格是稀疏选拔的对比检测点，其方向组织是选拔的伴随性质而非全局编码方案。

### 方法论常数（新增）
- **全格剖面检验协议**（median_h D 层剖面 + 保组置换单尾 + 逐层 p_l）入标配；事件级结构必须与全格基线对照报告（event-selection vs grid-wide 两分定位）。
- bit 级锚（d_pole13 max diff 0.0）确认零前向复算路径确定性的上限实践。

### 执行史
- 主脚本一次运行成功（1.2 s，无修复）；seal 正常。
- 零前向：无模型加载。

### 硬伤
- 事件级 rho +0.654 基于 n=13 混池（quasi-post-hoc）——不能排除轴内混杂；per-axis 事件级检验 n 太小（2/9/2）未做。
- 全格否定用中位数剖面——均值/分位数剖面或其他聚合未测；c 剖面噪声级（|c| ~ 1e-5 ~ 1e-4）与事件级 |d_pole| 0.7-1.8 的量级差异悬殊，中位可能被大量弱响应头淹没（head 子集分析未做）。
- size 全格反向的解释未深究（其 9 个事件最多、13 事件级信号主要由 size 贡献）。
- 单模型；n=1 run（bit 级锚缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2924_depth_polarity_gradient.py: e57b578d
- execution.json: 9ac2e862（created 2026-09-19T12:43:41）
- result.json: 27f034a7（final_verdict=depth_gradient_absent，runtime 1.2 s）
- depth_polarity_gradient.npz: 4db45a37（c_speed/size/moist/lang、rho_axis、d_pole13_ref、ev_layers）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2924/depth_polarity_gradient/
- Ledger：M2924_depth_polarity_gradient 入账，measurements 62->63，L14 connects 30->31，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2924 行完成）

### 接续（2925 候选）
- A（主选）：**事件选拔解剖**——13 个 sig 格 vs 同层 non-sig 格的特征判别（零前向：margin 之外的 PR_word、|D|、头一致性、pole 分离度等特征；回答"什么样的格成为事件"）。
- B：**探针相对性检验**——lang 词表换词级 lang 探针（dirs_word）重测事件集（一次前向 ~1 min）。
- C（零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道，roadmap 遗留项）。
- D：**事件格空间范围**——sig 格邻域（同头邻层/同层邻头）是否也 contrast-structured（事件格的空间延展，零前向）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2924: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2924 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2924")
