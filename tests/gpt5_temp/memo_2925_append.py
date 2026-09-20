# -*- coding: utf-8 -*-
"""memo_2925_append.py -- append Phase 2925 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T12:50:13 -> [2026-09-19 12:50]).
Idempotent: refuses if a 2925 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2925_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2925_event_selection_anatomy" in m_ids, \
    "run ledger_2925_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2925_event_selection_anatomy" in l14["connects"]
assert len(L["measurements"]) == 64 and len(l14["connects"]) == 32
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2925:" not in txt, "already appended"

SEC = """
## Phase 2925: 事件选拔解剖——events_margin_only + |phi| 事后诊断 [2026-09-19 12:50]

### 目的（2924 接续候选 A / 纲领 afa60f18 §4 2925 行）
2924 把层位结构定位为"事件选拔性质"。本 Phase 零前向回答"什么选拔了事件"：13 个 sig 格 vs 同层 non-sig 头的 7 特征判别（margin 参照排除出判决族）。zero-forward（1.0 s）。

### 设计（冻结口径）
- 特征（每格标量）：f1 margin（判据本身，参照）；f2 phi（pole x 响应符号 2x2 关联，r==0 计正侧——**带符号**）；f3 |d_pole|；f4 |m_hi - m_lo|；f5 PR_word；f6 top3_mass；f7 mean|r|；f8 neighbor_phi（同头邻层 max）。
- 百分位：pct = mean(池 <= sig 值)，池 = sig 格 + 同轴同层 non-sig 头。
- 检验：每特征 13 个 pct，sign test vs 0.5 单尾精确二项（p 地板 1/8192 << 0.05，粒度先检过）；族 = 7 非平凡特征，BH q=0.05（exploratory 注明）。
- 判决映射：n_BH >= 3 且 median pct(phi) >= 0.9 => events_polar_separation_selected；n_BH >= 3 => events_multifeature_selected；else => events_margin_only。

### 锚（3/3，全过）
- a1：2921 sign_M 重算 median ~6e-10、max 0；a2：d_pole13 vs 2923 npz **bit 级 0.0**；a3：2924 npz rho_axis[size] = -0.060746。

### 结果
- **冻结判决：events_margin_only（n_BH = 2/7 < 3）**。
- P1 特征判别：**d_pole_abs 中位 pct = 1.0（13/13 > 0.5，sign p = 1.22e-4，BH q = 4.27e-4 显著）**；**contrast_raw 中位 0.906（13/13，同 p/q 显著）**；mean_abs_r 中位 0.75（9/13，p=0.133 ns）；pr_word 0.50、top3_mass 0.32、neighbor_phi 0.50 全 ns——**响应形态（宽度/幅度）与空间延伸不参与选拔**。margin 参照中位 pct = 1.0（判据本身，trivial）。
- **登记缺陷：phi 检验无效**——预注册用带符号 phi + 单侧百分位，LOW 驱动格的 phi 是最极端负值，数学上不可能达到"中位 pct >= 0.9"判据。detail 显示双峰签名：8 格 pct ~1.0（HIGH 驱动）、5 格 pct ~0.03（LOW 驱动）——两端都是同层最极端。
- **事后 |phi| 诊断（descriptive，非冻结判决）**：**|phi| 百分位 13/13 全部 = 1.0**（sign p = 1.22e-4）——每个 sig 格都是其所在层全部 32 头中极间符号关联最强者；与 d_pole_abs pct 13/13 = 1.0、contrast_raw 13/13 合并：**事件由极间对比强度选拔**，margin 判据与 |phi|/|d_pole| 本质同源。

### 解读
1. **事件选拔的一维性**：sig 格在"极间对比强度"族（|phi|、|d_pole|、contrast_raw）全面碾压同层背景（|phi| 13/13 全层第一），而响应形态（PR_word/top3_mass/mean|r|）与空间延伸（neighbor_phi）完全不分离——事件不是"宽响应格"或"强响应格"，是"pole 两端反号对比极端的格"。margin（sign-Gram 一致性）与 |phi|/|d_pole| 是同一选拔量的不同投影。
2. **2924 "事件选拔性质"的具体化**：背景格对比弱/无方向 -> 不成事件；sig 格 = 层内极间对比最强点 -> 事件；层位结构（2923）是"对比方向"的组织，选拔（2925）是"对比强度"的组织。
3. **预注册缺陷的制度价值**：带符号 phi 的 pct 判据在混合符号事件集上数学不可达（设计期未察觉 phi 双峰）——判决按冻结映射保持 margin_only，缺陷与事后诊断如实入账；2926 用 |phi| 修正口径复测（预期翻转至 events_polar_separation_selected），完成"预注册 -> 缺陷发现 -> 修正复测"纪律闭环。
4. 理论对应：Cmp(o,r,v) 候选竞争图景下，(头,层) 事件 = 层内极间对比的极值点——"事件格"是对比竞争的胜出者，不是独立通道的开关；事件稀疏性源于每层只有少数格达到极值。

### 方法论常数（新增）
- **带符号量的百分位判据禁令**：预注册特征若带符号（phi、d_pole 等）且事件集符号混合，判据必须用 |值| 或符号对齐口径——否则判据不可达（2925 教训，与纪律 8 粒度先检并列的"判据可达性先检"）。
- **特征族判别协议**（sig vs 同层 non-sig 百分位 + sign test + BH，margin 参照排除）入标配。
- 事后诊断登记制度（2920 功效诊断先例）延续：|phi| 诊断不入判决，登记待修正复测。

### 执行史
- 主脚本一次运行成功（1.0 s，无修复）；seal 含 |phi| post-hoc 探针（独立脚本，报告并列）。
- 零前向：无模型加载。

### 硬伤
- phi 实现缺陷使冻结判决的信息量受限（n_BH=2 是缺陷下的保守下界；|phi| 修正后实质为 3/7 显著且 median pct 1.0——判决实质应为 polar_separation_selected，待 2926 预注册复测正式翻转）。
- BH 族 exploratory（7 特征非独立：d_pole_abs/contrast_raw/|phi| 同源）。
- |phi| 诊断为事后描述（13/13 = 1.0 极端整齐，或与 margin 判据同源Circular——2926 复测必须给出与 margin 独立的口径）。
- n=1 run（bit 级锚缓解）；单模型。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2925_event_selection_anatomy.py: 47113e12
- execution.json: ef748611（created 2026-09-19T12:50:13）
- result.json: 7846d213（final_verdict=events_margin_only，runtime 1.0 s）
- event_selection_anatomy.npz: d59402cf（pct_matrix 13x8 [margin+7 特征]、feat_keys、event_ids）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2925/event_selection_anatomy/
- Ledger：M2925_event_selection_anatomy 入账，measurements 63->64，L14 connects 31->32，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2925 行完成）

### 接续（2926 候选）
- A（主选）：**|phi| 修正口径复测**——同 2925 设计，phi 判据改 |phi|（判据可达性先检过），预期翻转 events_polar_separation_selected；同场加 margin-独立性问题（|phi| 与 margin 的格级秩相关——若 ~1 则两判据同源，事件选拔=对比强度极值，写死结论）。
- B：**探针相对性检验**——lang 词表换词级 lang 探针（dirs_word）重测事件集（一次前向 ~1 min）。
- C（零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道，roadmap 遗留项）。
- D：**对比天花板剖面**——各层 |phi| 分布形状（non-sig 格均匀弱 vs 与 sig 格有 gap），事件选拔的层间差异。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2925: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2925 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2925")
