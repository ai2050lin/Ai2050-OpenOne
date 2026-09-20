# -*- coding: utf-8 -*-
"""memo_2926_append.py -- append Phase 2926 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T13:01:00 -> [2026-09-19 13:01]).
Idempotent: refuses if a 2926 title already exists.
"""
import hashlib
import json
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
ROADMAP = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs"
           r"\lpf_multiaxis_gating_roadmap_v1.md")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2926_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2926_event_selection_polar_fix" in m_ids, \
    "run ledger_2926_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2926_event_selection_polar_fix" in l14["connects"]
assert len(L["measurements"]) == 65 and len(l14["connects"]) == 33
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2926:" not in txt, "already appended"

SEC = """
## Phase 2926: 事件选拔 |phi| 修正复测——events_polar_separation_selected 翻转确认 [2026-09-19 13:01]

### 目的（2925 接续候选 A）
2925 冻结判决 events_margin_only 但登记 phi 判据缺陷（带符号 phi + 单侧 pct 在混合符号事件集上数学不可达），事后 |phi| 诊断 13/13 pct=1.0。本 Phase 用修正口径正式复测：phi→|phi|、neighbor_phi→max|phi|（两处带符号缺陷同步修正），预注册 correction_note 如实标注修正源于 2925 事后诊断（confirmatory re-test，非盲预测）。zero-forward（0.4 s）。

### 设计（冻结口径，2925 verbatim 除两处修正）
- 特征：f1 margin（参照）；f2 **|phi|**（修正）；f3 |d_pole|；f4 |m_hi-m_lo|；f5 PR_word；f6 top3_mass；f7 mean|r|；f8 **max|phi| 同头邻层**（修正）。
- pct/sign test/BH、判决映射全部 2925 verbatim（phi→phi_abs）。
- **P2 新增（|phi|-margin 独立性）**：rho13 = Spearman(|phi|, margin) over 13 sig 格（平均秩防并列）；null = 5000 次置换 rng(2901) 双侧；coupled iff rho13 >= 0.5 且 p <= 0.05；全格逐轴 Spearman 描述性；P2 不改变 P1 判决。
- P3 描述性：sig 格层内 |phi| 排名（全层 32 头口径）。

### 锚（3/3，全过）
- a1：2921 sign_M 重算 median ~6e-10；a2：d_pole13 **bit 级 0.0**；a3：rho_axis[size] = -0.060746。

### 结果
- **判决翻转确认：events_polar_separation_selected**（n_BH = 3/7 >= 3 且 median pct(phi_abs) = 1.0 >= 0.9）——2925 事后诊断的预测精确兑现。
- P1 幸存特征 3/7：**phi_abs 中位 pct = 1.0（13/13 全部 = 1.0，sign p 1.22e-4，BH q 2.85e-4）**、d_pole_abs（中位 1.0，3 格 0.94-0.97 其余 1.0）、contrast_raw（中位 0.906）；出局不变：pr_word 0.50 / top3_mass 0.32 / mean_abs_r 0.75 / neighbor_phi_abs 0.41 全 ns——响应形态与空间延伸仍不参与选拔。
- **P2 |phi|-margin 独立性否定：rho13 = 0.978，p_perm = 0.0**（5000 次 null max 0.863、p95 0.561，0 次达到观测）；**全格逐轴 Spearman：speed 0.877 / size 0.897 / moist 0.809**——|phi| 与 sign-Gram margin 是同一底层量（符号一致性强度）的近单调别名，**不是新选拔维度**；选拔机制正式收敛为一维：极间对比/符号一致性强度极值。
- **P3 全层排名 12/13 rank-1**：唯一例外 (18,7)@L7 排名 2——被同层双事件 (21,7) 压制（|phi| 0.811 vs 0.606；margin 1.238 vs 0.654）；L7 是**双层选拔层**（2922 已知联动对 rho 0.676）。P1 池口径（非事件头）下 13/13 = 1.0 与 2925 事后诊断精确复现——口径差异澄清，非矛盾。

### 解读
1. **判决翻转的制度价值**：2925→2926 完成"预注册→缺陷发现→修正复测→翻转"完整纪律闭环；polar_separation_selected 的实质内容 = 13/13 sig 格在其层非事件头中 |phi| 百分位全为 1.0。
2. **选拔一维性正式化（P2 是本 Phase 最大增量）**：|phi|、|d_pole|、contrast_raw、margin 四个显著特征在格级近单调耦合（rho 0.98/0.81-0.90）——事件选拔不是多特征合取，是**单一底层量（极间符号一致性强度）的极值选拔**；"响应形态"与"空间延伸"两个族完全不参与。事件 = 层内对比竞争的胜出点，竞争泛函只有一项。
3. **L7 双层选拔层**：同轴同层两个事件并存（(21,7) 主、(18,7) 次），次事件在全层口径下让位但仍在非事件头池中居首——选拔是"每层非事件头中的相对极值"，允许多胜出者分层。
4. Cmp(o,r,v) 图景更新：候选竞争的读出端是一维强度泛函的稀疏极值化；层内"谁成事件"由对比强度决定，"往哪端偏"由层位符号组织（2923）决定——强度与方向是分离的自由度。

### 方法论常数
- **判据可达性先检**（2925 教训制度化后首次应用）：修正判据的可达性由事后诊断预先确认，复测一次通过。
- **修正复测的标注制度**：correction_note 写明修正来源与 confirmatory 性质——翻转判决的权重是"预测兑现"而非盲发现。
- **口径分辨**：同层多事件时"非事件头池 pct"与"全层 rank"不同——报告必须注明池口径（P1 vs P3）。

### 执行史
- 主脚本一次运行成功（0.4 s，无修复）；seal 探针：P3 rank-2 取证（L7 top-3 头 21/18/22）、P2 null sanity（max 0.863 < 0.978，p 地板真实）、P1 phi_abs 13/13 复现确认。

### 硬伤
- correction_note 使判决权重为 confirmatory（预注册前结果已知）——|phi| 13/13 的新信息量在于口径澄清与 P2/P3 增量，非首次发现。
- P2 耦合 rho13 基于已选拔的 13 格（selection bias 方向：sig 格内 margin 方差受限，rho 可能低估全格耦合；全格 0.81-0.90 补证但同为描述性）。
- 7 特征 BH 族 exploratory（三特征同源，P2 已量化其耦合）；n=1 run（bit 级锚缓解）；单模型。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2926_event_selection_polar_fix.py: c1eb659e
- execution.json: efc992a9（created 2026-09-19T13:01:00）
- result.json: 999151fa（final_verdict=events_polar_separation_selected，runtime 0.4 s）
- event_selection_polar_fix.npz: 83b1d027（pct_matrix 13x8、phi13/margin13、p2_null 5000、event_ids）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2926/event_selection_polar_fix/
- Ledger：M2926_event_selection_polar_fix 入账，measurements 64->65，L14 connects 32->33，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2926 行完成）

### 接续（2927 候选）
- A（主选）：**探针相对性检验**——lang 事件集在词级探针（dirs_word）下的复现性（一次前向 ~1 min；2920 "功能迁移≠几何同一"的正面检验：事件集是否探针不变）。
- B（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- C（零前向）：对比天花板剖面——各层 |phi| 分布形状（sig 格下方是连续背景还是 gap），选拔强度的层间差异。
- D（零前向）：L7 双层选拔解剖——同轴同层双事件 (21,7)/(18,7) 的头分工（词响应相关、极性、peak 层）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2926: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2926 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2926")
