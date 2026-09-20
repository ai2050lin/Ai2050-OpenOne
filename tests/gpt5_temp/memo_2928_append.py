# -*- coding: utf-8 -*-
"""memo_2928_append.py -- append Phase 2928 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T13:24:13 -> [2026-09-19 13:24]).
Idempotent: refuses if a 2928 title already exists.
"""
import hashlib
import json
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
ROADMAP = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs"
           r"\lpf_multiaxis_gating_roadmap_v1.md")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2928_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2928_survivor_core_anatomy" in m_ids, \
    "run ledger_2928_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2928_survivor_core_anatomy" in l14["connects"]
assert len(L["measurements"]) == 67 and len(l14["connects"]) == 35
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2928:" not in txt, "already appended"

SEC = """
## Phase 2928: 幸存核解剖——重叠无信息修正 + 双强度/响应结构不变性确立 [2026-09-19 13:24]

### 目的（2927 接续候选 A）
2927 报告 7/24 重叠与"不变硬核"。本 Phase 零前向（212 s）回答三问：(a) 重叠是否超机会；(b) 幸存组与丢失组被什么区分（双探针强度假说）；(c) 幸存事件的词响应结构是否探针不变。

### 设计（冻结口径）
- **P1 overlap 零假设校准**：maxT 机器作用于**固定 G 堆栈**（sign-Gram 矩阵由 B 决定，不随置换变），R=100 个独立置换集（rng 2902+i，各 200 置换，向量化 einsum）；每重复抽 E86' 与 Ewd' 计 overlap；p = mean(null >= 7)。保格间相关结构、随机化标签掩码——诚实 null。
- **P2 双强度判别（主检验）**：E17 的 24 事件各算 pct86/pctwd（各自口径下层内 margin 百分位），min_pct = min(两者)；survivor (7) vs lost (17) Mann-Whitney U 单侧 + 置换（rng 2903 x 10000）。
- **P3 响应结构不变性**：每事件 rho_e = Spearman(B86[h,:,l], B_word[h,:,l]) over 57 词；同构 U 检验。
- 锚 4/4：a1 sign_M diff **0.0**；a2 集合重建 24/32/**7**；a3 margin 双值 (1.34634, 1.01827)；a4 cos median 0.1651。

### 结果
- **冻结判决：survivor_core_not_established**（P1 fail 触发 else 分支）——但失败方式本身是本 Phase 最重要的方法论发现。
- **P1 重叠无信息（2927 硬核解读修正）**：overlap null **median 7.0 = 观测 7**（mean 7.22、max 9、p=0.8515）；null 集大小 24-33 与观测同量级；null 显著集强烈聚集在**固定热点层**（L6 47/10 集、L9 31、L8 25、L19 21）——热点层由 Gram 结构（r 模式）决定，两口径共享 → 重叠是 maxT+层热点的机械产物。**显著集重叠不能作为电路不变性证据；2927 "7 事件不变硬核"的选拔统计量错误（数据无误，解读修正）**。
- **P2 双强度通过**：survivor min_pct 中位 **0.9688** vs lost **0.8125**（U=100，p=0.0049）——幸存事件在两个口径下都是层内 top 格。
- **P3 响应结构不变性通过（机制性答案）**：survivor rho 中位 **0.825** vs lost **0.377**（U=104，p=0.0025）——幸存事件的 57 词响应模式跨探针不变，丢失事件的响应向量被探针重写；L+ 类事件破坏最重（(25,3) rho=-0.64、(26,5) -0.47、(17,28) -0.25）。
- **(27,24) 反例取证**：双口径层内第一（pct 1.000/1.000）却丢失——margin_wd 0.560、p_maxTwd **0.0647 差 0.0147 被拒**——双强度是必要不充分条件（maxT 族级判决带随机性）。
- **2918 类不对称（描述性）**：en+ 类 2/3 幸存（(7,19)(8,2)），**L+ 类 0/4 全灭**——词探针（lab0 组 − lab1 组方向）系统性抹除 L+ 事件，方向偏置登记待对照（2929 候选 B）。

### 解读
1. **重叠无信息定律**：两个 maxT 显著集的原始重叠在固定 Gram 结构下是机会水平——族级校正显著集的交叠被层热点结构支配；任何"两个条件下的显著集重叠"论证都必须先过这类 null 校准（与纪律 7 粒度先检同族：先检统计量的零假设分布）。
2. **探针不变性的正确统计量是事件级的**：双口径层内强度（P2）与词响应结构相关（P3）——幸存核的实质 = "双探针下都层内 top 且响应模式不变的格"，其数量恰好 7 个是巧合，机制才是本体。
3. **响应结构 rho 是电路固有坐标的候选**：rho(B86, B_word) 高 = 格的词级读出模式不依赖探针语义混合——2929 将其推广到全格（1152 格 rho 图谱），可望给出"探针不变电路骨架"。
4. (27,24) 的教训：maxT 族级判决对边界事件（p 0.065）的否决是统计机制而非电路性质——边界带事件（p 0.05-0.10）应单独登记。
5. L+ 全灭与 dirs_word 方向语义一致：词探针偏向 en-L 词差方向——探针族的方向偏置是图谱偏置之源（2929 候选 B 对照）。

### 方法论常数（新增）
- **重叠 null 校准**：报告任何"显著集重叠"前必须做固定统计结构 + 重抽样标签的 null 校准（2928 教训，与纪律 7/10 并列）。
- **边界带登记**：族级显著阈值 0.05-0.10 的边界事件单独登记（(27,24) 类），不与硬显著混池。
- **响应结构 rho**：Spearman(B_probe1, B_probe2) per cell 作为探针不变性的事件级/格级统计量——入标配。

### 执行史
- 主脚本一次运行成功（212 s，含 100 次 x 2 方向向量化 maxT；写入期修正 a3 锚口径——1.01827 是词级 margin，双断言）；seal 探针：null 集大小/热点层聚合、(27,24) 取证、2918 类幸存模式。

### 硬伤
- P1 null 保留 Gram 结构但只随机化标签掩码——它 null 掉的是"给定两网格下的 maxT 集合重叠"，不是"两独立电路的重叠"；作为"重叠=不变性证据"的反驳足够，作为重叠的精确期望分布是近似的。
- P2/P3 的 U 检验 n=7 vs 17 小样本（置换 p 精确但功效有限）；P3 的 rho 对 57 词共享词表（两口径同词表，rho 的高基线未校准——survivor rho 0.825 vs null 基线未估）。
- dirs_word 单一构造的方向偏置未对照（候选 B）；n=1 run（锚 bit 级缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2928_survivor_core_anatomy.py: ac0dddbc
- execution.json: 73f92a17（created 2026-09-19T13:24:13）
- result.json: 47b23cc2（final_verdict=survivor_core_not_established，runtime 212 s）
- survivor_core_anatomy.npz: 2db8dfca（overlap_null 100、E17/Ewd_ids、min_pct_rows、rho_rows）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2928/survivor_core_anatomy/
- Ledger：M2928_survivor_core_anatomy 入账，measurements 66->67，L14 connects 34->35，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2928 行完成）

### 接续（2929 候选）
- A（主选）：**全格响应结构图谱**——rho(B86[h,:,l], B_word[h,:,l]) 遍历 1152 格 + 阈值扫描定义"探针不变骨架"（零前向）；rho 与 margin/|phi| 的关系（不变骨架 vs 事件选拔的分离度）。
- B：方向偏置对照——dirs_word 反向组约定或每极平衡对重构（一次前向），检验 L+ 全灭是否方向伪象。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：rho 骨架的跨模型复现（glm4 双口径，一次前向）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2928: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2928 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2928")
