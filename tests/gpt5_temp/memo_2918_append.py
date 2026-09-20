# -*- coding: utf-8 -*-
"""memo_2918_append.py -- append Phase 2918 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T10:50:29 -> [2026-09-19 10:50]).
Idempotent: refuses if a 2918 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2918_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2918_event_anatomy" in m_ids, "run ledger_2918_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2918_event_anatomy" in l14["connects"]
assert len(L["measurements"]) == 57 and len(l14["connects"]) == 25
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2918:" not in txt, "already appended"

SEC = """
## Phase 2918: （头,层）事件解剖——极性交替链与三层密度门控 [2026-09-19 10:50]

### 目的（2917 接续候选 A）
判定 novel 早层事件与后段 known 事件 (7,19)/(27,24) 是同一机制的前段形态还是独立机制族；产出事件解剖（词位分解、稀疏度、时间曲线、头复用）作为密度门控操作化的实证基线。零前向产物域分析（runtime 2.3 s），对象 = 2917 npz。

### 设计（冻结口径）
联结统计 = 57 维词位响应向量 B_heads[h,:,l] 的 Spearman rho；null = 2000 次层距匹配的非显著 cell 对抽样；maxT 家族校正双族：276 对全事件族（图）+ 10-slot focus×known 族（判决 p_link）；同层 (d=0) 事件对存在（(26,6)/(21,6)/(5,6)/(1,6) 同在 L6）纳入族与 null（ha≠hb）；粒度先检：maxT p 地板 1/2001=0.0005 << 0.05，无 BH（2917 教训观测前应用）。focus = margin 前 5 novel：(26,6),(8,2),(25,3),(22,12),(24,23)；early novel = novel 且层 <=16（16/22）。

### 锚（3/3）
- a1：sign_M 由 npz B_heads 按 2917 公式重算——中位差 8.56e-10、最大差 4.71e-08、0 格 >0.02（公式复制逐位一致，无 fp32 符号翻转）。
- a2：(7,19) 为全矩阵 argmax、margin 1.346335、p=0.004975。
- registry：npz 重算显著集 == 24 == 2917 P1 n_sig。

### 结果
- **判决：early_events_same_mechanism**（n_linked 5/5；p_link = 0.0005 / 0.0180 / 0.0035 / 0.0085 / 0.0085）。|rho| 对 (7,19)/(27,24) = 0.42–0.72（均值 0.542）；early 内部 rho 均值 0.612。
- **核心发现——极性结构**：全部 24 个事件共享同一个 en-vs-非en 词位响应模式，分两个极性类：en+ {(8,2),(22,12),(7,19)}、L+ {(26,6),(25,3),(24,23),(27,24)}；同极性 +0.45…+0.72、异极性 −0.42…−0.60；**极性随深度交替**：L2 en+ -> L3 L+ -> L6 L+ -> L12 en+ -> L19 en+ -> L23 L+ -> L24 L+。类纯度逐事件不同（lang_align：(26,6) en 侧 1.0；(22,12) L 侧 0.943；(27,24) L 侧 0.886/en 侧 0.273）。
- **三层密度（P1/P3）**：词位层 DENSE——强事件 PR_word 37.6–39.7/57（同层 null 79–97 百分位，比基线更密），top-3 词仅承载 14–31% |r| 质量；弱事件更稀（(24,23) PR 20.9、6.5 百分位）。层位层 SHARP——temporal PR 6.3–10.3/36、相邻对比 0.76–1.15。头层 SPARSE——24/1152（2.1%）。幅度不敏感：早期事件幅度比 (7,19) 小 5–10 倍（~0.002–0.02 vs ~0.03–0.11），sign-Gram 仍检出——门控判据是符号对齐密度而非幅度。
- **头复用（P4）**：头内跨层词位模式 7/8 低于 null95（去相关或反相关）：h26 L5 vs L6 −0.620（相邻层极性翻转）、h4 L19 vs L22 −0.479、h21 L6 vs L16 −0.345、h7 L19 vs L34 −0.377；唯一真复用 h4 L1<->L19 +0.447 > null95 0.245。头通道是"被复用的导管"，不是"可复用的模式"。
- **已登记的 family 警告**：24x24 联结图在更严的 276 对 maxT 族下 0 边——联结过 10-slot 族不过 276 族；跨 cell 基线相关高（单对 null95 0.24–0.27），事件联结是中等效应（|rho| 0.42–0.72），只在预注册小族可检。与 2917"窗口族 vs 全族"同类现象。

### 解读
1. 判决含义：早层事件 = 同一语言分离机制的**前段形态**——不是独立机制，也不是静态载体的延伸，而是**极性交替的事件链**。2916 的"载体=(头,层)事件"与 2918 的"单模式双极性"合并为：语言轴沿深度被一系列 (头,层) 事件反复重表达，每次极性可选。
2. 对密度门控的直接证据：门控不在词位层收窄（宽门），而在（头,层）选择上收窄（稀门）+ 层位上尖锐（尖门）；幅度不是门控变量。
3. 相邻层同头极性翻转（h26 L5/L6 −0.620）提示"候选竞争"式交替：同一通道在相邻层携带相反取向的同一模式。

### 方法论常数（新增）
- 跨事件联结检验必须双 family 口径并行报告（小族预注册 + 大族 maxT）；单对 null95 与 family max 是两个门槛。
- sign-Gram 口径下"稀疏 vs 致密"可直接用 PR_word + 同层 null 百分位量化；结论：语言轴事件是致密宽门。

### 硬伤
- 276 族 0 边：联结结构效应量中等且 family 依赖；跨 run 复现未做（n=1 前向继承）。
- sign 口径丢弃幅度结构（5–10 倍差未入统计）。
- 词表 57、语言标签二类（en/L）；极性参照模式未预注册（由 data 极性类定义，属描述性）。
- rho_known 取 max 吸收了极性翻转——"同一机制"判据对全局符号翻转不变，若两极类实为两个反相机制则判决不变（信息等价），但解释已按极性链口径给出。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2918_event_anatomy.py: 15c545ab
- execution.json: 1d67c3c2（created 2026-09-19T10:50:29）
- result.json: d57378eb（final_verdict=early_events_same_mechanism，runtime 2.3 s）
- event_anatomy.npz: e5f29e4d（word_r, rho24, p_pair24, null_all, max_null276/10, 曲线头, 词表）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2918/event_anatomy/
- Ledger：M2918_event_anatomy 入账，measurements 56->57，L14 connects 24->25，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领文档 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（多轴坐标系 x 单通道多子空间 x 密度门控：思路评估 + LPF v6 形式化 + 2919–2925 路线）

### 接续（2919 候选）
- A（主选，对应纲领 2919）：多轴方向族构建——speed/size/moisture 轴逐层类间差方向 + 轴间共线先验检查。
- B（对应纲领 2923，可部分零前向）：极性交替形式化——极性翻转层 vs 方向族层间符号不稳定层（dirs 相邻层余弦）的重合率。
- C（对应纲领 2921 前置）：复用解剖——h4 L1<->L19 真复用的子空间主角度分析。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2918: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2918 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2918")
