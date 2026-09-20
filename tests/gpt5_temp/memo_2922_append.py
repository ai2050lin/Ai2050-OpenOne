# -*- coding: utf-8 -*-
"""memo_2922_append.py -- append Phase 2922 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T12:26:19 -> [2026-09-19 12:26]).
Idempotent: refuses if a 2922 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2922_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2922_attr_event_anatomy" in m_ids, \
    "run ledger_2922_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2922_attr_event_anatomy" in l14["connects"]
assert len(L["measurements"]) == 61 and len(l14["connects"]) == 29
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2922:" not in txt, "already appended"

SEC = """
## Phase 2922: 属性事件解剖——attr_events_linked_polar [2026-09-19 12:26]

### 目的（2921 接续候选 A / 纲领 afa60f18 §4 2922 行）
2921 发现 13 个属性 (头,层) 事件（speed 2 / size 9 / moist 2）。本 Phase 用 2918 协议 verbatim 对其做零前向解剖：词级分解、极性对齐、稀疏度、时间曲线、轴内联动（Spearman rho + layer-distance-matched null maxT），并与 24 个 lang 事件对比。zero-forward（1.1 s，artifact domain）。

### 设计（冻结口径）
- 分析单位：词响应向量 r_e = B_ax[h, :, l]（N_ax 维：speed 43 / size 48 / moist 35）；输入 2921 npz + 2917 npz（P5 对比）。
- 联动：同轴事件对 Spearman rho；null = 2000 次 layer-distance-matched non-sig 格对（各轴各自 non-sig 集，预构造层对池——2918 verbatim）；maxT over 合并 38 对族（speed 1 + size 36 + moist 1），q=0.05；p 地板 1/2001 = 0.0005 << 0.05，无 BH（粒度先检）。
- 极性对齐（新增正式检验）：p_pole = 2000 次保组大小 pole 标签置换（fresh default_rng(2897) 独立流）下 max(|m_hi|,|m_lo|) >= 观测的比例；p_pole <= 0.05 判 pole-aligned；d_pole = (mean B[hi] - mean B[lo]) / pooled_std。
- P5：lang 24 事件同 P1 指标（lang_align 替代 pole_align）对比表。

### 锚（3/3，全过）
- a1：每属性轴从 npz B 重算 sign-Gram margin（pole 划分）vs npz sign_M——median absdiff ~6e-10、max = 0、零格 > 0.02（2918 a1 口径全过且远优于容差）。
- a2：registry——sig 集大小 2/9/2（lang 24）；每轴 top1 格与 2921 P1 top15[0] 一致且 margin 差 < 1e-4。
- a3：(7,19) 仍是 2917 sign_M argmax，margin 1.346335，p 0.004975（2918 a2 verbatim）。

### 结果
- **判决：attr_events_linked_polar**。
- P1 极性：**13/13 事件 p_pole = 0.0005（地板）**——全部属性事件按极性清晰分离词表；|d_pole| 0.67-1.84（中位 1.086）。**d_pole 符号分裂：8/13 LOW 驱动**（lo 词响应均值更高：size top (21,7) d=-1.84、moist 两个全 LOW (8,15) -1.60/(10,9) -0.67、(18,7) -1.44、(22,2) -1.20、(25,15) -1.09、(14,12) -0.87、(11,27) -0.79）；5/13 HIGH 驱动（(18,16) +1.36、(12,18) +1.27、(24,19) +0.92、(11,23) +0.84、(26,21) +0.77）。
- P1 稀疏度/时间：PR_word 11-33、top3_mass 0.16-0.42、pr_pct_vs_layer_null 0-0.93；(14,12) 最集中（PR 16.7、top3 0.318）。
- P2 联动：**n_linked_all 4/38，全部在 size**：(21,7)-(18,7) rho 0.676 p 0.0295、(21,7)-(25,15) 0.571 p 0.0345、(11,27)-(18,7) 0.571 p 0.0345、(24,19)-(12,18) 0.572 p 0.0345；size 分量 = {(21,7),(25,15),(18,7),(11,27)} 4 事件分量 + {(24,19),(12,18)} 对 + 3 孤立；**speed/moist 事件对均不联动**（rho 远低于 null95）——size 是唯一内部连贯的属性轴。
- P3 层分布：属性事件 median peak **L15 vs lang L6**——属性事件系统性偏深层（lang 24 事件 16/24 在 L1-L16，属性 13 事件峰跨 L2-L23）。
- P4 跨类别同头：**8 个头承载 >1 类别事件**：h21（lang (21,6)/(21,16) + size (21,7)）、h18（speed (18,16) + size (18,7)）、h26（lang x2 + size (21 层)）、h8/h14/h22/h24/h25 各一对——头共享而格私有（2921 P2 shared=0），**类别私有性是共享头内的层级私有**。
- P5 对比：lang vs attr——PR_word 中位 30.6 vs 24.7、top3_mass 0.196 vs 0.207、pr_pct 0.582 vs 0.484、peak 6 vs 15、pr_time 9.3 vs 7.1、n_pos_layers 18 vs 16、对齐 |mean sign| 0.794 vs 0.826——同一宽响应机制形态，属性事件更集中、更深层。

### 解读
1. **属性事件是极性结构化的（polarity-structured）**：13/13 以 p=0.0005 地板分离 hi/lo 词——属性 (头,层) 事件不是无方向噪声，而是对 pole 维度的读出。与 2919 dirs（HIGH-LOW 差方向）注入有效、2917-2921 锚链共同构成属性轴证据闭环。
2. **极性符号事件特异**：8/13 LOW 驱动 vs 5/13 HIGH 驱动，且 size top 事件 (21,7) 是最强 LOW 驱动（d=-1.84）——margin 高低与"哪端驱动"无关；属性读出不是单向 HIGH 检测器，逐事件极性符号是自由参数（2923 解剖对象）。
3. **size 唯一内部连贯**：4/36 边全在 size，含一个 4 事件分量——size 9 事件非独立抽样，部分共享词响应模式；speed/moist 事件对不联动（各自孤立）。与 2921 "size 最强"一致。
4. **层深分离**：属性事件 median peak L15 vs lang L6——语言身份读出在浅层，属性读出在深层；与 2921 P3 迁移 argmax（属性 L12、lang L23）部分呼应。
5. 头共享/格私有：8 头跨类别——通道私有性（2921）在头级放宽、在格级保持；跨类别复用发生在头内不同层。
6. 理论对应：属性事件的极性结构与符号自由度支持 Cmp(o,r,v) 类"候选竞争"图景——同一头可承载多类别候选，格级选择决定通道。

### 方法论常数（新增）
- **极性对齐置换检验**（保组大小标签置换、独立 rng 流 2897、p 地板 1/2001）正式入协议——词级事件解剖标配。
- **同头跨类别登记**（head shared / cell private 描述口径）入协议。
- 层对池死区教训：d >= NL/2 时 la±d 存在双向越界死区，null 抽样必须预构造 valid_lp[d] 池（2918 verbatim 结构），不得 inline 翻转（run1 KeyError -7 根因）。

### 执行史
- run1 KeyError -7（层对死区，inline 翻转逻辑缺陷）-> 修复为 valid_lp 池；run2 P4 tuple(int) 笔误 -> 修复；run3 通过（1.1 s）。每次改脚本先删产物目录（python shutil.rmtree + 复核）。
- 零前向：无模型加载，锚从 npz 自洽复算。

### 硬伤
- speed/moist 事件对联动检验 n=1 对/轴——无统计功效可言，"不联动"仅指该单对。
- 极性符号（LOW vs HIGH 驱动）是观测后描述，未预注册方向假设——2923 需预注册检验。
- 联动 null 复用 38 对合并族（保守）；per-axis 族功效未单独评估。
- lang 对比（P5）为跨 npz 描述性（2917 vs 2921 产物），非同前向。
- n=1 run（npz 自洽锚缓解）；单模型。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2922_attr_event_anatomy.py: c9328ca9
- execution.json: 1e926972（created 2026-09-19T12:26:19）
- result.json: 629f94c7（final_verdict=attr_events_linked_polar，runtime 1.1 s）
- attr_event_anatomy.npz: 5ec4ec1a（word_r_speed/size/moist、event_ids、rho_obs38/p_pair38/null_all38/d_s38、pole_p13、lang_event_ids、sign_M_ref/sign17_ref）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2922/attr_event_anatomy/
- Ledger：M2922_attr_event_anatomy 入账，measurements 60->61，L14 connects 28->29，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2922 行完成）

### 接续（2923 候选）
- A（主选）：**极性符号结构解剖**——LOW/HIGH 驱动由什么决定：预注册检验（如 d_pole 符号 vs 事件格 o_proj 输入侧词响应 top 词的极性构成；hi/lo 响应不对称与 2919 dirs 极性约定的关系），零前向 + 可选一次受控前向验证。
- B：**探针相对性检验**——lang 词表换词级 lang 探针（dirs_word，2921 npz 已存）重测事件集（需一次前向，~1 min）。
- C（零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道，roadmap 遗留项）。
- D：**size 联动分量验证**——4 条 size 边的词响应 top 词重叠分析（零前向，cheap；分量 {(21,7),(25,15),(18,7),(11,27)} 共享哪些词）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2922: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2922 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2922")
