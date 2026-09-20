# -*- coding: utf-8 -*-
"""memo_2920_append.py -- append Phase 2920 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T11:45:25 -> [2026-09-19 11:45]).
Idempotent: refuses if a 2920 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2920_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2920_multiaxis_word_atlas" in m_ids, \
    "run ledger_2920_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2920_multiaxis_word_atlas" in l14["connects"]
assert len(L["measurements"]) == 59 and len(l14["connects"]) == 27
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2920:" not in txt, "already appended"

SEC = """
## Phase 2920: 多轴词级事件图谱（5 轴划分 x 4 Jacobian 族）+ 跨协议迁移检验 [2026-09-19 11:45]

### 目的（2919 接续候选 A+B / 纲领 afa60f18 §4 2920 行）
把 2917 (头,层) 事件图谱协议从单一语言轴推广到多轴：per-axis 事件集、跨轴共现（通道复用）、以及 2919 句子级方向族与词级方向的迁移检验。forward 协议（55.1 s，qwen3-4b，110 单 token 词）。

### 设计（冻结口径）
- 110 词 = lang 2887 verbatim 57 + 属性形容词 53（tokenizer 双形式筛查，12 词设计期排除：hasty/leisurely/plodding/unhurried/diminutive/soggy/drenched/arid/parched/dehydrated/withered/waterless）。
- 4 Jacobian 族：族 0 = lang+concept 共享探针（注入 dirs = 2886 lang 类间差，2917 verbatim float64 推导；同一 B 矩阵施加两个划分——同探针双划分设计）；族 1/2/3 = speed/size/moist（注入 dirs = 2919 dirs_all[1..3]）。
- 5 划分：lang（labels_lang 0=en,1=L）、concept（labels_concept，2887，22 类，== ck）、speed/size/moist（pole 1=HIGH fast/huge/wet，2919 极性约定）。
- conds same/func/null verbatim（lang 词 = 2917 same-language min-tid 上下文；属性词 = same-axis min-tid 上下文）；null tids 两段 rng(2896)（段 1 = 2917 verbatim 57 抽样）。
- 统计：sign-Gram margin（2910 口径）；每轴 200 次标签置换（fresh default_rng(2896)，lang 轴置换序列与 2917 逐位一致）；per-axis maxT（1152 格族，q=0.05）PRIMARY + joint 5x1152 maxT SECONDARY descriptive；p 地板先检 1/201 = 0.004975 << 0.05（纪律 7）。

### 锚（4/4，全逐位）
- a0：dirs2919[lang] vs 2886 推导 rel 2.86e-08。
- a1：B_lang fresh vs 2917 npz rel 3.53e-09。
- a2：m78 0.280360（== RECON_REF 0.28036）。
- a3：sign_M diff 4.71e-08 + 显著集 **24/24 集合相等**——lang 图谱第五次连续前向锚定，2917 完整复现。

### 结果
- **判决：nonlang_events_absent**。
- P1：lang 24（top 与 2917 逐位一致：(7,19) 1.34634、(26,6) 1.23584、(8,2) 1.12525）；**concept 0**；speed 0；size 0；moist 0。
- P2：跨轴共现**全空**（唯一非空显著集 = lang；零共享 cell）——多轴通道复用在词级 maxT 口径下无对象。
- P3 迁移 4/4 failed：median |cos| L1-35 = lang 0.166 / speed 0.237 / size 0.222 / moist 0.084（峰 |cos|：lang 0.283@L23、speed 0.322@L19、size 0.373@L9、moist 0.185@L9）。
- **功效诊断（post-hoc 零前向，descriptive）**：concept（n=57，与 lang 同词表同探针同前向 = 同功效）top margin 0.487 **低于其 null 中位 0.496**（124/200 perm >= top）-> **真 absence**；属性轴（n=15-21）margin 粒度化（size top-40 仅 9 个不同值）且 permuted null 常规到达观测 top（speed null p50 0.9709 == 观测 top；moist 100/200）-> **小词表功效失效**，0 事件 ≠ 事件不存在；size 最接近：(18,7) margin 1.5312、p_maxT 0.0796（15/200），距 0.05 一步。

### 解读
1. **事件图谱是划分相对的（partition-relative）**：同一响应矩阵 B 上，en/L 划分给出 24 事件、concept 划分给出 0 事件（功效充分）——(头,层) 事件语言当前只对"语言身份"划分成立。结合 2887（概念 token ID 跨语言共享）：概念组织不进 lang-探针的符号响应结构——**语言身份与概念内容在事件层解离**。
2. **功能迁移 ≠ 几何同一**：迁移 4/4 failed 但 2913-2918 已证明句子 dirs 在词位置因果有效——注入方向是"测量仪器"而非"编码方向"；2919 句子 dirs 不应默认当作词级属性编码方向。属性轴事件存在性问题回到两个自由度：功效（n）与探针选择。
3. 纲领修正：2920 预期"多轴图谱+共现矩阵"，实际产出负结果+功效诊断——复用机制分析（roadmap 2921 主角度）暂无跨轴目标，退回 lang 内部复用（h4 L1<->L19）。

### 方法论常数（新增）
- **小词表（n <~ 25）下 sign-Gram margin 粒度化使 maxT 失效**：margin 值域被组合学量化（size top-40 仅 9 个值），permuted null 常规到达观测 top——词级事件检验要求 n >~ 40，或改用幅度敏感统计；预注册时必须做 margin 粒度先检（与纪律 7 的 p 粒度检查并列）。
- 迁移检验三档（median |cos| >= 0.5 ok / 0.3-0.5 weak / < 0.3 failed）首次使用；4/4 failed 促成"因果有效性"与"几何同一性"的命题分离。

### 执行史
- run1 KeyError（属性词未登记 tid_map）修复重跑；run2 p_joint 广播错误（(200,) vs (32,36)）修复重跑；run3 通过（55.1 s）。每次改脚本先删产物目录（rm shim 劣化 -> python shutil.rmtree + Glob 复核）。
- 设计期探针 2 轮：2887 labels_concept == ck 确认（22 类：house/maison/Haus/casa 等）；12 候选词非单 token 排除；2917 锚风险预检（min sig margin 0.5657 vs max nonsig 0.5604，gap 5.3e-3）。

### 硬伤
- 属性轴 0 事件受小 n 功效限制（已量化）——不能解读为"属性轴无 (头,层) 事件"。
- sign 口径丢弃幅度结构（2918 已登记；concept 的"真 absence"仅指符号响应结构）。
- 概念轴只测了 lang-探针响应；概念差方向等其他探针未测——探针空间未穷尽。
- 迁移 dirs_word 用 func-cond（'the X'）上下文；句内上下文的词级方向未测。
- n=1 run（逐位锚缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2920_multiaxis_word_atlas.py: b96f3a19
- execution.json: 598764fe（created 2026-09-19T11:45:25）
- result.json: bf96339b（final_verdict=nonlang_events_absent，runtime 55.1 s）
- multiaxis_word_atlas.npz: b037bdb1（B x4 族、sign_M/p_M/p_maxT/p_joint x5 轴、max_perm+global_max_perm、dirs_used/dirs_word/cos_curve、词表+标签）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2920/multiaxis_word_atlas/
- Ledger：M2920_multiaxis_word_atlas 入账，measurements 58->59，L14 connects 26->27，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2920 行完成——判决偏离路线预期，负结果+功效诊断按纪律如实入账）

### 接续（2921 候选）
- A（主选）：**属性轴词表扩容复测**——speed/size/moist 各扩至 40-60 词（tokenizer 筛查 + margin 粒度先检），同 2920 冻结统计重测属性轴（lang 24 锚复验）；直接回答属性事件存在性。
- B：**探针相对性检验**——lang 词表用词级 lang 方向（dirs_word[lang] 或 2887 lang_dir）作探针重跑 lang 轴事件检验：事件集是否随探针改变（检验"事件=探针不变量"还是"事件=探针相对"）。
- C（roadmap 2921 原项，零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道）——跨轴复用无对象，退回 lang 内部复用解剖。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2920: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2920 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2920")
