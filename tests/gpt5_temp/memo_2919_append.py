# -*- coding: utf-8 -*-
"""memo_2919_append.py -- append Phase 2919 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T11:16:00 -> [2026-09-19 11:16]).
Idempotent: refuses if a 2919 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2919_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2919_multiaxis_families" in m_ids, "run ledger_2919_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2919_multiaxis_families" in l14["connects"]
assert len(L["measurements"]) == 58 and len(l14["connects"]) == 26
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2919:" not in txt, "already appended"

SEC = """
## Phase 2919: 多轴方向族构建（speed/size/moisture）+ 共线性审计 [2026-09-19 11:16]

### 目的（2918 接续候选 A / 纲领 afa60f18 §4 2919 行）
为 2920 多轴事件图谱构建属性轴逐层方向族并做共线先验检查：speed/size/moisture 三轴在 2886 last-token 口径下的类间差方向 dirs(axis, layer)，判定 (a) 各轴方向族质量是否达到图谱注入门槛、(b) 轴间共线是否需要残差化。forward 协议（17.9 s，qwen3-4b，200 冻结句）。

### 设计（冻结口径）
- 200 冻结句 = lang 2886 verbatim 80 + speed/size/moisture 各 20 同主题对（模板 "The {s} is fast./slow." / huge./tiny. / wet./dry.）；**同主题对设计**：每轴 20 个主题各出一对 HIGH/LOW 句，主题效应在类间差均值 dirs = unit(mean HIGH - mean LOW) 中**精确相消**——dirs 是纯类方向，无主题污染。
- rows [0,36) input-to-block 口径（与 2917/2918 消费的推导一致）。
- P1 质量：每轴逐层 LOO nearest-centroid acc（lang 锚 = 2886 S4 verbatim cosine NC；属性轴 = 同轴 20 对 LOO）；n_ready 只数 3 属性轴（lang 为锚不计入）。
- P2 共线：6 对 x 36 层 |cos| 矩阵，判据 global max |cos| < 0.5。
- P3：类间差 diff norm 深度剖面。

### 锚（3/3，全逐位）
- a1：重算 80 条 lang 句 S_last vs 2886 npz S_last，rel 0.00e+00（前向逐位确定）。
- a2：dirs_lang rows [0,36) vs 2917 消费的推导，rel 0.00e+00。
- BONUS 跨 Phase 交叉：重算 lang LOO 探针曲线与 2886 存储曲线**逐位一致**。

### 结果
- **判决：multiaxis_families_ready**（n_ready_attr = 3/3）。
- P1 per-axis best LOO acc：lang 1.0@L1、speed 1.0@L1、size 1.0@L1、moist 1.0@L1——门槛早层平凡通过，信息量在**深度剖面**。
- **核心发现——轴的深度剖面分化**：lang 全程 ~1.0 可读（与 2886 登记的 hourglass_validated=False 一致：无 probe dip）；speed 深层衰减 1.0 -> 0.65（L31–35）；size 轻度衰减 -> 0.875；moist ~0.95–1.0 保持。**属性语义深层被整合 away、语言身份持续**——轴有不同深度剖面；与 2918 极性链（语言事件集中早层 L1–L16）互洽。
- P2 共线：global max |cos| = 0.4472 @ speed-moist L32；attr-attr 0.4472、lang-attr 0.1274 -> **2920 无需残差化**。
- P3 diff norm 深度超线性增长：lang 320.7 vs 属性 78.7–94.0 @L35——语言主导末 token 类间差。
- **退化行登记**：layer-0 行精确为 0——末 token = 句号 "." 嵌入逐句相同 -> dirs[0] = 0 向量；解释 2917/2918 曲线 sign_M[:,0] = 0；rows 1..35 substantive。

### 解读
1. 纲领 2919 判据达成：四轴方向族就绪 + 共线低 -> 2920 多轴事件图谱可直接用 dirs，不需残差化。
2. 深度剖面分化 = 纲领"不同类别有各自路径"的第一个量化形态：语言轴"全程可读"型 vs 属性轴"早层峰值/深层衰减"型（speed 最陡、moist 最平、size 居中）——三类属性轴之间也有分化，非单一属性原型。
3. 同主题对设计是属性轴方向族的方法学基座：主题（被修饰名词）是句子间最大变异源，类间差均值中精确相消后才得到纯类方向。

### 方法论常数（新增）
- 方向族构建必须先验共线检查（global max |cos| 阈值），通过后才允许图谱化；本次 0.4472 < 0.5 通过。
- 末 token 为标点的句子协议下 layer-0 类间差恒为 0——跨 Phase 对比必须排除 row 0 或显式登记（2917/2918 的 sign_M[:,0]=0 由此得解，入账为跨 Phase 自洽锚）。

### 执行史
- run1 IndexError（boolean mask 80 vs 200：Sa=S[:80] 误用 (200,) 全局掩码）-> 修复 aid80/cls80 切片，删产物重跑通过。
- run 前自查修复：删除遗留占位死循环（每层重复计算 80x37x2560 mean）；n_ready 误含 lang 锚轴 -> 改 best[1:] 只数 3 属性轴。
- **2886 交叉核实**：记忆中"2886 hourglass 已验证"为错误——2886 result.json 判决原文 hourglass_validated=False（H1=False H2=False，probe mid 1.0 vs ends 0.915，peak CKA 0.839@L12 未达阈值）；重算曲线与存储曲线逐位一致 -> 跨 Phase 一致性锚，非数据矛盾。

### 硬伤
- 属性 dirs 来自句子上下文（"The {s} is {attr}."），2920 注入单 token/单词上下文——**跨协议迁移是 2920 显式检查项**（lang 先例：2886 句子 dirs 在 2913–2918 单 token 注入中有效，但属性轴无此先例）。
- n=1 run；前向逐位确定（a1/a2 rel 0.0）缓解但不消除设计单点。
- LOO NC acc 上限 1.0 -> 早层饱和，深度剖面分辨率依赖 acc 下降段；20 主题/轴，类间差均值的抽样噪声未做 bootstrap。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2919_multiaxis_direction_families.py: 2df1ad3b
- execution.json: 308bfc37（created 2026-09-19T11:16:00）
- result.json: a5381b06（final_verdict=multiaxis_families_ready，runtime 17.9 s）
- multiaxis_direction_families.npz: 3df331fe（dirs_all 4x36x2560、probe 曲线、cos 矩阵、diff norms、冻结句清单）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2919/multiaxis_direction_families/
- Ledger：M2919_multiaxis_families 入账，measurements 57->58，L14 connects 25->26，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（本 Phase = 纲领 §4 2919 行完成）

### 接续（2920 候选）
- A（主选，对应纲领 2920）：多轴事件图谱——2917 协议 x 4 轴（per-axis 词表标注：2887 57 词 + 概念类 token ID（2887 发现 labels_concept 跨语言共享 ~22 类，如 3753=house/Haus）+ 400b aligns），产出 A[a,h,l] 事件张量 + 跨轴共现矩阵；dirs 已就绪且无需残差化。
- B（便宜 de-risk，可先行或并入 A）：跨协议迁移检验——sentence dirs vs 2-token single-word dirs 逐层 cos 对比。
- C（对应纲领 2923）：极性交替形式化——2918 极性翻转层 vs dirs 相邻层符号不稳定层的重合率（部分零前向）。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2919: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2919 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2919")
