# -*- coding: utf-8 -*-
"""memo_2917_append.py -- append Phase 2917 section to AGI_GPT5_MEMO.md.
Title time = v2 execution created (2026-09-19T10:27:46 -> [2026-09-19 10:27]).
Idempotent: refuses if a 2917 title already exists.
"""
import hashlib
import json
import os
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2917_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2917_event_atlas" in m_ids, "run ledger_2917_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2917_event_atlas" in l14["connects"]
assert len(L["measurements"]) == 56 and len(l14["connects"]) == 24
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2917:" not in txt, "already appended"

SEC = """
## Phase 2917: 全层 36x32（头,层）事件图谱——BH 功效零勘误 + maxT 家族校正 [2026-09-19 10:27]

### 目的（2916 接续候选 A）
把 (头,层) 事件识别从窗口内相对检验升级为全层图谱：一次前向捕获扩展到全部 36 层（dirs = 2886 类间差方向逐层 unit），构建 32x36=1152 格 sign-margin 矩阵（2910 口径），以全头全层为单一 family 做选择校正显著性检验，回答两个预注册问题：(i) 已知 6 事件 (7,19),(27,24),(31,22),(8,23),(7,34),(8,34)（2916 LOO 锁定）是否在全族水平幸存；(ii) 窗口外是否存在 novel / satellite 事件。

### 执行史与勘误（如实入账）
- run1（v1）：per-event BH-FDR q=0.05 over 1152 -> 0/1152 显著，机械判决 event_atlas_not_replicated。
- 审计：v1 判据结构性功效为零，属我的设计失误——N_PERM=200 时最小可达置换 p = 1/201 = 0.004975，而 BH 首阈 q/m = 0.05/1152 = 4.34e-5；BH 仅当 >=115 个事件同时处于粒度地板才可能触发。与 2912 离散 p 校准失败（E[p]=0.5+tau/2 族）同类：置换 p 粒度与 family 大小的匹配必须在任何观测前先验检查。
- 处置：run1 冻结判决标签保留原样（执行史如实登记），判据重冻结为 v2 maxT（Westfall-Young 单步：p_maxT(event) = (1 + #{perms: max_{h,l} sign_M_perm >= sign_M(event)}) / (1+200)，其中 max_perm[pi] 为该置换下全 1152 格的最大值；per-event BH p 值降级为 descriptive），清产物目录重跑。execution.json PREREG.verdict_v2_note 冻结完整自责声明。

### 设计（v2 冻结口径）
SEED=2896，eps=1.0，pos 1，57 词 verbatim 2887；conds same/func/null；一次前向捕获全部 36 层 o_proj 输入；B_heads = einsum('nlhk,lhk->hnl', r_comb, G3) -> (32,57,36)；sign_M (32,36) 逐格 2910 sign-margin（sign 外积 Gram 固定，mask 置换 null，200 perms SEED=2896）；分类 known / satellite（同头 |dl|<=1）/ novel；判决映射 v1/v2 均冻结于 execution.json。

### 锚
- a1：B_heads[:,:,26:36] vs 2913 npz —— rel 3.16e-08，ok=True（连续第五次前向锚定一致：3.88e-8 / 3.16e-8 / 2.73e-8 / 3.16e-8 / 3.16e-8）。
- a2：{7,8} on [26,36) block margin 0.280360 vs 2913 0.28036，absdiff 0.0，ok=True。

### 结果（v2 maxT，runtime 32.3 s，qwen3-4b）
- 全族显著事件 24/1152（q=0.05）；known 幸存 2/6，satellite 0，novel 22。
- 判决：event_atlas_not_replicated（冻结映射要求 n_known_sig>=3 才算 partially_replicated，2/6 落 else 分支）。

known 6 事件明细：
| 事件 | margin | p_maxT | 判定 |
|---|---|---|---|
| (7,19) | 1.34634 | 0.0050 | 全族显著（全 1152 格第一名） |
| (27,24) | 0.81057 | 0.0100 | 全族显著 |
| (31,22) | 0.45609 | 0.1891 | 不显著 |
| (8,23) | 0.17081 | 1.0000 | 不显著 |
| (7,34) | 0.18701 | 1.0000 | 不显著 |
| (8,34) | 0.29734 | 0.8955 | 不显著 |

24 个显著事件（margin 降序，*=known）：
(7,19) 1.34634* | (26,6) 1.23584 | (8,2) 1.12525 | (25,3) 1.12278 | (22,12) 1.11380 | (24,23) 1.01633 | (21,6) 0.91948 | (13,22) 0.82254 | (5,6) 0.81462 | (6,19) 0.81057 | (27,24) 0.81057* | (4,1) 0.80581 | (21,16) 0.73379 | (4,22) 0.72903 | (20,8) 0.72604 | (1,6) 0.64759 | (4,19) 0.64495 | (14,9) 0.63896 | (1,4) 0.56835 | (15,13) 0.56817 | (17,28) 0.56817 | (26,5) 0.56659 | (28,1) 0.56659 | (24,9) 0.56571
（层 <=16 者占 16/24；(6,19) 与 (27,24) margin 同为 0.81057 是 57 词 sign-margin 离散化的并列，非笔误。）

### 解读（三点，诚实口径）
1. 判决 not_replicated 不等于"事件不存在"：known 2/6 全族幸存——(7,19) 是全图谱第一名（1.34634，与 2916 LOO 的 sign peak L19 1.3463 一致），(27,24) 亦全族显著；2916 锁定的两大单层驱动事件在最强校正下为真。
2. 后段窗口 4 事件 (31,22)/(8,23)/(7,34)/(8,34) 全族不显著：它们是"窗口族相对"事件——在约 32 头 x 单窗口的小 family 内显著，放到全 1152 格 family 即消失。2915 的层段绑定结构在窗口内成立，但不构成全族水平的独立事件证据。
3. novel 22 个事件集中于中早层（L1-L16 占 16/24）：h26@L6 1.23584、h8@L2 1.12525、h25@L3 1.12278、h22@L12 1.11380 等。窗口受限扫描（2915 只测 [16,36)）从未检验过早层——早层存在一族全族水平的 lang-diff (头,层) 事件，是图谱的净新增结构。

### 方法论常数（新增）
- 置换检验 p 粒度（1/(N_PERM+1)）x family 大小（BH 首阈 q/m）的匹配必须在任何观测前先验检查；大 family + 小置换数时 maxT（Westfall-Young）是正确选择，per-event BH 会让整个判据功效机械归零。
- "窗口内显著"与"全族显著"是两个不同强度的命题：maxT 家族校正下报告显著性必须注明 family 口径。

### 硬伤
- N_PERM=200 -> maxT p 地板 1/201 = 0.004975：6 个地板事件无法进一步排序，更细粒度需 N_PERM>=2000。
- sign_M 为单侧符号口径（仅正号 margin），负向事件不可见。
- 单一 run B（n=1 前向），novel 22 事件未做跨 run 复现；null 为 run 内 mask 置换，非 run 级重采样。
- dirs 仅 lang 类间差方向族；其他语义轴（size/moisture/speed 等）的全层图谱未测。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2917_event_atlas.py（v2）: 30e0ccbb
- execution.json: b548239f（created 2026-09-19T10:27:46；PREREG 含 verdict_v2_note 勘误声明）
- result.json: 1f988c40（final_verdict=event_atlas_not_replicated，runtime 32.3 s）
- event_atlas.npz: 02343146（B_heads, sign_M, p_M, p_maxT, max_perm, dirs, labels_lang, words）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2917/event_atlas/
- Ledger：M2917_event_atlas 入账，measurements 55->56，L14 connects 23->24，ledger sha256-8 = """ + LEDGER_SHA + """

### 接续（2918 候选）
- A：maxT 显著 novel 事件的中早层结构解剖——h26@L6 / h8@L2 / h25@L3 / h22@L12 的逐词 sign 贡献与响应曲线，判定它们与后段 L19/L24 事件是否同一机制的前段形态。
- B：窗口族 vs 全族显著性的形式化——两水平检验框架（window-family maxT + full-family maxT 双报告），把 2915/2917 的 family 口径差异变成可预注册判据。
- C：glm4-9b 跨模型 (头,层) 事件检验——2917 协议 verbatim 移植，检验早层事件簇与 (7,19) 是否跨模型存在。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

# verify on disk
with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2917: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2917 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s" % (n_before, n_after, bool(m), LEDGER_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2917")
