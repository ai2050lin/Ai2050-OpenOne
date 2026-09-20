# -*- coding: utf-8 -*-
"""memo_2921_append.py -- append Phase 2921 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T12:08:29 -> [2026-09-19 12:08]).
Idempotent: refuses if a 2921 title already exists.
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
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2921_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2921_attr_vocab_expansion" in m_ids, \
    "run ledger_2921_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2921_attr_vocab_expansion" in l14["connects"]
assert len(L["measurements"]) == 60 and len(l14["connects"]) == 28
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2921:" not in txt, "already appended"

SEC = """
## Phase 2921: 属性轴词表扩容复测——attribute_events_found [2026-09-19 12:08]

### 目的（2920 接续候选 A / 纲领 afa60f18 §4 2921 行）
2920 判决 nonlang_events_absent 中属性轴 0 事件被诊断小 n 功效失效（纪律 8）。本 Phase 把 speed/size/moist 词表扩到 lang 量级（n 35-48），同 2920 冻结统计复测，直接回答属性 (头,层) 事件存在性。forward 协议（79.8 s，qwen3-4b）。

### 设计（冻结口径）
- 183 词 = lang 2887 verbatim 57 + speed 43（21 hi / 22 lo）+ size 48（25 hi / 23 lo）+ moist 35（19 hi / 16 lo）；tokenizer 两轮筛查（spaced 单 token 优先、bare 兜底；moist 第一轮仅 25 词——wet/dry 系大量多 token，第二轮补 sweat/lush/sprayed/watering/raining/dipped/gritty/baking/roasted/parch 等达 35）。
- 4 Jacobian 族：lang 探针 = 2886 类间差 2917 verbatim；speed/size/moist 注入 2919 dirs_all[1..3]；concept 轴 2920 已结案（真 absence），剔除。
- 4 划分（pole 1 = HIGH fast/huge/wet，2919 极性约定）；conds same/func/null verbatim；null tids 两段 rng(2896)（段 1 = 2917 57 抽样，段 2 排除全 183 词 tid）。
- 统计：sign-Gram margin（2910 口径）+ per-axis maxT（1152 格族，q=0.05）PRIMARY + joint 4x1152 maxT SECONDARY descriptive；200 次标签置换 fresh default_rng(2896)。
- **粒度先检预注册（纪律 8 制度化）**：每属性轴 top-40 格 distinct margin 值 >= 20（GRAN_MIN = 20）方可判 null "powered"——从 2920 post-hoc 诊断升格为判决链内预注册先检。
- 判决映射：attribute_events_found（任一属性 n_sig > 0）/ attribute_events_absent_powered / attribute_null_granularity_limited / anchor_fail_all_void。

### 锚（4/4，全逐位）
- a0：dirs2919[lang] vs 2886 推导 rel 2.86e-08。
- a1：B_lang fresh vs 2917 npz rel 3.53e-09。
- a2：m78 0.280360（== RECON_REF 0.28036）。
- a3：sign_M diff 4.71e-08 + 显著集 **24/24 集合相等**——lang 图谱第六次连续前向锚定，2917 完整复现。

### 结果
- **判决：attribute_events_found**。
- P1：lang 24（top (7,19) 1.34634 与 2917 逐位一致）；**speed 2**：（18,16）1.01608、（14,12）0.76479；**size 9**：（21,7）1.23834、（24,19）0.98181、（12,18）0.86648、（26,21）0.75588、（25,15）0.75505、（11,23）0.75478、（11,27）0.6575、（22,2）0.65529、（18,7）0.65363；**moist 2**：（8,15）0.99258、（10,9）0.9917。
- P2：overlap 矩阵纯对角 [24,2,9,2]，**shared pairs = 0**——13 个属性事件与 24 个 lang 事件零共享通道。
- P3 迁移复检（扩容词表）4/4 failed：median |cos| L1-35 = lang 0.1664 / speed 0.2025 / size 0.2193 / moist 0.0943（argmax L23/L12/L12/L12）——与 2920 同型。
- P4 粒度先检：top-40 distinct = lang 30 / speed 26 / size 27 / moist 23，**全部 >= 20** -> 四轴 null 全部 powered；max_perm p50 = 0.358-0.549（moist 偏高 0.5487/p95 0.8294）。
- 跨相位格检查：2920 遗留 size 格 (18,7) margin 1.53125（小词表）-> 0.65363（扩容）**缩水 58% 但 p_maxT 0.024876 显著幸存**。

### 解读
1. **属性 (头,层) 事件存在**：size 最强（9 事件、top margin 1.23834 仅次于 lang top），speed/moist 各 2。2919 方向族就绪 -> 2920 功效诊断预言 -> 2921 正结果，证据链收敛。属性轴不是只存在句级方向族——词级扰动响应同样有通道化的符号结构。
2. **2920->2921 翻转（0 -> 13 事件）= 纪律 8 的直接验证**：方法论修正（扩容 + 粒度先检）直接改变科学结论。负结果必须区分"真 absence"（2920 concept 型：观测 top 低于 null 中位）与"功效失效"（2920 属性轴型：margin 量化 + null 常规到达 top）。该翻转复用 2917 p 粒度教训的同类结构。
3. **零跨轴共享通道**（shared pairs = 0）：属性事件与 lang 事件完全不重叠，与 2920"事件编码语言身份"合并 -> 事件图谱按划分类型组织为互斥私有通道（"不同类别各自路径"支持）。跨轴共现在词级 sign 口径下无对象。
4. 迁移扩容后仍 4/4 failed——句子级方向族与词级方向几何不同一维持（功能迁移 != 几何同一；注入方向 = 测量仪器）。属性轴的因果有效性（2913-2918）与几何同一性分离保持。
5. 理论对应：属性通道私有化 -> 属性信息的 (头,层) 载体独立于语言身份载体；size 事件跨层分布（L7/L18/L19/L21/L23/L27）提示非单点机制。为 Cmp(o,r,v) 候选竞争提供了属性轴的载体候选。

### 方法论常数（新增/升级）
- **粒度先检 GRAN_MIN = 20（top-40 格 distinct margin 值）制度化**：n < 40 的轴（moist 35）由先检裁决 null 是否 powered；预注册先检入判决链（与纪律 7 p 粒度检查并列，纪律 8 补完）。
- **跨词表 margin 非不变量**：(18,7) margin 随词表从 1.53125 缩到 0.65363 但显著性幸存——sign-Gram margin 是词表相对量，跨词表只比显著性/集合，不比 margin 绝对值。
- moist 事件对（(8,15) 0.99258 / (10,9) 0.9917）margin 几乎并列——成对事件结构首次出现（lang 无此形态）。

### 执行史
- 词表探针 2 轮（FIXED = 2920 幸存词 + CANDS 候选；单 token 复查；重复检测）。
- 主脚本一次运行成功（79.8 s）——吸收 2920 全部教训（tid_map 登记、p_joint 广播、H_func base 索引）初版完成；npz 键 axis_n 不存在的小修复仅影响 seal 探针（不影响主产物）。
- seal：created 2026-09-19T12:08:29；四 SHA256-8 登记（见下）。

### 硬伤
- moist n = 35 仍低于 n >~ 40 指南线——先检兜底（gran40 = 23 达标）但功效偏弱（max_perm p50 0.5487 / p95 0.8294 为四轴最高）；moist 2 事件的稳健性待复核。
- sign 口径丢弃幅度结构（事件 = 符号响应结构，非幅度响应）。
- n = 1 run（逐位锚缓解）。
- 单模型（qwen3-4b）；属性词表英文单语（hi/lo 池跨语混排，但无跨语划分）。
- (18,7) 等 2920 幸存词保留在 2921 词表——跨相位对比非独立样本。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2921_attr_vocab_expansion.py: c6ae4da8
- execution.json: 5d6a6f05（created 2026-09-19T12:08:29）
- result.json: e23cd5e9（final_verdict=attribute_events_found，runtime 79.8 s）
- attr_vocab_expansion.npz: 8bc6066d（B x4、sign_M/p_M/p_maxT/p_joint x4 轴、max_perm/global_max_perm、dirs_used/dirs_word/cos_curve、词表+标签）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2921/attr_vocab_expansion/
- Ledger：M2921_attr_vocab_expansion 入账，measurements 59->60，L14 connects 27->28，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2921 行完成——判决 attribute_events_found 如实入账）

### 接续（2922 候选）
- A（主选）：**属性事件解剖**——对 13 个新属性事件跑 2918 协议（词位分解 / 极性 / 密度 / 响应曲线，零前向）：属性事件是否像 lang 事件一样可分解为词位级响应。
- B：**探针相对性检验**——lang 词表换词级 lang 探针（dirs_word[lang]，2921 npz 已存）重测事件集：事件集是否随探针改变。
- C（零前向）：h4 L1<->L19 复用子空间主角度分析（2918 唯一真复用通道）——roadmap 2921 遗留项。
- D：属性事件共现加深——零共享可能是 sign 口径敏感度问题，幅度口径或更松阈值的共现复检。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2921: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2921 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2921")
