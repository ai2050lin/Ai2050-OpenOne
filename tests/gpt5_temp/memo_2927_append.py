# -*- coding: utf-8 -*-
"""memo_2927_append.py -- append Phase 2927 section to AGI_GPT5_MEMO.md.
Title time = execution created (2026-09-19T13:14:59 -> [2026-09-19 13:14]).
Idempotent: refuses if a 2927 title already exists.
"""
import hashlib
import json
import re

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
LEDGER = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas"
          r"\atlas_ledger.json")
ROADMAP = (r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs"
           r"\lpf_multiaxis_gating_roadmap_v1.md")
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_2927_report.txt"

with open(LEDGER, encoding="utf-8") as f:
    L = json.load(f)
m_ids = [m.get("meas_id") for m in L["measurements"]]
assert "M2927_probe_relativity" in m_ids, \
    "run ledger_2927_update.py first"
l14 = [c for c in L["linkage"]
       if c.get("link_id") == "L14_readout_spectrum_cross_model"][0]
assert "M2927_probe_relativity" in l14["connects"]
assert len(L["measurements"]) == 66 and len(l14["connects"]) == 34
LEDGER_SHA = hashlib.sha256(open(LEDGER, "rb").read()).hexdigest()[:8]
ROADMAP_SHA = hashlib.sha256(
    open(ROADMAP, "rb").read()).hexdigest()[:8]

with open(MEMO, encoding="utf-8") as f:
    txt = f.read()
n_before = txt.count("\n") + (0 if txt.endswith("\n") else 1)
assert "## Phase 2927:" not in txt, "already appended"

SEC = """
## Phase 2927: 探针相对性检验——events_probe_partially_invariant + 7 事件不变硬核 [2026-09-19 13:14]

### 目的（2926 接续候选 A）
2920 已证"功能迁移≠几何同一"（句子 dirs 与词级编码方向 median |cos| 0.166）。本 Phase 正面检验事件图谱本身：24 事件 lang 图谱是读出无关的电路性质，还是 2886 句子探针的伪象？一次前向（111 s），2917 协议 verbatim 但**双读出口径**：每次扰动同时算 dirs86（句子探针，锚口径）与 dirs_word（词级探针）。

### 设计（冻结口径）
- **dirs_word 构造**：本次前向 func 条件 [the, word] 的 pos-1 attn 输入残差流，57 词 lab_lang==0 组均值 − lab_lang==1 组均值逐层 unit。方向符号约定无关紧要——sign-Gram margin 对 B 全局符号翻转不变（outer(s,s)）。
- 两遍结构：pass 1 整体前向 171 序列存 attn 输入 → 构造 dirs_word；pass 2 逐层 attn_call 双方向扰动（共享 ref），per (condition, word) 存储。
- 锚（4/4，全部作用在本次前向的 2886 口径上——先认证前向，词级口径才可信）：a1 B86[:,:,26:36] vs 2913 rel **3.16e-08**（2917 谱系第七次连续前向锚定）；a2 m78 **0.280360** 精确；a3 sign_M vs 2917 npz diff **4.71e-08**；a4 maxT 事件集 **24/24 集合相等**——本次前向的 2886 口径是认证的 2917 复制品。
- P1 主检验：maxT（200 置换 rng2 2896 verbatim）on sign_M_word → E'；n_overlap = |E' ∩ E|。判决映射：overlap >= 12 且 (7,19) ∈ E' => events_probe_invariant；overlap >= 6 => events_probe_partially_invariant；else => events_probe_relative。

### 结果
- **判决：events_probe_partially_invariant**（overlap 7/24 ≥ 6，jaccard 0.143；top1 (7,19) ∈ E'）。
- **探针本身差异复现 2920**：cos(dirs_word, dirs86) median 0.165（min 0.000 / max 0.283）。
- **幸存核 7 事件：(1,6) (5,6) (7,19) (8,2) (14,9) (20,8) (21,6)**；(7,19) margin 1.346 → 1.018（−24%）但**层内排名双探针均 #1**——唯一排名稳定事件，不变硬核的锚。
- **丢失 17 事件——深层全灭**：l≥20 的 5 个深层事件（(4,22) (13,22) (17,28) (24,23) (27,24)）**全部丢失**；**新增 25 事件中 21/25（84%）在 l≤10**——词级探针把图谱整体推向早层、抹除深层事件。
- **P2 全格 Spearman(sign_M_word, sign_M86) = 0.176；E17 内部 24 事件 margin 排名相关 0.184**——探针更换不只是改显著性集合，是全格 margin 排序近乎重排。

### 解读
1. **事件图谱 = 探针 × 电路的交互产物**，非纯电路性质：句子 dirs（聚合语义）与词 dirs（词汇身份）读出电路的不同侧面——2917-2926 链条的所有图谱结构（24 事件、选拔机制、层位组织）都是"给定探针下"的命题。
2. **部分不变硬核**：7 事件跨探针幸存 + (7,19) 双探针层内第一——电路存在探针无关的强对比结构（19 层 h7 的极间对比在两种语义方向下都是层内极值）；图谱其余部分是探针相对的。
3. **层深选择的探针依赖**：句子探针看见深层事件、词探针看见早层事件——与 2922"属性事件 median peak L15 vs lang L6"的深度分离呼应：探针的"语义聚合度"决定它照亮电路的哪一段深度。
4. **与 2920 的闭环**：2920 证"注入方向几何≠词级编码几何"，2927 进一步证"换用词级几何后事件图谱实质改变但保留硬核"——功能迁移≠几何同一在图谱层面的具体化。
5. 理论对应：LPF v6 的极性/密度场是"读出方向条件化"的场——A(方向, 头, 层) 而非 A(头, 层)；探针族间的不变核才是电路的固有坐标。

### 方法论常数
- **双口径锚定协议**：换测量仪器的 Phase 用"旧口径完整复现旧 Phase（锚）+ 新口径主检验"结构——一次前向同时完成认证与检验，2927 首用入标配。
- **margin 的符号翻转不变性**：sign-Gram margin 判据对 B 全局符号翻转不变 → 探针方向约定（哪组为正）不影响事件判据，只影响方向语义解释——预注册时无需冻结探针方向约定。
- 事件图谱报告必须注明探针口径（"2917 图谱"= 句子探针图谱）。

### 执行史
- 主脚本一次运行成功（111 s；写入期自查修复 pass2 条件覆盖缺陷 + all_void 分支 NameError 风险 + res a4 表达式，均在运行前修复）；seal 探针：幸存核/丢失/新增分类、(7,19) 排名追踪、层深统计、E17 内部排名相关。

### 硬伤
- dirs_word 仅一种词级探针构造（func 条件 pos-1 attn 输入组差）——词级探针族内不变性未检验（候选 B 的第三探针族可部分覆盖）。
- E_word n=32 的 maxT 族校准同 2917（200 置换粒度对 1152 格族）；置换 null 只打乱标签掩码，G 不变——探针间的比较无共享置换结构，overlap 的零假设分布未校准（7/24 是否超随机重叠未检验——描述性对比，正式检验留 2928 候选 A）。
- n=1 run（锚 4/4 bit 级缓解）；单模型；英文/法语词表。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2927_probe_relativity.py: 082491b9
- execution.json: 6f52eedc（created 2026-09-19T13:14:59）
- result.json: 4302c248（final_verdict=events_probe_partially_invariant，runtime 111 s）
- probe_relativity.npz: 84fec594（B86/B_word fp32、sign_M86/word、p_maxT86/word、dirs_word、cos_profile）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2927/probe_relativity/
- Ledger：M2927_probe_relativity 入账，measurements 65->66，L14 connects 33->34，ledger sha256-8 = """ + LEDGER_SHA + """
- 理论纲领 research/gpt5/docs/lpf_multiaxis_gating_roadmap_v1.md: """ + ROADMAP_SHA + """（§4 2927 行完成）

### 接续（2928 候选）
- A（主选）：**幸存核解剖**——什么让 7 事件探针不变？词响应相关、极性、密度、(7,19) 的双探针 margin 剖面（零前向，2927 npz 上）；附 overlap 零假设校准（置换两个探针的标签流）。
- B：第三探针族检验（如逐层 PCA-1 或置乱标签 dirs，一次前向）——探针族内/间不变性谱系。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：dirs_word 口径的属性轴图谱（2921 协议换词探针，一次前向）——属性事件是否也探针部分不变。
"""

txt = txt.rstrip("\n") + "\n" + SEC.lstrip("\n")
with open(MEMO, "w", encoding="utf-8") as f:
    f.write(txt)

with open(MEMO, encoding="utf-8") as f:
    txt2 = f.read()
n_after = txt2.count("\n")
m = re.search(r"^## Phase 2927: .+ \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]$",
              txt2, re.M)
rep = ("MEMO 2927 appended: lines %d -> %d, title_ok=%s, "
       "ledger_sha=%s, roadmap_sha=%s"
       % (n_before, n_after, bool(m), LEDGER_SHA, ROADMAP_SHA))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(rep + "\n")
print("OK memo 2927")
