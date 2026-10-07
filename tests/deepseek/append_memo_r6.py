# -*- coding: utf-8 -*-
"""R6: 按用户新规范，把当前对话 R3-R5b 研究日志追加到 AGI_DEEPSEEK_MEMO.md。
铁律：append-only；写前断言前缀 == 冻结基线（4ba5e22f / 497406 B）；
写后复核（前缀逐字节不变 / BOM / CRLF / bare_lf 0 / 行数 / sha8）。
"""
import os, hashlib

P = r"D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md"
REP = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek\result\_append_r6_report.txt"

old_raw = open(P, "rb").read()
OLD_SHA = hashlib.sha256(old_raw).hexdigest()
OLD_N = len(old_raw)
old_txt = old_raw.decode("utf-8-sig")
assert OLD_SHA[:8] == "4ba5e22f", "前缀基线漂移：期望 4ba5e22f，实得 %s" % OLD_SHA[:8]
assert OLD_N == 497406, "前缀字节漂移：期望 497406，实得 %d" % OLD_N
assert old_txt.count("\r\n") == 4773 and (old_txt.count("\n") - old_txt.count("\r\n")) == 0

L = []
A = L.append
A("")
A("## A 闸门 R3–R5b：循环诊断 → 宪法/冻结队列 → Q01/Q02 → Q09/Q12 → seal 请求包（非 Phase 编号）[2026-10-03 02:20]")
A("")
A("### 0 登记约定变更 v3（用户指令，2026-10-03 02:18）")
A("- **所有研究日志一律追加到本文件（`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`），不再写入其他 `.md` 文件。**")
A("- 原「N 线写 deepseek 备忘录 / G 线写 gpt5 备忘录」的**双备忘录制对日志类记录废止**；G 线备忘录不再作为研究日志落点。")
A("- 本文件仍保持 **UTF-8+BOM + CRLF + `bare_lf 0`** 与 **append-only**（纠错只 append，不回改）。")
A("- 待确认（挂账）：既有 `research/gpt5/docs/*.md` 交付/治理文档（宪法、TESTPLAN、报告）是否需并入本文件。")
A("")
A("### 1 R3 循环诊断 + 研究宪法 + 冻结队列")
A("- **诊断量化**：gpt5 备忘录 1.90 MB / 14,901 行 / **402 Phase**；判决串出现 **685** 次但只有 **48 个唯一标签（复用率 1.10）** ⇒ 判决不可跨 Phase 比较、无法累积；A 级命题 5/62（8%）；「接续」字段 **380** 处（自动派生下一 Phase 的模板）。")
A("- **验收函数必须可失败**：把 I1 操作化为「一个 Phase 是否降低了共享指标」——扫描 397 个 Phase 节，**仅 26 个（6.55%）**报告了「指标名 + 小数 A→B」，窄词表 18 ／ 宽词表 26（区间报告）⇒ **93.45% 是目录条目**。")
A("- **产物**：`research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md`（`013d08a9`）；`research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md`（`01df6398`，I1–I11 全部落为判据）；`research/gpt5/atlas/phase_queue_v1.json`（`675836fd`，Q01–Q30 唯一议程来源）。独立复核 **PASS 44 / FAIL 0**。")
A("- **工程修正**：生成器须 `newline=\"\\n\"` 强制 LF，否则 Windows 文本模式产生 CRLF，使同一内容的 sha256 随平台变化（实测 10,388 vs 10,555 B）。")
A("")
A("### 2 R4：Q01 元层单一真源对账 + Q02 KPI 口径冻结（零 GPU）")
A("- **Q01 五处不自洽（D1–D5）**：① 分级口径混用（MEMO 记 A5/B34/C14/D21/E20，TESTPLAN 记 A5/B6/C10/D21/E20）；② 有效依赖比例量纲混淆；③ `atlas_ledger.json` 自声明哈希 `41d65a13` ≠ 实际 `bbda63df`；④ MEMO 记 `proposition_ledger` = `add57ba7` ≠ 实际 `9a3c6ff4`；⑤ `measurements` 两套 schema 并存（270/199/兼有 165）且 `evidence_level` **304/304 恒为常量**。")
A("- **口径结论**：真源 = 命题账本 62 条（review 57 + new 5）；**分量口径**复现 MEMO 的 A5/B34/C14/D21/E20（和 **94**）；**TESTPLAN 的 B6/C10 五口径全不可复现**；「约 55%（A+B）」「63%」= **分量数 ÷ 条数**的量纲混淆，同量纲唯一成立值 = **41.5%（39/94）**。")
A("- **哈希策略**：文件内自声明哈希在追加后必然失效（自指结构性失效）⇒ 规范为 `content_excluding_self`（`sha256(json.dumps(去掉该字段, ensure_ascii=False, indent=1))[:8]`），重算不变量实测 **`0dc6e57a`**。")
A("- **产物**：`META_SINGLE_SOURCE_Q01.md`（`95366ca9`）+ `meta_single_source_v4.json`（`f9d7ede6`）；更正表 **C1–C6 待 seal**。独立复核 **PASS 44 / FAIL 0**。")
A("- **Q02 KPI 口径冻结**：`metric_dict.json` v1→v2（`469c0ad1` → `03887e51`，备份 v1）；新增 `global_kpis`：`E_read` = `k1_model_report.b4_rel_readout_mean3seed` = **0.33162 / 0.39860 / 0.38984（0/3 过 5% 门）**；`E_ar(k)` / `C_steer` 口径冻结但未测；`metrics` 7 项 + `meta_rules` 原样保留 ⇒ 兼容 P3150 历史断言。报告 `METRIC_DICT_Q02.md`（`ba518fea`）；复核 **PASS 43 / FAIL 0**。")
A("")
A("### 3 R5：Q09 死线双轨重述（I3）+ Q12 D/E 命题引用审计（I4）（零 GPU）")
A("- **Q09 病理**：`RDC_TESTPLAN_v1.md` §8.2 三条死线的触发条件**全部写成了对模型数/族数的全称量词合取**；且 **K2/K3 从未被测量**（`phase3154*` 目录 0 命中；全库无「top-50 覆盖率」量，MEMO 中的「覆盖率」指微场普查 283 词，非同一条目）；K2 在同一文件内还有**两种互斥操作化**（§8.2「响应 cos 下降 > 50%」 vs 3153 预注册「交互份额 > 50%」）。")
A("- **K1 逐模型现场重算**（3151/3152 result，`margin = err_cand − err_B4`，负 = 候选优，过门需 `m ≤ −2·MDE`）：")
A("  - `qwen3-4b`：k*=3 / readout=35；B4@k* `0.0079107` / B4@readout `0.3316153`；margin@k* `[-0.003996, -0.001872, -0.002429]` / MDE `[0.001646, 0.002166, 0.001724]`；`above_add_gate=False`。")
A("  - `qwen3-14b`：k*=3 / readout=39；B4@k* `0.0806781` / B4@readout `0.3986008`；margin@k* `[-0.043157, -0.044846, -0.044502]` / MDE `[0.007399, 0.008100, 0.009959]`；`above_add_gate=True`。")
A("  - `glm4-9b`：k*=3 / readout=39；B4@k* `0.0461279` / B4@readout `0.3898353`；margin@k* `[-0.028947, -0.032304, -0.024767]` / MDE `[0.006778, 0.007540, 0.005948]`；`above_add_gate=False`。")
A("- **聚合**：池化 margin@k* **−0.025202**（MDE 0.005696，轨 A 通过）；池化 margin@readout **+0.940920**（MDE 0.193309）；`E_read` 池化 **0.373350**；模型级 CI bootstrap n=3 / 20000 次 / seed 20261003。")
A("- **双轨判决**：**k* 层 = `model_specific`**（轨 A 通过，轨 B 由 `qwen3-4b` 否决）；**读出层 = `fired_all_models`**（3/3 否决 + 轨 A 触发）⇒ **与旧结论「未触发、算子代数线保住」相反**。层位选择属 Q08（待 seal）。")
A("- **Q12 装置**：账本驱动 + 命名空间感知 + 文本指纹级。新推理链 = MEMO 自 **Phase 3104（第 13244 行）**起至文末 ∪ `docs/*.md`（26 个；**排除**归档快照 `AGI_GPT5_MEMO_2026*.md` 与自身输出）。")
A("  - **F1（id 级）**：命中**仅 1 处** —— MEMO L13546 / Phase 3113 / `R55`（复合等级 `E+A`）。")
A("  - **F3（文本指纹）**：以 claim 的 **6-CJK-gram** 为指纹，29 条可构造，新推理链**实质命中 0** ⇒ **无实质违规**。")
A("- **两项制度缺陷**：**RULE-UNDECIDABLE** —— D∪E = **38/62 = 61.29%**，其中**复合等级 26 条**、纯 E 仅 3 条（`R03/R15/R46`）、纯 D 6 条 ⇒ 「E 级命题禁止进入新推理链」对 **26/38 条不可机械判定**（撤回的是 E 分量，A/B/C 分量仍可引用）；**NS-COLLISION** —— 账本 id `R\\d\\d` **不是项目内唯一命名空间**（`LOOP_DIAGNOSIS_AND_EXIT_v1.md` 自带 §2 清单 `R1–R11`，其 `R10` =「真阴性记录：翻译轴不存在」与账本 `R10`「实体感紧凑性定律」毫无关系）⇒ **纯 id grep 必假阳性**，审计必须叠加文本指纹或人工核定表。")
A("- **产物**：`deadline_dual_track_v1.json`（`4d1853d3`）+ `DEADLINE_DUAL_TRACK_Q09.md`（`0d8633c7`）；`prop_citation_audit_v1.json`（`4c2ea9d2`）+ `PROP_CITATION_AUDIT_Q12.md`（`32933855`）；汇报页 `q09q12_gate_r5.html`（`ee439e4c`）。独立复核 `disk_verify_q09q12_r5.py` = **PASS 76 / FAIL 0 / ALL_PASS**。")
A("")
A("### 4 R5b：A 闸门 seal 请求包 + 幻影写入纠错")
A("- **seal 请求包（只读汇总，不代为改判）**：`seal_request_v1.json`（`d871a4b4`）+ `SEAL_REQUEST_A_GATE.md`（`7678cbe0`）+ `seal_request_a_gate.html`（`3976537e`）。内容：Q08 K1 判定层（读法甲 = 改判·推荐 / 读法乙 = 重述并显式解冻升版 / 暂缓）+ C1–C6 更正表逐条 + I1/I9 冻结确认 + 现场重算的冻结指纹 + seal 后执行顺序。一行回复模板：`seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认`。")
A("- **幻影写入实证（重要）**：上一轮 `patch_*.py` 自报「wlog 追加成功」「技能教训追加成功」，独立探针发现 wlog（`d4540e4b`）与 `SKILL.md`（`afe6215c`）**与写入前逐字节相同** ⇒ **脚本日志不得作为落盘证据**，必须用独立探针 re-hash 每个目标文件。补写后：wlog `9187a35f`（2,907→7,299 B）、`SKILL.md` `66704c89`（65,880→68,234 B，新增**教训 38** 禁入类规则可机械判定性 / **教训 39** 一次脚本多处写入必须逐处独立复核）、`MEMORY.md` `46db6970`（5,606→5,160 字符，20/20 锚点）。")
A("- **根因**：幻影写入的触发点是「匹配串凭记忆构造」⇒ 一律先 dump 真实磁盘再构造匹配串。")
A("- **未改动**：MEMO / TESTPLAN / Ledger / 队列 / 宪法 / 3151–3153 产物一律未触碰；指纹现场复核全 OK（`2a84776b` / `9a3c6ff4` / `71b85673` / `675836fd` / `01df6398`）。**全程未改任何既有判定。**")
A("")
A("### 5 机制拼图与限界")
A("- **本段新增拼图**：A 闸门把「死线免疫」的完整病理定位为两条 —— ① **触发条件写成合取**（K1/K3）；② **两条死线根本没有装置**（K2/K3 从未被测量）。一个合取 + 两个空装置 ⇒ 死线在结构上不可能触发。")
A("- **限界**：① Q09/Q12 均为**元层审计**，不产生新物理结论；② K1 的「层位选择」是**决策点**（Q08）而非测量结果，本线不代为改判；③ Q12 只能证明「无 id 级与指纹级违规」，**不覆盖语义级引用**；④ `R\\d\\d` 命名空间冲突意味着任何面向本账本的自动化引用检查都必须自带语料白名单。")
A("")
A("### 6 第一性原理")
A("死线的作用是「让一个研究纲领可被杀死」。若触发条件用了对异质系统的合取，或者判据挂在**被检验假设自己选定的层**上，那么这条死线不是判据，而是**同义反复的保护壳**。本段两轮（Q08/Q09）给出的是同一个病的两个切面：**判定层自选**与**合取式触发**。修法是可机械检验的：主判据用跨模型聚合量 + bootstrap CI，单模型反例只作否决权（降级为 `model_specific`）。")
A("")
A("### 7 后续死线")
A("- **最高（阻塞）**：用户 seal（Q08 / C1–C6 / I1 / I9）⇒ A 闸门关闭 ⇒ 才可启动 B 闸门 Q03（`E_read` 统一基线复算，需 GPU）。")
A("- **并列**：N 线 P3–P7 补登 Ledger（零 GPU，纯记账）。")
A("- 仍挂账：Q04–Q07（KPI 装置，需 GPU）、Q10/Q11、N2h1-α-1 权重级、N2h1-β 水果类崩塌、N3-β→N3-ε、R1 补强、K4。")
A("")
A("### 8 一句话 ×3")
A("1. **零 GPU 的 A 闸门四件（Q01/Q02/Q09/Q12）已全部完成并独立复核**（44/0、43/0、76/0）。")
A("2. **死线免疫的病理 = 合取式触发 + 两条从未被测量的死线**；K1 按行为读出层重算后**应当触发**，与旧结论相反。")
A("3. **剩余全部是 seal 决策，不是待做实验**；seal 之后才进 B 闸门（需 GPU）。")
A("")

base_lf = old_txt.replace("\r\n", "\n").rstrip("\n")
new_lf = base_lf + "\n" + "\n".join(L)
payload = b"\xef\xbb\xbf" + new_lf.replace("\n", "\r\n").encode("utf-8")
open(P, "wb").write(payload)

back = open(P, "rb").read()
back_txt = back.decode("utf-8-sig")
prefix_ok = back[:OLD_N] == old_raw
rep = []
rep.append("PRE  bytes=%d sha8=%s lines=%d bom=%s" % (OLD_N, OLD_SHA[:8], old_txt.count("\n")+1, old_raw.startswith(b"\xef\xbb\xbf")))
rep.append("POST bytes=%d sha8=%s lines=%d" % (len(back), hashlib.sha256(back).hexdigest()[:8], back_txt.count("\n")+1))
rep.append("prefix_byte_identical = %s" % prefix_ok)
rep.append("BOM=%s  CRLF=%d  bare_lf=%d" % (back.startswith(b"\xef\xbb\xbf"), back_txt.count("\r\n"), back_txt.count("\n")-back_txt.count("\r\n")))
rep.append("has_new_section = %s" % ("## A 闸门 R3–R5b" in back_txt))
rep.append("phase_headings = %d" % sum(1 for l in back_txt.split("\n") if l.startswith("## ")))
assert prefix_ok, "前缀被改动！"
assert (back_txt.count("\n") - back_txt.count("\r\n")) == 0, "出现裸 LF"
open(REP, "w", encoding="utf-8").write("\n".join(rep) + "\n")
print("\n".join(rep))
